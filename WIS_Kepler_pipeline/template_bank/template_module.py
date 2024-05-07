#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:44:48 2020

@author: orynaivashtenko
"""

import numpy as np
import batman
import copy
import os
import auxiliary_functions as af
from utils import whiten_vector
from pipeline_parameters import Default_template_bank_folder, Default_templates_filename, \
    Default_templates_shifts_filename, Default_shifts_filename, Default_durations_filename, Default_indices_of_templates_filename, \
    Default_templates_of_indices_filename, Default_template_params_filename, Precomputed_tables_location, \
    Log_occurrences_templates_periods_m13r_per_edges_filename, Log_second_derivs_cos2_shifts_templates_filename, \
    Log_cos2_derivs_templates_filename, Log_template_prior_m13r_per_edges_filename, Limb_darkening_array_file, Default_limb_darkening_coefs

# dt_lc = 1765.5 / (60 * 60 * 24)  # days long cadence interval
dt_lc = 0.02043368  # almost the same. measured from data
L_ban = 105  # Length of the templates  # based on the maximal considered duration
T_ban = L_ban * dt_lc
Min_edge_zeros = 7  # Minimal number of zeros in the edge of the templates

Durations = np.load(os.path.join(Default_template_bank_folder, Default_durations_filename), allow_pickle=True)  # days
Indices_of_templates = np.load(os.path.join(Default_template_bank_folder, Default_indices_of_templates_filename), allow_pickle=True)
Templates_of_indices = np.load(os.path.join(Default_template_bank_folder, Default_templates_of_indices_filename), allow_pickle=True)
N_templates = len(Indices_of_templates)
Max_cos = 0.97  # was used in the bank generation. If cos was smaller, it was added to the bank

def get_default_templates_params(folder=Default_template_bank_folder, filename=Default_template_params_filename):
    # m_1_3_r_s, period, a, inc, ecc, w, u1, u2
    params_temps = np.load(os.path.join(folder, filename),  allow_pickle=True)
    return params_temps


# noinspection PyUnresolvedReferences
class TemplateBank(object):
    def __init__(self, templates=None,
                 # from_parameters=False,
                 parameters=None,
                 template_bank_folder=Default_template_bank_folder,
                 templates_filename=Default_templates_filename,
                 templates_shifts_filename=Default_templates_shifts_filename,
                 shifts_filename=Default_shifts_filename,
                 durations_filename=Default_durations_filename,
                 indices_of_templates_filename=Default_indices_of_templates_filename,
                 templates_of_indices_filename=Default_templates_of_indices_filename,
                 # parameters_file_dir=Default_template_params_filename,
                 template_index='all', shifts=None, dt=dt_lc, support=None,
                 enforce_centering=False,
                 enforce_same_length=False,
                 enforce_normalization=False,
                 enforce_sorting=False,
                 just_load=True,
                 load_essentials=False):
        """
        load_essentials : will be only initial_templates_shifts, templates_shifts, durations, indices_of_templates and templates_of_indices

        Initializes a kepler light curve object.

                self.dt = dt
        self.templates_file_dir = templates_file_dir
        self.shifts_file_dir = shifts_file_dir
        self.durations_file_dir = durations_file_dir

        self.initial_templates
        self.shifts
        self.len_initial_template
        self.num_templates
        self.durations
        self.templates
        self.indices_of_templates
        self.templates_of_indices
        self.shifts_shifts
        self.initial_templates_shifts
        self.templates_shifts
        self.num_templates_shifts
        self.support

        """
        self.dt = dt
        self.templates_file_dir = os.path.join(template_bank_folder, templates_filename)
        self.templates_shifts_file_dir = os.path.join(template_bank_folder, templates_shifts_filename)
        self.shifts_file_dir = os.path.join(template_bank_folder, shifts_filename)
        self.durations_file_dir = os.path.join(template_bank_folder, durations_filename)
        self.indices_of_templates_file_dir = os.path.join(template_bank_folder, indices_of_templates_filename)
        self.templates_of_indices_file_dir = os.path.join(template_bank_folder, templates_of_indices_filename)
        # self.parameters_file_dir = os.path.join(template_bank_folder, parameters_file_dir)

        self.initial_templates = None
        self.shifts = None
        self.len_initial_template = None
        self.num_templates = None
        self.durations = None
        self.templates = None
        self.indices_of_templates = None
        self.templates_of_indices = None
        self.shifts_shifts = None
        self.initial_templates_shifts = None
        self.templates_shifts = None
        self.num_templates_shifts = None
        self.template_index = -1 if template_index == 'all' else int(template_index)
        self.parameters = parameters

        self.support = None

        ########################################################
        # Reading from the file:
        ########################################################
        if (templates is None) and load_essentials:
            self.load_npy_bank(only_essentials=True)
            if template_index != 'all':
                if not hasattr(template_index, '__len__'):
                    template_index = [int(template_index)]  # assert type int does not work with numpy.int64...
                # self.initial_templates_shifts = [[self.initial_templates_shifts[si]
                #                                   for si in range(self.indices_of_templates[templ_ind][0],
                #                                   self.indices_of_templates[templ_ind][1])] for templ_ind in template_index]
                self.initial_templates_shifts = [self.initial_templates_shifts[si] for templ_ind in template_index
                                                 for si in range(self.indices_of_templates[templ_ind][0],
                                                                 self.indices_of_templates[templ_ind][1])]
                self.templates_shifts = copy.deepcopy(self.initial_templates_shifts)
                self.durations = [self.durations[templ_ind] for templ_ind in template_index]
                indices_of_templates = []
                ind = 0
                for templ_ind in template_index:
                    step = self.indices_of_templates[templ_ind][1] - self.indices_of_templates[templ_ind][0]
                    indices_of_templates.append([ind, ind + step])
                    ind = ind + step
                self.indices_of_templates = indices_of_templates
                self.templates_of_indices = [k for k in range(len(self.indices_of_templates))
                                             for _ in range(self.indices_of_templates[k][0], self.indices_of_templates[k][1])]
            self.calculate_len_and_number()
        else:
            if (templates is None) and just_load:
                self.load_npy_bank(only_essentials=False)
            else:
                if templates is None:
                    assert self.templates_file_dir is not None, 'There must be provided either `templates` or `templates_file_dir`!'
                    if self.templates_file_dir[-4:] == '.npy':
                        self.initial_templates = np.load(self.templates_file_dir, allow_pickle=True)
                    elif self.templates_file_dir[-4:] == '.txt':
                        self.initial_templates = af.read_quarters_from_file(self.templates_file_dir, dimension=3)
                    else:
                        raise TypeError('Unrecognized template bank file type!')
                else:
                    self.initial_templates = templates

                if not hasattr(self.initial_templates[0][0], '__len__'):
                    self.initial_templates = [[t] for t in self.initial_templates]
                else:
                    self.initial_templates = self.initial_templates

                # Load or set the subbin time shifts of the templates:
                if shifts is None:
                    if (self.shifts_file_dir is not None) and (load_essentials or just_load):
                        if self.shifts_file_dir[-4:] == '.npy':
                            self.shifts = np.load(self.shifts_file_dir, allow_pickle=True)
                        elif self.shifts_file_dir[-4:] == '.txt':
                            self.shifts = af.read_quarters_from_file(self.shifts_file_dir, dimension=2)
                        else:
                            raise TypeError('Unrecognized shifts file type!')
                    else:
                        self.shifts = [[i / len(tt) * self.dt for i in range(len(tt))] for tt in self.initial_templates]
                        # self.shifts = [[0.] for _ in range(len(self.initial_templates))]
                else:
                    self.shifts = shifts

                # Load or measure the durations of the templates:
                if (self.durations_file_dir is not None) and (load_essentials or just_load):
                    if self.durations_file_dir[-4:] == '.npy':
                        self.durations = np.load(self.durations_file_dir, allow_pickle=True)
                    elif self.durations_file_dir[-4:] == '.txt':
                        self.durations = af.get_values_from_txt(file_dir=self.durations_file_dir, filename=None)
                        # TODO: make sure the units work (hours/days)!
                    else:
                        raise TypeError('Unrecognized durations file type!')
                else:
                    self.durations = [templ_dur_fun(tt[0]) * self.dt for tt in self.initial_templates]

            if template_index != 'all':
                if hasattr(template_index, '__len__'):
                    self.initial_templates = self.initial_templates[template_index]
                    self.shifts = self.shifts[template_index]
                    self.durations = self.durations[template_index]
                else:
                    template_index = int(template_index)  # assert type int does not work with numpy.int64...
                    self.initial_templates = [self.initial_templates[template_index]]
                    self.shifts = [self.shifts[template_index]]
                    self.durations = [self.durations[template_index]]

            ########################################################
            # Maybe make sure they are centered and normalized:
            ########################################################

            if enforce_centering:
                self.initial_templates = center_templates(self.initial_templates)
            if enforce_normalization:
                self.initial_templates = normalize_templates(self.initial_templates)

            self.calculate_len_and_number()

            ########################################################
            # Sorting putting the ones with shifts separately
            #########################################################
            if enforce_sorting:
                indices_for_reordering = []
                available_indices = list(range(len(self.initial_templates)))
                while len(available_indices) > 0:
                    next_i = available_indices[0]
                    for i in available_indices[1:]:
                        if len(self.initial_templates[i]) > len(self.initial_templates[next_i]):  # there are more shifts
                            next_i = i
                        elif len(self.initial_templates[i]) == len(
                                self.initial_templates[next_i]):  # same number of shifts, but the duration is shorter
                            if self.durations[i] < self.durations[next_i]:
                                next_i = i
                            elif self.durations[i] == self.durations[next_i]:
                                if np.max(self.initial_templates[i][0]) > np.max(self.initial_templates[next_i][0]):  # it is as if shorter
                                    next_i = i
                    indices_for_reordering.append(next_i)
                    available_indices.remove(next_i)
                self.initial_templates = [self.initial_templates[ind] for ind in indices_for_reordering]
                self.shifts = [self.shifts[ind] for ind in indices_for_reordering]
                self.durations = [self.durations[ind] for ind in indices_for_reordering]
                if self.parameters is not None:
                    self.parameters = [self.parameters[ind] for ind in indices_for_reordering]
            ###########################################################################################################################
            self.templates = copy.deepcopy(self.initial_templates)
            ########################################################
            # Rearranging by putting all the shifts subsequently and not in groups
            ########################################################
            self.make_flattened_shifts_arrays()

        ########################################################
        # Padding
        ########################################################
        if support is None:
            if enforce_same_length:
                self.support = self.len_initial_template
                self.pad_templates(support=support)
        else:
            self.support = support
            self.pad_templates(support=support)

    def calculate_len_and_number(self):
        if self.initial_templates is not None:
            self.len_initial_template = np.max([len(template[0]) for template in self.initial_templates])
        if self.templates is not None:
            self.num_templates = len(self.templates)
        elif self.indices_of_templates is not None:
            self.num_templates = len(self.indices_of_templates)
        if self.templates_shifts is not None:
            self.num_templates_shifts = len(self.templates_shifts)
        elif self.templates_of_indices is not None:
            self.num_templates_shifts = len(self.templates_of_indices)

    def normalize_templates(self):
        self.templates = normalize_templates(self.templates)
        if self.templates_shifts is not None:
            self.templates_shifts = normalize_templates(self.templates_shifts)

    def center_templates(self):
        self.templates = center_templates(self.templates)
        if self.templates_shifts is not None:
            self.templates_shifts = center_templates(self.templates_shifts)

    def make_flattened_shifts_arrays(self):
        indlen = list(np.cumsum([len(s) for s in self.shifts]))
        # pairs of indices denoting the beginning and the end of different shifts of one template
        self.indices_of_templates = np.array(list(zip(np.concatenate(([0], indlen)), indlen)))
        self.templates_of_indices = [k for k in range(len(self.indices_of_templates))
                                     for _ in range(self.indices_of_templates[k][0], self.indices_of_templates[k][1])]

        self.shifts_shifts = np.hstack(self.shifts)
        self.num_templates_shifts = len(self.shifts_shifts)
        self.initial_templates_shifts = np.concatenate(self.initial_templates)
        self.templates_shifts = np.concatenate(self.templates)

    def make_grouped_shifts_arrays(self):
        self.templates = []
        for inds in self.indices_of_templates:
            self.templates.append(self.templates_shifts[inds[0]:inds[1]])

    def pad_templates(self, support):
        self.support = support
        if self.templates is not None:
            self.templates = pad_templates(self.templates, support)
        if self.templates_shifts is not None:
            self.templates_shifts = pad_templates(self.templates_shifts, support)

    def roll_templates(self):
        if self.templates is not None:
            self.templates = roll_templates(self.templates)
        if self.templates_shifts is not None:
            self.templates_shifts = roll_templates(self.templates_shifts)

    def save_bank(self, templates_file_dir=None, shifts_file_dir=None, durations_file_dir=None,
                  templates_shifts_file_dir=None, indices_of_templates_file_dir=None, templates_of_indices_file_dir=None):
        if templates_file_dir is None:
            templates_file_dir = self.templates_file_dir
        if shifts_file_dir is None:
            shifts_file_dir = self.shifts_file_dir
        if durations_file_dir is None:
            durations_file_dir = self.durations_file_dir
        if templates_shifts_file_dir is None:
            templates_shifts_file_dir = self.templates_shifts_file_dir
        if indices_of_templates_file_dir is None:
            indices_of_templates_file_dir = self.indices_of_templates_file_dir
        if templates_of_indices_file_dir is None:
            templates_of_indices_file_dir = self.templates_of_indices_file_dir

        for file_dir, value in [[templates_file_dir, self.templates], [shifts_file_dir, self.shifts], [durations_file_dir, self.durations],
                                [templates_shifts_file_dir, self.templates_shifts],
                                [indices_of_templates_file_dir, self.indices_of_templates],
                                [templates_of_indices_file_dir, self.templates_of_indices]]:
            if file_dir[-4:] == '.npy':
                np.save(file_dir, np.array(value, dtype=object), allow_pickle=True)
            elif file_dir[-4:] == '.txt':
                with open(file_dir, 'w') as file:
                    print(*value, file=file)
            else:
                raise TypeError('Unrecognized template bank file type!')

    def save_parameters(self, parameters_file_dir=None):
        if parameters_file_dir is None:
            parameters_file_dir = self.parameters_file_dir

        file_dir, value = parameters_file_dir, self.parameters
        if file_dir[-4:] == '.npy':
            np.save(file_dir, np.array(value, dtype=object), allow_pickle=True)
        elif file_dir[-4:] == '.txt':
            with open(file_dir, 'w') as file:
                print(*value, file=file)
        else:
            raise TypeError('Unrecognized template bank file type!')

    def load_npy_bank(self, templates_file_dir=None, shifts_file_dir=None, durations_file_dir=None, templates_shifts_file_dir=None,
                      indices_of_templates_file_dir=None, templates_of_indices_file_dir=None, only_essentials=False):
        if templates_shifts_file_dir is None:
            templates_shifts_file_dir = self.templates_shifts_file_dir
        if durations_file_dir is None:
            durations_file_dir = self.durations_file_dir
        if indices_of_templates_file_dir is None:
            indices_of_templates_file_dir = self.indices_of_templates_file_dir
        if templates_of_indices_file_dir is None:
            templates_of_indices_file_dir = self.templates_of_indices_file_dir
        if not only_essentials:
            if templates_file_dir is None:
                templates_file_dir = self.templates_file_dir
            if shifts_file_dir is None:
                shifts_file_dir = self.shifts_file_dir
            self.initial_templates, self.initial_templates_shifts, self.shifts, self.durations, self.indices_of_templates, \
                self.templates_of_indices = load_npy_bank(templates_file_dir=templates_file_dir,
                                                          templates_shifts_file_dir=templates_shifts_file_dir,
                                                          shifts_file_dir=shifts_file_dir,
                                                          durations_file_dir=durations_file_dir,
                                                          indices_of_templates_file_dir=indices_of_templates_file_dir,
                                                          templates_of_indices_file_dir=templates_of_indices_file_dir,
                                                          only_essentials=False)
            self.templates = copy.deepcopy(self.initial_templates)
            self.templates_shifts = copy.deepcopy(self.initial_templates_shifts)
        else:
            self.initial_templates_shifts, self.durations, self.indices_of_templates, \
                self.templates_of_indices = load_npy_bank(templates_file_dir=templates_file_dir,
                                                          templates_shifts_file_dir=templates_shifts_file_dir,
                                                          shifts_file_dir=shifts_file_dir,
                                                          durations_file_dir=durations_file_dir,
                                                          indices_of_templates_file_dir=indices_of_templates_file_dir,
                                                          templates_of_indices_file_dir=templates_of_indices_file_dir,
                                                          only_essentials=True)
            self.templates_shifts = copy.deepcopy(self.initial_templates_shifts)
        self.calculate_len_and_number()


def load_npy_bank(templates_file_dir=os.path.join(Default_template_bank_folder, Default_templates_filename),
                  templates_shifts_file_dir=os.path.join(Default_template_bank_folder, Default_templates_shifts_filename),
                  shifts_file_dir=os.path.join(Default_template_bank_folder, Default_shifts_filename),
                  durations_file_dir=os.path.join(Default_template_bank_folder, Default_durations_filename),
                  indices_of_templates_file_dir=os.path.join(Default_template_bank_folder, Default_indices_of_templates_filename),
                  templates_of_indices_file_dir=os.path.join(Default_template_bank_folder, Default_templates_of_indices_filename),
                  only_essentials=False):
    initial_templates_shifts = list(np.load(templates_shifts_file_dir, allow_pickle=True))
    durations = list(np.load(durations_file_dir, allow_pickle=True))
    indices_of_templates = np.load(indices_of_templates_file_dir, allow_pickle=True)
    templates_of_indices = np.load(templates_of_indices_file_dir, allow_pickle=True)
    if not only_essentials:
        initial_templates = list(np.load(templates_file_dir, allow_pickle=True))
        shifts = list(np.load(shifts_file_dir, allow_pickle=True))
        return initial_templates, initial_templates_shifts, shifts, durations, indices_of_templates, templates_of_indices
    else:
        return initial_templates_shifts, durations, indices_of_templates, templates_of_indices


def pad_template(template, support):
    return np.concatenate((np.zeros((support - len(template)) // 2),
                           template,
                           np.zeros((support - len(template) + 1) // 2)))


def center_templates(templates):
    """
    Centers so that the peak (first peak) is either at the center or one past center.
    Accepts both single templates and the templates with shifts.
    :param templates:
    :type templates:
    :return:
    :rtype:
    """
    centered_templates = []
    if hasattr(templates[0][0], '__len__'):
        for template in templates:
            centered_template = []
            roll_value = len(template[0]) // 2 - np.argmax(template[0])
            for t_sh in template:
                centered_template.append(np.roll(t_sh, roll_value))
            centered_templates.append(centered_template)
    else:
        for template in templates:
            roll_value = len(template) // 2 - np.argmax(template)
            centered_templates.append(np.roll(template, roll_value))
    return centered_templates


def pad_templates(templates, support):
    padded_templates = []
    if hasattr(templates[0][0], '__len__'):
        for template in templates:
            padded_template = []
            for t_sh in template:
                padded_template.append(pad_template(t_sh, support))
            padded_templates.append(padded_template)
    else:
        for template in templates:
            padded_templates.append(pad_template(template, support))
    return padded_templates


def normalize_templates(templates, method='norm'):
    if hasattr(templates[0][0], '__len__'):
        return [[normalize(t_sh, method=method) for t_sh in temp] for temp in templates]
    else:
        return [normalize(temp, method=method) for temp in templates]


def roll_templates(templates):
    rolled_templates = []
    if hasattr(templates[0][0], '__len__'):
        for template in templates:
            rolled_template = []
            for t_sh in template:
                rolled_template.append(np.roll(t_sh, -(len(t_sh) // 2 - 1)))
            rolled_templates.append(rolled_template)
    else:
        for template in templates:
            rolled_templates.append(np.roll(template, -(len(template) // 2 - 1)))
    return rolled_templates


def shape_coef_fun(templ):
    """
    A coefficient characterizing the shape of the template (like FWHM).
    Compares duration at half max and the total non-zero duration.
    :param templ:
    :type templ:
    :return:
    :rtype:
    """
    dur_hmax = templ_dur_hmax_fun(templ)
    dur_full = len(np.where(np.abs(templ) > 0)[0])
    return dur_hmax / dur_full


def com_fun(vec):
    """
    Center of mass of a vector, linear weights
    :param vec:
    :type vec:
    :return:
    :rtype:
    """
    return np.sum(np.arange(len(vec)) * vec) / np.sum(vec)


def templ_dur_hmax_fun(templ):
    """
    Duration of a template at half-maximum
    :param templ:
    :type templ:
    :return:
    :rtype:
    """
    signmax = np.sign(templ[np.argmax(np.abs(templ))])
    templ = templ * signmax  # make it positive-oriented
    hmax = np.max(templ) / 2
    return len(np.where(templ >= hmax)[0])


def templ_dur_fun(templ, quadratic_threshold=0.9, abs_threshold=0.9, cumulative=True):
    """
    Calculates the template duration in some sophisticated way
    :param templ:
    :type templ:
    :param quadratic_threshold:
    :type quadratic_threshold:
    :param abs_threshold:
    :type abs_threshold:
    :param cumulative:
    :type cumulative:
    :return:
    :rtype:
    """
    if quadratic_threshold is not None:
        templ_work = templ ** 2
        threshold_ratio = quadratic_threshold
    else:
        templ_work = np.abs(templ)
        threshold_ratio = abs_threshold
    if not cumulative:
        templ_work = np.sort(templ_work)[::-1]
        cs = np.cumsum(templ_work)
        thresh_value = threshold_ratio * cs[-1]
        thresh_ind = np.searchsorted(cs, thresh_value)
        if thresh_value in cs:
            thresh_ind += 1
        return thresh_ind
    else:
        threshold = np.sum(templ_work) * threshold_ratio
        # The cumulative method only works with a monotonous template
        center_ind = np.argmax(templ_work)
        cen_l = center_ind  # left center
        while templ_work[cen_l - 1] == templ_work[center_ind]:
            cen_l -= 1
        cen_r = center_ind
        while templ_work[cen_r + 1] == templ_work[center_ind]:
            cen_r += 1
        sum_vals = np.sum(templ_work[cen_l:cen_r + 1])

        hwid = 1
        dur = cen_r + 1 - cen_l
        if sum_vals < threshold:
            while hwid <= len(templ) // 2 - 1:
                sum_vals_l = sum_vals + templ_work[cen_l - hwid]
                if sum_vals_l >= threshold:
                    dur = (cen_r + hwid) - (cen_l - hwid)
                    break
                sum_vals_r = sum_vals + templ_work[cen_r + hwid]
                if sum_vals_r >= threshold:
                    dur = (cen_r + hwid) - (cen_l - hwid)
                    break
                sum_vals = sum_vals_l + templ_work[cen_r + hwid]
                if sum_vals >= threshold:
                    dur = (cen_r + hwid) - (cen_l - hwid) + 1
                    break
                hwid += 1
        return dur

    # # # The following only worked if the template was centered in the center and symmetric:
    # cs = np.cumsum(templ_work)
    # ss_ind = np.searchsorted(cs, cs[-1] / 2)  # this was the way to find the center
    # threshold = cs[-1] * 0.9  # threshold of the integral
    # if (cs[-1] / 2 in cs) and (ss_ind == (len(cs) / 2) - 1):  # and (len(cs)%2==0):
    #     # center_in = True
    #     cen_l = ss_ind  # left center counting
    #     cen_r = ss_ind + 1  # right center counting
    #     sum_vals = templ_work[cen_l] + templ_work[cen_r]
    # else:
    #     # center_in = False
    #     cen_l = ss_ind  # left center counting
    #     cen_r = ss_ind  # right center counting
    #     sum_vals = templ_work[cen_l]
    # hwid = 1
    # dur = 0
    # while hwid <= len(templ) // 2 - 1:
    #     sum_vals_l = sum_vals + templ_work[cen_l - hwid]
    #     if sum_vals_l >= threshold:
    #         dur = (cen_r + hwid) - (cen_l - hwid)
    #         break
    #     sum_vals_r = sum_vals + templ_work[cen_r + hwid]
    #     if sum_vals_r >= threshold:
    #         dur = (cen_r + hwid) - (cen_l - hwid)
    #         break
    #     sum_vals = sum_vals_l + templ_work[cen_r + hwid]
    #     if sum_vals >= threshold:
    #         dur = (cen_r + hwid) - (cen_l - hwid) + 1
    #         break
    #     hwid += 1
    # return dur


def trim_templates(templates, leave_spare_bins=2):
    """
    The template is assumed to be centered in the center, and not in the beginning of the vector.
    :param templates:
    :type templates:
    :param leave_spare_bins:
    :type leave_spare_bins:
    :return:
    :rtype:
    """
    templates_trimmed = []
    if hasattr(templates[0][0], '__len__'):
        zer_len = len(templates[0][0]) // 2
        for template in templates:
            for templ_sh in template:
                non_zer = np.where(~np.isclose(templ_sh, 0))[0]
                zer_len = min(zer_len, non_zer[0], len(templ_sh) - 1 - non_zer[-1])
        zer_len = max(0, zer_len - leave_spare_bins)
        for template in templates:
            template_trimmed = []
            for templ_sh in template:
                template_trimmed.append(templ_sh[zer_len: len(templ_sh) - zer_len])
            templates_trimmed.append(template_trimmed)
    else:
        zer_len = len(templates[0]) // 2
        for template in templates:
            non_zer = np.where(~np.isclose(template, 0))[0]
            zer_len = min(zer_len, non_zer[0], len(template) - 1 - non_zer[-1])
        zer_len = max(0, zer_len - leave_spare_bins)
        for template in templates:
            templates_trimmed.append(template[zer_len: len(template) - zer_len])
    return templates_trimmed


# ATTENTION, HERE WE GENERATE THE TIME WITH ODD NUMBER OF POINTS!!
# we also generated even, it is enough to choose another time t which is commented
def time(t_tot=T_ban, dt=dt_lc, t_bins=None, parity='odd', fromzero=False):
    """
    generates time from -T/2 to T/2 (not exactly, rounding down) with interval dt
    can have t=0 (parity=odd) or not have (parity=even)
    Tbins is number of Bins that we want to have with spacing dt. Tbins overwrites T
    ATTENTION Tbins is the overall number, so not (-Tbins, Tbins)!

    Parameters
    ----------
    t_tot : float
        The full duratin of the time sample in some units (the same as `dt`).
        The default is 2*`T_ban`.
    dt : float
        The time step in the same units as T. The default is `dt_lc`.
    t_bins : int or None
        The number of time steps dt to be in the result. If None, then `Tbins`
        will not be used, and T will be used instead. The default is None.
    parity : str : 'even' or 'odd'
        Whether there should be an even number of Bins or not. The default is 'odd'.
    fromzero : bool
        Whether the time semple should start from zero or be symmetric around zero.
        The default is False.

    Returns
    -------
    numpy.array 1d of float
        Time array with step `dt`

    """
    if not fromzero:
        if t_bins is None:
            if parity == 'odd':
                return np.concatenate(((-1) * np.arange(0, t_tot / 2, dt)[:0:-1], np.arange(0, t_tot / 2, dt)))
            elif parity == 'even':
                return np.concatenate(((-1) * np.arange(dt / 2, t_tot / 2, dt)[::-1], np.arange(dt / 2, t_tot / 2, dt)))
            else:
                raise AssertionError('not recognized parity value')
        else:
            assert type(t_bins) == int, '`Tbins` should be int'
            if t_bins % 2 != 0:
                t_amp = (t_bins + 1) // 2
                return np.concatenate(((-dt) * np.arange(t_amp)[:0:-1], dt * np.arange(t_amp)))
            else:
                t_amp = t_bins // 2
                return np.concatenate(((-dt) * (np.arange(t_amp)[::-1] + 0.5), dt * (np.arange(t_amp) + 0.5)))
    else:
        if t_bins is None:
            return np.arange(0, t_tot, dt)
        else:
            assert type(t_bins) == int, '`Tbins` should be int'
            return dt * np.arange(t_bins)


def normalize(x, method='norm', t=None):
    """
    Normalizes the vector with following options:
    1) np.linalg.norm
    2) by the maximum (bymax=True)
    3) by the sum (bysum=True)
    4) can perform sum and integrals with respect to time axis (if t specified))

    Parameters
    ----------
    x : np.array
        a vector to normalize.
    t : float or np.array, optional
        If t is array, it is time series.
        If t is float it is time step dt.
        if specified, normalization is performed by time integral
        (squared or not squared if bysum). The default is 0.
    method : `norm`, `max`, or `sum`

    Returns
    -------
    np.array
        the normalized vector.

    """
    if method == 'norm':
        if t is None:
            return x / np.linalg.norm(x)
        elif type(t) == float or type(t) == int:
            return x / np.linalg.norm(t * x)
        else:
            dt = (np.roll(t, -1) - t)[:-1]
            dt = np.append(dt, dt[-1])
            return x / np.linalg.norm(dt * x)
    elif method == 'max':
        return x / np.max(np.abs(x))
    elif method == 'sum':
        if t is None:
            return x / np.abs(np.sum(x))
        elif type(t) == float or type(t) == int:
            return x / np.abs(np.sum(t * x))
        else:
            dt = (np.roll(t, -1) - t)[:-1]
            dt = np.append(dt, dt[-1])
            return x / np.abs(np.sum(dt * x))
    else:
        raise ValueError('Unrecognized `method`!')


def flux_par(t=None, params=None, per=365., rp=0.01, a=215., inc=90., ecc=0., w=90.,
             u1=Default_limb_darkening_coefs[0], u2=Default_limb_darkening_coefs[1], t0=0,
             supersample_factor=20, exp_time=None):
    """
    Generates a light curve with batman package from the specified parameters.
    The default parameters correspond to Earth.

    :param t: time, array (not necessarily evenly spaced), any units consistent with period
    :type t: array(float), time units
    :param params:
    :type params:
    :param per: period, any units consistent with time. The default is 365.
    :type per: float, time_units
    :param rp: planet radius divided by stellar radius. The default is 0.01.
    :type rp: float, unitless
    :param a: major semi-axis, in units of stellar radius. The default is 215.
    :type a: float, units of stellar radii
    :param inc: inclination in degrees. The default is 90., it orig_means there is no inclination, a straight transit.
    :type inc: float, deg, 90 is direct transit.
    :param ecc: eccentricity. The default is 0.
    :type ecc: float, from 0 to 1
    :param w: periapsis argument, degree. The default is 90., which is perihelion towards us
    :type w: float, degrees.
    :param u1: first limb darkening coefficient for the quadratic model, in [0,1]. The default for Sun is 0.47.
    :type u1: float
    :param u2: second limb darkening coefficient for the quadratic model, in [0,1]. The default for Sun is 0.23.
    :type u2: float
    :param t0: time of inferior conjunction. The default is 0.
    :type t0: float,
    :param supersample_factor: Number of samples used for integration
    :type supersample_factor:
    :param exp_time: exposure time
    :type exp_time:
    :return:
    :rtype:

    Defaults Earth:
    Rs = 1  # Rsun
    params = batman.TransitParams()
    params.t0 = 0.  # time of inferior conjunction
    params.per = 500.  # orbital period
    params.rp = 0.01 / Rs  # planet radius (in units of stellar radii)
    params.a = (params.per / 365) ** (2 / 3) * 215 / Rs  # semi-major axis (in units of stellar radii)
    params.inc = 90.  # orbital inclination (in degrees)
    params.ecc = 0.9  # eccentricity
    params.w = 270.  # longitude of periastron (in degrees)
    params.u = [0.47, 0.23]  # limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"  # limb darkening model

    Returns
    -------
    flux : numpy.array 1d of float
        flux (periodic) at corresponding times with max at 1 and dimmings on transits.
    #returns the flux as function of given physical parameters

    """
    if t is None:
        t = time()
    if params is None:
        params = batman.TransitParams()
        params.t0 = t0  # time of inferior conjunction
        params.per = per  # orbital period
        params.rp = rp  # planet radius (in units of stellar radii)
        params.a = a  # semi-major axis (in units of stellar radii)
        params.inc = inc  # orbital inclination (in degrees)
        params.ecc = ecc  # eccentricity
        params.w = w  # longitude of periastron (in degrees)
        params.u = [u1, u2]  # limb darkening coefficients [u1, u2]
        params.limb_dark = "quadratic"  # limb darkening model
    if supersample_factor is not None:
        if exp_time is None:
            exp_time = np.nanmean((t - np.roll(t, 1))[1:])
        m = batman.TransitModel(params, t, supersample_factor=supersample_factor, exp_time=exp_time)
    else:
        m = batman.TransitModel(params, t)  # initializes model
    flux = m.light_curve(params)  # calculates light curve
    return flux


def make_template_from_pars(L, per, a, inc, ecc, w,
                            u1=Default_limb_darkening_coefs[0], u2=Default_limb_darkening_coefs[1],
                            template_support=256, min_cos_shift=0.93, n_shifts=3, rp=0.02,
                            enforce_symmetrization=True, dt=dt_lc, effective_wf_fd=None):
    shift_step = dt / n_shifts
    if effective_wf_fd is None:
        from simulations.lightcurve_simulation import get_saved_wf_fd
        freqs = np.fft.rfftfreq(template_support, d=dt)
        effective_wf_fd = get_saved_wf_fd(freqs=freqs, dt=dt)

    t_red = time(t_tot=L * dt, parity='odd')
    flux_obs = 1. - flux_par(t=t_red, params=None, per=per, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u1, u2=u2, t0=0, supersample_factor=20,
                             exp_time=dt_lc)
    flux_obs_0 = np.array(flux_obs)
    flip = 1
    if enforce_symmetrization:
        flux_obs = np.roll(flux_obs, -np.argmax(flux_obs))
        l2 = (len(flux_obs) + 1) // 2
        s1 = np.sum(flux_obs[1:l2])
        s2 = np.sum(flux_obs[-l2 + 1:])
        if s1 < s2:
            flip = -1
            flux_obs = np.array(flux_obs_0)[::-1]
        else:
            flux_obs = np.array(flux_obs_0)
    init_roll = -(np.argmax(flux_obs) - (len(flux_obs) - 1) // 2)
    flux_obs = np.roll(flux_obs, init_roll)
    flux_obs = flux_obs / np.linalg.norm(flux_obs)

    flux_obs_to_wh = np.roll(flux_obs, -np.argmax(flux_obs))
    flux_obs_to_wh = np.concatenate((flux_obs_to_wh[:len(flux_obs_to_wh) // 2],
                                     np.zeros(template_support - len(flux_obs_to_wh)), flux_obs_to_wh[len(flux_obs_to_wh) // 2:]))
    wh_flux_obs = np.fft.rfft(flux_obs_to_wh) * effective_wf_fd
    wh_flux_obs = wh_flux_obs / np.linalg.norm(np.fft.irfft(wh_flux_obs))
    # wh_flux_obs_prepared_for_cos_fd = np.conj(wh_flux_obs) * cosine_coeffs

    return_temp = [np.array(flux_obs)]
    return_wh = [np.array(wh_flux_obs)]

    cos_angle_shift = np.fft.irfft(wh_flux_obs) @ np.roll(np.fft.irfft(wh_flux_obs), -1)
    # cos_angle_shift = np.real(wh_flux_obs_prepared_for_cos_fd @ np.fft.rfft(np.roll(np.fft.irfft(wh_flux_obs), -1)))
    if cos_angle_shift < min_cos_shift:
        for i_sh in range(1, n_shifts):
            flux_obs = 1. - flux_par(t=t_red, params=None, per=per, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u1, u2=u2,
                                     t0=i_sh * shift_step * flip,
                                     supersample_factor=20, exp_time=dt_lc)
            if flip == -1:
                flux_obs = flux_obs[::-1]
            flux_obs = np.roll(flux_obs, init_roll)
            flux_obs = flux_obs / np.linalg.norm(flux_obs)

            flux_obs_to_wh = np.roll(flux_obs, -np.argmax(flux_obs))
            flux_obs_to_wh = np.concatenate((flux_obs_to_wh[:len(flux_obs_to_wh) // 2],
                                             np.zeros(template_support - len(flux_obs_to_wh)), flux_obs_to_wh[len(flux_obs_to_wh) // 2:]))
            wh_flux_obs = np.fft.rfft(flux_obs_to_wh) * effective_wf_fd
            wh_flux_obs = wh_flux_obs / np.linalg.norm(np.fft.irfft(wh_flux_obs))

            return_temp.append(np.array(flux_obs))
            return_wh.append(np.array(wh_flux_obs))
    return return_temp, return_wh


def prepare_fd_for_cos(vec_fd):
    vec_fd = vec_fd / np.linalg.norm(np.fft.irfft(vec_fd))
    support = 2 * (len(vec_fd) - 1)
    # cosine_coeffs = np.ones(support // 2 + 1) * 2
    cosine_coeffs = np.ones(len(vec_fd)) * 2
    cosine_coeffs[0] = 1.
    cosine_coeffs[-1] = 1.
    cosine_coeffs = cosine_coeffs / support
    return np.conj(vec_fd) * cosine_coeffs


def load_volume_prior_precomputed_terms(
        precomputed_tables_location=Precomputed_tables_location,
        log_occurrences_templates_periods_m13r_per_edges_filename=Log_occurrences_templates_periods_m13r_per_edges_filename,
        log_second_derivs_cos2_shifts_templates_filename=Log_second_derivs_cos2_shifts_templates_filename,
        log_cos2_derivs_templates_filename=Log_cos2_derivs_templates_filename):
    """
    log_cos2_derivs_templates : time derivative of template shifting wrt itself,
    log_second_derivs_cos2_shifts_templates : cos squared derivative wrt the template number and
    log_occurrences_templates_periods_m13r_per_edges : the binned occurrence rates with bins.
    log_occurrences_templates_periods is normalized by the m_1_3_r to have the full axtrophysical occurrence rate
    in every m_1_3_r bin. units are rate! no probability normalization!
    :param precomputed_tables_location: in which directory all that resides
    :type precomputed_tables_location:
    :param log_occurrences_templates_periods_m13r_per_edges_filename:
    :type log_occurrences_templates_periods_m13r_per_edges_filename:
    :param log_second_derivs_cos2_shifts_templates_filename:
    :type log_second_derivs_cos2_shifts_templates_filename:
    :param log_cos2_derivs_templates_filename:
    :type log_cos2_derivs_templates_filename:
    :return:
    :rtype:
    """
    log_second_derivs_cos2_shifts_templates = np.load(
        os.path.join(precomputed_tables_location, log_second_derivs_cos2_shifts_templates_filename), allow_pickle=True)
    # -> time derivative of template shifting wrt itself
    log_cos2_derivs_templates = np.load(os.path.join(precomputed_tables_location, log_cos2_derivs_templates_filename),
                                        allow_pickle=True)
    log_occurrences_templates_periods, m13r_bin_edges, period_edges = np.load(
        os.path.join(precomputed_tables_location, log_occurrences_templates_periods_m13r_per_edges_filename), allow_pickle=True)
    # log_occurrences_templates_periods is normalized by the m_1_3_r!
    return log_second_derivs_cos2_shifts_templates, log_cos2_derivs_templates, \
        log_occurrences_templates_periods, m13r_bin_edges, period_edges


def load_template_prior(
        precomputed_tables_location=Precomputed_tables_location,
        log_template_prior_m13r_per_edges_filename=Log_template_prior_m13r_per_edges_filename):
    """
    log_template_prior_periods_m13r_per_edges : the binned normalized occurrence rates with bins.
    normalized to have sum one over the templates in each bin.
    :param precomputed_tables_location: in which directory all that resides
    :type precomputed_tables_location:
    :param log_template_prior_m13r_per_edges_filename:
    :type log_template_prior_m13r_per_edges_filename:
    :return:
    :rtype:
    """
    log_template_prior, m13r_bin_edges, period_edges = np.load(
        os.path.join(precomputed_tables_location, log_template_prior_m13r_per_edges_filename), allow_pickle=True)
    # log_occurrences_templates_periods is normalized by the m_1_3_r!
    return log_template_prior, m13r_bin_edges, period_edges


def calculate_depth_factor(params):
    """
    Calculates how deep is the transit wrt the area part (due to limb darkening and inclination)
    For the search it is not important because we are pnly looking at cosines. For these templates
    there can be different parameters with different depth factor.
    :param params:
    :type params:
    :return:
    :rtype:
    """
    was_many_params = True
    if not hasattr(params[0], '__len__'):
        params = [params]
        was_many_params = False
    rp_1 = 0.2
    depth_factors = []
    tim = time(t_tot=T_ban, parity='odd')[1:]
    for m_1_3_r_s_1, period_1, a_1, inc_1, ecc_1, w_1, u1_1, u2_1 in params:
        flux_best = 1. - flux_par(t=tim, params=None,
                                  per=period_1, rp=rp_1, a=a_1, inc=inc_1, ecc=ecc_1, w=w_1, u1=u1_1, u2=u2_1, t0=0,
                                  supersample_factor=20, exp_time=dt_lc)
        depth_factors.append(np.max(flux_best) / rp_1 ** 2)
    if not was_many_params:
        depth_factors = depth_factors[0]
    return depth_factors


def get_rp_inj(snr_req, wf_fd, n_transits, per_inj=300., m_1_3_r_s=1, inc_inj=90., ecc_inj=0, w_inj=90.,
               u1_inj=Default_limb_darkening_coefs[0], u2_inj=Default_limb_darkening_coefs[1],
               nperseg_psd=128, include_detrending=True, dt=dt_lc):
    """
    Returns the planetary raduis that needs to be injected to the data with the PSD corresponding to a
    whitening filter `wf_fd` and with `n_transits` transits.
    :param snr_req: Required SNR of the entire planet (not squared). The required SNR is the SNR expected from the pipeline assuming
        that the PSD will be measured and the detrending will be (maybe) done.
    :type snr_req: float
    :param wf_fd: whitening filter in the Fourier domain
    :type wf_fd: numpy.ndarray of float
    :param n_transits: required number of transits
    :type n_transits: int
    :param per_inj: Injected perios, days. Is used to determine the shape of the transit, and not the number of them.
    :type per_inj: float
    :param m_1_3_r_s: Mass of the star (in units of solar masses) to the power 1/3 divided by the radius of the
        star (in units of solar radii)
    :type m_1_3_r_s: float
    :param inc_inj: Required inclination, degrees
    :type inc_inj: float
    :param ecc_inj: Required eccentricity
    :type ecc_inj: float
    :param w_inj: Required periastron abgle, degrees
    :type w_inj: float
    :param u1_inj: Required first limb darkening coefficient
    :type u1_inj: float
    :param u2_inj: Required second limb darkening coefficient
    :type u2_inj: float
    :param nperseg_psd: Slice size that will be used to measure the PSD. Is needed to zero all the expected SNR in the low
        frequencies which are not resolved by this slice.
    :type nperseg_psd: int
    :param include_detrending: Whether to take into account that detrending will be done to the data, so that all the frequencies
        which are not resolved by the PSD measurement will have zero power. This will increase the needed radius for a given expected SNR.
    :type include_detrending: bool
    :param dt: Time step, days
    :type dt: float
    :return: rp_inj, injected planet radius in units of stellar radii
    :rtype: float
    """
    rp_generation = 0.02
    from pipeline_parameters import M_sun, G, R_sun
    a_inj = (G / (4 * np.pi ** 2) * (per_inj * 24 * 60 * 60) ** 2) ** (1 / 3) * m_1_3_r_s * (M_sun ** (1 / 3) / R_sun)

    flux_model = 1 - flux_par(per=per_inj, rp=rp_generation, a=a_inj, inc=inc_inj, ecc=ecc_inj, w=w_inj, u1=u1_inj, u2=u2_inj)
    depth_factor = np.max(flux_model) / rp_generation ** 2

    support = 2 * (len(wf_fd) - 1)
    if include_detrending:
        freqs_wf_fd = np.fft.rfftfreq(support, d=dt)
        freq_psd_resolution = 1 / (nperseg_psd * dt)
        wf_fd[:np.searchsorted(freqs_wf_fd, freq_psd_resolution)] = 0.

    flux_model = flux_model / np.nanmax(flux_model) * depth_factor
    flux_model = whiten_vector(flux_model, wf_fd=wf_fd)
    # snr_predic = np.linalg.norm(flux) * n_transits_max ** 0.5 * (rp_rs ** 2)
    rp_inj = (snr_req / (np.linalg.norm(flux_model) * n_transits ** 0.5)) ** 0.5
    return rp_inj


def make_limb_darkening_coefficients_array(limb_darkening_table_file, save_file=False):
    """
    Opens the Table: J/A+A/510/A21/table2.dat.gz (http://cdsarc.cds.unistra.fr)
    and returns the array of [Teff,logg,M/H,a,b], where a, b are the u1, u2 quadratic limb darkening coeffs.
    #Limb_darkening_full_table_file = '/Collaboration-Kepler/KeplerFiles/limb_darkening_kepler_table.txt'
    :param limb_darkening_table_file: The full path to the txt table with coefficietns
    :type limb_darkening_table_file: str
    :return: The array of relevant parameters and the limb darlening quadratic coefficients.
    :rtype: numpy.ndarray
    """
    with open(limb_darkening_table_file, 'r') as fil:
        # 'Table: J/A+A/510/A21/table2.dat.gz (http://cdsarc.cds.unistra.fr)'
        lines = fil.readlines()
        # Teff|logg|  M/H|      u|      a|      b|     c2|     c3|     c4|     a1|     a2|     a3|     a4
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line[0].isnumeric()]
    lines = [line.split('|') for line in lines]
    lines = [[line[0], line[1], line[2], line[4], line[5]] for line in lines]
    # [Teff,logg,M/H,a,b]
    lines = [[float(e.strip()) for e in line] for line in lines]
    lines = np.array(lines)
    lines = lines[lines[:, 0] != 1.]
    if save_file:
        np.save(os.path.join(os.path.dirname(limb_darkening_table_file)), 'limb_darkening_kepler_array.npy')
    return lines


def get_limb_darkening_coefs(teff, logg, metallicity, limb_darkening_array_file=Limb_darkening_array_file,
                             default=Default_limb_darkening_coefs):
    if (teff is None) or (logg is None) or (metallicity is None):
        return default
    if np.isnan(np.array([teff, logg, metallicity])).any():
        return default

    teff_s, logg_s, meh_s, a_s, b_s = np.load(limb_darkening_array_file, allow_pickle=True).T

    meh_set = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    logg_set = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    teff_set = np.array([3500.0, 3750.0, 4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5250.0, 5500.0, 5750.0, 6000.0, 6250.0, 6500.0, 6750.0,
                         7000.0, 7250.0, 7500.0, 7750.0, 8000.0, 8500.0, 8750.0, 9000.0, 9500.0, 9750.0, 10000.0, 10500.0, 11000.0, 11500.0,
                         12000.0, 12500.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
                         23000.0,
                         24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0,
                         37500.0,
                         40000.0, 42500.0, 45000.0, 47500.0, 50000.0])
    metallicity_table = meh_set[np.argmin(np.abs(meh_set - metallicity))]
    logg_table = logg_set[np.argmin(np.abs(logg_set - logg))]
    teff_table = teff_set[np.argmin(np.abs(teff_set - teff))]
    index = np.where((teff_s == teff_table) & (logg_s == logg_table) & (meh_s == metallicity_table))[0]
    index = index[0]
    return a_s[index], b_s[index]



"""
How I prepared the whitened array for injections:

sprt = 256  # we can make 8192, but it will take too much time in the injections code. 
# and also, in the real life, we measure the psd with a smaller resolution. the template length is only 105, so we can take this.
#tb = tm.TemplateBank(just_load=True)
tb.pad_templates(support=sprt)
swffd = sim.get_saved_wf_fd(freqs=np.fft.rfftfreq(sprt, d=tm.dt_lc))
# rolls = [-np.argmax(tt[0]) for tt in tb.templates]
# templates_rolled_left = [np.roll(tt, rolls[tb.templates_of_indices[i]]) for i, tt in enumerate(tb.templates_shifts)]
# -> rolled to the max because it is easier for detection in injections
templates_rolled_left = [np.roll(tt, -np.argmax(tt)) for i, tt in enumerate(tb.templates_shifts)]
# -> roll everyone to the left because how do real injections know what are the shifts? they will be rolled all, too!

templates_whitened_fd = [np.fft.rfft(tt) * swffd for tt in templates_rolled_left]
templates_whitened_fd = [tt/ np.linalg.norm(np.fft.irfft(tt)) for tt in templates_whitened_fd]

# templates_whitened_td = [np.fft.irfft(tt) for tt in templates_whitened_fd] 
# templates_whitened_td = [tt / np.linalg.norm(tt) for tt in templates_whitened_td]  
np.save('/Users/orynaivashtenko/Work/Kepler_Project/Kepler_Earths/WIS_Kepler_pipeline'
        +'/template_bank/templates_whitened_normalized_fd_for_injections.npy', templates_whitened_fd)
"""


"""


"""