#!/usr/bin/env python3

import os

from locations import Default_template_bank_folder

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num # export NUMEXPR_NUM_THREADS=6
import numpy as np

import pipeline_parameters as pp
import template_bank.template_module as tm
from template_bank.template_bank_generation_parameters import *
from template_bank.looking_at_masses_and_radii import get_masses_radii_densities_selection
import auxiliary_functions as af
import simulations.lightcurve_simulation as sim
import file_operations as fop
import utils as ut
import locations as locs
import copy


def prep_flux_for_cos(flux_obs, effective_wf_fd, enforce_symmetrization=True, pad=False, template_support=Template_support):
    flux_obs_0 = np.array(flux_obs)
#     flux_obs = np.roll(flux_obs, -(np.argmax(flux_obs) - (len(flux_obs)-1)//2))
    flux_obs = np.roll(flux_obs, -np.argmax(flux_obs))
    if enforce_symmetrization:
        l2 = (len(flux_obs)+1) // 2
        s1 = np.sum(flux_obs[1:l2])
        s2 = np.sum(flux_obs[-l2+1:])
        if s1 < s2:
            flux_obs = flux_obs_0[::-1]
        else:
            flux_obs = flux_obs_0
    flux_obs_to_wh = np.roll(flux_obs, -np.argmax(flux_obs))
    if pad:
        flux_obs_to_wh = np.concatenate((flux_obs_to_wh[:len(flux_obs_to_wh)//2],
                                         np.zeros(template_support-len(flux_obs_to_wh)), flux_obs_to_wh[len(flux_obs_to_wh)//2:]))
    wh_flux_obs = np.fft.rfft(flux_obs_to_wh) * effective_wf_fd
    wh_flux_obs = wh_flux_obs / np.linalg.norm(np.fft.irfft(wh_flux_obs))
    return wh_flux_obs


def get_fft_cosine_coeffs(template_support):
    """Is needed to calculate the correct cosines from the vectors in the Fourier domain. Only works for even length"""
    cosine_coeffs = np.ones(template_support // 2 + 1) * 2
    cosine_coeffs[0] = 1.
    cosine_coeffs[-1] = 1.
    cosine_coeffs = cosine_coeffs / template_support
    return cosine_coeffs


def filter_out_rare_templates(templates, template_bank_params, templates_parameter_sets, min_part=1e-5):
    counts_temps = np.array([len(templates_parameter_sets[i]) for i in range(len(templates_parameter_sets))])
    # saving only the templates with more than a certain number of triggers, because we will not find too rare planets in our star samlpe
    indices_to_save = np.where(counts_temps > np.sum(counts_temps) * min_part)[0]
    # -> we checked that it will also be okay when we split tht densities into bins
    templates = [templates[ind] for ind in indices_to_save]
    template_bank_params = [template_bank_params[ind] for ind in indices_to_save]
    templates_parameter_sets = [templates_parameter_sets[ind] for ind in indices_to_save]
    return templates, template_bank_params, templates_parameter_sets


def create_whitened_templates(tb=None, template_support=Template_support, template_bank_folder=Default_template_bank_folder,
                              save_to_file=False, templates_fd_for_injections_filename=locs.Default_templates_fd_for_injections_filename,
                              return_results=True):
    # we can make 8192, but it will take too much time in the injections code.
    # and also, in the real life, we measure the psd with a smaller resolution. the template length is only 105, so we can take this.
    # MAKING WHITENED TEMPLATES
    if tb is not None:
        tb_1 = copy.deepcopy(tb)
    else:
        tb_1 = tm.TemplateBank(just_load=True, template_bank_folder=template_bank_folder)
    tb_1.pad_templates(support=template_support)
    swffd = sim.get_saved_wf_fd(freqs=np.fft.rfftfreq(template_support, d=tm.dt_lc))
    # rolls = [-np.argmax(tt[0]) for tt in tb.templates]
    # templates_rolled_left = [np.roll(tt, rolls[tb.templates_of_indices[i]]) for i, tt in enumerate(tb.templates_shifts)]
    # -> rolled to the max because it is easier for detection in injections
    templates_rolled_left = [np.roll(tt, -np.argmax(tt)) for i, tt in enumerate(tb_1.templates_shifts)]
    # -> roll everyone to the left because how do real injections know what are the shifts? they will be rolled all, too!
    templates_whitened_fd = [np.fft.rfft(tt) * swffd for tt in templates_rolled_left]
    templates_whitened_fd = [tt / np.linalg.norm(np.fft.irfft(tt)) for tt in templates_whitened_fd]
    # templates_whitened_td = [np.fft.irfft(tt) for tt in templates_whitened_fd]
    # templates_whitened_td = [tt / np.linalg.norm(tt) for tt in templates_whitened_td]
    templates_whitened_fd = create_whitened_templates(tb=tb, save_to_file=save_to_file, return_results=True,
                                                      template_bank_folder=template_bank_folder,
                                                      templates_fd_for_injections_filename=templates_fd_for_injections_filename)
    if save_to_file:
        np.save(os.path.join(str(template_bank_folder), str(templates_fd_for_injections_filename)),
                np.array(templates_whitened_fd), allow_pickle=True)
    if return_results:
        return templates_whitened_fd


# --------------------------------------------------------------------------------------------------------------------
Build_new_bank = False
# --------------------------------------------------------------------------------------------------------------------
Default_templates_fd_for_injections_filename = locs.Default_templates_fd_for_injections_filename
Default_template_bank_folder = locs.Default_template_bank_folder
L = tm.L_ban
M_sun = pp.M_sun
R_sun = pp.R_sun
au = pp.au
G = pp.G
R_sun_R_e = pp.R_sun_R_e
# --------------------------------------------------------------------------------------------------------------------


def run_monte_carlo(build_new_bank=Build_new_bank, max_number=5000000,  # Maximal number of simulations
                    n_m_1_3_r=20000, template_bank_folder=Default_template_bank_folder,
                    m_1_3_r_selected_list_file=M_1_3_r_selected_list_file):
    m_1_3_r_kepler = np.load(m_1_3_r_selected_list_file, allow_pickle=True)
    per_min = Per_min  # days
    per_max = Per_max  # days
    n_w = N_w
    n_ecc = N_ecc
    n_cos_incl = N_cos_incl
    rp = Rp  # assume the shape does not depend on the planet radius, although it is not exactly correct
    n_shifts = N_shifts  # we checked that with our parameters this can be 3 for everyone who needs shifts
    min_cos = Min_cos
    min_cos_shift = Min_cos_shift
    us = Limb_darkening_coefs_template_bank_generation

    # Getting effective whitening filter in Fourier domain for overlap calculations
    template_support = Template_support
    dt = tm.dt_lc
    freqs = np.fft.rfftfreq(template_support, d=dt)
    effective_wf_fd = sim.get_saved_wf_fd(freqs=freqs, dt=tm.dt_lc)
    t = tm.time(t_tot=(template_support + 1) * tm.dt_lc, parity='odd')[1:]
    # -> make it conj for the ease of cosine
    # The following is needed to calculate the correct cosines from the vectors in the Fourier domain. Only works for even length
    cosine_coeffs = get_fft_cosine_coeffs(template_support=template_support)

    cos_angles_max = []
    params_all = []
    if not build_new_bank:
        templs_prepared_for_cos_fd = np.conj(
            np.load(os.path.join(template_bank_folder, Default_templates_fd_for_injections_filename), allow_pickle=True))
        # -> make it conj for the ease of cosine
        # The following is needed to calculate the correct cosines from the vectors in the Fourier domain. Only works for even length
        templs_prepared_for_cos_fd = templs_prepared_for_cos_fd * cosine_coeffs
        templates_of_indices = tm.Templates_of_indices
        templates_parameter_sets = [[] for k in range(len(tm.Indices_of_templates))]
        templates = None
        template_bank_params = None
        indices_of_templates = None
    else:
        m_1_3_r_s, period, a, inc, ecc, w, u1, u2 = Earth_default_parameters
        shifted_temp, wh_roll_shifted_temp = tm.make_template_from_pars(L=L, per=period, a=a, inc=inc, ecc=ecc,
                                                                        w=w, u1=u1, u2=u2,
                                                                        template_support=template_support,
                                                                        min_cos_shift=min_cos_shift, n_shifts=n_shifts, rp=rp,
                                                                        enforce_symmetrization=True,
                                                                        dt=tm.dt_lc,
                                                                        effective_wf_fd=effective_wf_fd)
        wh_roll_shifted_temp = np.conj(np.array(wh_roll_shifted_temp)) * cosine_coeffs
        template_bank_params = [[m_1_3_r_s, period, a, inc, ecc, w, u1, u2]]
        templates = [shifted_temp]
        templs_prepared_for_cos_fd = wh_roll_shifted_temp
        indices_of_templates = [[0, 1]]
        templates_of_indices = [k for k in range(len(indices_of_templates)) for _ in range(
            indices_of_templates[k][0], indices_of_templates[k][1])]
        templates_parameter_sets = [[] for _ in range(len(templates))]

    saving_numbers = [int(max_number*0.12), int(max_number*0.25), int(max_number*0.5)]
    _was_saved_1 = False  # Saving every once in a while for safety
    _was_saved_2 = False
    _was_saved_3 = False

    counter = 0
    for jjj, m_1_3_r_s in enumerate(np.concatenate(([1., 0.9, 1.1, 0.8, 1.2],
                                                    np.random.uniform(np.min(m_1_3_r_kepler),
                                                                      np.max(m_1_3_r_kepler), n_m_1_3_r - 5)))):
        for period in np.random.uniform(per_min, per_max, 5):
            if jjj < 500:   # First, make sure to sample well the Sun limb darkening
                u = Limb_darkening_coefs_sun
            else:
                u = us[int(np.random.choice(np.arange(len(us)), 1)[0])]
            # Attention, this is a linear uniform distribution, although we are using logarithmic occurrences.
            a = pp.get_a_from_p(period=period, m_1_3_r_s=m_1_3_r_s)  # 3rd Kepler law #in units of R
            # r_min = a*(1-e) -> 2*r_star=a*(1-e_max)
            e_max = min(1 - 2 / a, 0.97)
            assert e_max > 0, 'Semi-major axis is too small or stellar radius is too large!'
            # The eccentricity distribution is normalized to unity in the allowed range.
            # There can be no planets that clash the star, so it is uniform in the allowed range.
            e_range = ut.get_uniform_random_sequence(0, e_max, n_ecc)
            e_range = e_range[e_range <= e_max]
            for ecc in e_range:
                w_range = ut.get_uniform_random_sequence(0, 360, n_w)
                for w in w_range:
                    cos_i_max = 1 / a * (1 + ecc * np.cos((w - 90) / 180 * np.pi)) / (1 - ecc ** 2)  # ) * 180 / np.pi
                    cos_i_range = ut.get_uniform_random_sequence(0, 1, n_cos_incl)
                    cos_i_range = cos_i_range[cos_i_range <= cos_i_max]
                    incls = np.arccos(cos_i_range) * 180 / np.pi
                    for inc in incls:
                        flux_obs = 1. - tm.flux_par(t=t, params=None, per=period, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u[0], u2=u[1],
                                                    t0=0, supersample_factor=20, exp_time=tm.dt_lc)
                        # we built the bank to find only certain durations
                        len_zeros = len(np.where(np.isclose(flux_obs, 0))[0])  # to make sure transit is not wider than the time
                        dur = tm.templ_dur_hmax_fun(flux_obs) * tm.dt_lc
                        if (np.linalg.norm(flux_obs) > 0) and (Min_template_duration_days <= dur) and (
                                dur <= Max_template_duration_days) and len_zeros >= 4:
                            # -> these were the requirements for the bank on the duration
                            wh_flux_obs = prep_flux_for_cos(flux_obs, effective_wf_fd, enforce_symmetrization=True)
                            snrs_or_cos_angles = [np.real(wh_flux_obs @ templs_prepared_for_cos_fd[j])
                                                  for j in range(len(templs_prepared_for_cos_fd))]
                            ind_1 = np.argmax(snrs_or_cos_angles)
                            cos_max = snrs_or_cos_angles[ind_1]
                            if cos_max < min_cos and build_new_bank:
                                shifted_temp, wh_roll_shifted_temp = tm.make_template_from_pars(
                                    L=L, per=period, a=a, inc=inc, ecc=ecc, w=w, u1=u[0], u2=u[1], template_support=template_support,
                                    min_cos_shift=min_cos_shift, n_shifts=n_shifts, rp=rp, enforce_symmetrization=True, dt=tm.dt_lc,
                                    effective_wf_fd=effective_wf_fd)
                                templates.append(shifted_temp)
                                wh_roll_shifted_temp = np.conj(np.array(wh_roll_shifted_temp)) * cosine_coeffs
                                templs_prepared_for_cos_fd = np.concatenate((templs_prepared_for_cos_fd, wh_roll_shifted_temp))
                                template_bank_params.append([m_1_3_r_s, period, a, inc, ecc, w, u[0], u[1], dur])
                                templates_parameter_sets = [[] for k in range(len(templates))]
                                cos_angles_max = []
                                params_all = []
                                indlen = list(np.cumsum([len(s) for s in templates]))
                                indices_of_templates = np.array(list(zip(np.concatenate(([0], indlen)), indlen)))
                                templates_of_indices = [k for k in range(len(indices_of_templates))
                                                        for _ in range(indices_of_templates[k][0], indices_of_templates[k][1])]
                            else:
                                templates_parameter_sets[templates_of_indices[ind_1]].append(
                                    [m_1_3_r_s, period, a, inc, ecc, w, u[0], u[1]])
                                cos_angles_max.append(cos_max)
                                params_all.append([m_1_3_r_s, period, a, inc, ecc, w, u[0], u[1], dur])
                            counter += 1
                            if counter % 100 == 0:
                                os.system('cls')
                                print(counter, 'sims,', jjj + 1, 'densities,', len(templates_parameter_sets),
                                      'templates,', len(cos_angles_max), 'since last')
                        if counter > max_number:
                            break
                    if counter > max_number:
                        break
                if counter > max_number:
                    break
            if counter > max_number:
                break
        if ((not _was_saved_1 and counter > saving_numbers[0]) or (not _was_saved_2 and counter > saving_numbers[1])
                or (not _was_saved_3 and counter > saving_numbers[2])) or counter >= max_number:
            if build_new_bank:
                np.save(Templates_parameter_sets_file, np.array(templates_parameter_sets, dtype=object), allow_pickle=True)
                np.save(Cos_angles_max_file, np.array(cos_angles_max, dtype=float), allow_pickle=True)
                np.save(Params_all_file, np.array(params_all, dtype=object), allow_pickle=True)
                np.save(Templates_bank_parameters_file, np.array(template_bank_params, dtype=object), allow_pickle=True)
                np.save(Templs_prepared_for_cos_fd_file, np.array(templs_prepared_for_cos_fd, dtype=np.complex128), allow_pickle=True)
                np.save(Templates_file, np.array(templates, dtype=object), allow_pickle=True)
                np.save(Indices_of_templates_file, np.array(indices_of_templates, dtype=object), allow_pickle=True)
            else:
                np.save(Templates_parameter_sets_prior_file, np.array(templates_parameter_sets, dtype=object), allow_pickle=True)
                np.save(Cos_angles_max_prior_file, np.array(cos_angles_max, dtype=float), allow_pickle=True)
                np.save(Params_all_prior_file, np.array(params_all, dtype=object), allow_pickle=True)
            if not _was_saved_1:
                _was_saved_1 = True
            elif not _was_saved_2:
                _was_saved_2 = True
            elif not _was_saved_3:
                _was_saved_3 = True
        if counter > max_number:
            break


def construct_template_bank_from_simulation(templates_parameter_sets_file=Templates_parameter_sets_file,
                                            templates_bank_parameters_file=Templates_bank_parameters_file,
                                            templates_file=Templates_file,
                                            template_bank_folder=Default_template_bank_folder,
                                            templates_fd_for_injections_filename=Default_templates_fd_for_injections_filename,
                                            return_results=True, save_to_file=False):
    templates_parameter_sets = np.load(templates_parameter_sets_file, allow_pickle=True)
    template_bank_params = np.load(templates_bank_parameters_file, allow_pickle=True)
    templates = np.load(templates_file, allow_pickle=True)
    # indices_of_templates = np.load(Indices_of_templates_file, allow_pickle=True)
    templates, template_bank_params, templates_parameter_sets = filter_out_rare_templates(
        templates=templates, template_bank_params=template_bank_params, templates_parameter_sets=templates_parameter_sets)
    # indlen = list(np.cumsum([len(s) for s in templates]))
    # indices_of_templates = np.array(list(zip(np.concatenate(([0], indlen)), indlen)))
    # templates_of_indices = [k for k in range(len(indices_of_templates))
    #                         for _ in range(indices_of_templates[k][0], indices_of_templates[k][1])]
    tb = tm.TemplateBank(templates=templates,
                         parameters=template_bank_params,
                         template_index='all', shifts=None, dt=tm.dt_lc,
                         enforce_centering=True,
                         enforce_same_length=False,
                         enforce_normalization=False,
                         enforce_sorting=True,
                         just_load=False,
                         load_essentials=False,
                         template_bank_folder=template_bank_folder
                         )
    templates_whitened_fd = create_whitened_templates(tb=tb, save_to_file=save_to_file, return_results=True,
                                                      template_bank_folder=template_bank_folder,
                                                      templates_fd_for_injections_filename=templates_fd_for_injections_filename)
    if save_to_file:
        tb.save_bank()
        tb.save_parameters()
        np.save(os.path.join(str(template_bank_folder), str(templates_fd_for_injections_filename)),
                np.array(templates_whitened_fd), allow_pickle=True)
    if return_results:
        return tb, templates_whitened_fd


def calculate_occurrence_priors_from_simulation(
        templates_simulation_folder=Templates_preparation_folder, return_results=True, save_to_file=False,
        log_occurrences_templates_periods_m13r_per_edges_filename=locs.Log_occurrences_templates_periods_m13r_per_edges_filename,
        fie_dir=locs.Precomputed_tables_location,
        templates_parameter_sets_prior_filename=Templates_parameter_sets_prior_filename,
        m13r_bin_edges=M13r_bin_edges_template_prior
        # cos_angles_max_prior_filename=Cos_angles_max_prior_filename,
        # params_all_prior_filename=Params_all_prior_filename
        ):
    # # Calculating occurrence rates from the simulations
    templates_parameter_sets = np.load(
        os.path.join(templates_simulation_folder, templates_parameter_sets_prior_filename), allow_pickle=True)
    # cos_angles_max = np.load(os.path.join(templates_simulation_folder, cos_angles_max_prior_filename), allow_pickle=True)
    # params_all = np.load(os.path.join(templates_simulation_folder, params_all_prior_filename), allow_pickle=True)

    # Opening the previously known rates for periods and radii
    log2_rp_re_edges, log10_per_edges, occurrences_table = pp.get_occurrence_table_enlarged()
    log10_per_edges = log10_per_edges[1:]
    occurrences_table = occurrences_table[1:]
    period_edges = 10 ** log10_per_edges
    occurrences_per_periods = np.sum(occurrences_table, axis=1)
    # -> in principle, it is incorrect to sum because we are putting together all the radii
    # but for now we are doing this because we just want to compare scores of different periods and templates

    # Making a table for templates occurrence rates with stellar density bins and period bins
    # We normalize every density bin separately. We do not account for different occurrence rates of different densities.
    # The comparison between periods and templates will be made inside one star, so we need to select the corresponding bin.
    # Also, the occurrences are anyway not well-measured for F,G,K...
    m13r_bin_edges = np.array(m13r_bin_edges)
    occurrences_templates_periods = np.zeros((len(m13r_bin_edges) - 1, len(period_edges) - 1, len(tm.Indices_of_templates)))
    for bin_ind in range(len(period_edges) - 1):
        for m13_bin_ind in range(len(m13r_bin_edges) - 1):
            for tt in range(len(tm.Indices_of_templates)):
                parss = np.array(templates_parameter_sets[tt])
                parss = parss[(parss[:, 1] > period_edges[bin_ind]) & (parss[:, 1] <= period_edges[bin_ind + 1])
                              & (parss[:, 0] > m13r_bin_edges[m13_bin_ind]) & (parss[:, 0] <= m13r_bin_edges[m13_bin_ind + 1])]
                occurrences_templates_periods[m13_bin_ind, bin_ind, tt] = len(parss)  # / len(params_per) * occurrences_per_periods[bin_ind]
            occurrences_templates_periods[m13_bin_ind, bin_ind][occurrences_templates_periods[m13_bin_ind, bin_ind] == 0] += 1e-3
            total_counts_per = np.sum(occurrences_templates_periods[m13_bin_ind, bin_ind])
            occurrences_templates_periods[m13_bin_ind, bin_ind] = (occurrences_templates_periods[m13_bin_ind, bin_ind] / total_counts_per
                                                                   * occurrences_per_periods[bin_ind])

    log_occurrences_templates_periods = np.log(occurrences_templates_periods)
    if save_to_file:
        np.save(os.path.join(str(fie_dir), log_occurrences_templates_periods_m13r_per_edges_filename),
                np.array([log_occurrences_templates_periods, m13r_bin_edges, period_edges], dtype=object), allow_pickle=True)
    if return_results:
        return np.array([log_occurrences_templates_periods, m13r_bin_edges, period_edges], dtype=object)


if __name__ == '__main__':
    run_monte_carlo(build_new_bank=True, max_number=1000000,  # Maximal number of simulations
                    n_m_1_3_r=5000)
    construct_template_bank_from_simulation(template_bank_folder=Default_template_bank_folder,
                                            save_to_file=True, return_results=False)
    run_monte_carlo(build_new_bank=False, max_number=5000000,  # Maximal number of simulations
                    n_m_1_3_r=20000)
    calculate_occurrence_priors_from_simulation(save_to_file=True, return_results=False)
