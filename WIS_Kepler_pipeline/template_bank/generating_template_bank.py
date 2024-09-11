#!/usr/bin/env python3

import os
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
import auxiliary_functions as af
import simulations.lightcurve_simulation as sim
import file_operations as fop
import utils as ut

import batman


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
                                         np.zeros(template_support-len(flux_obs_to_wh)),
                                         flux_obs_to_wh[len(flux_obs_to_wh)//2:]))
    wh_flux_obs = np.fft.rfft(flux_obs_to_wh) * effective_wf_fd
    wh_flux_obs = wh_flux_obs / np.linalg.norm(np.fft.irfft(wh_flux_obs))
    return wh_flux_obs


M_sun = pp.M_sun
R_sun = pp.R_sun
au = pp.au
G = pp.G
R_sun_R_e = pp.R_sun_R_e

m_1_3_r_kepler = np.load(M_1_3_r_selected_list_file, allow_pickle=True)
mmrr13 = np.linspace(np.min(m_1_3_r_kepler), np.max(m_1_3_r_kepler), 1000)  # want to cover it faster and not to wait for the same things
per_min = pp.P_min_search  # days
per_max = pp.P_max_search  # days
n_w = 10
d_ecc = 0.1
d_inc = 1.0  # this is in degrees, and often until 85 is fine
n_shifts = 3  # we checked that with our parameters this can be 3 for everyone who needs shifts
shift_step = tm.dt_lc / n_shifts
min_cos = 0.97
min_cos_shift = min_cos
us = Limb_darkening_coefs_template_bank_generation

# Getting effective whitening filter in Fourier domain for overlap calculations
template_support = Template_support
dt = tm.dt_lc
freqs = np.fft.rfftfreq(template_support, d=dt)
effective_wf_fd = sim.get_saved_wf_fd(freqs=freqs, dt=tm.dt_lc)

t = tm.time(t_tot=(template_support + 1) * tm.dt_lc, parity='odd')[1:]
# -> make it conj for the ease of cosine
# The following is needed to calculate the correct cosines from the vectors in the Fourier domain. Only works for even length
cosine_coeffs = np.ones(template_support // 2 + 1) * 2
cosine_coeffs[0] = 1.
cosine_coeffs[-1] = 1.
cosine_coeffs = cosine_coeffs / template_support

rp = 0.02  # assume the shape does not depend on the planet radius, although it is not exactly correct
L = 105
m_1_3_r_s = 1
period = 365
a = 215
inc = 90
ecc = 0
w = 90
u1 = 0.47
u2 = 0.23
shifted_temp, wh_roll_shifted_temp = tm.make_template_from_pars(L=L, per=period, a=a, inc=inc, ecc=ecc,
                                                                w=w, u1=u1, u2=u2,
                                                                template_support=template_support,
                                                                min_cos_shift=0.93, n_shifts=3, rp=rp,
                                                                enforce_symmetrization=True,
                                                                dt=tm.dt_lc,
                                                                effective_wf_fd=effective_wf_fd)
wh_roll_shifted_temp = np.conj(np.array(wh_roll_shifted_temp)) * cosine_coeffs

cos_angles_max = []
params_all = []
template_bank_params = [[m_1_3_r_s, period, a, inc, ecc, w, u1, u2]]
templates = [shifted_temp]
templs_prepared_for_cos_fd = wh_roll_shifted_temp
indices_of_templates = [[0, 1]]
templates_of_indices = [k for k in range(len(indices_of_templates)) for _ in range(indices_of_templates[k][0], indices_of_templates[k][1])]
templates_parameter_sets = [[] for k in range(len(templates))]

MAX_NUMBER = 2000000  # Maximal number of simulations
FOR_NEW_TEMPS = 1000000
WAS_SAVED_1 = False  # Saving every once in a while for safety
WAS_SAVED_2 = False
WAS_SAVED_3 = False
N_m_1_3_r = 20000
counter = 0
for jjj, m_1_3_r_s in enumerate(np.concatenate(([1., 0.9, 1.1, 0.8, 1.2],
                                                np.random.uniform(np.min(m_1_3_r_kepler),
                                                                  np.max(m_1_3_r_kepler), N_m_1_3_r - 5)))):
    for period in np.random.uniform(per_min, per_max, 5):
        if jjj < 500:   # First, make sure to sample well the Sun limb darkening
            u = Limb_darkening_coefs_sun
        else:
            u = us[int(np.random.choice(np.arange(len(us)), 1)[0])]
        # Attention, this is a linear uniform distribution, although we are using logarithmic occurrences.
        a = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * m_1_3_r_s * (
                M_sun ** (1 / 3) / R_sun)  # 3rd Kepler law #in units of R
        # a = (G * M /(4*np.pi**2) * (per*24*60*60)**2)**(1/3) / R #3rd Kepler law #in units of R
        # r_min = a*(1-e) -> 2*r_star=a*(1-e_max)
        e_max = min(1 - 2 / a, 0.97)  # for some reason, 0.98 gets computationally stuck in batman sometimes, not clear why
        if e_max < 0:
            e_range = [0.]
        else:
            e_range = np.arange(0, e_max, d_ecc)
            if len(e_range) == 0:
                e_range = [0.]
            else:
                e_range += np.random.uniform(0, d_ecc, len(e_range))
                e_range = e_range[e_range < e_max]
        for ecc in e_range:
            ws = np.linspace(0, 360, n_w, endpoint=False)
            ws += np.random.uniform(0, (ws[1] - ws[0]), len(ws))
            for w in ws:
                i_max = np.arccos(1 / a * (1 + ecc * np.cos((w - 90) / 180 * np.pi)) / (1 - ecc ** 2)) * 180 / np.pi
                incls = np.arange(90, i_max, -d_inc)  # there will be added a random, so it is ok that the final point is not included
                if len(incls) > 1:
                    incls -= np.random.uniform(0, d_inc, len(incls))
                    incls = incls[incls >= i_max]
                for inc in incls:
                    rp = 0.02  # assume the shape does not depend on the planet radius, although it is not exactly correct
                    flux_obs = 1. - tm.flux_par(t=t, params=None, per=period, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u[0], u2=u[1], t0=0,
                                                supersample_factor=20, exp_time=tm.dt_lc)
                    # we built the bank to find only certain durations
                    len_zeros = len(np.where(np.isclose(flux_obs, 0))[0])  # to make sure transit is not wider than the time
                    dur = tm.templ_dur_hmax_fun(flux_obs) * tm.dt_lc
                    if ((np.linalg.norm(flux_obs) > 0) and (Min_template_duration_days <= dur) and (dur <= Max_template_duration_days)
                            and len_zeros >= 4):
                        # -> these were the requirements for the bank on the duration
                        wh_flux_obs = prep_flux_for_cos(flux_obs, effective_wf_fd, enforce_symmetrization=True)
                        snrs_or_cos_angles = [np.real(wh_flux_obs @ templs_prepared_for_cos_fd[j])
                                              for j in range(len(templs_prepared_for_cos_fd))]
                        ind_1 = np.argmax(snrs_or_cos_angles)
                        cos_max = snrs_or_cos_angles[ind_1]
                        if cos_max < min_cos:
                            shifted_temp, wh_roll_shifted_temp = tm.make_template_from_pars(L=L, per=period, a=a, inc=inc, ecc=ecc, w=w,
                                                                                            u1=u[0], u2=u[1],
                                                                                            template_support=template_support,
                                                                                            min_cos_shift=0.93, n_shifts=3, rp=rp,
                                                                                            enforce_symmetrization=True, dt=tm.dt_lc,
                                                                                            effective_wf_fd=effective_wf_fd)
                            templates.append(shifted_temp)
                            wh_roll_shifted_temp = np.conj(np.array(wh_roll_shifted_temp)) * cosine_coeffs
                            templs_prepared_for_cos_fd = np.concatenate((templs_prepared_for_cos_fd, wh_roll_shifted_temp))
                            template_bank_params.append([m_1_3_r_s, period, a, inc, ecc, w, u[0], u[1]])
                            templates_parameter_sets = [[] for k in range(len(templates))]
                            cos_angles_max = []
                            params_all = []
                            indlen = list(np.cumsum([len(s) for s in templates]))
                            indices_of_templates = np.array(list(zip(np.concatenate(([0], indlen)), indlen)))
                            templates_of_indices = [k for k in range(len(indices_of_templates))
                                                    for _ in range(indices_of_templates[k][0], indices_of_templates[k][1])]
                        else:
                            templates_parameter_sets[templates_of_indices[ind_1]].append([m_1_3_r_s, period, a, inc, ecc, w, u[0], u[1]])
                            cos_angles_max.append(cos_max)
                            params_all.append([m_1_3_r_s, period, a, inc, ecc, w, u[0], u[1]])
                        counter += 1

                        if counter > FOR_NEW_TEMPS:
                            min_cos = 0.
                        if counter % 100 == 0:
                            os.system('cls')
                            print(counter, 'sims,', jjj + 1, 'densities,', len(templates_parameter_sets), 'templates,', len(cos_angles_max),
                                  'since last')
                    if counter > MAX_NUMBER:
                        break
                if counter > MAX_NUMBER:
                    break
            if counter > MAX_NUMBER:
                break
        if counter > MAX_NUMBER:
            break
    if (not WAS_SAVED_1 and counter > 700000) or (not WAS_SAVED_2 and counter > 1000000) or (not WAS_SAVED_3 and counter > 1500000):
        # np.save('templates_parameter_sets_prior_for_last_bank.npy', np.array(templates_parameter_sets, dtype=object), allow_pickle=True)
        # np.save('cos_angles_max_last.npy', np.array(cos_angles_max, dtype=float), allow_pickle=True)
        # np.save('params_all_last.npy', np.array(params_all, dtype=object), allow_pickle=True)
        # np.save('template_bank_params_last.npy', np.array(template_bank_params, dtype=object), allow_pickle=True)
        # np.save('templs_prepared_for_cos_fd_last.npy', np.array(templs_prepared_for_cos_fd, dtype=np.complex128), allow_pickle=True)
        # np.save('templates_last.npy', np.array(templates, dtype=object), allow_pickle=True)
        # np.save('indices_of_templates_last.npy', np.array(indices_of_templates, dtype=object), allow_pickle=True)
        if not WAS_SAVED_1:
            WAS_SAVED_1 = True
        elif not WAS_SAVED_2:
            WAS_SAVED_2 = True
        elif not WAS_SAVED_3:
            WAS_SAVED_3 = True
    if counter > MAX_NUMBER:
        break

# TODO!
# np.save('templates_parameter_sets_prior_for_last_bank.npy', np.array(templates_parameter_sets, dtype=object), allow_pickle=True)
# np.save('cos_angles_max_last.npy', np.array(cos_angles_max, dtype=float), allow_pickle=True)
# np.save('params_all_last.npy', np.array(params_all, dtype=object), allow_pickle=True)
# np.save('template_bank_params_last.npy', np.array(template_bank_params, dtype=object), allow_pickle=True)
# np.save('templs_prepared_for_cos_fd_last.npy', np.array(templs_prepared_for_cos_fd, dtype=np.complex128), allow_pickle=True)
# np.save('templates_last.npy', np.array(templates, dtype=object), allow_pickle=True)
# np.save('indices_of_templates_last.npy', np.array(indices_of_templates, dtype=object), allow_pickle=True)
#
#
# # MAKING WHITENED TEMPLATES
# tb_1 = copy.deepcopy(tb)
# sprt = 256  # we can make 8192, but it will take too much time in the injections code.
# # and also, in the real life, we measure the psd with a smaller resolution. the template length is only 105, so we can take this.
# #tb = tm.TemplateBank(just_load=True)
# tb_1.pad_templates(support=sprt)
# swffd = sim.get_saved_wf_fd(freqs=np.fft.rfftfreq(sprt, d=tm.dt_lc))
# # rolls = [-np.argmax(tt[0]) for tt in tb.templates]
# # templates_rolled_left = [np.roll(tt, rolls[tb.templates_of_indices[i]]) for i, tt in enumerate(tb.templates_shifts)]
# # -> rolled to the max because it is easier for detection in injections
# templates_rolled_left = [np.roll(tt, -np.argmax(tt)) for i, tt in enumerate(tb_1.templates_shifts)]
# # -> roll everyone to the left because how do real injections know what are the shifts? they will be rolled all, too!
#
# templates_whitened_fd = [np.fft.rfft(tt) * swffd for tt in templates_rolled_left]
# templates_whitened_fd = [tt/ np.linalg.norm(np.fft.irfft(tt)) for tt in templates_whitened_fd]
#
# # templates_whitened_td = [np.fft.irfft(tt) for tt in templates_whitened_fd]
# # templates_whitened_td = [tt / np.linalg.norm(tt) for tt in templates_whitened_td]