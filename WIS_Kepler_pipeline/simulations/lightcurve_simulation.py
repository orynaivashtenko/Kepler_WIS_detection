#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 12:16:38 2021

@author: orynaivashtenko
"""

import numpy as np
# import sys
import os
# import search_pipeline as pip
import template_bank.template_module as tm
import auxiliary_functions as af
import file_operations as fop
import utils as ut
from pipeline_parameters import Pipeline_folder, Simulations_folder_name, \
    Default_stored_asd_file_filename, Default_effective_wf_fd_filename

dt_lc = tm.dt_lc
Default_stored_asd_file_dir = os.path.join(Pipeline_folder, Simulations_folder_name, Default_stored_asd_file_filename)
Default_stored_wf_fd_file_dir = os.path.join(Pipeline_folder, Simulations_folder_name, Default_effective_wf_fd_filename)


def get_saved_psd(freqs=None, dt=tm.dt_lc, file_dir=Default_stored_asd_file_dir):
    """

    :param freqs: the frequencies that we want to get  in the end
    :type freqs: ndarray(float)
    :param dt: The sampling rate that was allegedly used in the saved fft. By default, the Kepler long cadence.
    :type dt: float
    :param file_dir:
    :type file_dir:
    :return:
    :rtype:
    """
    rfft_med = np.array(fop.read_csv_file(file_dir))
    fr = np.fft.rfftfreq(2 * (len(rfft_med) - 1), d=dt)
    if freqs is not None:
        psd = np.abs(np.interp(freqs, fr, rfft_med))**2
    else:
        psd = rfft_med**2
    # wf_saved = 1 / rfft_med
    # # THIS WAS OLD, NOW NO NEED
    # #apply_tukey:
    # support = 8192, nperseg = 128
    # # We do this because this is what will probably happen in the pipeline, and we want to make sure that after this, the cosines
    # # will be also fine. This influences a lot the number of the templates!
    # first_measured_bin = 2 * int(np.ceil(support / nperseg))  # 8192 IS THE LENGTH OF THE SUPPORT, 128 IS THE NPERSEG
    # lentuk = first_measured_bin * 4  # 4 * first_measured_bin
    # wf_saved[:2 * first_measured_bin] = wf_saved[:2 * first_measured_bin] * ut.tukey(lentuk, alpha=3 * first_measured_bin / lentuk,
    #                                                                                      sym=True)[:2 * first_measured_bin]
    return psd


def get_saved_wf_fd(freqs=None, dt=dt_lc, file_dir=Default_stored_wf_fd_file_dir):
    wf_fd = np.load(file_dir, allow_pickle=True)
    fr = np.fft.rfftfreq(2 * (len(wf_fd) - 1), d=dt)
    if freqs is not None:
        wf_fd = np.abs(np.interp(freqs, fr, wf_fd))
    return wf_fd


def get_red_noise(support, n_vectors=0, dt=tm.dt_lc, psd='saved', return_psd=False):
    """
    Simulate the simplest red noise with the psd by default saved in the file
    :param support:
    :type support:
    :param n_vectors: If 0, then one vector will be given. If >=1, then it will be a matrix
    :type n_vectors:
    :param dt: Time resolution (sampling rate) of the `psd`
    :type dt: float
    :param psd:
    :type psd:
    :return:
    :rtype:
    """
    freqs = np.fft.rfftfreq(support, d=dt)
    if psd == 'saved':
        psd = get_saved_psd(freqs, dt=dt)
    if n_vectors > 0:
        noise = np.random.normal(0, 1, (n_vectors, support))
        f_noise = np.fft.rfft(noise, axis=1, norm='ortho')
        noise = np.fft.irfft(f_noise * psd**0.5, axis=1, norm='ortho')
    else:
        noise = np.random.normal(0, 1, support)
        f_noise = np.fft.rfft(noise, norm='ortho')
        noise = np.fft.irfft(f_noise * psd**0.5, norm='ortho')
    if not return_psd:
        return noise
    else:
        return noise, psd


def get_simulated_star(n_quarters=14, mean_flux=1e4, sigma_high_rel=1e-4, tmin=131, t_between=1.,
                       inject_planet=False, period=100, first_transit_center=141, rp=0.02, a=215., inc=90., ecc=0., w=90., u1=0.47,
                       u2=0.23, snr_required=None):
    lenquart = int(91 / tm.dt_lc)
    noises = []
    noise, psd = get_red_noise(support=lenquart, n_vectors=0, return_psd=True)
    for i in range(n_quarters):
        noise, psd = get_red_noise(support=lenquart, n_vectors=0, return_psd=True)
        noises.append(noise - np.nanmean(noise))
    sigma_there = np.mean(np.abs(np.fft.rfft(noises[0], norm='ortho')[-len(noises[0]) // 4:]) ** 2) ** 0.5
    factor = sigma_high_rel * mean_flux / sigma_there
    noises = [noises[i] * factor + mean_flux for i in range(len(noises))]
    psd_ortho_there = psd * factor ** 2

    times = [np.arange(len(noises[i])) * tm.dt_lc for i in range(len(noises))]
    for i in range(1, len(times)):
        times[i] = times[i] + times[i - 1][-1] + tm.dt_lc + t_between
    times = [times[i] + tmin for i in range(len(noises))]

    if inject_planet:
        snr = None
        fluxes = None
        for _ in range(1 if snr_required is None else 2):  # if there is an snr request, do it twice to adjust the snr
            fluxes, transits = inject_planet_to_data(times=times, fluxes=noises,
                                                     period=period, first_transit_center=first_transit_center, rp=rp, a=a,
                                                     inc=inc, ecc=ecc, w=w, u1=u1, u2=u2,
                                                     supersample_factor=20, return_template=True)
            transits = [transits[i] * mean_flux
                        for i in range(len(transits))]
            snr = 0
            for transit in transits:
                # getting expected snr
                if len(np.where(transit > 0)[0]) > 0:
                    transit = np.roll(transit, -np.argmax(transit))
                    snr_1 = np.sum(np.fft.irfft(np.fft.rfft(transit, norm='ortho') / psd_ortho_there ** 0.5, norm='ortho') ** 2) ** 0.5
                    snr = (snr ** 2 + snr_1 ** 2) ** 0.5
            if snr_required is None:
                snr_required = snr
            rp = rp * (snr_required / snr)**0.5

        return times, fluxes, snr
    else:
        return times, noises


def inject_planet_to_data(times=None, fluxes=None,
                          period=100., first_transit_center=500, rp=0.01, a=215., inc=90., ecc=0., w=90., u1=0.47, u2=0.23,
                          supersample_factor=20, return_template=False):
    """

    :param times:
    :type times:
    :param fluxes:
    :type fluxes:
    :param period:
    :type period:
    :param first_transit_center:
    :type first_transit_center:
    :param rp:
    :type rp:
    :param a:
    :type a:
    :param inc:
    :type inc:
    :param ecc:
    :type ecc:
    :param w:
    :type w:
    :param u1:
    :type u1:
    :param u2:
    :type u2:
    :param supersample_factor:
    :type supersample_factor:
    :param return_template:
    :type return_template:
    :return:
    :rtype:
    """
    times = [np.array(tt) for tt in times]  # should work like copy...
    fluxes = [np.array(ff) for ff in fluxes]

    times_int = [ut.interpolate_time(times[i]) for i in range(len(times))]
    dt = np.nanmean([np.nanmean((t - np.roll(t, 1))[1:]) for t in times_int])
    if first_transit_center - times_int[0][0] >= period:
        first_transit_center = times_int[0][0] + (first_transit_center - times_int[0][0]) % period

    transits_quarters = []
    for q in range(len(times)):
        trans_quart = tm.flux_par(t=times_int[q], per=period, rp=rp, a=a, inc=inc,
                                  ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center,
                                  supersample_factor=supersample_factor, exp_time=dt)
        trans_quart = 1. - trans_quart
        transits_quarters.append(trans_quart)

    for q in range(len(fluxes)):
        fluxes[q] -= transits_quarters[q] * np.nanmedian(fluxes[q])

    if return_template:
        return fluxes, transits_quarters
    else:
        return fluxes


# def generate_lc_quarters(t_tot=1400, t_min=0., t_max=None, dt=tm.dt_lc, flux0=1e5,
#                          source='stored_psd', asd_max=10000., asd_min=7.5, pow_asd=-3., freq_shift=0.02e-5,
#                          variations_relative=0.05, variations_absolute=1e3, std_mean_quarters=0.1, sensitivity_std=0.1,
#                          std_read_relative=0.,
#                          planet=False, period=100., template_num=20, transit_duration=1., first_transit_center=50,
#                          use_physical_parameters=False, rp=0.01, a=215., inc=90., ecc=0., w=90., u1=0.47, u2=0.23,
#                          supersampling=True, integration_points=20, n_quarters=17,
#                          snr=7., depth_ppm=None, depth_by_rp=False,
#                          return_template=False, return_psd=False):
#     """
#     PARAMETERS:
#     t_tot : float, days
#         The length of time of data
#
#     flux0 : float
#         Mean flux level
#     source : str : 'stored_psd', 'stored_fft', 'white_noise' or anything else
#         Source for the noise model.
#         If 'white noise': the white noise with parameters specified further.
#         If 'stored_psd' : will use some sampled PSD of the real star.
#         If 'stored_fft' : will use some fft of the real star.
#         Else : will use power law model for red noise.
#     asd_max : float
#         Maximal value of the ASD of the red noise in the power law model.
#     asd_min : float
#         Minimal value of the ASD of the red noise in the power law model.
#     pow_asd : float
#         Power of the power law in the red noise model.
#     freq_shift : float
#         Shift in the power law in the red noise model (without this shift,
#         there will be infinity at zero frequency)
#     variations_relative=0.05,
#     variations_absolute=1e3,
#     std_mean_quarters=0.1,
#     sensitivity_std=0.1,
#     # std_read_relative_to_variations=0.05,
#     planet=False, period=100.,
#     template_num=20,
#     transit_duration=100,
#     first_transit_center=500,
#     snr=7.,
#     n_quarters=17
#     ----------------------
#     RETURNS:
#     pixel_data : np.ndarray of the shape (t_tot, N, N)
#
#     """
#     if t_min is None:
#         t_min = 0.
#     first_transit_center = (1. * first_transit_center - t_min) % period
#     # for now, shift everything to zero beginning, will correct in the end
#     if t_max is not None:
#         t_tot = t_max - t_min
#
#     len_tot = int(t_tot / dt)
#     if len_tot % 2 != 0:
#         len_tot -= 1
#         t_tot -= dt
#     # go to absolute units...
#
#     # 2)generating ASD of the star
#     if source == 'stored_smoothed_asd_from_fft' and np.abs((dt - tm.dt_lc) / tm.dt_lc) < 0.05:
#         asd_smooth = np.array(fop.read_csv_file(os.path.join(os.path.dirname(__file__), 'stored_smoothed_asd_from_fft.txt')))
#         fr = np.fft.rfftfreq(2 * (len(asd_smooth) - 1), d=dt)
#         freqs = np.fft.rfftfreq(len_tot, d=dt)
#         asd = np.abs(np.interp(freqs, fr, asd_smooth))
#     elif source == 'stored_fft' and np.abs((dt - tm.dt_lc) / tm.dt_lc) < 0.05:
#         rfft_med = np.array(fop.read_csv_file(os.path.join(os.path.dirname(__file__), 'stored_fft.txt')))
#         fr = np.fft.rfftfreq(2 * (len(rfft_med) - 1), d=dt)
#         freqs = np.fft.rfftfreq(len_tot, d=dt)
#         asd = np.abs(np.interp(freqs, fr, rfft_med))
#     elif source == 'stored_psd' and np.abs((dt - tm.dt_lc) / tm.dt_lc) < 0.05:
#         psd_saved = np.array(fop.read_csv_file(os.path.join(os.path.dirname(__file__), 'stored_psd.txt')))
#         fr = np.fft.rfftfreq(2 * (len(psd_saved) - 1), d=dt)
#         freqs = np.fft.rfftfreq(len_tot, d=dt)
#         asd = np.abs(np.interp(freqs, fr, psd_saved)) ** 0.5
#     elif source == 'white_noise':
#         asd = np.ones(len_tot // 2 + 1)  # this will multiply the rfft of the white gaussian noise, so the
#         # normalization by length should not be applied
#     else:
#         freqs = np.fft.rfftfreq(len_tot, d=dt)  # 1/day
#         asd = np.zeros(len(freqs))
#         asd = (freqs / 60 / 60 / 24 + freq_shift) ** pow_asd
#         mm = np.max(asd)
#         m = np.min(asd)
#         asd = asd * ((asd_max - asd_min) / (mm - m)) + (mm * asd_min - asd_max * m) / (
#                                       mm - m)  # adjusting asd to have specified max and min
#
#     var_dat = 1 / ((len(asd) - 1) * 2) * (
#             np.sum(np.abs(asd) ** 2) + np.sum(np.abs(asd[1:-1]) ** 2))  # what would be the variance of data with this asd
#     # (there is no factor (1/len...) from normalization because data is rfft of the white noise multiplied by asd,
#     # so the normalization is authomatic
#     if variations_relative and flux0 != 0:  # if variations_relative is not None or False, there is a priority for it
#         asd = asd / var_dat ** 0.5  # we will give data different normalization
#         asd = asd * flux0 * variations_relative
#     elif variations_absolute:  # if there are some absolute variations specified
#         asd = asd / var_dat ** 0.5
#         asd = asd * variations_absolute
#     data = np.real(np.fft.irfft(np.fft.rfft(np.random.normal(0, 1, len_tot)) * asd, len_tot))
#     data = data - np.mean(data) + flux0
#     if std_read_relative is not None and std_read_relative != 0.:
#         data += std_read_relative * flux0 * np.random.normal(0, 1, len(data))
#     if flux0 != 0:
#         asd[0] = flux0 * len_tot
#     psdfull = np.concatenate((asd ** 2, asd[1:-1][::-1] ** 2))
#     psdfull[0] = np.inf
#
#     time = np.arange(len_tot) * dt  # for now, shift everything to zero beginning, will correct in the end
#
#     # 3)generating template
#     if planet:
#         if first_transit_center >= period:
#             first_transit_center = first_transit_center % period
#
#         if use_physical_parameters:
#             if supersampling:
#                 transiting_planet = tm.flux_par(time, per=period, rp=rp, a=a, inc=inc,
#                                                 ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center,
#                                                 supersample_factor=integration_points, exp_time=dt / integration_points)
#             else:
#                 transiting_planet = tm.flux_par(time, per=period, rp=rp, a=a, inc=inc,
#                                                 ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center)
#             transiting_planet = -(transiting_planet - np.max(transiting_planet))
#
#         else:
#             if template_num is None:
#                 widths = tm.Durations
#                 template_num = np.argmin(np.abs(widths - transit_duration))
#             P = period
#             dP = dt * P / t_tot  # at least should cover dP in the level of fine-graining
#             n_points_int_min = 50  # minimal number of initial points for one point after integrative interpolation
#             nn = max(int(np.ceil(n_points_int_min * dP / dt)), 1)
#             dt_int = dP / nn  # time step in the time sample used for integration
#             t_int = tm.time(dt=dt_int)
#             t = tm.time(dt=dt)
#             t_int = np.concatenate(([t_int[0] - dt_int], t_int, [t_int[-1] + dt_int]))
#             template_int = tm.TemplateBank(dt=dt_int, just_load=True).templates_shifts[tm.Indices_of_templates[template_num][0]]
#             template_int = np.concatenate(([0.], template_int, [0.]))
#             transiting_planet = np.zeros(len_tot)
#             for i in np.arange(int((t_tot - first_transit_center) / P) + 1):
#                 shift = (first_transit_center + i * P) % dt
#                 transit = tm.integ_interp(t, t_int, np.roll(template_int, int(np.round(shift / dt_int))))[1]
#                 tr_index = int(np.floor((first_transit_center + i * P) / dt))
#
#                 if (tr_index - len(transit) // 2) >= 0 and (tr_index + (
#                         len(transit) - 1) // 2) <= len_tot - 1:
#                   ATTENTION: if the distance is not enough, the edge transits can be missed!
#                     transiting_planet[tr_index - (len(transit)) // 2: tr_index + (len(transit) + 1) // 2] = np.array(transit)
#                 elif (tr_index - len(transit) // 2) >= 0:
#                     missed_space = (tr_index + (len(transit) - 1) // 2) - (len_tot - 1)
#                     transiting_planet[tr_index - (len(transit)) // 2:] = np.array(transit)[:len(transit) - missed_space]
#                 elif (tr_index + (
#                         len(transit) - 1) // 2) <= len_tot - 1:
#                       ATTENTION: if the distance is not enough, the edge transits can be missed!
#                     missed_space = len(transit) // 2 - tr_index
#                     transiting_planet[: tr_index + (len(transit) + 1) // 2] = np.array(transit)[missed_space:]
#
#         if (snr is not None) and (not depth_by_rp):
#             transiting_planet = transiting_planet / np.linalg.norm(transiting_planet)
#             denom_stat_f = np.abs(
#                 np.sum(np.abs(np.fft.fft(transiting_planet)) ** 2 / psdfull / len_tot)) ** 0.5  # equivalent to the real space formula
#
#             # transiting_planet = transiting_planet/np.linalg.norm(transiting_planet)
#             # whitening_filter = 1/psdfull**0.5
#             # whitening_filter[0] = 0.
#             # transit_whitened = np.fft.fft(transiting_planet)*whitening_filter
#             # denom_stat_f = np.abs(np.sum(np.abs(transit_whitened)**2/len_tot))**0.5 #equivalent to the real space formula
#
#             amplitude = - snr / denom_stat_f
#             # The signal is not exactly multiplicative, because we subtract from the star some part of this star which is not exactly
#             # proportional to the other parts of the flux. For example, there can be spot etc. In any case, the amplitude is defined with
#             # respect to the psd. But the mean flux level added in the end does not play here.
#             data += amplitude * transiting_planet
#
#         elif (not depth_by_rp) and (depth_ppm is not None):
#             data += transiting_planet / np.max(np.abs(transiting_planet)) * flux0 * depth_ppm / 10 ** 6
#
#         else:
#             data += transiting_planet * flux0  # depth is set byt the radius
#
#     quarters_sensitivity = np.random.normal(1, sensitivity_std, n_quarters)
#     quarters_additive = np.random.normal(0, std_mean_quarters, n_quarters) * flux0
#     quarter_len = len(data) // n_quarters
#     if len(data) % n_quarters != 0:
#         quarter_len += 1
#         quarter_0_len = len(data) - quarter_len * (n_quarters - 1)
#     else:
#         quarter_0_len = quarter_len
#     quarters = [data[:quarter_0_len] * quarters_sensitivity[0] + quarters_additive[0]]
#     time += t_min
#     times = [np.arange(quarter_0_len) * dt + t_min]
#     for i in range(0, n_quarters - 1):
#         quarters.append(
#             data[quarter_0_len + i * quarter_len:quarter_0_len + (1 + i) * quarter_len] * quarters_sensitivity[i] + quarters_additive[i])
#         # times.append(np.arange(quarter_len)*dt+(quarter_0_len+(i-1)*quarter_len)*dt)
#         times.append(time[quarter_0_len + i * quarter_len:quarter_0_len + (1 + i) * quarter_len])
#     if planet and return_template:
#         if return_psd:
#             return times, quarters, transiting_planet, psdfull
#         else:
#             return times, quarters, transiting_planet
#     else:
#         if return_psd:
#             return times, quarters, psdfull
#         else:
#             return times, quarters


# def generate_injection_general(times=None, fluxes=None, get_data=False, target_id=4158779, data_location='',
#                                nperseg=128, nsig_outliers=3, detlen=500, normalize_by_level=True,
#                                period=100., template_num=20, transit_duration_days=0.1, first_transit_center=500,
#                                use_physical_parameters=True, rp=0.01, a=215., inc=90., ecc=0., w=90., u1=0.47, u2=0.23,
#                                supersample_factor=20, snr=7., depth_ppm=None, depth_by_rp=False,
#                                return_template=False, return_wfs=False, return_zero_padding_ends=False):
#     """
#     NOT DEBUGGED, DOES NOT WORK FOR SURE
#
#     PARAMETERS:
#     T : float, days
#         The length of time of data
#
#     flux0 : float
#         Mean flux level
#     source : str : 'stored_psd', 'stored_fft', 'white_noise' or anything else
#         Source for the noise model.
#         If 'white noise': the white noise with parameters specified further.
#         If 'stored_psd' : will use some sampled PSD of the real star.
#         If 'stored_fft' : will use some fft of the real star.
#         Else : will use power law model for red noise.
#     asd_max : float
#         Maximal value of the ASD of the red noise in the power law model.
#     asd_min : float
#         Minimal value of the ASD of the red noise in the power law model.
#     pow_asd : float
#         Power of the power law in the red noise model.
#     freq_shift : float
#         Shift in the power law in the red noise model (without this shift,
#         there will be infinity at zero frequency)
#     variations_relative=0.05,
#     variations_absolute=1e3,
#     std_mean_quarters=0.1,
#     sensitivity_std=0.1,
#     # std_read_relative_to_variations=0.05,
#     planet=False, period=100.,
#     template_num=20,
#     transit_duration=100,
#     first_transit_center=500,
#     snr=7.,
#     n_quarters=17
#     ----------------------
#     RETURNS:
#     fluxes and more, see description
#
#     """
#     if get_data:
#         assert target_id is not None, 'Please specify target ID!'
#         times, fluxes = af.get_data_fun(target_id, location=data_location)
#     times = [np.array(tt) for tt in times]  # should work like copy...
#     fluxes = [np.array(ff) for ff in fluxes]
#
#     dts = []
#     for q in range(len(times)):
#         dts.append(np.nanmean((times[q] - np.roll(times[q], 1))[1:]))
#     dt = np.nanmean(dts)
#     times_int = [ut.interpolate_time(times[i]) for i in range(len(times))]
#     # time = np.arange(times_int[0][0], times_int[-1][-1]+dt)*dt
#
#     if first_transit_center - times_int[0][0] >= period:
#         first_transit_center = times_int[0][0] + (first_transit_center - times_int[0][0]) % period
#
#     if depth_by_rp or (snr is None):
#         transits_quarters = []
#         for q in range(len(times)):
#             if use_physical_parameters:
#                 trans_quart = tm.flux_par(times_int[q], per=period, rp=rp, a=a, inc=inc,
#                                           ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center,
#                                           supersample_factor=supersample_factor, exp_time=None)
#             else:
#                 if template_num is None:
#                     widths = tm.Durations  # days
#                     template_num = np.argmin(np.abs(widths - transit_duration_days))
#                 params = tm.get_default_templates_params()[tm.Indices_of_templates[template_num][0]]
#                 # m_1_3_r_s, period, a, ecc, w, inc
#                 trans_quart = tm.flux_par(times_int[q], per=params[1], rp=rp, a=params[2], inc=params[5], ecc=params[3], w=params[4],
#                                           u1=tm.Limb_darkening_coefs_sun[0], u2=tm.Limb_darkening_coefs_sun[1], t0=first_transit_center,
#                                           supersample_factor=supersample_factor, exp_time=None)
#             trans_quart = 1. - trans_quart
#             if (not depth_by_rp) and (depth_ppm is not None):
#                 trans_quart = trans_quart * depth_ppm / 10 ** 6
#             transits_quarters.append(trans_quart)
#
#         for q in range(len(fluxes)):
#             fluxes[q] -= transits_quarters[q] * np.nanmean(fluxes[q])
#
#         if return_template:
#             return fluxes, transits_quarters
#         else:
#             return fluxes
#
#     else:
#         transits_quarters = []
#         mov_medians = []
#         whitened_transits_quarters = []
#         wf_fds = []
#         padded_transits_quarters = []
#         padding_zerlens = []
#         tmax = np.max(np.array([len(t) for t in times]))  # maximum length of the time
#         # the closest (but bigger) power of 2 to the length        Tmax += tm.L_ban
#         support = tmax + nperseg * 4
#         if support % 2 != 0:
#             support += 1
#
#         for q in range(len(times)):
#             if use_physical_parameters:
#                 trans_quart = 1 - tm.flux_par(times_int[q], per=period, rp=rp, a=a, inc=inc,
#                                               ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center,
#                                               supersample_factor=supersample_factor, exp_time=None)
#             else:
#                 if template_num is None:
#                     widths = tm.Durations  # days
#                     template_num = np.argmin(np.abs(widths - transit_duration_days))
#
#                 params = tm.get_default_templates_params()[tm.Indices_of_templates[template_num][0]]
#                 # m_1_3_r_s, period, a, ecc, w, inc
#                 trans_quart = tm.flux_par(times_int[q], per=params[1], rp=rp, a=params[2], inc=params[5], ecc=params[3], w=params[4],
#                                           u1=tm.Limb_darkening_coefs_sun[0], u2=tm.Limb_darkening_coefs_sun[1], t0=first_transit_center,
#                                           supersample_factor=supersample_factor, exp_time=None)
#             transits_quarters.append(trans_quart)
#             # padding to avoid crosstalks in whitening and to make even support
#             padded_transits_quarter, zerlens = ut.pad_by_zeros(trans_quart, support=support, return_zero_padding_ends=True)
#             padded_transits_quarters.append(padded_transits_quarter)
#             padding_zerlens.append(zerlens)
#
#         normalization = np.linalg.norm(np.concatenate(transits_quarters))
#         transits_quarters = [transits_quarters[q] / normalization for q in range(len(transits_quarters))]
#         padded_transits_quarters = [padded_transits_quarters[q] / normalization for q in range(len(padded_transits_quarters))]
#
#         #     padded_transits_quarters = copy.deepcopy(transits_quarters)
#         #     padding_zerlens=[(0,0) for i in range(len(padding_zerlens))]
#         fluxes_detr = []
#         masks = []
#         for q in range(len(times)):
#             detlen_q = detlen
#             mask = ~np.isnan(fluxes[q])
#             if detlen_q > len(fluxes[q]) // 3:
#                 detlen_q = len(fluxes[q]) // 3 + (len(fluxes[q]) // 3) % 2
#             mov_median_left = [np.nanmedian(fluxes[q][0:i + detlen_q // 2]) for i in np.arange(0, detlen_q // 2)]
#             mov_median = [np.nanmedian(fluxes[q][i - detlen_q // 2:i + detlen_q // 2]) for i in
#                           np.arange(detlen_q // 2, len(fluxes[q]) - detlen_q // 2)]
#             mov_median_right = [np.nanmedian(fluxes[q][i - detlen_q // 2:]) for i in
#                                 np.arange(len(fluxes[q]) - detlen_q // 2, len(fluxes[q]))]
#             mov_median = np.concatenate((mov_median_left, mov_median, mov_median_right))
#             mov_medians.append(mov_median)
#             mask *= np.abs(fluxes[q] - mov_median) <= nsig_outliers * np.nanstd(fluxes[q] - mov_median)
#             if np.prod(mask) == 0:
#                 flux_interp = ut.interpolate_over_mask(np.array(fluxes[q]), mask=mask)
#             else:
#                 flux_interp = fluxes[q]
#             mask = ut.clean_small_holes(mask, max_hole=5)
#             masks.append(mask)
#             fluxes_detr.append(flux_interp - mov_median)
#         psds, wf_fds, wf_tds = ut.measure_psds_func(fluxes, masks=masks, nperseg=nperseg, noverlap=nperseg//2, window='hann',
#                                                     support=support, subtract_mean=True, rfft=True, smoothen_psds=True,
#                                                     tukey_steps_len=16, conditional_zero_zeroth_freq=True, always_zero_zeroth_freq=False,
#                                                     zero_to_high_freq_thresh_wf_ratio=0.6, dt=dt)
#
#         whitened_transits_quarters = [np.fft.rfft(padded_transits_quarters[q]) * wf_fds[q] for q in range(len(fluxes))]
#         #         whitened_transits_quarters.append(whit_pad_trans[padding_zerlens[q][0]:len(whit_pad_trans)-padding_zerlens[q][-1]])
#
#         L = np.sum([len(whitened_transits_quarters[q]) for q in range(len(whitened_transits_quarters))])
#         denom_stat_f = (np.sum(
#             [np.sum(np.abs(whitened_transits_quarters[q]) ** 2) for q in range(len(whitened_transits_quarters))]) / L) ** 0.5
#         amplitude = - snr / denom_stat_f
#         if normalize_by_level:
#             total_median = np.nanmedian(np.concatenate(fluxes))
#             if total_median / np.nanstd(np.concatenate(fluxes)) < 0.1:
#                 normalize_by_level = False  # if the mean level is too low, ignore it to avoin indtabilities
#
#         for q in range(len(fluxes)):
#             if normalize_by_level:
#                 fluxes[q] += transits_quarters[q] * mov_medians[q] / total_median * amplitude
#             else:
#                 fluxes[q] += transits_quarters[q] * amplitude
#
#         # WARNING:
#         # The PSD used for injection is measured -> there will be no loss due to error in measurement!
#         # It orig_means that the injected SNR is not exactly the stated.
#         # As well, the mean levels are measured, so the same issue.
#         if return_zero_padding_ends:
#             if return_template:
#                 if return_wfs:
#                     return fluxes, transits_quarters, wf_fds, return_zero_padding_ends
#                 else:
#                     return fluxes, transits_quarters, return_zero_padding_ends
#             else:
#                 if return_wfs:
#                     return fluxes, wf_fds, return_zero_padding_ends
#                 else:
#                     return fluxes, return_zero_padding_ends
#         else:
#             if return_template:
#                 if return_wfs:
#                     return fluxes, transits_quarters, wf_fds
#                 else:
#                     return fluxes, transits_quarters
#             else:
#                 if return_wfs:
#                     return fluxes, wf_fds
#                 else:
#                     return fluxes


def rangen(pars, len_tot, distx=0, disty=0):
    """
    generates random vector from given set of values pars
    of given length L
    with given numerical distribution (distx, disty)

    Parameters
    ----------
    pars : TYPE
        DESCRIPTION.
    len_tot : TYPE
        DESCRIPTION.
    distx : TYPE, optional
        DESCRIPTION. The default is 0.
    disty : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if type(distx) == int:
        distx = pars
        disty = np.ones(len(pars)) / np.sum(pars)
    if pars[0] < distx[0] or pars[-1] > distx[-1]:
        print('inconsistent dimensions')
        return []
    probdis = np.interp(pars, distx, disty)
    probdis = probdis / np.sum(probdis)
    return np.random.choice(pars, len_tot, replace=True, p=probdis)

