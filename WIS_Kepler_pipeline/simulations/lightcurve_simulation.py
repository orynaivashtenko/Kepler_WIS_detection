#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 12:16:38 2021

@author: orynaivashtenko
"""

import numpy as np
import sys
import os
import search_pipeline as pip
import templates.template_module as tm
import auxiliary_functions as af
import file_operations as fop

dt_lc = tm.dt_lc


def generate_lc_quarters(T=1400, t_min=0., t_max=None, dt=tm.dt_lc, flux0=1e5,
                         source='stored_psd', asd_max=10000., asd_min=7.5, pow_asd=-3., freq_shift=0.02e-5,
                         variations_relative=0.05, variations_absolute=1e3, std_mean_quarters=0.1, sensitivity_std=0.1,
                         std_read_relative=0.,
                         planet=False, period=100., template_num=20, transit_duration=1., first_transit_center=50,
                         use_physical_parameters=False, rp=0.01, a=215., inc=90., ecc=0., w=90., u1=0.47, u2=0.23,
                         supersampling=True, integration_points=20, n_quarters=17,
                         snr=7., depth_ppm=None, depth_by_rp=False,
                         return_template=False, return_psd=False):
    """
    PARAMETERS:
    T : float, days
        The length of time of data

    flux0 : float
        Mean flux level
    source : str : 'stored_psd', 'stored_fft', 'white_noise' or anything else
        Source for the noise model.
        If 'white noise': the white noise with parameters specified further.
        If 'stored_psd' : will use some sampled PSD of the real star.
        If 'stored_fft' : will use some fft of the real star.
        Else : will use power law model for red noise.
    asd_max : float
        Maximal value of the ASD of the red noise in the power law model.
    asd_min : float
        Minimal value of the ASD of the red noise in the power law model.
    pow_asd : float
        Power of the power law in the red noise model.
    freq_shift : float
        Shift in the power law in the red noise model (without this shift,
        there will be infinity at zero frequency)
    variations_relative=0.05,
    variations_absolute=1e3,
    std_mean_quarters=0.1,
    sensitivity_std=0.1,
    # std_read_relative_to_variations=0.05,
    planet=False, period=100.,
    template_num=20,
    transit_duration=100,
    first_transit_center=500,
    snr=7.,
    n_quarters=17
    ----------------------
    RETURNS:
    pixel_data : np.ndarray of the shape (T, N, N)

    """
    if t_min is None:
        t_min = 0.
    first_transit_center = (1. * first_transit_center - t_min) % period  # for now, shift everything to zero beginning, will correct in the end
    if t_max is not None:
        T = t_max - t_min

    L = int(T / dt)
    if L % 2 != 0:
        L -= 1
        T -= dt
    # go to absolute units...

    # 2)generating ASD of the star
    if source == 'stored_smoothed_asd_from_fft' and np.abs((dt - tm.dt_lc) / tm.dt_lc) < 0.05:
        asd_smooth = np.array(fop.read_csv_file(os.path.join(os.path.dirname(__file__), 'stored_smoothed_asd_from_fft.txt')))
        fr = np.fft.rfftfreq(2 * (len(asd_smooth) - 1), d=dt)
        freqs = np.fft.rfftfreq(L, d=dt)
        asd = np.abs(np.interp(freqs, fr, asd_smooth))
    elif source == 'stored_fft' and np.abs((dt - tm.dt_lc) / tm.dt_lc) < 0.05:
        rfft_med = np.array(fop.read_csv_file(os.path.join(os.path.dirname(__file__), 'stored_fft.txt')))
        fr = np.fft.rfftfreq(2 * (len(rfft_med) - 1), d=dt)
        freqs = np.fft.rfftfreq(L, d=dt)
        asd = np.abs(np.interp(freqs, fr, rfft_med))
    elif source == 'stored_psd' and np.abs((dt - tm.dt_lc) / tm.dt_lc) < 0.05:
        psd_saved = np.array(fop.read_csv_file(os.path.join(os.path.dirname(__file__), 'stored_psd.txt')))
        fr = np.fft.rfftfreq(2 * (len(psd_saved) - 1), d=dt)
        freqs = np.fft.rfftfreq(L, d=dt)
        asd = np.abs(np.interp(freqs, fr, psd_saved)) ** 0.5
    elif source == 'white_noise':
        asd = np.ones(L // 2 + 1)  # this will multiply the rfft of the white gaussian noise, so the
        # normalization by length should not be applied
    else:
        freqs = np.fft.rfftfreq(L, d=dt)  # 1/day
        asd = np.zeros(len(freqs))
        asd = (freqs / 60 / 60 / 24 + freq_shift) ** pow_asd
        M = np.max(asd)
        m = np.min(asd)
        asd = asd * ((asd_max - asd_min) / (M - m)) + (M * asd_min - asd_max * m) / (M - m)  # adjusting asd to have specified max and min

    var_dat = 1 / ((len(asd) - 1) * 2) * (
            np.sum(np.abs(asd) ** 2) + np.sum(np.abs(asd[1:-1]) ** 2))  # what would be the variance of data with this asd
    # (there is no factor (1/len...) from normalization because data is rfft of the white noise multiplied by asd,
    # so the normalization is authomatic
    if variations_relative and flux0 != 0:  # if variations_relative is not None or False, there is a priority for it
        asd = asd / var_dat ** 0.5  # we will give data different normalization
        asd = asd * flux0 * variations_relative
    elif variations_absolute:  # if there are some absolute variations specified
        asd = asd / var_dat ** 0.5
        asd = asd * variations_absolute
    data = np.real(np.fft.irfft(np.fft.rfft(np.random.normal(0, 1, L)) * asd, L))
    data = data - np.mean(data) + flux0
    if std_read_relative is not None and std_read_relative != 0.:
        data += std_read_relative * flux0 * np.random.normal(0, 1, len(data))
    if flux0 != 0:
        asd[0] = flux0 * L
    psdfull = np.concatenate((asd ** 2, asd[1:-1][::-1] ** 2))
    psdfull[0] = np.inf

    time = np.arange(L) * dt  # for now, shift everything to zero beginning, will correct in the end

    # 3)generating template
    if planet:
        if first_transit_center >= period:
            first_transit_center = first_transit_center % period

        if use_physical_parameters:
            if supersampling:
                transiting_planet = tm.flux_par(time, per=period, rp=rp, a=a, inc=inc,
                                                ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center,
                                                supersample_factor=integration_points, exp_time=dt / integration_points)
            else:
                transiting_planet = tm.flux_par(time, per=period, rp=rp, a=a, inc=inc,
                                                ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center)
            transiting_planet = -(transiting_planet - np.max(transiting_planet))

        else:
            if template_num is None:
                widths = tm.durations(tm.time(dt=dt), tm.template_bank(dt=dt))
                template_num = np.argmin(np.abs(widths - transit_duration))
            P = period
            dP = dt * P / T  # at least should cover dP in the level of fine-graining
            n_points_int_min = 50  # minimal number of initial points for one point after integrative interpolation
            nn = max(int(np.ceil(n_points_int_min * dP / dt)), 1)
            dt_int = dP / nn  # time step in the time sample used for integration
            t_int = tm.time(dt=dt_int)
            t = tm.time(dt=dt)
            t_int = np.concatenate(([t_int[0] - dt_int], t_int, [t_int[-1] + dt_int]))
            template_int = tm.template_bank(dt=dt_int)[template_num]
            template_int = np.concatenate(([0.], template_int, [0.]))
            transiting_planet = np.zeros(L)
            for i in np.arange(int((T - first_transit_center) / P) + 1):
                shift = (first_transit_center + i * P) % dt
                transit = tm.integ_interp(t, t_int, np.roll(template_int, int(np.round(shift / dt_int))))[1]
                tr_index = int(np.floor((first_transit_center + i * P) / dt))

                if (tr_index - len(transit) // 2) >= 0 and (tr_index + (
                        len(transit) - 1) // 2) <= L - 1:  # ATTENTION: if the distance is not enough, the edge transits can be missed!
                    transiting_planet[tr_index - (len(transit)) // 2: tr_index + (len(transit) + 1) // 2] = np.array(transit)
                elif (tr_index - len(transit) // 2) >= 0:
                    missed_space = (tr_index + (len(transit) - 1) // 2) - (L - 1)
                    transiting_planet[tr_index - (len(transit)) // 2:] = np.array(transit)[:len(transit) - missed_space]
                elif (tr_index + (
                        len(transit) - 1) // 2) <= L - 1:  # ATTENTION: if the distance is not enough, the edge transits can be missed!
                    missed_space = len(transit) // 2 - tr_index
                    transiting_planet[: tr_index + (len(transit) + 1) // 2] = np.array(transit)[missed_space:]

        if (snr is not None) and (not depth_by_rp):
            transiting_planet = transiting_planet / np.linalg.norm(transiting_planet)
            denom_stat_f = np.abs(
                np.sum(np.abs(np.fft.fft(transiting_planet)) ** 2 / psdfull / L)) ** 0.5  # equivalent to the real space formula

            # transiting_planet = transiting_planet/np.linalg.norm(transiting_planet)
            # whitening_filter = 1/psdfull**0.5
            # whitening_filter[0] = 0.
            # transit_whitened = np.fft.fft(transiting_planet)*whitening_filter
            # denom_stat_f = np.abs(np.sum(np.abs(transit_whitened)**2/L))**0.5 #equivalent to the real space formula

            amplitude = - snr / denom_stat_f
            # The signal is not exactly multiplicative, because we subtract from the star some part of this star which is not exactly
            # proportional to the other parts of the flux. For example, there can be spot etc. In any case, the amplitude is defined with
            # respect to the psd. But the mean flux level added in the end does not play here.
            data += amplitude * transiting_planet

        elif (not depth_by_rp) and (depth_ppm is not None):
            data += transiting_planet / np.max(np.abs(transiting_planet)) * flux0 * depth_ppm / 10 ** 6

        else:
            data += transiting_planet * flux0  # depth is set byt the radius

    quarters_sensitivity = np.random.normal(1, sensitivity_std, n_quarters)
    quarters_additive = np.random.normal(0, std_mean_quarters, n_quarters) * flux0
    quarter_len = len(data) // n_quarters
    if len(data) % n_quarters != 0:
        quarter_len += 1
        quarter_0_len = len(data) - quarter_len * (n_quarters - 1)
    else:
        quarter_0_len = quarter_len
    quarters = [data[:quarter_0_len] * quarters_sensitivity[0] + quarters_additive[0]]
    time += t_min
    times = [np.arange(quarter_0_len) * dt + t_min]
    for i in range(0, n_quarters - 1):
        quarters.append(
            data[quarter_0_len + i * quarter_len:quarter_0_len + (1 + i) * quarter_len] * quarters_sensitivity[i] + quarters_additive[i])
        # times.append(np.arange(quarter_len)*dt+(quarter_0_len+(i-1)*quarter_len)*dt)
        times.append(time[quarter_0_len + i * quarter_len:quarter_0_len + (1 + i) * quarter_len])
    if planet and return_template:
        if return_psd:
            return times, quarters, transiting_planet, psdfull
        else:
            return times, quarters, transiting_planet
    else:
        if return_psd:
            return times, quarters, psdfull
        else:
            return times, quarters


def generate_injection(times=None, fluxes=None, get_data=False, target_id=4158779, data_location='',
                       nperseg=128, nsig_outliers=3, detlen=500, normalize_by_level=True,
                       period=100., template_num=20, transit_duration=1., first_transit_center=500,
                       use_physical_parameters=True, rp=0.01, a=215., inc=90., ecc=0., w=90., u1=0.47, u2=0.23,
                       supersampling=True, integration_points=20,
                       snr=7., depth_ppm=None, depth_by_rp=False,
                       return_template=False, return_wfs=False, return_zero_padding_ends=False):
    """
    PARAMETERS:
    T : float, days
        The length of time of data

    flux0 : float
        Mean flux level
    source : str : 'stored_psd', 'stored_fft', 'white_noise' or anything else
        Source for the noise model.
        If 'white noise': the white noise with parameters specified further.
        If 'stored_psd' : will use some sampled PSD of the real star.
        If 'stored_fft' : will use some fft of the real star.
        Else : will use power law model for red noise.
    asd_max : float
        Maximal value of the ASD of the red noise in the power law model.
    asd_min : float
        Minimal value of the ASD of the red noise in the power law model.
    pow_asd : float
        Power of the power law in the red noise model.
    freq_shift : float
        Shift in the power law in the red noise model (without this shift,
        there will be infinity at zero frequency)
    variations_relative=0.05,
    variations_absolute=1e3,
    std_mean_quarters=0.1,
    sensitivity_std=0.1,
    # std_read_relative_to_variations=0.05,
    planet=False, period=100.,
    template_num=20,
    transit_duration=100,
    first_transit_center=500,
    snr=7.,
    n_quarters=17
    ----------------------
    RETURNS:
    fluxes and more, see description

    """
    if get_data:
        assert target_id is not None, 'Please specify target ID!'
        times, fluxes = pip.get_data_fun(target_id, location=data_location)
    times = [np.array(tt) for tt in times]  # should work like copy...
    fluxes = [np.array(ff) for ff in fluxes]

    dts = []
    for q in range(len(times)):
        dts.append(np.nanmean((times[q] - np.roll(times[q], 1))[1:]))
    dt = np.nanmean(dts)
    times_int = [pip.interpolate_time(times[i]) for i in range(len(times))]
    # time = np.arange(times_int[0][0], times_int[-1][-1]+dt)*dt

    if first_transit_center - times_int[0][0] >= period:
        first_transit_center = times_int[0][0] + (first_transit_center - times_int[0][0]) % period

    if (not depth_by_rp) and (snr is not None):
        transits_quarters = []
        mov_medians = []
        whitened_transits_quarters = []
        wffds = []
        padded_transits_quarters = []
        padding_zerlens = []
        for q in range(len(times)):
            if use_physical_parameters:
                if supersampling:
                    trans_quart = 1 - tm.flux_par(times_int[q], per=period, rp=rp, a=a, inc=inc,
                                                  ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center,
                                                  supersample_factor=integration_points, exp_time=dt / integration_points)
                else:
                    trans_quart = tm.flux_par(times_int[q], per=period, rp=rp, a=a, inc=inc,
                                              ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center)
            else:
                if template_num is None:
                    widths = tm.durations(tm.time(dt=dt), tm.template_bank(dt=dt))
                    template_num = np.argmin(np.abs(widths - transit_duration))
                trans_quart = 1 - tm.flux_par(prms=tm.template_bank_parameter_list[template_num])
            transits_quarters.append(trans_quart)
            # padding to avoid crosstalks in whitening and to make even support
            padded_transits_quarter, zerlens = af.pad_by_zeros(trans_quart, minimal_extension=nperseg, return_zero_padding_ends=True)
            padded_transits_quarters.append(padded_transits_quarter)
            padding_zerlens.append(zerlens)

        normalization = np.linalg.norm(np.concatenate(transits_quarters))
        transits_quarters = [transits_quarters[q] / normalization for q in range(len(transits_quarters))]
        padded_transits_quarters = [padded_transits_quarters[q] / normalization for q in range(len(padded_transits_quarters))]

        #     padded_transits_quarters = copy.deepcopy(transits_quarters)
        #     padding_zerlens=[(0,0) for i in range(len(padding_zerlens))]
        for q in range(len(times)):
            detlen_q = detlen
            mask = ~np.isnan(fluxes[q])
            if detlen_q > len(fluxes[q]) // 3:
                detlen_q = len(fluxes[q]) // 3 + (len(fluxes[q]) // 3) % 2
            mov_median_left = [np.nanmedian(fluxes[q][0:i + detlen_q // 2]) for i in np.arange(0, detlen_q // 2)]
            mov_median = [np.nanmedian(fluxes[q][i - detlen_q // 2:i + detlen_q // 2]) for i in
                          np.arange(detlen_q // 2, len(fluxes[q]) - detlen_q // 2)]
            mov_median_right = [np.nanmedian(fluxes[q][i - detlen_q // 2:]) for i in
                                np.arange(len(fluxes[q]) - detlen_q // 2, len(fluxes[q]))]
            mov_median = np.concatenate((mov_median_left, mov_median, mov_median_right))
            mov_medians.append(mov_median)
            mask *= np.abs(fluxes[q] - mov_median) <= nsig_outliers * np.nanstd(fluxes[q] - mov_median)
            if np.prod(mask) == 0:
                flux_interp = pip.interpolate_over_mask(np.array(fluxes[q]), mask=mask)
                frqs, psd_quarter = af.measure_psd(flux_interp, rfft=False, nperseg=nperseg)
            else:
                frqs, psd_quarter = af.measure_psd(fluxes[q], rfft=False, nperseg=nperseg)
            wf_td, wf_fd = af.psd2wf(psd_quarter, support=len(padded_transits_quarters[q]), rfft=False)
            wffds.append(wf_fd)
            whit_pad_trans = np.fft.fft(padded_transits_quarters[q]) * wf_fd
            #         whitened_transits_quarters.append(whit_pad_trans[padding_zerlens[q][0]:len(whit_pad_trans)-padding_zerlens[q][-1]])
            whitened_transits_quarters.append(whit_pad_trans)

        L = np.sum([len(whitened_transits_quarters[q]) for q in range(len(whitened_transits_quarters))])
        denom_stat_f = (np.sum(
            [np.sum(np.abs(whitened_transits_quarters[q]) ** 2) for q in range(len(whitened_transits_quarters))]) / L) ** 0.5
        amplitude = - snr / denom_stat_f
        if normalize_by_level:
            total_median = np.nanmedian(np.concatenate(fluxes))
            if total_median / np.nanstd(np.concatenate(fluxes)) < 0.1:
                normalize_by_level = False  # if the mean level is too low, ignore it to avoin indtabilities

        for q in range(len(fluxes)):
            if normalize_by_level:
                fluxes[q] += transits_quarters[q] * mov_medians[q] / total_median * amplitude
            else:
                fluxes[q] += transits_quarters[q] * amplitude

        # WARNING:
        # The PSD used for injection is measured -> there will be no loss due to error in measurement!
        # It means that the injected SNR is not exactly the stated.
        # As well, the mean levels are measured, so the same issue.
        if return_zero_padding_ends:
            if return_template:
                if return_wfs:
                    return fluxes, transits_quarters, wffds, return_zero_padding_ends
                else:
                    return fluxes, transits_quarters, return_zero_padding_ends
            else:
                if return_wfs:
                    return fluxes, wffds, return_zero_padding_ends
                else:
                    return fluxes, return_zero_padding_ends
        else:
            if return_template:
                if return_wfs:
                    return fluxes, transits_quarters, wffds
                else:
                    return fluxes, transits_quarters
            else:
                if return_wfs:
                    return fluxes, wffds
                else:
                    return fluxes
    else:
        transits_quarters = []
        for q in range(len(times)):
            if use_physical_parameters:
                if supersampling:
                    trans_quart = tm.flux_par(times_int[q], per=period, rp=rp, a=a, inc=inc,
                                              ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center,
                                              supersample_factor=integration_points, exp_time=dt / integration_points)
                else:
                    trans_quart = tm.flux_par(times_int[q], per=period, rp=rp, a=a, inc=inc,
                                              ecc=ecc, w=w, u1=u1, u2=u2, t0=first_transit_center)
            else:
                if template_num is None:
                    widths = tm.durations(tm.time(dt=dt), tm.template_bank(dt=dt))
                    template_num = np.argmin(np.abs(widths - transit_duration))
                trans_quart = tm.flux_par(prms=tm.template_bank_parameter_list[template_num])
            trans_quart = 1. - trans_quart
            if (not depth_by_rp) and (depth_ppm is not None):
                trans_quart = trans_quart * depth_ppm / 10 ** 6
            transits_quarters.append(trans_quart)

        for q in range(len(fluxes)):
            fluxes[q] -= transits_quarters[q] * np.nanmean(fluxes[q])

        if return_template:
            return fluxes, transits_quarters
        else:
            return fluxes


def rangen(pars, L, distx=0, disty=0):
    """
    generates random vector from given set of values pars
    of given length L
    with given numerical distribution (distx, disty)

    Parameters
    ----------
    pars : TYPE
        DESCRIPTION.
    L : TYPE
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
    return np.random.choice(pars, L, replace=True, p=probdis)


def scramble_data(times, fluxes, numbers, std_time=4.):
    np.random.seed(seed=None)
    # r = np.abs(np.random.normal(0,std_time,len(times)))
    # rr = np.cumsum(r)
    # rr = np.array([np.sum(r[0:i]) for i in range(len(r))])
    # a = 0.
    for i in range(len(times)):
        # c = np.round(np.abs(np.random.normal(a,std_time)), 3)
        # a += c
        times[i] = np.array(times[i]) + numbers[i]
    return times, fluxes

# def scramble_data(times, fluxes, shift_times=True, std_time=4., invert_fluxes=True, mix_quarters=True, invert_times=True):
#     if shift_times:
#         r = np.abs(np.random.normal(0,std_time,len(times)))
#         r = np.cumsum(r)
#         for i in range(len(times)):
#             times[i] = np.array(times[i]) + r[i]
#     if invert_fluxes:
#         for i in range(len(fluxes)):
#             mean = np.nanmean(fluxes[i])
#             fluxes[i] = mean - (np.array(fluxes[i])-mean)
#     if invert_times:
#         for i in range(len(fluxes)):
#             f = bool(np.random.randint(0, 2))
#             if f:
#                 fluxes[i] = fluxes[i][::-1]
#     if mix_quarters:
#         indices = np.arange(len(fluxes))
#         np.random.shuffle(indices)
#         fluxes = [fluxes[i] for i in indices]
#     return times, fluxes


# def flgen(t=tm.time(), NT=1000, maxdur=9):
#     """
#     Generates random fluxes from a feasible parameter range
#     Needed for testing the template bank to have good angles
#     returns fluxes and

#     Parameters
#     ----------
#     t : TYPE, optional
#         DESCRIPTION. The default is np.concatenate(((-1)*np.arange(0, T_ban, dt_lc)[:0:-1],np.arange(0, T_ban, dt_lc))).
#     NT : TYPE, optional
#         DESCRIPTION. The default is 1000.
#     maxdur : TYPE, optional
#         DESCRIPTION. The default is 9.

#     Returns
#     -------
#     None.

#     """
#     per1=50; per2=500 # days #take more to cover extreme cases of other parameters
#     nper = 501
#     pers = np.linspace(per1, per2, nper)
#     rp1 = 0.005; rp2 = 0.05; nrp = 201
#     rps = np.linspace(rp1, rp2, nrp)
#     ecc1 = 0.; ecc2 = 0.9; necc = 201
#     eccs = np.linspace(ecc1, ecc2, necc)
#     w1 = 90.; w2 = w1 + 360.; nw = 401
#     ws = np.linspace(w1, w2, nw)

#     #different variables are taken to be independent
#     #generating uniform distributions of parameters:
#     #(we generate twice more of them because we will then filter them out if they are bad)
#     pergen = rangen(pers, 2*NT)
#     Ms = np.linspace(0.5, 2.5, 201)
#     Mgen = rangen(Ms, 2*NT)
#     #Rs = np.linspace(1,10,10)
#     #Rsgen = rangen(Rs, 2*NT)
#     Rsgen = np.ones(len(Mgen))
#     agen = (pergen/365)**(2/3) * Mgen**(1/3) * 215/Rsgen
#     rpgen = rangen(rps, 2*NT)/Rsgen
#     eccgen = rangen(eccs, 2*NT)
#     wgen = rangen(ws, 2*NT)
#     #generating uniform inclinations from the range that we are able to observe
#     incgen = np.array([])
# #     eccgen = np.array([])
#     for i in range(2*NT):
# #         # the eccentricity is bounded from above by the radius of the star
# #         #Rs-rp < agen * (1 - eccgen)
# #         #1+rp' < agen' * (1 - eccgen)
# #         ecc1 = 0.0; ecc2 = np.min([0.9,1-(1+rpgen[i])/agen[i]]); necc = 101
# #         eccs = np.linspace(ecc1, ecc2, necc)
# #         ecc = rangen(eccs, 1)[0]
# #         eccgen = np.append(eccgen, ecc)

#         #the distance from the planet to the star:
#         #R = agen[i] * (1 - eccgen[i]**2)/(1+eccgen[i] * np.cos(np.pi/180*(wgen[i]-90)))
#         ecc=eccgen[i]
#         R = agen[i] * (1 - ecc**2)/(1+ecc * np.cos(np.pi/180*(wgen[i]-90)))
#         inc2 = 90.
#         inc1 = 90. - np.arcsin((1+rpgen[i])/R)/np.pi*180.
#         ninc = int((inc2-inc1)*200)
#         incs = np.linspace(inc1, inc2, ninc)
#         incgen = np.append(incgen, rangen(incs, 1)[0])

#     argen = np.vstack((pergen,rpgen, agen, incgen, eccgen, wgen)).T
#     flgen = np.array(list(map(
#         lambda v: flux_par(t, *v), argen)))

#     #cleaning out the generated fluxes.
#     #we only need symmetric and fitting the time.
#     t1 = t>=-maxdur/2
#     t2 = t<=maxdur/2
#     ttrun = t[t1 & t2]
#     shapes = np.array(list(map(lambda x: t_shape(ttrun,1-x[t1 & t2]),flgen)))
#     flgen = np.delete(flgen, np.where(shapes.T[0]<=2 * (t[1]-t[0])), axis=0)
#     argen = np.delete(argen, np.where(shapes.T[0]<=2 * (t[1]-t[0])), axis=0)
#     #we filtered out empty or non-symmwtric samples that could have occurred by chance
#     flgen = flgen[0:NT]
#     argen = argen[0:NT]
#     return flgen, argen
