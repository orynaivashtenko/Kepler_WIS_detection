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
N_quarters_simulation = 16
T_min_simulation = 131.
Dur_quarter_days_sim = 91.
Gap_between_quarters_days_sim = 1.


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


def get_simulated_star(n_quarters=N_quarters_simulation, mean_flux=1e4, sigma_high_rel=1e-4, tmin=T_min_simulation,
                       t_between=Gap_between_quarters_days_sim,
                       inject_planet=False, period=100, first_transit_center=141, rp=0.02, a=215., inc=90., ecc=0., w=90., u1=0.47,
                       u2=0.23, snr_required=None, red=True, random_seed=None):
    np.random.seed(random_seed)
    lenquart = int(Dur_quarter_days_sim / tm.dt_lc)
    noises = []
    if red:
        noise, psd = get_red_noise(support=lenquart, n_vectors=0, return_psd=True)
        for i in range(n_quarters):
            noise, psd = get_red_noise(support=lenquart, n_vectors=0, return_psd=True)
            noises.append(noise - np.nanmean(noise))
    else:
        psd = np.ones(lenquart//2 + 1)
        for i in range(n_quarters):
            noise = np.random.normal(0, 1, lenquart)
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

