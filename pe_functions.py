#!/usr/bin/env python3

import os

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6

import numpy as np

import sys

sys.path.append('/home/orynai/Kepler_Earths/WIS_Kepler_pipeline')
# import target_class
# import tce_class
# import pipeline_parameters as pp
import search_pipeline as spip
import template_bank.template_module as tm
import utils as ut
from pipeline_parameters import R_sun_R_e, get_a_from_p
from template_bank.template_bank_generation_parameters import E_crit
import auxiliary_functions as af
from locations import Gaia_kepler_file, Berger_table_file
# import simulations.lightcurve_simulation as sim
from IPython.display import clear_output
import json
from scipy.stats import skewnorm
from scipy.ndimage import binary_dilation
from pipeline_parameters import G, M_sun, R_sun


def simulate_whitened_noise_transit(ecc_sim, w_sim, inc_sim, rp_rs_sim, per,
                                    time, binwid, inds_to_zero, len_high_pass,
                                    u, rel_wf_fd, mask_flux, vars_flux,
                                    a_sim=None, stellar_mass=None, stellar_radius=None, return_injected_model=False):
    assert inc_sim <= 90., "inc_sim should be <= 90 degrees"
    # Simulation
    if a_sim is None:
        if stellar_mass is None or stellar_radius is None:
            raise ValueError("If a_sim is not given, stellar mass and radius must be given to compute it.")
        else:
            a_sim = get_a_from_p(period=per, m_1_3_r_s=None, m_star=stellar_mass, r_star=stellar_radius)

    flux_sim_padded_wh, f_flux_sim_padded_wh, flux_sim = get_flux_model_for_fit(
        time=time, per=per, rp=rp_rs_sim, a=a_sim, inc=inc_sim, ecc=ecc_sim, w=w_sim, u=u, rel_wf_fd=rel_wf_fd,
        inds_high_pass=inds_to_zero, len_high_pass=len_high_pass, exp_time=binwid, return_red=True)

    vars_flux = np.ones(len(vars_flux)) * np.median(vars_flux) + np.abs(np.random.normal(0, np.nanstd(vars_flux) * 0.1, len(vars_flux)))
    mask_flux = np.ones(len(mask_flux), dtype=bool)

    flux_sim_padded_wh = -flux_sim_padded_wh
    flux_sim_padded_wh_0 = np.array(flux_sim_padded_wh)
    flux_sim_padded_wh += np.random.normal(0, 1, size=len(flux_sim_padded_wh)) * (vars_flux) ** 0.5
    f_flux_sim_padded_wh = np.fft.rfft(flux_sim_padded_wh)

    flux = flux_sim_padded_wh
    f_flux = f_flux_sim_padded_wh
    log_dur_sim = np.log(get_transit_duration_approx(per=per, a=a_sim, ecc=ecc_sim, cos_i=np.cos(inc_sim / 180. * np.pi), w=w_sim))
    impact_par_sim = get_impact_parameter_approx(a=a_sim, ecc=ecc_sim, cos_i=np.cos(inc_sim / 180. * np.pi), w=w_sim)
    b_transf_sim = impact_par_transform_fun(impact_par_sim)

    if not return_injected_model:
        return flux, f_flux, vars_flux, mask_flux, a_sim, log_dur_sim, impact_par_sim, b_transf_sim
    else:
        return flux, f_flux, vars_flux, mask_flux, a_sim, log_dur_sim, impact_par_sim, b_transf_sim, flux_sim_padded_wh_0


def get_log_likelihood(flux, vars_flux, mask_flux, model_flux):
    """Compute the log-likelihood of the data given the model.
    Warning: vars should not be zero anywhere!
    :param flux: The folded flux values.
    :type flux: numpy.ndarray
    :param vars_flux: The variances of the folded flux values.
    :type vars_flux: numpy.ndarray
    :param mask_flux: A boolean mask indicating valid (True) and invalid (False) data points.
    :type mask_flux: numpy.ndarray
    :param model_flux: The model flux values to compare against the observed data.
    :type model_flux: numpy.ndarray

    :return: The log-likelihood value up to a constant
    """
    residuals = flux - model_flux
    chi2 = np.sum((residuals[mask_flux] ** 2) / vars_flux[mask_flux])
    log_det = np.sum(np.log(vars_flux[mask_flux]))
    n = np.sum(mask_flux)
    log_likelihood_value = -0.5 * (chi2 + log_det + n * np.log(2 * np.pi))
    return log_likelihood_value


def get_cos_i_max(a, ecc, w=None, overshoot_factor=1., w_rad=None):
    """w is in degrees"""
    if w_rad is None:
        w_rad = w / 180. * np.pi
    cos_i_max = 1 / a * (1 + ecc * np.cos(w_rad - np.pi / 2)) / (1 - ecc ** 2) * overshoot_factor
    return cos_i_max


def check_transit_condition(a, ecc, w, inc, e_crit=None, overshoot_factor=1.):
    """w and inc are in degrees"""
    is_transiting = True
    if e_crit is not None:
        e_max = min(1 - 2 / a, e_crit)
        is_transiting = is_transiting & (ecc <= e_max)
    cos_i_max = get_cos_i_max(a=a, ecc=ecc, w=w, overshoot_factor=overshoot_factor)
    cos_i = np.cos(inc / 180. * np.pi)
    is_transiting = is_transiting & (cos_i < cos_i_max)
    return is_transiting


def get_log_occurrence_prior_period_rp(rp_re, period, dp=1, drp_re=0.1):
    rp_re_min = rp_re - 0.5 * drp_re
    rp_re_max = rp_re + 0.5 * drp_re
    period_min = period - 0.5 * dp
    period_max = period + 0.5 * dp
    occurrence_rate_in_chunk = tm.get_occurrence_rate_per_rp_chunk(period_min=period_min, period_max=period_max, rp_re_min=rp_re_min,
                                                                   rp_re_max=rp_re_max)
    log_occurrence_rate_density = np.log(occurrence_rate_in_chunk) - np.log(dp * drp_re)
    return log_occurrence_rate_density


def return_inf_if_no_transiting(cos_i, sqrt_e_cos_w, sqrt_e_sin_w, rp_re, e_max, a):
    w_rad = np.arctan2(sqrt_e_sin_w, sqrt_e_cos_w)
    ecc = sqrt_e_cos_w ** 2 + sqrt_e_sin_w ** 2
    if ecc >= e_max or ecc < 0:
        return -np.inf
    cos_i_max = get_cos_i_max(a=a, ecc=ecc, w_rad=w_rad, overshoot_factor=1.01)
    if cos_i < 0 or cos_i > cos_i_max:
        return -np.inf
    else:
        return 0.


def get_log_prior_period_rp(rp_re, period, cos_i, sqrt_e_cos_w, sqrt_e_sin_w,  e_max, a, dp=1, drp_re=0.1):
    """
    Units of density. Full likelihood is integral of likelihood density dp dr etc. These dp and dr can be different. When
    we compare posteriors, we need to either keep dp dr etc the same, or multiply by them to compare the probabilities. But
    then we need to make sure that the bins are the same or something. So it is better to keep dp and dr and etc the same and
    compare the densities.

    For cos i, we take a uniform prior, so up to a constant it does not matter.
    Same for ecc, and for w which we replace together by sqrt_e_cos_w and sqrt_e_sin_w.

    inclination is from 0 to 90, with 90 being edge-on, perfect transit. 0 is face-on, no transit.

    """
    # TODO: Should the prior include the probability to transit depending on ecc and w and a? When we include all the parameters, then
    # bins are uniform, they do not have any weights. So perhaps this is fine.
    transiting_factor = return_inf_if_no_transiting(cos_i=cos_i, sqrt_e_cos_w=sqrt_e_cos_w, sqrt_e_sin_w=sqrt_e_sin_w,
                                                    rp_re=rp_re, e_max=e_max, a=a)
    log_occurrence_rate_density = get_log_occurrence_prior_period_rp(rp_re=rp_re, period=period, dp=dp, drp_re=drp_re)
    return log_occurrence_rate_density + transiting_factor


# def log_probability(flux, vars_flux, mask_flux, model_flux,
#                     cos_i, sqrt_e_cos_w, sqrt_e_sin_w, rp_re, e_max, a, period):
#     log_likelihood_value = get_log_likelihood(flux=flux, vars_flux=vars_flux, mask_flux=mask_flux, model_flux=model_flux)
#     log_prior_value = get_log_prior_period_rp(cos_i, sqrt_e_cos_w, sqrt_e_sin_w, rp_re, e_max=e_max, a=a, period=period)
#     if not np.isfinite(log_prior_value):
#         return -np.inf
#     return log_prior_value + log_likelihood_value


Index_sqrt_e_cos_w_grid, Index_sqrt_e_sin_w_grid, Index_cos_i_grid = 0, 1, 2


def get_geometry_grid(a, e_max, dcosi=0.01, dsqrt_e_cos_w=0.1, return_separate_grids=False):
    """rp_re_center is the initial guess for planetary radius in R_earth"""

    # sqrt_e_cos_w_grid = ut.get_uniform_random_sequence(0, np.sqrt(e_max), dx=dsqrt_e_cos_w)
    # sqrt_e_cos_w_grid = np.concatenate((-sqrt_e_cos_w_grid[::-1],
    # ut.get_uniform_random_sequence(0, np.sqrt(e_max), dx=dsqrt_e_cos_w)[1:]))

    # sqrt_e_sin_w_grid = ut.get_uniform_random_sequence(0, np.sqrt(e_max), dx=dsqrt_e_cos_w)
    # sqrt_e_sin_w_grid = np.concatenate((-sqrt_e_sin_w_grid[::-1],
    # ut.get_uniform_random_sequence(0, np.sqrt(e_max), dx=dsqrt_e_cos_w)[1:]))

    sqrt_e_cos_w_grid = np.arange(0, np.sqrt(e_max), dsqrt_e_cos_w)
    sqrt_e_cos_w_grid = np.concatenate((-sqrt_e_cos_w_grid[::-1], sqrt_e_cos_w_grid[1:]))
    sqrt_e_sin_w_grid = np.array(sqrt_e_cos_w_grid, copy=True)

    global Index_sqrt_e_cos_w_grid
    global Index_sqrt_e_sin_w_grid
    global Index_cos_i_grid
    Index_sqrt_e_cos_w_grid, Index_sqrt_e_sin_w_grid, Index_cos_i_grid = 0, 1, 2

    # sqrt_e_cos_w_edges = np.array([-np.sqrt(e_max), np.sqrt(e_max)])
    # sqrt_e_cos_w_grid = np.linspace(sqrt_e_cos_w_edges[0], sqrt_e_cos_w_edges[1], n_ecc)
    # sqrt_e_sin_w_edges = np.array([-np.sqrt(e_max), np.sqrt(e_max)])
    # sqrt_e_sin_w_grid = np.linspace(sqrt_e_sin_w_edges[0], sqrt_e_sin_w_edges[1], n_w)
    geometry_grid = []
    for i_cos in range(len(sqrt_e_cos_w_grid)):
        for i_sin in range(len(sqrt_e_sin_w_grid)):
            sqrt_e_cos_w_i = sqrt_e_cos_w_grid[i_cos]
            sqrt_e_sin_w_i = sqrt_e_sin_w_grid[i_sin]
            sqrt_e_cos_w_i += np.random.uniform(low=0, high=dsqrt_e_cos_w, size=None)
            sqrt_e_sin_w_i += np.random.uniform(low=0, high=dsqrt_e_cos_w, size=None)
            ecc = sqrt_e_cos_w_i ** 2 + sqrt_e_sin_w_i ** 2
            if ecc >= e_max or ecc < 0:
                pass
            else:
                w_rad = np.arctan2(sqrt_e_sin_w_i, sqrt_e_cos_w_i)
                cos_i_max = get_cos_i_max(a=a, ecc=ecc, w_rad=w_rad, overshoot_factor=1.01)
                # cos_i_grid = np.arange(0, min(1, cos_i_max), dcosi)
                cos_i_grid = ut.get_uniform_random_sequence(0, min(1, cos_i_max), dx=dcosi)
                for cos_i in cos_i_grid:
                    geometry_grid.append((sqrt_e_cos_w_i, sqrt_e_sin_w_i, cos_i))
    if not return_separate_grids:
        return geometry_grid

    else:
        max_cos_i = np.nanmax([geometry_grid[i][2] for i in range(len(geometry_grid))])
        # Here we do not do random because these are regular grids
        cos_i_grid = np.arange(0, max_cos_i, dcosi)
        return geometry_grid, sqrt_e_cos_w_grid, sqrt_e_sin_w_grid, cos_i_grid


def get_regular_grids_from_geometry_grid(geometry_grid, e_max=None,
                                         index_sqrt_e_cos_w_grid=Index_sqrt_e_cos_w_grid, index_sqrt_e_sin_w_grid=Index_sqrt_e_sin_w_grid,
                                         index_cos_i_grid=Index_cos_i_grid,
                                         dcosi=0.01, dsqrt_e_cos_w=0.1):
    if e_max is None:
        e_max = E_crit
    else:
        e_max = np.nanmax([geometry_grid[i][index_sqrt_e_cos_w_grid] ** 2 + geometry_grid[i][index_sqrt_e_sin_w_grid] ** 2 for i in
                           range(len(geometry_grid))])

    cos_i_max = np.nanmax([geometry_grid[i][index_cos_i_grid] for i in range(len(geometry_grid))])
    return get_regular_grids(e_max=e_max, cos_i_max=cos_i_max, dcosi=dcosi, dsqrt_e_cos_w=dsqrt_e_cos_w)


def get_regular_grids(e_max, cos_i_max, dcosi=0.01, dsqrt_e_cos_w=0.1):
    cos_i_grid = np.arange(0, cos_i_max, dcosi)
    sqrt_e_cos_w_grid = np.arange(0, np.sqrt(e_max), dsqrt_e_cos_w)
    sqrt_e_cos_w_grid = np.concatenate((-sqrt_e_cos_w_grid[::-1], sqrt_e_cos_w_grid[1:]))
    sqrt_e_sin_w_grid = np.array(sqrt_e_cos_w_grid, copy=True)
    return sqrt_e_cos_w_grid, sqrt_e_sin_w_grid, cos_i_grid


def get_rp_re_grid(rp_re_min, rp_re_max, drp_re=0.1):
    rp_re_grid = np.arange(rp_re_min, rp_re_max, drp_re)
    return rp_re_grid


def get_flux_model_for_fit(time, per, len_high_pass, rp, a, inc, ecc, w, u, rel_wf_fd, inds_high_pass, exp_time, return_red=False, t0=0):
    """
    len_high_pass is needed before high-passing befire the final cut
    Important! The model does not undergo any normalizations etc!"""
    # We do not do any rolling, we will maximize over shifts
    # We do not do any normalization, we want to preserve the radius units
    # WARNING: batman package for some reason is very slow for w=-90 or w=270, so we replace:
    if w == -90:
        w = -89.999
    if w == 270:
        w = 269.999
    flux_model = 1. - tm.flux_par(t=time, params=None, per=per, rp=rp, a=a, inc=inc, ecc=ecc, w=w, u1=u[0], u2=u[1],
                                  t0=t0, supersample_factor=20, exp_time=exp_time)

    # # High-passing:
    # f_flux_model = np.fft.rfft(flux_model)
    # f_flux_model[inds_to_zero_freqs] = 0
    # flux_model = np.fft.irfft(f_flux_model)
    # # -> Attention: a sinusoid is created because of the high-pass,
    # # This sinusoid formally makes the support infinite, but it is small
    # Padding to match the whitening filter size:
    len_final = len(time)
    support = 2 * (len(rel_wf_fd) - 1)  # Support is just the length of the whitening filter. It is already padded
    if len_final > support:
        raise ValueError("Final length is longer than the whitening filter support!")
    zerlen_1 = (support - len(flux_model)) // 2
    zerlen_2 = (support - len(flux_model) + 1) // 2
    flux_model_padded = np.concatenate((np.zeros(zerlen_1), flux_model, np.zeros(zerlen_2)))
    flux_model_padded_wh = np.fft.irfft(np.fft.rfft(flux_model_padded) * rel_wf_fd)
    if len(flux_model_padded_wh) - len_final < 0:
        raise ValueError("Whitening filter is too short!")

    if len(flux_model_padded_wh) - len_high_pass > 0:
        cut_len_1 = (len(flux_model_padded_wh) - len_high_pass) // 2
        cut_ind_2 = cut_len_1 + len_high_pass
        flux_model_padded_wh = flux_model_padded_wh[cut_len_1:cut_ind_2]
    elif len(flux_model_padded_wh) - len_high_pass == 0:
        pass
    else:
        raise ValueError("len(flux_model_padded_wh) < len_high_pass!")
    # High-passing:
    f_flux_model_padded_wh = np.fft.rfft(flux_model_padded_wh)
    f_flux_model_padded_wh[inds_high_pass] = 0
    flux_model_padded_wh = np.fft.irfft(f_flux_model_padded_wh)
    # -> Attention: a sinusoid is created because of the high-pass,
    # This sinusoid formally makes the support infinite, but it is small

    # Final cut:
    if len(flux_model_padded_wh) - len_final > 0:
        cut_len_1 = (len(flux_model_padded_wh) - len_final) // 2
        cut_ind_2 = cut_len_1 + len_final
        flux_model_padded_wh = flux_model_padded_wh[cut_len_1:cut_ind_2]
    elif len(flux_model_padded_wh) - len_final == 0:
        pass
    else:
        raise ValueError("len(flux_model_padded_wh) < len_final!")
    f_flux_model_padded_wh = np.fft.rfft(flux_model_padded_wh)
    if not return_red:
        return flux_model_padded_wh, f_flux_model_padded_wh
    else:
        if len(flux_model_padded_wh) - len_final > 0:
            cut_len_1 = (len(flux_model_padded) - len_final) // 2
            cut_ind_2 = cut_len_1 + len_final
            flux_model = flux_model_padded[cut_len_1:cut_ind_2]
        return flux_model_padded_wh, f_flux_model_padded_wh, flux_model


def exp_sum_no_overflow(arr, axis=-1):
    arr = np.array(arr, copy=True)

    arr[np.isinf(arr)] = np.nan
    nan_mask = np.isnan(arr)
    arr_ma = np.ma.array(arr, mask=nan_mask, fill_value=np.nan)
    max_vals = np.nanmax(arr_ma, axis=axis)
    idx = [slice(None)] * max_vals.ndim
    idx.insert(axis, None)
    summed = arr_ma - max_vals[tuple(idx)]  # to avoid overflow
    summed_mask = summed.mask
    summed = np.nansum(np.exp(summed), axis=axis)
    if hasattr(summed, 'filled'):
        summed = summed.filled(fill_value=np.nan)
    else:
        if summed_mask.all():
            summed = np.nan
    summed = np.log(summed)
    summed += max_vals
    return summed


def fit_amplitude(f_flux_wh, f_flux_model_padded_wh, len_flux, rp_rs_model, rp_rs_max=1.,
                  half_width_bins_limit_fit_indices=None):
    """Fit the amplitude of the flux model to the observed flux while finding the best timing
    `f_flux_wf` and `f_flux_model_padded_wh` are the FFTs of the flux and model flux. Both are
    whitened They should be of the same length. They should be in the same units, i.e no further
    normalization will be applied (e.g. by the mean flux baseline) to get the radius."""
    denom = (np.fft.irfft(f_flux_model_padded_wh * np.conjugate(f_flux_model_padded_wh), n=len_flux))[0]
    if denom == 0:
        ampl = np.nan
        rp_rs_est = np.nan
        ind_fit = 0
    else:
        num = (np.fft.irfft(f_flux_wh * np.conjugate(f_flux_model_padded_wh), n=len_flux))
        if half_width_bins_limit_fit_indices is not None:
            num[half_width_bins_limit_fit_indices: len(num) - half_width_bins_limit_fit_indices] = np.nan
        ind_fit = np.nanargmin(num)
        ampl = num[ind_fit] / denom
        if ampl < 0:
            rp_rs_est = np.abs(ampl) ** 0.5 * rp_rs_model
        else:
            rp_rs_est = 0.
        if rp_rs_max is not None:
            if rp_rs_est > rp_rs_max:
                # Due to the undersampling of very short transits, the depth can be eaten, and
                # then the fit will show unphysically large radius.
                rp_rs_est = rp_rs_max
                ampl = (rp_rs_est / rp_rs_model) ** 2
    return rp_rs_est, ampl, ind_fit


def cos_angle_fun(fm1, fm2):
    return fm1 @ fm2 / (fm1 @ fm1) ** 0.5 / (fm2 @ fm2) ** 0.5


def fit_flux_par(time, flux_wh, vars_flux, mask_flux, binwid, sqrt_e_cos_w, sqrt_e_sin_w, cos_i, per, a, u,
                 rel_wf_fd, inds_to_zero, len_high_pass, stellar_radius_for_prior, a_for_prior=None,
                 min_cos_shift=0.97, e_max=E_crit, rp_rs_mock=0.02, rp_rs_max=1., return_model=False,
                 use_geometry_for_prior=False, use_occurrence_rate_prior=True,
                 half_width_bins_limit_fit_indices=None):
    f_flux_wh = np.fft.rfft(flux_wh)
    len_flux = len(flux_wh)
    vars_flux = np.array(vars_flux)
    vars_flux[vars_flux <= 0] = np.inf  # we will divide by it. And zero var means that there was no data there
    mask_flux = mask_flux & ((~np.isinf(vars_flux)) & (vars_flux != 0) & (~np.isnan(vars_flux)))

    # Getting physical parameters:
    ecc = sqrt_e_cos_w ** 2 + sqrt_e_sin_w ** 2
    w = np.arctan2(sqrt_e_sin_w, sqrt_e_cos_w) / np.pi * 180
    inc = np.arccos(cos_i) / np.pi * 180
    flux_model_padded_wh, f_flux_model_padded_wh = get_flux_model_for_fit(
        time=time, per=per, rp=rp_rs_mock, a=a, inc=inc, ecc=ecc, w=w, u=u, rel_wf_fd=rel_wf_fd, t0=0.,
        inds_high_pass=inds_to_zero, len_high_pass=len_high_pass, exp_time=binwid, return_red=False)

    if np.linalg.norm(flux_model_padded_wh) == 0:  # Checking whether transits at all
        log_posterior, rp_rs_est, ampl, ind_fit = -np.inf, np.nan, np.nan, 0
        log_likelihood = np.nan
    else:
        # Getting subbin shifted models if needed:
        cos_angle_shift = flux_model_padded_wh @ np.roll(flux_model_padded_wh, -1) / (flux_model_padded_wh @ flux_model_padded_wh)

        if cos_angle_shift >= min_cos_shift:  # no need to get subbin shifts
            rp_rs_est, ampl, ind_fit = fit_amplitude(f_flux_wh=f_flux_wh, f_flux_model_padded_wh=f_flux_model_padded_wh,
                                                     len_flux=len_flux, rp_rs_model=rp_rs_mock, rp_rs_max=rp_rs_max,
                                                     half_width_bins_limit_fit_indices=half_width_bins_limit_fit_indices)
            # Fitting the amplitude, ignoring the radius - only the shape. The radius will be obtained from the amplitude.
            if np.isnan(ampl):
                log_posterior = -np.inf
                log_likelihood = np.nan
            else:
                # resid = flux - ampl * np.roll(flux_model_padded_wh, ind_fit)

                log_lik = get_log_likelihood(flux=flux_wh, vars_flux=vars_flux, mask_flux=mask_flux,
                                             model_flux=ampl * np.roll(flux_model_padded_wh, ind_fit))
                if use_occurrence_rate_prior:
                    # log_pri = get_log_occurrence_prior_period_rp(rp_re=rp_rs_est * stellar_radius_for_prior * R_sun_R_e,
                    #                                              period=per)
                    if use_geometry_for_prior:

                        log_pri = get_log_prior_period_rp(cos_i=cos_i, sqrt_e_cos_w=sqrt_e_cos_w, sqrt_e_sin_w=sqrt_e_sin_w,
                                                          rp_re=rp_rs_est * stellar_radius_for_prior * R_sun_R_e, e_max=e_max, a=a, period=per)
                    else:
                        assert a_for_prior is not None, "a_for_prior must be given if not using geometry for prior"
                        log_pri = get_log_prior_period_rp(cos_i=0, sqrt_e_cos_w=0, sqrt_e_sin_w=0,
                                                          rp_re=rp_rs_est * stellar_radius_for_prior * R_sun_R_e,
                                                          e_max=e_max, a=a_for_prior, period=per)
                    log_posterior = log_lik + log_pri
                else:
                    log_posterior = log_lik
                log_likelihood = log_lik
        else:  # need to get subbin shifts
            # Finding the proper subbin shifts number:
            n_shifts = 1  # Initially there were no subbin shifts
            shift_step = binwid / n_shifts
            while cos_angle_shift < min_cos_shift:
                n_shifts += 1
                shift_step = binwid / n_shifts
                models_wh = [np.array(flux_model_padded_wh)]  # take the non-shifted one here
                f_models_wh = [np.array(f_flux_model_padded_wh)]
                for i_sh in range(1, n_shifts):
                    fmw, ffmw = get_flux_model_for_fit(time, per=per, rp=rp_rs_mock, a=a, inc=inc, ecc=ecc, w=w, u=u, rel_wf_fd=rel_wf_fd,
                                                       t0=i_sh * shift_step, inds_high_pass=inds_to_zero, len_high_pass=len_high_pass,
                                                       exp_time=binwid, return_red=False)
                    models_wh.append(fmw)
                    f_models_wh.append(ffmw)
                cos_angle_shift = cos_angle_fun(models_wh[0], models_wh[1])
                # Fitting all the subbin shifts and finding the best:
            log_posteriors_shifts = []
            log_likelihoods_shifts = []
            rp_rs_est_shifts = []
            ampl_shifts = []
            ind_fits_shifts = []
            for i_shift in range(len(models_wh)):
                f_flux_model_padded_wh_shift = f_models_wh[i_shift]
                flux_model_padded_wh_shift = models_wh[i_shift]
                rp_rs_est, ampl, ind_fit = fit_amplitude(f_flux_wh=f_flux_wh, f_flux_model_padded_wh=f_flux_model_padded_wh_shift,
                                                         len_flux=len_flux, rp_rs_model=rp_rs_mock, rp_rs_max=rp_rs_max,
                                                         half_width_bins_limit_fit_indices=half_width_bins_limit_fit_indices)
                log_lik = get_log_likelihood(flux=flux_wh, vars_flux=vars_flux, mask_flux=mask_flux,
                                             model_flux=ampl * np.roll(flux_model_padded_wh_shift, ind_fit))
                if use_occurrence_rate_prior:
                    if use_geometry_for_prior:
                        # TODO! the semi-major axis etc are not physical! They should not be used for any prior!
                        log_pri = get_log_prior_period_rp(cos_i=cos_i, sqrt_e_cos_w=sqrt_e_cos_w, sqrt_e_sin_w=sqrt_e_sin_w,
                                                          rp_re=rp_rs_est * stellar_radius_for_prior * R_sun_R_e, e_max=e_max, a=a, period=per)
                    else:
                        assert a_for_prior is not None, "a_for_prior must be given if not using geometry for prior"
                        log_pri = get_log_prior_period_rp(cos_i=0, sqrt_e_cos_w=0, sqrt_e_sin_w=0,
                                                          rp_re=rp_rs_est * stellar_radius_for_prior * R_sun_R_e,
                                                          e_max=e_max, a=a_for_prior, period=per)
                    log_posterior = log_lik + log_pri
                else:
                    log_posterior = log_lik

                log_posteriors_shifts.append(log_posterior)
                log_likelihoods_shifts.append(log_lik)
                rp_rs_est_shifts.append(rp_rs_est)
                ampl_shifts.append(ampl)
                ind_fits_shifts.append(ind_fit)
            best_shift_index = np.nanargmax(log_posteriors_shifts)
            log_posterior = log_posteriors_shifts[best_shift_index]
            log_likelihood = log_likelihoods_shifts[best_shift_index]
            rp_rs_est = rp_rs_est_shifts[best_shift_index]
            ampl = ampl_shifts[best_shift_index]
            ind_fit = ind_fits_shifts[best_shift_index]
            flux_model_padded_wh = models_wh[best_shift_index]
    if not return_model:
        return log_posterior, rp_rs_est, ampl, ind_fit, log_likelihood
    else:
        return log_posterior, rp_rs_est, ampl, ind_fit, flux_model_padded_wh, log_likelihood


def prepare_grid(min_edge, max_edge, n_vals, return_edges=False):
    edges = np.linspace(min_edge, max_edge, n_vals + 1)
    centers = edges[:-1] + 0.5 * (edges[1] - edges[0])
    if not return_edges:
        return centers
    else:
        return centers, edges


def get_physical_parameters_from_uniform_parameters_grid(geometry_grid,
                                                         index_sqrt_e_cos_w_grid=Index_sqrt_e_cos_w_grid,
                                                         index_sqrt_e_sin_w_grid=Index_sqrt_e_sin_w_grid,
                                                         index_cos_i_grid=Index_cos_i_grid):
    eccs_list_fit = [geometry_grid[i][index_sqrt_e_cos_w_grid] ** 2 + geometry_grid[i][index_sqrt_e_sin_w_grid] ** 2 for i in
                     range(len(geometry_grid))]
    ws_list_fit = [np.arctan2(geometry_grid[i][index_sqrt_e_sin_w_grid], geometry_grid[i][index_sqrt_e_cos_w_grid]) / np.pi * 180 for i in
                   range(len(geometry_grid))]
    incs_list_fit = [np.arccos(geometry_grid[i][index_cos_i_grid]) / np.pi * 180 for i in range(len(geometry_grid))]
    return eccs_list_fit, ws_list_fit, incs_list_fit


def get_coarse_grids_from_physical_parameters(eccs_list_fit, ws_list_fit, incs_list_fit,
                                              ecc_min=0, ecc_max=1, w_min=-180, w_max=180, inc_min=None, inc_max=90., n_eccs=10, n_ws=10,
                                              n_incs=10):
    if ecc_min is None:
        ecc_min = np.nanmin(eccs_list_fit)
    if ecc_max is None:
        ecc_max = np.nanmax(eccs_list_fit)
    if w_min is None:
        w_min = np.nanmin(ws_list_fit)
    if w_max is None:
        w_max = np.nanmax(ws_list_fit)
    if inc_min is None:
        inc_min = np.nanmin(incs_list_fit)
    if inc_max is None:
        inc_max = np.nanmax(incs_list_fit)
    eccs_grid, eccs_grid_edges = prepare_grid(min_edge=ecc_min, max_edge=ecc_max, n_vals=n_eccs, return_edges=True)
    ws_grid, ws_grid_edges = prepare_grid(min_edge=w_min, max_edge=w_max, n_vals=n_ws, return_edges=True)
    incs_grid, incs_grid_edges = prepare_grid(min_edge=inc_min, max_edge=inc_max, n_vals=n_incs, return_edges=True)
    return eccs_grid, ws_grid, incs_grid, eccs_grid_edges, ws_grid_edges, incs_grid_edges


# ----------------------------------------------------------------------------------------------------------------------


def prepare_target_for_pe(targ, focus_period, focus_ftt=None, upsampling_factor=1):
    # ------------------------------------------------------------------------------------------------------------------
    # # Re-doing preliminary conditioning because I do not want to notch, (
    # note: perhaps it would be better to notch both the data and the template and take care of the normalization)
    # I want the least invasive processing to avoid biases
    # get depth in units of flux
    # ------------------------------------------------------------------------------------------------------------------
    assert upsampling_factor >= 1, "upsampling_factor should be >= 1"
    assert int(upsampling_factor) == upsampling_factor, "upsampling_factor should be an integer"
    upsampling_factor = int(upsampling_factor)

    if focus_ftt is None:
        focus_ftt = targ.best_ftt
    targ.initialize_kepler_light_curve()
    klc = targ.klc

    nsig_outliers_0 = spip.Nsig_outliers_0
    nsig_outliers_0_pos = spip.Nsig_outliers_0_pos
    nsig_outliers_0_neg = spip.Nsig_outliers_0_neg
    nsig_outliers_psd = spip.Nsig_outliers_psd
    nperseg_psd = spip.Nperseg_psd
    npsdsmin = spip.Npsdsmin
    reject_outliers_psd = spip.Reject_outliers_psd
    # nsig_notch=spip.Nsig_notch
    # nsig_outliers_w=spip.Nsig_outliers_w
    # nsig_outliers_w_pos=spip.Nsig_outliers_w_pos
    # nsig_outliers_w_neg=spip.Nsig_outliers_w_neg
    max_small_hole = spip.Max_small_hole
    detr_length_outliers = spip.Detr_length_outliers
    tukey_steps_len = spip.Tukey_steps_len_psd
    save_low_trends = False

    klc.nperseg_psd = nperseg_psd
    klc.subtract_means(save_means=True, masks=None)
    klc.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                           nsig_negative=nsig_outliers_0_neg,
                           nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                           in_whitened_fluxes=False, modify_clean_masks=True)
    klc.subtract_means(save_means=True, masks=klc.masks)
    klc.normalize_by_means()
    klc.make_big_hole_masks(max_small_hole=max_small_hole, exclude_individual_points=True)
    klc.interpolate_fluxes(interpolate_clean_fluxes=True)
    klc.subtract_low_trends(nperseg=klc.nperseg_psd, save_low_trends=save_low_trends)
    klc.fill_additional_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                 nsig_negative=nsig_outliers_0_neg,
                                 nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                 in_whitened_fluxes=False, modify_clean_masks=True,
                                 interpolate_clean_fluxes=True)  # again after piecewise detrending
    klc.prepare_for_psd(nperseg=klc.nperseg_psd, npsdsmin=npsdsmin, max_small_hole=max_small_hole)
    klc.pad_by_zeros()  # Needed to avoid crosstalk in whitening filter. And we want the end zeros to be actual zeros
    klc.measure_psds(nperseg=klc.nperseg_psd, tukey_steps_len=tukey_steps_len, reject_outliers=reject_outliers_psd)
    klc.whiten_fluxes(whiten_clean_fluxes=True)
    # ------------------------------------------------------------------------------------------------------------------
    # GETTING THE AVERAGE WHITENING FILTER
    # ------------------------------------------------------------------------------------------------------------------
    # Averaging the whitening filter while taking into account in which quarters the transits are to
    # average the proper filters. Warning: we do not account here for the masks. For example,
    # half a transit may be masked, but the averaging is not aware of that and will use the whitening filter
    # in the masked region as well. As a result, a bias between the model and the data may be introduced.
    # In order to avoid the bias, full model ephemeris should be created and whitened with quarter filters, then
    # the masks should be applied, and then the averaging should be made. It is more computationally expensive by
    # a factor of number of different quarters among the transits.
    # ------------------------------------------------------------------------------------------------------------------
    # Getting transit quarters
    expected_transit_times = ut.get_transit_times(first_transit_time=focus_ftt, period=focus_period, max_time=klc.tmax)

    min_quarter_times = [np.nanmin(klc.times[q]) for q in range(len(klc.times))]
    max_quarter_times = [np.nanmax(klc.times[q]) for q in range(len(klc.times))]

    transit_quarters = []
    for tim in expected_transit_times:
        for qq in range(len(klc.times)):
            if min_quarter_times[qq] <= tim <= max_quarter_times[qq]:
                transit_quarters.append(qq)
            elif qq > 0 and max_quarter_times[qq - 1] < tim < min_quarter_times[qq]:
                transit_quarters.append(np.nan)
    # -----------------------------------------------
    rel_wf_fds = np.array(klc.wf_fds) * klc.mean_total
    # rel_wf_fd = np.mean(np.array(rel_wf_fds) ** 2, axis=0) ** 0.5
    wf_fds_for_averaging = [rel_wf_fds[q] ** 2 for q in transit_quarters if q is not np.nan]
    rel_wf_fd = np.mean(wf_fds_for_averaging, axis=0) ** 0.5
    if upsampling_factor > 1:
        rel_wf_fd = upsample_wf_fd(rel_wf_fd=rel_wf_fd, oversampling_factor=upsampling_factor)
    mean_flux = klc.mean_total
    # -----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # GETTING THE FOLDED WHITENED FLUXES AND THEIR STD
    # ------------------------------------------------------------------------------------------------------------------
    dt_steps_high_pass = klc.nperseg_psd
    dt_days = klc.dt_days
    binwid = klc.dt_days / upsampling_factor
    bins_time, bins_flux, bins_counts, _, _, bins_std = af.fold_fluxes(
        times=klc.times, fluxes=klc.whitened_fluxes_td, period=focus_period, first_time=focus_ftt,
        masks=klc.big_hole_masks, binwid=binwid, dt=klc.dt_days, means=None,
        template=None, return_std=True, normalize_by_counts=True)
    # -> this will be normalized by counts to have the same mean in every bin

    # # If we had normalize_by_counts=False and wanted a variance normalization:
    # flux_folded = np.zeros(len(bins_flux))
    # flux_folded[bins_counts > 0] = bins_flux[bins_counts > 0] / bins_counts[bins_counts > 0] ** 0.5
    # # -> this would have the same variance for all the bins

    # Finding the original std to then correct for blueing:
    dur_avoid_bins = klc.nperseg_psd * 2 * upsampling_factor  # duration around the transit center to avoid when calculating std
    dur_take_bins = (len(bins_flux) - dur_avoid_bins) // 2
    sigma_binned_flux_orig = np.nanstd(np.concatenate((bins_flux[:dur_take_bins], bins_flux[-dur_take_bins:])))

    # We assume that the interesting part of the folded binned flux cannot be larger than the length of the whitening filter,
    # which should be a reasonable assumption because the filter covers the entire quarter
    len_rel_wf_td = (len(rel_wf_fd) - 1) * 2
    if len(bins_flux) > len_rel_wf_td:
        len_cut_left = (len(bins_flux) - len_rel_wf_td) // 2
        index_cut_right = len_rel_wf_td + len_cut_left
        bins_flux = bins_flux[len_cut_left:index_cut_right]
        bins_time = bins_time[len_cut_left:index_cut_right]
        bins_std = bins_std[len_cut_left:index_cut_right]
        bins_counts = bins_counts[len_cut_left:index_cut_right]

    # High-pass to remove the nom-informative low freqs
    flux_high_pass = af.filter_data_high_pass(data=bins_flux, dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                              in_fourier=False, return_cutoff_index=False, renormalize_to_tail=False)
    flux_high_pass = flux_high_pass[:2 * (len(flux_high_pass) // 2)]
    time_high_pass = bins_time[:len(flux_high_pass)]

    # Define the freqs and the high-pass for the future models
    freqs = np.fft.rfftfreq(len(flux_high_pass), d=binwid)
    dt_pass = klc.nperseg_psd * klc.dt_days
    inds_to_zero = freqs <= 1 / dt_pass  # Saving this and using to high-pass the templates

    # Renormalizing the std to account for high-passing:
    sigma_binned_flux_blue = np.nanstd(np.concatenate((flux_high_pass[:dur_take_bins], flux_high_pass[-dur_take_bins:])))
    bins_std = bins_std * sigma_binned_flux_blue / sigma_binned_flux_orig

    # # # CUT IT
    # I think this shouls be enough, including to neglect the ends crosstalk;
    # and it is shorter than any of the checked periods
    duration_folded_flux = klc.nperseg_psd * 6 * klc.dt_days
    assert duration_folded_flux < focus_period, "The cut folded flux duration is longer than the period"
    duration_residuals_bins = klc.nperseg_psd * 2 * upsampling_factor  # For residuals, take a shorter part

    bin_min = np.where(bins_time >= -duration_folded_flux / 2)[0][0]
    bin_max = np.where(bins_time <= duration_folded_flux / 2)[0][-1] + 1
    if len(bins_time[bin_min:bin_max]) % 2 == 1:
        bin_max += 1
    time = bins_time[bin_min:bin_max]
    flux = flux_high_pass[bin_min:bin_max]
    vars_flux = bins_std[bin_min:bin_max] ** 2
    mask_flux = (bins_counts[bin_min:bin_max] > 0)
    f_flux = np.fft.rfft(flux)

    res = {
        'time': time,
        'time_high_pass': time_high_pass,
        'flux': flux,
        'vars_flux': vars_flux,
        'mask_flux': mask_flux,
        'f_flux': f_flux,
        'binwid': binwid,
        'inds_to_zero': inds_to_zero,
        'len_high_pass': len(flux_high_pass),
        'duration_residuals_bins': duration_residuals_bins,
        'stellar_radius': targ.stellar_radius,
        'stellar_mass': targ.stellar_mass,
        'logg': targ.logg,
        'teff': targ.teff,
        'metallicity': targ.metallicity,
        'mean_flux': mean_flux,
        'limb_darkening_coefs': targ.limb_darkening_coefs,
        'rel_wf_fd': rel_wf_fd,
        'transit_quarters': transit_quarters
    }
    return res


def smooth_connect(v1, v2, overlap_window=None):
    if overlap_window is None:
        overlap_window = min(len(v1), len(v2)) // 4
    if overlap_window < 1:
        return np.concatenate((v1, v2))
    w = np.linspace(0, np.pi / 2, overlap_window)
    w1 = np.cos(w) ** 2
    w2 = np.sin(w) ** 2
    v_connect = v1[-overlap_window:] * w1 + v2[:overlap_window] * w2
    return np.concatenate((v1[:-overlap_window], v_connect, v2[overlap_window:]))


def upsample_wf_fd(rel_wf_fd, oversampling_factor=2, part_of_length_for_median=0.5, part_of_length_overlap=0.25):
    assert oversampling_factor >= 1, "oversampling_factor should be >= 1"
    assert oversampling_factor == int(oversampling_factor), "oversampling_factor should be an integer"
    oversampling_factor = int(oversampling_factor)

    rel_wf_fd = np.real(rel_wf_fd)
    extension_amplitude = np.median(np.real(rel_wf_fd[-int(np.round(len(rel_wf_fd) * part_of_length_for_median)):]))
    extension_length = (len(rel_wf_fd) - 1) * int(np.round(oversampling_factor - 1))
    overlap_window = min(extension_length, int(len(rel_wf_fd) * part_of_length_overlap))
    extension = np.ones(extension_length + overlap_window) * extension_amplitude
    rel_wf_fd_extended = smooth_connect(v1=rel_wf_fd, v2=extension, overlap_window=overlap_window)
    # #  If you use the convention that the whitening filter should look the same in the time domain after
    # # upsampling, uncomment the next code line. But if you use the whitening convention of
    # # wh_flux = np.fft.irfft(np.fft.rfft(flux) * wf_fd)
    # # and want to preserve the flux in time domain in the same units, then do not uncomment.
    # rel_wf_fd_extended = rel_wf_fd_extended * oversampling_factor
    return rel_wf_fd_extended


def calculate_posteriors_for_geometry_grid(geometry_grid, time, flux, vars_flux, mask_flux, binwid, rel_wf_fd, inds_to_zero, time_high_pass,
                                           rp_rs_mock, min_cos_shift,
                                           per, a, u, stellar_radius_for_prior, external_priors=None):
    """
    This is for exact grid fitting, meaning that the sampling is done according to the physical parameters, such as eccentricity and
    argument of periastron (through sqrt(e)sin(w) and sqrt(e)cos(w)). This is slow and degenerate,
    therefore this is only used for calibration and debug. The actual
    method that is used makes an anti-degenerate grid of transit durations and impact parameters.
    """
    if external_priors is not None:
        if not hasattr(external_priors, '__len__'):
            external_priors = np.ones(len(geometry_grid)) * external_priors
        log_external_priors = np.log(external_priors)
    else:
        log_external_priors = None

    log_posteriors = []
    ampls = []
    rp_rss = []
    ind_fits = []
    models = []

    for i_grid in range(len(geometry_grid)):
        log_posterior, rp_rs_est, ampl, ind_fit, flux_model_padded_wh, _ = fit_flux_par(time=time, flux_wh=flux, vars_flux=vars_flux,
                                                                                        mask_flux=mask_flux, binwid=binwid,
                                                                                        sqrt_e_cos_w=geometry_grid[i_grid][0],
                                                                                        sqrt_e_sin_w=geometry_grid[i_grid][1],
                                                                                        cos_i=geometry_grid[i_grid][2],
                                                                                        per=per, a=a, u=u, rel_wf_fd=rel_wf_fd,
                                                                                        inds_to_zero=inds_to_zero,
                                                                                        len_high_pass=len(time_high_pass),
                                                                                        stellar_radius_for_prior=stellar_radius_for_prior,
                                                                                        min_cos_shift=min_cos_shift, e_max=E_crit,
                                                                                        rp_rs_mock=rp_rs_mock, rp_rs_max=1.,
                                                                                        return_model=True,
                                                                                        use_geometry_for_prior=True)
        if log_external_priors is not None:
            log_posterior += log_external_priors[i_grid]

        log_posteriors.append(log_posterior)
        ampls.append(ampl)
        rp_rss.append(rp_rs_est)
        ind_fits.append(ind_fit)
        models.append(flux_model_padded_wh)
        if i_grid % 10 == 0:
            clear_output(wait=True)
            print(f"Grid {i_grid + 1}/{len(geometry_grid)} done", end='\r')
    # ------------------------------------------------------------------------------------------------------------------

    # np.save('grid_fit_results_ecc_09_w_150_inc_90_rp_rs_0.1.npy', {
    #     'geometry_grid': geometry_grid,
    #     'log_posteriors': log_posteriors,
    #     'ampls': ampls,
    #     'rp_rss': rp_rss,
    #     'ind_fits': ind_fits,
    # }, allow_pickle=True)

    # res_dict = np.load('grid_fit_results_ecc_09_w_150_inc_90_rp_rs_0.1.npy', allow_pickle=True).item()

    # ------------------------------------------------------------------------------------------------------------------
    # The global maximum posterior model and its amplitude
    # ------------------------------------------------------------------------------------------------------------------
    approx_durations = np.array([get_transit_duration_approx(  # Warning: this was not debugged...
        per=per, a=a,
        ecc=(geometry_grid[j][Index_sqrt_e_cos_w_grid]) ** 2 + geometry_grid[j][Index_sqrt_e_sin_w_grid] ** 2,
        cos_i=geometry_grid[j][Index_cos_i_grid],
        w=np.degrees(np.arctan2(geometry_grid[j][Index_sqrt_e_sin_w_grid], geometry_grid[j][Index_sqrt_e_cos_w_grid])))
        for j in range(len(geometry_grid))])
    goodness_index_mask = filter_good_fit_indices(ind_fits=ind_fits, len_flux=len(flux),
                                                  log_dur=np.log(approx_durations), binwid=binwid)
    goodness_mask = goodness_index_mask & (~np.isnan(log_posteriors)) & (~np.isinf(log_posteriors))

    ind_max_posterior_with_mask = np.nanargmax(np.array(log_posteriors)[goodness_mask])
    ind_max_posterior = np.arange(len(log_posteriors))[goodness_mask][ind_max_posterior_with_mask]
    sqrt_e_cos_w_best, sqrt_e_sin_w_best, cos_i_best = geometry_grid[ind_max_posterior]
    ecc_best = sqrt_e_cos_w_best ** 2 + sqrt_e_sin_w_best ** 2
    w_best = np.arctan2(sqrt_e_sin_w_best, sqrt_e_cos_w_best) / np.pi * 180
    inc_best = np.arccos(cos_i_best) / np.pi * 180
    ampl_best = ampls[ind_max_posterior]
    rp_rs_best = rp_rss[ind_max_posterior]
    ind_fit_best = ind_fits[ind_max_posterior]
    flux_model_padded_wh_best = models[ind_max_posterior]

    fit_best = ampl_best * np.roll(flux_model_padded_wh_best, ind_fit_best)
    resid_best = flux - fit_best
    resid_best_fourier = np.fft.rfft(resid_best)
    # ------------------------------------------------------------------------------------------------------------------
    return_dict = {
        'geometry_grid': geometry_grid,
        'log_posteriors': log_posteriors,
        'ampls': ampls,
        'rp_rss': rp_rss,
        'ind_fits': ind_fits,
        'models': models,
        'sqrt_e_cos_w_best': sqrt_e_cos_w_best,
        'sqrt_e_sin_w_best': sqrt_e_sin_w_best,
        'cos_i_best': cos_i_best,
        'ecc_best': ecc_best,
        'w_best': w_best,
        'inc_best': inc_best,
        'ampl_best': ampl_best,
        'rp_rs_best': rp_rs_best,
        'ind_fit_best': ind_fit_best,
        'flux_model_padded_wh_best': flux_model_padded_wh_best,
        'fit_best': fit_best,
        'resid_best': resid_best,
        'resid_best_fourier': resid_best_fourier,
        'goodness_mask': goodness_mask
    }
    return return_dict


def filter_good_fit_indices(ind_fits, len_flux, log_dur, binwid, nsig_outliers=2.5, n_iterations_outliers=6,
                            n_transit_durations_max_offset=1, min_number_of_samples=10, min_cut_offset_days=4./24,
                            max_cut_offset_days=10./24, mask_for_std_computation=None):
    min_cut_offset_bins = max(int(np.round(min_cut_offset_days / binwid)), 1)
    max_cut_offset_bins = max(int(np.round(max_cut_offset_days / binwid)), 1)
    if mask_for_std_computation is None:
        mask_for_std_computation = np.ones(len(ind_fits), dtype=bool)
    if not hasattr(log_dur, '__len__'):
        unity = 1.
        dur_bins = np.exp(log_dur) / binwid
    else:
        dur_bins = np.exp(np.array(log_dur)) / binwid
        unity = np.ones(len(dur_bins))
    offset_limits = np.maximum(unity * min_cut_offset_bins, dur_bins * n_transit_durations_max_offset)
    offset_limits = np.minimum(offset_limits, unity * max_cut_offset_bins)

    ind_fits = np.array(ind_fits)
    ind_fits = np.array(ind_fits)
    ind_fits_0 = np.array(ind_fits)
    ind_fits = np.array([ind if ind <= len_flux // 2 else ind - len_flux for ind in ind_fits_0])
    offset_limits = 2 * offset_limits
    goodness_index_mask = np.abs(ind_fits) < 2 * offset_limits  # some initial pre-filtering
    mask_for_std_computation = mask_for_std_computation & goodness_index_mask
    if np.sum(mask_for_std_computation.astype(int)) < min_number_of_samples:
        return np.array([False] * len(ind_fits))
    else:
        var_outlier_free, outlier_mask = ut.outlier_free_std(ind_fits,
                                                             n_iterations=n_iterations_outliers, mask=mask_for_std_computation,
                                                             nsig=nsig_outliers, return_var=True, return_mask=True)
        goodness_index_mask = outlier_mask | (np.abs(ind_fits) < offset_limits)
        return goodness_index_mask


def bin_exact_grid_to_log_dur_b_transf_grid(geometry_grid, log_posteriors, per, a,
                                            log_transit_duration_grid_edges=None, b_transf_grid_edges=None,
                                            log_transit_duration_grid=None, b_transf_grid=None, min_log_transit_duration=-7,
                                            max_log_transit_duration=0,
                                            n_log_transit_duration_grid=40, n_b_transf_grid=30, return_values=False):
    """
    Used when the fit is done to the exact (degenerate) geometry grid of sqrt(e)sin(w) and sqrt(e)cos(w). Then, to compare it to the
    anti-degenerate grid fit, we bin it to the grid of log transit durations and impact parameters with this function.
    """
    if log_transit_duration_grid_edges is None:
        if log_transit_duration_grid is None:
            log_transit_duration_grid_edges = np.linspace(min_log_transit_duration, max_log_transit_duration,
                                                          n_log_transit_duration_grid + 1)
            # log_transit_duration_grid = get_grid_centers_from_grid_edges(grid_edges=log_transit_duration_grid_edges)
        else:
            log_transit_duration_grid_edges = get_grid_edges_from_grid_centers(grid=log_transit_duration_grid)
    # if log_transit_duration_grid is None:
    #     log_transit_duration_grid = get_grid_centers_from_grid_edges(grid_edges=log_transit_duration_grid_edges)
    if b_transf_grid_edges is None:
        if b_transf_grid is None:
            b_transf_grid_edges = np.linspace(0, 1 - 1e-5, n_b_transf_grid + 1)
        else:
            b_transf_grid_edges = get_grid_edges_from_grid_centers(grid=b_transf_grid)

    eccs_list_fit, ws_list_fit, incs_list_fit = get_physical_parameters_from_uniform_parameters_grid(geometry_grid=geometry_grid)
    cos_i_list_fit = [geometry_grid[i][Index_cos_i_grid] for i in range(len(geometry_grid))]
    impact_parameters, transit_durations = get_durations_impact_parameters_from_orbital(per=per, a=a, cos_i_list_fit=cos_i_list_fit,
                                                                                        eccs_list_fit=eccs_list_fit,
                                                                                        ws_list_fit=ws_list_fit)

    valid_transit_mask = (impact_parameters <= 1) & (transit_durations > 0)
    log_transit_duration_set = np.zeros(len(transit_durations)) + np.nan
    log_transit_duration_set[valid_transit_mask] = np.log(transit_durations[valid_transit_mask])
    bt_set = np.zeros(len(impact_parameters)) + np.nan
    bt_set[valid_transit_mask] = impact_par_transform_fun(impact_parameters[valid_transit_mask])
    log_posteriors[~valid_transit_mask] = np.nan  # setting invalid transits to very low posterior

    log_posterior_grid = make_log_posterior_2d_grid(param_1_grid_edges=log_transit_duration_grid_edges,
                                                    param_2_grid_edges=b_transf_grid_edges,
                                                    param_1_fit_list=log_transit_duration_set, param_2_fit_list=bt_set,
                                                    log_posteriors=log_posteriors)

    if not return_values:
        return log_transit_duration_grid_edges, b_transf_grid_edges, log_posterior_grid, log_transit_duration_grid, b_transf_grid
    else:
        return (log_transit_duration_grid_edges, b_transf_grid_edges, log_posterior_grid, log_transit_duration_set,
                bt_set, valid_transit_mask, log_transit_duration_grid, b_transf_grid)


# -------------- TRANSIT SHAPE FUNCTIONS -------------------
def get_impact_parameter_approx(a, ecc, cos_i, w):
    """
    a is in units of stellar radii
    w is in degrees
    """
    w_rad = w * np.pi / 180
    b = a * cos_i * (1 - ecc ** 2) / (1 + ecc * np.sin(w_rad))
    return b


def get_transit_chord_length_approx(a, ecc, cos_i, w):
    """
    a is in units of stellar radii
    w is in degrees
    """
    b = get_impact_parameter_approx(a=a, ecc=ecc, cos_i=cos_i, w=w)
    if b >= 1:
        chord = 0
    else:
        chord = 2 * (1 - b ** 2) ** 0.5
    return chord


def get_transit_velocity_approx(per, a, ecc, cos_i, w):
    """
    a is in units of stellar radii
    w is in degrees
    per is in days
    """
    w_rad = w * np.pi / 180
    sin_2_i = (1 - cos_i ** 2)
    ecc_2 = ecc ** 2
    vel = 2 * np.pi * a / (per * (1 - ecc_2) ** 0.5) * (1 + 2 * ecc * np.sin(w_rad) + ecc_2 - ecc_2 * sin_2_i * np.cos(w_rad) ** 2) ** 0.5
    return vel


def get_transit_duration_approx(per, a, ecc, cos_i, w):
    """
    a is in units of stellar radii
    w is in degrees
    per is in days
    """
    vel = get_transit_velocity_approx(per=per, a=a, ecc=ecc, cos_i=cos_i, w=w)
    chord = get_transit_chord_length_approx(a=a, ecc=ecc, cos_i=cos_i, w=w)
    dur = chord / vel
    return dur


def get_a_cosi_from_impact_duration_e0(per, dur, b):
    """
    a is in units of stellar radii
    w is in degrees
    """
    chord = 2 * (1 - b ** 2) ** 0.5
    vel = chord / dur
    a = vel * per / (2 * np.pi)
    cos_i = b / a
    # # b = a * cos_i
    # cos_i = b / a
    # # vel = get_transit_velocity_e0(per=per, a=a)  # get_transit_velocity_e0  # Transit velovity for zero eccentricity
    # vel = 2 * np.pi * a / per

    # # dur = chord / vel
    # chord = dur * vel
    # # chord = 2 * (1 - b**2)**0.5
    # b = (1 - (chord / 2)**2)**0.5
    return a, cos_i


# ------------------------------------------------

def get_grid_edges_from_grid_centers(grid):
    grid_edges = np.zeros(len(grid) + 1)
    d_grid = grid[1] - grid[0]
    grid_edges[0] = grid[0] - d_grid / 2
    for i in range(1, len(grid_edges) - 1):
        d_grid = grid[i] - grid[i - 1]
        grid_edges[0] = grid[0] - d_grid / 2
        grid_edges[i] = grid[i - 1] + d_grid / 2
    d_grid = grid[-1] - grid[-2]
    grid_edges[-1] = grid[-1] + d_grid / 2
    return grid_edges


def get_grid_centers_from_grid_edges(grid_edges):
    grid = np.zeros(len(grid_edges) - 1)
    for i in range(len(grid)):
        d_grid = grid_edges[i + 1] - grid_edges[i]
        grid[i] = grid_edges[i] + d_grid / 2
    return grid


def digitize(values_list, grid_edges=None, grid=None):
    if grid_edges is None and grid is not None:
        if grid is None:
            raise ValueError("Either `grid_edges` or `grid` must be provided.")
        grid_edges = get_grid_edges_from_grid_centers(grid=grid)
    bad_index_value = -9999999
    assert np.abs(bad_index_value) > len(grid_edges)
    d_grid = np.abs(grid_edges[1] - grid_edges[0])
    indices = np.digitize(values_list + d_grid * 1e-6, grid_edges) - 1
    mask_in_bound = np.ones(len(values_list), dtype=bool)
    mask_in_bound[np.isnan(values_list)] = False
    mask_in_bound[np.isinf(values_list)] = False
    # indices = np.array([np.nanargmin(np.abs(grid - val)) if not np.isnan(val) else bad_index_value
    # for val in values_list], dtype=int)  # this is precise but 1000 times slower
    mask_in_bound[values_list >= grid_edges[-1]] = False
    mask_in_bound[values_list < grid_edges[0]] = False
    # mask_in_bound[indices == -1] = False
    n_bins = len(grid_edges) - 1
    mask_in_bound &= (indices >= 0) & (indices < n_bins)
    mask_out_bound = ~mask_in_bound
    indices[mask_out_bound] = bad_index_value
    indices[mask_out_bound] = bad_index_value
    return indices, mask_in_bound


def make_log_posterior_2d_grid(param_1_grid_edges, param_2_grid_edges,
                               param_1_fit_list, param_2_fit_list, log_posteriors, normalize=True):
    log_posteriors_2d = np.zeros((len(param_1_grid_edges) - 1, len(param_2_grid_edges) - 1)) + np.nan
    if isinstance(param_1_fit_list, list):
        param_1_fit_list = np.array(param_1_fit_list)
    if isinstance(param_2_fit_list, list):
        param_2_fit_list = np.array(param_2_fit_list)

    param_1_indices, mask_in_bound_1 = digitize(values_list=param_1_fit_list, grid_edges=param_1_grid_edges)
    param_2_indices, mask_in_bound_2 = digitize(values_list=param_2_fit_list, grid_edges=param_2_grid_edges)

    for i in range(len(log_posteriors)):
        if mask_in_bound_1[i] and mask_in_bound_2[i]:
            ind_p1 = param_1_indices[i]
            ind_p2 = param_2_indices[i]
            if (not np.isnan(log_posteriors[i])) and (not np.isinf(log_posteriors[i])):
                lp2d_value = log_posteriors_2d[ind_p1, ind_p2]
                if np.isnan(lp2d_value) or np.isinf(lp2d_value):
                    log_posteriors_2d[ind_p1, ind_p2] = log_posteriors[i]
                else:
                    log_posteriors_2d[ind_p1, ind_p2] = np.logaddexp(
                        lp2d_value, log_posteriors[i])
    # log_posteriors_2d[np.isinf(log_posteriors_2d)] = np.nan
    if normalize:
        mask_summation = (~np.isnan(log_posteriors_2d)) & (~np.isinf(log_posteriors_2d))
        log_posteriors_2d -= np.logaddexp.reduce(log_posteriors_2d[mask_summation])
    return log_posteriors_2d


def make_log_posterior_1d_grid(param_fit_list, log_posteriors, param_grid=None, param_grid_edges=None, normalize=True):
    if param_grid_edges is None:
        param_grid_edges = get_grid_edges_from_grid_centers(grid=param_grid)
    param_indices, param_mask_in_bound = digitize(values_list=param_fit_list, grid_edges=param_grid_edges)
    log_posteriors_1d = np.zeros(len(param_grid_edges) - 1) + np.nan
    for i in range(len(log_posteriors)):
        if param_mask_in_bound[i]:
            if (not np.isnan(log_posteriors[i])) and (not np.isinf(log_posteriors[i])):
                lp1d_value = log_posteriors_1d[param_indices[i]]
                if np.isnan(lp1d_value) or np.isinf(lp1d_value):
                    log_posteriors_1d[param_indices[i]] = log_posteriors[i]
                else:
                    log_posteriors_1d[param_indices[i]] = np.logaddexp(
                        lp1d_value, log_posteriors[i])

    # log_posteriors_2d[np.isinf(log_posteriors_2d)] = np.nan
    if normalize:
        mask_summation = (~np.isnan(log_posteriors_1d)) & (~np.isinf(log_posteriors_1d))
        log_posteriors_1d -= np.logaddexp.reduce(log_posteriors_1d[mask_summation])
    return log_posteriors_1d


# --------------------------------------------------------------------------------------------------
# Anti-degenerate grid fitting

def impact_par_transform_fun(b):
    return 1 - (1. - b)**0.3
    # return b
    # return 1. - np.sqrt(1. - b ** 2)


def inv_impact_par_transform_fun(bt):
    return 1 - (1 - bt)**(1/0.3)
    # return bt
    # return np.sqrt(1. - (1. - bt) ** 2)


def get_log_dur_b_transf_a_cosi_grids(min_log_transit_duration, max_log_transit_duration, n_log_transit_duration_grid, n_b_transf_grid,
                                      per):
    log_transit_duration_grid_edges = np.linspace(min_log_transit_duration, max_log_transit_duration, n_log_transit_duration_grid + 1)
    b_transf_max = impact_par_transform_fun(1. - 1e-9)
    b_transf_grid_edges = np.linspace(0, b_transf_max, n_b_transf_grid + 1)
    log_transit_duration_grid = get_grid_centers_from_grid_edges(grid_edges=log_transit_duration_grid_edges)
    b_transf_grid = get_grid_centers_from_grid_edges(grid_edges=b_transf_grid_edges)
    # log_transit_duration_grid = np.linspace(min_log_transit_duration, max_log_transit_duration, n_log_transit_duration_grid)
    # b_transf_grid = np.linspace(0, 1-1./(n_b_transf_grid-1)/2, n_b_transf_grid)
    impact_parameter_grid_edges = inv_impact_par_transform_fun(b_transf_grid_edges)
    impact_parameter_grid = inv_impact_par_transform_fun(b_transf_grid)

    a_cosi_grid_edges = np.array([
        [get_a_cosi_from_impact_duration_e0(per=per, dur=np.exp(log_dur_i), b=b_j)
         for b_j in impact_parameter_grid_edges] for log_dur_i in log_transit_duration_grid_edges])
    a_cosi_grid = np.array([
        [get_a_cosi_from_impact_duration_e0(per=per, dur=np.exp(log_dur_i), b=b_j)
         for b_j in impact_parameter_grid] for log_dur_i in log_transit_duration_grid])
    a_cosi_grid_goodness_mask = np.ones(a_cosi_grid.shape[:2], dtype=bool)
    for i_log_dur in range(len(log_transit_duration_grid)):
        for j_bt in range(len(b_transf_grid)):
            a_i_j = a_cosi_grid[i_log_dur][j_bt][0]
            cos_i_j = a_cosi_grid[i_log_dur][j_bt][1]
            if np.isnan(a_i_j) or np.isnan(cos_i_j) or cos_i_j >= 1. or a_i_j <= 1.:
                a_cosi_grid_goodness_mask[i_log_dur, j_bt] = False
    return (log_transit_duration_grid, b_transf_grid, a_cosi_grid, a_cosi_grid_goodness_mask,
            log_transit_duration_grid_edges, b_transf_grid_edges, a_cosi_grid_edges)


# --------------------------------------------------------------------------------------------------------------------
#

def get_durations_impact_parameters_from_orbital(per, a, cos_i_list_fit, eccs_list_fit, ws_list_fit):
    impact_parameters = []
    transit_durations = []

    for i in range(len(cos_i_list_fit)):
        ecc = eccs_list_fit[i]
        cos_i = cos_i_list_fit[i]
        w = ws_list_fit[i]

        impact_parameter = get_impact_parameter_approx(a=a, ecc=ecc, cos_i=cos_i, w=w)
        transit_duration = get_transit_duration_approx(per=per, a=a, ecc=ecc, cos_i=cos_i, w=w)
        impact_parameters.append(impact_parameter)
        transit_durations.append(transit_duration)

    impact_parameters = np.array(impact_parameters)
    transit_durations = np.array(transit_durations)
    return impact_parameters, transit_durations


def filter_valid_transit_parameters(result_dict, return_bad=False):
    """
    Filtering only valid transits
    """
    # Initialization
    result_dict_new = {}
    for key in result_dict.keys():
        result_dict_new[key] = result_dict[key]

    impact_parameters = result_dict['impact_parameter_set']
    transit_durations = result_dict['transit_duration_set']
    eccs_list_fit = result_dict['ecc_set']
    ws_list_fit = result_dict['w_set']
    incs_list_fit = result_dict['inc_set']
    geometry_grid = result_dict['geometry_grid']
    a_grid = result_dict['a']
    a_prior = result_dict['a_prior']

    valid_transit_mask = (impact_parameters <= 1) & (transit_durations > 0)

    ecc_set = np.array(eccs_list_fit)[valid_transit_mask]
    w_set = np.array(ws_list_fit)[valid_transit_mask]
    inc_set = np.array(incs_list_fit)[valid_transit_mask]
    transit_duration_set = np.array(transit_durations)[valid_transit_mask]
    impact_parameter_set = np.array(impact_parameters)[valid_transit_mask]
    a_grid = np.array(a_grid)[valid_transit_mask]
    a_prior = np.array(a_prior)[valid_transit_mask]
    geometry_grid = [geometry_grid[vi] for vi in np.where(valid_transit_mask)[0]]

    result_dict_new['geometry_grid'] = geometry_grid
    result_dict_new['ecc_set'] = ecc_set
    result_dict_new['w_set'] = w_set
    result_dict_new['inc_set'] = inc_set
    result_dict_new['transit_duration_set'] = transit_duration_set
    result_dict_new['impact_parameter_set'] = impact_parameter_set
    result_dict_new['a'] = a_grid
    result_dict_new['a_prior'] = a_prior

    # --------------------------------
    if return_bad:
        ecc_bad = np.array(eccs_list_fit)[~valid_transit_mask]
        w_bad = np.array(ws_list_fit)[~valid_transit_mask]
        inc_bad = np.array(incs_list_fit)[~valid_transit_mask]
        transit_duration_bad = np.array(transit_durations)[~valid_transit_mask]
        impact_parameter_bad = np.array(impact_parameters)[~valid_transit_mask]
        a_bad = np.array(a_grid)[~valid_transit_mask]
        a_prior_bad = np.array(a_prior)[~valid_transit_mask]

        result_dict_new['ecc_bad'] = ecc_bad
        result_dict_new['inc_bad'] = inc_bad
        result_dict_new['w_bad'] = w_bad
        result_dict_new['transit_duration_bad'] = transit_duration_bad
        result_dict_new['impact_parameter_bad'] = impact_parameter_bad
        result_dict_new['a_bad'] = a_bad
        result_dict_new['a_prior_bad'] = a_prior_bad

    return result_dict_new


# ----------------------------------------------------------------------------------------------


def prepare_geometric_parameters_for_fit(per, a_grid, a_prior, e_max, d_cosi, d_sqrt_e_cos_w, return_bad=False):
    result_dict_all_a = None
    for i_a, a in enumerate(a_grid):

        geometry_grid = get_geometry_grid(a=a, e_max=e_max, dcosi=d_cosi, dsqrt_e_cos_w=d_sqrt_e_cos_w, return_separate_grids=False)
        eccs_list_fit, ws_list_fit, incs_list_fit = get_physical_parameters_from_uniform_parameters_grid(geometry_grid=geometry_grid)
        cos_i_list_fit = [geometry_grid[i][Index_cos_i_grid] for i in range(len(geometry_grid))]

        # ----------------------------------------------------------------------------------------------
        # Getting the transit shape parameters for the degenerate grid extimation

        impact_parameters, transit_durations = get_durations_impact_parameters_from_orbital(per=per, a=a, cos_i_list_fit=cos_i_list_fit,
                                                                                            eccs_list_fit=eccs_list_fit,
                                                                                            ws_list_fit=ws_list_fit)
        result_dict_one_a = {
            'a': a * np.ones(len(eccs_list_fit)),
            'a_prior': a_prior[i_a] * np.ones(len(eccs_list_fit)),
            'geometry_grid': geometry_grid,
            'ecc_set': eccs_list_fit,
            'w_set': ws_list_fit,
            'inc_set': incs_list_fit,
            'transit_duration_set': transit_durations,
            'impact_parameter_set': impact_parameters
        }
        result_dict_one_a = filter_valid_transit_parameters(result_dict=result_dict_one_a, return_bad=return_bad)
        lens = [len(result_dict_one_a[key]) for key in result_dict_one_a.keys()]
        assert np.all(np.array(lens) == lens[0]), "Lengths of the result_dict_one_a entries are not equal"
        assert lens[0] > 0, "No valid transit parameters found for a = {a}".format(a=a)
        if result_dict_all_a is None:
            result_dict_all_a = result_dict_one_a
        else:
            for key in result_dict_one_a.keys():
                if key in result_dict_all_a:
                    if key in result_dict_all_a and (len(result_dict_all_a[key]) > 0) and (len(result_dict_one_a[key]) > 0):
                        result_dict_all_a[key] = np.concatenate((result_dict_all_a[key], result_dict_one_a[key]))
                else:
                    result_dict_all_a[key] = result_dict_one_a[key]

    if result_dict_all_a is not None:
        result_dict_all_a['per'] = per
        result_dict_all_a['e_max'] = e_max
        result_dict_all_a['d_cosi'] = d_cosi
        result_dict_all_a['d_sqrt_e_cos_w'] = d_sqrt_e_cos_w
    return result_dict_all_a


# PREPARING THE PRIORS FROM THE DEGENERATE GRID
def get_priors_for_log_dur_b_transf_from_fine_grid_sets(log_transit_duration_grid_edges, b_transf_grid_edges, geometry_dict_for_prior):
    counts_models = np.zeros((len(log_transit_duration_grid_edges) - 1, len(b_transf_grid_edges) - 1), dtype=float)
    # -> counts are the geometry priors. The period and radius priors should be computed separately

    # # Preparing prior weights:
    log_transit_duration_set = np.log(geometry_dict_for_prior['transit_duration_set'])
    bt_set = impact_par_transform_fun(geometry_dict_for_prior['impact_parameter_set'])
    # d_log_transit_duration = np.abs(log_transit_duration_grid[1] - log_transit_duration_grid[0])
    # ltd_grid_indices = np.digitize(log_transit_duration_set + d_log_transit_duration * 1e-6, log_transit_duration_grid) - 1
    # d_bt = np.abs(b_transf_grid[1] - b_transf_grid[0])
    # bt_grid_indices = np.digitize(bt_set + d_bt * 1e-6, b_transf_grid) - 1
    ltd_grid_indices, mask_in_bound_ltd = digitize(values_list=log_transit_duration_set, grid_edges=log_transit_duration_grid_edges)
    bt_grid_indices, mask_in_bound_bt = digitize(values_list=bt_set, grid_edges=b_transf_grid_edges)
    a_priors = geometry_dict_for_prior['a_prior']

    total_prior = 0
    for i in range(len(ltd_grid_indices)):
        if mask_in_bound_ltd[i] and mask_in_bound_bt[i]:
            counts_models[ltd_grid_indices[i], bt_grid_indices[i]] += a_priors[i]
            total_prior += a_priors[i]
    assert total_prior > 0, "Total prior is zero!"
    geometric_priors_models = counts_models / total_prior
    return geometric_priors_models


def pre_filter_one_fine_grid(log_posteriors_grid,
                             b_transf_grid_coarse, log_transit_duration_grid_coarse,
                             log_transit_duration_grid_fine, b_transf_grid_fine,
                             add_extra_bins_fine_follow_up_log_dur=1,
                             min_frac_posterior_for_fine_follow_up_log_dur=0.01):

    indices_closest_of_fine_in_coarse_b_transf = nn_map_1d(fine_x=b_transf_grid_fine, coarse_x=b_transf_grid_coarse)
    ind_groups_fine_b_transf_inds_for_coarse = [np.flatnonzero(indices_closest_of_fine_in_coarse_b_transf == index)
                                                for index in range(len(b_transf_grid_coarse))]
    pre_filtering_mask_log_dur_b_transf_fine = np.ones((len(log_transit_duration_grid_fine),
                                                        len(b_transf_grid_fine)), dtype=bool)
    # Going over all the coarse posteriors and setting selection mask to false if the posterior is
    # low. This will halp us to avoid multiple calculations in the fine grid

    for j_bt_coarse in range(len(log_posteriors_grid[0])):
        max_posterior_row_index = np.nanargmax(log_posteriors_grid[:, j_bt_coarse])
        mask, ind_max, thr = hill_around_global_max_1d(
            data=np.exp(log_posteriors_grid[:, j_bt_coarse] - log_posteriors_grid[max_posterior_row_index, j_bt_coarse]),
            ind_max=max_posterior_row_index, frac=min_frac_posterior_for_fine_follow_up_log_dur,
            add_extra=add_extra_bins_fine_follow_up_log_dur)
        mask_indices = np.where(mask)[0]
        min_log_dur = log_transit_duration_grid_coarse[mask_indices[0]]
        max_log_dur = log_transit_duration_grid_coarse[mask_indices[-1]]
        mask_log_dur_fine = (log_transit_duration_grid_fine >= min_log_dur) & \
                            (log_transit_duration_grid_fine <= max_log_dur)

        rows_to_false = np.flatnonzero(~mask_log_dur_fine)
        cols_to_false = ind_groups_fine_b_transf_inds_for_coarse[j_bt_coarse]
        pre_filtering_mask_log_dur_b_transf_fine[np.ix_(rows_to_false, cols_to_false)] = False
    return pre_filtering_mask_log_dur_b_transf_fine


def get_pre_filtering_fine_grid_mask_for_all_u(log_posteriors_grids_for_u_list,
                                               b_transf_grid_coarse, log_transit_duration_grid_coarse,
                                               log_transit_duration_grid_fine, b_transf_grid_fine,
                                               add_extra_bins_fine_follow_up_log_dur=1,
                                               min_frac_posterior_for_fine_follow_up_log_dur=0.01):
    masks_for_us = [pre_filter_one_fine_grid(log_posteriors_grids_for_u_list[i_u],
                                             b_transf_grid_coarse, log_transit_duration_grid_coarse,
                                             log_transit_duration_grid_fine, b_transf_grid_fine,
                                             add_extra_bins_fine_follow_up_log_dur=add_extra_bins_fine_follow_up_log_dur,
                                             min_frac_posterior_for_fine_follow_up_log_dur=min_frac_posterior_for_fine_follow_up_log_dur)
                    for i_u in range(len(log_posteriors_grids_for_u_list))]
    pre_filtering_mask_total = np.array(masks_for_us[0], dtype=bool)
    for mask_u in masks_for_us[1:]:
        pre_filtering_mask_total = pre_filtering_mask_total | mask_u
    return pre_filtering_mask_total


def get_log_posteriors_for_log_dur_b_transf_grid_coarse_fine_selection_u_list(
        log_transit_duration_grid_coarse, b_transf_grid_coarse,
        priors_dur_b_coarse, a_cosi_grid_coarse, grid_goodness_mask_coarse,
        time, flux_wh, vars_flux, mask_flux, binwid,
        per, u_list, rel_wf_fd, inds_to_zero, len_high_pass,
        stellar_radius_for_prior, a_for_prior, min_cos_shift, e_max, rp_rs_mock,
        u_list_priors=None,
        print_progress=False,
        use_occurrence_rate_prior=True,
        pre_filter_coarse_grid=True,
        pre_estimate_log_duration=None,
        half_window_pre_estimate_lower=None,
        half_window_pre_estimate_upper=None,
        log_transit_duration_grid_fine=None,
        pre_filter_fine_grid=True,
        b_transf_grid_fine=None,
        priors_dur_b_fine=None,
        a_cosi_grid_fine=None,
        grid_goodness_mask_fine=None,
        min_frac_posterior_for_fine_follow_up_log_dur=0.01,
        add_extra_bins_fine_follow_up_log_dur=1,
        n_transit_durations_max_offset=1, min_cut_offset_days=4./24,):
    if half_window_pre_estimate_lower is None and (half_window_pre_estimate_upper is not None):
        half_window_pre_estimate_lower = half_window_pre_estimate_upper
    if half_window_pre_estimate_upper is None and (half_window_pre_estimate_lower is not None):
        half_window_pre_estimate_upper = half_window_pre_estimate_lower

    if pre_filter_coarse_grid and pre_estimate_log_duration is not None and half_window_pre_estimate_lower is not None:
        mask_log_dur_coarse = (log_transit_duration_grid_coarse >= pre_estimate_log_duration - half_window_pre_estimate_lower) & \
                       (log_transit_duration_grid_coarse <= pre_estimate_log_duration + half_window_pre_estimate_upper)
        grid_goodness_mask_coarse[~mask_log_dur_coarse, :] = False

    fit_dict_for_u_list_coarse = get_log_posteriors_for_log_dur_b_transf_grid_u_list(
        log_transit_duration_grid=log_transit_duration_grid_coarse,
        b_transf_grid=b_transf_grid_coarse,
        priors_dur_b=priors_dur_b_coarse, a_cosi_grid=a_cosi_grid_coarse, grid_goodness_mask=grid_goodness_mask_coarse,
        time=time, flux_wh=flux_wh, vars_flux=vars_flux, mask_flux=mask_flux, binwid=binwid,
        per=per, u_list=u_list, rel_wf_fd=rel_wf_fd, inds_to_zero=inds_to_zero,
        len_high_pass=len_high_pass, stellar_radius_for_prior=stellar_radius_for_prior,
        a_for_prior=a_for_prior, min_cos_shift=min_cos_shift, e_max=e_max, rp_rs_mock=rp_rs_mock,
        u_priors=u_list_priors, use_occurrence_rate_prior=use_occurrence_rate_prior,
        print_progress=print_progress, add_str_to_print_progress='coarse',
        n_transit_durations_max_offset=n_transit_durations_max_offset,
        min_cut_offset_days=min_cut_offset_days)

    if pre_filter_fine_grid and log_transit_duration_grid_fine is not None:
        pre_filtering_mask_total = get_pre_filtering_fine_grid_mask_for_all_u(
            log_posteriors_grids_for_u_list=fit_dict_for_u_list_coarse['log_posteriors_grid'],
            b_transf_grid_coarse=b_transf_grid_coarse,
            log_transit_duration_grid_coarse=log_transit_duration_grid_coarse,
            log_transit_duration_grid_fine=log_transit_duration_grid_fine,
            b_transf_grid_fine=b_transf_grid_fine,
            add_extra_bins_fine_follow_up_log_dur=add_extra_bins_fine_follow_up_log_dur,
            min_frac_posterior_for_fine_follow_up_log_dur=min_frac_posterior_for_fine_follow_up_log_dur)

        grid_goodness_mask_fine = grid_goodness_mask_fine & pre_filtering_mask_total

    fit_dict_for_u_list_fine = get_log_posteriors_for_log_dur_b_transf_grid_u_list(
        log_transit_duration_grid=log_transit_duration_grid_fine,
        b_transf_grid=b_transf_grid_fine,
        priors_dur_b=priors_dur_b_fine, a_cosi_grid=a_cosi_grid_fine, grid_goodness_mask=grid_goodness_mask_fine,
        time=time, flux_wh=flux_wh, vars_flux=vars_flux, mask_flux=mask_flux, binwid=binwid,
        per=per, u_list=u_list, rel_wf_fd=rel_wf_fd, inds_to_zero=inds_to_zero,
        len_high_pass=len_high_pass, stellar_radius_for_prior=stellar_radius_for_prior,
        a_for_prior=a_for_prior, min_cos_shift=min_cos_shift, e_max=e_max, rp_rs_mock=rp_rs_mock,
        u_priors=u_list_priors, use_occurrence_rate_prior=use_occurrence_rate_prior,
        print_progress=print_progress, add_str_to_print_progress='fine',
        n_transit_durations_max_offset=n_transit_durations_max_offset,
        min_cut_offset_days=min_cut_offset_days)
    return fit_dict_for_u_list_coarse, fit_dict_for_u_list_fine


def get_log_posteriors_for_log_dur_b_transf_grid_u_list(log_transit_duration_grid, b_transf_grid,
                                                        priors_dur_b, a_cosi_grid, grid_goodness_mask,
                                                        time, flux_wh, vars_flux, mask_flux, binwid,
                                                        per, u_list, rel_wf_fd, inds_to_zero, len_high_pass,
                                                        stellar_radius_for_prior, a_for_prior, min_cos_shift, e_max,
                                                        rp_rs_mock, u_priors=None, use_occurrence_rate_prior=True, print_progress=False,
                                                        add_str_to_print_progress='',
                                                        n_transit_durations_max_offset=1, min_cut_offset_days=4. / 24):
    fit_dict_for_u_list = None
    for j_u, u_j in enumerate(u_list):
        fit_dict_j = get_log_posteriors_for_log_dur_b_transf_grid(
            log_transit_duration_grid=log_transit_duration_grid, b_transf_grid=b_transf_grid,
            priors_dur_b=priors_dur_b, a_cosi_grid=a_cosi_grid, grid_goodness_mask=grid_goodness_mask,
            time=time, flux_wh=flux_wh, vars_flux=vars_flux, mask_flux=mask_flux, binwid=binwid,
            per=per, u=u_j, rel_wf_fd=rel_wf_fd, inds_to_zero=inds_to_zero,
            len_high_pass=len_high_pass, stellar_radius_for_prior=stellar_radius_for_prior,
            a_for_prior=a_for_prior, min_cos_shift=min_cos_shift, e_max=e_max,
            rp_rs_mock=rp_rs_mock, use_occurrence_rate_prior=use_occurrence_rate_prior,
            print_progress=print_progress,
            add_str_to_print_progress=str(u_j) + ' ' + add_str_to_print_progress,
            n_transit_durations_max_offset=n_transit_durations_max_offset,
            min_cut_offset_days=min_cut_offset_days)
        if u_priors is not None:
            fit_dict_j['log_posteriors'] = fit_dict_j['log_posteriors'] + np.log(u_priors[j_u])  # adding prior for u
            fit_dict_j['log_posteriors_grid'] = fit_dict_j['log_posteriors_grid'] + np.log(u_priors[j_u])  # adding prior for u
            fit_dict_j['u_prior'] = u_priors[j_u]
        else:
            fit_dict_j['u_prior'] = 1.0

        if fit_dict_for_u_list is None:
            fit_dict_for_u_list = {}
            for key in fit_dict_j.keys():
                fit_dict_for_u_list[key] = [fit_dict_j[key]]
        else:
            for key in fit_dict_j.keys():
                fit_dict_for_u_list[key].append(fit_dict_j[key])
    return fit_dict_for_u_list


# def get_log_posteriors_for_log_dur_b_transf_grid_coarse_fine_selection_for_one_u(
#         log_transit_duration_grid_coarse, b_transf_grid_coarse,
#         priors_dur_b_coarse, a_cosi_grid_coarse, grid_goodness_mask_coarse,
#         time, flux_wh, vars_flux, mask_flux, binwid,
#         per, u, rel_wf_fd, inds_to_zero, len_high_pass,
#         stellar_radius_for_prior, a_for_prior, min_cos_shift, e_max,
#         rp_rs_mock, print_progress=True,
#         use_occurrence_rate_prior=True,
#         pre_filter_coarse_grid=True,
#         pre_estimate_log_duration=None,
#         half_window_pre_estimate_lower=None,
#         half_window_pre_estimate_upper=None,
#         log_transit_duration_grid_fine=None,
#         pre_filter_fine_grid=True,
#         b_transf_grid_fine=None,
#         priors_dur_b_fine=None,
#         a_cosi_grid_fine=None,
#         grid_goodness_mask_fine=None,
#         min_frac_posterior_for_fine_follow_up_log_dur=0.01,
#         add_extra_bins_fine_follow_up_log_dur=1,
#         n_transit_durations_max_offset=1, min_cut_offset_days=4./24):
#     if half_window_pre_estimate_lower is None and (half_window_pre_estimate_upper is not None):
#         half_window_pre_estimate_lower = half_window_pre_estimate_upper
#     if half_window_pre_estimate_upper is None and (half_window_pre_estimate_lower is not None):
#         half_window_pre_estimate_upper = half_window_pre_estimate_lower
#
#     if pre_filter_coarse_grid and pre_estimate_log_duration is not None and half_window_pre_estimate_lower is not None:
#         mask_log_dur_coarse = (log_transit_duration_grid_coarse >= pre_estimate_log_duration - half_window_pre_estimate_lower) & \
#                        (log_transit_duration_grid_coarse <= pre_estimate_log_duration + half_window_pre_estimate_upper)
#         grid_goodness_mask_coarse[~mask_log_dur_coarse, :] = False
#
#     fit_dict_coarse = get_log_posteriors_for_log_dur_b_transf_grid(
#         log_transit_duration_grid=log_transit_duration_grid_coarse, b_transf_grid=b_transf_grid_coarse,
#         priors_dur_b=priors_dur_b_coarse, a_cosi_grid=a_cosi_grid_coarse, grid_goodness_mask=grid_goodness_mask_coarse,
#         time=time, flux_wh=flux_wh, vars_flux=vars_flux, mask_flux=mask_flux, binwid=binwid,
#         per=per, u=u, rel_wf_fd=rel_wf_fd, inds_to_zero=inds_to_zero, len_high_pass=len_high_pass,
#         stellar_radius_for_prior=stellar_radius_for_prior, min_cos_shift=min_cos_shift, e_max=e_max,
#         use_occurrence_rate_prior=use_occurrence_rate_prior, a_for_prior=a_for_prior,
#         rp_rs_mock=rp_rs_mock, print_progress=print_progress, add_str_to_print_progress='coarse',
#         n_transit_durations_max_offset=n_transit_durations_max_offset, min_cut_offset_days=min_cut_offset_days)
#
#     indices_closest_of_fine_in_coarse_b_transf = nn_map_1d(fine_x=b_transf_grid_fine, coarse_x=b_transf_grid_coarse)
#     ind_groups_fine_b_transf_inds_for_coarse = [np.flatnonzero(indices_closest_of_fine_in_coarse_b_transf == index)
#                                                 for index in range(len(b_transf_grid_coarse))]
#
#     if pre_filter_fine_grid and log_transit_duration_grid_fine is not None:
#         # Going over all the coarse posteriors and setting selection mask to false if the posterior is
#         # low. This will halp us to avoid multiple calculations in the fine grid
#         log_posteriors_grid = fit_dict_coarse['log_posteriors_grid']
#
#         for j_bt_coarse in range(len(log_posteriors_grid[0])):
#             max_posterior_row_index = np.nanargmax(log_posteriors_grid[:, j_bt_coarse])
#             mask, ind_max, thr = hill_around_global_max_1d(
#                 data=np.exp(log_posteriors_grid[:, j_bt_coarse] - log_posteriors_grid[max_posterior_row_index, j_bt_coarse]),
#                 ind_max=max_posterior_row_index, frac=min_frac_posterior_for_fine_follow_up_log_dur,
#                 add_extra=add_extra_bins_fine_follow_up_log_dur)
#             mask_indices = np.where(mask)[0]
#             min_log_dur = log_transit_duration_grid_coarse[mask_indices[0]]
#             max_log_dur = log_transit_duration_grid_coarse[mask_indices[-1]]
#             mask_log_dur_fine = (log_transit_duration_grid_fine >= min_log_dur) & \
#                                 (log_transit_duration_grid_fine <= max_log_dur)
#
#             rows_to_false = np.flatnonzero(~mask_log_dur_fine)
#             cols_to_false = ind_groups_fine_b_transf_inds_for_coarse[j_bt_coarse]
#             grid_goodness_mask_fine[np.ix_(rows_to_false, cols_to_false)] = False
#
#     fit_dict_fine = get_log_posteriors_for_log_dur_b_transf_grid(
#         log_transit_duration_grid=log_transit_duration_grid_fine, b_transf_grid=b_transf_grid_fine,
#         priors_dur_b=priors_dur_b_fine, a_cosi_grid=a_cosi_grid_fine, grid_goodness_mask=grid_goodness_mask_fine,
#         time=time, flux_wh=flux_wh, vars_flux=vars_flux, mask_flux=mask_flux, binwid=binwid,
#         per=per, u=u, rel_wf_fd=rel_wf_fd, inds_to_zero=inds_to_zero, len_high_pass=len_high_pass,
#         use_occurrence_rate_prior=use_occurrence_rate_prior, a_for_prior=a_for_prior,
#         stellar_radius_for_prior=stellar_radius_for_prior, min_cos_shift=min_cos_shift, e_max=e_max,
#         rp_rs_mock=rp_rs_mock, print_progress=print_progress, add_str_to_print_progress='fine',
#         n_transit_durations_max_offset=n_transit_durations_max_offset, min_cut_offset_days=min_cut_offset_days)
#     return fit_dict_coarse, fit_dict_fine


def get_log_posteriors_for_log_dur_b_transf_grid(log_transit_duration_grid, b_transf_grid,
                                                 priors_dur_b, a_cosi_grid, grid_goodness_mask,
                                                 time, flux_wh, vars_flux, mask_flux, binwid,
                                                 per, u, rel_wf_fd, inds_to_zero, len_high_pass,
                                                 stellar_radius_for_prior, a_for_prior, min_cos_shift, e_max,
                                                 rp_rs_mock, use_occurrence_rate_prior=True, print_progress=True,
                                                 add_str_to_print_progress='',
                                                 n_transit_durations_max_offset=1, min_cut_offset_days=4./24):
    min_cut_offset_bins = max(int(np.round(min_cut_offset_days / binwid)), 1)
    log_posteriors_grid = np.zeros((len(log_transit_duration_grid), len(b_transf_grid))) + np.nan
    # log_likelihoods_grid = np.zeros((len(log_transit_duration_grid), len(b_transf_grid))) + np.nan
    log_posteriors = []
    log_likelihoods = []
    log_priors = []
    ampls = []
    rp_rss = []
    ind_fits = []
    models = []
    log_durs = []
    b_transfs = []

    if print_progress:
        counter = 0
        printed_next_dec_percentage = 1
        print('Of ', len(log_transit_duration_grid) * len(b_transf_grid), add_str_to_print_progress + ':', end='\n')
        print('-' * 10, end='\n')
    for j_bt in range(len(b_transf_grid)):
        for i_log_dur in range(len(log_transit_duration_grid)):
            dur = np.exp(log_transit_duration_grid[i_log_dur])
            half_width_index_limit_bins = max(int(np.round(dur / binwid)), min_cut_offset_bins)
            prior_weight = priors_dur_b[i_log_dur, j_bt]
            if grid_goodness_mask[i_log_dur, j_bt]:
                # Weighting the log posterior by the prior weight from the grid
                if prior_weight > 0:
                    a_i_j = a_cosi_grid[i_log_dur][j_bt][0]
                    cos_i_j = a_cosi_grid[i_log_dur][j_bt][1]
                    log_posterior, rp_rs_est, ampl, ind_fit, flux_model_padded_wh, log_likelihood = fit_flux_par(
                        time=time, flux_wh=flux_wh, vars_flux=vars_flux, mask_flux=mask_flux, binwid=binwid,
                        sqrt_e_cos_w=0., sqrt_e_sin_w=0., cos_i=cos_i_j,
                        per=per, a=a_i_j, u=u, rel_wf_fd=rel_wf_fd, inds_to_zero=inds_to_zero, len_high_pass=len_high_pass,
                        stellar_radius_for_prior=stellar_radius_for_prior, min_cos_shift=min_cos_shift, e_max=e_max, rp_rs_mock=rp_rs_mock,
                        rp_rs_max=1., return_model=True, use_geometry_for_prior=False, a_for_prior=a_for_prior,
                        use_occurrence_rate_prior=use_occurrence_rate_prior,
                        half_width_bins_limit_fit_indices=half_width_index_limit_bins)
                    log_posterior += np.log(prior_weight)
                else:
                    log_posterior, rp_rs_est, ampl, ind_fit, flux_model_padded_wh, log_likelihood = (
                        -np.inf, np.nan, np.nan, 0, np.zeros_like(flux_wh) + np.nan, np.nan)
            else:
                log_posterior, rp_rs_est, ampl, ind_fit, flux_model_padded_wh, log_likelihood = (
                    -np.inf, np.nan, np.nan, 0, np.zeros_like(flux_wh) + np.nan, np.nan)

            log_durs.append(log_transit_duration_grid[i_log_dur])
            b_transfs.append(b_transf_grid[j_bt])
            log_priors.append(np.log(prior_weight))
            log_posteriors_grid[i_log_dur, j_bt] = log_posterior
            log_posteriors.append(log_posterior)
            # log_likelihoods_grid[i_log_dur, j_bt] = log_likelihood
            log_likelihoods.append(log_likelihood)
            ampls.append(ampl)
            rp_rss.append(rp_rs_est)
            ind_fits.append(ind_fit)
            models.append(flux_model_padded_wh)

            if print_progress:
                counter += 1
                printed_dec_percentage = int(counter / (len(log_transit_duration_grid) * len(b_transf_grid)) * 10)
                if printed_dec_percentage > printed_next_dec_percentage:
                    print('=', end='')
                    printed_next_dec_percentage += 1

    if print_progress:
        clear_output(wait=False)

    log_durs = np.array(log_durs)
    b_transfs = np.array(b_transfs)
    log_posteriors = np.array(log_posteriors)
    log_priors = np.array(log_priors)
    mask_select_for_good_indices = ~np.isnan(log_posteriors) & ~np.isinf(log_posteriors)
    goodness_index_mask = filter_good_fit_indices(ind_fits=ind_fits, len_flux=len(flux_wh),
                                                  log_dur=log_durs, binwid=binwid,
                                                  n_transit_durations_max_offset=int(n_transit_durations_max_offset),
                                                  min_cut_offset_days=min_cut_offset_days,
                                                  mask_for_std_computation=mask_select_for_good_indices)
    log_posteriors[~goodness_index_mask] = np.nan  # Attention!
    goodness_mask = goodness_index_mask & (~np.isnan(log_posteriors)) & (~np.isinf(log_posteriors))

    return_dict = {}
    return_dict['log_posteriors'] = log_posteriors
    return_dict['log_likelihoods'] = log_likelihoods
    return_dict['log_priors'] = log_priors
    return_dict['log_posteriors_grid'] = log_posteriors_grid
    return_dict['ampls'] = ampls
    return_dict['rp_rss'] = rp_rss
    return_dict['ind_fits'] = ind_fits
    # return_dict['models'] = models
    return_dict['log_durs'] = log_durs
    return_dict['b_transfs'] = b_transfs
    return_dict['goodness_index_mask'] = goodness_index_mask

    # ------------------------------------------------------------------------------------------------------------------
    # The global maximum posterior model and its amplitude
    # ------------------------------------------------------------------------------------------------------------------
    if np.sum(goodness_mask) > 0:
        has_good_ind_samples = True
        ind_max_posterior_with_mask = np.nanargmax(np.array(log_posteriors)[goodness_mask])
        ind_max_posterior = np.arange(len(log_posteriors))[goodness_mask][ind_max_posterior_with_mask]
        log_dur_best = log_durs[ind_max_posterior]
        bt_best = b_transfs[ind_max_posterior]
        a_best, cos_i_best = get_a_cosi_from_impact_duration_e0(per=per, dur=np.exp(log_dur_best), b=bt_best)
        inc_best = np.arccos(cos_i_best) / np.pi * 180
        ampl_best = ampls[ind_max_posterior]
        rp_rs_best = rp_rss[ind_max_posterior]
        ind_fit_best = ind_fits[ind_max_posterior]
        flux_model_padded_wh_best = models[ind_max_posterior]

        fit_best = ampl_best * np.roll(flux_model_padded_wh_best, ind_fit_best)
        resid_best = flux_wh - fit_best
        resid_best_fourier = np.fft.rfft(resid_best)
    else:
        has_good_ind_samples = False
        ind_max_posterior = None
        log_dur_best = np.nan
        bt_best = np.nan
        a_best = np.nan
        cos_i_best = np.nan
        inc_best = np.nan
        ampl_best = np.nan
        rp_rs_best = np.nan
        ind_fit_best = None
        flux_model_padded_wh_best = np.array(flux_wh) + np.nan
        fit_best = np.array(flux_wh) + np.nan
        resid_best = np.array(flux_wh) + np.nan
        resid_best_fourier = np.array(flux_wh) + np.nan

    return_dict['ind_max_posterior'] = ind_max_posterior
    return_dict['log_dur_best'] = log_dur_best
    return_dict['bt_best'] = bt_best
    return_dict['a_best'] = a_best
    return_dict['cos_i_best'] = cos_i_best
    return_dict['inc_best'] = inc_best
    return_dict['ampl_best'] = ampl_best
    return_dict['rp_rs_best'] = rp_rs_best
    return_dict['ind_fit_best'] = ind_fit_best
    return_dict['flux_model_padded_wh_best'] = flux_model_padded_wh_best
    return_dict['fit_best'] = fit_best
    return_dict['resid_best'] = resid_best
    return_dict['resid_best_fourier'] = resid_best_fourier
    return_dict['has_good_ind_samples'] = has_good_ind_samples
    return_dict['u'] = u

    return return_dict


def find_median_and_percentiles_from_log_posterior(parameter_values, log_posteriors, mask=None, percentile_lower=16., percentile_upper=84.):
    if mask is None:
        mask = np.ones(len(parameter_values), dtype=bool)
    mask = np.array(mask, dtype=bool)
    mask &= (~np.isnan(parameter_values))
    mask &= (~np.isinf(parameter_values))
    mask &= (~np.isnan(log_posteriors))
    mask &= (~np.isinf(log_posteriors))
    parameter_values = np.array(parameter_values)[mask]
    log_posteriors = np.array(log_posteriors)[mask]
    median = weighted_percentile(values=parameter_values, percentile=50, weights=np.exp(log_posteriors - np.nanmax(log_posteriors)),
                                 mask=None)
    percentile_lower = weighted_percentile(values=parameter_values, percentile=percentile_lower,
                                           weights=np.exp(log_posteriors - np.nanmax(log_posteriors)),
                                           mask=None)
    percentile_upper = weighted_percentile(values=parameter_values, percentile=percentile_upper,
                                           weights=np.exp(log_posteriors - np.nanmax(log_posteriors)),
                                           mask=None)
    return median, percentile_lower, percentile_upper


# ----------------------------------------------------------------------------------------------------------------------
# Parameters distribution fitting
def _skewed_normal_difference(params, median, perc_value_lower, perc_value_upper, perc_lower=0.16, perc_upper=0.84):
    from scipy.stats import skewnorm
    loc, scale, a = params
    distr = skewnorm(a=a, loc=loc, scale=scale)
    perc_value_lower_here = distr.ppf(perc_lower)
    perc_value_upper_here = distr.ppf(perc_upper)
    median_here = distr.median()
    return ((perc_value_upper_here - perc_value_upper) ** 2 + (perc_value_lower_here - perc_value_lower) ** 2 + (
                median_here - median) ** 2) ** 0.5


def fit_skew_normal_to_percetiles(value_median, value_lower, value_upper, perc_lower=0.16, perc_upper=0.84):
    from scipy.optimize import minimize

    std_0 = (value_upper - value_lower) / 2
    params = {'median': value_median, 'perc_value_lower': value_lower, 'perc_value_upper': value_upper,
              'perc_lower': perc_lower, 'perc_upper': perc_upper}
    sol = minimize(lambda x: _skewed_normal_difference(x, **params), x0=[value_median, std_0, 0])
    #    bounds=[(r_star_lower, r_star_upper), (std_0/5, std_0*5), (0, None)])
    loc_sol, scale_sol, a_sol = sol.x
    return loc_sol, scale_sol, a_sol, sol


def get_sample_of_a(period, mass, mass_lower, mass_upper, radius, radius_lower, radius_upper,
                    sample_size, perc_lower=0.16, perc_upper=0.84):
    loc_radius, scale_radius, a_radius, sol_radius = fit_skew_normal_to_percetiles(
        value_median=radius, value_lower=radius_lower, value_upper=radius_upper,
        perc_lower=perc_lower, perc_upper=perc_upper)
    loc_mass, scale_mass, a_mass, sol_mass = fit_skew_normal_to_percetiles(
        value_median=mass, value_lower=mass_lower, value_upper=mass_upper,
        perc_lower=perc_lower, perc_upper=perc_upper)
    sample_radius = skewnorm.rvs(a=a_radius, loc=loc_radius, scale=scale_radius, size=sample_size)
    sample_mass = skewnorm.rvs(a=a_mass, loc=loc_mass, scale=scale_mass, size=sample_size)
    a_coef = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * M_sun ** (1 / 3) / R_sun
    a_sim_in_radii = a_coef * (sample_mass ** (1 / 3) / sample_radius)
    a_sim = a_sim_in_radii * sample_radius
    return a_sim_in_radii, a_sim, sample_mass, sample_radius


def get_stratified_sample_of_a(period, mass, mass_lower, mass_upper, radius, radius_lower, radius_upper,
                               sample_size, perc_lower=0.16, perc_upper=0.84):
    """Get either a in stellar radii or in solar radii"""
    loc_radius, scale_radius, a_radius, sol_radius = fit_skew_normal_to_percetiles(
        value_median=radius, value_lower=radius_lower, value_upper=radius_upper,
        perc_lower=perc_lower, perc_upper=perc_upper)
    loc_mass, scale_mass, a_mass, sol_mass = fit_skew_normal_to_percetiles(
        value_median=mass, value_lower=mass_lower, value_upper=mass_upper,
        perc_lower=perc_lower, perc_upper=perc_upper)
    sample_radius = ut.get_stratified_skewnorm_rvs(n=sample_size, a=a_radius, loc=loc_radius, scale=scale_radius)
    np.random.shuffle(
        sample_radius)  # Shuffle is very important! otherwise, they will be ordered, and the ratio mass/radius will be biased!!!!
    sample_mass = ut.get_stratified_skewnorm_rvs(n=sample_size, a=a_mass, loc=loc_mass, scale=scale_mass)
    np.random.shuffle(
        sample_mass)  # Shuffle is very important! otherwise, they will be ordered, and the ratio mass/radius will be biased!!!!
    a_coef = (G / (4 * np.pi ** 2) * (period * 24 * 60 * 60) ** 2) ** (1 / 3) * M_sun ** (1 / 3) / R_sun
    a_sim_in_radii = a_coef * (sample_mass ** (1 / 3) / sample_radius)
    a_sim = a_sim_in_radii * sample_radius  # this is in solar radii
    return a_sim_in_radii, a_sim, sample_mass, sample_radius


def get_emulated_prior_for_a(period, mass, mass_lower, mass_upper,
                             radius, radius_lower, radius_upper, sample_size_simulation, perc_lower, perc_upper,
                             perc_lower_a_grid, perc_upper_a_grid, n_a_grid):
    a_sim_in_radii_strat, a_sim_strat, sample_mass_strat, sample_radius_strat = get_stratified_sample_of_a(
        period=period, mass=mass, mass_lower=mass_lower, mass_upper=mass_upper,
        radius=radius, radius_lower=radius_lower, radius_upper=radius_upper,
        sample_size=sample_size_simulation, perc_lower=perc_lower, perc_upper=perc_upper)

    # Making the grid of a

    a_sim_in_radii_strat = a_sim_in_radii_strat[(~np.isnan(a_sim_in_radii_strat)) & (~np.isinf(a_sim_in_radii_strat))]
    # a_sim_in_radii_strat_median = np.median(a_sim_in_radii_strat)
    a_sim_in_radii_strat_lower = np.percentile(a_sim_in_radii_strat, perc_lower_a_grid * 100)
    a_sim_in_radii_strat_upper = np.percentile(a_sim_in_radii_strat, perc_upper_a_grid * 100)
    a_grid_edges = np.linspace(a_sim_in_radii_strat_lower, a_sim_in_radii_strat_upper, n_a_grid + 1)
    d_a_grid = a_grid_edges[1] - a_grid_edges[0]
    a_grid = a_grid_edges[:-1] + d_a_grid / 2
    a_prior_counts, _ = np.histogram(a_sim_in_radii_strat, bins=a_grid_edges)
    a_prior = a_prior_counts / np.sum(a_prior_counts) * (perc_upper_a_grid - perc_lower_a_grid)  # / d_a_grid  # Normalized prior density
    # TODO think whether priors should be integrals with measure -  normalized of sum-to-unity-normalized
    return a_prior, a_grid, a_grid_edges


# ------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# Functions of stellar parematers
def get_gaia_stellar_parameters_for_target_id(target_id, gaia_kepler_file=Gaia_kepler_file):
    from catalog_functions import read_gaia_table_for_target_id
    tid_gaia_table = read_gaia_table_for_target_id(tid_list=[target_id], file_path=gaia_kepler_file)
    res_dict = {}
    for field_name in [
        'radius_gspphot',
        'radius_gspphot_lower',
        'radius_gspphot_upper',
        'radius_flame',
        'radius_flame_lower',
        'radius_flame_upper',
        'mass_flame',
        'mass_flame_lower',
        'mass_flame_upper',
        'teff_gspphot_lower',
        'teff_gspphot',
        'teff_gspphot_upper',
        'logg_gspphot_lower',
        'logg_gspphot',
        'logg_gspphot_upper',
        'mh_gspphot_lower',
        'mh_gspphot',
        'mh_gspphot_upper']:
        res_dict[field_name] = tid_gaia_table[field_name].data.data[0]
    return res_dict


def get_berger_stellar_parameters_for_target_id(target_id, berger_file_path=Berger_table_file):
    from catalog_functions import read_berger_table
    berger_table = read_berger_table(berger_table_file=berger_file_path)
    berger_indices = np.where(berger_table['target_id'].data == target_id)[0]
    berger_names_dict = {
        'stellar_mass': 'Mass',
        'stellar_radius': 'Rad',
        'teff': 'Teff',
        'logg': 'logg',
        'metallicity': '[Fe/H]'
    }
    berger_lower_error_name_dict = {
        'stellar_mass': 'e_Mass',
        'stellar_radius': 'e_Rad',
        'teff': 'e_Teff',
        'logg': 'e_logg',
        'metallicity': 'e_[Fe/H]'
    }
    berger_upper_error_name_dict = {
        'stellar_mass': 'E_Mass',
        'stellar_radius': 'E_Rad',
        'teff': 'E_Teff',
        'logg': 'E_logg',
        'metallicity': 'E_[Fe/H]'
    }
    res_dict = {}
    if len(berger_indices) > 0:
        berger_index = berger_indices[0]
        for key in berger_names_dict.keys():
            field_name = berger_names_dict[key]
            lower_error_name = berger_lower_error_name_dict[key]
            upper_error_name = berger_upper_error_name_dict[key]
            value = berger_table[field_name].data[berger_index]
            lower_error = berger_table[lower_error_name].data[berger_index]
            upper_error = berger_table[upper_error_name].data[berger_index]
            lower_value = value - lower_error
            upper_value = value + upper_error
            res_dict[key] = value
            res_dict[key + '_lower'] = lower_value
            res_dict[key + '_upper'] = upper_value
    else:
        raise ValueError("Target ID {tid} not found in Berger et al. (2020) table".format(tid=target_id))
    return res_dict


def get_stellar_parameters_for_target_id(target_id, gaia_kepler_file=Gaia_kepler_file, berger_file_path=Berger_table_file):
    gaia_names_dict = {
        'stellar_mass': 'mass_flame',
        'stellar_mass_lower': 'mass_flame_lower',
        'stellar_mass_upper': 'mass_flame_upper',
        'stellar_radius': 'radius_gspphot',
        'stellar_radius_lower': 'radius_gspphot_lower',
        'stellar_radius_upper': 'radius_gspphot_upper',
        'teff': 'teff_gspphot',
        'teff_lower': 'teff_gspphot_lower',
        'teff_upper': 'teff_gspphot_upper',
        'logg': 'logg_gspphot',
        'logg_lower': 'logg_gspphot_lower',
        'logg_upper': 'logg_gspphot_upper',
        'metallicity': 'mh_gspphot',
        'metallicity_lower': 'mh_gspphot_lower',
        'metallicity_upper': 'mh_gspphot_upper'
    }

    res_dict_gaia = get_gaia_stellar_parameters_for_target_id(target_id=target_id, gaia_kepler_file=gaia_kepler_file)
    res_dict = {}
    for key in gaia_names_dict.keys():
        field_name = gaia_names_dict[key]
        res_dict[key] = res_dict_gaia[field_name]

    if (np.isnan(res_dict['stellar_mass']) or np.isnan(res_dict['stellar_radius'])):
        res_dict = get_berger_stellar_parameters_for_target_id(target_id=target_id, berger_file_path=berger_file_path)
    return res_dict


def get_teff_logg_mh_for_target_id(target_id, gaia_kepler_file=Gaia_kepler_file):
    from catalog_functions import read_gaia_table_for_target_id
    tid_gaia_table = read_gaia_table_for_target_id(tid_list=[target_id], file_path=gaia_kepler_file)

    teff_gspphot_lower = tid_gaia_table['teff_gspphot_lower'].data.data[0]
    teff_gspphot = tid_gaia_table['teff_gspphot'].data.data[0]
    teff_gspphot_upper = tid_gaia_table['teff_gspphot_upper'].data.data[0]

    logg_gspphot_lower = tid_gaia_table['logg_gspphot_lower'].data.data[0]
    logg_gspphot = tid_gaia_table['logg_gspphot'].data.data[0]
    logg_gspphot_upper = tid_gaia_table['logg_gspphot_upper'].data.data[0]

    mh_gspphot_lower = tid_gaia_table['mh_gspphot_lower'].data.data[0]
    mh_gspphot = tid_gaia_table['mh_gspphot'].data.data[0]
    mh_gspphot_upper = tid_gaia_table['mh_gspphot_upper'].data.data[0]
    return (teff_gspphot_lower, teff_gspphot, teff_gspphot_upper,
            logg_gspphot_lower, logg_gspphot, logg_gspphot_upper,
            mh_gspphot_lower, mh_gspphot, mh_gspphot_upper)


def add_systematic_errors_lower_upper(value_median, value_lower, value_upper, systematic_rel_error_lower=0.05,
                                      systematic_rel_error_upper=0.05,
                                      systematic_abs_error_lower=0.0, systematic_abs_error_upper=0.0):
    if systematic_abs_error_lower is None:
        systematic_abs_error_lower = 0.0
    if systematic_abs_error_upper is None:
        systematic_abs_error_upper = 0.0
    if systematic_rel_error_lower is None:
        systematic_rel_error_lower = 0.0
    if systematic_rel_error_upper is None:
        systematic_rel_error_upper = 0.0

    value_lower_diff = np.abs(value_median - value_lower)
    value_lower = value_median - (
                value_lower_diff ** 2 + (systematic_rel_error_lower * value_median) ** 2 + (systematic_abs_error_lower) ** 2) ** 0.5
    value_upper_diff = np.abs(value_upper - value_median)
    value_upper = value_median + (
                value_upper_diff ** 2 + (systematic_rel_error_upper * value_median) ** 2 + (systematic_abs_error_upper) ** 2) ** 0.5
    return value_lower, value_upper


# ----------------------------------------------------------------------------------------------------------------------
# Limb darkening functions

def get_ldc_sample_for_target_id(target_id, sample_size=100, systematic_rel_error_teff=0.035, systematic_rel_error_logg=0.15,
                                 systematic_abs_error_mh=0.15,
                                 teff_lower=None, teff=None, teff_upper=None,
                                 logg_lower=None, logg=None, logg_upper=None,
                                 mh_lower=None, mh=None, mh_upper=None):
    if (teff is not None) or (logg is not None) or (mh is not None) or (teff_lower is not None) or (teff_upper is not None) \
            or (logg_lower is not None) or (logg_upper is not None) or (mh_lower is not None) or (mh_upper is not None):
        assert (teff is not None) and (logg is not None) and (mh is not None) and (teff_lower is not None) and (teff_upper is not None) \
               and (logg_lower is not None) and (logg_upper is not None) and (mh_lower is not None) and (mh_upper is not None), \
            "If any of the stellar parameters is provided, all of them must be provided"
    else:
        teff_gspphot_lower, teff_gspphot, teff_gspphot_upper, logg_gspphot_lower, logg_gspphot, \
            logg_gspphot_upper, mh_gspphot_lower, mh_gspphot, mh_gspphot_upper = get_teff_logg_mh_for_target_id(target_id)
        teff_lower, teff, teff_upper, logg_lower, logg, logg_upper, mh_lower, mh, mh_upper = \
            teff_gspphot_lower, teff_gspphot, teff_gspphot_upper, logg_gspphot_lower, logg_gspphot, \
                logg_gspphot_upper, mh_gspphot_lower, mh_gspphot, mh_gspphot_upper
    # adding systematic errors:
    teff_lower, teff_upper = add_systematic_errors_lower_upper(
        value_median=teff, value_lower=teff_lower, value_upper=teff_upper,
        systematic_rel_error_lower=systematic_rel_error_teff, systematic_rel_error_upper=systematic_rel_error_teff)
    logg_lower, logg_upper = add_systematic_errors_lower_upper(
        value_median=logg, value_lower=logg_lower, value_upper=logg_upper,
        systematic_rel_error_lower=systematic_rel_error_logg, systematic_rel_error_upper=systematic_rel_error_logg)
    mh_lower, mh_upper = add_systematic_errors_lower_upper(
        value_median=mh, value_lower=mh_lower, value_upper=mh_upper,
        systematic_rel_error_lower=0.0, systematic_rel_error_upper=0.0,
        systematic_abs_error_lower=systematic_abs_error_mh, systematic_abs_error_upper=systematic_abs_error_mh)

    return get_ldc_sample_for_stellar_params(teff_lower=teff_lower, teff=teff,
                                             teff_upper=teff_upper, logg_lower=logg_lower, logg=logg,
                                             logg_upper=logg_upper, mh_lower=mh_lower, mh=mh,
                                             mh_upper=mh_upper, sample_size=sample_size)


def get_ldc_sample_for_stellar_params(teff_lower, teff, teff_upper, logg_lower, logg,
                                      logg_upper, mh_lower, mh, mh_upper, sample_size=100):
    # Making effective distributions and sampling parameters:
    loc_teff, scale_teff, a_teff, sol_teff = fit_skew_normal_to_percetiles(
        value_median=teff, value_lower=teff_lower, value_upper=teff_upper)

    sample_teff = ut.get_stratified_skewnorm_rvs(n=sample_size, a=a_teff, loc=loc_teff, scale=scale_teff)
    np.random.shuffle(sample_teff)

    loc_logg, scale_logg, a_logg, sol_logg = fit_skew_normal_to_percetiles(
        value_median=logg, value_lower=logg_lower, value_upper=logg_upper)
    sample_logg = ut.get_stratified_skewnorm_rvs(n=sample_size, a=a_logg, loc=loc_logg, scale=scale_logg)
    np.random.shuffle(sample_logg)

    loc_mh, scale_mh, a_mh, sol_mh = fit_skew_normal_to_percetiles(
        value_median=mh, value_lower=mh_lower, value_upper=mh_upper)
    sample_mh = ut.get_stratified_skewnorm_rvs(n=sample_size, a=a_mh, loc=loc_mh, scale=scale_mh)
    np.random.shuffle(sample_mh)

    # calculating limb-darkening coefficients
    u1_list, u2_list = tm.get_limb_darkening_coefs(teff=sample_teff, logg=sample_logg, metallicity=sample_mh, interp=True)
    return u1_list, u2_list


phi_ldc_correlation = 40. / 180 * np.pi
lcd_cor_coef = 0.93


def get_u_from_gamma(gamma_1, gamma_2, phi=phi_ldc_correlation):
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    u1 = gamma_1 * cos_phi + gamma_2 * sin_phi
    u2 = - gamma_1 * sin_phi + gamma_2 * cos_phi
    return u1, u2


def get_gamma_from_u(u1, u2, phi=phi_ldc_correlation):
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    gamma_1 = u1 * cos_phi - u2 * sin_phi
    gamma_2 = u1 * sin_phi + u2 * cos_phi
    return gamma_1, gamma_2


def get_q_from_gamma(gamma_1, gamma_2, phi=phi_ldc_correlation):
    u1, u2 = get_u_from_gamma(gamma_1=gamma_1, gamma_2=gamma_2, phi=phi)
    q1 = (u1 + u2) ** 2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2


def get_u_from_q(q1, q2):
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return u1, u2


def get_u_condition_mask(us_1, us_2):
    """
    Checks which limb darkening coefficient pairs are physical and which are not. See Kipping 2013.
    us_1 and us_2 should be arrays or floats"""
    mask = ((us_1 + us_2) < 1) & (us_1 > 0) & ((us_1 + 2 * us_2) > 0)
    return mask


def get_u_uncertainty_samples(u, n_samples, std_max_rel=0.2, cor_coef=lcd_cor_coef, phi=phi_ldc_correlation):
    """
    Docstring for get_u_uncertainty_samples

    :param u: Central values of limb darkening coefficients (u1, u2)
    :param n_samples: Number of samples to generate
    :param std_max_rel: Maximal relative standard deviation in gamma space (std for the larger axis of the correlation ellipse)
    :param cor_coef: Correlation coefficient between u1 and u2
    :param phi: Rotation angle between u and gamma spaces (correlation ellipse rotation)
    """
    u1, u2 = u
    gamma1, gamma2 = get_gamma_from_u(u1=u1, u2=u2, phi=phi)

    std_ratio = ((1 + cor_coef) / (1 - cor_coef)) ** 0.5
    std_max = max(gamma1, gamma2) * std_max_rel
    std_min = std_max / std_ratio
    gammas_1 = np.random.normal(loc=gamma1, scale=std_max, size=n_samples)
    gammas_2 = np.random.normal(loc=gamma2, scale=std_min, size=n_samples)

    us_1, us_2 = get_u_from_gamma(gamma_1=gammas_1, gamma_2=gammas_2, phi=phi)
    mask = get_u_condition_mask(us_1=us_1, us_2=us_2)
    us_1 = us_1[mask]
    us_2 = us_2[mask]
    return us_1, us_2


# Making a histogram on a grid and then selecting cells that have high enough prior

def make_grid_prior_u_from_sample(us_1, us_2, n_grid_u_0=10, min_prob=0.02, return_full_grid=False):
    """
    n_grid_u_0 : Intial number of grid cells, sets the fineness of the final grid
    min_prob :  Minimal prior in a cell
    """
    u_1_grid_edges_0 = np.linspace(0, 2, n_grid_u_0 + 1)
    u_2_grid_edges_0 = np.linspace(-1, 1, n_grid_u_0 + 1)
    u_1_grid_centers = 0.5 * (u_1_grid_edges_0[1:] + u_1_grid_edges_0[:-1])
    u_2_grid_centers = 0.5 * (u_2_grid_edges_0[1:] + u_2_grid_edges_0[:-1])
    hist_u = np.histogram2d(us_1, us_2, bins=[u_1_grid_edges_0, u_2_grid_edges_0], density=True)[0]

    mgr = np.array(np.meshgrid(u_1_grid_centers, u_2_grid_centers, indexing='xy')).transpose(2, 1, 0)
    assert (mgr.shape[:2] == hist_u.shape)
    us_pairs_grid = np.concatenate(mgr)
    us_pairs_hist = np.concatenate(hist_u)

    # Selecting cells
    grid_indices_selected = np.where(us_pairs_hist > min_prob)[0]
    us_pairs_selected = us_pairs_grid[grid_indices_selected]
    us_pairs_hist_selected = us_pairs_hist[grid_indices_selected]
    if not return_full_grid:
        return us_pairs_selected, us_pairs_hist_selected  # , hist_u, u_1_grid_0, u_2_grid_0
    else:
        return us_pairs_selected, us_pairs_hist_selected, hist_u, u_1_grid_edges_0, u_2_grid_edges_0


def get_u_prior_and_grid_for_target_id(target_id, n_centers=100, n_samples_unc=100, std_max_rel=0.2, n_grid_u_0=10, min_prob=0.05,
                                       cor_coef=lcd_cor_coef, phi=phi_ldc_correlation, return_full_grid=False,
                                       systematic_rel_error_teff=0.035, systematic_rel_error_logg=0.15,
                                       systematic_abs_error_mh=0.15,
                                       teff_lower=None, teff=None, teff_upper=None,
                                       logg_lower=None, logg=None, logg_upper=None,
                                       mh_lower=None, mh=None, mh_upper=None):
    """
    Docstring for get_u_prior_and_grid_for_target_id

    :param target_id:
    :param n_centers:
    :param n_samples_unc:
    :param u: Central values of limb darkening coefficients (u1, u2)
    :param n_samples: Number of samples to generate
    :param std_max_rel: Maximal relative standard deviation in gamma space (std for the larger axis of the correlation ellipse)
    :param cor_coef: Correlation coefficient between u1 and u2
    :param phi: Rotation angle between u and gamma spaces (correlation ellipse rotation)
    :param n_grid_u_0:
    :param min_prob:
    :param return_full_grid:
    :param systematic_rel_error_teff:
    :param systematic_rel_error_logg:
    :param systematic_abs_error_mh:
    """
    u1_centers, u2_centers = get_ldc_sample_for_target_id(target_id=target_id, sample_size=n_centers,
                                                          systematic_rel_error_teff=systematic_rel_error_teff,
                                                          systematic_rel_error_logg=systematic_rel_error_logg,
                                                          systematic_abs_error_mh=systematic_abs_error_mh,
                                                          teff_lower=teff_lower, teff=teff, teff_upper=teff_upper,
                                                          logg_lower=logg_lower, logg=logg, logg_upper=logg_upper,
                                                          mh_lower=mh_lower, mh=mh, mh_upper=mh_upper)
    u1_s = np.empty(shape=(0,))
    u2_s = np.empty(shape=(0,))
    for i in range(len(u1_centers)):
        u1_s_unc, u2_s_unc = get_u_uncertainty_samples((u1_centers[i], u2_centers[i]), n_samples=n_samples_unc, std_max_rel=std_max_rel,
                                                       cor_coef=cor_coef, phi=phi)
        u1_s = np.concatenate((u1_s, u1_s_unc))
        u2_s = np.concatenate((u2_s, u2_s_unc))

    res = make_grid_prior_u_from_sample(us_1=u1_s, us_2=u2_s, n_grid_u_0=n_grid_u_0, min_prob=min_prob, return_full_grid=return_full_grid)
    us_pairs_selected, us_pairs_hist_selected = res[:2]
    if not return_full_grid:
        return us_pairs_selected, us_pairs_hist_selected
    else:
        hist_u, u_1_grid_edges_0, u_2_grid_edges_0 = res[2:]
        return us_pairs_selected, us_pairs_hist_selected, hist_u, u_1_grid_edges_0, u_2_grid_edges_0


def get_depth_factor_for_u_pairs(u_pairs, per=100, a=90, inc=90., ecc=0., w=90., t0=0, rp=0.02, dt=tm.dt_lc):
    was_array = False

    # u_pairs should be pairs, so it is an array by definition. but check whether there are many pairs
    len_u = None
    for i in range(len(u_pairs)):
        # u_pair can be nan sometimes, so we do not check u[0], we iterate over all
        if hasattr(u_pairs[i], '__len__'):
            len_u = len(u_pairs)

    len_of_someone = len_u
    lens_params = [len_u]
    for param in [per, a, inc, ecc, w, t0, rp]:
        if hasattr(param, '__len__'):
            len_of_someone = len(param)
            lens_params.append(len(param))
        else:
            lens_params.append(None)

    lens_params_having_length = [l for l in lens_params if l is not None]
    if len(lens_params_having_length) > 0:
        len_of_someone = lens_params_having_length[0]
        was_array = True
        assert (np.array(
            lens_params_having_length) == len_of_someone).all(), "All parameters should be either scalars or arrays of the same length"
    else:
        len_of_someone = 1
        was_array = False

    # Broadcasting parameters
    if len_u is None:
        u_pairs = [u_pairs] * len_of_someone
    if not hasattr(per, '__len__'):
        per = np.ones(len_of_someone) * per
    if not hasattr(a, '__len__'):
        a = np.ones(len_of_someone) * a
    if not hasattr(inc, '__len__'):
        inc = np.ones(len_of_someone) * inc
    if not hasattr(ecc, '__len__'):
        ecc = np.ones(len_of_someone) * ecc
    if not hasattr(w, '__len__'):
        w = np.ones(len_of_someone) * w
    if not hasattr(t0, '__len__'):
        t0 = np.ones(len_of_someone) * t0
    if not hasattr(rp, '__len__'):
        rp = np.ones(len_of_someone) * rp
    duration_approx = np.nanmax(
        [get_transit_duration_approx(per=per[index], a=a[index], cos_i=np.cos(np.radians(inc[index])), ecc=ecc[index], w=w[index]) for index
         in range(len_of_someone)])
    if duration_approx < dt:
        import warnings
        warnings.warn('Transit duration approx is smaller than time resolution dt')
    duration_approx = max(duration_approx, 3 * dt)

    time = tm.time(t_tot=2 * duration_approx, dt=dt)
    depths_maxs = []
    depth_factors = []
    for index, u_pair in enumerate(u_pairs):
        if not isinstance(u_pair, float):  # this would mean it is nan
            if not np.isnan(u_pair).any():
                u1, u2 = u_pair
                flux_model = 1. - tm.flux_par(t=time, params=None, per=per[index], rp=rp[index], a=a[index], inc=inc[index], ecc=ecc[index],
                                              w=w[index], u1=u1, u2=u2,
                                              t0=t0[index], supersample_factor=20, exp_time=dt)
                depths_maxs.append(np.nanmax(flux_model))
                depth_factors.append(np.nanmax(flux_model) / rp[index] ** 2)
            else:
                depths_maxs.append(np.nan)
                depth_factors.append(np.nan)
        else:
            depths_maxs.append(np.nan)
            depth_factors.append(np.nan)
    depth_factors = np.array(depth_factors)
    if not was_array:
        depth_factors = depth_factors[0]
    return depth_factors


# ----------------------------------------------------------------------------------------------------------------------
# Templete-based pre-estimation

def _downsample_by_nanmean(x, f):
    """f must be integer factor"""
    assert f == int(f), "Downsampling factor must be integer"
    f = int(f)
    n = (x.size // f) * f
    return np.nanmean(x[:n].reshape(-1, f), axis=1)


# Old naive way to easimate radii
def find_best_template_in_bank_and_rp_rs(flux, vars_flux, mask_flux, transit_quarters, mean_flux,
                                         indices_of_templates, whitened_templates_td, templates_shifts,
                                         f_flux=None, upsampling_factor_flux_wrt_templates=1):
    """Old naive way to easimate radii used in the pipeline before. Does not take into account
    that depth is not just proportional to rp_rs^2"""
    if upsampling_factor_flux_wrt_templates != 1:
        flux = _downsample_by_nanmean(flux, upsampling_factor_flux_wrt_templates)
        vars_flux = _downsample_by_nanmean(vars_flux, upsampling_factor_flux_wrt_templates)
        mask_flux = _downsample_by_nanmean(mask_flux.astype(float), upsampling_factor_flux_wrt_templates)
        mask_flux = np.isclose(mask_flux, 1)
        f_flux = None  # need to recompute!
    if f_flux is None:
        f_flux = np.fft.rfft(flux)
    l1 = (len(flux) + 1) // 2
    l2 = len(flux) // 2

    rp_rs_est_for_temps = []
    log_liks_for_temps = []
    for temp_index in range(len(indices_of_templates)):
        rp_rs_est_for_shifts = []
        log_liks_for_shifts = []
        for k in range(indices_of_templates[temp_index][0], indices_of_templates[temp_index][1]):
            temp_from_bank_for_averaging = [np.array(whitened_templates_td[k][q]) for q in transit_quarters if not np.isnan(q)]
            # This is a sloppy way to find the average template to fit to the folded flux, but okay:
            temp_from_bank = np.mean(temp_from_bank_for_averaging, axis=0)
            temp_from_bank = np.concatenate((temp_from_bank[:l1], temp_from_bank[-l2:]))
            f_temp_from_bank = np.fft.rfft(temp_from_bank)

            rp_rs_ref_from_templ_norm = (np.max(np.abs(templates_shifts[k])) / mean_flux) ** 0.5
            rp_rs_est, ampl, ind_fit = fit_amplitude(f_flux_wh=f_flux, f_flux_model_padded_wh=f_temp_from_bank,
                                                     len_flux=len(flux), rp_rs_model=rp_rs_ref_from_templ_norm, rp_rs_max=None)
            mask_flux[np.isclose(vars_flux, 0)] = False
            log_lik = get_log_likelihood(flux=flux, vars_flux=vars_flux, mask_flux=mask_flux,
                                         model_flux=ampl * np.roll(temp_from_bank, ind_fit))
            rp_rs_est_for_shifts.append(rp_rs_est)
            log_liks_for_shifts.append(log_lik)

        index_shifts_best = np.nanargmax(log_liks_for_shifts)
        rp_rs_est = rp_rs_est_for_shifts[index_shifts_best]
        log_lik = log_liks_for_shifts[index_shifts_best]
        rp_rs_est_for_temps.append(rp_rs_est)
        log_liks_for_temps.append(log_lik)
    best_template_index = np.nanargmax(log_liks_for_temps)
    best_rp_rs = rp_rs_est_for_temps[best_template_index]

    return best_rp_rs, best_template_index, rp_rs_est_for_temps, log_liks_for_temps


# ----------------------------------------------------------------------------------------------------------------------

def make_json_safe(obj):
    """Recursively convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.datetime64,)):
        return str(obj)
    elif isinstance(obj, (complex)):
        return str(obj)
    elif isinstance(obj, (np.complex128, np.complex64, complex)):
        return str(obj)
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)


def save_as_json(dictionary, saving_dir, name, keys_to_save=None, keys_to_exclude_includes='fourier'):
    if keys_to_save is not None:
        keys_to_save = [key for key in keys_to_save if (key in dictionary.keys())]
    else:
        keys_to_save = list(dictionary.keys())
    if keys_to_exclude_includes is not None:
        if isinstance(keys_to_exclude_includes, str):
            keys_to_save = [key for key in keys_to_save if keys_to_exclude_includes not in key]
        else:
            for exclude_key in keys_to_exclude_includes:
                keys_to_save = [key for key in keys_to_save if exclude_key not in key]
    dictionary = {key: dictionary[key] for key in keys_to_save}
    dict_new = make_json_safe(obj=dictionary)

    saving_file = os.path.join(saving_dir, name)
    with open(saving_file, 'w') as f:
        json.dump(dict_new, f, indent=4)


# ----------------------------------------------------------------------------------------------------------------------

def weighted_percentile(values, weights, percentile, mask=None):
    """
    Compute a weighted percentile, ignoring NaNs, with an optional mask.

    Parameters
    ----------
    values : array_like
        1D array of data values.
    weights : array_like
        1D array of non-negative weights (probability weights).
    percentile : float
        Desired percentile in [0, 100].
    mask : array_like of bool, optional
        Boolean mask: True = keep the element, False = ignore.

    Returns
    -------
    float
        Weighted percentile value.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        values = values[mask]
        weights = weights[mask]

    # Remove NaNs in both values and weights
    valid = ~(np.isnan(values) | np.isnan(weights))
    v = values[valid]
    w = weights[valid]

    if v.size == 0:
        return np.nan

    # Sort by values
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]

    # Cumulative normalized weights
    cw = np.cumsum(w)
    cw /= cw[-1]

    # Percentile position
    p = percentile / 100.0

    # Index where cumulative weight crosses percentile
    i = np.searchsorted(cw, p)

    return v[min(i, len(v) - 1)]


def _overlap_fraction_1d(fine_edges, coarse_edges):
    """
    Compute overlap fractions between fine bins and coarse bins in 1D.

    Returns
    -------
    frac : ndarray, shape (n_fine, n_coarse)
        frac[i, k] = overlap_length(fine_bin_i, coarse_bin_k) / coarse_width_k
        so that distributing coarse mass uniformly gives correct mass split.
    """
    fe = np.asarray(fine_edges, dtype=float)
    ce = np.asarray(coarse_edges, dtype=float)

    if fe.ndim != 1 or ce.ndim != 1:
        raise ValueError("Edges must be 1D arrays.")
    if fe.size < 2 or ce.size < 2:
        raise ValueError("Edges must have length >= 2.")
    if not (np.all(np.diff(fe) > 0) and np.all(np.diff(ce) > 0)):
        raise ValueError("Edges must be strictly increasing.")

    nf = fe.size - 1
    nc = ce.size - 1
    cw = ce[1:] - ce[:-1]

    frac = np.zeros((nf, nc), dtype=float)

    # For each fine bin, find intersecting coarse bins and accumulate overlap
    for i in range(nf):
        a = fe[i]
        b = fe[i + 1]
        if b <= ce[0] or a >= ce[-1]:
            continue

        # coarse bin indices that could overlap
        k0 = np.searchsorted(ce, a, side="right") - 1
        k1 = np.searchsorted(ce, b, side="left")
        k0 = max(k0, 0)
        k1 = min(k1, nc)

        for k in range(k0, k1):
            left = max(a, ce[k])
            right = min(b, ce[k + 1])
            ol = right - left
            if ol > 0:
                frac[i, k] = ol / cw[k]

    return frac


def interpolate_prior_grid(
    coarse_prior,
    coarse_param_0_edges,
    coarse_param_1_edges,
    fine_param_0_edges,
    fine_param_1_edges,
    renormalize=False,
):
    """
    Mass-conserving interpolation (rebinning) of a 2D binned prior from coarse to fine edges.

    Assumption: coarse_prior[k, l] is probability mass in coarse cell
    [coarse0[k], coarse0[k+1]) x [coarse1[l], coarse1[l+1]), and mass is uniform inside each cell.

    Parameters
    ----------
    coarse_prior : array_like, shape (n0_coarse, n1_coarse)
        Probability mass per coarse cell.
    coarse_param_0_edges, coarse_param_1_edges : array_like
        Bin edges for coarse grid in each dimension.
    fine_param_0_edges, fine_param_1_edges : array_like
        Bin edges for fine grid in each dimension.
    renormalize : bool, default False
        If True, renormalize fine grid to sum to coarse grid sum (useful if fine extends outside coarse).

    Returns
    -------
    fine_prior : ndarray, shape (n0_fine, n1_fine)
        Probability mass per fine cell. Total mass is preserved within the overlapping region.
    """
    cp = np.asarray(coarse_prior, dtype=float)
    c0 = np.asarray(coarse_param_0_edges, dtype=float)
    c1 = np.asarray(coarse_param_1_edges, dtype=float)
    f0 = np.asarray(fine_param_0_edges, dtype=float)
    f1 = np.asarray(fine_param_1_edges, dtype=float)

    n0c = c0.size - 1
    n1c = c1.size - 1
    if cp.shape != (n0c, n1c):
        raise ValueError(
            f"coarse_prior shape {cp.shape} must match "
            f"(len(coarse_param_0_edges)-1, len(coarse_param_1_edges)-1)=({n0c},{n1c})."
        )
    if np.any(cp < 0):
        raise ValueError("coarse_prior must be non-negative (probability masses).")

    frac0 = _overlap_fraction_1d(f0, c0)  # (n0f, n0c): overlap / coarse_width0
    frac1 = _overlap_fraction_1d(f1, c1)  # (n1f, n1c): overlap / coarse_width1

    # Mass split: fine[i,j] = sum_{k,l} coarse[k,l] * frac0[i,k] * frac1[j,l]
    fine = frac0 @ cp @ frac1.T

    if renormalize:
        coarse_sum = np.sum(cp)
        fine_sum = np.sum(fine)
        if fine_sum > 0:
            fine *= (coarse_sum / fine_sum)

    return fine


def hill_around_global_max_1d(data, ind_max=None, frac=0.01, add_extra=0):
    """
    Return a boolean mask for the contiguous 1D region ("hill") around the global
    maximum with y >= frac * max(y).

    Parameters
    ----------
    data : array_like
        1D array of values. NaNs are ignored.
    frac : float, default 0.99
        Threshold as fraction of the global maximum.

    Returns
    -------
    mask : ndarray of bool, shape y.shape
        True for indices in the hill.
    ind_max : int or None
        Index of a global maximum (first occurrence).
    thr : float
        Absolute threshold used (frac * max(y)).
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("y must be a 1D array.")

    # Handle all-NaN or empty-valid case
    if not np.isfinite(data).any():
        return np.zeros_like(data, dtype=bool), None, np.nan

    # Global max ignoring NaNs
    if ind_max is None:
        ind_max = int(np.nanargmax(data))
    data_max = data[ind_max]
    thr = frac * data_max

    # Eligible points
    eligible = np.isfinite(data) & (data >= thr)

    # Expand left
    left = ind_max
    while left - 1 >= 0 and eligible[left - 1]:
        left -= 1
    if add_extra > 0:
        left = max(0, left - add_extra)

    # Expand right
    right = ind_max
    n = data.size
    while right + 1 < n and eligible[right + 1]:
        right += 1
    if add_extra > 0:
        right = min(n - 1, right + add_extra)

    mask = np.zeros_like(data, dtype=bool)
    mask[left:right + 1] = True

    return mask, ind_max, thr


def nn_map_1d(fine_x, coarse_x):
    """
    Map each fine_x to the index of the nearest coarse_x (coarse_x must be sorted).

    Parameters
    ----------
    fine_x : array_like
        Fine coordinates (any shape).
    coarse_x : array_like
        1D sorted coarse coordinates (centers).

    Returns
    -------
    idx : ndarray of int
        Indices into coarse_x for the nearest neighbor of each fine_x.
        Shape matches fine_x.
    """
    fine_x = np.asarray(fine_x)
    coarse_x = np.asarray(coarse_x)

    if coarse_x.ndim != 1:
        raise ValueError("coarse_x must be 1D.")
    if not np.all(np.diff(coarse_x) > 0):
        raise ValueError("coarse_x must be strictly increasing.")

    j = np.searchsorted(coarse_x, fine_x, side="left")
    j = np.clip(j, 1, coarse_x.size - 1)

    left = j - 1
    right = j

    choose_right = (fine_x - coarse_x[left]) > (coarse_x[right] - fine_x)
    return np.where(choose_right, right, left)


def pad_region_with_value(
    a,
    pad=2,
    val_min=0.0,
    val_to_replace=np.nan,
    include_neg_inf=False,
    connectivity=8,
):
    """
    Pad a region of a 2D array defined by a replaceable value by a given number
    of bins and fill the padded region with a constant value.

    Parameters
    ----------
    a : array_like
        2D array.
    pad : int, optional
        Number of dilation steps (bins) to pad outward.
    val_min : float, optional
        Value to assign to the padded region.
    val_to_replace : float or np.nan, optional
        Value defining the region to be replaced / padded.
        If ``np.nan``, NaNs define the replaceable region.
    include_neg_inf : bool, optional
        If True, treat ``-np.inf`` as replaceable as well.
    connectivity : {4, 8}, optional
        Neighborhood definition for dilation.

    Returns
    -------
    out : ndarray
        Array with padded region filled with ``val_min``.

    Notes
    -----
    The padding is performed by binary dilation of the *non-replaceable*
    region. The padded ring is defined as the dilated region minus the original
    non-replaceable region.

    Existing non-replaceable values are never overwritten.
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("`a` must be a 2D array.")
    if pad < 0:
        raise ValueError("`pad` must be >= 0.")

    out = a.copy()

    # Determine replaceable mask
    if np.isnan(val_to_replace):
        replaceable = np.isnan(out)
    else:
        replaceable = (out == val_to_replace)

    if include_neg_inf:
        replaceable |= np.isneginf(out)

    valid = ~replaceable

    if pad == 0:
        return out

    if connectivity == 4:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool)
    elif connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        raise ValueError("`connectivity` must be 4 or 8.")

    dilated = binary_dilation(valid, structure=structure, iterations=pad)

    # Ring = newly reached cells only
    ring = dilated & ~valid

    # Fill only cells that are replaceable
    out[ring & replaceable] = val_min

    return out


def get_approx_low_pass_snr_from_simulation_params(ecc_sim, w_sim, inc_sim, rp_rs_sim, per, a_sim, time, binwid, inds_to_zero,
                                                   len_high_pass, u, rel_wf_fd, mask_flux, vars_flux):
    (flux_wh, f_flux, vars_flux, mask_flux, a_sim, log_dur_sim, impact_par_sim, b_transf_sim,
     flux_sim_padded_wh_0) = simulate_whitened_noise_transit(
        ecc_sim=ecc_sim, w_sim=w_sim, inc_sim=inc_sim, rp_rs_sim=rp_rs_sim, per=per,
        time=time, binwid=binwid, inds_to_zero=inds_to_zero, len_high_pass=len_high_pass,
        u=u, rel_wf_fd=rel_wf_fd, mask_flux=mask_flux, vars_flux=vars_flux,
        a_sim=a_sim, stellar_mass=None, stellar_radius=None, return_injected_model=True)
    dur_approx = get_transit_duration_approx(per=per, a=a_sim, cos_i=np.cos(np.radians(inc_sim)), ecc=ecc_sim, w=w_sim)
    dur_approx_bins = int(np.ceil(dur_approx / binwid))
    no_std_half_wing = max(dur_approx_bins, 30)

    # Estimating SNR of this simulated folded transit
    flux_for_std = np.concatenate((flux_wh[:len(flux_wh)//2 - no_std_half_wing], flux_wh[len(flux_wh)//2+no_std_half_wing:]))
    sigma_approx = ut.outlier_free_std(flux_for_std)
    snr_estimated = np.sqrt(flux_sim_padded_wh_0@flux_sim_padded_wh_0) / sigma_approx
    return snr_estimated
