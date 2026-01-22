#!/usr/bin/env python3


import os

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import json
import time
import copy
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
from locations import Efficiency_output_folder, Data_location
from runfiles_parameters import Jobs_monitor_folder
import pipeline_parameters as pp
import search_pipeline as spip
import template_bank.template_module as tm
import auxiliary_functions as af
import simulations.lightcurve_simulation as sim
import utils as ut
from pe_functions import get_cos_i_max
from template_bank.template_bank_generation_parameters import E_crit
from file_operations import report_job_done

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number prefix", default="011305996")
parser.add_argument("--period", type=str, nargs=1, help="", default="200")
parser.add_argument("--job_name", type=str, nargs=1, help="", default="")
parser.add_argument('--saving_dir', type=str, nargs=1, help="", default="")
args = parser.parse_args()
target_id = args.target_id
if isinstance(target_id, list) or isinstance(target_id, tuple):
    target_id = target_id[0]
Period = args.period
if isinstance(Period, list) or isinstance(Period, tuple):
    Period = Period[0]
Period = float(Period)

job_name = args.job_name
if isinstance(job_name, list) or isinstance(job_name, tuple):
    job_name = job_name[0]
saving_dir = args.saving_dir
if isinstance(saving_dir, list) or isinstance(saving_dir, tuple):
    saving_dir = saving_dir[0]

# prefix = target_id[:4]  # Extracting the prefix from the target_id
# last_digit = target_id[-1]  # Extracting the last digit from the target_id
# efficiency_output_folder_prefix_last_digit = os.path.join(Efficiency_output_folder, prefix, last_digit)
# saving_dir = efficiency_output_folder_prefix_last_digit

Saving_json_scramble_postfix = '_scrambles.json'
Saving_json_injection_postfix = '_injections.json'

# N_scrambles_per_target = 7
N_scrambles_per_target = 3
Bright_candidate_mes_here = 7.5
Bright_candidate_ses_here = 4.0
# -> We do not want to leave also KOIs that for our pipeline are large signals, we want to probe the efficiency without them

#%%
# --------------------------------------------------------------------------------------------
# ######################################################################################################################
# SCRAMBLE EMPTY
# ######################################################################################################################


def get_all_scores(target_id,
                   period=None,
                   saving_dict_file=None,
                   data_location=Data_location,
                   output_location=None,
                   scramble=False, seed=None, n_scrambles=1,
                   save_json=True,
                   keys_saving=()):
    target_id = str(target_id).rjust(9, '0')
    targ = target_class.KeplerTarget(target_id=target_id,
                                     data_location=data_location,
                                     output_location=output_location,
                                     load_pickled_attributes=False,
                                     problems_folder=None)
    targ.initialize_kepler_light_curve()

    targ.get_connected_klc(reinitialize=False, allow_loading=False,
                           file_path_connected_data=None,
                           mask_koi_planet=True, period_avoid_masking=None, ftt_avoid_masking=None,
                           mask_candidates=True,
                           mask_false_positives=True,
                           mask_bright_candidates=True,
                           bright_candidate_mes=Bright_candidate_mes_here,
                           bright_candidate_ses=Bright_candidate_ses_here,
                           save_connected_stats=False, save_conditioned_data=False)
    targ.set_periodicity_search_parameters(force_overwrite=False)
    periods_in_chunks = targ.klc.make_periods_chunks(p_min=targ.p_min, p_max=targ.p_max, duty_cycle=targ.duty_cycle,
                                                     per_phases_per_chunk=targ.per_phases_per_chunk,
                                                     dt=targ.klc.dt_days, t_mission=targ.klc.t_mission)
    closest_chunk, _ = ut.find_closest_chunk(periods_in_chunks, period, overlap_bins=0)
    periods_to_search = closest_chunk
    targ.calculate_baseline_integral_score_rate()

    if scramble:
        np.random.seed(seed)
        n_repetitions = n_scrambles
    else:
        n_repetitions = 1

    for i_repetition in range(n_repetitions):
        if scramble:
            targ.klc.scramble_full_stats()

        targ.klc.delete_mes()
        targ.klc.periodicity_search(p_min=None, p_max=None, periods=periods_to_search,
                                    delete_full_stats=False, force_interleave_shifts=True,
                                    delete_periodic_interleaved=False,
                                    save_uncorrected_periodic=True, delete_non_marginalized=False,
                                    overlap_period_bins=spip.P_bins_overlap)

        targ.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False, save_veto_masks=True,
                                  log_baseline_integral_score_rate=targ.log_baseline_int_score_rate, minimal_trackable_score=5)
        targ.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True, save_veto_masks=True,
                                  log_baseline_integral_score_rate=targ.log_baseline_int_score_rate, minimal_trackable_score=5)
        targ.klc.make_best_periodic_scores()

        marg_score = targ.klc.best_stat
        int_score = targ.klc.best_int_score
        best_period = targ.klc.best_period
        best_ftt = targ.klc.best_ftt

        if np.isnan(targ.klc.best_template_index):
            ftt_ind = np.nan
            per_ind = np.nan
            cor_score = np.nan
            uncor_score = np.nan
            temp_int = np.nan
        else:
            if targ.klc.best_template_index in targ.klc.interleaved_numbers:
                ftt_phases = targ.klc._phases_interleaved + targ.klc.tmin
                uncor_stats = targ.klc.uncorrected_stats_periodic[0]
                cor_stats = targ.klc.statistics_periodic_interleaved
                temp_int = targ.klc.best_template_index
            else:
                ftt_phases = targ.klc._phases_not_interleaved + targ.klc.tmin
                uncor_stats = targ.klc.uncorrected_stats_periodic[1]
                cor_stats = targ.klc.statistics_periodic_not_interleaved
                temp_int = targ.klc.best_template_index - len(targ.klc.statistics_periodic_interleaved[0])

            ftt_ind = np.argmin(np.abs(ftt_phases - best_ftt))
            per_ind = np.argmin(np.abs(targ.klc.periods - best_period))

            cor_score = (cor_stats[per_ind, temp_int, ftt_ind]) ** 2
            uncor_score = (uncor_stats[per_ind, temp_int, ftt_ind]) ** 2

        # If we were selecting the best uncorrected:
        min_interl = np.nanmin(targ.klc.uncorrected_stats_periodic[0])
        min_non_interl = np.nanmin(targ.klc.uncorrected_stats_periodic[1])
        if min_interl < min_non_interl:
            ftt_phases_uncor = targ.klc._phases_interleaved + targ.klc.tmin
            uncor_stats_uncor = targ.klc.uncorrected_stats_periodic[0]
            cor_stats_uncor = targ.klc.statistics_periodic_interleaved
        else:
            ftt_phases_uncor = targ.klc._phases_not_interleaved + targ.klc.tmin
            uncor_stats_uncor = targ.klc.uncorrected_stats_periodic[1]
            cor_stats_uncor = targ.klc.statistics_periodic_not_interleaved

        per_ind_uncor = np.nanargmin(np.nanmin(uncor_stats_uncor[:, :, :], axis=(1, 2)))
        temp_int_ind_uncor = np.nanargmin(np.nanmin(uncor_stats_uncor[per_ind_uncor, :, :], axis=-1))
        phas_ind_uncor = np.nanargmin(uncor_stats_uncor[per_ind_uncor, temp_int_ind_uncor, :])
        ftt_uncor = ftt_phases_uncor[phas_ind_uncor]
        cor_score_uncor = (cor_stats_uncor[per_ind_uncor, temp_int_ind_uncor, phas_ind_uncor]) ** 2
        uncor_score_uncor = (uncor_stats_uncor[per_ind_uncor, temp_int_ind_uncor, phas_ind_uncor]) ** 2
        phas_ind_marg_uncor = np.nanargmin(np.abs(targ.klc._phases_not_interleaved + targ.klc.tmin - ftt_uncor))
        marg_score_uncor = targ.klc.stats_periodic_mar_temp[per_ind_uncor, phas_ind_marg_uncor]
        best_period_uncor = targ.klc.periods[per_ind_uncor]
        best_ftt_uncor = ftt_uncor
        # --------------------------------------------------------------------------------------------
        # If we were selecting the best corrected:
        min_interl = np.nanmin(targ.klc.statistics_periodic_interleaved)
        min_non_interl = np.nanmin(targ.klc.statistics_periodic_not_interleaved)
        if min_interl < min_non_interl:
            ftt_phases_cor = targ.klc._phases_interleaved + targ.klc.tmin
            uncor_stats_cor = targ.klc.uncorrected_stats_periodic[0]
            cor_stats_cor = targ.klc.statistics_periodic_interleaved
        else:
            ftt_phases_cor = targ.klc._phases_not_interleaved + targ.klc.tmin
            uncor_stats_cor = targ.klc.uncorrected_stats_periodic[1]
            cor_stats_cor = targ.klc.statistics_periodic_not_interleaved

        per_ind_cor = np.nanargmin(np.nanmin(cor_stats_cor[:, :, :], axis=(1, 2)))
        temp_int_ind_cor = np.nanargmin(np.nanmin(cor_stats_cor[per_ind_cor, :, :], axis=-1))
        phas_ind_cor = np.nanargmin(cor_stats_cor[per_ind_cor, temp_int_ind_cor, :])
        ftt_cor = ftt_phases_cor[phas_ind_cor]
        cor_score_cor = (cor_stats_cor[per_ind_cor, temp_int_ind_cor, phas_ind_cor]) ** 2
        uncor_score_cor = (uncor_stats_cor[per_ind_cor, temp_int_ind_cor, phas_ind_cor]) ** 2
        phas_ind_marg_cor = np.nanargmin(np.abs(targ.klc._phases_not_interleaved + targ.klc.tmin - ftt_cor))
        marg_score_cor = targ.klc.stats_periodic_mar_temp[per_ind_cor, phas_ind_marg_cor]
        best_period_cor = targ.klc.periods[per_ind_cor]
        best_ftt_cor = ftt_cor

        # --------------------------------------------------------------------------------------------
        if save_json:
            if i_repetition == 0:
                dict_save = {}
                for key in keys_saving:
                    dict_save[key] = []
            else:
                with open(saving_dict_file, "r") as file:
                    dict_save = json.load(file)

            for key in keys_saving:
                val = locals()[key]
                if not isinstance(val, str):
                    val = float(val)
                dict_save[key].append(val)

            # Dict saving
            with open(saving_dict_file, "w") as file:
                pass

            tmp = saving_dict_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(dict_save, f, indent=4)
            os.replace(tmp, saving_dict_file)

            # with open(saving_dict_file, "w") as file:
            #     json.dump(dict_save, file, indent=4)  # 'indent=4' makes it pretty-printed
        # --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------
        # res = [target_id, period, uncor_score, cor_score, marg_score, int_score, best_period, best_ftt,
        #        cor_score_uncor, uncor_score_uncor, phas_ind_marg_uncor, marg_score_uncor, best_period_uncor, best_ftt_uncor,
        #        cor_score_cor, uncor_score_cor, phas_ind_marg_cor, marg_score_cor, best_period_cor, best_ftt_cor]
    del targ


def scrambled_run_wrapper(target_id, period, saving_folder_json, ):
    seed = int(target_id) + int(time.time()*1000000) % (2**25)

    if not os.path.isdir(saving_folder_json):
        os.mkdir(saving_folder_json, mode=0o777)

    saving_file_json = os.path.join(saving_folder_json, target_id + Saving_json_scramble_postfix)

    # # INITIALIZING THE DICT
    keys = ['target_id', 'period', 'uncor_score', 'cor_score', 'marg_score', 'int_score', 'best_period', 'best_ftt',
            'cor_score_uncor', 'uncor_score_uncor', 'phas_ind_marg_uncor', 'marg_score_uncor', 'best_period_uncor', 'best_ftt_uncor',
            'cor_score_cor', 'uncor_score_cor', 'phas_ind_marg_cor', 'marg_score_cor', 'best_period_cor', 'best_ftt_cor']

    get_all_scores(target_id=target_id,
                   period=period,
                   saving_dict_file=saving_file_json,
                   data_location=Data_location,
                   output_location=None,
                   scramble=True, seed=seed, n_scrambles=N_scrambles_per_target,
                   save_json=True,
                   keys_saving=keys)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    scrambled_run_wrapper(target_id=target_id, period=Period, saving_folder_json=saving_dir)


#%%

# ######################################################################################################################
# INJECTIONS
# ######################################################################################################################


# --------------------------------------------------------------------------------------------
snr2_approx_injected_array = np.array([60, 70, 80, 100])
snr_min_injected_array = snr2_approx_injected_array ** 0.5
snr_max_1_injected = 7.5  # I do not want to scale these with period, let it be uniform. 6.5 was the outlier cut, so let it be this
snr_max_full_injected_array = (snr2_approx_injected_array + 5) ** 0.5
N_injections_per_snrs = [5, 5, 5, 1]
# --------------------------------------------------------------------------------------------


def get_rp_min_max(targ, period_inj, snr_max_1_injected,
                   snr_max_full_injected, snr_min_injected, data_location=Data_location, output_location=None):
    # --------------------------------------------------------------------------------------------
    # GETTING THE PSD
    target_id = targ.target_id
    klc = spip.KeplerLightCurve(target_id=target_id,
                                data_location=data_location,
                                output_location=output_location,
                                allow_data_folder_creation=False, allow_output_folder_creation=False)

    nsig_outliers_0 = spip.Nsig_outliers_0
    nsig_outliers_0_pos = spip.Nsig_outliers_0_pos
    nsig_outliers_0_neg = spip.Nsig_outliers_0_neg
    nsig_outliers_psd = spip.Nsig_outliers_psd
    nperseg_psd = spip.Nperseg_psd
    npsdsmin = spip.Npsdsmin
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
    klc.measure_psds(nperseg=klc.nperseg_psd, tukey_steps_len=tukey_steps_len)

    # Getting parameters needed for estimation:
    n_quarters = len(klc.quarters)
    wf_fds = np.array(klc.wf_fds) * klc.mean_total  # -> relative wf_fd, take mean flux into account
    wf_fd = np.mean(np.array(np.abs(wf_fds)) ** 2, axis=0) ** 0.5

    # --------------------------------------------------------------------------------------------
    # Getting flux
    inc_inj = 90.
    ecc_inj = 0.
    rp_generation = 0.02
    w_inj = 90.
    u = targ.limb_darkening_coefs
    if u is None:
        u = pp.Default_limb_darkening_coefs

    a = pp.get_a_from_p(period=period_inj, m_1_3_r_s=targ.m_1_3_r_s)

    t = tm.time(t_tot=(256 + 1) * tm.dt_lc, parity='odd')[1:]  # Making an even points number but odd parity
    flux_obs = 1. - tm.flux_par(t=t, params=None, per=period_inj, rp=rp_generation, a=a, inc=inc_inj, ecc=ecc_inj, w=w_inj, u1=u[0],
                                u2=u[1], t0=0, supersample_factor=20, exp_time=tm.dt_lc)

    # dur = tm.templ_dur_fun(flux_obs) * tm.dt_lc
    depth_factor = np.nanmax(flux_obs) / rp_generation ** 2
    rp_rs_min, rp_rs_max = targ.get_detectable_radius_flux(flux_model=flux_obs, period=period_inj,
                                                           depth_factor=depth_factor, snr_min=snr_min_injected,
                                                           n_quarters=n_quarters, wf_fd=wf_fd,
                                                           snr_max_1=snr_max_1_injected,
                                                           snr_max_full=snr_max_full_injected,
                                                           stellar_radius=targ.stellar_radius, units='star',
                                                           include_detrending=targ, nperseg_psd=spip.Nperseg_psd)
    del klc
    return rp_rs_min, rp_rs_max


def scramble_fluxes_by_quarters(fluxes, times=None, seed=np.nan):
    if ~np.isnan(seed):
        np.random.seed(int(seed))
    for q in range(len(fluxes)):
        mask = ~np.isnan(fluxes[q])
        big_mask = ut.clean_small_holes(np.array(mask))
        big_mask = np.array(~ut.clean_small_holes(np.array(~big_mask), max_hole=1))

        inds_pieces_scramble = ut.get_pieces_indices(mask=big_mask)
        # Warning: if there is a hole in the beginning or the end, it may (?) be treated incorrectly and the
        # length of the scrambled data will be effectively bigger
        tot_hole_len_scramble = len(fluxes[q]) - np.sum([ip[1] - ip[0] for ip in inds_pieces_scramble])
        np.random.shuffle(inds_pieces_scramble)

        if len(inds_pieces_scramble) > 1:
            len_holes = ut.gen_uniform_piece_lengths(tot_len=tot_hole_len_scramble, n_pieces=len(inds_pieces_scramble) - 1, seed=seed)
            # big_mask, inds_pieces_in_scrambled = ut.rearrange_pieces(
            #     big_mask, inds_pieces=inds_pieces_scramble, hole_lens=len_holes, fill_hole_with=False, return_inds_target_res=True)
            fluxes[q] = ut.rearrange_pieces(fluxes[q], inds_pieces=inds_pieces_scramble, hole_lens=len_holes, fill_hole_with=np.nan)
        if np.random.choice([True, False]):
            fluxes[q] = fluxes[q][::-1]
    indices_quarters = np.arange(len(fluxes))
    np.random.shuffle(indices_quarters)
    fluxes = [fluxes[iq] for iq in indices_quarters]

    if times is not None:
        times_interp = [ut.interpolate_time(np.array(times[q]), mask=None) for q in range(len(times))]
        time_min = np.nanmin([np.nanmin(times_interp[q]) for q in range(len(times_interp))])
        time_max = np.nanmax([np.nanmax(times_interp[q]) for q in range(len(times_interp))])
        dt = ut.get_average_dt(times=times)
        occupied_time_bins = np.sum([len(times[q]) for q in range(len(times))])
        total_time_bins = (time_max - time_min) / dt
        total_hole_len_bins = int(total_time_bins - occupied_time_bins)
        len_holes_between_quarters = ut.gen_uniform_piece_lengths(tot_len=total_hole_len_bins, n_pieces=len(times) - 1, seed=seed)
        times_new = []
        current_time = time_min
        for q in range(len(fluxes)):
            time_array_new = np.arange(len(fluxes[q]), dtype=float) * dt + current_time
            times_new.append(time_array_new)
            current_time = time_array_new[-1] + dt
            if q < len(fluxes) - 1:
                current_time = current_time + len_holes_between_quarters[q] * dt
        return fluxes, times_new
    else:
        return fluxes


# --------------------------------------------------------------------------------------------
def get_all_scores_inj(target_id,
                       per_inj=None,
                       snr_max_1_injected=snr_max_1_injected,
                       snr_min_injected_array=snr_min_injected_array,
                       n_injections_per_snrs=tuple(N_injections_per_snrs),
                       saving_dict_file=None,
                       data_location=Data_location,
                       output_location=None,
                       scramble=True, seed=None,
                       save_json=True,
                       keys_saving=()):
    target_id = str(target_id).rjust(9, '0')
    targ = target_class.KeplerTarget(target_id=target_id,
                                     data_location=data_location,
                                     output_location=output_location,
                                     load_pickled_attributes=False,
                                     problems_folder=None)
    targ_rp_estimation = copy.deepcopy(targ)  # -> will use this later

    # Will take times, fluxes, masks from here:
    targ.klc = spip.KeplerLightCurve(target_id=targ.target_id, get_data=True,
                                     data_location=targ.data_location, m_1_3_r_s=targ.m_1_3_r_s,
                                     output_location=targ.output_location)
    # If there is a need to mask koi (checking it here), we mask it:
    if targ.is_koi:  # Masking koi planets is only possible if we did not scramble! So we selected stars not to be kois
        targ.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False,
                             mask_candidates=True,
                             mask_false_positives=True,
                             mask_bright_candidates=True,
                             bright_candidate_mes=Bright_candidate_mes_here,
                             bright_candidate_ses=Bright_candidate_ses_here,
                             avoid_period=None,
                             avoid_time_first=None, dp_avoid=0.002, dt_avoid=0.04)
    tms_0 = copy.deepcopy(targ.klc.times)
    flxs_0 = copy.deepcopy(targ.klc.fluxes)
    for q in range(len(flxs_0)):
        flxs_0[q][~targ.klc.masks[q]] = np.nan  # Applying masks

    per_inj_0 = per_inj
    for i_snr, snr2_approx_injected in enumerate(snr2_approx_injected_array):
        n_injections = n_injections_per_snrs[i_snr]
        snr_min_injected = snr_min_injected_array[i_snr]
        snr_max_full_injected = snr_max_full_injected_array[i_snr]

        rp_min, rp_max = get_rp_min_max(targ=targ_rp_estimation, period_inj=per_inj, snr_max_1_injected=snr_max_1_injected,
                                        snr_max_full_injected=snr_max_full_injected, snr_min_injected=snr_min_injected,
                                        data_location=Data_location, output_location=None)
        rp_generation = 0.02
        u = targ.limb_darkening_coefs
        if u is None:
            u = pp.Default_limb_darkening_coefs

        rp_inj = rp_min
        for i_injection in range(n_injections):
            # Making a bit of freedom for the period to make sure there is no template sampling bias
            per_inj = per_inj_0 + np.random.rand() * 1.
            a_inj = pp.get_a_from_p(period=per_inj, m_1_3_r_s=targ.m_1_3_r_s)
            t = tm.time(t_tot=(256 + 1) * tm.dt_lc, parity='odd')[1:]  # Making an even points number but odd parity

            norm_flux = 0
            inc_inj = None
            ecc_inj = None
            w_inj = None
            while np.isclose(norm_flux, 0):
                e_max = min(1 - 2 / a_inj, E_crit)
                ecc_inj = ut.get_uniform_random_sequence(0, e_max, n=1)[0]
                w_inj = ut.get_uniform_random_sequence(0, 360, n=1)[0]
                cos_i_max = get_cos_i_max(a=a_inj, ecc=ecc_inj, w=w_inj, overshoot_factor=1.)
                inc_inj = np.arccos(ut.get_uniform_random_sequence(0, cos_i_max, n=1)[0]) * 180. / np.pi

                flux_obs = 1. - tm.flux_par(t=t, params=None, per=per_inj, rp=rp_generation, a=a_inj, inc=inc_inj, ecc=ecc_inj, w=w_inj,
                                            u1=u[0],
                                            u2=u[1], t0=0, supersample_factor=20, exp_time=tm.dt_lc)
                norm_flux = np.linalg.norm(flux_obs)

            seed = int(per_inj * a_inj * rp_inj * ecc_inj * w_inj) % (2 ** 32)
            phase_inj = np.random.uniform(0, float(per_inj))

            # tms, flxs = af.get_data_fun(target_id=targ.target_id, location=targ.data_location)
            tms = copy.deepcopy(tms_0)
            flxs = copy.deepcopy(flxs_0)

            # Warning! We scramble to destroy all possible signal from before (KOI for example)
            if scramble:
                flxs, tms = scramble_fluxes_by_quarters(fluxes=flxs, times=tms, seed=seed)

            first_transit_time_inj = np.nanmin(tms[0]) + phase_inj

            flxs_inj = sim.inject_planet_to_data(times=tms, fluxes=flxs, period=per_inj, first_transit_center=first_transit_time_inj,
                                                 rp=rp_inj, a=a_inj, inc=inc_inj, ecc=ecc_inj, w=w_inj, u1=u[0], u2=u[1],
                                                 supersample_factor=20, return_template=False)

            targ.klc = spip.KeplerLightCurve(target_id=targ.target_id, times=tms, fluxes=flxs_inj, get_data=False,
                                             data_location=targ.data_location, m_1_3_r_s=targ.m_1_3_r_s,
                                             output_location=targ.output_location)

            # masked_dur = 1.5 * tm.Durations[-1]
            # targ.klc.mask_ephemeris(period=targ.best_per, durations=masked_dur, transit_times=targ.best_ftt)
            # # Mask the suspected real transit. Warning, can kill some of injected signal
            targ.klc.condition_data()
            targ.klc.condition_templates()
            targ.klc.detect_single_transits(apply_veto=True, gaussianize=True)
            targ.klc.connect_quarters()
            targ.klc.delete_irrelevant()  # For memory
            targ.klc.interleave_shifts(delete_fulls=True)

            targ.set_periodicity_search_parameters(force_overwrite=False)
            periods_in_chunks = targ.klc.make_periods_chunks(p_min=targ.p_min, p_max=targ.p_max, duty_cycle=targ.duty_cycle,
                                                             per_phases_per_chunk=targ.per_phases_per_chunk,
                                                             dt=targ.klc.dt_days, t_mission=targ.klc.t_mission)
            closest_chunk, _ = ut.find_closest_chunk(periods_in_chunks, per_inj, overlap_bins=0)
            periods_to_search = closest_chunk
            targ.calculate_baseline_integral_score_rate()

            targ.klc.delete_mes()
            targ.klc.periodicity_search(p_min=None, p_max=None, periods=periods_to_search,
                                        delete_full_stats=False, force_interleave_shifts=True,
                                        delete_periodic_interleaved=False,
                                        save_uncorrected_periodic=True, delete_non_marginalized=False,
                                        overlap_period_bins=spip.P_bins_overlap)

            targ.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=False, save_veto_masks=True,
                                      log_baseline_integral_score_rate=targ.log_baseline_int_score_rate, minimal_trackable_score=5)
            targ.klc.get_best_trigger(apply_veto_mes=True, in_gaussianized=True, save_veto_masks=True,
                                      log_baseline_integral_score_rate=targ.log_baseline_int_score_rate, minimal_trackable_score=5)
            targ.klc.make_best_periodic_scores()

            marg_score = targ.klc.best_stat
            int_score = targ.klc.best_int_score
            best_period = targ.klc.best_period
            best_ftt = targ.klc.best_ftt

            if np.isnan(targ.klc.best_template_index):
                ftt_ind = np.nan
                per_ind = np.nan
                cor_score = np.nan
                uncor_score = np.nan
                temp_int = np.nan
            else:
                if targ.klc.best_template_index in targ.klc.interleaved_numbers:
                    ftt_phases = targ.klc._phases_interleaved + targ.klc.tmin
                    uncor_stats = targ.klc.uncorrected_stats_periodic[0]
                    cor_stats = targ.klc.statistics_periodic_interleaved
                    temp_int = targ.klc.best_template_index
                else:
                    ftt_phases = targ.klc._phases_not_interleaved + targ.klc.tmin
                    uncor_stats = targ.klc.uncorrected_stats_periodic[1]
                    cor_stats = targ.klc.statistics_periodic_not_interleaved
                    temp_int = targ.klc.best_template_index - len(targ.klc.statistics_periodic_interleaved[0])

                ftt_ind = np.argmin(np.abs(ftt_phases - best_ftt))
                per_ind = np.argmin(np.abs(targ.klc.periods - best_period))

                cor_score = (cor_stats[per_ind, temp_int, ftt_ind]) ** 2
                uncor_score = (uncor_stats[per_ind, temp_int, ftt_ind]) ** 2

            # If we were selecting the best uncorrected:
            min_interl = np.nanmin(targ.klc.uncorrected_stats_periodic[0])
            min_non_interl = np.nanmin(targ.klc.uncorrected_stats_periodic[1])
            if min_interl < min_non_interl:
                ftt_phases_uncor = targ.klc._phases_interleaved + targ.klc.tmin
                uncor_stats_uncor = targ.klc.uncorrected_stats_periodic[0]
                cor_stats_uncor = targ.klc.statistics_periodic_interleaved
            else:
                ftt_phases_uncor = targ.klc._phases_not_interleaved + targ.klc.tmin
                uncor_stats_uncor = targ.klc.uncorrected_stats_periodic[1]
                cor_stats_uncor = targ.klc.statistics_periodic_not_interleaved

            per_ind_uncor = np.nanargmin(np.nanmin(uncor_stats_uncor[:, :, :], axis=(1, 2)))
            temp_int_ind_uncor = np.nanargmin(np.nanmin(uncor_stats_uncor[per_ind_uncor, :, :], axis=-1))
            phas_ind_uncor = np.nanargmin(uncor_stats_uncor[per_ind_uncor, temp_int_ind_uncor, :])
            ftt_uncor = ftt_phases_uncor[phas_ind_uncor]
            cor_score_uncor = (cor_stats_uncor[per_ind_uncor, temp_int_ind_uncor, phas_ind_uncor]) ** 2
            uncor_score_uncor = (uncor_stats_uncor[per_ind_uncor, temp_int_ind_uncor, phas_ind_uncor]) ** 2
            phas_ind_marg_uncor = np.nanargmin(np.abs(targ.klc._phases_not_interleaved + targ.klc.tmin - ftt_uncor))
            marg_score_uncor = targ.klc.stats_periodic_mar_temp[per_ind_uncor, phas_ind_marg_uncor]
            best_period_uncor = targ.klc.periods[per_ind_uncor]
            best_ftt_uncor = ftt_uncor
            # --------------------------------------------------------------------------------------------
            # If we were selecting the best corrected:
            min_interl = np.nanmin(targ.klc.statistics_periodic_interleaved)
            min_non_interl = np.nanmin(targ.klc.statistics_periodic_not_interleaved)
            if min_interl < min_non_interl:
                ftt_phases_cor = targ.klc._phases_interleaved + targ.klc.tmin
                uncor_stats_cor = targ.klc.uncorrected_stats_periodic[0]
                cor_stats_cor = targ.klc.statistics_periodic_interleaved
            else:
                ftt_phases_cor = targ.klc._phases_not_interleaved + targ.klc.tmin
                uncor_stats_cor = targ.klc.uncorrected_stats_periodic[1]
                cor_stats_cor = targ.klc.statistics_periodic_not_interleaved

            per_ind_cor = np.nanargmin(np.nanmin(cor_stats_cor[:, :, :], axis=(1, 2)))
            temp_int_ind_cor = np.nanargmin(np.nanmin(cor_stats_cor[per_ind_cor, :, :], axis=-1))
            phas_ind_cor = np.nanargmin(cor_stats_cor[per_ind_cor, temp_int_ind_cor, :])
            ftt_cor = ftt_phases_cor[phas_ind_cor]
            cor_score_cor = (cor_stats_cor[per_ind_cor, temp_int_ind_cor, phas_ind_cor]) ** 2
            uncor_score_cor = (uncor_stats_cor[per_ind_cor, temp_int_ind_cor, phas_ind_cor]) ** 2
            phas_ind_marg_cor = np.nanargmin(np.abs(targ.klc._phases_not_interleaved + targ.klc.tmin - ftt_cor))
            marg_score_cor = targ.klc.stats_periodic_mar_temp[per_ind_cor, phas_ind_marg_cor]
            best_period_cor = targ.klc.periods[per_ind_cor]
            best_ftt_cor = ftt_cor

            # --------------------------------------------------------------------------------------------
            if save_json:
                if i_injection == 0 and i_snr == 0:
                    dict_results = {}
                    for key in keys_saving:
                        dict_results[key] = []
                else:
                    # Old dict retrieval
                    with open(saving_dict_file, "r") as file:
                        dict_results = json.load(file)

                for key in keys_saving:
                    val = locals()[key]
                    if not isinstance(val, str):
                        val = float(val)
                    dict_results[key].append(val)

                # Dict saving
                tmp = saving_dict_file + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(dict_results, f, indent=4)
                os.replace(tmp, saving_dict_file)

                # with open(saving_dict_file, "w") as file:
                #     json.dump(dict_results, file, indent=4)  # 'indent=4' makes it pretty-printed
            # --------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------

            # res = [target_id, per_inj, first_transit_time_inj, rp_inj, uncor_score, cor_score, marg_score, int_score, best_period, best_ftt,
            #        cor_score_uncor, uncor_score_uncor, phas_ind_marg_uncor, marg_score_uncor, best_period_uncor, best_ftt_uncor,
            #        cor_score_cor, uncor_score_cor, phas_ind_marg_cor, marg_score_cor, best_period_cor, best_ftt_cor]
    return None


def injection_run_wrapper(target_id, period=Period, saving_folder_json=saving_dir):
    # # KEYS FOR THE DICT
    keys = ['target_id', 'per_inj', 'a_inj', 'inc_inj', 'ecc_inj', 'w_inj', 'phase_inj', 'rp_inj',
            'snr2_approx_injected', 'first_transit_time_inj', 'seed',
            'uncor_score', 'cor_score', 'marg_score', 'int_score', 'best_period', 'best_ftt',
            'cor_score_uncor', 'uncor_score_uncor', 'phas_ind_marg_uncor', 'marg_score_uncor', 'best_period_uncor', 'best_ftt_uncor',
            'cor_score_cor', 'uncor_score_cor', 'phas_ind_marg_cor', 'marg_score_cor', 'best_period_cor', 'best_ftt_cor']
    saving_file_json = os.path.join(saving_folder_json, target_id + Saving_json_injection_postfix)
    seed = int(target_id) + int(time.time() * 1000000) % (2 ** 25) + int(period)
    get_all_scores_inj(target_id=target_id,
                       per_inj=period,
                       saving_dict_file=saving_file_json,
                       data_location=Data_location,
                       output_location=None,
                       scramble=True, seed=seed,
                       save_json=True, keys_saving=keys)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    injection_run_wrapper(target_id, period=Period)


# ######################################################################################################################
# report_job_done(job_name=job_name, jobs_monitor_folder=Jobs_monitor_folder)



# ########################################################################################
