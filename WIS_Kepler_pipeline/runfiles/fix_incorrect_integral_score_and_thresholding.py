#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import search_pipeline as spip
from pipeline_parameters import Stat_candidate_search, Threshold_crossing_lists_folder
import target_class
import template_bank.template_module as tm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prefix", type=str, nargs=1, help="KOI prefix", default="0030")
args = parser.parse_args()
Prefix = args.prefix[0]

Input_folder = '/home/labs/barakz/Collaboration-Kepler/KeplerData/'
Output_folder = '/home/labs/barakz/Collaboration-Kepler/orynai/output_run_11/'
New_threshold_crossing_list_filename = 'all_threshold_crossings.txt'
New_g_threshold_crossing_list_filename = 'all_g_threshold_crossings.txt'

new_threshold_crossing_list_file = os.path.join(Threshold_crossing_lists_folder, New_threshold_crossing_list_filename)
new_g_threshold_crossing_list_file = os.path.join(Threshold_crossing_lists_folder, New_g_threshold_crossing_list_filename)


def strip_integral_scores_from_old_occurrences(integr_score_old, m_1_3_r, period):
    # OLD PART TO UNDO:
    _, __, log_occurrences_templates_periods, m13r_bin_edges, period_edges = tm.load_volume_prior_precomputed_terms()
    m_1_3_r_bin = min(max(0, np.searchsorted(m13r_bin_edges, m_1_3_r) - 1), len(m13r_bin_edges) - 2)
    per_bin = min(max(0, np.searchsorted(period_edges, period) - 1), len(period_edges) - 2)
    log_occurrence = log_occurrences_templates_periods[m_1_3_r_bin, per_bin]
    log_occurrence = np.log(np.sum(np.exp(log_occurrence)))  # sum over templates
    # log_occurrences_templates_periods is normalized by the m_1_3_r to have the full astrophysical occurrence rate
    # in every m_1_3_r bin. units are rate! no probability normalization!
    just_integral = integr_score_old - log_occurrence
    return just_integral


for prefix in [Prefix]:
    for tid in os.listdir(os.path.join(Output_folder, prefix)):
        output_folder_tid = os.path.join(Output_folder, prefix, tid)
        fils = [ff for ff in os.listdir(output_folder_tid) if 'best_periodic_scores_all_p' in ff]
        if len(fils) > 0:
            target_1 = target_class.KeplerTarget(tid, data_location=os.path.join(Input_folder, tid[:4], tid),
                                                 output_location=output_folder_tid)
            target_1.unpickle_small_attributes()

            # ############################################################################################################
            # 1) Opening the best periodic scores file

            fil_best_periodic_scores = os.path.join(output_folder_tid, fils[0])
            best_periodic_scores = np.load(fil_best_periodic_scores, allow_pickle=True)

            ind_gauss = 0
            stats_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_stats][ind_gauss] for i in range(len(best_periodic_scores))])
            periods_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_per][ind_gauss] for i in range(len(best_periodic_scores))])
            old_integral_scores_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_best_integral][ind_gauss] for i in range(len(best_periodic_scores))])
            ftts_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_ftt][ind_gauss] for i in range(len(best_periodic_scores))])
            stats_gauss_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_stats_gauss][ind_gauss] for i in range(len(best_periodic_scores))])
            max_templs_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_template][ind_gauss] for i in range(len(best_periodic_scores))])

            ind_gauss = 1
            g_stats_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_stats][ind_gauss] for i in range(len(best_periodic_scores))])
            g_periods_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_per][ind_gauss] for i in range(len(best_periodic_scores))])
            g_old_integral_scores_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_best_integral][ind_gauss] for i in range(len(best_periodic_scores))])
            g_ftts_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_ftt][ind_gauss] for i in range(len(best_periodic_scores))])
            g_stats_gauss_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_stats_gauss][ind_gauss] for i in range(len(best_periodic_scores))])
            g_max_templs_res = np.array(
                [best_periodic_scores[i][spip.Best_per_results_index_template][ind_gauss] for i in range(len(best_periodic_scores))])

            # ############################################################################################################
            # 2) Defining the reference rate density for the smallest period and corresponding phase space measure

            target_1.log_baseline_integral_score_rate_density = np.log(tm.get_number_of_transiting_planets_density_per_dp(
                                                                    target_1.periods_in_chunks[0][0], m_1_3_r_s=target_1.m_1_3_r_s, dp=1.
                                                                    ) / target_1.periods_in_chunks[0][0])
            # -> the function gives density per dp, and dividing it by the period, we get it also for dt.
            target_1.dp_0 = target_1.periods_in_chunks[0][1] - target_1.periods_in_chunks[0][0]
            target_1.d_ftt_0 = tm.dt_lc  # Should be taken from KeplerLightCurve._phases, byt hopefully, this is correct
            target_1.log_baseline_phase_space_measure = np.log(target_1.dp_0 * target_1.d_ftt_0)
            target_1.log_baseline_integral_score_rate = (target_1.log_baseline_integral_score_rate_density
                                                         + target_1.log_baseline_phase_space_measure)
            # Calculating rates fot all the chunks
            # Density per d_period per d_first_transit_time. When multiplied by dp * dftt that was already done in the integral
            # score calculation, gives the number of transiting planets in this phase space element
            log_occurrence_densities_per_dp_dftt = np.array([np.log(tm.get_number_of_transiting_planets_density_per_dp(
                                                             period=periods_res[i], m_1_3_r_s=target_1.m_1_3_r_s, dp=1.) / periods_res[i])
                                                             for i in range(len(periods_res))])

            # ############################################################################################################
            # 3) CALCULATING NEW INTEGRAL SCORES.
            # Integral scores without rates:
            bare_integral_scores_res = np.array([strip_integral_scores_from_old_occurrences(integr_score_old=old_integral_scores_res[i],
                                                 m_1_3_r=target_1.m_1_3_r_s, period=periods_res[i])
                                                 for i in range(len(best_periodic_scores))])
            integral_scores_with_rate = bare_integral_scores_res + log_occurrence_densities_per_dp_dftt  # dp dftt are inside the integral
            integral_scores_with_rate_to_baseline = integral_scores_with_rate - target_1.log_baseline_integral_score_rate

            # Integral scores gaussianized without rates:
            g_bare_integral_scores_res = np.array([strip_integral_scores_from_old_occurrences(integr_score_old=g_old_integral_scores_res[i],
                                                   m_1_3_r=target_1.m_1_3_r_s, period=g_periods_res[i])
                                                   for i in range(len(best_periodic_scores))])
            g_integral_scores_with_rate = g_bare_integral_scores_res + log_occurrence_densities_per_dp_dftt  # dp dftt are inside integral
            g_integral_scores_with_rate_to_baseline = g_integral_scores_with_rate - target_1.log_baseline_integral_score_rate

            # ############################################################################################################
            # 4) CORRECTING THE BEST_PERIODIC_SCORES FILES
            # NOW, THE INTEGRAL SCORE SAVED THERE IS DEFINED WITH THE RATE DENSITY IN TEH INTEGRAL AND WRT THE REFERENCE
            for i in range(len(best_periodic_scores)):
                best_periodic_scores[i][spip.Best_per_results_index_best_integral][0] = integral_scores_with_rate_to_baseline[i]
                best_periodic_scores[i][spip.Best_per_results_index_best_integral][1] = g_integral_scores_with_rate_to_baseline[i]
            # re-writing the file:
            np.save(fil_best_periodic_scores, np.array(best_periodic_scores, dtype=object), allow_pickle=True)

            # ############################################################################################################
            # 5) FINDING NEWLY DEFINED THRESHOLD CROSSINGS
            # Determining TCEs among [chunks maxima in non-gaussianized sense]
            trigger_mask = integral_scores_with_rate_to_baseline >= Stat_candidate_search
            triggering_integral_scores_with_rate_to_baseline = integral_scores_with_rate_to_baseline[trigger_mask]
            triggering_stats = stats_res[trigger_mask]
            triggering_periods = periods_res[trigger_mask]
            triggering_ftts = ftts_res[trigger_mask]
            triggering_stats_gauss = stats_gauss_res[trigger_mask]
            triggering_max_templs = max_templs_res[trigger_mask]

            with open(new_threshold_crossing_list_file, 'a') as fil1:  # will save here these good targets
                for i in range(len(triggering_integral_scores_with_rate_to_baseline)):
                    triggering_integral_score_with_rate_to_baseline = triggering_integral_scores_with_rate_to_baseline[i]
                    triggering_stat = triggering_stats[i]
                    triggering_period = triggering_periods[i]
                    triggering_ftt = triggering_ftts[i]
                    triggering_max_templ = triggering_max_templs[i]
                    triggering_stat_gauss = triggering_stats_gauss[i]
                    print(tid, triggering_integral_score_with_rate_to_baseline, triggering_stat, triggering_period, triggering_ftt,
                          triggering_max_templ, triggering_stat_gauss, sep=' ', end='\n', file=fil1)

            # Determining TCEs among [chunks maxima in gaussianized sense]
            # SAME FOR GAUSS BUT A DIFFERENT ORDER OF SAVING: GAUSS STAT GOES FIRST< AND THE INTEGRAL WAS ALSO MADE ON GAUSS
            g_trigger_mask = g_integral_scores_with_rate_to_baseline >= Stat_candidate_search
            g_triggering_integral_scores_with_rate_to_baseline = g_integral_scores_with_rate_to_baseline[g_trigger_mask]
            g_triggering_stats = g_stats_res[g_trigger_mask]
            g_triggering_periods = g_periods_res[g_trigger_mask]
            g_triggering_ftts = g_ftts_res[g_trigger_mask]
            g_triggering_stats_gauss = g_stats_gauss_res[g_trigger_mask]
            g_triggering_max_templs = g_max_templs_res[g_trigger_mask]

            with open(new_g_threshold_crossing_list_file, 'a') as fil1:  # will save here these good targets
                for i in range(len(g_triggering_integral_scores_with_rate_to_baseline)):
                    g_triggering_integral_score_with_rate_to_baseline = g_triggering_integral_scores_with_rate_to_baseline[i]
                    g_triggering_stat = g_triggering_stats[i]
                    g_triggering_period = g_triggering_periods[i]
                    g_triggering_ftt = g_triggering_ftts[i]
                    g_triggering_max_templ = g_triggering_max_templs[i]
                    g_triggering_stat_gauss = g_triggering_stats_gauss[i]
                    print(tid, g_triggering_integral_score_with_rate_to_baseline, g_triggering_stat_gauss, g_triggering_period,
                          g_triggering_ftt, g_triggering_max_templ, g_triggering_stat, sep=' ', end='\n', file=fil1)

            target_1.pickle_small_attributes()  # will save the new rate fields
            target_1.print_dict_file()  # will save the new rate fields
