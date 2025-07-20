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
import target_class
import trigger_functions
from pipeline_parameters import Data_location
from confirmed_koi_search_create import *
from astropy.table import Table

confirmed_trigger_file = os.path.join(Kepler_confirmed_koi_run_folder, Kepler_confirmed_koi_threshold_crossing_list_filename)
(target_ids_conf, integral_scores_conf, stats_conf, pers_conf, ftts_conf,
 gauss_stats_conf, templates_conf) = trigger_functions.load_triggers_unique(file=confirmed_trigger_file)


# CALCULATING SIGNIFICANCES HERE
p_planets_for_target_ids = []
false_alarm_prob_trigger_for_target_ids = []
transit_probability_for_target_ids = []
transiting_planets_of_radii_in_per_range_one_star_for_target_ids = []
transit_and_range_probability_for_target_ids = []
transiting_planets_in_relevant_range_for_target_ids = []

for i, tid in enumerate(target_ids_conf):
    output_location = str(os.path.join(Kepler_confirmed_koi_output_folder, tid))
    target_1 = target_class.KeplerTarget(tid, data_location=os.path.join(Data_location, tid[:4], tid),
                                         output_location=output_location, problems_folder=Kepler_confirmed_koi_problems_folder)
    target_1.unpickle_small_attributes()
    if not np.isnan(target_1.best_per) and not np.isnan(target_1.stellar_radius) and target_1.is_good_for_processing:
        target_1.analyze_injections_and_background()
        if (target_1.prob_density_planet_not_noise < 0.1) and (target_1.best_int_score > 60):
            target_1.analyze_injections_and_background(bin_width=3.)
            # -> Sometimes the injection range is so broad that there can be holes between bins
        p_planets_for_target_ids.append(target_1.prob_density_planet_not_noise)
        false_alarm_prob_trigger_for_target_ids.append(target_1.false_alarm_prob_trigger)
        transit_probability_for_target_ids.append(target_1.transit_probability)
        transiting_planets_of_radii_in_per_range_one_star_for_target_ids.append(target_1.transiting_planets_of_radii_in_per_range_one_star)
        transit_and_range_probability_for_target_ids.append(target_1.transit_and_range_probability)
        transiting_planets_in_relevant_range_for_target_ids.append(target_1.transiting_planets_in_relevant_range)
    else:
        p_planets_for_target_ids.append(np.nan)
        false_alarm_prob_trigger_for_target_ids.append(np.nan)
        transit_probability_for_target_ids.append(np.nan)
        transiting_planets_of_radii_in_per_range_one_star_for_target_ids.append(np.nan)
        transit_and_range_probability_for_target_ids.append(np.nan)
        transiting_planets_in_relevant_range_for_target_ids.append(np.nan)

dict_significance = {
    'target_id': list(target_ids_conf),
    'p_planet': p_planets_for_target_ids,
    'false_alarm_prob': false_alarm_prob_trigger_for_target_ids,
    'transit_probability': transit_probability_for_target_ids,
    'transiting_planets_of_radii_in_per_range_one_star': transiting_planets_of_radii_in_per_range_one_star_for_target_ids,
    'transit_and_range_probability': transit_and_range_probability_for_target_ids,
    'transiting_planets_in_relevant_range': transiting_planets_in_relevant_range_for_target_ids}

table_results = Table(dict_significance)
p_planet_table_file = os.path.join(Kepler_confirmed_koi_run_folder, Kepler_confirmed_koi_p_planet_table)
table_results.write(p_planet_table_file, format='ascii.ecsv', overwrite=True)

# bsub -q physics-short -R "rusage[mem=3.9GB]"  source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/confirmed_koi_runfile_p_planet.py































#
# import os
# os.environ["OMP_NUM_THREADS"] = "1"  #
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"  #
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# import tce_class
# import target_class
# from pipeline_parameters import *
# from confirmed_koi_search_create import Kepler_confirmed_koi_output_folder, Kepler_confirmed_koi_problems_folder
# import locations as locs
#
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
# parser.add_argument("--period", type=str, nargs=1, help="period, days", default="100")
# args = parser.parse_args()
# Target_id = args.target_id[0]
# Period = float(args.period[0])
#
# output_location = str(os.path.join(Kepler_confirmed_koi_output_folder, Target_id))
# CREATE DIRECTORY FOR PLOTS AND MAKE SURE IT WORKS
# tce = tce_class.WisTCE(target_id=Target_id,
#                        output_location=output_location, data_location=Data_location,
#                        target_pixels_location=Target_pixels_location,
#                        problems_folder=Kepler_confirmed_koi_problems_folder,
#                        load_pickled_attributes=False, load_target_attributes=False)
# tce.unpickle_small_attributes(load_target_attributes=True, overwrite_existing_by_target=True,
#                               overwrite_location_parameters=False)
# tce.run_all_checks_exclusive(print_message_to_history=True, save_byproducts_to_file=True,
#                              produce_plots=True, no_show=True, save_plots=True,
#                              update_pickling=True, enforce_all_checks=True, include_veto_short_periodicity=False)
# tce.pickle_small_attributes()
# tce.print_dict_file()
#
# targ = target_class.KeplerTarget(target_id=Target_id, output_location=output_location,
#                                  problems_folder=Kepler_confirmed_koi_problems_folder,
#                                  load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
# targ.unpickle_small_attributes()
# ...
# plot_all_after_slides_injections(tce=None, target=None, save_fig=Save_fig, no_show=No_show,
#                                      plots_output_location=locs.Triggers_plots_location,
#                                      include_periodogram=FALSE)
# # ----------------------------------------------------------------------
# # targ.klc = spip.KeplerLightCurve(target_id=targ.target_id, get_data=True, data_location=targ.data_location,
# #                                  output_location=targ.output_location, m_1_3_r_s=targ.m_1_3_r_s,
# #                                  simulated_red_star=False, simulated_white_star=False)
# # targ.mask_koi_planet(modify_self_klc=True, return_durations_transit_times=False,
# #                      mask_candidates=Mask_kepler_candidates,
# #                      mask_false_positives=Mask_kepler_false_positives,
# #                      mask_bright_candidates=Mask_kepler_bright_candidates,
# #                      bright_candidate_mes=Kepler_bright_candidate_mes,
# #                      bright_candidate_ses=Kepler_bright_candidate_ses)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # ----------------------------------------------------------------------------------------------------
# # TCE VETO
# import os
# os.environ["OMP_NUM_THREADS"] = "1"  #
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"  #
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# import sys
#
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# import tce_class
# from pipeline_parameters import (Threshold_crossing_list_file, Triggers_prefiltered_file,
#                                  Output_location, Data_location, Target_pixels_location,
#                                  Score_candidate_postprocess, Per_min_candidate)
# # from locations import Pickle_tce_small_attributes_filename_postfix
# import locations as locs
# from runfiles_parameters import Error_list_file
# from trigger_functions import copy_trigger_between_files, Indices_dict, get_trigger_from_target_id
# from auxiliary_functions import get_saving_file_path
# import traceback
# import datetime
#
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
# args = parser.parse_args()
# Target_id = args.target_id[0]
#
# # Taking the trigger values for the score and the period
# trigger_main_search = get_trigger_from_target_id(Target_id, trigger_list_file=Threshold_crossing_list_file)
# int_score_search = float(trigger_main_search[Indices_dict['int_score']])
# per_search = float(trigger_main_search[Indices_dict['per']])
#
# # Only if the score and the period are good enough for postprocessing, do the postprocessing:
# if int_score_search >= Score_candidate_postprocess and per_search >= Per_min_candidate:
#     try:
#         tce = tce_class.WisTCE(target_id=Target_id,
#                                output_location=Output_location, data_location=Data_location,
#                                target_pixels_location=Target_pixels_location,
#                                load_pickled_attributes=False, load_target_attributes=False)
#         # ---------------------------------
#         prefixes_to_clean = [
#             locs.Dict_tce_txt_filename_postfix,
#             locs.Pickle_tce_small_attributes_filename_postfix,
#             locs.Folded_binned_wh_flux_model_even_odd_filename_postfix,
#             locs.Periodic_stats_centroid_y_filename_postfix,
#             locs.Connected_stats_centroid_y_filename_postfix,
#             locs.Periodic_stats_centroid_x_filename_postfix,
#             locs.Connected_stats_centroid_x_filename_postfix,
#             locs.Periodic_stats_bg_pixels_filename_postfix,
#             locs.Connected_stats_bg_pixels_filename_postfix,
#             locs.Periodic_stats_star_pixels_filename_postfix,
#             locs.Connected_stats_star_pixels_filename_postfix,
#             locs.Fold_residuals_flux_td_chi_filename_postfix,
#             locs.Folded_binned_flux_template_filename_postfix,
#             locs.Mf_scores_residuals_filename_postfix,
#             locs.Fold_residuals_flux_fd_chi2_filename_postfix
#         ]
#         for ff in os.listdir(tce.output_location):
#             for prefix in prefixes_to_clean:
#                 if prefix in ff:
#                     os.remove(os.path.join(tce.output_location, ff))
#         # Clean the pickle file to make sure we run from scratch:
#         # file_path_unpickling = get_saving_file_path(output_location=tce.output_location, target_id=tce.target_id,
#         #                                             filename_postfix=Pickle_tce_small_attributes_filename_postfix,
#         #                                             file_format='pickle')
#         # if os.path.exists(file_path_unpickling):
#         #     os.remove(file_path_unpickling)
#         # ---------------------------------
#         tce.unpickle_small_attributes(load_target_attributes=True, overwrite_existing_by_target=True,
#                                       overwrite_location_parameters=False)
#         tce.run_all_checks_exclusive(print_message_to_history=True, save_byproducts_to_file=True,
#                                      produce_plots=True, no_show=True, save_plots=True,
#                                      update_pickling=True, enforce_all_checks=True)
#         tce.pickle_small_attributes()
#         tce.print_dict_file()
#
#         if tce.is_good_pre_filtering is True:
#             # copy_trigger_between_files(target_id=Target_id, source_file=pp.Threshold_crossing_list_file,
#             #                            destination_file=pp.Triggers_for_slides_injections_file)
#             copy_trigger_between_files(target_id=Target_id, source_file=Threshold_crossing_list_file,
#                                        destination_file=Triggers_prefiltered_file)
#     except Exception as exc:
#         tb = sys.exc_info()
#         list_str_exc = traceback.format_exception(exc, value=tb[1], tb=tb[2])
#         with open(Error_list_file, 'a') as err_list_fil:
#             print(Target_id, file=err_list_fil)
#             print(datetime.datetime.now(), file=err_list_fil)
#             for str_exc in list_str_exc:
#                 print(str_exc.strip(), file=err_list_fil)
#             print('', file=err_list_fil)
#
#
# # ---------------------------------------------------------------------------------------------------------------------
# # SHORT PERIODICITY
# # ---------------------------------------------------------------------------------------------------------------------
# #!/usr/bin/env python3
#
# import os
#
# # import auxiliary_functions
#
# os.environ["OMP_NUM_THREADS"] = "1"  #
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"  #
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# import numpy as np
# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# import target_class
# import search_pipeline as spip
# import pipeline_parameters as pp
# from trigger_functions import print_trigger_line  # , get_trigger_from_target_id, Indices_dict, copy_trigger_between_files
# from auxiliary_functions import get_saving_file_path
# import traceback
# from runfiles_parameters import Error_list_file
# import datetime
#
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
# args = parser.parse_args()
# Target_id = args.target_id[0]
#
# Saving_maximal_scores_file = pp.Maximal_short_periodicity_triggers_file
# Saving_maximal_g_scores_file = pp.Maximal_short_periodicity_g_triggers_file
# P_short_min = pp.P_short_min
# P_short_max = pp.P_short_max
#
# try:
#     targ = target_class.KeplerTarget(target_id=Target_id,
#                                      load_pickled_attributes=False)  # all the arguments are set to the defaults like main search
#
#     # We will have here a simplified version of the search. We do not change anything in the saved parameters after the main search!
#
#     targ.unpickle_small_attributes()
#     # Just in case:
#     targ.best_stat = None
#     targ.best_per = None
#     targ.best_temp = None
#     targ.best_ftt = None
#     targ.best_gauss_stat = None
#     targ.best_int_score = None
#     targ.best_g_stat = None
#     targ.best_g_per = None
#     targ.best_g_temp = None
#     targ.best_g_ftt = None
#     targ.best_g_gauss_stat = None
#     targ.best_g_int_score = None
#
#     targ.get_connected_klc(reinitialize=True, mask_koi_planet=True, save_conditioned_data=True, save_connected_stats=True)
#     targ.klc.interleave_shifts()
#
#     duty_cycle = spip.Duty_cycle
#     per_phases_per_chunk = pp.Per_phases_per_chunk
#     overlap_period_bins = spip.P_bins_overlap
#     min_required_nonzero_transits_number = spip.Min_required_nonzero_transits_number
#     min_required_nonzero_transits_fraction = spip.Min_required_nonzero_transits_fraction
#
#     targ.p_min = P_short_min
#     targ.p_max = P_short_max
#     targ.periods_in_chunks = targ.klc.make_periods_chunks(p_min=targ.p_min, p_max=targ.p_max,
#                                                           per_phases_per_chunk=targ.per_phases_per_chunk,
#                                                           duty_cycle=targ.duty_cycle, dt=targ.klc.dt_days, t_mission=targ.klc.t_mission)
#     best_candidates_concat, failures = targ.search_periodicity_over_chunks(
#         period_chunks=targ.periods_in_chunks,
#         force_interleave_shifts=False,
#         min_required_nonzero_transits_number=min_required_nonzero_transits_number,
#         min_required_nonzero_transits_fraction=min_required_nonzero_transits_fraction, overlap_period_bins=overlap_period_bins,
#         report_exceptions=True, problem_description='Short: ',
#         return_failures=True, save_search_time=False, save_veto_masks=False)
#
#     file_path = get_saving_file_path(output_location=targ.output_location, target_id=targ.target_id,
#                                      filename_postfix=pp.Best_short_periodic_scores_filename_postfix, file_format='npy')
#     np.save(file_path, np.array(best_candidates_concat, dtype=object), allow_pickle=True)
#     targ.save_best_candidate(candidate_threshold=pp.Score_candidate_search,
#                              threshold_crossing_list_file=None,
#                              threshold_crossing_list_location=None,
#                              threshold_crossing_list_filename=None,
#                              threshold_g_crossing_list_filename=None,
#                              save_to_threshold_crossing_list=False,
#                              best_candidates_concat=best_candidates_concat)
#
#     with open(Saving_maximal_scores_file, 'a') as fil1:  # will save here these good targets
#         print_trigger_line(target_id=targ.target_id, integral_score=targ.best_int_score, stat=targ.best_stat, per=targ.best_per,
#                            ftt=targ.best_ftt, gauss_stat=targ.best_gauss_stat, template=targ.best_temp,
#                            file=fil1, sep=' ', end='\n')
#
#     with open(Saving_maximal_g_scores_file, 'a') as fil1:  # will save here these good targets
#         print_trigger_line(target_id=targ.target_id, integral_score=targ.best_g_int_score, stat=targ.best_g_stat, per=targ.best_g_per,
#                            ftt=targ.best_g_ftt, gauss_stat=targ.best_g_gauss_stat, template=targ.best_g_temp,
#                            file=fil1, sep=' ', end='\n')
#
#     # trigger_main_search = get_trigger_from_target_id(Target_id, trigger_list_file=pp.Threshold_crossing_list_file)
#     # int_score_main = trigger_main_search[Indices_dict['int_score']]
#     # if int_score_main > targ.best_int_score:
#     #     copy_trigger_between_files(target_id=Target_id, source_file=pp.Threshold_crossing_list_file,
#     #                                destination_file=pp.Triggers_for_slides_injections_file)
# except Exception as exc:
#     tb = sys.exc_info()
#     list_str_exc = traceback.format_exception(exc, value=tb[1], tb=tb[2])
#     with open(Error_list_file, 'a') as err_list_fil:
#         print(Target_id, file=err_list_fil)
#         print(datetime.datetime.now(), file=err_list_fil)
#         for str_exc in list_str_exc:
#             print(str_exc.strip(), file=err_list_fil)
#         print('', file=err_list_fil)
#
# # bsub -q physics-short -R "rusage[mem=3.9GB]" /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_short_periodicity_search.py --target_id
# # ---------------------------------------------------------------------------------------------------------------------
# # SIGNIFICANCE