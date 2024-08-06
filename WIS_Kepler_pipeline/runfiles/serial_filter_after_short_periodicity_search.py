#!/usr/bin/env python3

# #########################################################################################################################
# THIS FILE CHECKS THAT THE TRIGGER IS NOT SOMETHING WITH SHORT PERIODICITY
# #########################################################################################################################

import os

core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pipeline_parameters as pp
from pipeline_parameters import (Target_ids_prefiltered_file, Triggers_short_periodicity_filtered_file,
                                 Threshold_crossing_list_file, Maximal_short_periodicity_triggers_file,
                                 Non_slidable_tids_short_period_file)

from trigger_class import load_triggers, print_trigger_line

Data_location = pp.Data_location
Output_location = pp.Output_location

# ##### CLEANING #########
with open(Triggers_short_periodicity_filtered_file, 'w'):
    pass
with open(Non_slidable_tids_short_period_file, 'w'):
    pass
# ######################

non_slidable_tids = set()

target_ids, integral_scores, stats, pers, ftts, gauss_stats, templates = load_triggers(file=Threshold_crossing_list_file)
target_ids_sp, integral_scores_sp, stats_sp, pers_sp, ftts_sp, gauss_stats_sp, templates_sp = load_triggers(
    file=Maximal_short_periodicity_triggers_file)

with open(Target_ids_prefiltered_file, 'r') as fil:
    pre_filtered_tids = fil.readlines()
pre_filtered_tids = [tid.split()[0] for tid in pre_filtered_tids]

for tid in pre_filtered_tids:
    if tid in target_ids and tid in target_ids_sp:
        index = np.where(target_ids == tid)[0][0]
        index_sp = np.where(target_ids_sp == tid)[0][0]
        integral_score = integral_scores[index]
        integral_score_sp = integral_scores_sp[index_sp]
        stat = stats[index]
        stat_sp = stats_sp[index_sp]

        delta_integral = integral_score_sp - integral_score
        delta_stat = stat_sp - stat
        # The following is instead of if integral_score_sp >= integral_score so that it is the same as in the pipeline.
        if delta_integral >= - delta_stat * (pp.Delta_integral_close / pp.Delta_stat_close):
            reason = 'integral score ' + str(np.round(integral_score)) + ', short integral score ' + str(
                np.round(integral_score_sp)) + 'stat ' + str(np.round(stat)) + ', short stat ' + str(np.round(stat_sp))
            non_slidable_tids.add(tid)
            with open(Non_slidable_tids_short_period_file, "a") as fil:
                print(tid, reason, file=fil)
        else:
            with open(Triggers_short_periodicity_filtered_file, "a") as fil:
                print_trigger_line(target_id=target_ids[index], integral_score=integral_scores[index], stat=stats[index],
                                   per=pers[index], ftt=ftts[index], gauss_stat=gauss_stats[index], template=templates[index],
                                   file=fil)

        # # The following commented code should be used if there was something wrong with the triggers files, and we need to go to the best
        # periodic scores saved and define the best score again. Otherwise, there is no need, the best scores are saved in the
        # trigger list files.
        # import target_class
        # import search_pipeline as spip
        # from file_operations import report_exception
        # Problems_file = '/home/labs/barakz/orynai/run_11/problems_slidable_triggers_run_11.txt'
        # is_slidable = True
        # reason = ''
        #
        # if tid in non_slidable_tids:
        #     is_slidable = False
        #
        # if is_slidable:
        #     data_location = os.path.join(Data_location, tid[:4], tid)
        #     output_location = os.path.join(Output_location, tid[:4], tid)
        #     try:
        #         target_1 = target_class.KeplerTarget(tid, data_location=data_location, output_location=output_location)
        #         target_1.unpickle_small_attributes()
        #
        #         # Loading all the short periodicity triggers
        #         target_1.best_candidates_concat = None
        #         # target_1.save_all_triggers_from_per_chunks(save_to_threshold_crossing_list=False,
        #         #                                            best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix)
        #         target_1.retrieve_best_candidates_concat(best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix)
        #         integral_scores_normal = np.array(
        #             [target_1.best_candidates_concat[i][spip.Best_per_results_index_best_integral][0] for i in
        #              range(len(target_1.best_candidates_concat))])
        #         integral_g_scores_normal = np.array(
        #             [target_1.best_candidates_concat[i][spip.Best_per_results_index_best_integral][1] for i in
        #              range(len(target_1.best_candidates_concat))])
        #         max_normal_trigger = np.nanmax([np.nanmax(integral_scores_normal), np.nanmax(integral_g_scores_normal)])
        #
        #         target_1.best_candidates_concat = None
        #         # target_1.save_all_triggers_from_per_chunks(save_to_threshold_crossing_list=False,
        #         #                                            best_periodic_scores_filename_postfix='best_periodic_scores_10_30.npy')
        #         # max_short_trigger = np.max(target_1.triggers[spip.Best_per_results_index_best_integral])
        #         target_1.retrieve_best_candidates_concat(best_periodic_scores_filename_postfix='best_periodic_scores_10_30.npy')
        #         integral_scores_short = np.array(
        #             [target_1.best_candidates_concat[i][spip.Best_per_results_index_best_integral][1] for i in
        #              range(len(target_1.best_candidates_concat))])
        #         integral_g_scores_short = np.array(
        #             [target_1.best_candidates_concat[i][spip.Best_per_results_index_best_integral][0] for i in
        #              range(len(target_1.best_candidates_concat))])
        #         max_short_trigger = np.nanmax([np.nanmax(integral_scores_short), np.nanmax(integral_g_scores_short)])
        #         if max_short_trigger >= max_normal_trigger:
        #             is_slidable = False
        #             reason = 'there is a short trigger of score ' + str(np.round(max_normal_trigger))
        #
        #     except Exception as exc:
        #         is_slidable = False
        #         reason = 'there was an exception !!!!!!!!!!!!!! '
        #         report_exception(exception=exc, file_to_append=Problems_file,
        #                          title=tid, info='')
        #
        # if is_slidable:
        #     with open(Triggers_short_periodicity_filtered_file, "a") as fil:
        #         print_trigger_line(target_id=target_ids[index], integral_score=integral_scores[index], stat=stats[index],
        #                            per=pers[index], ftt=ftts[index], gauss_stat=gauss_stats[index], template=templates[index],
        #                            file=fil)
        #
        # else:
        #     non_slidable_tids.add(tid)
        #     with open(Non_slidable_tids_short_period_file, "a") as fil:
        #         print(tid, reason, file=fil)

# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_filter_after_short_periodicity_search.py
