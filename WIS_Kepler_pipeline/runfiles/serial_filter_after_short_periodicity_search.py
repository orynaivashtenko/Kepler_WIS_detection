#!/usr/bin/env python3

# TODO work on this
# #########################################################################################################################
# THIS FILE CHECKS THAT THE TRIGGER IS NOT A BRIGHT KOI AND IS NOT TOO SHORT
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
import target_class
import search_pipeline as spip
import pipeline_parameters as pp
from pipeline_parameters import Best_periodic_scores_filename_postfix
from file_operations import report_exception

Data_location = pp.Data_location
Output_location = pp.Output_location
Previous_slidable_triggers_file = '/home/labs/barakz/orynai/run_11/slidable_triggers_run_11_not_koi_not_short.txt'
Slidable_triggers_file = '/home/labs/barakz/orynai/run_11/slidable_triggers_run_11.txt'
# Non_slidable_triggers_file = '/home/labs/barakz/orynai/run_11/non_slidable_triggers_run_11.txt'
Non_slidable_triggers_file = '/home/labs/barakz/orynai/run_11/non_slidable_triggers_run_11_having_10_30_trigger.txt'
Problems_file = '/home/labs/barakz/orynai/run_11/problems_slidable_triggers_run_11.txt'

# ##### CLEANING #########
with open(Slidable_triggers_file, 'w'):
    pass
with open(Non_slidable_triggers_file, 'w'):
    pass
# ######################

non_slidable_tids = set()

all_threshold_crossings_file = Previous_slidable_triggers_file
with open(all_threshold_crossings_file, 'r') as fil:
    lines = fil.readlines()

lines = [line.strip().split() for line in lines]
all_triggers = [[str(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), int(float(x[5])), float(x[6])] for x in lines]
all_triggers = np.array(all_triggers, dtype='object')

for iii, trigger in enumerate(all_triggers):
    tid = str(trigger[0])

    is_slidable = True
    reason = ''

    if tid in non_slidable_tids:
        is_slidable = False

    if is_slidable:
        data_location = os.path.join(Data_location, tid[:4], tid)
        output_location = os.path.join(Output_location, tid[:4], tid)
        try:
            target_1 = target_class.KeplerTarget(tid, data_location=data_location, output_location=output_location)
            target_1.unpickle_small_attributes()

            # Loading all the short periodicity triggers
            target_1.best_candidates_concat = None
            # target_1.save_all_triggers_from_per_chunks(save_to_threshold_crossing_list=False,
            #                                            best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix)
            target_1.retrieve_best_candidates_concat(best_periodic_scores_filename_postfix=Best_periodic_scores_filename_postfix)
            integral_scores_normal = np.array(
                [target_1.best_candidates_concat[i][spip.Best_per_results_index_best_integral][0] for i in
                 range(len(target_1.best_candidates_concat))])
            integral_g_scores_normal = np.array(
                [target_1.best_candidates_concat[i][spip.Best_per_results_index_best_integral][1] for i in
                 range(len(target_1.best_candidates_concat))])
            max_normal_trigger = np.nanmax([np.nanmax(integral_scores_normal), np.nanmax(integral_g_scores_normal)])

            target_1.best_candidates_concat = None
            # target_1.save_all_triggers_from_per_chunks(save_to_threshold_crossing_list=False,
            #                                            best_periodic_scores_filename_postfix='best_periodic_scores_10_30.npy')
            # max_short_trigger = np.max(target_1.triggers[spip.Best_per_results_index_best_integral])
            target_1.retrieve_best_candidates_concat(best_periodic_scores_filename_postfix='best_periodic_scores_10_30.npy')
            integral_scores_short = np.array(
                [target_1.best_candidates_concat[i][spip.Best_per_results_index_best_integral][1] for i in
                 range(len(target_1.best_candidates_concat))])
            integral_g_scores_short = np.array(
                [target_1.best_candidates_concat[i][spip.Best_per_results_index_best_integral][0] for i in
                 range(len(target_1.best_candidates_concat))])
            max_short_trigger = np.nanmax([np.nanmax(integral_scores_short), np.nanmax(integral_g_scores_short)])
            if max_short_trigger >= max_normal_trigger:
                is_slidable = False
                reason = 'there is a short trigger of score ' + str(np.round(max_normal_trigger))

        except Exception as exc:
            is_slidable = False
            reason = 'there was an exception !!!!!!!!!!!!!! '
            report_exception(exception=exc, file_to_append=Problems_file,
                             title=tid, info='')

    if is_slidable:
        with open(Slidable_triggers_file, 'a') as fil:
            print(*trigger, file=fil)
    else:
        non_slidable_tids.add(tid)
        with open(Non_slidable_triggers_file, 'a') as fil:
            print(tid, reason, file=fil)

# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_filter_after_short_periodicity_search.py
