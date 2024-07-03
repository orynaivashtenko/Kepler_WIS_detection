#!/usr/bin/env python3

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
from pipeline_parameters import Best_periodic_scores_filename_postfix
from file_operations import report_exception

# TODO write this in an organized way!
Data_location = '/home/labs/barakz/Collaboration-Kepler/KeplerData'
Output_location = '/home/labs/barakz/Collaboration-Kepler/orynai/output_run_11'
Slidable_triggers_file = '/home/labs/barakz/orynai/run_11/slidable_triggers_run_11.txt'
Non_slidable_triggers_file = '/home/labs/barakz/orynai/run_11/non_slidable_triggers_run_11.txt'
Problems_file = '/home/labs/barakz/orynai/run_11/problems_slidable_triggers_run_11.txt'

# ##### CLEANING #########
with open(Slidable_triggers_file, 'w'):
    pass
# ######################


Threshold_considered_period = 50.
Threshold_mes_koi_candidate = 20
Threshold_ses_koi_candidate = 8.

non_slidable_tids = set()

all_threshold_crossings_file = '/home/labs/barakz/orynai/run_11/all_threshold_crossings.txt'
with open(all_threshold_crossings_file, 'r') as fil:
    lines = fil.readlines()

lines = [line.strip().split() for line in lines]
all_triggers = [[str(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), int(float(x[5])), float(x[6])] for x in lines]
all_triggers = np.array(all_triggers, dtype='object')


for iii, trigger in enumerate(all_triggers):
    is_slidable = True
    reason = ''
    tid = str(trigger[0])

    if tid in non_slidable_tids:
        is_slidable = False

    if is_slidable:
        data_location = os.path.join(Data_location, tid[:4], tid)
        output_location = os.path.join(Output_location, tid[:4], tid)

        try:
            target_1 = target_class.KeplerTarget(tid, data_location=data_location, output_location=output_location)
            target_1.unpickle_small_attributes()

            # Maybe it is not slidable because it is KOI:
            if target_1.is_koi:
                max_ses_koi = target_1.koi_dict['koi_max_sngle_ev']
                max_ses_koi = [e for e in max_ses_koi if e != '']
                if len(max_ses_koi) > 0:
                    max_ses_koi = np.nanmax(np.array(max_ses_koi).astype(float))
                    if max_ses_koi > Threshold_ses_koi_candidate:
                        is_slidable = False
                        reason += 'max_ses_koi>' + str(Threshold_ses_koi_candidate) + '; '
                max_mes_koi = target_1.koi_dict['koi_max_mult_ev']
                max_mes_koi = [e for e in max_mes_koi if e != '']
                if len(max_mes_koi) > 0:
                    max_mes_koi = np.nanmax(np.array(max_mes_koi).astype(float))
                    if max_mes_koi > Threshold_mes_koi_candidate:
                        is_slidable = False
                        reason += 'max_mes_koi>' + str(Threshold_mes_koi_candidate) + '; '

            if is_slidable:
                # Maybe it is not slidable because it has the loudest bad stat which was not removed:
                target_1.save_all_triggers_from_per_chunks(save_to_threshold_crossing_list=False)
                max_trigger_index = np.nanargmax(target_1.triggers[spip.Best_per_results_index_best_integral])
                max_trigger_per = target_1.triggers[spip.Best_per_results_index_per][max_trigger_index]
                if max_trigger_per < Threshold_considered_period:
                    is_slidable = False
                    reason += 'max_trigger_per<' + str(Threshold_considered_period)

        except Exception as exc:
            is_slidable = False
            report_exception(exception=exc, file_to_append=Problems_file,
                             title=tid, info='')

    if is_slidable:
        with open(Slidable_triggers_file, 'a') as fil:
            print(*trigger, file=fil)
    else:
        non_slidable_tids.add(tid)
        with open(Non_slidable_triggers_file, 'a') as fil:
            print(tid, reason, file=fil)

# bsub -q short -R "rusage[mem=2GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/serial_filter_slidable_kids_all_stats_only_max.py
