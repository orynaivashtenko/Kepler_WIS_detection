#!/usr/bin/env python3

# #########################################################################################################################
# THIS FILE OPENS TRIGGER LIST, SELECTS THOSE TRIGGERS THAT HAVE INTEGRAL SCORE ABOVE THRESHOLD AND PRINTS THEM TO ANOTHER FILE
# #########################################################################################################################

import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = core_num  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = core_num  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = core_num  # export NUMEXPR_NUM_THREADS=6

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import trigger_functions

Trigger_list_file = '/home/labs/barakz/orynai/run_11/threshold_crossing_scrambled_kid_info.txt'
Filtered_trigger_list_file = '/home/labs/barakz/orynai/run_11/threshold_crossing_scrambled_triggers_to_slide.txt'
Threshold_integral_score = 49

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Trigger_list_file)
parser.add_argument("--filtered_trigger_list_file", type=str, nargs=1, help="path to file where to put triggers",
                    default=Filtered_trigger_list_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file
if isinstance(trigger_list_file, list):
    trigger_list_file = trigger_list_file[0]
filtered_trigger_list_file = args.filtered_trigger_list_file
if isinstance(filtered_trigger_list_file, list):
    filtered_trigger_list_file = filtered_trigger_list_file[0]


target_ids, integral_scores, stats, pers, ftts, gauss_stats, templates = trigger_class.load_triggers(file=trigger_list_file)

with open(filtered_trigger_list_file, 'w') as fil1:  # will save here these good targets
    for i in range(len(target_ids)):
        if integral_scores[i] >= Threshold_integral_score:
            trigger_class.print_trigger_line(
                target_id=target_ids[i],
                stat=stats[i],
                per=pers[i],
                ftt=ftts[i],
                gauss_stat=gauss_stats[i],
                integral_score=integral_scores[i],
                template=templates[i], file=fil1, sep=' ', end='\n')

