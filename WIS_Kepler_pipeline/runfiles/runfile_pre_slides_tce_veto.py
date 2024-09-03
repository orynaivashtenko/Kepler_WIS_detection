#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tce_class
from pipeline_parameters import (Threshold_crossing_list_file, Triggers_prefiltered_file,
                                 Output_location, Data_location, Target_pixels_location,
                                 Score_candidate_postprocess, Per_min_candidate)
# from templates.template_module import get_templates_durations_from_file
from trigger_functions import copy_trigger_between_files, Indices_dict, get_trigger_from_target_id

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'font.serif': 'Times New Roman', 'lines.linewidth': 2})
def_col = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.ioff()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]

# Taking the trigger values for the score and the period
trigger_main_search = get_trigger_from_target_id(Target_id, trigger_list_file=Threshold_crossing_list_file)
int_score_search = trigger_main_search[Indices_dict['int_score']]
per_search = trigger_main_search[Indices_dict['per']]

# Only if the score and the period are good enough for postprocessing, do the postprocessing:
if int_score_search >= Score_candidate_postprocess and per_search >= Per_min_candidate:
    tce = tce_class.KeplerTCE(target_id=Target_id,
                              output_location=Output_location, data_location=Data_location,
                              target_pixels_location=Target_pixels_location,
                              load_pickled_attributes=False, load_target_attributes=False)
    tce.unpickle_small_attributes(load_target_attributes=True, overwrite_existing_by_target=True,
                                  overwrite_location_parameters=False)

    tce.run_all_checks_exclusive()
    tce.pickle_small_attributes()
    tce.print_dict_file()

    if tce.is_good_pre_filtering is True:
        copy_trigger_between_files(target_id=Target_id, source_file=Threshold_crossing_list_file,
                                   destination_file=Triggers_prefiltered_file)
