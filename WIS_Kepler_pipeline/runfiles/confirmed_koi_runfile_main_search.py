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
import target_class
from locations import Run_outputs_folder, Run_name
from pipeline_parameters import N_injections_chunk, N_injections_total
from confirmed_koi_search_create import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--koi_index", type=str, nargs=1,
                    help="Index of KOI because the target mat have a few KOIs",
                    default="0")

args = parser.parse_args()
Target_id = args.target_id
if isinstance(Target_id, list):
    Target_id = Target_id[0]
koi_index = args.koi_index
if isinstance(koi_index, list):
    koi_index = int(koi_index[0])

output_location = str(os.path.join(Kepler_confirmed_koi_output_folder, Target_id))
if not os.path.isdir(output_location):
    os.mkdir(output_location)

with open(Confirmed_to_run_koi_dict_file, 'rb') as koi_catalog_dict_file:
    koi_info_dict = pickle.load(koi_catalog_dict_file)
periods_koi = koi_info_dict[Target_id]['koi_period']
periods_koi = [float(p) for p in periods_koi]
ftts_koi = koi_info_dict[Target_id]['koi_time0bk']
ftts_koi = [float(t) for t in ftts_koi]

# FOR NOW, ONLY ONE PERIOD IS USED
# for period in periods_koi:
period = periods_koi[koi_index]
targ = target_class.KeplerTarget(target_id=Target_id,
                                 output_location=output_location,
                                 load_pickled_attributes=False,
                                 problems_folder=Kepler_confirmed_koi_problems_folder,
                                 kill_villanova_binary=False
                                 )  # all the arguments are set to the defaults of the class

targ.run_one_chunk_search(period,
                          saving_folder=targ.output_location, allow_saving_folder_creation=False,
                          add_saving_folder_prefix=False,
                          candidate_threshold=30.,
                          mask_koi_planet=True, avoid_masking_central_period=True,
                          mask_kepler_candidates=True,
                          mask_kepler_false_positives=True,
                          mask_kepler_bright_candidates=True,
                          delete_irrelevant=True,
                          save_connected_stats=True,
                          add_period_to_connected_statistic_data_filename_postfix=True,
                          best_periodic_scores_filename_postfix=Kepler_confirmed_koi_best_periodic_scores_filename_postfix,
                          add_period_to_best_periodic_scores_filename_postfix=True,
                          save_to_threshold_crossing_list=True,
                          threshold_crossing_list_file=None,
                          threshold_g_crossing_list_file=None,
                          threshold_crossing_list_location=Kepler_confirmed_koi_run_folder,
                          threshold_crossing_list_filename=Kepler_confirmed_koi_threshold_crossing_list_filename,
                          threshold_g_crossing_list_filename=Kepler_confirmed_koi_threshold_g_crossing_list_filename,
                          save_multiple_triggers=True,
                          threshold_crossing_list_multiple_triggers_filename=
                          Kepler_confirmed_koi_threshold_crossing_list_multiple_triggers_filename,
                          threshold_g_crossing_list_multiple_triggers_filename=
                          Kepler_confirmed_koi_threshold_g_crossing_list_multiple_triggers_filename,
                          seeds_scrambling_run_file=None,
                          scramble=False, print_no_trigger_found=False)
targ.pickle_small_attributes()
targ.print_dict_file()
