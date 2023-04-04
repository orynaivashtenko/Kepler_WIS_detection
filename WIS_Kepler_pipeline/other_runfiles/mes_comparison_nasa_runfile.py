#!/usr/bin/env python3

import pickle
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
from mes_comparison_nasa_job_creator import Mes_comparison_folder
import file_operations as fop
from pipeline_parameters import *
import traceback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
parser.add_argument("--period", type=str, nargs=1, help="KOI period around which the search will be, in days", default="100")
args = parser.parse_args()
Target_id = args.target_id[0]
Period = float(args.period[0])

Output_location = '/home/labs/barakz/Collaboration-Kepler/orynai/run_6_mes_comparison_output'
Run_results_list_file = os.path.join(Mes_comparison_folder, 'Faint_nasa_kois_results.txt')

targ = target_class.KeplerTarget(target_id=Target_id,
                                 load_pickled_attributes=False,
                                 output_location=Output_location)  # all the arguments are set to the defaults of the class

if not targ.is_good_for_processing:  # is villanova binary or is not in Gaia
    targ.print_dict_file()
else:  # for now, run the search only if it is not a villanova binary
    targ.run_main_search(save_to_threshold_crossing_list=False,
                         clean_dir_if_not_threshold_crossing=False,
                         mask_koi_planet=False,
                         p_min=Period-1,
                         p_max=Period+1,
                         p_chunks=None)

    with open(Run_results_list_file, 'a') as fil1:  # will save here these target_ids outputs
        print(targ.target_id, targ.best_stat, targ.best_per, targ.best_temp, targ.best_phase,
              targ.best_first_transit_time, targ.best_ampl, targ.best_ppm, sep=' ', end='\n', file=fil1)
