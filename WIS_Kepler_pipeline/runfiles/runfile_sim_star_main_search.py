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
import datetime
from runfiles_parameters import *
import traceback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="000000000")
args = parser.parse_args()
Target_id = args.target_id
if isinstance(Target_id, list):
    Target_id = Target_id[0]

targ = target_class.KeplerTarget(target_id=Target_id,
                                 simulated_star=True,
                                 load_pickled_attributes=False,
                                 data_location=None,
                                 output_location='/home/labs/barakz/Collaboration-Kepler/orynai/simulated_star')

targ.run_main_search(save_to_threshold_crossing_list=False,
                     clean_dir_if_not_threshold_crossing=False,
                     save_multiple_triggers=False)

# bsub -q short -R "rusage[mem=3.9G]" -sp 100 -J sim /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/runfile_sim_star_main_search.py
