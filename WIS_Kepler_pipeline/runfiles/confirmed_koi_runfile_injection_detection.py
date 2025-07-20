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
import argparse
from pipeline_parameters import N_injections_chunk, N_injections_total
from confirmed_koi_search_create import Kepler_confirmed_koi_output_folder, Kepler_confirmed_koi_problems_folder
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--first_injection_load_parameters_index", type=str, nargs=1,
                    help="Index in the list of all the detection parameters, by chunks of N_injections_chunk",
                    default="0")

args = parser.parse_args()
Target_id = args.target_id[0]
First_injection_load_parameters_index = int(int(args.first_injection_load_parameters_index[0]) * N_injections_chunk)

output_location = str(os.path.join(Kepler_confirmed_koi_output_folder, Target_id))

for i in range(First_injection_load_parameters_index, min(First_injection_load_parameters_index + N_injections_chunk, N_injections_total)):
    targ = target_class.KeplerTarget(target_id=Target_id, output_location=output_location,
                                     problems_folder=Kepler_confirmed_koi_problems_folder,
                                     load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
    targ.unpickle_small_attributes()
    # The KeplerLightCurve object is created inside the method, no need in initialization.
    if targ.is_good_for_processing:
        targ.run_injection_search(injection_load_parameters_index=i, period_avoid_masking=targ.best_per)
    # I did not remove the injection parameters file because it can be not the last job
