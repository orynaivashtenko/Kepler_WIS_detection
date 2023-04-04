#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
import argparse
from pipeline_parameters import N_injections_chunk, N_injections_total
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
parser.add_argument("--first_injection_load_parameters_index", type=str, nargs=1,
                    help="Index in the list of all the detection parameters, by chunks of N_injections_chunk",
                    default="0")

args = parser.parse_args()
Target_id = args.target_id[0]
# In the run_6, it used to be this:
# First_injection_load_parameters_index = int(args.first_injection_load_parameters_index[0])
# this should have been multiplied by N_injections_chunk????????:
# So correct to this:
First_injection_load_parameters_index = int(int(args.first_injection_load_parameters_index[0]) * N_injections_chunk)

for i in range(First_injection_load_parameters_index, min(First_injection_load_parameters_index + N_injections_chunk, N_injections_total)):
    targ = target_class.KeplerTarget(target_id=Target_id,
                                     load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
    targ.unpickle_small_attributes()
    # The KeplerLightCurve object is created inside the method, no need in initialization.
    targ.run_injection_search(injection_load_parameters_index=i)
    # I did not remove the injection parameters file because it can be not the last job
