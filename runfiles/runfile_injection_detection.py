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
from file_operations import report_job_done
from runfiles_parameters import Jobs_monitor_folder
import argparse
from pipeline_parameters import N_injections_chunk, N_injections_total
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--first_injection_load_parameters_index", type=str, nargs=1,
                    help="Index in the list of all the detection parameters, by chunks of N_injections_chunk",
                    default="0")
parser.add_argument("--job_name", type=str, nargs=1, help="name of the job to report", default="003097222")

args = parser.parse_args()
Target_id = args.target_id[0]
First_injection_load_parameters_index = int(int(args.first_injection_load_parameters_index[0]) * N_injections_chunk)
job_name = args.job_name[0]

for i in range(First_injection_load_parameters_index, min(First_injection_load_parameters_index + N_injections_chunk, N_injections_total)):
    targ = target_class.KeplerTarget(target_id=Target_id,
                                     load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
    targ.unpickle_small_attributes()
    # The KeplerLightCurve object is created inside the method, no need in initialization.
    targ.run_injection_search(injection_load_parameters_index=i)
    # I did not remove the injection parameters file because it can be not the last job
    # Writing that this index was completed:
    last_digit = Target_id[-1]
    report_job_done(job_name=job_name, jobs_monitor_folder=Jobs_monitor_folder, str_to_add='_'+str(i))

