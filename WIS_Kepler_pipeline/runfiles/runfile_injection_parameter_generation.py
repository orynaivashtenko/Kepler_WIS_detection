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
from pipeline_parameters import *
import argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--period", type=str, nargs=1, help="period, days", default="100")
parser.add_argument("--job_name", type=str, nargs=1, help="name of the job to report", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]
Period = float(args.period[0])
job_name = args.job_name[0]

targ = target_class.KeplerTarget(target_id=Target_id,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
# read the fields
targ.unpickle_small_attributes()
targ.generate_injection_parameters(central_period=Period)

report_job_done(job_name=job_name, jobs_monitor_folder=Jobs_monitor_folder)