#!/usr/bin/env python3

import time
import numpy.random as npr
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
from runfiles_parameters import Jobs_monitor_folder
from file_operations import report_job_done
from pipeline_parameters import Slides_clean_res_filename_postfix
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
parser.add_argument("--period", type=str, nargs=1, help="period, days", default="100")
parser.add_argument("--ftt", type=str, nargs=1, help="first transit time, days Kepler time", default="140")
parser.add_argument("--job_name", type=str, nargs=1, help="name of the job to report", default="003097222")

args = parser.parse_args()
Target_id = args.target_id
Period = args.period
Ftt = args.ftt
job_name = args.job_name[0]

if isinstance(Target_id, list):
    Target_id = Target_id[0]
if isinstance(Period, list):
    Period = Period[0]
Period = float(Period)
if isinstance(Ftt, list):
    Ftt = Ftt[0]
Ftt = float(Ftt)

targ = target_class.KeplerTarget(target_id=Target_id,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
targ.unpickle_small_attributes()
# The KeplerLightCurve object is created inside the method, no need in initialization.
npr.seed(None)
rt = npr.rand() * 10.  # make it sleep random number of seconds because maybe many jobs will want to access the same file...
time.sleep(rt)
targ.run_timeslides(central_period=Period, use_period_chunks=True, use_shuffling_pieces=True,
                    clean_trigger=True, trigger_ftt=Ftt, trigger_period=Period,
                    slides_res_filename_postfix=Slides_clean_res_filename_postfix)
report_job_done(job_name=job_name, jobs_monitor_folder=Jobs_monitor_folder)
