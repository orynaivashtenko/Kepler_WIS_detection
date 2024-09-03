#!/usr/bin/env python3

import time
# import datetime
import os

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from runfiles_parameters import *
from file_operations import check_number_of_submitted_jobs
from pipeline_parameters import Run_root_folder

# ################################################################################################
# LAUNCH MANAGER WHICH DOES NOT MOVE OR DELETE OUTPUTFILES AND ERRORFILES AND DOES NOT MOVE JOB FILES
# ################################################################################################

Max_work_time = 6.9 * 24 * 60 * 60  # in seconds; the job in physics-short is killed after 7 days
Max_jobs_submitted = 4950  # Maximal number of jobs that we can have in bjobs at a certain time

# generate a random number and name the output file accordingly
# if "Successfully completed." in the output, then delete it

queues = Queues_check
# Take job files from the running list randomly to avoid two timeslides file running simultaneously

minutes_between_checks = 10.
timeout_start = time.time()
time_last = timeout_start - (minutes_between_checks + 0.01) * 60  # I want it to start immediately
while (time.time() < timeout_start + Max_work_time) and (len(os.listdir(Job_files_folder)) > 0):
    if time.time() - time_last > minutes_between_checks * 60:  # check every 10 min
        time_last = time.time()
        # wc = check_number_of_submitted_jobs(output_dir=Run_root_folder)
        wcs = check_number_of_submitted_jobs(output_dir=Run_root_folder, queues=queues)
        if hasattr(wcs, '__len__'):
            wc = sum(wcs)
        else:
            wc = wcs
        for ff in os.listdir(Job_files_folder)[:Max_jobs_submitted - wc]:
            os.system('bsub < ' + os.path.join(Job_files_folder, ff))
            time.sleep(0.1)
            os.remove(os.path.join(Job_files_folder, ff))
            # -> In slides and injections, just remove the submitted job; there are too many of them
    else:
        time.sleep(30)  # sleep 30 seconds if the time to check did not yet come

# bsub -q physics-long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_slides_injections.py
# bsub -q long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_slides_injections.py
