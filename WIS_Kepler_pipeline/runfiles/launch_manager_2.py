#!/usr/bin/env python3

import time
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from runfiles_parameters import *
from file_operations import check_number_of_submitted_jobs
from pipeline_parameters import Threshold_crossing_lists_folder

Max_work_time = 6.9 * 24 * 60 * 60  # in seconds; the job in physics-short is killed after 7 days
Max_jobs_submitted = 4800  # Maximal number of jobs that we can have in bjobs at a certain time

# generate a random number and name the output file accordingly
# if "Successfully completed." in the output, then delete it


# Take job files from the running list randomly to avoid two timeslides file running simultaneously

queues = ['physics-short', 'new-short']

timeout_start = time.time()
time_counter = 0
while (time.time() < timeout_start + Max_work_time) and (len(os.listdir(Job_files_folder)) > 0):
    if time.time() - timeout_start > time_counter * 2 * 60:  # check every 2 min
        wcs = check_number_of_submitted_jobs(output_dir=Threshold_crossing_lists_folder, queues=queues)
        wc = wcs[0] + wcs[1]
        for ff in [fff for fff in os.listdir(Job_files_folder) if 'injection_' in fff][:Max_jobs_submitted - wc]:
            os.system('bsub < '+os.path.join(Job_files_folder, ff))
            time.sleep(0.1)
            os.remove(os.path.join(Job_files_folder, ff))
        time_counter += 1




# bsub -q physics-long -R "rusage[mem=3.9GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_2.py
