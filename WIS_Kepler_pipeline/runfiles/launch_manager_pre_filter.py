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
# LAUNCH MANAGER WHICH DOES NOT MOVE OR DELETE OUTPUTFILES AND ERRORFILES BUT MOVES THE JOB FILES DEPENDING ON STATUS
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


timeout_start = time.time()
time_last = timeout_start - 10.01 * 60  # I want it to start immediately
while (time.time() < timeout_start + Max_work_time) and ((len(os.listdir(Job_files_folder)) > 0)
                                                         or len(os.listdir(Output_files_folder)) > 0 or len(
            os.listdir(Error_files_folder)) > 0):
    if time.time() - time_last > 10 * 60:  # check every 10 min
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
            # os.remove(os.path.join(Job_files_folder, ff))
            # Moving it to the submitted jobs' folder. Will delete it after it is done
            os.replace(os.path.join(Job_files_folder, ff), os.path.join(Submitted_jobs_folder, ff))

        # ----------------------------------------------------------------------------------------------
        # Simultaneously, this loop looks at the submitted jobs which are done and moves them to the appropriate folder
        for js in os.listdir(Submitted_jobs_folder):
            target_id = ''.join(list(map(lambda x: x if x.isnumeric() else '', list(js.split('.')[0]))))  # will extract numerical part
            for fo in os.listdir(Output_files_folder):
                if target_id in fo:
                    with open(os.path.join(Output_files_folder, fo), 'r') as out_fil:
                        text_out = out_fil.read()
                    # If the job is still running, it will make itself a new file.
                    job_was_done = False
                    job_was_success = False
                    job_was_bad = False
                    # check that the job is finished and how:
                    if 'Successfully completed' in text_out:
                        job_was_done = True
                        job_was_success = True

                    if ('Exited with exit code' in text_out
                            or 'Exited.' in text_out
                            or ('Terminated' in text_out and 'Terminated at ' not in text_out)
                            or 'Killed' in text_out or 'killed' in text_out):
                        job_was_done = True
                        job_was_success = False
                        job_was_bad = True

                    # If done, figure out what is its fate:
                    if job_was_done:
                        job_filename = os.path.split(text_out[(text_out.index('Job <') + len('Job <')):text_out.index('> was submitted')])[1]
                        # Depending on the success, either delete the job, or move to failed jobs' folder:
                        if job_was_bad:
                            os.replace(os.path.join(Submitted_jobs_folder, job_filename), os.path.join(Failed_jobs_folder, job_filename))

                        if job_was_success:
                            os.remove(os.path.join(Submitted_jobs_folder, job_filename))
    else:
        time.sleep(30)  # sleep 30 seconds if the time to check did not yet come

# bsub -q physics-long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_pre_filter.py
# bsub -q long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_pre_filter.py
