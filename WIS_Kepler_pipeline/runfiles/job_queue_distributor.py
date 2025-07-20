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
from locations import Run_technical_folder

# ################################################################################################
# MANAGER WHICH MOVES JOBS BETWEEN QUEUES TO MAXIMIZE RUNNING
# ################################################################################################

Max_work_time = 6.9 * 24 * 60 * 60  # in seconds; the job in physics-short is killed after 7 days

queue_submit = 'physics-short'
queues_distribute = ['risk', 'short']  # order is important for the ratio: the first one will get fewer slots if needed

minutes_between_checks = 10.
timeout_start = time.time()
time_last = timeout_start - (minutes_between_checks + 0.01) * 60  # I want it to start immediately
while time.time() < timeout_start + Max_work_time:
    if time.time() - time_last > minutes_between_checks * 60:  # check every .. min
        time_last = time.time()
        wcs = check_number_of_submitted_jobs(output_dir=Run_technical_folder, queues=queue_submit, running=False, pending=True)
        if hasattr(wcs, '__len__'):
            wc = sum(wcs)
        else:
            wc = wcs
        if wc > 300:
            wcs_distr_run = [check_number_of_submitted_jobs(output_dir=Run_technical_folder, queues=queue, running=True, pending=False) for
                             queue in queues_distribute]
            wcs_distr_run = [sum(w) if hasattr(w, '__len__') else w for w in wcs_distr_run]
            wcs_distr_pen = [check_number_of_submitted_jobs(output_dir=Run_technical_folder, queues=queue, running=False, pending=True) for
                             queue in queues_distribute]
            wcs_distr_pen = [sum(w) if hasattr(w, '__len__') else w for w in wcs_distr_pen]
            queues_distribute_active = []
            for i in range(len(queues_distribute)):
                if wcs_distr_pen[i] > 0:
                    if (wcs_distr_run[i] / wcs_distr_pen[i] < 3 and wcs_distr_pen[i] > 1000) or wcs_distr_pen[i] > 5000:
                        pass
                else:
                    queues_distribute_active.append(queues_distribute[i])
            if len(queues_distribute_active) > 1:
                # If the first queue is more active, switch the order of the queues
                if wcs_distr_pen[0] == 0 and wcs_distr_run[0] > 0 and wcs_distr_pen[1] > 0:
                    queues_distribute_active = queues_distribute_active[::-1]
                    wcs_distr_run = wcs_distr_run[::-1]
                    wcs_distr_pen = wcs_distr_pen[::-1]
                if wcs_distr_run[0] < wcs_distr_run[1] / 3:
                    ratio = 4
                elif wcs_distr_run[0] < wcs_distr_run[1] / 2:
                    ratio = 3
                else:
                    ratio = 2
                text = """index=0  
for job_id in $(bjobs -pe -q """ + queue_submit + """ | awk '{print $1}'); do
    if [ -n "$job_id" ]; then
        if (( index % """ + str(ratio) + """ == 0 )); then
            bmod -q """ + queues_distribute_active[0] + """ "$job_id"  
        else
            bmod -q """ + queues_distribute_active[1] + """ "$job_id"  
        fi
        index=$((index + 1))
    fi
done
"""
                os.system(text)
            elif len(queues_distribute_active) == 1:
                text = """for job_id in $(bjobs -pe -q """ + queue_submit + """ | awk '{print $1}'); do
    if [ -n "$job_id" ]; then
        bmod -q """ + queues_distribute_active[0] + """ "$job_id" 
    fi
done
"""
                os.system(text)
            else:
                # If all the alternative queues are inactive, wait until the next check
                pass
        else:
            # If we ran out of jobs at all, break:
            wcs = check_number_of_submitted_jobs(output_dir=Run_technical_folder, queues=queue_submit, running=True, pending=False)
            if hasattr(wcs, '__len__'):
                wc = sum(wcs)
            else:
                wc = wcs
            if wc == 0:
                break
    else:
        time.sleep(30)  # sleep 30 seconds if the time to check did not yet come

# bsub -q long -sp 100 -R "rusage[mem=500MB]" /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/job_queue_distributor.py
# bsub -q physics-long -R "rusage[mem=500MB]" /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/job_queue_distributor.py
# bsub -q risk -R "rusage[mem=500MB]" /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/job_queue_distributor.py
