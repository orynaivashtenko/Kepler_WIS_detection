#!/usr/bin/env python3

##############################################
# delete_selected_jobs
##############################################

# import os
# import sys
#
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# from pipeline_parameters import *
# from runfiles_parameters import *
#
# # N_slides_total_for_now = N_deep_slides_total
# Snr_threshold_lower = 6.8  # this will produce of the order of 100000 jobs, which is enough for now...
# Snr_threshold_normal = 7.0
# Period_threshold_trash = 50.  # days
# Period_threshold_lower_snr = 100.  # days
#
# threshold_crossing_list_file = os.path.join(Threshold_crossing_lists_folder, Threshold_crossing_list_filename)
# # threshold_inv_crossing_list_file = os.path.join(Threshold_crossing_lists_folder, Threshold_inv_crossing_list_filename)
#
# with open(threshold_crossing_list_file, 'r') as trig_file:
#     lines = trig_file.readlines()
#
# trig_ids_to_delete = []
# # trig_snrs = []
# for i in range(len(lines)):
#     line = lines[i]
#     if line.strip()[0] != '#' and len(line.strip()) >= 9:
#         parameters = line.split()
#         if np.abs(float(parameters[1])) < Snr_threshold_normal:
#             if ((float(parameters[2]) < Period_threshold_trash)
#                     or (float(parameters[2]) > Period_threshold_lower_snr)):
#                 trig_ids_to_delete.append(parameters[0])
#         # reject too low periods, because they are dominated by trash from even lower periods and by 30 days artifacts
#         # also, we decided not to keep lower bar for long periods, because it is useless anyway, p_planet will be too small
#
# trig_ids_to_delete = list(set(trig_ids_to_delete))
# job_file_dir = Job_files_folder
#
# for tid in trig_ids_to_delete:
#     for job_filename in os.listdir(job_file_dir):
#         if (Injection_detection_job_filename_prefix in job_filename) and (tid in job_filename):
#             os.remove(os.path.join(job_file_dir, job_filename))


##############################################
# delete_selected_outputs
##############################################

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from pipeline_parameters import *

tid_to_run_file = '/home/labs/barakz/Collaboration-Kepler/orynai/tids_to_run_first_timeslides_runs_6_7_pastro_0_9.txt'
with open(tid_to_run_file, 'r') as ttrf:
    trig_ids = ttrf.readlines()
trig_ids = [tid.strip() for tid in trig_ids]

slides_mes_inv_filename_postfix = Slides_mes_inv_filename_postfix
slides_mes_filename_postfix = Slides_mes_filename_postfix
slides_pers_inv_filename_postfix = Slides_pers_inv_filename_postfix
slides_pers_filename_postfix = Slides_pers_filename_postfix
slides_first_times_inv_filename_postfix = Slides_first_times_inv_filename_postfix
slides_first_times_filename_postfix = Slides_first_times_filename_postfix

Output_location_wexac = '/home/labs/barakz/Collaboration-Kepler/orynai/output_run_7'
# noinspection DuplicatedCode
for tid in trig_ids:
    output_location = os.path.join(Output_location_wexac, tid[:4], tid)

    fls = os.listdir(output_location)
    for fl in fls:
        if tid + '_' + slides_mes_inv_filename_postfix + '_' in fl:
            os.remove(os.path.join(output_location, fl))
        elif tid + '_' + slides_mes_filename_postfix + '_' in fl:
            os.remove(os.path.join(output_location, fl))
        elif tid + '_' + slides_pers_inv_filename_postfix + '_' in fl:
            os.remove(os.path.join(output_location, fl))
        elif tid + '_' + slides_pers_filename_postfix + '_' in fl:
            os.remove(os.path.join(output_location, fl))
        elif tid + '_' + slides_first_times_inv_filename_postfix + '_' in fl:
            os.remove(os.path.join(output_location, fl))
        elif tid + '_' + slides_first_times_filename_postfix + '_' in fl:
            os.remove(os.path.join(output_location, fl))

# bsub -q physics-short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/delete_selected_jobs.py
