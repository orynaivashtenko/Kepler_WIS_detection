#!/usr/bin/env python3

import os

from runfiles_parameters import (Run_jobs_outputs_folder, Job_files_folder, Submitted_jobs_folder,
                                 Failed_jobs_folder, Output_files_folder, Error_files_folder, Output_stars_folder,
                                 Error_list_file, List_of_failed_jobs_file, List_of_success_jobs_file)
from pipeline_parameters import Trigger_lists_folder, Problems_folder, Data_location


if not os.path.exists(Run_jobs_outputs_folder) or not os.path.isdir(Run_jobs_outputs_folder):
    os.makedirs(Run_jobs_outputs_folder)
if not os.path.exists(Trigger_lists_folder) or not os.path.isdir(Trigger_lists_folder):
    os.makedirs(Trigger_lists_folder)

if not os.path.exists(Error_list_file):
    os.remove(Error_list_file)
if not os.path.exists(List_of_failed_jobs_file):
    os.remove(List_of_failed_jobs_file)
if not os.path.exists(List_of_success_jobs_file):
    os.remove(List_of_success_jobs_file)

for folder in [Job_files_folder, Submitted_jobs_folder, Failed_jobs_folder, Output_files_folder, Error_files_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
    if len(os.listdir(folder)) != 0:
        os.system('rm ' + folder + '/*')

prefixes = [p for p in os.listdir(Data_location) if len(p) == 4 and os.path.isdir(os.path.join(Data_location, p))]
for prefixes_folder in [Output_stars_folder]:
    for fold in os.listdir(prefixes_folder):
        if os.path.isdir(os.path.join(prefixes_folder, fold)):
            if len(os.listdir(os.path.join(prefixes_folder, fold))) != 0:
                os.system('rm ' + os.path.join(prefixes_folder, fold) + '/*')
        else:
            os.remove(os.path.join(prefixes_folder, fold))
    for prefix in prefixes:
        if not os.path.exists(os.path.join(prefixes_folder, prefix)):
            os.makedirs(os.path.join(prefixes_folder, prefix))

# bsub -q risk -R "rusage[mem=1GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/prepare_run_jobs_folder.py
