#!/usr/bin/env python3

import os
import sys
# noinspection DuplicatedCode
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import *
from runfiles_parameters import *

tid_to_run_file = '/home/labs/barakz/orynai/' + Run_name + '/threshold_crossing_kid_info.txt'
with open(tid_to_run_file, 'r') as ttrf:
    trig_ids = ttrf.readlines()
trig_ids = [tid.split()[0] for tid in trig_ids]

Output_location_wexac = '/home/labs/barakz/Collaboration-Kepler/orynai/output_' + Run_name
for tid in trig_ids:
    for i in np.arange(int(N_injections_total // N_injections_chunk)  # how many times we need to repeat
                       + int(bool(N_injections_total % N_injections_chunk))):
        fop.write_job_file(code_file=None,
                           code_filename=Injection_detection_runfile_filename,
                           code_file_dir=Run_files_folder,
                           jobname=Injection_detection_job_filename_prefix+'_'+tid+'_'+str(i)+'.bsub',
                           job_file=None,
                           job_filename=Injection_detection_job_filename_prefix+'_'+tid+'_'+str(i)+'.bsub',
                           job_file_dir=Job_files_folder,
                           queue=Queue,
                           mem_per_job_gb=Mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=Injection_detection_output_filename_prefix+'_'+tid+'_'+str(i)+'.out',
                           output_file_dir=Output_files_folder,
                           error_file=None,
                           error_filename=Injection_detection_error_filename_prefix+'_'+tid+'_'+str(i)+'.out',
                           error_file_dir=Error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid], ['first_injection_load_parameters_index', i]])


# bsub -q risk /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_runfiles_injections_detection.py
