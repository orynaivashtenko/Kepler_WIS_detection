#!/usr/bin/env python3

import os
import sys
# noinspection DuplicatedCode
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import np, Triggers_for_slides_injections_file, N_injections_total, N_injections_chunk
from runfiles_parameters import *
from trigger_functions import Index_tid

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Triggers_for_slides_injections_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file
if isinstance(trigger_list_file, list):
    trigger_list_file = trigger_list_file[0]

with open(trigger_list_file, 'r') as tlf:
    trig_ids = tlf.readlines()
trig_ids = [tid.split()[Index_tid] for tid in trig_ids]

job_type_prefix = Injection_detection_job_prefix
fop.arrange_job_folders(job_type_prefix=job_type_prefix,
                        jobs_folder=Job_files_folder,
                        output_files_folder=Output_files_folder,
                        error_files_folder=Error_files_folder,
                        jobs_monitor_folder=Jobs_monitor_folder,
                        use_last_digit=False)
# -> do not use last digit here because it is only for triggers, and they are not that many

# Folders for this specific type of lob (main search):
job_files_folder = os.path.join(Job_files_folder, job_type_prefix)
output_files_folder = os.path.join(Output_files_folder, job_type_prefix)
error_files_folder = os.path.join(Error_files_folder, job_type_prefix)


for tid in trig_ids:
    for i in np.arange(int(N_injections_total // N_injections_chunk)  # how many times we need to repeat
                       + int(bool(N_injections_total % N_injections_chunk))):
        job_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                        target_id=tid, additional_params=(str(i),))
        out_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                        target_id=tid, additional_params=())
        fop.write_job_file(code_file=None,
                           code_filename=Injection_detection_runfile_filename,
                           code_file_dir=Run_files_folder,
                           jobname=job_name,
                           job_file=None,
                           job_filename=job_name+'.bsub',
                           job_file_dir=job_files_folder,
                           queue=Queue_submit,
                           mem_per_job_gb=Mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=out_name+'.out',
                           output_file_dir=output_files_folder,
                           error_file=None,
                           error_filename=out_name+'.err',
                           error_file_dir=tid[-1],
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid], ['first_injection_load_parameters_index', i],
                                       ['job_name', job_name]])


# bsub -q short /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_injections_detection.py
