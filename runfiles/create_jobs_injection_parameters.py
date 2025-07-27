#!/usr/bin/env python3

import os
import sys
# noinspection DuplicatedCode
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import Triggers_for_slides_injections_file
from runfiles_parameters import *
from trigger_functions import Index_tid, Index_per

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Triggers_for_slides_injections_file)
parser.add_argument("--tid_selected_list_file", type=str, nargs=1,
                    help="path to file where to take target ids of the triggers which are to be selected from the list",
                    default='')
args = parser.parse_args()
trigger_list_file = args.trigger_list_file
if isinstance(trigger_list_file, list):
    trigger_list_file = trigger_list_file[0]
tid_selected_list_file = args.tid_selected_list_file
if isinstance(tid_selected_list_file, list):
    tid_selected_list_file = tid_selected_list_file[0]

mem_per_job_gb = '7.9'

with open(trigger_list_file, 'r') as tlf:
    trigs = tlf.readlines()
trig_ids = [trig.split()[Index_tid] for trig in trigs]
periods = [float(trig.split()[Index_per]) for trig in trigs]

if tid_selected_list_file != '':
    with open(tid_selected_list_file, 'r') as tlf:
        tids_selected = tlf.readlines()
    tids_selected = [line.split()[0] for line in tids_selected]

    inds_selected = [i for i in range(len(trig_ids)) if trig_ids[i] in tids_selected]
    trig_ids = [trig_ids[i] for i in inds_selected]
    periods = [periods[i] for i in inds_selected]


job_type_prefix = Injection_parameters_job_prefix
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


# noinspection DuplicatedCode
for ii in range(len(trig_ids)):
    tid = trig_ids[ii]
    per = periods[ii]
    job_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                    target_id=tid, additional_params=())
    fop.write_job_file(code_file=None,
                       code_filename=Injection_parameters_runfile_filename,
                       code_file_dir=Run_files_folder,
                       jobname=job_name,
                       job_file=None,
                       job_filename=job_name+'.bsub',
                       job_file_dir=job_files_folder,
                       queue=Queue_submit,
                       mem_per_job_gb=mem_per_job_gb,
                       array_indices=None,
                       output_file=None,
                       output_filename=job_name+'.out',
                       output_file_dir=output_files_folder,
                       error_file=None,
                       error_filename=job_name+'.out',
                       error_file_dir=error_files_folder,
                       output_error_overwrite=False,
                       arg_jobind=False,
                       other_args=[['target_id', tid], ['period', per], ['job_name', job_name]])


# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_injection_parameters.py
# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_injection_parameters.py --trigger_list_file /home/projects/barakz/orynai/run_11/triggers_for_slides_injections.txt --tid_selected_list_file /home/projects/barakz/orynai/run_11/not_run_tids.txt
