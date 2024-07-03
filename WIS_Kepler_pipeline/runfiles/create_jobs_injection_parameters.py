#!/usr/bin/env python3

import os
import sys
# noinspection DuplicatedCode
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import Triggers_for_slides_injections_file
from runfiles_parameters import *
from trigger_class import Index_tid, Index_per

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Triggers_for_slides_injections_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file[0]

with open(trigger_list_file, 'r') as tlf:
    trigs = tlf.readlines()
trig_ids = [trig.split()[Index_tid] for trig in trigs]
periods = [float(trig.split()[Index_per]) for trig in trigs]

# noinspection DuplicatedCode
for ii in range(len(trig_ids)):
    tid = trig_ids[ii]
    per = periods[ii]
    fop.write_job_file(code_file=None,
                       code_filename=Injection_parameters_runfile_filename,
                       code_file_dir=Run_files_folder,
                       jobname=Injection_parameters_job_filename_prefix+'_'+tid+'.bsub',
                       job_file=None,
                       job_filename=Injection_parameters_job_filename_prefix+'_'+tid+'.bsub',
                       job_file_dir=Job_files_folder,
                       queue=Queue_submit,
                       mem_per_job_gb=Mem_per_job_gb,
                       array_indices=None,
                       output_file=None,
                       output_filename=Injection_parameters_output_filename_prefix+'_'+tid+'.out',
                       output_file_dir=Output_files_folder,
                       error_file=None,
                       error_filename=Injection_parameters_error_filename_prefix+'_'+tid+'.out',
                       error_file_dir=Error_files_folder,
                       output_error_overwrite=False,
                       arg_jobind=False,
                       other_args=[['target_id', tid], ['period', per]])


# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_injection_parameters.py
