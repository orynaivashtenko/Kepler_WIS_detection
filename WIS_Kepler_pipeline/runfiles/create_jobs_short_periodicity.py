#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from runfiles_parameters import *
from pipeline_parameters import Triggers_prefiltered_file
from trigger_functions import Index_tid

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trigger_list_file", type=str, nargs=1, help="path to file where to take triggers",
                    default=Triggers_prefiltered_file)
args = parser.parse_args()
trigger_list_file = args.trigger_list_file
if isinstance(trigger_list_file, list):
    trigger_list_file = trigger_list_file[0]

with open(trigger_list_file, 'r') as tlf:
    trig_ids = tlf.readlines()
trig_ids = [tid.split()[Index_tid] for tid in trig_ids]
trig_ids = list(set(trig_ids))  # remove duplicates

for tid in trig_ids:
    # noinspection DuplicatedCode
    fop.write_job_file(code_file=None,
                       code_filename=Short_periodicity_code_filename,
                       code_file_dir=Run_files_folder,
                       jobname=Short_periodicity_job_filename_prefix + '_' + tid + '.bsub',
                       job_file=None,
                       job_filename=Short_periodicity_job_filename_prefix + '_' + tid + '.bsub',
                       job_file_dir=Job_files_folder,
                       queue=Queue_submit,
                       mem_per_job_gb=Mem_per_job_gb,
                       array_indices=None,
                       output_file=None,
                       output_filename=Short_periodicity_output_filename_prefix + '_' + tid + '.out',
                       output_file_dir=Output_files_folder,
                       error_file=None,
                       error_filename=Short_periodicity_error_filename_prefix + '_' + tid + '.err',
                       error_file_dir=Error_files_folder,
                       output_error_overwrite=False,
                       arg_jobind=False,
                       other_args=[['target_id', tid]])


# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_short_periodicity.py
# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_short_periodicity.py --trigger_list_file /home/labs/barakz/orynai/run_11/triggers/triggers_prefiltered.txt

