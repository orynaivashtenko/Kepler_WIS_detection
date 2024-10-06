#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from pipeline_parameters import *
from runfiles.runfiles_parameters import *
import file_operations as fop

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--runfile", type=str, nargs=1, help="whether main or scrambled runfile",
                    default="main")
# Runfile can be main or scrambled
args = parser.parse_args()
runfile = args.runfile
if isinstance(runfile, list):
    runfile = runfile[0]

if runfile == 'main':
    search_runfile_filename = Main_search_runfile_filename
else:
    search_runfile_filename = Main_search_scramble_runfile_filename

for prefix in os.listdir(Data_location):
    for target_id in os.listdir(os.path.join(Data_location, prefix)):
        fop.write_job_file(code_file=None,
                           code_filename=search_runfile_filename,
                           code_file_dir=Run_files_folder,
                           jobname=Main_search_job_filename_prefix + '_' + target_id + '.bsub',
                           job_file=None,
                           job_filename=Main_search_job_filename_prefix + '_' + target_id + '.bsub',
                           job_file_dir=Job_files_folder,
                           queue=Queue_submit,
                           mem_per_job_gb=Mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=Main_search_output_filename_prefix + '_' + target_id + '.out',
                           output_file_dir=Output_files_folder,
                           error_file=None,
                           error_filename=Main_search_error_filename_prefix + '_' + target_id + '.out',
                           error_file_dir=Error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', target_id]])

# bsub -q physics-short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_all_stars.py
# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_all_stars.py --runfile main
# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_all_stars.py --runfile scrambled
