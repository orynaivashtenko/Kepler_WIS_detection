#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from runfiles.runfiles_parameters import *
import pipeline_parameters as pp
import file_operations as fop

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tids_list_file", type=str, nargs=1, help="path to file where to take tids",
                    default=os.path.join(pp.Run_technical_folder, pp.Tids_that_did_not_run_filename))
parser.add_argument("--runfile", type=str, nargs=1, help="whether main or alternative runfile (differs by output location)",
                    default="main")
args = parser.parse_args()
tids_list_file = args.tids_list_file
runfile = args.runfile
if isinstance(tids_list_file, list):
    tids_list_file = tids_list_file[0]
if isinstance(runfile, list):
    runfile = runfile[0]

if runfile == 'main':
    main_search_runfile_filename = Main_search_runfile_filename
else:
    main_search_runfile_filename = Main_search_scramble_runfile_filename

with open(tids_list_file, 'r') as ttrf:
    targ_ids = ttrf.readlines()
targ_ids = [tid.split()[0] for tid in targ_ids]

for target_id in targ_ids:
    fop.write_job_file(code_file=None,
                       code_filename=main_search_runfile_filename,
                       code_file_dir=Run_files_folder,
                       jobname=target_id + '_' + Main_search_job_filename_prefix + '.bsub',
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

# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_some_stars.py
# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_some_stars.py --tids_list_file /home/labs/barakz/orynai/run_11/not_run_tids.txt --runfile alternative
# bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_some_stars.py --tids_list_file /home/labs/barakz/orynai/run_11/not_run_tids_scrambled_run.txt --runfile scrambled
