#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from pipeline_parameters import *
from runfiles.runfiles_parameters import *
from locations import Good_tids_search_postfix, Pre_search_filtering_folder
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
    job_type_prefix = Main_search_job_prefix

else:
    search_runfile_filename = Main_search_scramble_runfile_filename
    job_type_prefix = Main_search_scramble_job_prefix


for folder in [Run_outputs_folder, Output_location,
               Triggers_plots_location, Trigger_lists_folder, Problems_folder]:
    if not os.path.isdir(folder):
        os.mkdir(folder, mode=0o777)


fop.arrange_job_folders(job_type_prefix=job_type_prefix,
                        jobs_folder=Job_files_folder,
                        output_files_folder=Output_files_folder,
                        error_files_folder=Error_files_folder,
                        jobs_monitor_folder=Jobs_monitor_folder,
                        use_last_digit=True)

# Folders for this specific type of lob (main search):
job_files_folder = os.path.join(Job_files_folder, job_type_prefix)
output_files_folder = os.path.join(Output_files_folder, job_type_prefix)
error_files_folder = os.path.join(Error_files_folder, job_type_prefix)

for prefix in os.listdir(Data_location):
    output_location_prefix = os.path.join(Output_location, prefix)
    if not os.path.isdir(output_location_prefix):
        os.mkdir(output_location_prefix, mode=0o777)
    problems_location_prefix = os.path.join(Problems_folder, prefix)
    if not os.path.isdir(problems_location_prefix):
        os.mkdir(problems_location_prefix, mode=0o777)

    if (len(prefix) == 4) and prefix.isnumeric():
        good_tids_prefix_file = os.path.join(Pre_search_filtering_folder, prefix + '_' + Good_tids_search_postfix)
        with open(good_tids_prefix_file, 'r') as f:
            target_ids = f.readlines()
        target_ids = list(set([tid.split()[0] for tid in target_ids]))

        for target_id in target_ids:
            output_location_tid = os.path.join(Output_location, target_id[:4], target_id)
            if not os.path.isdir(output_location_tid):
                os.mkdir(output_location_tid, mode=0o777)

            job_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                            target_id=target_id, additional_params=())
            fop.write_job_file(code_file=None,
                               code_filename=search_runfile_filename,
                               code_file_dir=Run_files_folder,
                               jobname=job_name,
                               job_file=None,
                               job_filename=job_name,
                               job_file_dir=os.path.join(job_files_folder, target_id[-1]),
                               queue=Queue_submit_main_search,
                               mem_per_job_gb=Mem_per_job_gb,
                               array_indices=None,
                               output_file=None,
                               output_filename=job_name + '.out',
                               output_file_dir=os.path.join(output_files_folder, target_id[-1]),
                               error_file=None,
                               error_filename=job_name + '.err',
                               error_file_dir=os.path.join(error_files_folder, target_id[-1]),
                               output_error_overwrite=False,
                               arg_jobind=False,
                               other_args=[['target_id', target_id], ['job_name', job_name]])

# bsub -q physics-short source ~/.bashrc; conda activate planets_env_amd_epyc;  /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_all_stars.py
# bsub -q physics-short source ~/.bashrc; conda activate planets_env_amd_epyc;  /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_all_stars.py --runfile main
# bsub -q short source ~/.bashrc; conda activate planets_env_amd_epyc;  /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_main_search_all_stars.py --runfile scrambled
