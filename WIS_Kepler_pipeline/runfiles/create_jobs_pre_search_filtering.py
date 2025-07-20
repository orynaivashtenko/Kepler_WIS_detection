#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from pipeline_parameters import *
from runfiles.runfiles_parameters import *
import file_operations as fop

job_type_prefix = Pre_search_job_prefix


fop.arrange_job_folders(job_type_prefix=job_type_prefix,
                        jobs_folder=Job_files_folder,
                        output_files_folder=Output_files_folder,
                        error_files_folder=Error_files_folder,
                        jobs_monitor_folder=Jobs_monitor_folder,
                        use_last_digit=False, make_monitor_folder=False,
                        use_last_digit_monitor=False)

# Folders for this specific type of lob (main search):
job_files_folder = os.path.join(Job_files_folder, job_type_prefix)
output_files_folder = os.path.join(Output_files_folder, job_type_prefix)
error_files_folder = os.path.join(Error_files_folder, job_type_prefix)

for prefix in os.listdir(Data_location):
    if (len(prefix) == 4) and prefix.isnumeric():
        for last_digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            name_postfix = '_' + prefix + '_' + last_digit
            fop.write_job_file(code_file=None,
                               code_filename=Pre_search_runfile_filename,
                               code_file_dir=Run_files_folder,
                               jobname=Pre_search_job_prefix + name_postfix,
                               job_file=None,
                               job_filename=Pre_search_job_prefix + name_postfix + '.bsub',
                               job_file_dir=job_files_folder,
                               queue=Queue_submit,
                               mem_per_job_gb=Mem_per_job_pre_filtering_gb,
                               array_indices=None,
                               output_file=None,
                               output_filename=Pre_search_job_prefix + name_postfix + '.out',
                               output_file_dir=output_files_folder,
                               error_file=None,
                               error_filename=Pre_search_job_prefix + name_postfix + '.err',
                               error_file_dir=error_files_folder,
                               output_error_overwrite=False,
                               arg_jobind=False,
                               other_args=[['prefix', prefix], ['last_digit', last_digit]])

# bsub -q physics-short source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_pre_search_filtering.py
