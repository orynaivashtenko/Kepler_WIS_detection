#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from pipeline_parameters import *
from runfiles.runfiles_parameters import *
from locations import Good_tids_search_postfix, Pre_search_filtering_folder
import file_operations as fop

# Arranging global folders
job_type_prefix = Efficiency_job_prefix
jobs_folder_all_periods, output_files_folder_all_periods, error_files_folder_all_periods = fop.arrange_job_folders(
    job_type_prefix=job_type_prefix,
    jobs_folder=Job_files_folder,
    output_files_folder=Output_files_folder,
    error_files_folder=Error_files_folder,
    jobs_monitor_folder=Jobs_monitor_folder,
    make_monitor_folder=False,
    use_last_digit=False,
    return_folders=True)

Periods = np.arange(50., 500, 50)  # 30.,
if not os.path.isdir(Efficiency_output_folder):
    os.mkdir(Efficiency_output_folder, mode=0o777)

for period in Periods:
    str_period = str(int(np.round(period)))
    # Arranging folders for periods
    job_type_prefix = Efficiency_job_prefix + '_' + str_period
    job_files_folder_period = os.path.join(jobs_folder_all_periods, str_period)
    if not os.path.isdir(job_files_folder_period):
        os.mkdir(job_files_folder_period, mode=0o777)
    output_files_folder_period = os.path.join(output_files_folder_all_periods, str_period)
    if not os.path.isdir(output_files_folder_period):
        os.mkdir(output_files_folder_period, mode=0o777)
    error_files_folder_period = os.path.join(error_files_folder_all_periods, str_period)
    if not os.path.isdir(error_files_folder_period):
        os.mkdir(error_files_folder_period, mode=0o777)
    efficiency_output_folder = os.path.join(Efficiency_output_folder, str_period)
    if not os.path.isdir(efficiency_output_folder):
        os.mkdir(efficiency_output_folder, mode=0o777)

    for prefix in os.listdir(Data_location):
        if (len(prefix) == 4) and prefix.isnumeric():
            efficiency_output_folder_prefix = os.path.join(efficiency_output_folder, prefix)
            if not os.path.isdir(efficiency_output_folder_prefix):
                os.mkdir(efficiency_output_folder_prefix, mode=0o777)

            for last_digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                job_files_folder_period_last_digit = os.path.join(job_files_folder_period, last_digit)
                if not os.path.isdir(job_files_folder_period_last_digit):
                    os.mkdir(job_files_folder_period_last_digit, mode=0o777)
                output_files_folder_period_last_digit = os.path.join(output_files_folder_period, last_digit)
                if not os.path.isdir(output_files_folder_period_last_digit):
                    os.mkdir(output_files_folder_period_last_digit, mode=0o777)
                error_files_folder_period_last_digit = os.path.join(error_files_folder_period, last_digit)
                if not os.path.isdir(error_files_folder_period_last_digit):
                    os.mkdir(error_files_folder_period_last_digit, mode=0o777)
                efficiency_output_folder_prefix_last_digit = os.path.join(efficiency_output_folder_prefix, last_digit)
                if not os.path.isdir(efficiency_output_folder_prefix_last_digit):
                    os.mkdir(efficiency_output_folder_prefix_last_digit, mode=0o777)

            good_tids_prefix_file = os.path.join(Pre_search_filtering_folder, prefix + '_' + Good_tids_search_postfix)
            with open(good_tids_prefix_file, 'r') as f:
                target_ids = f.readlines()
            target_ids = list(set([tid.split()[0] for tid in target_ids]))

            for target_id in target_ids:
                job_name = fop.compose_job_name(job_type_prefix=job_type_prefix,
                                                target_id=target_id, additional_params=())
                fop.write_job_file(code_file=None,
                                   code_filename=Efficiency_runfile_filename,
                                   code_file_dir=Run_files_folder,
                                   jobname=job_name,
                                   job_file=None,
                                   job_filename=job_name,
                                   job_file_dir=os.path.join(job_files_folder_period, target_id[-1]),
                                   queue=Queue_submit,
                                   mem_per_job_gb='7.9',
                                   array_indices=None,
                                   output_file=None,
                                   output_filename=job_name + '.out',
                                   output_file_dir=os.path.join(output_files_folder_period, target_id[-1]),
                                   error_file=None,
                                   error_filename=job_name + '.err',
                                   error_file_dir=os.path.join(error_files_folder_period, target_id[-1]),
                                   output_error_overwrite=False,
                                   arg_jobind=False,
                                   other_args=[['target_id', target_id], ['period', period],
                                               ['job_name', job_name],
                                               ['saving_dir', os.path.join(efficiency_output_folder_prefix, target_id[-1])]])

# bsub -sp 100 -q physics-short source ~/.bashrc; conda activate planets_env_amd_epyc;  /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_efficiency.py
