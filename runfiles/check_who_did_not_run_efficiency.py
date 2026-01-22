#!/usr/bin/env python3

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from pipeline_parameters import *
from runfiles.runfiles_parameters import *
from locations import Good_tids_search_postfix, Pre_search_filtering_folder
import file_operations as fop

Efficiency_output_dir = '/home/projects/barakz/Collaboration-Kepler/orynai/run_15/efficiency_outputs_run_15'
Jobs_dir = '/home/projects/barakz/orynai/run_15/hpc/jobs/efficiency'
Periods = np.arange(50., 500, 50)  # 30.,

# jobs_not_run_scramble_all_periods = []
# jobs_not_run_injections_all_periods = []
# for period in Periods:
#     str_period = str(int(np.round(period)))
#     # Arranging folders for periods
#     efficiency_output_dir_period = os.path.join(Efficiency_output_dir, str_period)
#
#     tids_not_run_scramble_period = []
#     tids_not_run_injections_period = []
#     for prefix in os.listdir(efficiency_output_dir_period):
#         if (len(prefix) == 4) and prefix.isnumeric():
#             tids_output_scramble = []
#             tids_output_injections = []
#             efficiency_output_dir_period_prefix = os.path.join(efficiency_output_dir_period, prefix)
#             for last_digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
#                 efficiency_output_dir_period_prefix_last_digit = os.path.join(efficiency_output_dir_period_prefix, last_digit)
#                 tids_output_scramble_last_digit = [f[:9] for f in os.listdir(efficiency_output_dir_period_prefix_last_digit) if
#                                                    f.endswith('_scrambles.json')]
#                 tids_output_injections_last_digit = [f[:9] for f in os.listdir(efficiency_output_dir_period_prefix_last_digit) if
#                                                      f.endswith('_injections.json')]
#                 tids_output_scramble += tids_output_scramble_last_digit
#                 tids_output_injections += tids_output_injections_last_digit
#
#             good_tids_prefix_file = os.path.join(Pre_search_filtering_folder, prefix + '_' + Good_tids_search_postfix)
#             with open(good_tids_prefix_file, 'r') as f:
#                 target_ids = f.readlines()
#             target_ids = [str(tid.strip()) for tid in target_ids]
#             tids_not_run_scramble_prefix = list(set(target_ids) - set(tids_output_scramble))
#             tids_not_run_injections_prefix = list(set(target_ids) - set(tids_output_injections))
#
#             tids_not_run_scramble_period += tids_not_run_scramble_prefix
#             tids_not_run_injections_period += tids_not_run_injections_prefix
#
#     # jobs_not_run_period_scramble = []
#     for tid in tids_not_run_scramble_period:
#         job_filename = 'efficiency_' + str_period + '_' + tid + '.bsub'
#         job_file = os.path.join(Jobs_dir, str_period, tid[-1], job_filename)
#         # jobs_not_run_period_scramble.append('bsub -R "rusage[mem=4GB]" ~/.bashrc; conda activate planets_env_amd_epyc; ' + job_file)
#         new_job_file = '/home/projects/barakz/orynai/run_15/tids_not_run_efficiency/jobs_to_run_scramble/' + job_filename
#         os.system('cp ' + job_file + ' ' + new_job_file)
#     # jobs_not_run_period_injections = []
#     for tid in tids_not_run_injections_period:
#         job_filename = 'efficiency_' + str_period + '_' + tid + '.bsub'
#         job_file = os.path.join(Jobs_dir, str_period, tid[-1], 'efficiency_' + str_period + '_' + tid + '.bsub')
#         new_job_file = '/home/projects/barakz/orynai/run_15/tids_not_run_efficiency/jobs_to_run_injections/' + job_filename
#         os.system('cp ' + job_file + ' ' + new_job_file)
#         # jobs_not_run_period_injections.append('bsub -R "rusage[mem=4GB]" ~/.bashrc; conda activate planets_env_amd_epyc; ' + job_file)
#     # jobs_not_run_scramble_all_periods += jobs_not_run_period_scramble
#     # jobs_not_run_injections_all_periods += jobs_not_run_period_injections
#
# # with open('/home/projects/barakz/orynai/run_15/tids_not_run_efficiency/jobs_not_run_efficiency_scramble.txt', 'w'
# #           ) as f:
# #     print(*jobs_not_run_scramble_all_periods, file=f, sep='\n')
# # with open('/home/projects/barakz/orynai/run_15/tids_not_run_efficiency/jobs_not_run_efficiency_injections.txt',
# #           'w') as f:
# #     print(*jobs_not_run_injections_all_periods, file=f, sep='\n')


for period in Periods:
    str_period = str(int(np.round(period)))
    with open('/home/projects/barakz/Collaboration-Kepler/orynai/run_15/jobs_not_run_efficiency_injections.txt', 'r') as file:
        jobs_not_run_injections = file.readlines()

    tids_not_run_injections = [jobs_not_run_injections[i][-15:-6] for i in range(len(jobs_not_run_injections)) if '_' + str_period + '_' in jobs_not_run_injections[i][-20:-15]]

    with open('/home/projects/barakz/Collaboration-Kepler/orynai/run_15/jobs_not_run_efficiency_scramble.txt', 'r') as file:
        jobs_not_run_scramble = file.readlines()
    tids_not_run_scramble = [jobs_not_run_scramble[i][-15:-6] for i in range(len(jobs_not_run_scramble)) if '_' + str_period + '_' in jobs_not_run_scramble[i][-20:-15]]

    tids_to_add_injections = list(set(set(tids_not_run_injections) - set(tids_not_run_scramble)))

    for tid in tids_to_add_injections:
        job_filename = 'efficiency_' + str_period + '_' + tid + '.bsub'
        job_file = os.path.join(Jobs_dir, str_period, tid[-1], 'efficiency_' + str_period + '_' + tid + '.bsub')
        new_job_file = '/home/projects/barakz/orynai/run_15/tids_not_run_efficiency/jobs_to_run_injections/' + job_filename
        os.system('cp ' + job_file + ' ' + new_job_file)

# bsub -sp 100 -R "rusage[mem=4G]" ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/check_who_did_not_run_efficiency.py
