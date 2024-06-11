#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from runfiles_parameters import *

# Run_output_directory = '/home/labs/barakz/Collaboration-Kepler/orynai/output_run_11'
# Slidable_triggers_file = '/home/labs/barakz/orynai/run_11/slidable_triggers_run_11.txt'
# with open(Slidable_triggers_file, 'r') as fil:
#     lines = fil.readlines()
# tids = [line.strip().split()[0] for line in lines]

output_filename_prefix = 'out_short_periodicity'
error_filename_prefix = 'err_short_periodicity'
code_filename = 'short_periodicity_search.py'

for tid in ['011137065', '007605258', '007594395', '003840173']:
    # noinspection DuplicatedCode
    fop.write_job_file(code_file=None,
                       code_filename=code_filename,
                       code_file_dir=Run_files_folder,
                       jobname='short_periodicity_' + tid + '.bsub',
                       job_file=None,
                       job_filename='short_periodicity_' + tid + '.bsub',
                       job_file_dir=Job_files_folder,
                       queue=Queue,
                       mem_per_job_gb=Mem_per_job_gb,
                       array_indices=None,
                       output_file=None,
                       output_filename=output_filename_prefix + '_' + tid + '.out',
                       output_file_dir=Output_files_folder,
                       error_file=None,
                       error_filename=error_filename_prefix + '_' + tid + '.err',
                       error_file_dir=Error_files_folder,
                       output_error_overwrite=False,
                       arg_jobind=False,
                       other_args=[['target_id', tid]])

# # noinspection DuplicatedCode
# for tid in tids:
#     output_dir_tid = os.path.join(Run_output_directory, tid[:4], tid)
#     files_short_periodicity = [ff for ff in os.listdir(output_dir_tid) if 'best_periodic_scores_10_30' in ff]
#     if len(files_short_periodicity) > 0:
#         pass
#         # file_short_periodicity = os.path.join(output_dir_tid, files_short_periodicity[0])
#         # os.system('mv ' + file_short_periodicity + ' ' + os.path.join(output_dir_tid, tid + '_best_periodic_scores_10_30.npy'))
#     else:
#         if tid not in ['011137065', '007605258', '007594395', '003840173']:
#             # noinspection DuplicatedCode
#             fop.write_job_file(code_file=None,
#                                code_filename=code_filename,
#                                code_file_dir=Run_files_folder,
#                                jobname='short_periodicity_' + tid + '.bsub',
#                                job_file=None,
#                                job_filename='short_periodicity_' + tid + '.bsub',
#                                job_file_dir=Job_files_folder,
#                                queue=Queue,
#                                mem_per_job_gb=Mem_per_job_gb,
#                                array_indices=None,
#                                output_file=None,
#                                output_filename=output_filename_prefix + '_' + tid + '.out',
#                                output_file_dir=Output_files_folder,
#                                error_file=None,
#                                error_filename=error_filename_prefix + '_' + tid + '.err',
#                                error_file_dir=Error_files_folder,
#                                output_error_overwrite=False,
#                                arg_jobind=False,
#                                other_args=[['target_id', tid]])

# bsub -q risk /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_runfiles_short_periodicity.py

