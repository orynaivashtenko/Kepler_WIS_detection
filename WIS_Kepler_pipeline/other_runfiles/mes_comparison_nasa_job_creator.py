#!/usr/bin/env python3

import pickle
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import *

Max_nasa_mes_to_consider = 20.

Queue = 'physics-short'
Mem_per_job_gb = '3.9'

Mes_comparison_folder = os.path.join(Run_root_folder, 'mes_comparison')  # where the job files and errorfiles will be stored
Job_files_folder = os.path.join(Mes_comparison_folder, 'jobs')
Running_jobs_folder = os.path.join(Mes_comparison_folder, 'running_jobs')
Output_file = os.path.join(Mes_comparison_folder, 'output_file.out')
Error_file = os.path.join(Mes_comparison_folder, 'error_file.out')
Mes_comparison_job_filename_postfix = 'mes_comp'

target_ids_nasa_faint = []
periods_nasa_faint = []
with open(Koi_dict_file, 'rb') as koi_catalog_dict_file:
    koi_info_dict = pickle.load(koi_catalog_dict_file)

for i in range(len(koi_info_dict['kepid'])):
    if len(koi_info_dict['koi_max_mult_ev'][i]) == 0:  # some FPs do not have MES assigned
        if ((len(koi_info_dict['koi_max_mult_ev'][i]) > 0
             or np.abs(float(koi_info_dict['koi_max_mult_ev'][i])) <= Max_nasa_mes_to_consider)
                and 20 <= float(koi_info_dict['koi_period'][i]) <= 500):
            target_ids_nasa_faint.append(koi_info_dict['kepid'][i])
            periods_nasa_faint.append(float(koi_info_dict['koi_period'][i]))

for i in range(len(target_ids_nasa_faint)):
    target_id = target_ids_nasa_faint[i]
    period = periods_nasa_faint[i]
    fop.write_job_file(code_file=None,
                       code_filename='mes_comparison_nasa_runfile.py',
                       code_file_dir=os.path.dirname(__file__),
                       jobname=target_id+'_'+Mes_comparison_job_filename_postfix+'.bsub',
                       job_file=None,
                       job_filename=target_id+'_'+Mes_comparison_job_filename_postfix+'.bsub',
                       job_file_dir=Job_files_folder,
                       queue=Queue,
                       mem_per_job_gb=Mem_per_job_gb,
                       array_indices=None,
                       output_file=Output_file,
                       output_filename=None,
                       output_file_dir=None,
                       error_file=Error_file,
                       error_filename=None,
                       error_file_dir=None,
                       output_error_overwrite=False,
                       arg_jobind=False,
                       other_args=[['target_id', target_id], ['period', str(period)]])

# bsub -q physics-short -sp 100 /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/other_runfiles/mes_comparison_nasa_job_creator.py
