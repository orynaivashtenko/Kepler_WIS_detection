#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pipeline_parameters import Run_name


Run_files_folder = os.path.dirname(__file__)
Core_folder = os.path.join('/home/labs/barakz/orynai', Run_name)
Run_jobs_outputs_folder = os.path.join(Core_folder, 'hpc')

Job_files_folder = os.path.join(Run_jobs_outputs_folder, 'jobs')
Submitted_jobs_folder = os.path.join(Run_jobs_outputs_folder, 'submitted_jobs')
Failed_jobs_folder = os.path.join(Run_jobs_outputs_folder, 'failed_jobs')
Output_files_folder = os.path.join(Run_jobs_outputs_folder, 'outputfiles')
Error_files_folder = os.path.join(Run_jobs_outputs_folder, 'errorfiles')
Output_stars_folder = os.path.join(Run_jobs_outputs_folder, 'outputs_stars')
Star_full_output_filename_postfix = 'full_output.txt'

Error_list_file = os.path.join(Run_jobs_outputs_folder, 'error_list.txt')
List_of_failed_jobs_file = os.path.join(Run_jobs_outputs_folder, 'failed_jobs_list.txt')
List_of_success_jobs_file = os.path.join(Run_jobs_outputs_folder, 'success_jobs_list.txt')

# Queue_submit = 'physics-short'  # 'risk'
Queue_submit = 'risk'  # '"short medium risk"'  # dynamic queue submission
Queues_check = ['all']  # in dynamic queue submission, im not sure what it does to the names so i just check all
# ['short', 'medium']  # check these queues in launch manager
Mem_per_job_gb = '3.9'

Slides_runfile_filename = 'runfile_timeslides.py'
Slides_clean_runfile_filename = 'runfile_timeslides_clean.py'
Injection_parameters_runfile_filename = 'runfile_injection_parameter_generation.py'
Injection_detection_runfile_filename = 'runfile_injection_detection.py'
Main_search_runfile_filename = 'runfile_main_search.py'
Main_search_scramble_runfile_filename = 'runfile_main_search_scramble.py'
Short_periodicity_code_filename = 'runfile_short_periodicity_search.py'
Plotting_runfile_filename = 'runfile_plotting_results.py'
Pre_slides_tce_veto_code_filename = 'runfile_pre_slides_tce_veto.py'

Main_search_job_filename_prefix = 'search'
Main_search_output_filename_prefix = 'output_search'
Main_search_error_filename_prefix = 'error_search'

Short_periodicity_job_filename_prefix = 'short_periodicity'
Short_periodicity_output_filename_prefix = 'output_short_periodicity'
Short_periodicity_error_filename_prefix = 'error_short_periodicity'

Slides_job_filename_prefix = 'slides'
Slides_output_filename_prefix = 'output_timeslides'
Slides_error_filename_prefix = 'error_timeslides'

Slides_clean_job_filename_prefix = 'slides_clean'
Slides_clean_output_filename_prefix = 'output_timeslides_clean'
Slides_clean_error_filename_prefix = 'error_timeslides_clean'

Injection_parameters_job_filename_prefix = 'injection_parameters'
Injection_parameters_output_filename_prefix = 'output_injection_parameters'
Injection_parameters_error_filename_prefix = 'error_injection_parameters'

Injection_detection_job_filename_prefix = 'injection_detection'
Injection_detection_output_filename_prefix = 'output_injection_detection'
Injection_detection_error_filename_prefix = 'error_injection_detection'

Plotting_job_filename_prefix = 'plotting'
Plotting_output_filename_prefix = 'output_plotting'
Plotting_error_filename_prefix = 'error_plotting'

Pre_slides_tce_veto_job_filename_prefix = 'pre_slides_tce_veto'
Pre_slides_tce_veto_output_filename_prefix = 'pre_slides_tce_veto'
Pre_slides_tce_veto_error_filename_prefix = 'pre_slides_tce_veto'
