#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pipeline_parameters import Run_name


Run_files_folder = os.path.dirname(__file__)
Run_foldername = Run_name
Run_jobs_outputs_folder = '/home/labs/barakz/orynai'

Job_files_folder = Run_jobs_outputs_folder + '/' + str(Run_foldername) + '/jobs'
Future_job_files_folder = Run_jobs_outputs_folder + '/' + str(Run_foldername) + '/future_jobs'
Running_jobs_folder = Run_jobs_outputs_folder + '/' + str(Run_foldername) + '/running_jobs'
Output_files_folder = Run_jobs_outputs_folder + '/' + str(Run_foldername) + '/outputfiles'
Error_files_folder = Run_jobs_outputs_folder + '/' + str(Run_foldername) + '/errorfiles'
Output_stars_folder = Run_jobs_outputs_folder + '/' + str(Run_foldername) + '/outputs_stars'
Star_full_output_filename_postfix = 'full_output.txt'

Error_list_file = Run_jobs_outputs_folder + '/' + str(Run_foldername) + '/error_list.txt'

# Queue = 'physics-short'
Queue = 'risk'
Mem_per_job_gb = '3.9'

Slides_runfile_filename = 'timeslides_runfile.py'
Injection_parameters_runfile_filename = 'injection_parameter_generation_runfile.py'
Injection_detection_runfile_filename = 'injection_detection_runfile.py'
Main_search_runfile_filename = 'main_search_runfile.py'
Plotting_runfile_filename = 'plotting_results_runfile.py'
Target_pixels_veto_runfile_filename = 'target_pixels_veto_runfile.py'

Main_search_job_filename_prefix = 'search'
Main_search_output_filename_prefix = 'output_search'
Main_search_error_filename_prefix = 'error_search'

Slides_job_filename_prefix = 'slides'
Slides_output_filename_prefix = 'output_timeslides'
Slides_error_filename_prefix = 'error_timeslides'

Injection_parameters_job_filename_prefix = 'injection_parameters'
Injection_parameters_output_filename_prefix = 'output_injection_parameters'
Injection_parameters_error_filename_prefix = 'error_injection_parameters'

Injection_detection_job_filename_prefix = 'injection_detection'
Injection_detection_output_filename_prefix = 'output_injection_detection'
Injection_detection_error_filename_prefix = 'error_injection_detection'

Plotting_job_filename_prefix = 'plotting'
Plotting_output_filename_prefix = 'output_plotting'
Plotting_error_filename_prefix = 'error_plotting'

Target_pixels_veto_job_filename_prefix = 'targ_pix_veto'
Target_pixels_veto_output_filename_prefix = 'output_targ_pix_veto'
Target_pixels_veto_error_filename_prefix = 'error_targ_pix_veto'
