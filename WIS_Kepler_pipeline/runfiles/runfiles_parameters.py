#!/usr/bin/env python3

import os

Run_files_folder = os.path.dirname(__file__)

Job_files_folder = '/home/labs/barakz/orynai/run_6/jobs'
Future_job_files_folder = '/home/labs/barakz/orynai/run_6/future_jobs'
Running_jobs_folder = '/home/labs/barakz/orynai/run_6/running_jobs'
Output_files_folder = '/home/labs/barakz/orynai/run_6/outputfiles'
Error_files_folder = '/home/labs/barakz/orynai/run_6/errorfiles'
Output_stars_folder = '/home/labs/barakz/orynai/run_6/outputs_stars'
Star_full_output_filename_postfix = 'full_output.txt'

Error_list_file = '/home/labs/barakz/orynai/run_6/error_list.txt'

Queue = 'physics-short'
Mem_per_job_gb = '3.9'

Slides_runfile_filename = 'deep_timeslides_runfile.py'
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
