# #!/usr/bin/env python3
#
# import os
# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# import file_operations as fop
# from pipeline_parameters import Data_location
#
# folder = '/home/labs/barakz/orynai/run_11/fixing_rates_mutliple_triggers'
# code_file = '/home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/fix_incorrect_integral_score_and_thresholding.py'
#
# for prefix in os.listdir(Data_location):
#     job_file = os.path.join(folder, 'fixing_' + prefix + '.bsub')
#     fop.write_job_file(
#         code_file=code_file, code_filename=None, code_file_dir=None,
#         jobname=prefix + '_fixing', job_file=job_file, job_filename=None, job_file_dir=None,
#         env_name='', queue='short', mem_per_job_gb='3.9', array_indices=None,
#         output_file=None, output_filename=prefix + '.out', output_file_dir=folder,
#         error_file=None, error_filename=prefix + '.err', error_file_dir=folder,
#         output_error_overwrite=True,
#         arg_jobind=False, other_args=[['prefix', prefix]])
#
#     os.system('bsub < ' + job_file)
#
# # bsub -q short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_jobs_for_prefixes.py
