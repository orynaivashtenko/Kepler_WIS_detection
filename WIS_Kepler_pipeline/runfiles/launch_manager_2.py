#!/usr/bin/env python3

import time
import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from runfiles_parameters import *
from file_operations import check_number_of_submitted_jobs
from pipeline_parameters import Run_root_folder

# ################################################################################################
# LAUNCH MANAGER WHICH DOES NOT MOVE OR DELETE ERRORFILES

Max_work_time = 6.9 * 24 * 60 * 60  # in seconds; the job in physics-short is killed after 7 days
Max_jobs_submitted = 4950  # Maximal number of jobs that we can have in bjobs at a certain time

# generate a random number and name the output file accordingly
# if "Successfully completed." in the output, then delete it

queues = Queues_check
# Take job files from the running list randomly to avoid two timeslides file running simultaneously

minutes_between_checks = 10.
timeout_start = time.time()
time_last = timeout_start - (minutes_between_checks + 0.01) * 60  # I want it to start immediately
while (time.time() < timeout_start + Max_work_time) and (len(os.listdir(Job_files_folder)) > 0):
    if time.time() - time_last > minutes_between_checks * 60:  # check every 10 min
        time_last = time.time()
        # wc = check_number_of_submitted_jobs(output_dir=Run_root_folder)
        wcs = check_number_of_submitted_jobs(output_dir=Run_root_folder, queues=queues, running=False)
        if hasattr(wcs, '__len__'):
            wc = sum(wcs)
        else:
            wc = wcs
        if wc < Max_jobs_submitted:
            for ff in os.listdir(Job_files_folder)[:Max_jobs_submitted - wc]:
                os.system('bsub < ' + os.path.join(Job_files_folder, ff))
                time.sleep(0.1)
                os.remove(os.path.join(Job_files_folder, ff))
    else:
        time.sleep(30)  # sleep 30 seconds if the time to check did not yet come

# bsub -q long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_2.py
# bsub -q risk -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_2.py


# ################################################################################################
# ################################################################################################

# Max_work_time = 6.9 * 24 * 60 * 60  # in seconds; the job in physics-short is killed after 7 days
# Max_jobs_submitted = 4900  # Maximal number of jobs that we can have in bjobs at a certain time
#
# # generate a random number and name the output file accordingly
# # if "Successfully completed." in the output, then delete it
#
# # queues = ['physics-short', 'new-short']
# queues = ['risk']
# # Take job files from the running list randomly to avoid two timeslides file running simultaneously
#
# timeout_start = time.time()
# time_counter = 0
# while (time.time() < timeout_start + Max_work_time) and (len(os.listdir(Job_files_folder)) > 0):
# # len(os.listdir(Job_files_folder)) > 0 or len(os.listdir(Running_jobs_folder)) > 0):
# # or len(os.listdir(Output_files_folder)) > 0 or len(os.listdir(Error_files_folder)) > 0):
#     if time.time() - timeout_start > time_counter * 5 * 60:  # check every 10 min
#         # wc = check_number_of_submitted_jobs(output_dir=Run_root_folder)
#         wcs = check_number_of_submitted_jobs(output_dir=Run_root_folder, queues=queues)
#         if hasattr(wcs, '__len__'):
#             wc = sum(wcs)
#         else:
#             wc = wcs
#         for ff in os.listdir(Job_files_folder)[:Max_jobs_submitted - wc]:
#             time.sleep(1.)
#             os.system('bsub < '+os.path.join(Job_files_folder, ff))
#             # os.replace(os.path.join(Job_files_folder, ff), os.path.join(Running_jobs_folder, ff))  # move the jobs to the running folder
#             os.remove(os.path.join(Job_files_folder, ff))
#         time_counter += 1
#
#     # simultaneously, this loop should take care of the outputs. maybe make that when it takes the output, it deletes the job file
#     for fo in os.listdir(Output_files_folder):
#         with open(os.path.join(Output_files_folder, fo), 'r') as out_fil:
#             text = out_fil.read()
#         if 'Exited with exit code' in text or 'Successfully completed' in text:  # check that the job is finished
#             job_filename = os.path.split(text[(text.index('Job <') + len('Job <')):text.index('> was submitted')])[1]  # take the job file name
#             target_id = ''.join(list(map(lambda x: x if x.isnumeric() else '', list(fo.split('.')[0]))))  # will extract numerical part
#             # if 'Successfully completed' not in text:
#             #     with open(os.path.join(Output_stars_folder, target_id + '_' + Star_full_output_filename_postfix), 'a') as fs:
#             #         print('FILE:  ' + fo, file=fs)
#             #         dtt = time.localtime()
#             #         print(str(dtt.tm_mday) + '.' + str(dtt.tm_mon) + '.' + str(dtt.tm_year) + ', ' + str(dtt.tm_hour) + ':' + str(
#             #             dtt.tm_min) + ':' + str(dtt.tm_sec), file=fs)
#             #         # print(file=fs)
#             #         print(text, file=fs)  # we append the output text to the general output of this star
#             #         print(file=fs)
#             #         print('########################################################################################################################################################################',
#             #               file=fs)
#             #     with open(Error_list_file, 'a') as elf:
#             #         print(job_filename, file=elf)
#             # os.remove(os.path.join(Output_files_folder, fo))
#             #
#             # # If it has some output, then it is finished, and let us look at the error
#             # fe = fo[:fo.index('output')] + 'error' + fo[fo.index('output') + len('output'):]
#             # try:
#             #     with open(os.path.join(Error_files_folder, fe)) as err_fil:
#             #         text_err = err_fil.read()
#             #         if len(text_err) > 10:  # just that it is not empty
#             #             with open(os.path.join(Output_stars_folder, target_id + '_' + Star_full_output_filename_postfix), 'a') as fs:
#             #                 print('FILE:  ' + fe, file=fs)
#             #                 dtt = time.localtime()
#             #                 print(str(dtt.tm_mday) + '.' + str(dtt.tm_mon) + '.' + str(dtt.tm_year) + ', ' + str(dtt.tm_hour) + ':' + str(
#             #                     dtt.tm_min) + ':' + str(dtt.tm_sec), file=fs)
#             #                 print(file=fs)
#             #                 print(text_err, file=fs)  # we append the output text to the general output of this star
#             #                 print(file=fs)
#             #                 print('########################################################################################################################################################################',
#             #                       file=fs)
#             #         os.remove(os.path.join(Error_files_folder, fe))
#             # except:
#             #     pass
#
#             # if job_filename in os.listdir(Running_jobs_folder):  # I don't know why, but for some reason files are not always there
#             #     os.remove(os.path.join(Running_jobs_folder, job_filename))  # remove the file which finished
#
#
# # bsub -q physics-long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_2.py
# # bsub -q long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_2.py
# # ################################################################################################
# # ################################################################################################
#
#
#
# # ################################################################################################
# # ################################################################################################
# # LAUNCH_MANAGER COPYING ERRORS AND OUTPUTS TO THE SPECIFIC STAR FOLDER
#
# #!/usr/bin/env python3
#
# import time
# import os
# os.environ["OMP_NUM_THREADS"] = "1"  #
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"  #
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# from runfiles_parameters import *
# from file_operations import check_number_of_submitted_jobs
# from pipeline_parameters import Run_root_folder
#
# Max_work_time = 6.9 * 24 * 60 * 60  # in seconds; the job in physics-short is killed after 7 days
# Max_jobs_submitted = 4900  # Maximal number of jobs that we can have in bjobs at a certain time
#
# # generate a random number and name the output file accordingly
# # if "Successfully completed." in the output, then delete it
#
# # queues = ['short', 'medium']

# # Take job files from the running list randomly to avoid two timeslides file running simultaneously
#
# timeout_start = time.time()
# time_counter = 0
# while (time.time() < timeout_start + Max_work_time) and ((len(os.listdir(Job_files_folder)) > 0)
#         or len(os.listdir(Output_files_folder)) > 0 or len(os.listdir(Error_files_folder)) > 0):
#     if time.time() - timeout_start > time_counter * 5 * 60:  # check every 10 min
#         # wc = check_number_of_submitted_jobs(output_dir=Run_root_folder)
#         wcs = check_number_of_submitted_jobs(output_dir=Run_root_folder, queues=queues)
#         if hasattr(wcs, '__len__'):
#             wc = sum(wcs)
#         else:
#             wc = wcs
#         for ff in os.listdir(Job_files_folder)[:Max_jobs_submitted - wc]:
#             time.sleep(1.)
#             os.system('bsub < '+os.path.join(Job_files_folder, ff))
#             os.remove(os.path.join(Job_files_folder, ff))
#         time_counter += 1
#
#     # simultaneously, this loop should take care of the outputs. maybe make that when it takes the output, it deletes the job file
#     for fo in os.listdir(Output_files_folder):
#         with open(os.path.join(Output_files_folder, fo), 'r') as out_fil:
#             text = out_fil.read()
#         if 'Exited with exit code' in text or 'Successfully completed' in text or 'Terminated' in text or 'Killed' in text:
#             # -> check that the job is finished
#             job_filename = os.path.split(text[(text.index('Job <') + len('Job <')):text.index('> was submitted')])[1]  # take the job file name
#             target_id = ''.join(list(map(lambda x: x if x.isnumeric() else '', list(fo.split('.')[0]))))  # will extract numerical part
#             if 'Exited with exit code' in text or 'Terminated' in text or 'Killed' in text:
#                 with open(os.path.join(Output_stars_folder, target_id[:4], target_id + '_' + Star_full_output_filename_postfix), 'a') as fs:
#                     print('FILE:  ' + fo, file=fs)
#                     dtt = time.localtime()
#                     print(str(dtt.tm_mday) + '.' + str(dtt.tm_mon) + '.' + str(dtt.tm_year) + ', ' + str(dtt.tm_hour) + ':' + str(
#                         dtt.tm_min) + ':' + str(dtt.tm_sec), file=fs)
#                     # print(file=fs)
#                     print(text, file=fs)  # we append the output text to the general output of this star
#                     print(file=fs)
#                     print('########################################################################################################################################################################',
#                           file=fs)
#                 with open(List_of_exited_jobs_file, 'a') as elf:
#                     print(job_filename, file=elf)
#             os.remove(os.path.join(Output_files_folder, fo))
#
#             # If it has some output, then it is finished, and let us look at the error
#             fe = fo[:fo.index('output')] + 'error' + fo[fo.index('output') + len('output'):]
#             try:
#                 with open(os.path.join(Error_files_folder, fe)) as err_fil:
#                     text_err = err_fil.read()
#                     if len(text_err) > 10:  # just that it is not empty
#                         with open(os.path.join(Output_stars_folder, target_id[:4],
#                                                target_id + '_' + Star_full_output_filename_postfix), 'a') as fs:
#                             print('FILE:  ' + fe, file=fs)
#                             dtt = time.localtime()
#                             print(str(dtt.tm_mday) + '.' + str(dtt.tm_mon) + '.' + str(dtt.tm_year) + ', ' + str(dtt.tm_hour) + ':' + str(
#                                 dtt.tm_min) + ':' + str(dtt.tm_sec), file=fs)
#                             print(file=fs)
#                             print(text_err, file=fs)  # we append the output text to the general output of this star
#                             print(file=fs)
#                             print('########################################################################################################################################################################',
#                                   file=fs)
#                     os.remove(os.path.join(Error_files_folder, fe))
#             except:
#                 pass
#
#             # if job_filename in os.listdir(Running_jobs_folder):  # I don't know why, but for some reason files are not always there
#             #     os.remove(os.path.join(Running_jobs_folder, job_filename))  # remove the file which finished
#
#
# # bsub -q physics-long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_2.py
# # bsub -q long -R "rusage[mem=500MB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_2.py
#
#
#
#
#
#
#
# # ################################################################################################


