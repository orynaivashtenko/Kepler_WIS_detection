#!/usr/bin/env python3

import time
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from runfiles_parameters import *
from file_operations import check_number_of_submitted_jobs
from pipeline_parameters import Threshold_crossing_lists_folder

Max_work_time = 6.9 * 24 * 60 * 60  # in seconds; the job in physics-short is killed after 7 days
Max_jobs_submitted = 4900  # Maximal number of jobs that we can have in bjobs at a certain time

# generate a random number and name the output file accordingly
# if "Successfully completed." in the output, then delete it


# Take job files from the running list randomly to avoid two timeslides file running simultaneously

timeout_start = time.time()
time_counter = 0
while (time.time() < timeout_start + Max_work_time) and (
        len(os.listdir(Job_files_folder)) > 0 or len(os.listdir(Running_jobs_folder)) > 0
        or len(os.listdir(Output_files_folder)) > 0 or len(os.listdir(Error_files_folder)) > 0):
    if time.time() - timeout_start > time_counter * 10 * 60:  # check every 10 min
        wc = check_number_of_submitted_jobs(output_dir=Threshold_crossing_lists_folder)
        for ff in os.listdir(Job_files_folder)[:Max_jobs_submitted - wc]:
            os.replace(os.path.join(Job_files_folder, ff), os.path.join(Running_jobs_folder, ff))  # move the jobs to the running folder
            os.system('bsub < '+os.path.join(Running_jobs_folder, ff))
        time_counter += 1

    # simultaneously, this loop should take care of the outputs. maybe make that when it takes the output, it deletes the job file
    for fo in os.listdir(Output_files_folder):
        with open(os.path.join(Output_files_folder, fo), 'r') as out_fil:
            text = out_fil.read()
        #os.remove(os.path.join(Output_files_folder, fo))  # this may be redundant if the file is empty
        with open(os.path.join(Output_files_folder, fo), 'w') as out_fil:  # instead of removing
            pass
        # maybe if 'Exited with exit code 1' then do not do anything?..
        if 'Exited with exit code' in text or 'Successfully completed' in text:  # check that the job is finished
            job_filename = os.path.split(text[(text.index('Job <') + len('Job <')):text.index('> was submitted')])[1]  # take the job file name
            target_id = ''.join(list(map(lambda x: x if x.isnumeric() else '', list(fo.split('.')[0]))))  # will extract numerical part
            if 'job killed by owner' in text:
                pass
            elif 'Successfully completed' not in text:
                with open(os.path.join(Output_stars_folder, target_id + '_' + Star_full_output_filename_postfix), 'a') as fs:
                    print('FILE:  ' + fo, file=fs)
                    dtt = time.localtime()
                    print(str(dtt.tm_mday) + '.' + str(dtt.tm_mon) + '.' + str(dtt.tm_year) + ', ' + str(dtt.tm_hour) + ':' + str(
                        dtt.tm_min) + ':' + str(dtt.tm_sec), file=fs)
                    # print(file=fs)
                    print(text, file=fs)  # we append the output text to the general output of this star
                    print(file=fs)
                    print('##########################################################################################################################################',
                          file=fs)
                with open(Error_list_file, 'a') as elf:
                    print(job_filename, file=elf)

            if job_filename in os.listdir(Running_jobs_folder):  # I don't know why, but for some reason files are not always there
                os.remove(os.path.join(Running_jobs_folder, job_filename))  # remove the file which finished

        # Even if it does not have output, still let us look at the error
    for fe in os.listdir(Error_files_folder):
        target_id = ''.join(list(map(lambda x: x if x.isnumeric() else '', list(fe.split('.')[0]))))  # will extract numerical part
        with open(os.path.join(Error_files_folder, fe), 'r') as err_fil:
            text_err = err_fil.read()
        # os.remove(os.path.join(Error_files_folder, fe))
        with open(os.path.join(Error_files_folder, fe), 'w') as err_fil:  # instead of removing
            pass
        if len(text_err) > 3:  # just that it is not empty
            with open(os.path.join(Output_stars_folder, target_id + '_' + Star_full_output_filename_postfix), 'a') as fs:
                print('FILE:  ' + fe, file=fs)
                dtt = time.localtime()
                print(str(dtt.tm_mday) + '.' + str(dtt.tm_mon) + '.' + str(dtt.tm_year) + ', ' + str(dtt.tm_hour) + ':' + str(
                    dtt.tm_min) + ':' + str(dtt.tm_sec), file=fs)
                print(file=fs)
                print(text_err, file=fs)  # we append the output text to the general output of this star
                print(file=fs)
                print('###########################################################################################################################################',
                      file=fs)
# TODO: MAYBE SHUFFLE THE FILES BEFORE SUBMITTING TO AVOID RUNNING THE SAME STAR IN PARALLEL WITH ITSELF TOO MUCH?














# bsub -q physics-long -R "rusage[mem=3.9GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/launch_manager_2.py
