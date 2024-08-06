#!/usr/bin/env python3

import time
import os

# ################################################################################################
# LAUNCH MANAGER: CHECKS HOW MANY JOBS ARE IN THE SERVER AND SUBMITS MORE IF THERE IS ROOM


def check_number_of_submitted_jobs(output_file=None, output_filename='bjobs_output.txt', output_dir=None,
                                   queues=('all',), running=False):
    """Performs bjobs and checks how many submitted jobs there
    For WEXAC IBM job manager. Returns the number of running, waiting and all the other jobs.
    :param running:
    :type running:
    :param output_file: file to which the output of the bjobs command will be written. Not important because will be removed in the end.
    :type output_file:
    :param output_filename: if `output_file` is None, it will be created from `output_filename` which is the name of the file without the
    path to its folder, and `output_file_dir`, which is the path to its folder.
    :type output_filename: str
    :param output_dir: if `output_file` is None, it will be created from `output_filename` which is the name of the file without the
    path to its folder, and `output_file_dir`, which is the path to its folder.
    :type output_dir: str
    :param queues:
    :type queues:
    :return:
    :rtype: int or list
    """
    had_len = True
    if isinstance(queues, str):
        queues = (queues,)
        had_len = False

    if output_file is None:
        if output_dir is None:
            output_dir = os.getcwd()
        output_file = os.path.join(output_dir, output_filename)
    if (output_file[-4] != '.') and (output_file[-3] != '.') and (output_file[-5] != '.'):
        output_file += '.txt'
    wcs = []
    if queues[0] == 'all':
        with open(output_file, 'w'):  # clean file if it was there or create if not
            pass
        try:
            if not running:
                os.system('bjobs | wc > ' + output_file)
            else:
                os.system('bjobs -r | wc > ' + output_file)
        except Exception as e:
            with open(output_file) as fil1:
                print(str(e), file=fil1)

        with open(output_file, 'r') as out_file:
            text = out_file.read()
            if 'No unfinished job found' in text:
                wc = 0
            else:
                wc = int((text.split())[0])
        os.remove(output_file)
        wcs.append(wc)
    else:
        for queue in queues:
            with open(output_file, 'w'):  # clean file if it was there or create if not
                pass
            try:
                if not running:
                    os.system('bjobs -q ' + queue + ' | wc > ' + output_file)
                else:
                    os.system('bjobs -r -q ' + queue + ' | wc > ' + output_file)
            except Exception as e:
                with open(output_file) as fil1:
                    print(str(e), file=fil1)

            with open(output_file, 'r') as out_file:
                text = out_file.read()
                if 'No unfinished job found' in text:
                    wc = 0
                else:
                    wc = int((text.split())[0])
            os.remove(output_file)
            wcs.append(wc)
    res = wcs
    if not had_len:
        res = wcs[0]
    return res


# ################################################################################################
# Defining parameters
Job_files_folder = os.path.join(os.getcwd(), 'jobs')
Max_work_time = 6.9 * 24 * 60 * 60  # in seconds; the job in physics-long is killed after 7 days
Max_jobs_submitted = 4950  # Maximal number of jobs that we can have in bjobs at a certain time
queues = 'short'  # Which queues to check. The submission queues have to be specified inside the jobs
minutes_between_checks = 10.  # How often to check the number of submitted jobs

# ################################################################################################
# Loop of monitoring:
timeout_start = time.time()
time_last = timeout_start - (minutes_between_checks + 0.01) * 60  # I want it to start immediately
while (time.time() < timeout_start + Max_work_time) and (len(os.listdir(Job_files_folder)) > 0):
    if time.time() - time_last > minutes_between_checks * 60:  # check every 10 min
        time_last = time.time()
        wcs = check_number_of_submitted_jobs(output_dir=os.getcwd(), queues=queues, running=False)
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

