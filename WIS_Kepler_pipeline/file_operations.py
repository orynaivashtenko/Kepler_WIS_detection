#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:36:34 2021

@author: orynaivashtenko
"""
import numpy as np
import os
import pickle
import time
import datetime

from numba.core.removerefctpass import remove_unnecessary_nrt_usage


def is_int(string):
    assert type(string) == str, 'Input must be string!'
    if string.isdecimal():
        return True
    elif string[0] == '-' and (string[1:]).isdecimal():
        return True
    else:
        return False


def is_simple_num(string):
    """
    Checks whether the function is a simple number, that is float
    not in scientific"e+..." format
    """

    if is_int(string):
        return True
    elif string.count('.') == 1 and is_int(string.replace('.', '')):
        return True
    else:
        return False


def is_float_number(string):
    string = string.lower()
    if is_simple_num(string):
        return True
    elif ((string.count('e') == 1 and is_simple_num(string.replace('e', '')))
          or (string.count('e-') == 1 and is_simple_num(string.replace('e-', '')))
          or (string.count('e+') == 1 and is_simple_num(string.replace('e+', '')))):
        return True
    else:
        return False


def is_float(string):
    string = string.lower()
    if is_float_number(string):
        return True
    elif string == 'nan' or string == 'inf' or string == '-inf':
        return True
    else:
        return False


def is_bool(string):
    if string == 'True' or string == 'False':
        return True
    else:
        return False


def is_boolable(string):
    if (string == 'True' or string == 'False'
            or string == '0' or string == '1'
            or string == '0.' or string == '1.'):
        return True
    else:
        return False


# noinspection PyTypeChecker
def read_quarters_from_file(file_name, location='', dimension=2, to_array=True):
    """
    Reads an array (numpy.array or list of float, maximum 2d) printed to the txt file
    named name and returns the output in form of list.

    Parameters
    ----------
     file_name : str
        Name of the file ('.txt' is not needed).
    location : str
        Location where to create the file. If None, the current local directory
        will be used
    dimension : int
        The dimensionality of the array to be read. either 2 or 3, maybe 1.
    to_array : bool
    Whether convert to array

    Returns
    -------
    res : List (possibly with nested lists) of data read from the file.

    """
    # noinspection DuplicatedCode
    if location is None:
        if os.path.split(file_name)[0] != '':
            file_dir = file_name
        else:
            location = os.getcwd()
            file_dir = os.path.join(location, file_name)
    else:
        file_dir = os.path.join(location, file_name)
    if file_dir[-4:] != '.txt':
        file_dir = file_dir + '.txt'

    with open(file_dir, 'r') as file:
        res = file.read()
        if dimension < 3:
            res = res.replace('array', '')
            res = res.replace('(', '')
            res = res.replace(')', '')
            res = res.replace(',', ' ')
            res = list(res.split('] ['))
            for i in range(len(res)):
                res[i] = res[i].replace(']', '')
                res[i] = res[i].replace('[', '')
                res[i] = res[i].replace('\n', '')
                res[i] = list(map(float, res[i].split()))
            if len(res) == 1:
                res = res[0]
        elif dimension == 3:
            res = res.replace('array', '')
            res = res.replace('(', '')
            res = res.replace(')', '')
            res = res.replace(',', ' ')
            res = res.replace('  ', ' ')  # important to do in because
            # otherwise ']  [' with two spaces can appear (which we can't see)
            res = list(res.split(']] [['))
            for j in range(len(res)):
                res[j] = list(res[j].split('] ['))
                # noinspection DuplicatedCode
                for i in range(len(res[j])):
                    res[j][i] = res[j][i].replace(']', '')
                    res[j][i] = res[j][i].replace('[', '')
                    res[j][i] = res[j][i].replace('\n', '')
                    if to_array:
                        res[j][i] = np.array(list(map(float, res[j][i].split())))
                    else:
                        res[j][i] = list(map(float, res[j][i].split()))
    return res


def write_quarters_to_txt(file_name, var, location=None):
    """
    Prints array (list or numpy.array of float) variable var to file named name in the location `location`.
    If the var is not given, the name of the variable is the same as name

    Parameters
    ----------
    file_name : str
        Name of the file ('.txt' is not needed). If var is None, it is also
        the name of the variable to be printed to the file.
    var : list or array
        Variable to be printed to the file. If var is None, name is the name
        of the variable to be printed to the file.
    location : str
        Location where to create the file. If None, the current local directory
        will be used

    Returns
    -------
    None.

    """
    # noinspection DuplicatedCode
    if location is None:
        if os.path.split(file_name)[0] != '':
            file_dir = file_name
        else:
            location = os.getcwd()
            file_dir = os.path.join(location, file_name)
    else:
        file_dir = os.path.join(location, file_name)
    if file_dir[-4:] != '.txt':
        file_dir = file_dir + '.txt'
    # if var == None:
    #     var = globals()[file_dir]
    if location is None:
        location = os.getcwd()
        fileloc = os.path.join(location, file_dir) + '.txt'
    else:
        fileloc = file_dir + '.txt'
    with open(fileloc, 'w') as file:
        print(*var, file=file)


def read_csv_file(file, location=None, to_type=None):
    if location is not None:
        file = os.path.join(location, file)
    file = open(file, 'r')
    res = file.read()
    file.close()
    res = res.split(',')
    res = list(map(lambda x: x.strip(), res))
    if to_type is None:
        isfloat = True
        isint = True
        isbool = True
        for entry in res:
            if not is_int(entry):
                isint = False
            if not is_float(entry):
                isfloat = False
            if not is_bool(entry):
                isbool = False
            if (not isint) and (not isfloat) and (not isbool):
                break
        if isbool:
            to_type = bool
        if isint:
            to_type = float
        if isfloat:
            to_type = float

    if to_type == int or to_type == float or to_type == bool:
        res = list(map(to_type, res))
    elif to_type == str:
        pass
    else:
        raise AssertionError('Incorrect `to_type` specified!')
    return res


def write_job_file(code_file=None, code_filename=None, code_file_dir=None, jobname='job_noname',
                   job_file=None, job_filename='job_test.bsub', job_file_dir=None, env_name='',
                   queue='"short"', mem_per_job_gb='3.9', array_indices=None,
                   output_file=None, output_filename=None, output_file_dir=None,
                   error_file=None, error_filename=None, error_file_dir=None,
                   output_error_overwrite=True,
                   arg_jobind=False, other_args=None):
    """Creates a job or job array file which can be submitted to WEXAC IBM bsub command.

    :param job_file: bash file in which the job will be written
    :type job_file: str or None
    :param job_filename: if `job_file` is None, it will be created from `job_filename` which is the name of the file without the
    path to its folder, and `job_file_dir`, which is the path to its folder.
    :type job_filename: str or None
    :param job_file_dir: if `job_file` is None, it will be created from `job_filename` which is the name of the file without the
    path to its folder, and `job_file_dir`, which is the path to its folder.
    :type job_file_dir: str or None
    :param code_file: the full path to the file of the code which needs to be executed in the job
    :type code_file: str or None
    :param code_filename: if `code_file` is None, it will be created from `code_filename` which is the name of the file without the
    path to its folder, and `code_file_dir`, which is the path to its folder.
    :type code_filename: str or None
    :param code_file_dir: if `code_file` is None, it will be created from `code_filename` which is the name of the file without the
    path to its folder, and `code_file_dir`, which is the path to its folder.
    :type code_file_dir: str or None
    :param jobname: the name of the job to appear in the job code. Works like job identifier to see it easier in the jobs list
    :type jobname: str
    :param queue: queue in the WEXAC IBM submission machinery to which the job will be submitted. The default is physics-short.
    Another recommended option can be physics-long.
    :type queue: str
    :param mem_per_job_gb: RAM memory in GB per job. Should be less than 4 GB.
    :type mem_per_job_gb: str or float
    :param array_indices: If we want to create a job array and not a single job, this will be added to the jobname in a format
    `jobname`[array_indices[0]-array_indices[1]].
    :type array_indices: tuple or None
    :param output_file: full path to the file in which the output will be collected.
    :type output_file: str or None
    :param output_filename: if `output_file` is None, it will be created from `output_filename` which is the name of the file without the
    path to its folder, and `output_file_dir`, which is the path to its folder.
    :type output_filename: str or None
    :param output_file_dir: if `output_file` is None, it will be created from `output_filename` which is the name of the file without the
    path to its folder, and `output_file_dir`, which is the path to its folder.
    :type output_file_dir: str or None
    :param error_file: full path to the file in which the error info will be collected.
    :type error_file: str or None
    :param error_filename: if `error_file` is None, it will be created from `error_filename` which is the name of the file without the
    path to its folder, and `error_file_dir`, which is the path to its folder.
    :type error_filename: str or None
    :param error_file_dir: if `error_file` is None, it will be created from `error_filename` which is the name of the file without the
    path to its folder, and `error_file_dir`, which is the path to its folder.
    :type error_file_dir: str or None
    :param output_error_overwrite: whether to overwrite `error_file` and `output_file` or to append the new outputs to the end.
    :type output_error_overwrite: bool
    :param arg_jobind: whether to parse the job index as an argument of the code which will be run in the job.
    :type arg_jobind: bool
    :param env_name: Name of the conda environment to use. The default is the empty string, which is to run in the base env.
    :type env_name: str
    :param other_args: list of lists with pairs of argument name and argument value, all strings
    :type other_args: list[list[str]]  or None
    :return: 
    :rtype: 
    """
    if code_file is None:
        code_file = os.path.join(code_file_dir, code_filename)
    assert code_file is not None, 'Please specify which code to run in the job!'

    if job_file is None:
        if job_file_dir is None:
            job_file_dir = os.getcwd()
        job_file = os.path.join(job_file_dir, job_filename)
    if (job_file[-4] != '.') and (job_file[-3] != '.') and (job_file[-5] != '.'):
        job_file += '.bsub'

    # noinspection DuplicatedCode
    if (output_file is None) and (output_filename is not None) and (output_file_dir is not None):
        output_file = os.path.join(output_file_dir, output_filename)
    if output_file is not None:
        if (output_file[-4] != '.') and (output_file[-3] != '.') and (output_file[-5] != '.'):
            output_file += '.out'
    # noinspection DuplicatedCode
    if (error_file is None) and (error_filename is not None) and (error_file_dir is not None):
        error_file = os.path.join(error_file_dir, error_filename)
    if error_file is not None:
        if (error_file[-4] != '.') and (error_file[-3] != '.') and (error_file[-5] != '.'):
            error_file += '.out'

    if array_indices is not None:
        array_indices = '[' + str(int(array_indices[0])) + '-' + str(int(array_indices[1])) + ']'
    else:
        array_indices = ''

    text_to_write = '#!/usr/bin/bash\n'
    text_to_write += '#\n'
    text_to_write += '#BSUB -q ' + queue + '\n'
    text_to_write += '#BSUB -R "rusage[mem=' + str(mem_per_job_gb) + 'GB]"\n'
    text_to_write += '#BSUB -J ' + str(jobname) + array_indices + '\n'
    if output_file is not None:
        if output_error_overwrite:
            text_to_write += '#BSUB -oo ' + str(output_file) + '\n'
        else:
            text_to_write += '#BSUB -o ' + str(output_file) + '\n'
    if error_file is not None:
        if output_error_overwrite:
            text_to_write += '#BSUB -eo ' + str(error_file) + '\n'
        else:
            text_to_write += '#BSUB -e ' + str(error_file) + '\n'

    # text_to_write += "'conda init; conda activate " + str(env_name) + "; "
    text_to_write += "source ~/.bashrc\n"
    text_to_write += "conda activate " + env_name + "\n"
    # text_to_write += "conda activate " + str(env_name) + "\n"
    text_to_write += str(code_file)
    if arg_jobind:
        text_to_write += " $LSB_JOBINDEX"
    if other_args is not None:
        assert isinstance(other_args, list), 'Parameter other_args should be list of strings!'
        for arg in other_args:
            text_to_write += " --" + str(arg[0]) + " " + str(arg[1])
    # text_to_write += "'"
    with open(job_file, 'w') as fil1:
        print(text_to_write, file=fil1)

    # !/usr/bin/env python3


def check_number_of_submitted_jobs(output_file=None, output_filename='bjobs_output.txt', output_dir=None,
                                   queues=('all',), running=False, pending=True):
    """Performs bjobs and checks how many submitted jobs there
    For WEXAC IBM job manager. Returns the number of running, waiting and all the other jobs.
    :param running:
    :type running:
    :param pending:
    :type pending:
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
            command = 'bjobs '
            if running:
                command += '-r '
            if pending:
                command += '-pe '
            os.system(command + '| wc > ' + output_file)
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
                command = 'bjobs '
                if running:
                    command += '-r '
                if pending:
                    command += '-pe '
                command += '-q ' + queue + ' '
                os.system(command + '| wc > ' + output_file)
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


def report_exception(exception, file_to_append, title='', info=''):
    import sys
    import traceback
    tb = sys.exc_info()
    list_str_exc = traceback.format_exception(exception, value=tb[1], tb=tb[2])
    with open(file_to_append, 'a') as err_list_fil:
        print('NEW', file=err_list_fil)
        print(datetime.datetime.now(), file=err_list_fil)
        if title != '':
            print(title, file=err_list_fil)
        if info != '':
            print(info, file=err_list_fil)
        for str_exc in list_str_exc:
            print(str_exc.strip(), file=err_list_fil)
        print('', file=err_list_fil)
        print('SHORTLY:', file=err_list_fil)
        print(str(exception), file=err_list_fil)
        print('END', file=err_list_fil)
        print('', file=err_list_fil)


def print_problem(text, file=None, add_time=True):
    res = ''
    if add_time:
        res += str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' : '
    res += text
    with open(file, 'a') as fil1:
        print(text, file=fil1)
        print('----------------------------------------------------------', file=fil1)


# ####################################################################################################################
# FILE LOCKING
import fcntl
Dt_wait_locker = 2.  # s  # -> this number is just a scale, the waiting time will be random
Waiting_time_limit_locker = 120  # s


class Locker:
    """
    Class used to lock a file when some process used it, and then unlock whan the usage is finished. The locking is
    done by creating a file with extension '.locker' in the same directory as the file that needs to be locked.
    The class checks whether such a file exists. If yes, then it waits. Before accessing the file, the class creates
    the locker file. After finishing the work, the class deleted the file. The syntax is:

    with Locker(file=file_dir):
        do_stuff(file_dir)

    """
    def __init__(self, file, dt_wait=Dt_wait_locker, max_waiting_time=Waiting_time_limit_locker):
        self.file = file
        self.locker_file = get_locker_file(self.file)
        self.locker_file_fd = None
        self.dt_wait = dt_wait
        self.max_waiting_time = max_waiting_time

    def __enter__(self):
        self.wait_and_lock()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.unlock()

    def wait_and_lock(self):
        fd = os.open(self.locker_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        locker_file_fd = None

        dt = self.dt_wait
        max_time = self.max_waiting_time
        t0 = time.time()
        waiting_time = 0.
        while waiting_time < max_time:
            try:
                # The LOCK_EX means that only one process can hold the lock
                # The LOCK_NB means that the fcntl.flock() is not blocking,
                # and we are able to implement termination of while loop, when timeout is reached.
                # More information here: https://docs.python.org/3/library/fcntl.html#fcntl.flock
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                pass
            else:
                locker_file_fd = fd
                break
            time.sleep(np.random.rand() * dt * 2)
            waiting_time = time.time() - t0
        if locker_file_fd is None:
            os.close(fd)
            raise ValueError('File could not be locked!')
        self.locker_file_fd = locker_file_fd
        return self.locker_file_fd

    def unlock(self):
        # Do not remove the lockfile:
        #   https://github.com/benediktschmitt/py-filelock/issues/31
        #   https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
        fcntl.flock(self.locker_file_fd, fcntl.LOCK_UN)
        os.close(self.locker_file_fd)
        # os.chmod(self.locker_file_fd, 777)  # when I added this, i got Bad file descriptor error
        return None


def get_locker_file(file):
    no_ext_file, ext = os.path.splitext(file)
    if ext == '.locker':
        raise ValueError(file+' is a locker file!')
    locker_file = no_ext_file + '.locker'
    return locker_file


# def is_locked(file):
#     locker_file = get_locker_file(file)
#     if os.path.exists(locker_file):
#         return True
#     else:
#         return False

# def lock_file(f):
#     if f.writable(): fcntl.lockf(f, fcntl.LOCK_EX)
#
# def unlock_file(f):
#     if f.writable(): fcntl.lockf(f, fcntl.LOCK_UN)

# def lock_file(file):
#     locker_file = get_locker_file(file)
#     if os.path.exists(locker_file):
#         raise ValueError(file+' is locked!')
#     else:
#         with open(locker_file, 'w') as _:
#             pass
#
#
# def unlock_file(file):
#     locker_file = get_locker_file(file)
#     if not os.path.exists(locker_file):
#         raise ValueError(file+' is not locked!')
#     else:
#         os.remove(locker_file)

# ####################################################################################################################
