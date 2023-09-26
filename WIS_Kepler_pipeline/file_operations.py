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


def variables_from_file(file):
    """
    Opens the file with a list of variable names and values. Recognizes the data
    types of the values. Creates variables with corresponding names and assigns
    them the values. Format of the file (example):
        dummy_coefficient = 1.
        curse='hann'

        best_number=13
        is_life_great = True

    Parameters
    ----------
    file : str
        Name (full path) of the file from which the variables should be read.
    Returns
    -------
    None.

    Creates
    -------
    The variables and assigns values
    """
    f = open(file, 'r')
    lines = f.readlines()
    f.close()
    for line in lines:
        if '=' in line:
            text = line.split('#')[0]  # remove comments
            text = text.split('\\')[0]  # remove endlines
            name, val = text.split('=')[0:2]
            name = name.strip()
            val = val.strip()
            if name != '' and val != '':
                if val == 'None':
                    val = None
                elif val == 'True':
                    val = True
                elif val == 'False':
                    val = False
                elif is_int(val):
                    val = int(val)
                elif is_float(val):
                    val = float(val)
                else:
                    assert (((val[0] == '"') and (val[-1] == '"'))
                            or ((val[0] == "'") and (
                                        val[-1] == "'"))), 'Undefined type of value: not int, bool, float, str or None! ( line "' + str(
                        line) + '" )'
                    val = val[1:-1]

                assert ((65 <= ord(name[0]) <= 90) or (97 <= ord(name[0]) <= 122) or name[0] == '_'), 'Incorrect variable name specified!'
                if len(name) > 1:
                    for e in name[1:]:
                        assert ((48 <= ord(e) <= 57) or (65 <= ord(e) <= 90) or (
                                    97 <= ord(e) <= 122) or e == '_'), 'Incorrect variable name specified!'
                globals()[name] = val


def read_lists_from_file(file, location=None, dimension=2, to_array=True):
    """
    Reads an array (numpy.array or list of float, maximum 2d) printed to the txt file
    named name and returns the output in form of list.

    Parameters
    ----------
     file : str
        Name of the file ('.txt' is not needed).
    location : str
        Location where to create the file. If None, the current local directory
        will be used
    dimension : int
        3, 2, maybe 1. Dimensionality of the array to be read
    to_array : bool
        whether to convert to numpy array or stay with list

    Returns
    -------
    res : list
        List (possibly with nested lists) of data read from the file.

    """
    if location is not None:
        file = os.path.join(location, file)
    else:
        if file[-4:] != '.txt':
            file = file + '.txt'
    file = open(file, 'r')
    res = file.read()
    if dimension < 3:
        res = res.replace('array', '')
        res = res.replace('(', '')
        res = res.replace(')', '')
        res = res.replace(',', ' ')
        res = (res.split('] ['))
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
        res = (res.split(']] [['))
        for j in range(len(res)):
            res[j] = (res[j].split('] ['))
            for i in range(len(res[j])):
                res[j][i] = res[j][i].replace(']', '')
                res[j][i] = res[j][i].replace('[', '')
                res[j][i] = res[j][i].replace('\n', '')
                if to_array:
                    res[j][i] = np.array(list(map(float, res[j][i].split())))
                else:
                    res[j][i] = list(map(float, res[j][i].split()))
        file.close()
    return res


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


############################################################################################################################################
# MAKING SOME DICTIONARIES AND OTHER FILES USED IN THE PIPELINE
############################################################################################################################################

Default_files_folder = '/Users/orynaivashtenko/Work/Kepler_Project/'
Default_koi_catalog_filename = 'cumulative_2022.10.18_03.41.17.csv'
Default_koi_dict_filename = 'KOI_catalog_dict.pickle'
Default_ttv_catalog_filename = 'TTV_Catalog_Table3_O-C.txt'
Default_ttv_dict_filename = 'TTV_catalog_dict.pickle'
Default_villanova_catalog_filename = 'villanova_catalog_timings.txt'
Default_villanova_dict_filename = 'villanova_catalog_dict.pickle'


def make_koi_catalog_dict(koi_catalog_path=os.path.join(Default_files_folder, Default_koi_catalog_filename),
                          pickling_file_path=os.path.join(Default_files_folder, Default_koi_dict_filename)):
    """

    # This file was produced by the NASA Exoplanet Archive  http://exoplanetarchive.ipac.caltech.edu
    # Tue Oct 18 03:41:17 2022
    #
    # User preference: *
    #
    # COLUMN kepid:          KepID
    # COLUMN kepoi_name:     KOI Name
    # COLUMN kepler_name:    Kepler Name
    # COLUMN koi_disposition: Exoplanet Archive Disposition
    # COLUMN koi_pdisposition: Disposition Using Kepler Data
    # COLUMN koi_score:      Disposition Score
    # COLUMN koi_fpflag_nt:  Not Transit-Like False Positive Flag
    # COLUMN koi_fpflag_ss:  Stellar Eclipse False Positive Flag
    # COLUMN koi_fpflag_co:  Centroid Offset False Positive Flag
    # COLUMN koi_fpflag_ec:  Ephemeris Match Indicates Contamination False Positive Flag
    # COLUMN koi_period:     Orbital Period [days]
    # COLUMN koi_time0bk:    Transit Epoch [BKJD]
    # COLUMN koi_eccen:      Eccentricity
    # COLUMN koi_longp:      Long. of Periastron [deg]
    # COLUMN koi_impact:     Impact Parameter
    # COLUMN koi_duration:   Transit Duration [hrs] Duration is measured from first contact between the planet and star until last contact.
    # COLUMN koi_ingress:    Ingress Duration [hrs]
    # COLUMN koi_depth:      Transit Depth [ppm]
    # COLUMN koi_ror:        Planet-Star Radius Ratio
    # COLUMN koi_prad:       Planetary Radius [Earth radii]
    # COLUMN koi_sma:        Orbit Semi-Major Axis [au]
    # COLUMN koi_incl:       Inclination [deg]
    # COLUMN koi_ldm_coeff2: Limb Darkening Coeff. 2
    # COLUMN koi_ldm_coeff1: Limb Darkening Coeff. 1
    # COLUMN koi_max_sngle_ev: Maximum Single Event Statistic
    # COLUMN koi_max_mult_ev: Maximum Multiple Event Statistic
    # COLUMN koi_model_snr:  Transit Signal-to-Noise
    # COLUMN koi_num_transits: Number of Transits
    # COLUMN koi_tce_plnt_num: TCE Planet Number
    # COLUMN koi_tce_delivname: TCE Delivery
    # COLUMN koi_bin_oedp_sig: Odd-Even Depth Comparision Statistic
    # COLUMN koi_steff:      Stellar Effective Temperature [K]
    # COLUMN koi_slogg:      Stellar Surface Gravity [log10(cm/s**2)]
    # COLUMN koi_srad:       Stellar Radius [Solar radii]
    # COLUMN koi_smass:      Stellar Mass [Solar mass]
    # COLUMN ra:             RA [decimal degrees]
    # COLUMN dec:            Dec [decimal degrees]
    # COLUMN koi_kepmag:     Kepler-band [mag]

    """
    if pickling_file_path[-7:] != '.pickle':
        pickling_file_path = pickling_file_path + '.pickle'
    with open(koi_catalog_path, 'r') as fil1:
        lines = fil1.readlines()
        lines = [line.strip().split(',') for line in lines if line.strip()[0] != '#']
        headers = lines[0]
        lines = lines[1:]

        dict_koi = {}
        for key in headers:
            dict_koi[key] = []
        for line in lines:
            for i in range(len(headers)):
                dict_koi[headers[i]].append(line[i])

        for j in range(len(dict_koi['kepid'])):
            dict_koi['kepid'][j] = dict_koi['kepid'][j].rjust(9, '0')

        with open(pickling_file_path, 'wb') as pickle_file:
            pickle.dump(dict_koi, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def make_ttv_dict(koi_catalog_dict_path=os.path.join(Default_files_folder, Default_koi_dict_filename),
                  ttv_catalog_path=os.path.join(Default_files_folder, Default_ttv_catalog_filename),
                  pickling_ttv_catalog_dict_path=os.path.join(Default_files_folder, Default_ttv_dict_filename)):
    """
    # WARNING: SOMETIMES, IN THE TTV LIST SOME TRANSITS ARE MISSION, E.G. 011853905 time 1373.75

    Takes the TTV catalog, makes from it a dict. But it does not have a target_id name, only a useless KOI identifier.
    Therefore, this function opens the KOI catalog dict generated by another function from this file, and finds the corresponding
    name for each entry. It is added as a new key-value pair in the dict. Then, the dict is pickled.
    :param koi_catalog_dict_path:
    :type koi_catalog_dict_path:
    :param ttv_catalog_path:
    :type ttv_catalog_path:
    :param pickling_ttv_catalog_dict_path:
    :type pickling_ttv_catalog_dict_path:
    :return:
    :rtype:
    """
    with open(koi_catalog_dict_path, 'rb') as koi_catalog_dict_file:
        koi_info_dict = pickle.load(koi_catalog_dict_file)
    kois_from_koi_dict = [str(float(kk[1:])) for kk in koi_info_dict['kepoi_name']]

    with open(ttv_catalog_path, 'r') as fil1:
        lines = fil1.readlines()
        lines = [line.split() for line in lines if line.strip()[0] != '#']

    headers = 'KOI, n, T, O_C, O_C_Err, TDV, TDV_Err, TPV, TPV_Err, Flag_outlier, Flag_collision'.split(', ')
    explanations = 'Kepler Object of Interest number; Transit number; Expected transit time of the linear ephemeris (BJD-2454900); \
    O-C difference min; O-C uncertainty min; Fractional duration variation; TDV uncertainty; Fractional depth variation; TPV uncertainty; \
    Outlier flag. Sum of: 0 = ok, 1 = pvalue > 0.025, 2= local TTV, 4= global TTV, 8= global TDV, 16 =global TPV, 32 = TTV,TDV or TPV \
    uncertainty outlier; Collision flag. 0=ok, 1 = another relatively large (>0.25 area of transit) interfering transit close by \
    (<2*transit duration + 1 interfering transit duration)'

    dict_ttv = {}
    for key in headers:
        dict_ttv[key] = []
    for line in lines:
        for i in range(len(headers)):
            dict_ttv[headers[i]].append(line[i])

    dict_ttv['kepid'] = []
    for i in range(len(dict_ttv['KOI'])):
        arg = kois_from_koi_dict.index(dict_ttv['KOI'][i])
        dict_ttv['kepid'].append(koi_info_dict['kepid'][arg])

    with open(pickling_ttv_catalog_dict_path, 'wb') as pickle_file:
        pickle.dump(dict_ttv, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def make_villanova_dict(villanova_catalog_path=os.path.join(Default_files_folder, Default_villanova_catalog_filename),
                        pickling_dict_path=os.path.join(Default_files_folder, Default_villanova_dict_filename)):
    # villanova_catalog_timings.txt
    if pickling_dict_path[-7:] != '.pickle':
        pickling_dict_path = pickling_dict_path + '.pickle'
    with open(villanova_catalog_path, 'r') as fil1:
        lines = fil1.readlines()
        lines = [line.strip().split(',') for line in lines if len(line.strip()) > 0 and line.strip()[0] != '#']
        headers = lines[0]
        headers = [hh for hh in headers if hh != '']  # the line might end with ',', and then the last header will be ''
        lines = lines[1:]

        dict_villanova = {}
        for key in headers:
            dict_villanova[key] = []
        for line in lines:
            for i in range(len(headers)):
                if headers[i] == 'KIC':
                    dict_villanova[headers[i]].append(line[i].rjust(9, '0'))
                else:
                    dict_villanova[headers[i]].append(line[i])

        with open(pickling_dict_path, 'wb') as pickle_file:
            pickle.dump(dict_villanova, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def write_job_file(code_file=None, code_filename=None, code_file_dir=None, jobname='job_noname',
                   job_file=None, job_filename='job_test.bsub', job_file_dir=None,
                   queue='physics-short', mem_per_job_gb='3.9', array_indices=None,
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
    :param other_args: list of lists with pairs of argument name and argument value, all strings
    :type other_args: list[list[str, str]]  or None
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
            text_to_write += '#BSUB -oe ' + str(error_file) + '\n'
        else:
            text_to_write += '#BSUB -e ' + str(error_file) + '\n'

    text_to_write += str(code_file)
    if arg_jobind:
        text_to_write += ' $LSB_JOBINDEX'
    if other_args is not None:
        assert isinstance(other_args, list), 'Parameter other_args should be list of strings!'
        for arg in other_args:
            text_to_write += ' --' + str(arg[0]) + ' ' + str(arg[1])

    with open(job_file, 'w') as fil1:
        print(text_to_write, file=fil1)

    # !/usr/bin/env python3


def check_number_of_submitted_jobs(output_file=None, output_filename='bjobs_output.txt', output_dir=None,
                                   queue='physics-short'):
    """Performs bjobs and checks how many submitted jobs there
    For WEXAC IBM job manager. Returns the number of running, waiting and all the other jobs.
    :param output_file: file to which the output of the bjobs command will be written. Not important because will be removed in the end.
    :type output_file:
    :param output_filename: if `output_file` is None, it will be created from `output_filename` which is the name of the file without the
    path to its folder, and `output_file_dir`, which is the path to its folder.
    :type output_filename: str
    :param output_dir: if `output_file` is None, it will be created from `output_filename` which is the name of the file without the
    path to its folder, and `output_file_dir`, which is the path to its folder.
    :type output_dir: str
    :param queue:
    :type queue:
    :return:
    :rtype: int
    """
    if output_file is None:
        if output_dir is None:
            output_dir = os.getcwd()
        output_file = os.path.join(output_dir, output_filename)
    if (output_file[-4] != '.') and (output_file[-3] != '.') and (output_file[-5] != '.'):
        output_file += '.txt'
    with open(output_file, 'w'):  # clean file if it was there or create if not
        pass
    try:
        os.system('bjobs -q ' + queue + ' | wc > ' + output_file)
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
    return wc
