#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import target_class
import file_operations as fop
import pipeline_parameters as pp
from runfiles_parameters import *
import traceback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KOI number", default="003097222")
args = parser.parse_args()
Target_id = args.target_id[0]

targ = target_class.KeplerTarget(target_id=Target_id,
                                 load_pickled_attributes=False)  # all the arguments are set to the defaults of the class
targ.clean_output_directory()

if not targ.is_good_for_processing:  # is villanova binary or is not in Gaia
    targ.print_dict_file()
else:  # for now, run the search only if it is not a villanova binary
    try:
        targ.run_main_search(save_to_threshold_crossing_list=True,
                             clean_dir_if_not_threshold_crossing=True)
    # except Exception as exc:
    #     targ.failures += ' ' + str(exc)
    #     with open(Error_list_file, 'a') as err_list_fil:
    #         print(targ.target_id + ' ' + str(exc), file=err_list_fil)
    except Exception as exc:
        tb = sys.exc_info()
        list_str_exc = traceback.format_exception(exc, value=tb[1], tb=tb[2])
        with open(Error_list_file, 'a') as err_list_fil:
            print(targ.target_id, file=err_list_fil)
            for str_exc in list_str_exc:
                print(str_exc.strip(), file=err_list_fil)
            print('', file=err_list_fil)
    # # If the target is threshold-crossing, we write job files to run the timeslides and injections:
    # if targ.is_threshold_crossing:
    #     # noinspection DuplicatedCode
    #     for i in range(int(pp.N_deep_slides_total // pp.N_deep_slides_chunk)  # how many times we need to repeat timeslides
    #                    + int(bool(pp.N_deep_slides_total % pp.N_deep_slides_chunk))):  # stupid way to make np.ceil (otherwise numpy import)
    #         # noinspection DuplicatedCode
    #         fop.write_job_file(code_file=None,
    #                            code_filename=Slides_runfile_filename,
    #                            code_file_dir=Run_files_folder,
    #                            jobname=Slides_job_filename_prefix+'_'+targ.target_id+'.bsub',
    #                            job_file=None,
    #                            job_filename=Slides_job_filename_prefix+'_'+targ.target_id+'_'+str(i)+'.bsub',
    #                            job_file_dir=Job_files_folder,
    #                            queue=Queue,
    #                            mem_per_job_gb=Mem_per_job_gb,
    #                            array_indices=None,
    #                            output_file=None,
    #                            output_filename=Slides_output_filename_prefix+'_'+targ.target_id+'.out',
    #                            output_file_dir=Output_files_folder,
    #                            error_file=None,
    #                            error_filename=Slides_error_filename_prefix+'_'+targ.target_id+'.out',
    #                            error_file_dir=Error_files_folder,
    #                            output_error_overwrite=False,
    #                            arg_jobind=False,
    #                            other_args=[['target_id', targ.target_id]])
    #
    #     # noinspection DuplicatedCode
    #     fop.write_job_file(code_file=None,
    #                        code_filename=Injection_parameters_runfile_filename,
    #                        code_file_dir=Run_files_folder,
    #                        jobname=Injection_parameters_job_filename_prefix+'_'+targ.target_id+'.bsub',
    #                        job_file=None,
    #                        job_filename=Injection_parameters_job_filename_prefix+'_'+targ.target_id+'.bsub',
    #                        job_file_dir=Job_files_folder,
    #                        queue=Queue,
    #                        mem_per_job_gb=Mem_per_job_gb,
    #                        array_indices=None,
    #                        output_file=None,
    #                        output_filename=Injection_parameters_output_filename_prefix+'_'+targ.target_id+'.out',
    #                        output_file_dir=Output_files_folder,
    #                        error_file=None,
    #                        error_filename=Injection_parameters_error_filename_prefix+'_'+targ.target_id+'.out',
    #                        error_file_dir=Error_files_folder,
    #                        output_error_overwrite=False,
    #                        arg_jobind=False,
    #                        other_args=[['target_id', targ.target_id]])
    #
    #     for i in range(int(pp.N_injections_total // pp.N_injections_chunk)  # how many times we need to repeat
    #                    + int(bool(pp.N_injections_total % pp.N_injections_chunk))):
    #         fop.write_job_file(code_file=None,
    #                            code_filename=Injection_detection_runfile_filename,
    #                            code_file_dir=Run_files_folder,
    #                            jobname=Injection_detection_job_filename_prefix+'_'+targ.target_id+'_'+str(i)+'.bsub',
    #                            job_file=None,
    #                            job_filename=Injection_detection_job_filename_prefix+'_'+targ.target_id+'_'+str(i)+'.bsub',
    #                            job_file_dir=Future_job_files_folder,
    #                            queue=Queue,
    #                            mem_per_job_gb=Mem_per_job_gb,
    #                            array_indices=None,
    #                            output_file=None,
    #                            output_filename=Injection_detection_output_filename_prefix+'_'+targ.target_id+'.out',
    #                            output_file_dir=Output_files_folder,
    #                            error_file=None,
    #                            error_filename=Injection_detection_error_filename_prefix+'_'+targ.target_id+'.out',
    #                            error_file_dir=Error_files_folder,
    #                            output_error_overwrite=False,
    #                            arg_jobind=False,
    #                            other_args=[['target_id', targ.target_id], ['first_injection_load_parameters_index', i]])
    #
    #     # noinspection DuplicatedCode
    #     fop.write_job_file(code_file=None,
    #                        code_filename=Plotting_runfile_filename,
    #                        code_file_dir=Run_files_folder,
    #                        jobname=Plotting_job_filename_prefix + '_' + targ.target_id + '.bsub',
    #                        job_file=None,
    #                        job_filename=Plotting_job_filename_prefix + '_' + targ.target_id + '.bsub',
    #                        job_file_dir=Future_job_files_folder,
    #                        queue=Queue,
    #                        mem_per_job_gb=Mem_per_job_gb,
    #                        array_indices=None,
    #                        output_file=None,
    #                        output_filename=Plotting_output_filename_prefix + '_' + targ.target_id + '.out',
    #                        output_file_dir=Output_files_folder,
    #                        error_file=None,
    #                        error_filename=Plotting_error_filename_prefix + '_' + targ.target_id + '.out',
    #                        error_file_dir=Error_files_folder,
    #                        output_error_overwrite=False,
    #                        arg_jobind=False,
    #                        other_args=[['target_id', targ.target_id]])

# bsub -q physics-short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/all_stars_main_search_job_creator.py
