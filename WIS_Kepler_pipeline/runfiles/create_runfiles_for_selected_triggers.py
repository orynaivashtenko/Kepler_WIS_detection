#!/usr/bin/env python3

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import file_operations as fop
from pipeline_parameters import *
from runfiles_parameters import *

# N_slides_total_for_now = N_deep_slides_total
N_slides_total_for_now = 20000  # we have too many targets to afford 100000, so for now, let us have this
Snr_threshold_lower = 6.8  # this will produce of the order of 100000 jobs, which is enough for now...
Snr_threshold_normal = 7.0
Period_threshold_trash = 50.  # days
Period_threshold_lower_snr = 100.  # days
N_injection_chunks_for_now = 50

threshold_crossing_list_file = os.path.join(Threshold_crossing_lists_folder, Threshold_crossing_list_filename)
threshold_inv_crossing_list_file = os.path.join(Threshold_crossing_lists_folder, Threshold_inv_crossing_list_filename)

with open(threshold_crossing_list_file, 'r') as trig_file:
    lines = trig_file.readlines()

trig_ids = []
# trig_snrs = []
for i in range(len(lines)):
    line = lines[i]
    if line.strip()[0] != '#' and len(line.strip()) >= 9:
        parameters = line.split()
        # reject too low periods, because they are dominated by trash from even lower periods and by 30 days artifacts:
        if (float(parameters[2]) > Period_threshold_trash) and (np.abs(float(parameters[1])) > Snr_threshold_lower):
            # for long periods, make snr threshold higher because anyway p_astro will be small
            if (np.abs(float(parameters[1])) >= Snr_threshold_normal) or (float(parameters[2]) < Period_threshold_lower_snr):
                trig_ids.append(parameters[0])
                # trig_snrs.append(np.abs(float(parameters[1])))

# # TODO: FOR NOW, WE ONLY RUN ON THE NOT-INVERTED!!!! BECAUSE IT TAKES TOO LONG
# with open(threshold_inv_crossing_list_file, 'r') as trig_file:
#     lines = trig_file.readlines()
# trig_ids = trig_ids + [line.split()[0] for line in lines if (line.strip()[0] != '#' and len(line.strip()) >= 9)]
# trig_snrs = trig_snrs + [np.abs(float(line.split()[1])) for line in lines if (line.strip()[0] != '#' and len(line.strip()) >= 9)]

trig_ids = list(set(trig_ids))

Output_location_wexac = '/home/labs/barakz/Collaboration-Kepler/orynai/output_run_6'
# noinspection DuplicatedCode
for tid in trig_ids:
    # f_sl = [ff for ff in os.listdir(os.path.join('/home/labs/barakz/Collaboration-Kepler/orynai/output_run_6', tid[:4], tid)) if 'mes_slides' in ff]
    #
    # if len(f_sl) > 0:
    #     n_slides = int(f_sl[0][f_sl[0].index('mes_slides_') + len('mes_slides_'): f_sl[0].index('.')])
    # else:
    #     n_slides = 0
    #
    #     # noinspection DuplicatedCode
    #     if n_slides < N_slides_total_for_now:
    #         for i in range(int((N_slides_total_for_now-n_slides) // N_deep_slides_chunk)  # how many times we need to repeat timeslides
    #                        + int(bool((N_slides_total_for_now-n_slides) % N_deep_slides_chunk))):  # stupid way to make np.ceil (otherwise numpy import)
    #             # noinspection DuplicatedCode
    #             fop.write_job_file(code_file=None,
    #                                code_filename=Slides_runfile_filename,
    #                                code_file_dir=Run_files_folder,
    #                                jobname=Slides_job_filename_prefix + '_' + tid + '.bsub',
    #                                job_file=None,
    #                                job_filename=Slides_job_filename_prefix + '_' + tid + '_' + str(i) + '.bsub',
    #                                job_file_dir=Job_files_folder,
    #                                queue=Queue,
    #                                mem_per_job_gb=Mem_per_job_gb,
    #                                array_indices=None,
    #                                output_file=None,
    #                                output_filename=Slides_output_filename_prefix + '_' + tid + '.out',
    #                                output_file_dir=Output_files_folder,
    #                                error_file=None,
    #                                error_filename=Slides_error_filename_prefix + '_' + tid + '.out',
    #                                error_file_dir=Error_files_folder,
    #                                output_error_overwrite=False,
    #                                arg_jobind=False,
    #                                other_args=[['target_id', tid]])

    # if len([ff for ff in os.listdir(os.path.join(Output_location_wexac, tid[:4], tid)) if Injection_parameters_filename_postfix in ff]) == 0:
    #     # noinspection DuplicatedCode
    #     fop.write_job_file(code_file=None,
    #                        code_filename=Injection_parameters_runfile_filename,
    #                        code_file_dir=Run_files_folder,
    #                        jobname=Injection_parameters_job_filename_prefix+'_'+tid+'.bsub',
    #                        job_file=None,
    #                        job_filename=Injection_parameters_job_filename_prefix+'_'+tid+'.bsub',
    #                        job_file_dir=Job_files_folder,
    #                        queue=Queue,
    #                        mem_per_job_gb=Mem_per_job_gb,
    #                        array_indices=None,
    #                        output_file=None,
    #                        output_filename=Injection_parameters_output_filename_prefix+'_'+tid+'.out',
    #                        output_file_dir=Output_files_folder,
    #                        error_file=None,
    #                        error_filename=Injection_parameters_error_filename_prefix+'_'+tid+'.out',
    #                        error_file_dir=Error_files_folder,
    #                        output_error_overwrite=False,
    #                        arg_jobind=False,
    #                        other_args=[['target_id', tid]])

    for i in np.arange(int(N_injections_total // N_injections_chunk)  # how many times we need to repeat
                       + int(bool(N_injections_total % N_injections_chunk)))[N_injection_chunks_for_now:]:
        fop.write_job_file(code_file=None,
                           code_filename=Injection_detection_runfile_filename,
                           code_file_dir=Run_files_folder,
                           jobname=Injection_detection_job_filename_prefix+'_'+tid+'_'+str(i)+'.bsub',
                           job_file=None,
                           job_filename=Injection_detection_job_filename_prefix+'_'+tid+'_'+str(i)+'.bsub',
                           job_file_dir=Job_files_folder,
                           queue=Queue,
                           mem_per_job_gb=Mem_per_job_gb,
                           array_indices=None,
                           output_file=None,
                           output_filename=Injection_detection_output_filename_prefix+'_'+tid+'.out',
                           output_file_dir=Output_files_folder,
                           error_file=None,
                           error_filename=Injection_detection_error_filename_prefix+'_'+tid+'.out',
                           error_file_dir=Error_files_folder,
                           output_error_overwrite=False,
                           arg_jobind=False,
                           other_args=[['target_id', tid], ['first_injection_load_parameters_index', i]])

# bsub -q physics-short -sp 100 /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_runfiles_for_triggers.py
# bsub -q physics-short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_runfiles_for_triggers.py
