# #!/usr/bin/env python3
#
# import os
# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
# import file_operations as fop
# from pipeline_parameters import *
# from runfiles_parameters import *
#
# tid_to_run_file = '/home/labs/barakz/orynai/' + Run_name + '/threshold_crossing_kid_info.txt'
# with open(tid_to_run_file, 'r') as ttrf:
#     trig_ids = ttrf.readlines()
# trig_ids = [tid.split()[0] for tid in trig_ids]
#
# Output_location_wexac = '/home/labs/barakz/Collaboration-Kepler/orynai/output_' + Run_name
# # # noinspection DuplicatedCode
# for tid in trig_ids:
#     for i in range(int(N_slides_total // N_slides_chunk)  # how many times we need to repeat timeslides
#                    + int(bool(N_slides_total % N_slides_chunk))):  # stupid way to make np.ceil (otherwise numpy import)
#         # noinspection DuplicatedCode
#         fop.write_job_file(code_file=None,
#                            code_filename=Slides_runfile_filename,
#                            code_file_dir=Run_files_folder,
#                            jobname=Slides_job_filename_prefix + '_' + tid + '.bsub',
#                            job_file=None,
#                            job_filename=Slides_job_filename_prefix + '_' + tid + '_' + str(i) + '.bsub',
#                            job_file_dir=Job_files_folder,
#                            queue=Queue,
#                            mem_per_job_gb=Mem_per_job_gb,
#                            array_indices=None,
#                            output_file=None,
#                            output_filename=Slides_output_filename_prefix + '_' + tid + '.out',
#                            output_file_dir=Output_files_folder,
#                            error_file=None,
#                            error_filename=Slides_error_filename_prefix + '_' + tid + '.out',
#                            error_file_dir=Error_files_folder,
#                            output_error_overwrite=False,
#                            arg_jobind=False,
#                            other_args=[['target_id', tid]])
#
#     # f_sl = [ff for ff in os.listdir(os.path.join('/home/labs/barakz/Collaboration-Kepler/orynai/output_run_8', tid[:4], tid)) if 'mes_slides' in ff]
#     #
#     # if len(f_sl) > 0:
#     #     n_slides = int(f_sl[0][f_sl[0].index('mes_slides_') + len('mes_slides_'): f_sl[0].index('.')])
#     # else:
#     #     n_slides = 0
#     #
#     #     # noinspection DuplicatedCode
#     #     if n_slides < N_slides_total_for_now:
#     #         for i in range(int((N_slides_total_for_now-n_slides) // N_slides_chunk)  # how many times we need to repeat timeslides
#     #                        + int(bool((N_slides_total_for_now-n_slides) % N_slides_chunk))):  # stupid way to make np.ceil (otherwise numpy import)
#     #             # noinspection DuplicatedCode
#     #             fop.write_job_file(code_file=None,
#     #                                code_filename=Slides_runfile_filename,
#     #                                code_file_dir=Run_files_folder,
#     #                                jobname=Slides_job_filename_prefix + '_' + tid + '.bsub',
#     #                                job_file=None,
#     #                                job_filename=Slides_job_filename_prefix + '_' + tid + '_' + str(i) + '.bsub',
#     #                                job_file_dir=Job_files_folder,
#     #                                queue=Queue,
#     #                                mem_per_job_gb=Mem_per_job_gb,
#     #                                array_indices=None,
#     #                                output_file=None,
#     #                                output_filename=Slides_output_filename_prefix + '_' + tid + '.out',
#     #                                output_file_dir=Output_files_folder,
#     #                                error_file=None,
#     #                                error_filename=Slides_error_filename_prefix + '_' + tid + '.out',
#     #                                error_file_dir=Error_files_folder,
#     #                                output_error_overwrite=False,
#     #                                arg_jobind=False,
#     #                                other_args=[['target_id', tid]])
#
#     # if (len([ff for ff in os.listdir(os.path.join(Output_location_wexac, tid[:4], tid))
#     #         if Injection_parameters_filename_postfix in ff]) == 0
#     #     and len([ff for ff in os.listdir(os.path.join(Output_location_wexac, tid[:4], tid))
#     #              if Slides_stats_filename_postfix in ff]) > 0):
#     # if (len([ff for ff in os.listdir(os.path.join(Output_location_wexac, tid[:4], tid))
#     #          if Slides_stats_filename_postfix in ff]) > 0):
#     # if (len([ff for ff in os.listdir(os.path.join(Output_location_wexac, tid[:4], tid))
#     #          if Injection_parameters_filename_postfix in ff]) > 0):
#     #     # noinspection DuplicatedCode
#     #     fop.write_job_file(code_file=None,
#     #                        code_filename=Injection_parameters_runfile_filename,
#     #                        code_file_dir=Run_files_folder,
#     #                        jobname=Injection_parameters_job_filename_prefix+'_'+tid+'.bsub',
#     #                        job_file=None,
#     #                        job_filename=Injection_parameters_job_filename_prefix+'_'+tid+'.bsub',
#     #                        job_file_dir=Job_files_folder,
#     #                        queue=Queue,
#     #                        mem_per_job_gb=Mem_per_job_gb,
#     #                        array_indices=None,
#     #                        output_file=None,
#     #                        output_filename=Injection_parameters_output_filename_prefix+'_'+tid+'.out',
#     #                        output_file_dir=Output_files_folder,
#     #                        error_file=None,
#     #                        error_filename=Injection_parameters_error_filename_prefix+'_'+tid+'.out',
#     #                        error_file_dir=Error_files_folder,
#     #                        output_error_overwrite=False,
#     #                        arg_jobind=False,
#     #                        other_args=[['target_id', tid]])
#
#     # for i in np.arange(int(N_injections_total // N_injections_chunk)  # how many times we need to repeat
#     #                    + int(bool(N_injections_total % N_injections_chunk))):
#     #     fop.write_job_file(code_file=None,
#     #                        code_filename=Injection_detection_runfile_filename,
#     #                        code_file_dir=Run_files_folder,
#     #                        jobname=Injection_detection_job_filename_prefix+'_'+tid+'_'+str(i)+'.bsub',
#     #                        job_file=None,
#     #                        job_filename=Injection_detection_job_filename_prefix+'_'+tid+'_'+str(i)+'.bsub',
#     #                        job_file_dir=Job_files_folder,
#     #                        queue=Queue,
#     #                        mem_per_job_gb=Mem_per_job_gb,
#     #                        array_indices=None,
#     #                        output_file=None,
#     #                        output_filename=Injection_detection_output_filename_prefix+'_'+tid+'_'+str(i)+'.out',
#     #                        output_file_dir=Output_files_folder,
#     #                        error_file=None,
#     #                        error_filename=Injection_detection_error_filename_prefix+'_'+tid+'_'+str(i)+'.out',
#     #                        error_file_dir=Error_files_folder,
#     #                        output_error_overwrite=False,
#     #                        arg_jobind=False,
#     #                        other_args=[['target_id', tid], ['first_injection_load_parameters_index', i]])
#
#     # if len([ff for ff in os.listdir(os.path.join(Output_location_wexac, tid[:4], tid)) if Injections_filename_postfix in ff]) > 0:
#     #     # noinspection DuplicatedCode
#     # fop.write_job_file(code_file=None,
#     #                    code_filename=Plotting_runfile_filename,
#     #                    code_file_dir=Run_files_folder,
#     #                    jobname=Plotting_job_filename_prefix+'_'+tid+'.bsub',
#     #                    job_file=None,
#     #                    job_filename=Plotting_job_filename_prefix+'_'+tid+'.bsub',
#     #                    job_file_dir=Job_files_folder,
#     #                    queue=Queue,
#     #                    mem_per_job_gb=Mem_per_job_gb,
#     #                    array_indices=None,
#     #                    output_file=None,
#     #                    output_filename=Plotting_output_filename_prefix+'_'+tid+'.out',
#     #                    output_file_dir=Output_files_folder,
#     #                    error_file=None,
#     #                    error_filename=Plotting_error_filename_prefix+'_'+tid+'.out',
#     #                    error_file_dir=Error_files_folder,
#     #                    output_error_overwrite=False,
#     #                    arg_jobind=False,
#     #                    other_args=[['target_id', tid]])
#
#     # if len([ff for ff in os.listdir(os.path.join(Output_location_wexac, tid[:4], tid)) if Injections_filename_postfix in ff]) > 0:
#     #     # noinspection DuplicatedCode
#     #     fop.write_job_file(code_file=None,
#     #                        code_filename=Target_pixels_veto_runfile_filename,
#     #                        code_file_dir=Run_files_folder,
#     #                        jobname=Target_pixels_veto_job_filename_prefix+'_'+tid+'.bsub',
#     #                        job_file=None,
#     #                        job_filename=Target_pixels_veto_job_filename_prefix+'_'+tid+'.bsub',
#     #                        job_file_dir=Job_files_folder,
#     #                        queue=Queue,
#     #                        mem_per_job_gb=Mem_per_job_gb,
#     #                        array_indices=None,
#     #                        output_file=None,
#     #                        output_filename=Target_pixels_veto_output_filename_prefix+'_'+tid+'.out',
#     #                        output_file_dir=Output_files_folder,
#     #                        error_file=None,
#     #                        error_filename=Target_pixels_veto_error_filename_prefix+'_'+tid+'.out',
#     #                        error_file_dir=Error_files_folder,
#     #                        output_error_overwrite=False,
#     #                        arg_jobind=False,
#     #                        other_args=[['target_id', tid]])
#
#
#
# # bsub -q physics-short -sp 100 /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_runfiles_for_triggers.py
# # bsub -q physics-short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/create_runfiles_for_triggers.py
