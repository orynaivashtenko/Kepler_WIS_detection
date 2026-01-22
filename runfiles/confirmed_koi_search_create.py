#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import pickle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from pipeline_parameters import Kepler_bright_candidate_ses, N_slides_total, N_slides_chunk, N_injections_total, N_injections_chunk
from runfiles.runfiles_parameters import *
from locations import Run_outputs_folder, Run_name, Table_koi_file, Run_technical_folder
import file_operations as fop

Kepler_confirmed_koi_run_folder = os.path.join(Run_outputs_folder, 'kepler_confirmed_' + str(Run_name))
Kepler_confirmed_koi_technical_folder = os.path.join(Run_technical_folder, str(Run_name) + '_confirmed')
Kepler_confirmed_koi_output_folder = os.path.join(Kepler_confirmed_koi_run_folder, 'output_confirmed_' + str(Run_name))
Kepler_confirmed_koi_plots_folder = os.path.join(Kepler_confirmed_koi_run_folder, 'plots_confirmed_' + str(Run_name))
Kepler_confirmed_koi_problems_folder = os.path.join(Kepler_confirmed_koi_run_folder, 'problems_confirmed_' + str(Run_name))
Kepler_confirmed_koi_main_search_runfile_filename = 'confirmed_koi_runfile_main_search.py'
Kepler_confirmed_koi_slides_runfile_filename = 'confirmed_koi_runfile_timeslides.py'
Kepler_confirmed_koi_slides_clean_runfile_filename = 'confirmed_koi_runfile_timeslides_clean.py'
Kepler_confirmed_koi_injection_parameters_runfile_filename = 'confirmed_koi_runfile_injection_parameter_generation.py'
Kepler_confirmed_koi_injection_detection_runfile_filename = 'confirmed_koi_runfile_injection_detection.py'
Kepler_confirmed_koi_download_target_pixels_runfile_filename = Download_data_code_filename
Kepler_confirmed_koi_pre_slides_tce_veto_runfile_filename = 'confirmed_koi_runfile_pre_slides_tce_veto.py'
Kepler_confirmed_koi_jobs_search_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'jobs_search')
Kepler_confirmed_koi_jobs_slides_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'jobs_slides')
Kepler_confirmed_koi_jobs_slides_clean_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'jobs_slides_clean')
Kepler_confirmed_koi_jobs_injection_parameters_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'jobs_injection_parameters')
Kepler_confirmed_koi_jobs_injections_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'jobs_injections')
Kepler_confirmed_koi_jobs_download_target_pixels_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'jobs_download_target_pixels')
Kepler_confirmed_koi_jobs_tce_veto_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'jobs_pre_slides_tce_veto')
Kepler_confirmed_koi_output_files_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'output_files')
Kepler_confirmed_koi_error_files_folder = os.path.join(Kepler_confirmed_koi_technical_folder, 'error_files')
Output_files_folder = Kepler_confirmed_koi_output_files_folder
Error_files_folder = Kepler_confirmed_koi_error_files_folder
Kepler_confirmed_koi_best_periodic_scores_filename_postfix = 'best_periodic_scores_confirmed_koi'
Kepler_confirmed_koi_threshold_crossing_list_filename = 'threshold_crossing_kid_info_kepler_confirmed_koi.txt'
Kepler_confirmed_koi_threshold_g_crossing_list_filename = 'threshold_g_crossing_kid_info_kepler_confirmed_koi.txt'
Kepler_confirmed_koi_threshold_crossing_list_multiple_triggers_filename = 'threshold_crossing_list_multiple_kepler_confirmed_koi.txt'
Kepler_confirmed_koi_threshold_g_crossing_list_multiple_triggers_filename = 'threshold_g_crossing_list_multiple_kepler_confirmed_koi.txt'
Kepler_confirmed_koi_p_planet_table = 'p_planet_table_confirmed_koi.ecsv'

Confirmed_to_run_koi_dict_file = os.path.join(Kepler_confirmed_koi_run_folder, 'confirmed_to_run_koi_dict.pickle')

if __name__ == "__main__":
    for path_dir in [Kepler_confirmed_koi_run_folder, Kepler_confirmed_koi_technical_folder,
                     Kepler_confirmed_koi_output_folder, Kepler_confirmed_koi_problems_folder, Kepler_confirmed_koi_plots_folder,
                     Kepler_confirmed_koi_jobs_search_folder, Kepler_confirmed_koi_jobs_slides_folder,
                     Kepler_confirmed_koi_jobs_slides_clean_folder, Kepler_confirmed_koi_jobs_injection_parameters_folder,
                     Kepler_confirmed_koi_jobs_download_target_pixels_folder, Kepler_confirmed_koi_jobs_tce_veto_folder,
                     Kepler_confirmed_koi_jobs_injections_folder, Kepler_confirmed_koi_output_files_folder,
                     Kepler_confirmed_koi_error_files_folder]:
        if not os.path.exists(path_dir) or not os.path.isdir(path_dir):
            os.mkdir(path_dir, mode=0o777)

    koi_catalog_path = Table_koi_file
    pickling_file_path = Confirmed_to_run_koi_dict_file
    # Opening KOI table and making a dictionary:
    with open(koi_catalog_path, 'r') as fil1:
        lines = fil1.readlines()
        lines = [line.strip().split(',') for line in lines if line.strip()[0] != '#']
        headers = lines[0]
        lines = lines[1:]

        koi_info_dict = {}
        for key in headers:
            koi_info_dict[key] = []
        for line in lines:
            for i in range(len(headers)):
                koi_info_dict[headers[i]].append(line[i])

        for j in range(len(koi_info_dict['kepid'])):
            koi_info_dict['kepid'][j] = koi_info_dict['kepid'][j].rjust(9, '0')

    # Filtering and making a per-target-id dictionary of things that we want to search:
    target_confirmed_kid_dict = {}
    for i in range(len(koi_info_dict['kepid'])):
        add_entry = False
        if koi_info_dict['koi_disposition'][i] == 'CONFIRMED':
            add_entry = True
        if koi_info_dict['koi_max_mult_ev'][i] == '' or float(koi_info_dict['koi_max_mult_ev'][i]) > 15:  # pp.Kepler_bright_candidate_mes:
            add_entry = False
        if koi_info_dict['koi_max_sngle_ev'][i] == '' or float(koi_info_dict['koi_max_sngle_ev'][i]) > Kepler_bright_candidate_ses:
            add_entry = False
        if float(koi_info_dict['koi_period'][i]) < 30:
            add_entry = False
        if float(koi_info_dict['koi_period'][i]) > 500:
            add_entry = False
        if add_entry:
            tid = koi_info_dict['kepid'][i].rjust(9, '0')
            if tid not in target_confirmed_kid_dict.keys():
                target_confirmed_kid_dict[tid] = {}
                for key in koi_info_dict.keys():
                    target_confirmed_kid_dict[tid][key] = []
            for key in koi_info_dict.keys():
                target_confirmed_kid_dict[tid][key].append(koi_info_dict[key][i])

    # # Saving the dictionary:
    # if pickling_file_path[-7:] != '.pickle':
    #     pickling_file_path = pickling_file_path + '.pickle'
    # with open(pickling_file_path, 'wb') as pickle_file:
    #     pickle.dump(target_confirmed_kid_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    Queue_submit = 'short'
    Queue_submit_2 = 'physics-short'
    # Making the jobs:
    for target_id in list(target_confirmed_kid_dict.keys()):
        for koi_index in [0]:  # range(len(target_confirmed_kid_dict[target_id]['kepoi_name'])):
            # -> Do it only for one koi per star. by that, we leave 2 kois unprocessed.
            # -> we do it because all the naming and saving notations were not prepared for multiple kois per star.
            naming_addition = '_' + target_id + '_' + str(koi_index)
            # fop.write_job_file(code_file=None,
            #                    code_filename=Kepler_confirmed_koi_main_search_runfile_filename,
            #                    code_file_dir=Run_files_folder,
            #                    jobname=Main_search_job_prefix + naming_addition,
            #                    job_file=None,
            #                    job_filename=Main_search_job_prefix + naming_addition + '.bsub',
            #                    job_file_dir=Kepler_confirmed_koi_jobs_search_folder,
            #                    queue=Queue_submit_2,
            #                    mem_per_job_gb=Mem_per_job_gb,
            #                    array_indices=None,
            #                    output_file=None,
            #                    output_filename=Main_search_job_prefix + naming_addition + '.out',
            #                    output_file_dir=Output_files_folder,
            #                    error_file=None,
            #                    error_filename=Main_search_job_prefix + naming_addition + '.err',
            #                    error_file_dir=Error_files_folder,
            #                    output_error_overwrite=False,
            #                    arg_jobind=False,
            #                    other_args=[['target_id', target_id], ['koi_index', koi_index]])
            # Slides job files:
            per = target_confirmed_kid_dict[target_id]['koi_period'][koi_index]
            ftt = target_confirmed_kid_dict[target_id]['koi_time0bk'][koi_index]
            # for i in range(int(N_slides_total // N_slides_chunk)  # how many times we need to repeat timeslides
            #                + int(bool(N_slides_total % N_slides_chunk))):  # stupid way to make np.ceil (otherwise numpy import)
            #     fop.write_job_file(code_file=None,
            #                        code_filename=Kepler_confirmed_koi_slides_runfile_filename,
            #                        code_file_dir=Run_files_folder,
            #                        jobname=Slides_job_prefix + naming_addition,
            #                        job_file=None,
            #                        job_filename=Slides_job_prefix + naming_addition + '_' + str(i) + '.bsub',
            #                        job_file_dir=Kepler_confirmed_koi_jobs_slides_folder,
            #                        queue=Queue_submit,
            #                        mem_per_job_gb=Mem_per_job_gb,
            #                        array_indices=None,
            #                        output_file=None,
            #                        output_filename=Slides_job_prefix + naming_addition + '.out',
            #                        output_file_dir=Output_files_folder,
            #                        error_file=None,
            #                        error_filename=Slides_job_prefix + naming_addition + '.err',
            #                        error_file_dir=Error_files_folder,
            #                        output_error_overwrite=False,
            #                        arg_jobind=False,
            #                        other_args=[['target_id', target_id], ['period', per]])
            #     fop.write_job_file(code_file=None,
            #                        code_filename=Kepler_confirmed_koi_slides_clean_runfile_filename,
            #                        code_file_dir=Run_files_folder,
            #                        jobname=Slides_clean_job_prefix + naming_addition,
            #                        job_file=None,
            #                        job_filename=Slides_clean_job_prefix + naming_addition + '_' + str(i) + '.bsub',
            #                        job_file_dir=Kepler_confirmed_koi_jobs_slides_clean_folder,
            #                        queue=Queue_submit,
            #                        mem_per_job_gb=Mem_per_job_gb,
            #                        array_indices=None,
            #                        output_file=None,
            #                        output_filename=Slides_clean_job_prefix + naming_addition + '.out',
            #                        output_file_dir=Output_files_folder,
            #                        error_file=None,
            #                        error_filename=Slides_clean_job_prefix + naming_addition + '.err',
            #                        error_file_dir=Error_files_folder,
            #                        output_error_overwrite=False,
            #                        arg_jobind=False,
            #                        other_args=[['target_id', target_id], ['period', per], ['ftt', ftt]])
            # fop.write_job_file(code_file=None,
            #                    code_filename=Kepler_confirmed_koi_injection_parameters_runfile_filename,
            #                    code_file_dir=Run_files_folder,
            #                    jobname=Injection_parameters_job_prefix + naming_addition,
            #                    job_file=None,
            #                    job_filename=Injection_parameters_job_prefix + naming_addition + '.bsub',
            #                    job_file_dir=Kepler_confirmed_koi_jobs_injection_parameters_folder,
            #                    queue=Queue_submit_2,
            #                    mem_per_job_gb='7.9',
            #                    array_indices=None,
            #                    output_file=None,
            #                    output_filename=Injection_parameters_job_prefix + naming_addition + '.out',
            #                    output_file_dir=Output_files_folder,
            #                    error_file=None,
            #                    error_filename=Injection_parameters_job_prefix + naming_addition + '.err',
            #                    error_file_dir=Error_files_folder,
            #                    output_error_overwrite=False,
            #                    arg_jobind=False,
            #                    other_args=[['target_id', target_id], ['period', per]])
            # for i in list(range((int(N_injections_total // N_injections_chunk)  # how many times we need to repeat
            #                     + int(bool(N_injections_total % N_injections_chunk))))):
            #     fop.write_job_file(code_file=None,
            #                        code_filename=Kepler_confirmed_koi_injection_detection_runfile_filename,
            #                        code_file_dir=Run_files_folder,
            #                        jobname=Injection_detection_job_prefix + naming_addition + '_' + str(i),
            #                        job_file=None,
            #                        job_filename=Injection_detection_job_prefix + naming_addition + '_' + str(i) + '.bsub',
            #                        job_file_dir=Kepler_confirmed_koi_jobs_injections_folder,
            #                        queue=Queue_submit,
            #                        mem_per_job_gb=Mem_per_job_gb,
            #                        array_indices=None,
            #                        output_file=None,
            #                        output_filename=Injection_detection_job_prefix + naming_addition + '.out',
            #                        output_file_dir=Output_files_folder,
            #                        error_file=None,
            #                        error_filename=Injection_detection_job_prefix + naming_addition + '.err',
            #                        error_file_dir=Error_files_folder,
            #                        output_error_overwrite=False,
            #                        arg_jobind=False,
            #                        other_args=[['target_id', target_id], ['first_injection_load_parameters_index', i]])

            # fop.write_job_file(code_file=None,
            #                    code_filename=Kepler_confirmed_koi_download_target_pixels_runfile_filename,
            #                    code_file_dir=Run_files_folder,
            #                    jobname=Download_data_job_prefix + naming_addition,
            #                    job_file=None,
            #                    job_filename=Download_data_job_prefix + naming_addition + '.bsub',
            #                    job_file_dir=Kepler_confirmed_koi_jobs_download_target_pixels_folder,
            #                    queue=Queue_submit,
            #                    mem_per_job_gb=Mem_per_job_gb,
            #                    array_indices=None,
            #                    output_file=None,
            #                    output_filename=Download_data_job_prefix + naming_addition + '.out',
            #                    output_file_dir=Output_files_folder,
            #                    error_file=None,
            #                    error_filename=Download_data_job_prefix + naming_addition + '.err',
            #                    error_file_dir=Error_files_folder,
            #                    output_error_overwrite=False,
            #                    arg_jobind=False,
            #                    other_args=[['target_id', target_id]])
            #
            fop.write_job_file(code_file=None,
                               code_filename=Kepler_confirmed_koi_pre_slides_tce_veto_runfile_filename,
                               code_file_dir=Run_files_folder,
                               jobname=Pre_slides_tce_veto_job_prefix + naming_addition,
                               job_file=None,
                               job_filename=Pre_slides_tce_veto_job_prefix + naming_addition + '.bsub',
                               job_file_dir=Kepler_confirmed_koi_jobs_tce_veto_folder,
                               queue=Queue_submit,
                               mem_per_job_gb='6.9',
                               array_indices=None,
                               output_file=None,
                               output_filename=Pre_slides_tce_veto_job_prefix + naming_addition + '.out',
                               output_file_dir=Output_files_folder,
                               error_file=None,
                               error_filename=Pre_slides_tce_veto_job_prefix + naming_addition + '.err',
                               error_file_dir=Error_files_folder,
                               output_error_overwrite=False,
                               arg_jobind=False,
                               other_args=[['target_id', target_id]])

# bsub -q short -sp 100  source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/confirmed_koi_search_create.py
# bsub -q physics-short source ~/.bashrc; conda activate planets_env_amd_epyc; /home/projects/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/confirmed_koi_search_create.py
