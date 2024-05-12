#!/usr/bin/env python3

import os

Slides_mes_inv_filename_postfix = 'scramble_mes_inversion'
Slides_mes_filename_postfix = 'scramble_mes'
Slides_pers_inv_filename_postfix = 'scramble_pers_inversion'
Slides_pers_filename_postfix = 'scramble_pers'
Slides_first_times_inv_filename_postfix = 'scramble_first_times_inversion'
Slides_first_times_filename_postfix = 'scramble_first_times'

postfixes_list = [
    Slides_mes_inv_filename_postfix,
    Slides_mes_filename_postfix,
    Slides_pers_inv_filename_postfix,
    Slides_pers_filename_postfix,
    Slides_first_times_inv_filename_postfix,
    Slides_first_times_filename_postfix
]

tid_to_run_file = '/home/labs/barakz/Collaboration-Kepler/orynai/tids_to_run_first_timeslides_runs_6_7_pastro_0_9.txt'
with open(tid_to_run_file, 'r') as ttrf:
    trig_ids = ttrf.readlines()
trig_ids = [tid.strip() for tid in trig_ids]

Output_location_wexac = '/home/labs/barakz/Collaboration-Kepler/orynai/output_run_7'
for tid in trig_ids:
    fold = os.path.join(Output_location_wexac, tid[:4], tid)
    for fl in os.listdir(fold):
        for postfix in postfixes_list:
            if postfix in fl:
                os.remove(os.path.join(fold, fl))
                break

# bsub -q physics-short /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/runfiles/remover_file.py
