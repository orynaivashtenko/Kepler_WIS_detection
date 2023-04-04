#!/usr/bin/env python3

import os

directory = '/Collaboration-Kepler/orynai/output_run_6'
file_no_slides = os.path.join(directory, 'tids_no_slides.txt')
file_numbers_slides = os.path.join(directory, 'tids_numbers_slides.txt')
for prefix in os.listdir(directory):
    if len(prefix)==4:
        for tid in os.listdir(os.path.join(directory, prefix)):
            f_sl = [ff for ff in os.listdir(os.path.join(directory, prefix, tid)) if 'mes_slides' in ff]
            if len(f_sl) < 0:
                with open(file_no_slides, 'a') as fil1:
                    print(tid, file=fil1)
            else:
                n_slides = int(f_sl[0][f_sl[0].index('mes_slides_')+len('mes_slides_'): f_sl[0].index('.')])
                with open(file_numbers_slides, 'a') as fil1:
                    print(tid, n_slides, file=fil1)

# bsub -q physics-short /home/labs/barakz/orynai/run_6/search_for_unsuccessful_timeslides.py
