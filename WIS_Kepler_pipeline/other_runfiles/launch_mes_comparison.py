#!/usr/bin/env python3

import os

jobs_folder = '/home/labs/barakz/orynai/run_8/mes_comparison/jobs'
for ff in os.listdir(jobs_folder):
    os.system('bsub < '+os.path.join(jobs_folder, ff))

# bsub -q physics-short -R "rusage[mem=1GB]" /home/labs/barakz/orynai/Kepler_Earths/WIS_Kepler_pipeline/other_runfiles/launch_mes_comparison.py