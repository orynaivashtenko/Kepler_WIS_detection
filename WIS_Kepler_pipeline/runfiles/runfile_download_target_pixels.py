#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from auxiliary_functions import get_correct_tid_folder, download_target_pixel_files
from locations import Target_pixels_location

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target_id", type=str, nargs=1, help="KIC number", default="003097222")
args = parser.parse_args()
target_id = args.target_id
if isinstance(target_id, list):
    target_id = target_id[0]


target_id = str(target_id).rjust(9, "0")
data_folder = get_correct_tid_folder(target_id=target_id, location=Target_pixels_location,
                                     allow_creation=True, with_prefix_folder=True)

# noinspection DuplicatedCode
for ff in os.listdir(data_folder):
    os.remove(os.path.join(data_folder, ff))

download_target_pixel_files(target_id, data_folder)
