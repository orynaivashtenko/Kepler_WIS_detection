#!/usr/bin/env python3

import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num
import numpy as np

from astropy.table import Table
from astropy.io import fits

# #####################################################################################################################
# TCE FUNCTIONS
# #####################################################################################################################
from tce_parameters import (Pixel_scale, Neighbors_search_radius_pix, Gaia_kepler_file,
                            Gaia_fields, Neighbors_folder, Gaia_trigger_file)


def ensure_list_str(data):
    if isinstance(data, str):
        data = [data]
    elif isinstance(data, int):
        data = [str(data)]
    return [str(d) for d in data]


def read_gaia_kepler_table(gaia_kepler_file_path=Gaia_kepler_file):
    """
    Getting the Gaia-Kepler cross-match info
    '/Users/orynai/Work/Kepler_Project/Kepler_Earths/KeplerFiles/kepler_dr3_good.fits'
    """
    # Getting the Gaia-Kepler cross-match info
    gaia_hdu_list = fits.open(gaia_kepler_file_path, memmap=True)
    gaia_kepler_table = Table(gaia_hdu_list[1].data)
    gaia_hdu_list.close()
    return gaia_kepler_table


def read_table(file_path):
    if os.path.splitext(file_path)[-1] == '.ecsv':
        res = Table.read(file_path, format="ascii.ecsv")
    elif os.path.splitext(file_path)[-1] == '.csv':
        res = Table.read(file_path, format="csv")
    else:
        res = Table.read(file_path)
    return res


def read_trigger_table(file_path):
    """
    Getting the Kepler ID list
    'trlist_df.csv'
    """
    trigger_table = read_table(file_path)
    # Make sure trigger_id is in the correct format:
    tid_list = np.array(trigger_table['target_id'], dtype=str)
    tid_list = [e.rjust(9, '0') for e in tid_list]
    trigger_table['target_id'] = tid_list
    return trigger_table


def get_neighbors_file_name(target_id, neighbors_folder=Neighbors_folder, search_radius_pix=Neighbors_search_radius_pix):
    return os.path.join(neighbors_folder, str(target_id) + '_neigh_' + str(search_radius_pix) + '_pix.ecsv')


def read_neighbor_info(target_id, neighbors_file_path=None, neighbors_file_name=None, neighbors_folder=Neighbors_folder,
                       search_radius_pix=Neighbors_search_radius_pix):
    if neighbors_file_path is None:
        if neighbors_file_name is None:
            neighbors_file_name = get_neighbors_file_name(target_id=target_id, neighbors_folder=neighbors_folder,
                                                          search_radius_pix=search_radius_pix)
        neighbors_file_path = os.path.join(neighbors_folder, neighbors_file_name)
    neighbors_table = Table.read(neighbors_file_path, format="ascii.ecsv")  # , fill_values=None)
    return neighbors_table


def read_gaia_trigger_table(file_path=Gaia_trigger_file):
    return read_table(file_path)


def retrieve_line_for_target_id(target_id, table):
    tid_list = ensure_list_str(target_id)
    if 'target_id' in table.keys():
        key_target_id = 'target_id'
    else:
        key_target_id = 'kepid'
    inds = [np.where(np.array(table[key_target_id]).astype(int) == int(tid))[0][0] for tid in tid_list]
    return table[inds]


def read_gaia_kepler_crossmatch_table_for_target_id(tid_list, gaia_kepler_file_path=Gaia_kepler_file):
    tid_list = ensure_list_str(tid_list)
    gaia_kepler_table = read_gaia_kepler_table(gaia_kepler_file_path=gaia_kepler_file_path)
    return retrieve_line_for_target_id(tid_list, gaia_kepler_table)


def read_gaia_table_for_target_id(tid_list, file_path=Gaia_trigger_file):
    tid_list = ensure_list_str(tid_list)
    gaia_trigger_table = read_gaia_trigger_table(file_path=file_path)
    return retrieve_line_for_target_id(tid_list, gaia_trigger_table)
