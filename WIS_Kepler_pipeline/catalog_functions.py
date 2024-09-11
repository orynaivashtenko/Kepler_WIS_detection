#!/usr/bin/env python3

import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num
import numpy as np
import tce_parameters as tp

from astropy.table import Table
from astropy.io import fits

# #####################################################################################################################
# TCE FUNCTIONS
# #####################################################################################################################
from tce_parameters import (Pixel_scale, Neighbors_search_radius_pix, Gaia_kepler_file,
                            Gaia_fields, Neighbors_folder, Gaia_trigger_file)
from pipeline_parameters import (Koi_dict_file, Ttv_dict_file, Kepler_tce_table_file)


def ensure_list_str(data):
    if isinstance(data, str):
        data = [data]
    elif isinstance(data, int):
        data = [str(data)]
    return [str(d) for d in data]


def write_trigger_table(trigger_table, file_path=tp.Trigger_table_file, overwrite=True,
                        trigger_table_format=tp.Trigger_table_format,
                        fill_values=None):
    """
    , fill_values=[(aio.ascii.masked, '')]
    :param trigger_table:
    :type trigger_table:
    :param file_path:
    :type file_path:
    :param overwrite:
    :type overwrite:
    :param trigger_table_format:
    :type trigger_table_format:
    :param fill_values:
    :type fill_values:
    :return: None
    :rtype: None
    """
    if fill_values is not None:
        trigger_table.write(file_path, format=trigger_table_format, overwrite=overwrite, fill_values=fill_values)
    else:
        trigger_table.write(file_path, format=trigger_table_format, overwrite=overwrite)


def read_trigger_table(file_path=tp.Trigger_table_file,
                       trigger_table_format=tp.Trigger_table_format,
                       fill_values=None):
    """
    fill_values=[(aio.ascii.masked, '')]
    :param file_path:
    :type file_path:
    :param trigger_table_format:
    :type trigger_table_format:
    :param fill_values:
    :type fill_values:
    :return:
    :rtype: astropy.table.table.Table
    """
    trigger_table = Table.read(file_path, format=trigger_table_format, fill_values=fill_values)
    return trigger_table


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


def get_koi_ttv_dicts(kepler_id, koi_dict_file=Koi_dict_file, ttv_dict_file=Ttv_dict_file):
    import pickle
    with open(koi_dict_file, 'rb') as koi_catalog_dict_file:
        koi_info_dict = pickle.load(koi_catalog_dict_file)
    target_id = str(kepler_id).rjust(9, '0')

    is_koi = False
    koi_dict = {}
    ttv_dict = {}
    if target_id in koi_info_dict['kepid']:
        is_koi = True
        inds = np.where(np.array(koi_info_dict['kepid']) == target_id)[0]
        for ind in inds:
            for key in koi_info_dict:
                if key != 'kepid':
                    if key not in koi_dict:
                        koi_dict[key] = []
                    koi_dict[key].append(koi_info_dict[key][ind])

        with open(ttv_dict_file, 'rb') as ttv_catalog_dict_file:
            ttv_info_dict = pickle.load(ttv_catalog_dict_file)
        for koi_name in koi_dict['kepoi_name']:
            koi_inds = np.where((np.array(ttv_info_dict['KOI']) == str(float(koi_name[1:])))
                                & (np.array(ttv_info_dict['kepid']) == target_id))[0]
            if len(koi_inds) > 0:
                for key in ttv_info_dict:
                    if key != 'kepid':
                        if key not in ttv_dict:
                            ttv_dict[key] = []
                        ttv_dict[key].append(np.array(ttv_info_dict[key])[koi_inds])
        # Units of Kepler times : unit = 'BJD - 2454833'
        # Units in the TTV catalog :  BJD-2454900
        if 'T' in ttv_dict.keys():  # it may be not there if the object is koi but not ttv
            for j in range(len(ttv_dict['T'])):
                ttv_dict['T'][j] = ttv_dict['T'][j].astype(float) + 2454900 - 2454833
    return is_koi, koi_dict, ttv_dict


def read_kepler_tce_table(kepler_tce_table_file=Kepler_tce_table_file, return_headers=True):
    """
    '# COLUMN kepid:          KepID\n',
    '# COLUMN tce_plnt_num:   Planet Number\n',
    '# COLUMN tce_period:     Orbital Period [days]\n',
    '# COLUMN tce_time0bk:    Transit Epoch [BKJD]\n',
    '# COLUMN tce_ror:        Planet-Star Radius Ratio\n',
    '# COLUMN tce_duration:   Transit Duration [hrs]\n',
    '# COLUMN tce_depth:      Transit Depth [ppm]\n',
    '# COLUMN tce_model_snr:  Transit Signal-to-Noise (SNR)\n',
    '# COLUMN tce_prad:       Planetary Radius [Earth radii]\n',
    '# COLUMN tce_max_mult_ev: Multiple Event Statistic (MES)\n',
    '# COLUMN boot_fap:       Bootstrap False Alarm Probability\n',
    '# COLUMN boot_mesthresh: Bootstrap MES Threshold\n'
    """
    with open(kepler_tce_table_file, 'r') as table_tce_file:
        tce_table = table_tce_file.readlines()
    tce_table = [line for line in tce_table if line.strip()[0] != '#']
    headers = (tce_table[0].strip()).split(',')
    tce_table = tce_table[1:]  # remove header
    tce_table = [(line.strip()).split(',') for line in tce_table]
    if return_headers:
        return tce_table, headers
    else:
        return tce_table


def get_kepler_tce_dict(kepler_id, kepler_tce_table_file=Kepler_tce_table_file):
    # ################################################################################################################
    # TCEs
    tce_table, headers = read_kepler_tce_table(kepler_tce_table_file=kepler_tce_table_file, return_headers=True)
    target_id = str(kepler_id).rjust(9, '0')

    tce_kids = np.array([line[headers.index('kepid')].rjust(9, '0') for line in tce_table])
    is_tce = False
    tce_dict = {}
    if target_id in tce_kids:
        is_tce = True
        indices = np.where(tce_kids == target_id)[0]
        for ind in indices:
            for i_key, key in enumerate(headers):
                if key != 'kepid':
                    if key not in tce_dict:
                        tce_dict[key] = []
                    tce_dict[key].append(tce_table[ind][i_key])
    return is_tce, tce_dict

