#!/usr/bin/env python3

import os
core_num = "1"
os.environ["OMP_NUM_THREADS"] = core_num
os.environ["OPENBLAS_NUM_THREADS"] = core_num
os.environ["MKL_NUM_THREADS"] = core_num
os.environ["VECLIB_MAXIMUM_THREADS"] = core_num
os.environ["NUMEXPR_NUM_THREADS"] = core_num
import numpy as np
import locations as locs
from utils import check_transit_overlap
# from tce_parameters import (Neighbors_search_radius_pix, Pixel_scale, Gaia_fields)

from astropy.table import Table, Column, vstack
from astropy.io import fits


# #####################################################################################################################
# TCE FUNCTIONS
# #####################################################################################################################


def ensure_list_str(data):
    if isinstance(data, str):
        data = [data]
    elif isinstance(data, int):
        data = [str(data)]
    return [str(d) for d in data]


def write_trigger_table(trigger_table, file_path=locs.Trigger_table_file, overwrite=True,
                        trigger_table_format=locs.Trigger_table_format,
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


def read_gaia_kepler_table(gaia_kepler_file_path=locs.Gaia_kepler_file):
    """
    Getting the Gaia-Kepler cross-match info
    '/Users/orynai/Work/Kepler_Project/Kepler_Earths/KeplerFiles/kepler_dr3_good.fits'
    """
    # Getting the Gaia-Kepler cross-match info
    gaia_hdu_list = fits.open(gaia_kepler_file_path, memmap=True)
    gaia_kepler_table = Table(gaia_hdu_list[1].data)
    gaia_hdu_list.close()
    return gaia_kepler_table


def read_table(file_path, fill_values=None, table_format=None):
    if table_format is None:
        if os.path.splitext(file_path)[-1] == '.ecsv':
            table_format = "ascii.ecsv"
        elif os.path.splitext(file_path)[-1] == '.csv':
            table_format = "csv"
    if table_format is None:
        if fill_values is not None:
            res = Table.read(file_path, fill_values=fill_values)
        else:
            res = Table.read(file_path)
    else:
        if fill_values is not None:
            res = Table.read(file_path, format=table_format, fill_values=fill_values)
        else:
            res = Table.read(file_path, format=table_format)
    return res


def read_trigger_table(file_path=locs.Trigger_table_file, trigger_table_format=locs.Trigger_table_format,
                       fill_values=None):
    """
    Getting the Kepler ID list
    'trlist_df.csv'

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
    trigger_table = read_table(file_path=file_path, fill_values=fill_values, table_format=trigger_table_format)
    # Make sure trigger_id is in the correct format:
    tid_list = np.array(trigger_table['target_id'], dtype=str)
    tid_list = [e.rjust(9, '0') for e in tid_list]
    trigger_table['target_id'] = tid_list
    return trigger_table


def get_neighbors_file_name(target_id, neighbors_folder=locs.Neighbors_folder, search_radius_pix=None):
    if search_radius_pix is None:
        from tce_parameters import Neighbors_search_radius_pix
        search_radius_pix = Neighbors_search_radius_pix
    return os.path.join(neighbors_folder, str(target_id) + '_neigh_' + str(search_radius_pix) + '_pix.ecsv')


def read_neighbor_info(target_id, neighbors_file_path=None, neighbors_file_name=None, neighbors_folder=locs.Neighbors_folder,
                       search_radius_pix=None):
    if search_radius_pix is None:
        from tce_parameters import Neighbors_search_radius_pix
        search_radius_pix = Neighbors_search_radius_pix
    if neighbors_file_path is None:
        if neighbors_file_name is None:
            neighbors_file_name = get_neighbors_file_name(target_id=target_id, neighbors_folder=neighbors_folder,
                                                          search_radius_pix=search_radius_pix)
        neighbors_file_path = os.path.join(neighbors_folder, neighbors_file_name)
    neighbors_table = Table.read(neighbors_file_path, format="ascii.ecsv")  # , fill_values=None)
    return neighbors_table


def read_gaia_trigger_table(file_path=locs.Gaia_trigger_table_file):
    return read_table(file_path)


def retrieve_line_for_target_id(target_id, table, only_if_found=False):
    tid_list = ensure_list_str(target_id)
    if 'target_id' in table.keys():
        key_target_id = 'target_id'
    else:
        key_target_id = 'kepid'
    if not only_if_found:
        inds = [np.where(np.array(table[key_target_id]).astype(int) == int(tid))[0][0] for tid in tid_list]
    else:
        inds = [np.where(np.array(table[key_target_id]).astype(int) == int(tid))[0] for tid in tid_list]
        inds = [ind[0] for ind in inds if len(ind) > 0]
    return table[inds]


def read_gaia_kepler_crossmatch_table_for_target_id(tid_list, gaia_kepler_file_path=locs.Gaia_kepler_file,
                                                    only_if_found=False):
    tid_list = ensure_list_str(tid_list)
    gaia_kepler_table = read_gaia_kepler_table(gaia_kepler_file_path=gaia_kepler_file_path)
    return retrieve_line_for_target_id(tid_list, gaia_kepler_table, only_if_found=only_if_found)


def read_gaia_table_for_target_id(tid_list, file_path=locs.Gaia_trigger_table_file):
    tid_list = ensure_list_str(tid_list)
    gaia_trigger_table = read_gaia_trigger_table(file_path=file_path)
    return retrieve_line_for_target_id(tid_list, gaia_trigger_table)


def match_koi(target_id, period, ftt, n_transits, min_overlap_fraction=0.5, match_periods=True, period_tolerance=0.1,
              koi_dict_file=locs.Koi_dict_file, ttv_dict_file=locs.Ttv_dict_file, overlap_timing_error=0.08):
    """
    Gets matching KOI of this star to the trigger that is specified by `period`, `ftt`
    min_overlap_fraction is not important because we already require period match, and the transit time wll match somehow. it can
    be small sometimes because we cut the first and the last semi-quarters
    """
    period = float(period)
    ftt = float(ftt)
    res_dict = None
    is_koi, koi_dict, ttv_dict = get_koi_ttv_dicts(kepler_id=target_id, koi_dict_file=koi_dict_file, ttv_dict_file=ttv_dict_file)
    if is_koi:
        transit_overlap_mask = np.zeros(n_transits).astype(bool)
        if len(koi_dict) > 0:
            res_dict = {key: [] for key in koi_dict.keys()}
            for ind in range(len(koi_dict['koi_disposition'])):
                per_str = koi_dict['koi_period'][ind]
                if per_str == '':
                    per_str = 'nan'
                    per_float = np.nan
                    ftt_float = np.nan
                else:
                    per_float = float(per_str)
                    per_str = str(np.round(per_float, 1))
                    ftt_float = float(koi_dict['koi_time0bk'][ind])

                if not match_periods or np.abs(per_float - period) < period_tolerance:
                    transit_overlap_mask = check_transit_overlap(period=period, ftt=ftt,
                                                                 period_ref=per_float, ftt_ref=ftt_float,
                                                                 n_transits=n_transits, timing_error=overlap_timing_error)
                    if np.sum(transit_overlap_mask) > min_overlap_fraction * len(transit_overlap_mask):
                        for key in res_dict.keys():
                            res_dict[key].append(koi_dict[key][ind])
    return res_dict


def get_koi_ttv_dicts(kepler_id, koi_dict_file=locs.Koi_dict_file, ttv_dict_file=locs.Ttv_dict_file):
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


def read_kepler_tce_table(kepler_tce_table_file=locs.Kepler_tce_table_file, return_headers=True):
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


def get_kepler_tce_dict(kepler_id, kepler_tce_table_file=locs.Kepler_tce_table_file):
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


def test_gaia_table(table):
    # Some columns are masked columns, so they have fields `data` and `mask`. We use True for bad
    # because the masked will be automatically replaced by False, and the masked
    # cannot be classified as bad because there is just no information.

    res_masks = []
    res_reasons = []
    res_masks.append(table['phot_bp_rp_excess_factor'] > 3.)
    res_reasons.append('phot_bp_rp_excess_factor')
    if hasattr(table['ruwe'], 'mask'):
        res_masks.append(table['ruwe'].filled(0).data > 1.37)
    else:
        res_masks.append(table['ruwe'] > 1.37)
    res_reasons.append('ruwe')
    res_masks.append((table['astrometric_excess_noise'] > 0) & (table['astrometric_excess_noise_sig'] > 2.))
    res_reasons.append('astrometric_excess_noise')
    if hasattr(table['radial_velocity_error'], 'mask'):
        radial_velocity_error = table['radial_velocity_error'].filled(0)
    else:
        radial_velocity_error = table['radial_velocity_error']
    if hasattr(table['rv_chisq_pvalue'], 'mask'):
        rv_chisq_pvalue = table['rv_chisq_pvalue'].filled(1e20)
    else:
        rv_chisq_pvalue = table['rv_chisq_pvalue']
    if hasattr(table['rv_renormalised_gof'], 'mask'):
        rv_renormalised_gof = table['rv_renormalised_gof'].filled(0)
    else:
        rv_renormalised_gof = table['rv_renormalised_gof']
    res_masks.append((radial_velocity_error.data > 10.) | ((rv_chisq_pvalue.data < 0.01)
                                                           & (rv_renormalised_gof.data > 4)))  # & (table['rv_nb_transits'] >= 10) ))
    res_reasons.append('radial_velocity')
    res_masks.append(table['non_single_star'] != 0)
    res_reasons.append('non_single_star')
    res_masks.append(table['phot_variable_flag'] != 'NOT_AVAILABLE')
    res_reasons.append('phot_variable')

    return res_masks, res_reasons


# #####################################################################################################################
# FUNCTIONS TO WORK WITH KIC (Kepler Input Catalog)
# Were designed to get sky coordinates of villanova stars which were not well identified in Gaia
# #####################################################################################################################

def hms2deg(hh, mm, ss):
    """
    to go from hours, minutes, seconds to degrees
    """
    return (hh * 60 * 60 + mm * 60 + ss) / (24 * 60 * 60) * 360


def degamasec2deg(deg, am, asec):
    return deg + am / 60 + asec / 3600


Kic_retrieval_fields = (
    ('kic_kepler_id', 'target_id'),
    ('kic_degree_ra', 'ra'),
    ('kic_dec', 'dec'),
    ('kic_pmra', 'pmra'),
    ('kic_pmdec', 'pmdec'),
    ('kic_kepmag', 'kepmag'),
    ('kic_gmag', 'phot_g_mean_mag'),
    ('kic_variable', 'phot_variable_flag'),
    ('kic_teff', 'teff'),
    ('kic_logg', 'logg'),
    ('kic_feh', 'feh'),
    ('kic_radius', 'radius'),
    ('kic_pq', 'phot_quality'),
    ('kic_parallax', 'parallax'))

Villanova_table_fields = tuple([f[1] for f in Kic_retrieval_fields] + ['period', 't0'])


def query_kic(target_id_list, return_table=True,
              saving_file=locs.Villanova_kic_query_relevant_file,
              saving_folder=None, saving_filename='kic_query_out.txt',
              save_table_to_file=True, table_file=None, table_filename='kic_query_table.ecsv',
              table_folder=None,
              kic_retrieval_fields=Kic_retrieval_fields):
    """

    """
    import urllib.request
    import os
    if saving_file is None:
        if saving_folder is None:
            saving_folder = os.getcwd()
        saving_file = os.path.join(saving_folder, saving_filename)

    chunk_len = 200
    if len(target_id_list) > chunk_len:
        target_id_chunks = [target_id_list[i:i + chunk_len] for i in range(0, len(target_id_list), chunk_len)]
    else:
        target_id_chunks = [target_id_list]

    objects_table = None
    for i_ch, tids_chunk in enumerate(target_id_chunks):
        kic_ids_str = ','.join(tids_chunk)
        url = 'https://archive.stsci.edu/kepler/kic10/search.php?&outputformat=CSV&action=Search'
        # url += '&kic_kepler_id='+str(kic_kepler_id)
        url += '&kic_kepler_id=' + str(kic_ids_str)
        url += '&selectedColumnsCsv=' + ','.join([f[0] for f in kic_retrieval_fields])
        # FOR SOME REASON, IF I SPECIFY kic_ra, IT GIVES ANSWER IN DECIMAL HOURS, AND
        # kic_degree_ra GIVES IN hh mm ss.
        res = urllib.request.urlretrieve(url, saving_file)

        objects_table_1 = build_kic_table_from_query_result(
            query_result_file=saving_file, kic_retrieval_fields=kic_retrieval_fields,
            save_table_to_file=False, table_file=table_file,
            table_filename=table_filename, table_folder=table_folder)
        if i_ch == 0:
            objects_table = objects_table_1
        else:
            objects_table = vstack([objects_table, objects_table_1])
        os.remove(saving_file)

    if save_table_to_file:
        if table_file is None:
            if table_folder is None:
                table_folder = os.getcwd()
            table_file = os.path.join(table_folder, table_filename)
        write_trigger_table(trigger_table=objects_table, file_path=table_file)

    if return_table:
        return objects_table


def build_kic_table_from_query_result(query_result_file, kic_retrieval_fields=Kic_retrieval_fields,
                                      save_table_to_file=True, table_file=None,
                                      table_filename='kic_query_table.ecsv', table_folder=None):
    if table_file is None:
        if table_folder is None:
            table_folder = os.getcwd()
        table_file = os.path.join(table_folder, table_filename)

    with open(query_result_file, 'r') as fil:
        lines = fil.readlines()

    headers = lines[0].split(',')
    units = lines[1].split(',')
    lines = lines[2:]
    lines = [line.split(',') for line in lines]
    for names in kic_retrieval_fields:
        name = names[1]

    objects_table = Table(masked=True)
    for i in range(len(kic_retrieval_fields)):
        name = kic_retrieval_fields[i][1]
        col = Column(data=np.empty(0, dtype='<U100' if name == 'target_id' else float),
                     name=name,
                     dtype='<U100' if name == 'target_id' else float,
                     description=headers[i],
                     unit=units[i],
                     format=None, meta=None, copy=False, copy_indices=True)
        objects_table.add_column(col)

    for line in lines:
        objects_table.add_row()
        for i in range(len(kic_retrieval_fields)):
            name = kic_retrieval_fields[i][1]
            val = line[i].strip()
            if val != '':
                if name == 'ra':
                    val = val.split(' ')
                    val = hms2deg(hh=float(val[0]), mm=float(val[1]), ss=float(val[2]))
                elif name == 'dec':
                    val = val.split(' ')
                    val = degamasec2deg(deg=float(val[0]), am=float(val[1]), asec=float(val[2]))
                elif name == 'target_id':
                    val = str(val).rjust(9, '0')
                else:
                    val = float(line[i].strip())
                objects_table[name][-1] = val

    if save_table_to_file:
        write_trigger_table(trigger_table=objects_table, file_path=table_file, overwrite=True)

    return objects_table
