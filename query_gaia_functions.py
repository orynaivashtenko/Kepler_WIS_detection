#!/usr/bin/env python3

"""
This module contains functions for querying the Gaia database for Kepler targets. It is used to get the
Gaia info for the targets and to query for the neighbors of the targets.
--------------------------------------------------------------------------------------------
This file is to be used only on the machine which supports querying Gaia (Marvin and Wexac do not).
--------------------------------------------------------------------------------------------
This module uses functions from the `catalog_functions.py` file. The module `catalog_functions.py`
does not involve querying, it will operate with the tables obtained from querying done here.
This module should be used once to get those tables, and then not used again.

To copy the tables to Marvin or Wexac:
--------------------------------------------------------------------------------------------
scp -r -O /Users/orynai/Work/Kepler_Project/Intermediate_tasks/Forming_catalog/Neighbors_run_11_1
orynai@accessv01.wexac.weizmann.ac.il:/home/projects/barakz/Collaboration-Kepler/orynai
# use -O option if it refuses to copy the folder. This is to use the old version of scp which for
# some reason did not have this problem.
scp /Users/orynai/Work/Kepler_Project/Intermediate_tasks/Forming_catalog/tids_table_gaia.ecsv
orynai@accessv01.wexac.weizmann.ac.il:/home/projects/barakz/Collaboration-Kepler/orynai
--------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# GAIA TARGET INFO
# To get the Gaia info for the target ids listed in `target_id_list`, do the following:
    tids_table_gaia, failed_tids_reasons = get_gaia_table_for_kepler_id(target_id_list)
    tids_table_gaia.write('tids_table_gaia.ecsv', format='ascii.ecsv', overwrite=True)

# To read this table:
tids_table_gaia = Table.read('tids_table_gaia.ecsv', format="ascii.ecsv")  #, fill_values=None)
# BE CAREFUL ABOUT MASKED

# If the catalog was saved as astropy.Table with csv format:
trigger_table = Table.read('trlist_df.csv', format="csv")

# Can (and preferred) to use the `read_trigger_table` function from `catalog_functions.py`

# --------------------------------------------------------------------------------------------
# NEIGHBORS
# To query for the neighbors, do the following:
    query_neighbors(tids_table_gaia=None, target_id_list=None, search_radius_pix=Neighbors_search_radius_pix, pixel_scale=Pixel_scale,
                    gaia_fields=Gaia_fields, neighbors_folder=Neighbors_folder)

# To read the neighbors table:
    neighbors_file = [os.path.join(neighbors_folder, ff) for ff in os.listdir(neighbors_folder) if tid in ff][0]
    neighbors_table = Table.read(neighbors_file, format="ascii.ecsv")  #, fill_values=None)
    # BE CAREFUL ABOUT MASKED

"""

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from tce_parameters import (Pixel_scale, Neighbors_search_radius_pix, Gaia_fields, Neighbors_folder,
                            Gaia_trigger_table_file)

from astropy import units
from astropy.coordinates import SkyCoord
from astropy.table import Table
import warnings
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Select Data Release 3


from catalog_functions import (ensure_list_str, read_gaia_kepler_crossmatch_table_for_target_id,
                               get_neighbors_file_name)


def query_gaia_id_for_kepler_id(tid_list):
    """
    Query the Gaia database for the Gaia ID of the Kepler targets.
    :param tid_list:
    :type tid_list:
    :return:
    :rtype:
    """
    tid_list = ensure_list_str(tid_list)
    gaia_text = 'Gaia DR3'
    gaia_id_list = []
    failed_tids_info = []
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for i_tid, tid in enumerate(tid_list):
            try:
                result_table = Simbad.query_objectids("KID " + tid)  # ("KIC " + tid)
                res_gaia = [e[0] for e in result_table if gaia_text in e[0]]
                if len(result_table.errors) > 0:
                    failed_tids_info.append(tid + ' : Search failed')  # failed_tids.append(result_table.errors)
                elif len(res_gaia) > 0:
                    if len(res_gaia) > 1:
                        failed_tids_info.append(tid + ' : More than one Gaia result!')
                    else:
                        gaia_id = res_gaia[0][len(gaia_text):].strip()
                        gaia_id_list.append(gaia_id)
                else:
                    failed_tids_info.append(tid + ' : No Gaia ID found')
            except:
                failed_tids_info.append(tid + ' : Search failed')
    return gaia_id_list, failed_tids_info


def query_gaia_id(gaia_id_list):
    gaia_id_list = ensure_list_str(gaia_id_list)
    query = "SELECT * FROM gaiadr3.gaia_source gs WHERE gs.source_id in (" + ", ".join(gaia_id_list) + ");"
    # job = Gaia.launch_job_async(query, verbose=False)
    job = Gaia.launch_job(query, verbose=False)
    table_gaia = job.get_results()
    return table_gaia


def get_gaia_id_from_gaia_kepler_crossmatch(tid_list, only_if_found=False):
    tid_list = ensure_list_str(tid_list)
    gaia_kepler_table = read_gaia_kepler_crossmatch_table_for_target_id(tid_list, only_if_found=only_if_found)
    if 'SOURCE_ID' in gaia_kepler_table.keys():
        return list(gaia_kepler_table['SOURCE_ID'])
    elif 'source_id' in gaia_kepler_table.keys():
        return list(gaia_kepler_table['source_id'])
    else:
        raise ValueError('`source_id` key not found in the table')


def get_gaia_table_for_kepler_id(tid_list, write_to_file=False,
                                 file_path=Gaia_trigger_table_file, overwrite_file=True):
    tid_list = ensure_list_str(tid_list)
    gaia_id_list, failed_tids_info = query_gaia_id_for_kepler_id(tid_list)
    failed_tids = [e[:9] for e in failed_tids_info]
    tids_found = [tid for tid in tid_list if tid not in failed_tids]

    failed_gaia_ids = get_gaia_id_from_gaia_kepler_crossmatch(failed_tids, only_if_found=True)
    # # We checked that the coords here and in the gaia-kepler cross-match are the same.
    # # Therefore we will use the results from the gaia-kepler cross-match for the not found tids.
    gaia_id_list = np.concatenate((gaia_id_list, failed_gaia_ids))
    gaia_id_list = gaia_id_list.astype(int)
    combined_tid_list = np.concatenate((tids_found, failed_tids))
    combined_tid_list = np.array([str(tid).rjust(9, '0') for tid in combined_tid_list])
    inds = [np.where(combined_tid_list == tid)[0][0] for tid in tid_list]
    # The full gaia_id list for all the tid_list in the correct order:
    gaia_id_list = gaia_id_list[inds]

    # Querying to get the full gaia info for those IDs:
    table_gaia = query_gaia_id(gaia_id_list)
    inds = [np.where(table_gaia['SOURCE_ID'] == int(gaia_id))[0][0] for gaia_id in gaia_id_list]
    table_gaia = table_gaia[inds]
    table_gaia.add_column(tid_list, name='kepid', index=0)

    if write_to_file:
        if os.path.splitext(file_path)[-1] == '.ecsv':
            table_gaia.write(file_path, format='ascii.ecsv', overwrite=overwrite_file)
        elif os.path.splitext(file_path)[-1] == '.csv':
            table_gaia.write(file_path, format='.csv', overwrite=overwrite_file)
        else:
            table_gaia.write(file_path, overwrite=overwrite_file)

    return table_gaia, failed_tids_info


def query_neighbors(tids_table_gaia=None, target_id_list=None, search_radius_pix=Neighbors_search_radius_pix, pixel_scale=Pixel_scale,
                    gaia_fields=Gaia_fields, neighbors_folder=Neighbors_folder):
    """
    Query the Gaia database for the neighbors of the Kepler targets.
    :param tids_table_gaia: Table containing the Gaia info for the targets. Can be obtained from `get_gaia_table_for_kepler_id`.
        Should contain the columns 'kepid', 'ra', 'dec'. If None, it will be queried for the list `target_id_list`
    :type tids_table_gaia: astropy.Table
    :param target_id_list: List of Kepler IDs for which to query the neighbors. Superseded by `tids_table_gaia`.
        Will be used if `tids_table_gaia` is None.
    :type target_id_list: array-like
    :param search_radius_pix: Search radius to query the neighbors, in pixels
    :type search_radius_pix: float
    :param pixel_scale: Size of one pixel in arcsec.
    :type pixel_scale: float
    :param gaia_fields: The fields from the Gaia table which will be saved
    :type gaia_fields: List of str
    :param neighbors_folder: Where to put all the tables for the neighbors
    :type neighbors_folder: str
    :return: Writes the astropy.Table tables for the neighbors in the `neighbors_folder`.
    :rtype: astropy.Table for every neighbor
    """
    search_radius_arcsec = search_radius_pix * pixel_scale * units.arcsec
    # ------------------------------------------------------------------
    # Getting the Gaia matches of those stars
    if tids_table_gaia is None:
        tids_table_gaia, failed_tids_reasons = get_gaia_table_for_kepler_id(target_id_list)
        # failed_tids = [e[:9] for e in failed_tids_reasons]
    # ------------------------------------------------------------------
    # Query for every tid:
    for i in range(len(tids_table_gaia)):
        tid = tids_table_gaia['kepid'][i]
        ra = tids_table_gaia['ra'][i]
        dec = tids_table_gaia['dec'][i]

        coords = str(ra) + ' ' + str(dec)  # ra and dec in degrees
        coords = SkyCoord(coords, frame='icrs', unit=(units.deg, units.deg), obstime="J2000")

        # Do the cone search for the neighbors:
        job = Gaia.cone_search_async(coordinate=coords, radius=search_radius_arcsec, table_name='gaiadr3.gaia_source')
        gaia_result = job.get_results()
        gaia_result_relevant = Table([gaia_result.columns[nam] for nam in gaia_fields])
        # Writing the file for every target neighbors:
        gaia_result_relevant.write(get_neighbors_file_name(target_id=tid, neighbors_folder=neighbors_folder,
                                                           search_radius_pix=search_radius_pix), format='ascii.ecsv', overwrite=True)


def query_kid_in_vizier(target_id, radius_arcsec=1, catalog='I/355/gaiadr3'):
    from astroquery.vizier import Vizier
    res = Vizier.query_object("KID " + target_id.rjust(9, '0'), radius=radius_arcsec * units.arcsec)[catalog]
    return res
