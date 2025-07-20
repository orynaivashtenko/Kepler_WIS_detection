#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 2 2024

@author: orynaivashtenko
"""
import numpy as np
import os
import pickle
import time

import pipeline_parameters
############################################################################################################################################
# MAKING SOME DICTIONARIES AND OTHER FILES USED IN THE PIPELINE
############################################################################################################################################

from pipeline_parameters import (Koi_dict_filename, Ttv_dict_filename, Villanova_dict_filename,
                                 Limb_darkening_array_filename, Flares_file)

from locations import (Earth_pointing_times_file, Quarter_start_times_file, Quarter_end_times_file,
                       Villanova_catalog_file, Table_koi_file, Suspicious_frequency_reasons_triggers_file,
                       Threshold_crossing_list_file, Suspicious_frequency_filtered_triggers_file,
                       Villanova_periods_phases_kmags_relevant_file, Villanova_kic_relevant_file)
from catalog_functions import (Column, Kic_retrieval_fields, Villanova_table_fields, query_kic,
                               write_trigger_table)

Default_files_folder = '/Users/orynai/Work/Kepler_Project/Kepler_Earths/KeplerFiles'
Default_koi_catalog_filename = 'KOI_table_2024.csv'  # 'cumulative_2022.10.18_03.41.17.csv'  # TODO replace by the new one
# TODO make the quality mask thing
# TODO include here the earth pointing times etc
# TODO prepare handy villanova and koi for postprocess check
# todo enter all that in the tce class
# todo check what is going on with the limb darkening
# todo check whether there were some preparations of priors and if yes include
Default_koi_dict_filename = Koi_dict_filename
Default_ttv_catalog_filename = 'TTV_Catalog_Table3_O-C.txt'
Default_ttv_dict_filename = Ttv_dict_filename
Default_villanova_catalog_filename = 'villanova_catalog_timings.txt'
Default_villanova_dict_filename = Villanova_dict_filename


def make_koi_catalog_dict(koi_catalog_path=os.path.join(Default_files_folder, Default_koi_catalog_filename),
                          pickling_file_path=os.path.join(Default_files_folder, Default_koi_dict_filename)):
    """

    # This file was produced by the NASA Exoplanet Archive  http://exoplanetarchive.ipac.caltech.edu
    # Tue Oct 18 03:41:17 2022
    #
    # User preference: *
    #
    # COLUMN kepid:          KepID
    # COLUMN kepoi_name:     KOI Name
    # COLUMN kepler_name:    Kepler Name
    # COLUMN koi_disposition: Exoplanet Archive Disposition
    # COLUMN koi_pdisposition: Disposition Using Kepler Data
    # COLUMN koi_score:      Disposition Score
    # COLUMN koi_fpflag_nt:  Not Transit-Like False Positive Flag
    # COLUMN koi_fpflag_ss:  Stellar Eclipse False Positive Flag
    # COLUMN koi_fpflag_co:  Centroid Offset False Positive Flag
    # COLUMN koi_fpflag_ec:  Ephemeris Match Indicates Contamination False Positive Flag
    # COLUMN koi_period:     Orbital Period [days]
    # COLUMN koi_time0bk:    Transit Epoch [BKJD]
    # COLUMN koi_eccen:      Eccentricity
    # COLUMN koi_longp:      Long. of Periastron [deg]
    # COLUMN koi_impact:     Impact Parameter
    # COLUMN koi_duration:   Transit Duration [hrs] Duration is measured from first contact between the planet and star until last contact.
    # COLUMN koi_ingress:    Ingress Duration [hrs]
    # COLUMN koi_depth:      Transit Depth [ppm]
    # COLUMN koi_ror:        Planet-Star Radius Ratio
    # COLUMN koi_prad:       Planetary Radius [Earth radii]
    # COLUMN koi_sma:        Orbit Semi-Major Axis [au]
    # COLUMN koi_incl:       Inclination [deg]
    # COLUMN koi_ldm_coeff2: Limb Darkening Coeff. 2
    # COLUMN koi_ldm_coeff1: Limb Darkening Coeff. 1
    # COLUMN koi_max_sngle_ev: Maximum Single Event Statistic
    # COLUMN koi_max_mult_ev: Maximum Multiple Event Statistic
    # COLUMN koi_model_snr:  Transit Signal-to-Noise
    # COLUMN koi_num_transits: Number of Transits
    # COLUMN koi_tce_plnt_num: TCE Planet Number
    # COLUMN koi_tce_delivname: TCE Delivery
    # COLUMN koi_bin_oedp_sig: Odd-Even Depth Comparision Statistic
    # COLUMN koi_steff:      Stellar Effective Temperature [K]
    # COLUMN koi_slogg:      Stellar Surface Gravity [log10(cm/s**2)]
    # COLUMN koi_srad:       Stellar Radius [Solar radii]
    # COLUMN koi_smass:      Stellar Mass [Solar mass]
    # COLUMN ra:             RA [decimal degrees]
    # COLUMN dec:            Dec [decimal degrees]
    # COLUMN koi_kepmag:     Kepler-band [mag]

    """
    if pickling_file_path[-7:] != '.pickle':
        pickling_file_path = pickling_file_path + '.pickle'
    with open(koi_catalog_path, 'r') as fil1:
        lines = fil1.readlines()
        lines = [line.strip().split(',') for line in lines if line.strip()[0] != '#']
        headers = lines[0]
        lines = lines[1:]

        dict_koi = {}
        for key in headers:
            dict_koi[key] = []
        for line in lines:
            for i in range(len(headers)):
                dict_koi[headers[i]].append(line[i])

        for j in range(len(dict_koi['kepid'])):
            dict_koi['kepid'][j] = dict_koi['kepid'][j].rjust(9, '0')

        with open(pickling_file_path, 'wb') as pickle_file:
            pickle.dump(dict_koi, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def make_ttv_dict(koi_catalog_dict_path=os.path.join(Default_files_folder, Default_koi_dict_filename),
                  ttv_catalog_path=os.path.join(Default_files_folder, Default_ttv_catalog_filename),
                  pickling_ttv_catalog_dict_path=os.path.join(Default_files_folder, Default_ttv_dict_filename)):
    """
    # WARNING: SOMETIMES, IN THE TTV LIST SOME TRANSITS ARE MISSION, E.G. 011853905 time 1373.75

    Takes the TTV catalog, makes from it a dict. But it does not have a target_id name, only a useless KOI identifier.
    Therefore, this function opens the KOI catalog dict generated by another function from this file, and finds the corresponding
    name for each entry. It is added as a new key-value pair in the dict. Then, the dict is pickled.
    :param koi_catalog_dict_path:
    :type koi_catalog_dict_path:
    :param ttv_catalog_path:
    :type ttv_catalog_path:
    :param pickling_ttv_catalog_dict_path:
    :type pickling_ttv_catalog_dict_path:
    :return:
    :rtype:
    """
    with open(koi_catalog_dict_path, 'rb') as koi_catalog_dict_file:
        koi_info_dict = pickle.load(koi_catalog_dict_file)
    kois_from_koi_dict = [str(float(kk[1:])) for kk in koi_info_dict['kepoi_name']]

    with open(ttv_catalog_path, 'r') as fil1:
        lines = fil1.readlines()
        lines = [line.split() for line in lines if line.strip()[0] != '#']

    headers = 'KOI, n, T, O_C, O_C_Err, TDV, TDV_Err, TPV, TPV_Err, Flag_outlier, Flag_collision'.split(', ')
    # explanations = 'Kepler Object of Interest number; Transit number; Expected transit time of the linear ephemeris (BJD-2454900); \
    # O-C difference min; O-C uncertainty min; Fractional duration variation; \
    # TDV uncertainty; Fractional depth variation; TPV uncertainty; \
    # Outlier flag. Sum of: 0 = ok, 1 = pvalue > 0.025, 2= local TTV, 4= global TTV, 8= global TDV, 16 =global TPV, 32 = TTV,TDV or TPV \
    # uncertainty outlier; Collision flag. 0=ok, 1 = another relatively large (>0.25 area of transit) interfering transit close by \
    # (<2*transit duration + 1 interfering transit duration)'

    dict_ttv = {}
    for key in headers:
        dict_ttv[key] = []
    for line in lines:
        for i in range(len(headers)):
            dict_ttv[headers[i]].append(line[i])

    dict_ttv['kepid'] = []
    for i in range(len(dict_ttv['KOI'])):
        arg = kois_from_koi_dict.index(dict_ttv['KOI'][i])
        dict_ttv['kepid'].append(koi_info_dict['kepid'][arg])

    with open(pickling_ttv_catalog_dict_path, 'wb') as pickle_file:
        pickle.dump(dict_ttv, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def make_villanova_dict(villanova_catalog_path=os.path.join(Default_files_folder, Default_villanova_catalog_filename),
                        pickling_dict_path=os.path.join(Default_files_folder, Default_villanova_dict_filename)):
    """
    Beware:
    Kepler time is measured in BJD - 2454833
    Villanova phases are in BJD - 2400000
    We corrected for this in `make_villanova_periods_phases_kmags_for_relevant_periods_postprocessing`
    :param villanova_catalog_path:
    :type villanova_catalog_path:
    :param pickling_dict_path:
    :type pickling_dict_path:
    :return:
    :rtype:
    """
    # villanova_catalog_timings.txt
    if pickling_dict_path[-7:] != '.pickle':
        pickling_dict_path = pickling_dict_path + '.pickle'
    with open(villanova_catalog_path, 'r') as fil1:
        lines = fil1.readlines()
        lines = [line.strip().split(',') for line in lines if len(line.strip()) > 0 and line.strip()[0] != '#']
        headers = lines[0]
        headers = [hh for hh in headers if hh != '']  # the line might end with ',', and then the last header will be ''
        lines = lines[1:]

        dict_villanova = {}
        for key in headers:
            dict_villanova[key] = []
        for line in lines:
            for i in range(len(headers)):
                if headers[i] == 'KIC':
                    dict_villanova[headers[i]].append(line[i].rjust(9, '0'))
                else:
                    dict_villanova[headers[i]].append(line[i])

        with open(pickling_dict_path, 'wb') as pickle_file:
            pickle.dump(dict_villanova, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def make_villanova_periods_phases_kmags_for_relevant_periods_postprocessing(
        villanova_periods_phases_kmags_relevant_file=Villanova_periods_phases_kmags_relevant_file,
        return_results=False, save_to_file=True):
    # ################################################################################################
    # LOADING VILLANOVA
    # # KIC,period,period_err,bjd0,bjd0_err,morph,GLon,GLat,kmag,Teff,SC,

    with open(Villanova_catalog_file, 'r') as vil_file:
        lines = vil_file.readlines()
    lines = [(line.strip()).split(',') for line in lines if ((len(line.strip()) > 0) and (line.strip())[0] != '#')]
    vil_kids = np.array([line[0].rjust(9, '0') for line in lines])
    vil_periods = np.array([float(line[1]) for line in lines])  # error is about 0.0000002 days
    vil_phases = np.array([float(line[3]) for line in lines])  # error is about one lc exposure time
    vil_kmags = np.array([float(line[8]) for line in lines])
    vil_relevant_period_indices = vil_periods > 1  # we keep harmonics, but we do not need some 0.1 day periods

    vil_kids = vil_kids[vil_relevant_period_indices]
    vil_periods = vil_periods[vil_relevant_period_indices]
    vil_phases = vil_phases[vil_relevant_period_indices]
    vil_kmags = vil_kmags[vil_relevant_period_indices]
    # Kepler time is measured in BJD - 2454833
    # Villanova phases are in BJD - 2400000
    vil_phases = vil_phases + 2400000. - 2454833

    if save_to_file:
        np.save(villanova_periods_phases_kmags_relevant_file, np.vstack((vil_kids, vil_periods, vil_phases, vil_kmags)),
                allow_pickle=True)
    if return_results:
        return vil_kids, vil_periods, vil_phases, vil_kmags


def make_villanova_table_from_kic_for_postprocessing(
        villanova_kic_relevant_file=Villanova_kic_relevant_file,
        kic_retrieval_fields=Kic_retrieval_fields,
        villanova_table_fields=Villanova_table_fields):
    """
    Some Villanova targets cannot be identified with a Gaia ID, apparently exactly because they are
    binaries and not one star. Therefore, to pinpoint them as Kepler targets, we use KIC (Kepler Input Catalog).
    For example, if we need their coordinates
    :param villanova_kic_relevant_file:
    :type villanova_kic_relevant_file:
    :param kic_retrieval_fields:
    :type kic_retrieval_fields:
    :param villanova_table_fields:
    :type villanova_table_fields:
    :return:
    :rtype:
    """
    # ################################################################################################
    # LOADING VILLANOVA
    # # KIC,period,period_err,bjd0,bjd0_err,morph,GLon,GLat,kmag,Teff,SC,

    vil_kids, vil_periods, vil_phases, vil_kmags = make_villanova_periods_phases_kmags_for_relevant_periods_postprocessing(
        return_results=True, save_to_file=False)

    objects_table = query_kic(target_id_list=vil_kids, return_table=True,
                              save_table_to_file=True, table_file=villanova_kic_relevant_file,
                              table_filename=None, table_folder=None,
                              kic_retrieval_fields=kic_retrieval_fields)
    for name in ['period', 't0']:
        col = Column(data=np.empty(len(objects_table['target_id']), dtype=float),
                     name=name, dtype=float,
                     description=name, unit='days',
                     format=None, meta=None, copy=False, copy_indices=True)
        objects_table.add_column(col)

    for i_kid, kid in enumerate(np.array(objects_table['target_id'])):
        kid = str(kid).rjust(9, '0')
        if kid in vil_kids:
            ind = np.where(vil_kids == kid)[0][0]
            objects_table['period'][i_kid] = vil_periods[ind]
            objects_table['t0'][i_kid] = vil_phases[ind]
            objects_table['kepmag'][i_kid] = vil_kmags[ind]

    write_trigger_table(trigger_table=objects_table, file_path=villanova_kic_relevant_file, overwrite=True)


def make_limb_darkening_coefficients_array(limb_darkening_table_file, save_file=False):
    """
    Opens the Table: J/A+A/510/A21/table2.dat.gz (http://cdsarc.cds.unistra.fr)
    and returns the array of [Teff,logg,M/H,a,b], where a, b are the u1, u2 quadratic limb darkening coeffs.
    #Limb_darkening_full_table_file = '/Collaboration-Kepler/KeplerFiles/limb_darkening_kepler_table.txt'
    :param limb_darkening_table_file: The full path to the txt table with coefficients
    :type limb_darkening_table_file: str
    :return: The array of relevant parameters and the limb darkening quadratic coefficients.
    :rtype: numpy.ndarray
    :param save_file: Whether to save the file. Default is yes
    :type save_file: bool
    """
    with open(limb_darkening_table_file, 'r') as fil:
        # 'Table: J/A+A/510/A21/table2.dat.gz (http://cdsarc.cds.unistra.fr)'
        lines = fil.readlines()
        # Teff|logg|  M/H|      u|      a|      b|     c2|     c3|     c4|     a1|     a2|     a3|     a4
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line[0].isnumeric()]
    lines = [line.split('|') for line in lines]
    lines = [[line[0], line[1], line[2], line[4], line[5]] for line in lines]
    # [Teff,logg,M/H,a,b]
    lines = [[float(e.strip()) for e in line] for line in lines]
    lines = np.array(lines)
    lines = lines[lines[:, 0] != 1.]
    if save_file:
        np.save(os.path.join(os.path.dirname(limb_darkening_table_file)), 'limb_darkening_kepler_array.npy')
    return lines


def get_flares_table(file_flares_table=Flares_file):
    """
    'Title: Stellar Flares Observed in Long Cadence Data From the Kepler Mission\n',
    'Authors: Van Doorsselaere T., Shariati H., Debosscher J. \n',
    'Table: Catalog of Kepler flare stars\n',
    '================================================================================\n',
    'Byte-by-byte Description of file: apjsaa8f9at1_mrt.txt\n',
    '--------------------------------------------------------------------------------\n',
    '   Bytes Format Units   Label   Explanations\n',
    '--------------------------------------------------------------------------------\n',
    '   1-  8 I8     ---     KIC     Kepler Input Catalog identifier\n',
    '  10- 20 F11.6  d       Date    Kepler mission time of flare (1)\n',
    '  22- 35 F14.6  ---     a       Amplitude in flux units\n',
    '  37- 46 F10.6  d-1     k       Flare decay \n',
    '  48- 54 F7.1   K       Teff    Effective temperature\n',
    '  56- 62 F7.3   solRad  Radius  Radius\n',
    '  64- 68 F5.3   [cm/s2] log(g)  log surface gravity\n',
    '--------------------------------------------------------------------------------\n',
    'Note (1): BJD-2454833.\n',
    '--------------------------------------------------------------------------------\n',
    """
    with open(file_flares_table, 'r') as fil1:
        flare_data = fil1.readlines()
    ind_first = 1 + len(flare_data) - [i for i in range(len(flare_data)) if flare_data[-i].strip()[0] == '-'][
        0]  # finding where the headers end
    flare_data = flare_data[ind_first:]
    flare_data = [line.strip().split() for line in flare_data if len(line) > 5]
    flare_data = [[e[0].rjust(9, '0'), float(e[1]), float(e[2]), float(e[3]), float(e[4]), float(e[5]), float(e[6])] for e in flare_data]
    return flare_data


def get_flares_data_for_target(target_id, file_flares_table=Flares_file):
    """
    '   Bytes Format Units   Label   Explanations\n',
    '--------------------------------------------------------------------------------\n',
    '  10- 20 F11.6  d       Date    Kepler mission time of flare, BJD-2454833.\n',
    '  22- 35 F14.6  ---     a       Amplitude in flux units\n',
    '  37- 46 F10.6  d-1     k       Flare decay \n',
    '  48- 54 F7.1   K       Teff    Effective temperature\n',
    '  56- 62 F7.3   solRad  Radius  Radius\n',
    '  64- 68 F5.3   [cm/s2] log(g)  log surface gravity\n',
    :param target_id:
    :type target_id:
    :param file_flares_table:
    :type file_flares_table:
    :return:
    :rtype:
    """
    flare_data = get_flares_table(file_flares_table=file_flares_table)
    flare_data_tids = np.array([e[0] for e in flare_data])
    target_id_inds = np.where(flare_data_tids == target_id)[0]
    flare_data_target = np.array([flare_data[index][1:] for index in target_id_inds])
    return flare_data_target


def prepare_simulated_star(reference_target_id='009833034', input_folder=pipeline_parameters.Data_location,
                           output_folder=pipeline_parameters.Output_location,):
    from search_pipeline import KeplerLightCurve
    from utils import interpolate_time
    klc = KeplerLightCurve(target_id=reference_target_id, data_location=input_folder,
                           output_location=output_folder,
                           remove_zeroth_quarter=False, remove_quarter_17=False, cut_edge_nans=True)
    klc.interpolate_fluxes()
    times = [interpolate_time(np.array(klc.times[q]), mask=None) for q in range(len(klc.fluxes))]
    fluxes = [10000. * np.random.normal(1, 1e-4, len(klc.fluxes[q])) for q in range(len(klc.fluxes))]
    sim_star_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulations', 'simulated_white_star.npy')
    np.save(sim_star_file, np.array([times, fluxes], dtype=object), allow_pickle=True)


