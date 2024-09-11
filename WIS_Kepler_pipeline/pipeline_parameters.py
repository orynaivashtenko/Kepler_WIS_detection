#!/usr/bin/env python3

import numpy as np
import os
from locations import *
# from scipy.interpolate import RegularGridInterpolator

######################################################################################################################################
# PHYSICAL CONSTANTS AND RATES
# ------------------------------------------------------------------------------------------------------------------------------------

# Physical constants is CGS
R_sun = 6.9634e10  # cm #solar radius
au = 1.495978707e13  # cm #astronomical unit
M_sun = 1.989e33  # g #solar mass
G = 6.67430e-8  # dyn⋅cm2/g2 #gravitational constant
R_sun_R_e = 109.076  # sun radii to Earth radii
Default_limb_darkening_coefs = (0.47, 0.23)


# Way to use the occurrence table:
def get_occurrence_table_old(stellar_radius):
    """
    Per one star
    :param stellar_radius:
    :type stellar_radius:
    :return:
    :rtype:
    """
    # We are not putting the following occurrence rate info to a separate file because it is easier like this.
    # Generating rp according to the prior
    log_rp_edges = np.log(np.array([0.5, 1, 2, 4, 8, 20]) / stellar_radius / R_sun_R_e)
    # -> This was measured in Earth radii, we convert to stellar radii
    period_edges = 10 ** (np.linspace(1, 3, 11))[2:9]  # An exact empirical formula for the values in the table.
    occurrences_table = np.array([
        # zhu and dong, planets per 100 stars (WARNING: WE ONLY DETECT ONE PER STAR, DISTRIBUTION CAN BE DIFFERENT!)
        [4.7, 4.1, 7.2, 0.66, 0.23],  # P = 25.1  - 39.8  days
        [4.2, 4.2, 8.1, 0.58, 0.52],  # P = 39.8  - 63.1  days
        [3.1, 3.1, 6.9, 1.50, 0.44],  # P = 63.0  - 100.0 days
        [4.2, 4.2, 6.2, 1.30, 1.30],  # P = 100.0 - 158.5 days
        [4.9, 4.9, 7.5, 2.40, 1.20],  # P = 158.5 - 251.2 days
        [10.0, 10.0, 6.9, 1.70, 1.70]]  # P = 251.2 - 398.1 days
    ) / 100
    return log_rp_edges, period_edges, occurrences_table


# Way to use the occurrence table:
def get_occurrence_table():
    """
    Per one star
    :return:
    :rtype:
    """
    # We are not putting the following occurrence rate info to a separate file because it is easier like this.
    # Generating rp according to the prior
    log2_rp_re_edges = np.array([-1, 0, 1, 2, 3, np.log2(20)])  # planetary radius in Earth radii
    log10_per_edges = np.linspace(1, 3, 11)[2:9]  # days # An exact empirical formula for the values in the table.
    occurrences_table = np.array([
        # zhu and dong, planets per 100 stars (WARNING: WE ONLY DETECT ONE PER STAR, DISTRIBUTION CAN BE DIFFERENT!)
        [4.7, 4.1, 7.2, 0.66, 0.23],  # P = 25.1  - 39.8  days
        [4.2, 4.2, 8.1, 0.58, 0.52],  # P = 39.8  - 63.1  days
        [3.1, 3.1, 6.9, 1.50, 0.44],  # P = 63.0  - 100.0 days
        [4.2, 4.2, 6.2, 1.30, 1.30],  # P = 100.0 - 158.5 days
        [4.9, 4.9, 7.5, 2.40, 1.20],  # P = 158.5 - 251.2 days
        [10.0, 10.0, 6.9, 1.70, 1.70]]  # P = 251.2 - 398.1 days
    ) / 100  # -> we report per one star
    return log2_rp_re_edges, log10_per_edges, occurrences_table


def get_occurrence_table_enlarged():
    """
    """
    # We are not putting the following occurrence rate info to a separate file because it is easier like this.
    # zhu and dong, planets per 100 stars (WARNING: WE ONLY DETECT ONE PER STAR, DISTRIBUTION CAN BE DIFFERENT!)
    # I repeated some rows to extrapolate in order to interpolate
    # log2_rp_re_edges = np.array([-1, 0, 1, 2, 3, np.log2(20)])  # planetary radius in Earth radii
    log2_rp_re_edges_enlarged = np.array([-1, 0., 1, 2, 3, np.log2(20)])
    log10_per_edges_enlarged = np.array([1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8])
    occurrences_table_enlarged = np.array([
        [0.052, 0.046, 0.064, 0.0045, 0.0017],
        [0.047, 0.041, 0.072, 0.0066, 0.0023],
        [0.042, 0.042, 0.081, 0.0058, 0.0052],
        [0.031, 0.031, 0.069, 0.015,  0.0044],
        [0.042, 0.042, 0.062, 0.013,  0.013],
        [0.049, 0.049, 0.075, 0.024,  0.012],
        [0.1,   0.1,   0.069, 0.017,  0.017],
        [0.1,   0.1,   0.069, 0.017,  0.017]])
    return log2_rp_re_edges_enlarged, log10_per_edges_enlarged, occurrences_table_enlarged


# MASKING KOIs
Mask_kepler_candidates = False
Mask_kepler_bright_candidates = True
Kepler_bright_candidate_mes = 20.
Kepler_bright_candidate_ses = 8.
Mask_kepler_false_positives = False

# TARGET CLASS PARAMETERS

Min_quarters_to_process = 4  # If there are fewer quarters in the data, the search will not run
# Bad fft check:
Amp_max_bad_fft = 300.  # Limit amplitude of the fft maximum (above it the maximum is bad)
Dif_max_bad_fft = 11.  # Maximal difference between the value at point and the surrounding. Bigger is bad.

# Search
P_min_search = 30  # days
P_max_search = 500  # days
P_short_min = 10  # days - short periodicity search for verification
P_short_max = 30  # days - short periodicity search for verification
Per_min_candidate = 50.  # days  # Only consider such triggers for follow-up after the search
Baseline_rate_period = P_min_search
Rp_re_max = 11  # this number of earth radii we search for now. 11 is Jupiter
Rp_rs_max = 0.05  # this relative radius we search for now. 0.04 is like Neptune to Sun
# this one is used for the main search
# P_chunks_search = 360  # will give approx 2GB per core, I increased it when changed duty cycle. # is not used for now
# -> this one used to be when I had 25 templates. Now I have more, so we must increase
# P_chunks_search = 1000  # 1440
Per_phases_per_chunk = 1200000  # It was 1300000 in run 12, but injection search took too much memory # With 1500000 some jobs got killed
# -> instead of specifying number of chunks, it is easier to specify the time-size of the array going to the memory
# this one is used for the timeslides:
# N_periods_per_chunk = 2000  # will give 1.2GB per core, but it was when I only ran for 1 template
# N_periods_per_chunk = 80  # Now I run for the full bank
# I don't know why the previous line (with P_chunks_search = 360) is so (it would give 222 periods per chunk)
Score_candidate_search = 40.  # mes in units of SNR SQUARED TEMPLATE-MARGINALIZED
Score_candidate_postprocess = 49.
# The following `close` parameters are for the maximal score selection. Sometimes the maximal integral corresponds
# to the chunk that does not capture the true peak. But the integral does capture the true peak because of the
# period overlapping. We check whether there is a score which is no less
# than the maximum minus `Delta_integral_close` and gives the gain in statistic more than `Delta_stat_close`. If
# there is such a score, it is selected as a maximum.
Delta_integral_close = 0.2
Delta_stat_close = 1.

N_slides_total = 50000  # for one template and narrow periods for all the runs
# I thought to make chunk 50000 for one template and narrow periods per one run (about 14 hours) (rough 1 sec per one periodicity search)
# But there was a timeout, so
N_slides_chunk = 200
N_injections_total = 10000  # total number of all injections (all radii and periods etc.)
N_injections_chunk = 40  # the previous one was perhaps run_8, but now we have full bank
# N_injections_chunk = 20  # number of injections per job so that it will take 10/3 hours (120/3 sec per one inj detection)
# I decreased it from 50 to 20 in run_7 because now we need more time for veto
Per_wing_injection_and_slides_relative = 2 / 50  # Range to the left and to the right of the best period in units of period. Will be used
# to generate injections and time slides with variable periods around the best one
Per_wing_injection_and_slides_absolute = 2 / 50  # Range to the left and to the right of the best period in days. Will be used
Per_wing_injection_detection = 0.5  # days. Range to the left and to the right of the injected period used in the injection recovery
# Max_phase_mismatch_rel = 0.001  # Maximal relative phase difference for defining successful injections
Max_phase_mismatch_to_dur = 1.5  # Maximal phase difference in units of template duration for defining successful injections
Default_min_plot_stat = 10
Default_max_plot_stat = 140


#########################################################################################################################
# Class parameters
#########################################################################################################################
Histogram_mes_binwid = 1.  # Width of the bin in the MES[SNR^2] histogram of background and injections

Location_attributes = ['output_location', 'data_location', 'gaia_kepler_file', 'koi_dict_file',
                       'ttv_dict_file', 'villanova_dict_file', 'limb_darkening_array_file']

Star_attributes = ['stellar_radius', 'stellar_mass', 'kepmag', 'phot_g_mean_mag', 'parallax',
                   'gaia_abs_gmag', 'gaia_b_r', 'teff', 'ra', 'dec', 'm_1_3_r_s']


