#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:11:32 2020

@author: orynaivashtenko
version May 2024


"""
import os

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import warnings
from scipy.signal import sosfiltfilt
# import numba
from numba import njit
from scipy.stats import chi2

import template_bank.template_module as tm
import auxiliary_functions as af
import utils as ut
from utils import rfft, irfft
from search_pipeline_parameters import *
from simulations.lightcurve_simulation import get_simulated_star, N_quarters_simulation
    
import ssl

# noinspection PyUnresolvedReferences,PyProtectedMember
ssl._create_default_https_context = ssl._create_unverified_context  # this is needed to make requests work

##############################################################################
"""

Quick example of usage:
----------------------------
klc = pip.KeplerLightCurve(target_id='000757280', get_data=True, data_location='/home/orynai/Kepler_Notebooks/Data_sample_gaia_1', 
                           scramble=False, output_location='/SMALLDATA/orynai/trial_outputs')
klc.condition_data()
klc.condition_templates()
klc.detect_single_transits()
klc.periodicity_search(pmin=50, pmax=52)
----------------------------

If conditioned data is loaded:
----------------------------
klc1 = pip.KeplerLightCurve(target_id='000757280', get_data=True, data_location='/home/orynai/Kepler_Notebooks/Data_sample_gaia_1', 
                            scramble=False, output_location='/SMALLDATA/orynai/trial_outputs')
klc1.load_conditioned_data('/SMALLDATA/orynai/trial_outputs/000757280_conditioned_data.npy')
klc1.initialize_templates()
klc1.detect_single_transits()

"""

"""
Naming notations
"""

##############################################################################


class KeplerLightCurve(object):
    def __init__(self, target_id=None, times=None, fluxes=None, get_data=True,
                 data_location=Default_data_location, output_location=Default_output_location,
                 remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True,
                 m_1_3_r_s=Default_m_1_3_r_s, allow_data_folder_creation=True, allow_output_folder_creation=True,
                 simulated_white_star=False, simulated_red_star=False, n_quarters_simulation=N_quarters_simulation,
                 power_2_support=False):
        """
        Initializes a kepler light curve object. See the description in the attribute comments in the __init__.
        :param target_id: Kepler ID of the target. A 9-digit number, where some first digits may be zero.
            May be specified as integer or as string, should only contain digits.
            If `get_data`==True, target_id must be specified, it cannot be None.
        :type target_id: str or None
        :param times: List of photometry times for different quarters. Time inside quarters must be evenly
            sampled. If there is no quarters subdivision, formatting has single quarter: time=[time]
        :type times: list of np.ndarray 1d of float or np.nan
        :param fluxes: List of fluxes of Kepler PDC light curves for different quarters. Might contain numpy.nan.
            If there is no quarters subdivision, formatting has single quarter: flux=[flux]
        :type fluxes: list of np.ndarray 1d of float or np.nan
        :param get_data: Whether to take data from local or remote storage (if True) or to `times` and `fluxes`
            as input data (if False). If True, data will be searched for in the
            specified `data_location`. There should be a folder named as a 9-digit target_id.
            In this folder, fits files of Kepler data should be stored.
            If such a folder is not found, it will be created. Data will be then downloaded from
            https://archive.stsci.edu/pub/kepler/lightcurves/ and placed in this folder.
        :type get_data: bool
        :param data_location: Local path to the folder where data files are stored. The target_id will
            be searched in there. If light curves are not found in the folder, they will be downloaded.
            If folder is not found, it will be created.
        :type data_location: str
        :param output_location: Directory where the output is to be saved. The target ID named folder
            will be created if it is not there. See the description in naming notations.
        :type output_location: str
        :param remove_zeroth_quarter: Whether to remove quarter 0. This quarter is small and often bad,
            not intended for science.
        :type remove_zeroth_quarter: bool
        :param remove_quarter_17: Whether to remove quarter 17.
            This quarter is small, there is a hole, and there might be glitch. We remove it
        :type remove_quarter_17: bool
        :param m_1_3_r_s: Mass of the star in units of solar mass divided by the radius of the star to
            the power 1/3. Characterizes stellar density. If not provided, will be replaced by the default.
        :type m_1_3_r_s: float
        """

        # INITIALIZING THE STAR
        if target_id is not None:
            target_id = str(target_id).rjust(9, "0")  # Convert Kepler ID to string of appropriate length
        else:
            target_id = '0'.rjust(9, "0")
            get_data = False

        if data_location is not None:
            if get_data and (not simulated_white_star) and (not simulated_red_star):
                data_location = af.get_correct_tid_folder(target_id, data_location,
                                                          allow_creation=allow_data_folder_creation)
            else:
                data_location = None
                # -> if there is no need to download the data, no need in the folder

        if output_location is not None:
            output_location = af.get_correct_tid_folder(target_id, output_location,
                                                        allow_creation=allow_output_folder_creation)
        self.data_location = data_location
        """Local path where the data is saved"""
        self.output_location = output_location
        """Local path where the output will be saved"""
        self.target_id = target_id
        """Kepler ID of the target"""
        self.m_1_3_r_s = m_1_3_r_s
        """Mass of the star in units of solar mass divided by the radius of the star to
            the power 1/3. Characterizes stellar density"""

        # ########################################################################################################
        # GETTING DATA

        self.orig_times = None
        """Initial time arrays of quarters provided by user input or downloaded from location"""
        self.orig_fluxes = None
        """Initial flux arrays of quarters provided by user input or downloaded from location"""
        if simulated_white_star:
            if get_data:
                sim_star_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulations', 'simulated_white_star.npy')
                times, fluxes = np.load(sim_star_file, allow_pickle=True)
            else:
                np.random.seed(seed=None)
                times, fluxes = get_simulated_star(n_quarters=n_quarters_simulation)
            self.orig_times = [np.array(times[q]) for q in range(len(times))]
            self.orig_fluxes = [np.array(fluxes[q]) for q in range(len(fluxes))]
        elif simulated_red_star:
            np.random.seed(seed=None)
            times, fluxes = get_simulated_star(n_quarters=n_quarters_simulation)
            self.orig_times = [np.array(times[q]) for q in range(len(times))]
            self.orig_fluxes = [np.array(fluxes[q]) for q in range(len(fluxes))]
        elif not get_data:
            if not hasattr(times[0], '__len__'):
                times = [times]
            if not hasattr(fluxes[0], '__len__'):
                fluxes = [fluxes]
            assert len(times) == len(fluxes), 'Number of time pieces is supposed to be equal to the number of light curves!'
            for i in range(len(times)):
                assert len(times[i]) == len(fluxes[i]), 'Length of fluxes[' + str(i) + '] = ' + str(
                    len(fluxes[i])) + ' does not match its time length = ' + str(len(times[i])) + ' !'
            self.orig_times = [np.array(times[q]) for q in range(len(times))]
            self.orig_fluxes = [np.array(fluxes[q]) for q in range(len(fluxes))]
        else:
            assert target_id is not None, 'Please specify target ID!'
            self.orig_times, self.orig_fluxes = af.get_data_fun(target_id, location=data_location)
            # -> they should come out ordered
        if len(self.orig_times) < 1:
            raise AssertionError('There is no data to process!')

        # REMOVE ZEROTH AND 17TH QUARTER
        if remove_zeroth_quarter:
            if len(self.orig_times[0]) < 800:  # Make sure that the zeroth quarter is indeed the short zeroth quarter
                self.orig_times = self.orig_times[1:]
                self.orig_fluxes = self.orig_fluxes[1:]
        if len(self.orig_times) < 1:
            raise AssertionError('There is no data to process!')

        if remove_quarter_17:  # This quarter is small, there is a hole, and there might be glitch. We remove it
            if ((1570 >= np.nanmin(self.orig_times[-1])) & (
                    1570 <= np.nanmax(self.orig_times[-1]))):  # Check that Q17 is here.
                self.orig_times = self.orig_times[:-1]
                self.orig_fluxes = self.orig_fluxes[:-1]
        if len(self.orig_times) < 1:
            raise AssertionError('There is no data to process!')

        if cut_edge_nans:
            for q in range(len(self.orig_fluxes)):
                non_nans = np.where(~np.isnan(self.orig_fluxes[q]))[0]
                self.orig_fluxes[q] = self.orig_fluxes[q][non_nans[0]: non_nans[-1] + 1]
                self.orig_times[q] = self.orig_times[q][non_nans[0]: non_nans[-1] + 1]

        # ########################################################################################################
        # INITIALIZING FLUXES AND TIMES

        self.times = copy.deepcopy(self.orig_times)
        """List of time arrays for quarters that will be processed and modified"""
        self.fluxes = copy.deepcopy(self.orig_fluxes)
        """List of flux arrays for quarters that will be processed and modified"""
        self.clean_fluxes = copy.deepcopy(self.orig_fluxes)
        """Flux arrays for quarters used for technical purposes such as PDS measurement. 
            Will be processed more strictly so that signal SNR may be destroyed"""
        self.quarters = range(len(self.fluxes))
        """Array enumerating all the quarters of Kepler data"""
        self.masks = [~np.isnan(self.fluxes[i]) for i in self.quarters]  # mask things which need to be removed from data
        """List of flux mask arrays for quarters. True where data is good, False where data is bad"""
        self.clean_masks = copy.deepcopy(self.masks)  # mask all the outliers, needed for PSD and filters, might kill transits
        """More strict masks for quarters of self.clean_fluxes. True where data is good, False where data is bad"""
        self.psd_masks = copy.deepcopy(self.masks)  # clean_masks with maybe only big holes to have enough slices for PSD
        """Masks for quarters of data suitable for PSD measurement (some small holed may be ignored to increase 
                    slices size). True where data is good, False where data is bad"""
        # self.big_hole_masks = [np.ones(len(self.masks[q])).astype(bool) for q in self.quarters]
        self.big_hole_masks = []
        for q in self.quarters:
            big_mask = ut.clean_small_holes(np.array(self.masks[q]), max_hole=Max_small_hole)
            big_mask = np.array(~ut.clean_small_holes(np.array(~big_mask), max_hole=1))
            self.big_hole_masks.append(big_mask)
        """List of flux mask arrays for quarters where very small holes are omitted. The small holes 
                    can be interpolated without damage. True where data is good, False where data is bad"""

        # ########################################################################################################
        # FINDING MEANS LIMITS AND STEPS

        self.mean_total = ut.get_average_mean(self.clean_fluxes)
        """Weighted total mean of all quarters"""
        self.orig_means = None
        """Mean values of `self.orig_fluxes` per quarter, that will be subtracted from the data"""

        self.dt_days = ut.get_average_dt(times=self.times)
        """Mean time step in days"""
        self.dt_sec = self.dt_days * 24 * 60 * 60
        """Time step in seconds"""
        self.tmin = np.nanmin([np.nanmin(t) for t in self.times])
        """Minimal Kepler time of the considered data"""
        self.tmax = np.nanmax([np.nanmax(t) for t in self.times])
        """Maximal Kepler time of the considered data"""
        self.t_mission = self.tmax - self.tmin  # NOT CHANGED IN TIME SLIDES
        """Total time span in days of the considered data"""
        lmax = np.max(np.array([len(t) for t in self.orig_times]))  # maximum length of the time
        # the closest (but bigger) power of 2 to the length        Tmax += tm.L_ban
        if power_2_support:
            self.support = 2 ** int(np.ceil(np.log2(lmax)))
        else:
            safety_pad = max(2 * tm.L_ban, 4 * Nperseg_psd)
            self.support = lmax + 2 * safety_pad
            self.support = int(2 * np.ceil(self.support / 2))  # make it even
        """Length in bins of each quarter padded by zeros. The closest power of 2 greater or equal to the longest
            flux. Could include the length of the template from the template bank, but it does not."""
        self.fft_support = self.support // 2 + 1  # use_rfft
        """Length in fft domain of each quarter padded by zeros. Differs from `self.support` because rfft is used"""

        # ########################################################################################################
        # INITIALIZING FLAGS

        self.is_fluxes_conditioned = False
        self.is_zero_padded = False
        self.is_fluxes_whitened = False
        self.is_fluxes_inpainted = False
        self.is_holes_interpolated = False
        self.is_prepared_for_psd = False
        self.is_template_whitened = False
        self.is_periodic_scores_corrected = False
        self.is_shifts_interleaved = False
        self.was_notched = None
        self.is_scrambled = False

        # ########################################################################################################
        # INITIALIZING njitted functions

        # self.inpaint_fluxes_fast = numba.njit(inpaint_fluxes_fun)  # it is important to use it because if shrinks ram from 4G to 1.5G
        self.inpaint_fluxes_fast = inpaint_fluxes_fun
        # initialization of the numba function on some small data in order  not to
        # spend time (AND MEMORY) on big data:
        # _ = self.inpaint_fluxes_fast(fluxes=np.array(np.random.normal(0, 1, (2, 32)), dtype=Dtype_r),
        #                              masks=np.random.choice([0, 1], (2, 32)).astype(bool),
        #                              wf_tds=np.array([np.concatenate((np.zeros(32 // 2 - 2), [-0.5, 1, 1, -0.5], np.zeros(32 // 2 - 2)))
        #                                               for _ in range(2)]))
        # self.sum_periods_fast = numba.njit(sum_periods_fun, parallel=False, fastmath=False)
        self.sum_periods_fast = sum_periods_fun  # was initialized, compiled and dummy-run outside the class
        # initialization of the numba function on some small data in order  not to
        # spend time (AND MEMORY) on big data:
        # _ = self.sum_periods_fast(inputs_array=np.ones((192, 10), dtype=Dtype_r),
        #                     dt=Dtype_timings(0.02), tmax=Dtype_timings(3.), tmin=Dtype_timings(0.),
        #                     periods=np.arange(1., 2., 0.5).astype(Dtype_timings))
        # _ = self.sum_periods_fast(statistics_by_trends=np.ones((2, 192)),
        #                           trends_sq_by_vars=np.ones((2, 192)),
        #                           corrections=np.ones((2, 192)),
        #                           statistics_by_trends_gaussianized=np.ones((2, 192)),
        #                           dt=0.02, tmax=3., tmin=0., periods=np.arange(1., 2., 0.5))
        self.normalize_stats_fast = normalize_stats_fun
        # ########################################################################################################
        # WHITENING ETC

        self.whitened_fluxes_fd = None
        """Whitened `self.fluxes` per quarter in Fourier domain"""
        self.whitened_fluxes_td = None
        """Whitened `self.fluxes` per quarter in time domain"""
        self.whitened_clean_fluxes_fd = None
        """Whitened `self.clean_fluxes` per quarter in Fourier domain"""
        self.whitened_clean_fluxes_td = None
        """Whitened `self.clean_fluxes` per quarter in Fourier domain"""
        self.trends_low = None
        """Low-frequency trends which were removed before PSD measurement"""
        self.nperseg_psd = None
        """Slice size in bins used in PSD measurement"""
        self.npsds0 = None
        """Number of pieces per quarter available for psd measurement"""
        self.npsds = None
        """Number of pieces per quarter used for psd measurement"""
        self.freqs = None
        """Arrays of frequency per quarter"""
        self.psdfreqs = None
        """Arrays of frequency of PSD slices per quarter"""
        self.psds = None
        """PSD per quarter"""
        self.orig_psds = None
        """PSD per quarter originated from pure Welch, before any modifications"""
        self.wf_fds_meas = None
        """Whitening filters in Fourier domain per quarter originated from pure Welch, before any modifications"""
        self.wf_tds = None
        """Whitening filters per quarter in time domain"""
        self.wf_fds = None
        """Whitening filters per quarter in Fourier domain"""
        self.wf_fds_unwindowed = None
        """Whitening filters per quarter in Fourier domain without applying Tuckey window"""
        self.zero_padding_ends = None
        """Number of zeros added at the beginning and the end of every quarter to match `self.support`, per quarter"""
        self.notch_mask = None
        """Masks per quarter of indices (in Fourier) that were subjected to notch filtering"""
        self.notch_soss = None
        """Filters to suppress outliers in frequencies in `self.notch_mask` locations"""
        self.unnotched_whitened_clean_fluxes_fd = None
        """Whitening filters in Fourier domain without notch filter"""

        # ########################################################################################################
        # templates
        self.tb = None
        """Template bank class"""
        self.indices_of_templates = None
        """Indices corresponding to the beginning and the end of enumeration of the same template 
            with different subbin shifts"""
        self.templates_of_indices = None
        """In the list of templates with different subbin shifts, showing to which actual template 
            this corresponds"""
        self.whitened_templates_td = None
        """Whitened templates for all templates for all quarters in the time domain"""
        self.whitened_templates_fd = None
        """Whitened templates for all templates for all quarters in the Fourier domain"""

        # ########################################################################################################
        # STATISTICS BY QUARTERS

        self.statistics_raw = None
        """Matched-filtering score per template (subbin-shifts in a row) per quarter"""
        self.statistics_snr = None
        """Matched-filtering score in units of SNR per template (subbin-shifts in a row) per quarter"""
        self.stat_masks = None
        """Goodness masks of `self.statistics_raw` per template (subbin-shifts in a row) per quarter. 
            True if good False if bad."""
        self.stat_moving_vars = None
        """Moving variance of matched-filtering score per template (subbin-shifts in a row) per quarter"""
        self.stat_tot_vars = None
        """Total scalar variance of matched-filtering score per template (subbin-shifts in a row) per quarter"""

        # VETO AND DISTRIBUTION CORRECTION
        self.stats_raw_no_veto_no_score_cor = None
        """Original `self.statistics_raw` before it underwent veto and single-event statistic distribution correction"""
        self.veto_masks = None
        """Single-event quality veto passing masks of `self.statistics_raw` per template (subbin-shifts in a row) 
            per quarter. True if good False if bad."""
        self.trigger_indices_unpadded = None
        self.trigger_indices_padded = None
        """All single-event threshold-crossing indices which underwent single-event veto. For templates and quarters"""
        self.distance_to_hole_masks = None
        self.no_step_contamination_masks = None
        self.min_neighbor_and_template_ses_snr = None
        self.tail_stat_masks = None
        self.too_large_neighbor_masks = None

        self.veto_td_p_vals = None
        self.veto_fd_p_vals = None
        self.veto_p_vals = None
        self.veto_p_val_masks = None
        # Old veto:
        self.trigger_indices = None
        self.veto_bit_masks = None
        """Single-event quality veto passing masks of `self.statistics_raw` per template (subbin-shifts in a row) 
            per quarter for different veto tests. Binary digits code different tests."""

        # SINGLE_EVENT STATISTIC DISTRIBUTION CORRECTION
        self.statistics_gaussianized = None
        """Gaussianized matched-filtering score per template (subbin-shifts in a row) per quarter"""
        self.statistics_snr_gaussianized = None
        """Gaussianized matched-filtering score in units of SNR per template (subbin-shifts in a row) per quarter"""
        self.stat_non_gauss_scales_coefs = None
        """Parameters of distributions of single-event scores. Per template, one list is for negative statistics, 
            the other of positive, and inside are pairs of scale and amplitude coefficients of the distribution."""
        self.log_non_gaussian_corrections = None
        """Difference of log of probabilities under the standard distribution and the measured distribution 
            characterized by `self.stat_non_gauss_scales_coefs`"""

        # FULL STATISTICS
        self.time_full = None
        """Times of all quarters combined together"""
        self.statistics_full = None
        """Statistics `self.statistics_raw` of all quarters combined together, per template"""
        self.stat_vars_full = None
        """Statistic variances `self.stat_moving_vars` of all quarters combined together, per template"""
        self.stat_masks_full = None
        """Statistic masks `self.stat_masks` of all quarters combined together, per template"""
        self.big_hole_masks_full = None  # used only in the statistic shuffling
        """Masks `self.big_hole_masks` of all quarters combined together"""
        self.corrections_full = None
        """Log distribution corrections `self.log_non_gaussian_corrections` of all quarters combined together, per template"""
        self.statistics_gaussianized_full = None
        """Gaussianized matched-filtering score of all quarters combined together, per template"""

        # INTERLEAVED STATISTICS
        self.interleaved_numbers = None
        """Indices of templates which had subbin shifts and therefore require interleaving scores for subbin shifts"""
        self.not_interleaved_numbers = None
        """Indices of templates which had no subbin shifts and therefore do not require interleaving"""
        self.n_shifts = None
        """Number of subbin shifts per template. Same for all who have subbin shifts"""
        self.time_interleaved = None
        """High-resolution full time axis for statistics whose templates were given with subbin shifts."""
        self.statistics_interleaved = None
        """Full statistics for templates who required subbin shifts, interleaving subbin shifts to the same time axis"""
        self.stat_masks_interleaved = None
        """Interleaved `self.stat_masks_full` for templates who required subbin shifts"""
        self.stat_vars_interleaved = None
        """Interleaved `self.stat_vars_full` for templates who required subbin shifts"""
        self.corrections_interleaved = None
        """Interleaved `self.corrections_full` for templates who required subbin shifts"""
        self.statistics_gaussianized_interleaved = None
        """Interleaved `self.statistics_gaussianized_full` for templates who required subbin shifts"""
        self.time_not_interleaved = None
        """Original resolution full time axis for statistics whose templates were given without subbin shifts."""
        self.statistics_not_interleaved = None
        """Full statistics for templates who did not require subbin shifts, `self.not_interleaved_numbers`"""
        self.stat_masks_not_interleaved = None
        """`self.stat_masks_full` for templates who did not require subbin shifts, `self.not_interleaved_numbers`"""
        self.stat_vars_not_interleaved = None
        """`self.stat_vars_full` for templates who did not require subbin shifts, `self.not_interleaved_numbers`"""
        self.corrections_not_interleaved = None
        """`self.corrections_full` for templates who did not require subbin shifts, `self.not_interleaved_numbers`"""
        self.statistics_gaussianized_not_interleaved = None
        """`self.statistics_gaussianized_full` for templates who did not require subbin shifts, `self.not_interleaved_numbers`"""

        # ########################################################################################################
        # SCRAMBLING
        self.time_slide_shifts = None
        """If time slides are implemented, list of random shifts per quarter"""
        self.inds_pieces_scramble = None
        """If time scrambling are implemented, list of indices of randomly selected pieces of `self.statistics_full`"""
        self.tot_hole_len_scramble = None
        """If time scrambling are implemented, total length of holes in `self.statistics_full`"""
        self.inds_pieces_in_scrambled = None
        """If time scrambling are implemented, list of indices pieces `self.inds_pieces_scramble` which were randomly scrambled"""

        # ########################################################################################################
        # PERIODICITY SEARCH
        self.pmin = None
        """Minimal period of the search, days"""
        self.pmax = None
        """Maximal period of the search, days"""
        self.periods = None
        """All periods of the search, days"""
        self.period_overlap_bins = None
        """Spare bins added to `self.periods` to avoid triggers being on the edge of `self.periods`."""
        self.statistics_periodic_interleaved = None
        """Period-folded matched-filtering statistics, brought to units of SNR. 
            For periods, interleaved templates and phases."""
        self.corrections_periodic_interleaved = None
        """Period-folded `self.corrections_interleaved`. 
            For periods, interleaved templates and phases."""
        self._phases_interleaved = None
        """Initial phases (with respect to `self.tmin`) of `self.statistics_periodic_interleaved`."""
        self._phases_not_interleaved = None
        """Initial phases (with respect to `self.tmin`) of `self.statistics_periodic_not_interleaved`"""
        self.statistics_periodic_not_interleaved = None
        """Period-folded matched-filtering statistics, brought to units of SNR. 
            For periods, not-interleaved templates and phases."""
        self.corrections_periodic_not_interleaved = None
        """Period-folded `self.corrections_not_interleaved`. For periods, not-interleaved templates and phases."""
        self.uncorrected_stats_periodic_interleaved = None
        """Period-folded matched-filtering statistics, brought to units of SNR, which will not undergo 
            distribution correction. For periods, interleaved templates and phases."""
        self.uncorrected_stats_periodic_not_interleaved = None
        """Period-folded matched-filtering statistics, brought to units of SNR, which will not undergo 
            distribution correction. For periods, not-interleaved templates and phases."""
        self.stats_gaussianized_periodic_interleaved = None
        """Period-folded gaussianized matched-filtering statistics, brought to units of SNR. 
            For periods, interleaved templates and phases."""
        self.stats_gaussianized_periodic_not_interleaved = None
        """Period-folded gaussianized matched-filtering statistics, brought to units of SNR. 
                    For periods, interleaved templates and phases."""

        # ########################################################################################################
        # AFTER PERIODICITY SEARCH

        self.statistics_periodic = None
        """Period-folded matched-filtering statistics, brought to units of SNR after applying the distribution
            correction and downsampling interleaved templates. For periods, templates and phases."""
        self.corrections_periodic = None
        """Period-folded single-event statistic distribution corrections after downsampling interleaved templates. 
            For periods, templates and phases."""
        self.stats_gaussianized_periodic = None
        """Period-folded gaussianized matched-filtering statistics, brought to units of SNR after downsampling 
            interleaved templates. For periods, templates and phases."""
        self.uncorrected_stats_periodic = None
        """Period-folded matched-filtering statistics, brought to units of SNR which will not undergo 
            distribution correction. For periods, templates and phases."""
        self._phases = None
        """Initial phases (with respect to `self.tmin`) of `self.statistics_periodic`"""
        self.stats_periodic_mar_temp = None
        """`self.statistics_periodic` marginalized over the templates. Only negative were preserved. 
            For periods and phases."""
        self.stats_gaussianized_periodic_mar_temp = None
        """`self.stats_gaussianized_periodic` marginalized over the templates. Only negative were preserved.
            For periods and phases."""
        self.best_phase_inds = None
        """Indices of best phase (with maximal `self.stats_periodic_mar_temp`) for every period. 
            One phase per period."""
        self.stats_periodic_mar_temp_best_ph = None
        """One maximal `self.stats_periodic_mar_temp` per every period"""
        self.stats_gaussianized_periodic_mar_temp_best_ph = None
        """`self.stats_gaussianized_periodic_mar_temp` corresponding to `self.best_phase_inds`"""
        self.first_transit_times_best_ph = None
        """First transit times corresponding to `self.stats_periodic_mar_temp_best_ph`"""
        self.best_g_phase_inds = None
        """Indices of best phase (with maximal `self.stats_gaussianized_periodic_mar_temp`) 
            for every period. One phase per period."""
        self.stats_periodic_mar_temp_best_g_ph = None
        """`self.stats_periodic_mar_temp` corresponding to `self.best_g_phase_inds`"""
        self.stats_gaussianized_periodic_mar_temp_best_g_ph = None
        """One maximal `self.stats_gaussianized_periodic_mar_temp` per every period"""
        self.first_transit_times_best_g_ph = None
        """First transit times corresponding to `self.stats_periodic_mar_temp_best_g_ph`"""

        # ########################################################################################################
        # FINDING BEST TRIGGER
        self.best_periodic_scores_data = None
        """Summarizes best trigger found in the search."""
        self.veto_mes_masks = None
        """Periods and phases whose triggers were best triggers but were masked out by MES veto."""
        self.veto_g_mes_masks = None
        """Periods and phases whose triggers were best gaussianized triggers but were masked out by MES veto."""

        # ########################################################################################################
        # BEST TRIGGER PROPERTIES
        self.best_period = None
        """Period corresponding to the maximal `self.stats_periodic_mar_temp`"""
        self.best_ftt = None
        """First transit time corresponding to the maximal `self.stats_periodic_mar_temp`"""
        self.best_stat = None
        """Maximal `self.stats_periodic_mar_temp` (that passed MES veto if it was applied)"""
        self.best_stat_gaussianized = None
        """`self.stats_gaussianized_periodic_mar_temp` corresponding to the 
            maximal `self.stats_periodic_mar_temp`"""
        self.best_int_score = None
        """Sum of prior-weighted statistic around the maximal `self.stats_periodic_mar_temp`"""
        self.best_phase_index = None
        """Period index corresponding to the maximal `self.stats_periodic_mar_temp`"""
        self.best_per_index = None
        """Phase index corresponding to the maximal `self.stats_periodic_mar_temp`"""
        self.best_template_index = None
        """Template index of the best (negative and prior-weighted if applied) template 
            corresponding to the maximal `self.stats_periodic_mar_temp`"""

        self.best_g_period = None
        """Period corresponding to the maximal `self.stats_gaussianized_periodic_mar_temp`"""
        self.best_g_ftt = None
        """First transit time corresponding to the maximal `self.stats_gaussianized_periodic_mar_temp`"""
        self.best_g_stat = None
        """`self.stats_gaussianized_periodic_mar_temp` corresponding to the maximal 
            `self.stats_gaussianized_periodic_mar_temp`"""
        self.best_g_stat_gaussianized = None
        """Maximal `self.stats_gaussianized_periodic_mar_temp` (that passed MES veto if it was applied)"""
        self.best_g_int_score = None
        """Sum of prior-weighted statistis around the maximal `self.stats_gaussianized_periodic_mar_temp`"""
        self.best_g_phase_index = None
        """Period index corresponding to the maximal `self.stats_gaussianized_periodic_mar_temp`"""
        self.best_g_per_index = None
        """Phase index corresponding to the maximal `self.stats_gaussianized_periodic_mar_temp`"""
        self.best_g_template_index = None
        """Template index of the best (nagetive and prior-weighted if applied) template 
                    corresponding to the maximal `self.stats_gaussianized_periodic_mar_temp`"""

        # ########################################################################################################
        # SUMMED TERMS OF THE BEST TRIGGER
        self.summed_times = None
        """Approximate times when the individual transits of the best trigger happened"""
        self.is_summed_interleaved = None
        """Whether the maximal template before marginalization at the `self.summed_times` 
            was one requiring interleaving or not"""
        self.summed_indices = None
        """Indices in interleaved or non-interleaved arrays (depending on `self.is_summed_interleaved`)
            corresponding to the `self.summed_times`."""
        self.summed_stats_raw = None
        """Single-event matched-filtering scores corresponding to the `self.summed_times`"""
        self.summed_vars = None
        """Variance of the single-event matched-filtering scores corresponding to the `self.summed_times`"""
        self.summed_stats_raw_gauss = None
        """Gaussianized single-event matched-filtering scores corresponding to the `self.summed_times`"""
        self.summed_corrections = None
        """Log single-event distribution corrections corresponding to the `self.summed_times`"""

    def condition_data(self, nsig_outliers_0=Nsig_outliers_0, nsig_outliers_0_pos=Nsig_outliers_0_pos,
                       nsig_outliers_0_neg=Nsig_outliers_0_neg,
                       nsig_outliers_psd=Nsig_outliers_psd, nperseg_psd=Nperseg_psd,
                       npsdsmin=Npsdsmin, reject_outliers_psd=Reject_outliers_psd,
                       nsig_notch=Nsig_notch,
                       nsig_outliers_w=Nsig_outliers_w, nsig_outliers_w_pos=Nsig_outliers_w_pos,
                       nsig_outliers_w_neg=Nsig_outliers_w_neg,
                       max_small_hole=Max_small_hole,
                       detr_length_outliers=Detr_length_outliers,
                       tukey_steps_len=Tukey_steps_len_psd,
                       save_low_trends=False):
        """
        Preparing data for detection. Subtract mean, mask outliers, measure psds,
        padding by zeros, notching, inpainting, whitening.

        See the description of the non-described parameters in the docstrings of the corresponding functions.
        :param nsig_outliers_0: Minimal deviation from trend considered to be outlier (in units of std)
        :type nsig_outliers_0: float
        :param nsig_outliers_0_pos: Minimal deviation from trend in the positive direction considered
            to be outlier (in units of std)
        :type nsig_outliers_0_pos: float
        :param nsig_outliers_0_neg: Minimal deviation from trend in the negative direction considered
            to be outlier (in units of std)
        :type nsig_outliers_0_neg: float
        :param nsig_outliers_psd:
        :type nsig_outliers_psd:
        :param nperseg_psd: Length of slice used in measuring PSD
        :type nperseg_psd: int
        :param npsdsmin: Minimal number of slices required to measure PSD with Welch method
        :type npsdsmin:  int
        :param reject_outliers_psd: Whether to reject slice power outliers when measuring the PSD
        :type reject_outliers_psd: bool
        :param nsig_notch: Minimal deviation from trend of whitened fluxes in Fourier domain
            considered to be outlier (in units of std of whitened fluxes in Fourier space)
        :type nsig_notch: float
        :param nsig_outliers_w: Minimal deviation from trend of whitened fluxes in real domain
           considered to be outlier (in units of their std)
        :type nsig_outliers_w: float
        :param nsig_outliers_w_pos: Minimal positive deviation from trend of whitened fluxes in
            time domain considered to be outlier (in units of their std)
        :type nsig_outliers_w_pos: float
        :param nsig_outliers_w_neg: Minimal negative deviation from trend of whitened fluxes in
            time domain considered to be outlier (in units of their std)
        :type nsig_outliers_w_neg: float
        :param max_small_hole: Maximal hole to be considered small, so that it can be
            interpolated linearly and ignored
        :type max_small_hole: int
        :param detr_length_outliers: Detrending length to detrend before measuring outlier level
        :type detr_length_outliers: int
        :param tukey_steps_len: Size of Tukey window used in whitening filtering, see inside the method
        :type tukey_steps_len: int
        :param save_low_trends: Whether to save the low trends subtracted before the PSD measurement as class attribute
        :type save_low_trends: bool
        """
        self.nperseg_psd = nperseg_psd
        self.subtract_means(save_means=True, masks=None)
        self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                nsig_negative=nsig_outliers_0_neg,
                                nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                in_whitened_fluxes=False, modify_clean_masks=True)
        self.subtract_means(save_means=True, masks=self.masks)
        self.normalize_by_means()
        self.make_big_hole_masks(max_small_hole=max_small_hole, exclude_individual_points=True)
        self.interpolate_fluxes(interpolate_clean_fluxes=True)
        self.subtract_low_trends(nperseg=self.nperseg_psd, save_low_trends=save_low_trends)
        self.fill_additional_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                      nsig_negative=nsig_outliers_0_neg,
                                      nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                      in_whitened_fluxes=False, modify_clean_masks=True,
                                      interpolate_clean_fluxes=True)  # again after piecewise detrending
        self.prepare_for_psd(nperseg=self.nperseg_psd, npsdsmin=npsdsmin, max_small_hole=max_small_hole)
        self.pad_by_zeros()  # Needed to avoid crosstalk in whitening filter. And we want the end zeros to be actual zeros
        self.measure_psds(nperseg=self.nperseg_psd, tukey_steps_len=tukey_steps_len, reject_outliers=reject_outliers_psd)
        self.whiten_fluxes(whiten_clean_fluxes=True)
        self.fill_additional_outliers(nsig=nsig_outliers_w, nsig_positive=nsig_outliers_w_pos,
                                      nsig_negative=nsig_outliers_w_neg,
                                      nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                      in_whitened_fluxes=True, modify_clean_masks=True, interpolate_clean_fluxes=True)
        self.whiten_fluxes(whiten_clean_fluxes=True)
        self.notch_fluxes(nsig=nsig_notch)
        self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                nsig_negative=nsig_outliers_0_neg,
                                nsig_clean=nsig_outliers_psd,
                                detr_length_outliers=detr_length_outliers,
                                in_whitened_fluxes=False, modify_clean_masks=True)
        self.make_big_hole_masks(max_small_hole=max_small_hole, exclude_individual_points=False)
        self.inpaint_fluxes()
        self.whiten_fluxes(whiten_clean_fluxes=True)
        self.is_fluxes_conditioned = True

        """
        Values diverging from the large trend more than by 3 standard deviations are added to the mask 
        as False (before, mask marked bins containing NaN). psd_masks are also modified, because we don't 
        want outliers to take part in PSD measurement:

        Zero padding is needed to apply the whitening filter to avoid crosstalk between ends. 
        Applying filters in Fourier domain is equivalent to circular convolution, therefore 
        the ends of the data will influence each other which is unphysical. As well, at this step 
        the time array is interpolated and extended on the zero-padded region.

        # zero padding is needed to apply the whitening filter to avoid crosstalk between ends.
        # When padding by zeros is done before the interpolation, in the interpolation the padded zeros become non-zeros,
        # and due to this the overall trends will not produce  outliers inside the data at the ends.
        # REM: the floating mean is subtracted with whitening

        Will fill all the bad data masked by mask with linear interpolation. Interpolation 
        is required for whitening, notching and other filters because filtering is making linear 
        combinations of points in the kernel, and adding bad data or NaN there will spoil all 
        the data. These points still remain masked for the PSD measurement if there is enough 
        slices. Otherwise, the masks will not be cleaned from small holes. Therefore data in masked
        points will be used only if necessary. Remark: when padding by zeros is done before the 
        interpolation, in the interpolation the padded zeros become constant values equal to the 
        last edge data value, and due to this outliers will not be produced in the data at the ends:

        # Interpolation is required for whitening, notching etc.
        # But in the preparation for PSD, the masks will not be cleaned if not needed,
        # therefore data in holes will be used only if necessary

        In the preparation for PSD, the code is making sure that there is enough slices (npsdsmin) 
        to ensure the needed precision in PSD measurement. If there is not enough, 
        clean_small_holes() is used to remove small holes from the PSD mask. They are not 
        significant, so we easily can take interpolated values in them:

        Use Welch method to measure Power Spectral Density. Split data into multiple slices 
        of length nperseg_psd, calculate their abs squared of Fourier transform, and average. 
        The resulting array will have length nperseg_psd, therefore for making whitening filters, 
        (psd)^{-0.5} is converted to time domain and padded by zeros (finite response filter). 
        PSD and whitening filters are quarter-specific:

        Whitening filters are applied to the fluxes to create whitened fluxes. (Remark: the 
        floating mean is subtracted with whitening):

        Masking outliers in whitened fluxes. Effectively we are making detection in the whitened
        data with whitened template, so outliers are unwanted there:

        Interpolation to cover also new outliers. Note that outliers were masked in the whitened 
        flux, but interpolation is made in the red flux. It is possible to make interpolation in 
        the whitened flux and then redden it, but the inverse whitening filter is unstable, so 
        the red flux changes significantly. As well, in most cases interpolation in the red flux 
        indeed fixes the outliers because the corresponding point in the red flux has the biggest 
        impact on the outlier:

        # Before notch need to interpolate over mask because holes will not allow to do whitening which is needed in notch,
        # and need to freshly whiten the fluxes because need whitened_fd for notch

        Whiten the interpolated fluxes again before applying the Notch filter because in it, 
        the whitened fluxes in Fourier domain are needed:

        In whitened_fluxes_fd, common peaks in all the quarters are searched, and Notch filters 
        are designed to suppressed them. They are applied to the fluxes:

        # There is an option to remeasure PSD after notch. Usually, it leaves wider notched gaps in the whitened flux,
        # but removes remnant excess in power around peaks.

        After notching, some outliers can appear, they are also added to the mask: 

        All the bad points ever masked are filled using the optimal inpainting filter which zeroes 
        the impact of these point to the statistic. Note: PSD should not be remeasured after 
        inpainting because the same C^{-1} has to be used for inpainting and for detection: 

        These fluxes will be used for detection.
        """

    def save_conditioned_data(self, file=None, path=None, filename_postfix=Conditioned_data_filename_postfix):
        """
        ATTENTION WARNING if you change this function, make sure to change load_mean_rel_wf_fds in the target_class
        Saves the essential conditioning results as .npy file that can be read
        with `load_conditioned_data` function:
            self.support
            self.nperseg_psd
            self.mean_total
            self.zero_padding_ends
            self.times
            self.fluxes
            self.whitened_fluxes_td
            self.whitened_fluxes_fd
            self.wf_tds
            self.wf_fds
            self.big_hole_masks
            self.whitened_templates_td
            self.whitened_templates_fd

        :param file: Exact file where to save the result. Overrides `path` and `filename_postfix`.
        :type file: str
        :param path : Path to the folder in which the data should be saved. Used in case `file` is not specified.
        :type path : str
        :param filename_postfix: Used in the name convention "target_id + filename_postfix" in
            case `file` is not specified. The location will be `self.output_location`
        :type filename_postfix: str
        """
        if file is None:
            if path is None:
                path = self.output_location
            file = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                           filename_postfix=filename_postfix, file_format='npy')
        len_data = sum([self.support for _ in self.quarters])  # is assumed to be support length times number of quarters
        conditioned_data = np.zeros((7 + len(self.whitened_templates_td), len_data), dtype=Dtype_r)
        conditioned_data[0][10] = self.support
        conditioned_data[0][11] = len(self.whitened_templates_td)
        conditioned_data[0][12] = self.nperseg_psd
        conditioned_data[0][13] = self.mean_total
        ind_last = 0
        for q in self.quarters:
            conditioned_data[0][ind_last + 1] = self.zero_padding_ends[q][0]
            conditioned_data[0][ind_last + 2] = self.zero_padding_ends[q][1]
            conditioned_data[0][ind_last + 3] = len(
                self.whitened_fluxes_td[q])  # this depends on whether data is cut from zeros or not
            conditioned_data[0][ind_last + 4] = len(self.orig_fluxes[q])
            conditioned_data[1][ind_last:ind_last + len(self.fluxes[q])] = self.times[q]
            conditioned_data[2][ind_last:ind_last + len(self.fluxes[q])] = self.fluxes[q]
            conditioned_data[3][ind_last:ind_last + len(self.whitened_fluxes_td[q])] = self.whitened_fluxes_td[q]
            conditioned_data[4][ind_last:ind_last + self.support] = self.wf_tds[q]
            conditioned_data[5][ind_last:ind_last + len(self.fluxes[q])] = self.big_hole_masks[q]
            for k in range(len(self.whitened_templates_td)):
                conditioned_data[k + 7][ind_last:ind_last + len(self.whitened_templates_td[k][q])] = self.whitened_templates_td[k][q]
            ind_last += self.support
        if self.target_id is not None:
            file_path = file
            np.save(file_path, np.array(conditioned_data, dtype=object), allow_pickle=True)
        else:
            np.save(os.path.join(path, filename_postfix), np.array(conditioned_data, dtype=object), allow_pickle=True)

    def load_conditioned_data(self, file=None, filename_postfix=Conditioned_data_filename_postfix,
                              path=None, no_zero_padding=False):
        """
        Loads the previously prepared conditioned data needed for the single-event search.
        Only works for data created by `save_conditioned_data`.

        Sets:
        self.support
        self.nperseg_psd
        self.mean_total
        self.zero_padding_ends
        self.times
        self.fluxes
        self.whitened_fluxes_td
        self.whitened_fluxes_fd
        self.wf_tds
        self.wf_fds
        self.big_hole_masks
        self.whitened_templates_td
        self.whitened_templates_fd

        :param file: Exact .npy file where the data is to be looked for. Overrides `path` and `filename_postfix`.
        :type file: str
        :param path : Path to the folder in which the data should be saved. Used in case `file` is not specified.
        :type path : str
        :param filename_postfix: Used in the name convention "target_id + filename_postfix" in
            case `file` is not specified. The location will be `self.output_location`
        :type filename_postfix: str
        :param no_zero_padding: Put `True` if conditioned data that was saved was unpadded from zeros
        :type no_zero_padding: bool
        """
        if file is None:
            if path is None:
                path = self.output_location
            file = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                           filename_postfix=filename_postfix, file_format='npy')
        conditioned_data = np.load(file, allow_pickle=True)
        self.support = int(conditioned_data[0][10])
        len_whitened_templates_td = int(conditioned_data[0][11])
        self.nperseg_psd = int(conditioned_data[0][12])
        self.mean_total = float(conditioned_data[0][13])

        self.zero_padding_ends = [[0, 0] for _ in self.quarters]
        self.times = []
        self.fluxes = []
        self.whitened_fluxes_td = []
        self.whitened_fluxes_fd = []
        self.wf_tds = []
        self.wf_fds = []
        self.big_hole_masks = []
        self.whitened_templates_td = [[] for _ in range(len(conditioned_data) - 7)]
        self.whitened_templates_fd = [[] for _ in range(len(conditioned_data) - 7)]
        ind_last = 0
        was_zero_padded = self.is_zero_padded
        self.is_zero_padded = True
        for q in self.quarters:
            zpe1 = int(conditioned_data[0][ind_last + 1])
            zpe2 = int(conditioned_data[0][ind_last + 2])
            self.zero_padding_ends[q][0] = zpe1
            self.zero_padding_ends[q][1] = zpe2
            lenflux = int(conditioned_data[0][ind_last + 3])
            if lenflux != self.support:
                self.is_zero_padded = False
            self.times.append(np.array(conditioned_data[1][ind_last: ind_last + lenflux]).astype(Dtype_r))
            self.fluxes.append(np.array(conditioned_data[2][ind_last: ind_last + lenflux]).astype(Dtype_r))
            whfltd1 = np.array(conditioned_data[3][ind_last: ind_last + lenflux]).astype(Dtype_r)
            if (not self.is_zero_padded) and (not no_zero_padding):
                whfltd1 = np.concatenate((np.zeros(zpe1), whfltd1, np.zeros(zpe2)))
            self.whitened_fluxes_td.append(whfltd1)
            self.whitened_fluxes_fd.append(rfft(whfltd1))
            wftd1 = np.array(conditioned_data[4][ind_last: ind_last + self.support]).astype(Dtype_r)
            self.wf_tds.append(wftd1)
            self.wf_fds.append(np.abs(rfft(wftd1)))
            self.big_hole_masks.append((conditioned_data[5][ind_last: ind_last + lenflux]).astype(bool))
            for k in range(len_whitened_templates_td):
                whtltd1 = np.array(conditioned_data[k + 7][ind_last:ind_last + lenflux]).astype(Dtype_r)
                if (not self.is_zero_padded) and (not no_zero_padding):
                    whtltd1 = np.concatenate(
                        (whtltd1[:(len(whtltd1) + 1) // 2], np.zeros(zpe1 + zpe2), whtltd1[(len(whtltd1) + 1) // 2:]))
                self.whitened_templates_td[k].append(whtltd1)
                self.whitened_templates_fd[k].append(rfft(whtltd1))
            ind_last += self.support
        if (not self.is_zero_padded) and (not no_zero_padding):
            for q in self.quarters:
                zerlen1 = self.zero_padding_ends[q][0]
                zerlen2 = self.zero_padding_ends[q][1]
                self.fluxes[q] = np.concatenate((np.zeros(zerlen1),
                                                 self.fluxes[q], np.zeros(zerlen2))).astype(Dtype_r)
                self.big_hole_masks[q] = np.concatenate((np.zeros(zerlen1),
                                                         self.big_hole_masks[q], np.zeros(zerlen2))).astype(bool)
                self.times[q] = np.concatenate((np.zeros(zerlen1) + np.nan,
                                                self.times[q], np.zeros(zerlen2) + np.nan)).astype(Dtype_r)
                self.times[q] = ut.interpolate_time(np.array(self.times[q]), mask=None).astype(Dtype_r)
            self.is_zero_padded = True
        if self.is_zero_padded and (not was_zero_padded):
            for q in self.quarters:
                zerlen1 = self.zero_padding_ends[q][0]
                zerlen2 = self.zero_padding_ends[q][1]
                self.masks[q] = np.concatenate((np.zeros(zerlen1),
                                                self.masks[q], np.zeros(zerlen2))).astype(bool)

        self.is_template_whitened = True
        self.wf_tds = np.array(self.wf_tds, dtype=Dtype_r)

    def save_mean_psd_for_injections(self, file=None, filename_postfix=Psd_for_injections_filename_postfix, path=None):
        """
        Take the mean of the psds pf all the quarters to get the average noise. Will be used in estimations
        :param file: Local directory where to save the result. Overrides `filename_postfix`
        :type file: str
        :param filename_postfix: Used in the name convention "target_id + filename_postfix" in
            case `file` is not specified. The location will be `path`
        :type filename_postfix: str
        :param path: Where to save the result if `file` is not specified. if None, the
            location will be `self.output_location`
        :type path: str or None
        :return:
        :rtype:
        """
        if file is None:
            if path is None:
                path = self.output_location
            file = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                           filename_postfix=filename_postfix, file_format='npy')
        np.save(file, np.mean(self.psds, axis=0) / self.mean_total ** 2, allow_pickle=True)

    def save_connected_stats(self, filename_postfix=Connected_statistic_data_filename_postfix, output_location=None,
                             file_path=None):
        """
        Saving all the connected stats and other fields which will be used in periodicity search or timeslides.
        :param filename_postfix: Used in the name convention "target_id + filename_postfix" in
            case `file` is not specified. The location will be `self.output_location`
        :type filename_postfix: str
        :param output_location: Where to save the result if `file` is not specified. if None, the
            location will be `self.output_location`
        :type output_location: str or None
        :param file_path:
        :type file_path:
        """
        num_temp_shifts = len(self.statistics_full)
        len_one = len(self.time_full)
        connected_data = np.zeros((3 + num_temp_shifts * 5, len_one + 1), dtype=Dtype_r)

        connected_data[0, 1:] = self.time_full
        connected_data[2, 1:] = self.big_hole_masks_full.astype(Dtype_r)
        line_num_0 = 3
        for kk in range(num_temp_shifts):
            connected_data[line_num_0 + kk, 0] = self.templates_of_indices[kk]
            connected_data[line_num_0 + kk, 1:] = self.statistics_full[kk]
            connected_data[line_num_0 + 1 * num_temp_shifts + kk, 1:] = self.stat_masks_full[kk].astype(float)
            connected_data[line_num_0 + 2 * num_temp_shifts + kk, 1:] = self.stat_vars_full[kk]
            connected_data[line_num_0 + 3 * num_temp_shifts + kk, 1:] = self.corrections_full[kk]
            connected_data[line_num_0 + 4 * num_temp_shifts + kk, 1:] = self.statistics_gaussianized_full[kk]

        if file_path is None:
            if output_location is None:
                output_location = self.output_location
            file_path = af.get_saving_file_path(output_location=output_location, target_id=self.target_id,
                                                filename_postfix=filename_postfix, file_format='npy')
        np.save(file_path, connected_data, allow_pickle=True)

    def save_no_veto_no_score_cor_stats(self, filename_postfix=No_veto_no_score_cor_stats_filename_postfix):
        """
        Save statistics and their moving variances before veto or gaussianization were applied
        :param filename_postfix: Used in the name convention "target_id + filename_postfix".
            The location will be `self.output_location`
        :type filename_postfix: str

        """
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=filename_postfix, file_format='npy')
        np.save(file_path, np.array([self.stats_raw_no_veto_no_score_cor,
                                     self.stat_moving_vars], dtype=object), allow_pickle=True)

    def load_no_veto_no_score_cor_stats(self, filename_postfix=No_veto_no_score_cor_stats_filename_postfix):
        """
        Save statistics and their moving variances before veto or gaussianization were applied
        :param filename_postfix: Used in the name convention "target_id + filename_postfix".
            The location will be `self.output_location`
        :type filename_postfix: str

        """
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=filename_postfix, file_format='npy')
        self.stats_raw_no_veto_no_score_cor, self.stat_moving_vars = np.load(file_path, allow_pickle=True)

    def load_connected_stats(self, file_path=None, filename_postfix=Connected_statistic_data_filename_postfix, path=None):
        """
        Load all the connected stats and other fields which will be used in periodicity search or timeslides.
        :param file_path: The exact local file which to load. Overrides `filename_postfix` and `path`
        :type file_path: str
        :param filename_postfix: Used in the name convention "target_id + filename_postfix" in
            case `file` is not specified. The location will be `self.output_location`
        :type filename_postfix: str
        :param path: Where to save the result if `file` is not specified. if None, the
            location will be `self.output_location`
        :type path: str
        """
        if file_path is None:
            if path is None:
                path = self.output_location
            file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                                filename_postfix=filename_postfix, file_format='npy')
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"
        connected_data = np.load(file_path, allow_pickle=True)

        num_temp_shifts = (len(connected_data) - 3) // 5  # int(connected_data[0, 0])
        self.time_full = connected_data[0, 1:]
        self.big_hole_masks_full = (connected_data[2, 1:]).astype(bool)
        templates_of_indices = []
        line_num_0 = 3

        self.statistics_full = np.zeros((num_temp_shifts, len(self.time_full)), dtype=Dtype_r)
        self.stat_vars_full = np.zeros((num_temp_shifts, len(self.time_full)), dtype=Dtype_r)
        self.stat_masks_full = np.zeros((num_temp_shifts, len(self.time_full)), dtype=bool)
        self.corrections_full = np.zeros((num_temp_shifts, len(self.time_full)), dtype=Dtype_r)
        self.statistics_gaussianized_full = np.zeros((num_temp_shifts, len(self.time_full)), dtype=Dtype_r)

        for kk in range(num_temp_shifts):
            templates_of_indices.append(connected_data[line_num_0 + kk, 0])
            self.statistics_full[kk] = connected_data[line_num_0 + kk, 1:]
            self.stat_masks_full[kk] = (connected_data[line_num_0 + 1 * num_temp_shifts + kk, 1:]).astype(bool)
            self.stat_vars_full[kk] = connected_data[line_num_0 + 2 * num_temp_shifts + kk, 1:]
            self.corrections_full[kk] = connected_data[line_num_0 + 3 * num_temp_shifts + kk, 1:]
            self.statistics_gaussianized_full[kk] = connected_data[line_num_0 + 4 * num_temp_shifts + kk, 1:]

        # making indices_of_templates
        temp_now = templates_of_indices[0]
        indices_of_templates = [[0]]
        for i, temp in enumerate(templates_of_indices):
            if temp != temp_now:
                indices_of_templates[-1].append(i)
                indices_of_templates.append([i])
                temp_now = temp
        indices_of_templates[-1].append(len(templates_of_indices))
        self.indices_of_templates = np.array(indices_of_templates)
        self.templates_of_indices = np.array(templates_of_indices)
        self.is_shifts_interleaved = False

    def condition_templates(self, templates=None, shifts=None, template_index='all'):
        """
        Initializes and conditions the template bank. Most of the template properties and methods are inside the `TemplateBank` class,
        but notching, whitening and similar are performed here.
        :param templates: Array of transit models used for detection, usually produced with Batman package centered at the center.
            See `template_module`
        :type templates: list numpy.ndarrays 1d of float, see `template_module`
        :param shifts: Sub-bin shifts of templates, see `template_module`
        :type shifts:
        :param template_index: which templates from the bank to use
        :type template_index: int or 'all'
        """
        self.initialize_templates(templates=templates, shifts=shifts, template_index=template_index)
        self.notch_templates()  # why notch is before roll, is written inside the method
        self.tb.roll_templates()
        self.whiten_templates()

    # noinspection PyIncorrectDocstring
    def detect_single_transits(self, nsig_outliers_stat=Nsig_outliers_stat,
                               save_no_veto_no_score_cor_stats_as_attr=False,
                               apply_veto=True, gaussianize=Get_gaussianized_ses,
                               apply_veto_only_to_neg=True,
                               margin_snr_veto=Margin_snr_veto,
                               max_abs_stat_veto=Max_abs_stat_veto,
                               fn_fit_veto=Fn_fit_veto,
                               n_parts_veto_snr=N_parts_veto_snr,
                               min_bins_veto_part_time=Min_bins_veto_part_time,
                               lobe_hlen_veto_part_time=Lobe_hlen_veto_part_time,
                               n_parts_veto_fourier=N_parts_veto_fourier,
                               wind_len_overlap_fourier=Wind_len_overlap_fourier,
                               fraction_template_duration_edge=1., days_edge=None, days_hole_max=0.75,
                               fraction_template_duration_hole=1., kill_edges=False, kill_around_hole=False,
                               fraction_template_duration_minimal_neighbors=1.5,
                               delta_snr_kill_sf_minimal_neighbors=5e-2, veto_extreme_statistics=True,
                               threshold_fn_in_quarter_step=1.,
                               veto_positive_outliers=True,
                               veto_tail_statistics=True,
                               use_dist_to_hole_masks_for_threshold=True, fn_close_to_hole=5e-2,
                               use_snr_dependent_threshold=True,
                               update_stat_and_veto_masks=True, zero_vetted_stats=True, renormalize_stats_variances=True,
                               save_masks_as_attributes=False):
        """
        Performs the search of single-transit events in all the quarters if the conditioned and whitened
        data and for all the conditioned and whitened templates.
        Applies shape quality veto to the statistics. Investigates the distribution of the resulting score,
        corrects its variance, calculates correction for its non-gaussianity, calculates the gaussianized
        score which is a metric for estimating the lower limit of the likelihood.
        Please see the parameters documentation inside each method.
        :param save_no_veto_no_score_cor_stats_as_attr: Whether to save the statistics before they underwent the
            vetting and the score distribution gaussianization. These statistics may be needed in the postprocessing.
        :type save_no_veto_no_score_cor_stats_as_attr: bool.
        :param nsig_outliers_stat: Deviation in units of standard deviation to be considered as an outlier in the
            outlier-free variance calculation for the statistic.
        :type nsig_outliers_stat: float
        :param apply_veto: Whether to apply transit shape quality veto
        :type apply_veto: bool
        :param gaussianize: Whether to calculate the gaussianized statistics, that is statistics which have the
            same ranking as the given statistics but follow exactly the Gaussian distribution.
        :type gaussianize: bool
        """
        self.matched_filtering(nsig_outliers_stat=nsig_outliers_stat)
        if save_no_veto_no_score_cor_stats_as_attr:
            self.stats_raw_no_veto_no_score_cor = copy.deepcopy(self.statistics_raw)
        if apply_veto:
            self.veto_statistics(apply_only_to_neg=apply_veto_only_to_neg,
                                 margin_snr_veto=margin_snr_veto,
                                 max_abs_stat_veto=max_abs_stat_veto,
                                 n_parts_veto_snr=n_parts_veto_snr,
                                 min_bins_veto_part_time=min_bins_veto_part_time,
                                 lobe_hlen_veto_part_time=lobe_hlen_veto_part_time,
                                 n_parts_veto_fourier=n_parts_veto_fourier,
                                 wind_len_overlap_fourier=wind_len_overlap_fourier,
                                 fraction_template_duration_edge=fraction_template_duration_edge,
                                 days_edge=days_edge, days_hole_max=days_hole_max,
                                 fraction_template_duration_hole=fraction_template_duration_hole,
                                 kill_edges=kill_edges, kill_around_hole=kill_around_hole,
                                 fraction_template_duration_minimal_neighbors=fraction_template_duration_minimal_neighbors,
                                 delta_snr_kill_sf_minimal_neighbors=delta_snr_kill_sf_minimal_neighbors,
                                 veto_extreme_statistics=veto_extreme_statistics,
                                 veto_tail_statistics=veto_tail_statistics,
                                 threshold_fn_in_quarter_step=threshold_fn_in_quarter_step,
                                 veto_positive_outliers=veto_positive_outliers,
                                 use_dist_to_hole_masks_for_threshold=use_dist_to_hole_masks_for_threshold,
                                 fn=fn_fit_veto, fn_close_to_hole=fn_close_to_hole,
                                 use_snr_dependent_threshold=use_snr_dependent_threshold,
                                 update_stat_and_veto_masks=update_stat_and_veto_masks,
                                 zero_vetted_stats=zero_vetted_stats, renormalize_stats_variances=renormalize_stats_variances,
                                 save_masks_as_attributes=save_masks_as_attributes)
        self.cut_zero_padding_ends()
        self.calculate_ses_distr_corrections()
        if gaussianize:
            self.calculate_gaussianized_ses()

    def periodicity_search(self, p_min=None, p_max=None, dp=None, duty_cycle=Duty_cycle, periods=None,
                           delete_full_stats=True, force_interleave_shifts=True, delete_periodic_interleaved=True,
                           save_uncorrected_periodic=False, delete_non_marginalized=False,
                           overlap_period_bins=P_bins_overlap):
        """
        Wrapper for several methods in creating the final product of the pipeline, SNR of planets of different
        periods and first transit times.
        Starts from the connected-quarters statistics. Performs folding over periods, applies the single-event
        statistic distribution non-gaussianity correction, marginalizes over templates.
        :param p_min: Minimal period in the search. If None, `self.pmin` is used. Overriden by `periods`
        :type p_min: float or None
        :param p_max: Maximal period in the search. If None, `self.pmax` is used. Overriden by `periods`
        :type p_max: float or None
        :param dp: Period step. If None, will be calculated from the `duty_cycle`. Usually is always None.
        :type dp: float or None
        :param duty_cycle: Maximal accumulated deviation due to the period error in the end of the mission.
            In units of days. Overriden by `periods`.
        :type duty_cycle: float or None
        :param periods: Specifies all the periods, and then the previous parameters are not needed.
            Overrides `p_min`, `p_max`, `dp`, `duty_cycle`
        :type periods: numpy.ndarray of float
        :param delete_full_stats: Whether to delete full statistics after interleaving them
        :type delete_full_stats: bool
        :param force_interleave_shifts: Whether to create interleaved shifts anyway even if they are created.
            This can be important if there was scrambling made, and the interleaving should be done with new
            full statistics.
        :type force_interleave_shifts: bool
        :param delete_periodic_interleaved: whether to delete the previous arrays with interleaved and
            non-interleaved templates, or no
        :type delete_periodic_interleaved: bool
        :param save_uncorrected_periodic: Whether to save the periodic statistics before applying the
            single-event statistic distribution non-gaussianity correction
        :type save_uncorrected_periodic: bool
        :param delete_non_marginalized: Whether to delete the periodic statistics before marginalization
            over templates.
        :type delete_non_marginalized: bool
        :param overlap_period_bins: Number of spare period bins added to the periods array for the best trigger
            not to be at the boundary of the array. This is needed because the integral score will not be full
            if the trigger is on the boundary.
        :type overlap_period_bins: int
        """
        if force_interleave_shifts or (not hasattr(self, 'statistics_interleaved') or self.statistics_interleaved is None):
            self.interleave_shifts(delete_fulls=delete_full_stats)  # quarters will be connected inside if they are not
        self.sum_periods(p_min=p_min, p_max=p_max, dp=dp, duty_cycle=duty_cycle, periods=periods, safety_bins=overlap_period_bins)
        self.calculate_periodic_scores_with_correction(save_uncorrected=save_uncorrected_periodic)
        self.downsample_interleaved(delete_periodic_interleaved=delete_periodic_interleaved)
        self.marginalize_over_templates(m_1_3_r_s=self.m_1_3_r_s, delete_non_marginalized=delete_non_marginalized)

    def subtract_means(self, masks=None, save_means=True, method=Average_flux_method):
        """
        Subtracts mean value from each quarter's original flux
        Mean values of the flux are subtracted to have signal centered around zero.
        Means are saved in self.orig_means:
        :param save_means: Whether to create a new attribute self.orig_means and save the mean
            values of the fluxes there. The default is True.
        :type save_means: bool
        :param method: which function to use in calculating the average
        :type method: str, 'mean' or 'median'
        :param masks: which points to use in the average calculation
        :type masks: list of numpy.ndarrays 1d o bool. True means to take this point, False means to exclude
        """
        assert method == 'mean' or method == 'median', "`method` must be 'mean' or 'median'!"
        new_means = self.get_averages(masks=masks, method=method)
        # If there were saved old means, replace them by the new ones
        if self.orig_means is None:
            old_means = np.zeros(len(self.quarters))
        else:
            old_means = self.orig_means
        self.fluxes = [np.array(self.fluxes[q] + old_means[q] - new_means[q]) for q in self.quarters]
        self.clean_fluxes = [np.array(self.clean_fluxes[q] + old_means[q] - new_means[q]) for q in self.quarters]
        if save_means:
            self.orig_means = new_means
            lens = [len(self.clean_fluxes[q][~np.isnan(self.clean_fluxes[q])]) for q in self.quarters]
            self.mean_total = np.sum([self.orig_means[q] * lens[q] for q in self.quarters]) / np.sum(lens)

    def get_averages(self, masks=None, method=Average_flux_method):
        """
        Get the averages of `self.orig_fluxes` for every quarter
        :param masks: Masking as True the points which should participate in the average.
            If None, will take not-nen masks.
        :type masks: list of numpy.ndarrays of bool, or None
        :param method: how to calculate the average: mean or median
        :type method: str, 'mean' or 'median'
        :return: averages
        :rtype: list of float
        """
        if masks is None:
            masks = [(~np.isnan(self.orig_fluxes[q])) & (np.isfinite(self.orig_fluxes[q])) for q in self.quarters]
        if method.lower() == 'mean':
            averages = [np.nanmean(self.orig_fluxes[q][masks[q]]) for q in self.quarters]
        elif method.lower() == 'median':
            averages = [np.nanmedian(self.orig_fluxes[q][masks[q]]) for q in self.quarters]
        else:
            raise ValueError("`method` should be 'mean' or 'median'!")
        return averages

    def normalize_by_means(self):
        """
        Makes all the flux means to be the same, so that we work in the relative units in which
        the amplitudes of all the fluxes are supposed to be the same. The data is corrected for the crowding metric,
        so that should really be true. All the formulae are now simplified because we do not have to drag the changing
        mean flux trends. The actual value of the mean anyway cancels in the final formula. We do renormalize the fluxes by the
        new mean though because we do not what to work with too small numbers. We note that this changes also the values of
        the PSDs etc.
        :return:
        :rtype:
        """
        for q in self.quarters:
            self.fluxes[q] = self.fluxes[q] / self.orig_means[q] * self.mean_total
            self.clean_fluxes[q] = self.clean_fluxes[q] / self.orig_means[q] * self.mean_total

    def mask_flux_outliers(self, nsig=Nsig_outliers_0, nsig_positive=Nsig_outliers_0_pos,
                           nsig_negative=Nsig_outliers_0_neg, nsig_clean=Nsig_outliers_psd,
                           detr_length_outliers=Detr_length_outliers, detr_method_outliers=Detr_method_outliers,
                           detr_endmethod_outliers=Detr_endmethod_outliers,
                           in_whitened_fluxes=False, modify_clean_masks=True):
        """
        Detects outliers of the flux with respect to the trend. The trend is found by detrending.
        Modifies self.masks by setting False where the outliers were found.
        :param nsig: Minimal deviation from the trend considered as outlier. Is overriden by nsig_positive and nsig_negative.
        :type nsig: float
        :param nsig_positive: Maximum positive deviation from the trend in units of std of data (more extreme
            values will be marked as outliers). Overrides nsig. If is None, will be replaced by nsig.
        :type nsig_positive: float
        :param nsig_negative: Maximum negative deviation from the trend in units of std of data (more extreme
            values will be marked as outliers). Overrides nsig. If is None, will be replaced by nsig.
        :type nsig_negative: float
        :param nsig_clean: Outliers threshold for clean masks
        :type nsig_clean: float
        :param detr_length_outliers: Detrending length to detrend before measuring outlier level
        :type detr_length_outliers: int
        :param detr_method_outliers: The type of moving average using to define the trend.
            See the function documentation.
        :type detr_method_outliers: str, 'mean' or 'median'
        :param detr_endmethod_outliers: The type of moving average using to define the trend at the ends of the
            data. See the function documentation.
        :type detr_endmethod_outliers: str, 'mean' or 'median'
        :param in_whitened_fluxes: If true, the search will be performed not in self.fluxes but in self.whitened_fluxes_td
        :type in_whitened_fluxes: bool
        :param modify_clean_masks: If True, self.psd_masks will be modified setting False to the
            outliers positions. If False, only self.masks will be modified
        :type modify_clean_masks: bool
        :return: modifies self.masks, self.psd_masks
        :rtype:
        """
        if not in_whitened_fluxes:
            for q in self.quarters:
                outlier_mask, clean_mask = ut.mask_outliers_fun(np.array(self.fluxes[q]),
                                                                mask=np.array(self.clean_masks[q]),
                                                                nsig0=nsig, nsig1=nsig_clean,
                                                                nsig_positive=nsig_positive, nsig_negative=nsig_negative,
                                                                detr_length=detr_length_outliers,
                                                                detr_method=detr_method_outliers,
                                                                detr_endmethod=detr_endmethod_outliers,
                                                                return_symmetric_mask=True)
                self.masks[q] *= outlier_mask
                if modify_clean_masks:
                    self.clean_masks[q] *= clean_mask
        else:
            assert self.is_fluxes_whitened, 'Cannot mask outliers in whitened fluxes because fluxes are nbnot whitened!'

            for q in self.quarters:
                outlier_mask, clean_mask = ut.mask_outliers_fun(np.array(self.whitened_fluxes_td[q]),
                                                                mask=np.array(self.clean_masks[q]),
                                                                nsig0=nsig, nsig1=nsig_clean,
                                                                nsig_positive=nsig_positive, nsig_negative=nsig_negative,
                                                                detr_length=detr_length_outliers,
                                                                detr_method=None, detr_endmethod=detr_endmethod_outliers,
                                                                return_symmetric_mask=True)
                self.masks[q] *= outlier_mask
                if modify_clean_masks:
                    self.clean_masks[q] *= clean_mask

    def fill_additional_outliers(self, nsig=Nsig_outliers_0, nsig_positive=Nsig_outliers_0_pos,
                                 nsig_negative=Nsig_outliers_0_neg, nsig_clean=Nsig_outliers_psd,
                                 detr_length_outliers=Detr_length_outliers, detr_method_outliers=Detr_method_outliers,
                                 detr_endmethod_outliers=Detr_endmethod_outliers,
                                 in_whitened_fluxes=False, modify_clean_masks=True, interpolate_clean_fluxes=True):
        """
        Detects outliers of the flux with respect to the trend and interpolates them without changing the interpolation
        that was dona before in the pre-existing mask. Modifies only the points of the new outliers if any.
        The trend is found by detrending. Modifies self.masks by setting False where the outliers were found.
        :param nsig: Minimal deviation from the trend considered as outlier. Is overriden by nsig_positive and nsig_negative.
        :type nsig: float
        :param nsig_positive: Maximum positive deviation from the trend in units of std of data (more extreme
            values will be marked as outliers). Overrides nsig. If is None, will be replaced by nsig.
        :type nsig_positive: float
        :param nsig_negative: Maximum negative deviation from the trend in units of std of data (more extreme
            values will be marked as outliers). Overrides nsig. If is None, will be replaced by nsig.
        :type nsig_negative: float
        :param nsig_clean: Outliers threshold for clean masks
        :type nsig_clean: float
        :param detr_length_outliers: Detrending length to detrend before measuring outlier level
        :type detr_length_outliers: int
        :param detr_method_outliers: The type of moving average using to define the trend.
            See the function documentation.
        :type detr_method_outliers: str, 'mean' or 'median'
        :param detr_endmethod_outliers: The type of moving average using to define the trend at the ends of the
            data. See the function documentation.
        :type detr_endmethod_outliers: str, 'mean' or 'median'
        :param in_whitened_fluxes: If true, the search will be performad not in self.fluxes but in self.whitened_fluxes_td
        :type in_whitened_fluxes: bool
        :param modify_clean_masks: If True, self.psd_masks will be modified setting False to the
            outliers positions. If False, only self.masks will be modified
        :type modify_clean_masks: bool
        :param interpolate_clean_fluxes:
        :type interpolate_clean_fluxes:
        :return: modifies self.masks, self.psd_masks
        :rtype:
        """
        new_outlier_masks = []
        new_clean_outlier_masks = []

        if not in_whitened_fluxes:
            for q in self.quarters:
                outlier_mask, clean_mask = ut.mask_outliers_fun(np.array(self.fluxes[q]),
                                                                mask=np.array(self.clean_masks[q]),
                                                                nsig0=nsig, nsig1=nsig_clean,
                                                                nsig_positive=nsig_positive, nsig_negative=nsig_negative,
                                                                detr_length=detr_length_outliers,
                                                                detr_method=detr_method_outliers,
                                                                detr_endmethod=detr_endmethod_outliers,
                                                                return_symmetric_mask=True)
                new_outlier_masks.append(outlier_mask)
                new_clean_outlier_masks.append(clean_mask)
                self.masks[q] *= outlier_mask
                if modify_clean_masks:
                    self.clean_masks[q] *= clean_mask
        else:
            assert self.is_fluxes_whitened, 'Cannot mask outliers in whitened fluxes because fluxes are nbnot whitened!'

            for q in self.quarters:
                outlier_mask, clean_mask = ut.mask_outliers_fun(np.array(self.whitened_fluxes_td[q]),
                                                                mask=np.array(self.clean_masks[q]),
                                                                nsig0=nsig, nsig1=nsig_clean,
                                                                nsig_positive=nsig_positive, nsig_negative=nsig_negative,
                                                                detr_length=detr_length_outliers,
                                                                detr_method=None, detr_endmethod=detr_endmethod_outliers,
                                                                return_symmetric_mask=True)
                new_outlier_masks.append(outlier_mask)
                new_clean_outlier_masks.append(clean_mask)
                self.masks[q] *= outlier_mask
                if modify_clean_masks:
                    self.clean_masks[q] *= clean_mask

        for q in self.quarters:
            if (~new_outlier_masks[q]).any():
                self.fluxes[q] = ut.interpolate_over_mask(np.array(self.fluxes[q]), np.array(new_outlier_masks[q]),
                                                          interp_method='linear', ends='continue',
                                                          mitigate_hole_ends=False, endhole_method='median')
            if (~new_clean_outlier_masks[q]).any():
                if interpolate_clean_fluxes:
                    self.clean_fluxes[q] = ut.interpolate_over_mask(np.array(self.clean_fluxes[q]),
                                                                    np.array(new_clean_outlier_masks[q]),
                                                                    interp_method='linear', ends='continue',
                                                                    mitigate_hole_ends=False, endhole_method='median')

    def make_big_hole_masks(self, max_small_hole=Max_small_hole, exclude_individual_points=False, individual_point_size=2):
        """
        Remove from the masks small holes, that is holes that can be linearly interpolated and ignored.
        :param max_small_hole: Maximal hole to be considered small, so that it can be
            interpolated linearly and ignored
        :type max_small_hole: int
        :param exclude_individual_points: If i between two holes there is one non-hole point, also make it hole
        :type exclude_individual_points: bool
        :param individual_point_size: The size in bins of good pieces inside a hole which are considered too small to keep.
        :type individual_point_size: int
        """
        self.big_hole_masks = []
        for q in self.quarters:
            # Warning: the exclusion of individual points from the regular masks was added after the run_12 main search was done!
            # It was needed because the individual points in the mask make problems after that, and not always it happens
            # that the big hole mask individual point removal removes them.,
            if exclude_individual_points:
                self.masks[q] *= np.array(~ut.clean_small_holes(np.array(~self.masks[q]),
                                                                max_hole=individual_point_size)).astype(bool)
            big_mask = ut.clean_small_holes(np.array(self.masks[q]), max_hole=max_small_hole)
            if exclude_individual_points:
                big_mask = np.array(~ut.clean_small_holes(np.array(~big_mask), max_hole=individual_point_size))
                self.masks[q] *= big_mask
                self.clean_masks[q] *= big_mask
            self.big_hole_masks.append(big_mask)

    def mask_ephemeris(self, period=None, durations=None, transit_times=None, absolute_phase=None, time_edges_to_mask=None):
        """
        Adds `False` to the mask where there are transits.
        :param period: Period in days of transits that should be masked. Used only if `time_edges_to_mask`
            is None and `transit_times` is None. This is useful when there are no TTVs or TDVs.
        :type period: float
        :param durations: Duration or durations in days of transits that should be masked. Used only if
            `time_edges_to_mask` is None. If `float`, then it is used as constant duration for all transits.
            If `list`, then its length should coincide with the length of `transit_times` if it is also a list.
            This is useful when there are TDVs.
        :type durations: float or list[float]
        :param transit_times: Middle-time of the transit or transits in days. If `float`, it is counted as
            middle-time of one transit, and all the rest are assumed to pe periodic with `period`.
            If `list`, will be counted as all the mid-times that should be masked. Should coincide in length
            with `durations` if it is also a list. This is useful when there are TTVs.
            Is overriden by time_edges_to_mask.
        :type transit_times: list[float] or float or ndarray
        :param absolute_phase: Phase of one of the transits with respect to the beginning of the Kepler time
            in days. Used only if `time_edges_to_mask` is None and `transit_times` is None.
            This is useful when there are no TTVs or TDVs.
        :type absolute_phase: float
        :param time_edges_to_mask: list of tuples marking the beginning and the end of the things that should
            be masked in days. Overriding all the other parameters which are used if it is None.
            This is useful when there are TTVs and TDVs.
        :type time_edges_to_mask: list[list[float]] or ndarray
        """
        self.masks, self.big_hole_masks = af.mask_ephemeris(times=self.times, quarters=self.quarters, masks=self.masks,
                                                            big_hole_masks=self.big_hole_masks, dt_days=self.dt_days,
                                                            period=period, durations=durations, transit_times=transit_times,
                                                            absolute_phase=absolute_phase,
                                                            time_edges_to_mask=time_edges_to_mask)

    def mask_ephemeris_connected_quarters(self, period=None, durations=None, transit_times=None, absolute_phase=None,
                                          time_edges_to_mask=None):
        """
        Same as `mask_ephemeris` but for the connected quarters.
        :param period:
        :type period:
        :param durations:
        :type durations:
        :param transit_times:
        :type transit_times:
        :param absolute_phase:
        :type absolute_phase:
        :param time_edges_to_mask:
        :type time_edges_to_mask:
        :return:
        :rtype:
        """
        big_hole_mask, _ = af.mask_ephemeris(times=[np.array(self.time_full)], quarters=[0], masks=[np.array(self.big_hole_masks_full)],
                                             big_hole_masks=[np.array(self.big_hole_masks_full)], dt_days=self.dt_days,
                                             period=period, durations=durations, transit_times=transit_times,
                                             absolute_phase=absolute_phase, time_edges_to_mask=time_edges_to_mask)
        big_hole_mask = big_hole_mask[0]
        self.big_hole_masks_full = big_hole_mask
        self.stat_masks_full &= big_hole_mask
        for k in range(len(self.stat_masks_full)):
            self.statistics_full[k][~self.stat_masks_full[k]] = 0.
            self.stat_vars_full[k][~self.stat_masks_full[k]] = 0.
            self.corrections_full[k][~self.stat_masks_full[k]] = 0.
            self.statistics_gaussianized_full[k][~self.stat_masks_full[k]] = 0.

    def prepare_for_psd(self, nperseg=None, noverlap=Noverlap_psd, npsdsmin=Npsdsmin,
                        max_small_hole=Max_small_hole):
        """
        Checks whether the number of pieces for the Welch method is enough compare to npsdsmin, and
        if no, interpolates over mask and updates the psd mask to measure psd for the first time like that.
        Creates  self.psd_masks.
        :param nperseg: length of slices in Welch method
        :type nperseg: int
        :param noverlap: If int: length of overlap of slices in Welch method.
                         If float<1 : relative overlap to nperseg.
                         Length of overlap will be int(nperseg*noverlap)
        :type noverlap: int or float<1
        :param npsdsmin: minimum number of slices to measure psd with Welch. If the actual number is smaller, interpolation will be made.
        :type npsdsmin: int
        :param max_small_hole: Maximal hole which can be considered small and ignored if we lack pieces for PSD
        :type max_small_hole: int
        """
        if nperseg is None:
            nperseg = self.nperseg_psd
        self.psd_masks = copy.deepcopy(self.clean_masks)
        if not self.is_holes_interpolated:
            warnings.warn("Data is not interpolated. Cleaning small holes may cause NaN in the fluxes.")
        if noverlap < 1:
            noverlap = int(nperseg * noverlap)
        if noverlap == -1:  # an old piece of code for the case is we still use notation of noverlap=-1
            noverlap = nperseg // 2
        # Calculating the number of pieces available for psd measurement. If this number is too low, the interpolation
        # over holes will be done first
        self.npsds0 = np.zeros(len(self.fluxes))
        for i in self.quarters:
            for ind1 in range(0, len(self.fluxes[i]) - nperseg, nperseg - noverlap):
                if np.sum((~self.psd_masks[i])[ind1: ind1 + nperseg]) == 0:
                    self.npsds0[i] += 1
            if self.npsds0[i] < npsdsmin:  # if the number of psd samples is insufficient, first try to interpolate small holes
                mask_big = ut.clean_small_holes(np.array(self.psd_masks[i]), max_hole=max_small_hole)  # mask of only big and edge holes
                npsds1 = 0
                # new number of psd samples will be:
                for ind1 in range(0, len(self.fluxes[i]) - nperseg, nperseg - noverlap):
                    if np.sum((~mask_big)[ind1: ind1 + nperseg]) == 0:
                        npsds1 += 1
                if npsds1 >= npsdsmin:  # if it is sufficient, go with it
                    self.psd_masks[i] = mask_big
                else:  # if the new number is still not sufficient, then interpolate all:
                    if self.is_zero_padded:
                        self.psd_masks[i] = np.concatenate((np.zeros((self.zero_padding_ends[i])[0]),
                                                            np.ones(len(self.orig_fluxes[i])),
                                                            np.zeros((self.zero_padding_ends[i])[1]))).astype(bool)
                    else:
                        self.psd_masks[i] = np.ones(len(self.psd_masks[i])).astype(bool)
                        # in psd_masks masks, inner holes are interpolated to increase the number of psd samples
        self.is_prepared_for_psd = True

    def window_fluxes(self, alpha=Alpha_windowing_fluxes, window_clean_fluxes=True):
        self.fluxes = ut.window_data(self.fluxes, alpha=alpha)
        if window_clean_fluxes:
            self.clean_fluxes = ut.window_data(self.clean_fluxes, alpha=alpha)

    def pad_by_zeros(self):
        """
        Pads fluxes and masks by zeros; extrapolates times.
        The new length will be `self.support`.
        The times will be also interpolated to close the holes.
        """
        self.zero_padding_ends = []
        for i in self.quarters:
            zerlen1 = (self.support - len(self.fluxes[i])) // 2
            zerlen2 = (self.support - len(self.fluxes[i]) + 1) // 2
            self.zero_padding_ends.append((zerlen1, zerlen2))
            self.fluxes[i] = np.array(np.concatenate((np.zeros(zerlen1), self.fluxes[i], np.zeros(zerlen2))), dtype=Dtype_r)
            self.clean_fluxes[i] = np.array(np.concatenate((np.zeros(zerlen1), self.clean_fluxes[i], np.zeros(zerlen2))), dtype=Dtype_r)
            self.masks[i] = np.concatenate((np.zeros(zerlen1), self.masks[i], np.zeros(zerlen2))).astype(bool)
            self.big_hole_masks[i] = np.concatenate((np.zeros(zerlen1), self.big_hole_masks[i], np.zeros(zerlen2))).astype(bool)
            self.clean_masks[i] = np.concatenate((np.zeros(zerlen1), self.clean_masks[i], np.zeros(zerlen2))).astype(bool)
            self.psd_masks[i] = np.concatenate((np.zeros(zerlen1), self.psd_masks[i], np.zeros(zerlen2))).astype(bool)
            self.times[i] = np.array(np.concatenate((np.zeros(zerlen1) + np.nan, self.times[i], np.zeros(zerlen2) + np.nan)),
                                     dtype=Dtype_r)
            # NaNs are needed for the `interpolate_time` function to treat these regions
            # as the ones to be interpolated. The self.masks[i] is not transferred to the
            # time interpolation because not all the masked flux regions are NaNs in time,
            # and we do not want to lose precision rejecting data.
            self.times[i] = np.array(ut.interpolate_time(np.array(self.times[i]), mask=None), dtype=Dtype_r)
        self.fluxes = np.array(self.fluxes, dtype=Dtype_r)  # need them to be numpy.arrays for numba inpainting
        self.masks = np.array(self.masks)
        self.is_zero_padded = True

    def measure_psds(self, nperseg=None, noverlap=Noverlap_psd, window=Window_psd,
                     subtract_mean=Subtract_mean_in_psd, multi_resolution_psds=Multi_resolution_psds,
                     tukey_steps_len=Tukey_steps_len_psd,
                     always_zero_zeroth_freq=Always_zero_zeroth_freq_psd,
                     feature_sizes=(1, 9, 3.5, 4, 2., 3),
                     reject_outliers=Reject_outliers_psd, outlier_sf=Psd_outlier_sf,
                     n_repetitions_outliers=N_repetitions_outliers_psd):
        """
        PSDS as function of quarter id.
        The PSD measuring algorithm makes the evenly sampled steps and not adjust the search for
        good pieces according to the placement of the zeros in mask.
        Creates self.psds, self.psdfreqs, self.wf_tds, self.wf_fds, self.freqs, self.npsds.

        :param nperseg: the length of a sample used in Welch method
        :type nperseg: int
        :param noverlap: If int: length of overlap of slices in Welch method.
            If float<1 : relative overlap to nperseg.
            Length of overlap will be int(nperseg*noverlap)
        :type noverlap: int or float<1
        :param window: window function by which the signals are multiplied to avoid boundary effects and crosstalk
            Possible options: 'hann', 'boxcar', where 'boxcar' is just normalized bow, or no window
            Default is 'hann'. It is not recommended to use 'boxcar'.
        :type window: str, 'hann' or 'boxcar'
        :param subtract_mean: If True, the mean of the signal is subtracted. It is recommended to do it because
            otherwise unwanted power may appear in the zero frequency
            Default is True
        :type subtract_mean: bool
        :param multi_resolution_psds: whether to apply multi-resolution smoothing to mitigate the noise effects
        :type multi_resolution_psds: bool
        :param tukey_steps_len: length taken for zeroing in the tukey windowing
        :type tukey_steps_len: int or float
        :param always_zero_zeroth_freq: whether to always put the lowest frequencies to zero
        :type always_zero_zeroth_freq: bool
        :param feature_sizes: length and depth of the deviations that will trigger increasing nperseg, see in the code
        :type feature_sizes:
        :param reject_outliers: Whether to reject outliers in power when averaging PSD slices
        :type reject_outliers: bool
        :param outlier_sf: Survival function of rejected outliers (instead of "n sigma")
        :type outlier_sf: float
        :param n_repetitions_outliers: How many times iteratively re-measure the mean PSD and then reject outliers
            from the exponential distribution with this mean. If 0, no iterations are done. Iterations introduce
            bias if the distribution is purely exponential because of the measured mean which is used for the rejection.
            However, when real outliers are present, iterations help to mitigate the mean measuring bias coming from the outliers.
        :type n_repetitions_outliers: int
        """
        d1, w1, d2, w2, d3, w3 = feature_sizes
        if nperseg is None:
            nperseg = self.nperseg_psd

        if not self.is_prepared_for_psd:
            warnings.warn("Data is not prepared for PSD. Use prepare_for_psd method.")

        if self.is_fluxes_inpainted:
            warnings.warn("Fluxes have been inpainted, new PSD will make inpainting invalid for detection.")

        fftfun = rfft
        ifftfun = irfft
        freqfun = np.fft.rfftfreq
        dtype_res = np.float64

        if subtract_mean:
            clean_fluxes = [self.clean_fluxes[i] - np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]]) for i in
                            range(len(self.clean_fluxes))]
        else:
            clean_fluxes = self.clean_fluxes

        self.freqs = [freqfun(len(time), d=self.dt_days) for time in self.times]

        #############################################################################################################
        # Multi-resolution part (Smoothing)
        #############################################################################################################
        if multi_resolution_psds:
            # Lowest freqs cannot be resolved by small nperseg by definition. And also, there will be leakage.
            # So we just put there better resolution straight away
            nperseg_max = 2 * nperseg
            log2_factor_max = int(np.round(np.log2(nperseg_max / nperseg)))
            log2_factor_min = -3  # 1/8 of the nperseg
            npersegs = [int(np.round(2. ** log2_factor * nperseg)) for log2_factor in np.arange(log2_factor_max, log2_factor_min - 1, -1)]
            len_interpolation_wf_coarse = nperseg_max
            self.psdfreqs = [freqfun(nperseg_max, d=self.dt_days) for _ in self.quarters]
            self.orig_psds = []
            self.psds = []
            self.wf_fds_meas = []
            wfs_coarse_res_quarters = []

            for q in self.quarters:
                psds_resolutions_quarter = [
                    ut.welch_psds(clean_fluxes=[clean_fluxes[q]], psd_masks=[self.psd_masks[q]], nperseg=nperseg_res, noverlap=noverlap,
                                  window=window, reject_outliers=reject_outliers, outlier_sf=outlier_sf,
                                  n_repetitions_outliers=n_repetitions_outliers, return_freqs=False, dt=self.dt_days,
                                  return_npsds=False)[0] for nperseg_res in npersegs]
                len_wf_coarse_res = len(psds_resolutions_quarter[0])
                wfs_coarse_res = [1 / psds_resolutions_quarter[0] ** 0.5] + [
                    (ut.sinc_interp(1 / psds_resolutions_quarter[ind_res] ** 0.5, len_interpolation_wf_coarse,
                                    tukey_steps_len=tukey_steps_len)) for ind_res in np.arange(1, len(npersegs))]

                max_residuals = wfs_coarse_res[0] - wfs_coarse_res[-1]
                std_noisy = ut.outlier_free_std(max_residuals[len(max_residuals) // 2:], nsig=3, n_iterations=2)

                wfs_coarse_res_quarters.append(wfs_coarse_res)
                combined_wf_coarse_res = np.array(wfs_coarse_res[-1])

                for ind_res in np.arange(len(npersegs))[1:-1][::-1]:  # the last one is already here, the first is too long nperseg
                    replace_bins = int(np.floor(1 / 2 * len_wf_coarse_res * (npersegs[-1] / npersegs[ind_res])))
                    nps_ratio = int(nperseg / npersegs[ind_res])
                    combined_wf_coarse_res, actual_repl_size = ut.replace_beginning(vector=combined_wf_coarse_res,
                                                                                    replacement=wfs_coarse_res[ind_res],
                                                                                    min_repl_size=replace_bins, return_repl_size=True)
                    combined_wf_coarse_res = ut.replace_if_deviating(combined_wf_coarse_res, wfs_coarse_res[ind_res], tester=None,
                                                                     mindev=(std_noisy * 16 / nps_ratio), minwid=2 * nps_ratio)

                # This is a mystery and was found totally empirically. I tried to remove it, but with it, it works better:
                residuals = wfs_coarse_res[0] - combined_wf_coarse_res
                nsig_sign_minwids_repl_vecs_i_min = [[d3, -1., w3, wfs_coarse_res[1], 0],
                                                     [d1, -1., w1, wfs_coarse_res[2], 16],  # for mismatch behavior
                                                     [d1, +1., w1, wfs_coarse_res[2], 16],
                                                     [d2, -1., w2, wfs_coarse_res[1], 0],
                                                     [d2, +1., w2, wfs_coarse_res[1], 0]]  # ,  # for real features

                for nsig, sign, min_wid, repl_vec, i_min in nsig_sign_minwids_repl_vecs_i_min:
                    i = i_min
                    while i < len(combined_wf_coarse_res):
                        if sign * max_residuals[i] - nsig * std_noisy > 0.:
                            i_1 = i
                            i_2 = i
                            while (i_2 < len(max_residuals)) and (sign * max_residuals[i_2] - nsig * std_noisy > 0):
                                i_2 += 1
                            if (i_2 - i_1) >= min_wid or (i_1 <= 2 and (i_2 - i_1) >= min_wid / 2):
                                # find the beginning and the end of the replacement region:
                                j_1 = i_1
                                while (j_1 - 1 >= 0) and (sign * residuals[j_1 - 1] > 0):
                                    j_1 -= 1
                                j_2 = i_2
                                while (j_2 < len(max_residuals)) and (sign * residuals[j_2] > 0):
                                    j_2 += 1  # this is the last slice index, so it can be bigger
                                combined_wf_coarse_res[j_1:j_2] = repl_vec[j_1:j_2]
                                i = j_2 + 1
                            else:
                                i = i_2 + 1
                        else:
                            i += 1

                # Making the lowest freqs wf_fd non-decreasing

                # We do not want the zeroth freq to be an outlier.
                # We assume that the PSD should not drop towards the zero bin. If it does in reality, it is anyway
                # not well-measured and is prone to under-whitening. So we make it non-decreasing by punishing more
                # the lowest frequencies

                for j in [1, 2, 3]:  # Fix at these indices
                    i = 0
                    while combined_wf_coarse_res[i + j] < combined_wf_coarse_res[i]:
                        combined_wf_coarse_res[:i + j] = combined_wf_coarse_res[i + j]
                        i += 1
                        if i >= len(combined_wf_coarse_res) - j - 2:
                            break

                self.orig_psds.append(psds_resolutions_quarter[npersegs.index(nperseg)])
                self.wf_fds_meas.append(combined_wf_coarse_res)
                self.psds.append(1 / combined_wf_coarse_res ** 2)
        else:
            self.orig_psds, self.psdfreqs = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks, nperseg=nperseg,
                                                          noverlap=noverlap, window=window, reject_outliers=True, outlier_sf=outlier_sf,
                                                          n_repetitions_outliers=n_repetitions_outliers, return_freqs=True, dt=self.dt_days,
                                                          return_npsds=False)
            self.psds = copy.deepcopy(self.orig_psds)
            self.wf_fds_meas = [1 / self.psds[q] ** 0.5 for q in range(len(self.psds))]

        #############################################################################################################
        # Whitening filter
        #############################################################################################################
        # compare value of the wf_fd at low and high freqs. If the low one drops too fast, we will apply the Tukey
        # window and zero the wf_fd around zero. This will help to avoid unresolved residual low-freq red noise.
        assert self.support % 2 == 0, 'self.support assumed to be even!'
        self.wf_tds = []
        self.wf_fds = []
        self.wf_fds_unwindowed = []
        for q, wpsd in enumerate(self.wf_fds_meas):
            wf_td = ifftfun(wpsd)
            meas_len = len(wf_td)
            wf_td = np.fft.ifftshift(np.fft.fftshift(wf_td) * ut.tukey(meas_len, alpha=tukey_steps_len / meas_len, sym=True))
            mid_point = len(wf_td) // 2
            result = np.zeros(self.support, dtype=dtype_res)
            result[:mid_point] = wf_td[:mid_point]
            result[-mid_point:] = wf_td[-mid_point:]

            wf_fd = fftfun(result)
            wf_fd[0] = 0.  # MAYBE THIS IS WRONG! also, the flux was already supposed to have zero mean. see below what I did.
            wf_fd = np.abs(wf_fd)  # anyway in the formula it appears twice, so the phase will get canceled
            # but maybe this exactly orig_means that we should not take this abs?
            self.wf_fds_unwindowed.append(np.array(wf_fd))  # it is unwindowed in the large-window sense of this world

            whitened_clean_flux_fd = fftfun(self.clean_fluxes[q]) * wf_fd
            first_measured_bin = int(np.ceil(self.support / nperseg))
            too_noisy = np.where(np.abs(whitened_clean_flux_fd[:first_measured_bin]) ** 2 / np.mean(np.abs(
                whitened_clean_flux_fd[len(whitened_clean_flux_fd) // 2:]) ** 2) > chi2.isf(1 / (1 * first_measured_bin), df=1))[0]
            if len(too_noisy) > 0 and not always_zero_zeroth_freq:
                len_kill = int(np.ceil(too_noisy[-1] * 1.1))
                lentuk = max(len_kill + 2, first_measured_bin + 2) * 2  # if we need to cover more, we will do that after notch
                # the multiplication by 2 is because the window is symmetric
                wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
            elif always_zero_zeroth_freq:
                lentuk = first_measured_bin * 2
                wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
            # wf_td = ifftfun(wf_fd)  # this wll be done in `self.window_wf_td()`
            self.wf_tds.append(np.array(wf_td, dtype=Dtype_r))
            self.wf_fds.append(np.array(wf_fd, dtype=Dtype_c))
        # We want to somehow make the correlation length smaller but still okay in Fourier:
        self.window_wf_td()
        self.wf_tds = np.array(self.wf_tds, dtype=Dtype_r)  # for numba need numpy.arrays

    def measure_psds_run_12(self, nperseg=None, noverlap=Noverlap_psd, window=Window_psd,
                            subtract_mean=Subtract_mean_in_psd, multi_resolution_psds=Multi_resolution_psds,
                            tukey_steps_len=Tukey_steps_len_psd,
                            always_zero_zeroth_freq=Always_zero_zeroth_freq_psd,
                            feature_sizes=(1, 9, 3.5, 4, 2., 3)):
        """
        PSDS as function of quarter id.
        The PSD measuring algorithm makes the evenly sampled steps and not adjust the search for
        good pieces according to the placement of the zeros in mask.
        Creates self.psds, self.psdfreqs, self.wf_tds, self.wf_fds, self.freqs, seld.npsds.

        :param nperseg: the length of a sample used in Welch method
        :type nperseg: int
        :param noverlap: If int: length of overlap of slices in Welch method.
            If float<1 : relative overlap to nperseg.
            Length of overlap will be int(nperseg*noverlap)
        :type noverlap: int or float<1
        :param window: window function by which the signals are multiplied to avoid boundary effects and crosstalk
            Possible options: 'hann', 'boxcar', where 'boxcar' is just normalized bow, or no window
            Default is 'hann'. It is not recommended to use 'boxcar'.
        :type window: str, 'hann' or 'boxcar'
        :param subtract_mean: If True, the mean of the signal is subtracted. It is recommended to do it because
            otherwise unwanted power may appear in the zero frequency
            Default is True
        :type subtract_mean: bool
        :param multi_resolution_psds: whether to apply multi-resolution smoothing to mitigate the noise effects
        :type multi_resolution_psds: bool
        :param tukey_steps_len: length taken for zeroing in the tukey windowing
        :type tukey_steps_len: int or float
        :param always_zero_zeroth_freq: whether to always put the lowest frequencies to zero
        :type always_zero_zeroth_freq: bool
        :param feature_sizes: length and depth of the deviations that will trigger increasing nperseg, see in the code
        :type feature_sizes:
        """
        d1, w1, d2, w2, d3, w3 = feature_sizes
        if nperseg is None:
            nperseg = self.nperseg_psd

        if not self.is_prepared_for_psd:
            warnings.warn("Data is not prepared for PSD. Use prepare_for_psd method.")

        if self.is_fluxes_inpainted:
            warnings.warn("Fluxes have been inpainted, new PSD will make inpainting invalid for detection.")

        fftfun = rfft
        ifftfun = irfft
        freqfun = np.fft.rfftfreq
        dtype_res = np.float64

        if subtract_mean:
            clean_fluxes = [self.clean_fluxes[i] - np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]]) for i in
                            range(len(self.clean_fluxes))]
        else:
            clean_fluxes = self.clean_fluxes

        self.orig_psds, self.psdfreqs = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks, nperseg=nperseg,
                                                      noverlap=noverlap, window=window, return_freqs=True, dt=self.dt_days,
                                                      return_npsds=False)

        self.freqs = [freqfun(len(time), d=self.dt_days) for time in self.times]

        #############################################################################################################
        # Smoothing
        #############################################################################################################
        if multi_resolution_psds:
            nperseg_two = 2 * nperseg
            nperseg_coarse = nperseg // 2
            nperseg_rude = nperseg // 4
            nperseg_smooth = nperseg // 8
            psds_two = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks, nperseg=nperseg_two, noverlap=noverlap,
                                     window=window, return_freqs=False, dt=self.dt_days, return_npsds=False)
            psds_coarse = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks, nperseg=nperseg_coarse, noverlap=noverlap,
                                        window=window, return_freqs=False, dt=self.dt_days, return_npsds=False)
            psds_smooth = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks, nperseg=nperseg_smooth, noverlap=noverlap,
                                        window=window, return_freqs=False, dt=self.dt_days, return_npsds=False)
            psds_rude = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks, nperseg=nperseg_rude, noverlap=noverlap,
                                      window=window, return_freqs=False, dt=self.dt_days, return_npsds=False)

            self.psdfreqs = [freqfun(nperseg_two, d=self.dt_days) for _ in range(len(self.orig_psds))]
            self.psds = []
            self.wf_fds_meas = []

            ########################################################################################################################
            # Multi-resolution part
            ########################################################################################################################
            # Lowest freqs cannot be resolved by small nperseg by definition. And also, there will be leakage.
            # So we just put there better resolution straight away
            for q in range(len(self.orig_psds)):
                noisy_vec = 1 / psds_two[q] ** 0.5
                orig_vec = ut.sinc_interp(1 / self.orig_psds[q] ** 0.5, nperseg_two, tukey_steps_len=tukey_steps_len)
                coarse_vec = ut.sinc_interp(1 / psds_coarse[q] ** 0.5, nperseg_two, tukey_steps_len=tukey_steps_len)
                rude_vec = ut.sinc_interp(1 / psds_rude[q] ** 0.5, nperseg_two, tukey_steps_len=tukey_steps_len)
                mean_vec = ut.sinc_interp(1 / psds_smooth[q] ** 0.5, nperseg_two, tukey_steps_len=tukey_steps_len)

                noisy_resid_vec = noisy_vec - mean_vec
                std_noisy = ut.outlier_free_std(noisy_resid_vec[len(noisy_resid_vec) // 2:], nsig=3, n_iterations=2)

                combined_vec = np.array(mean_vec)
                vecs = [rude_vec, coarse_vec, orig_vec]
                npersegs_lim = [nperseg_smooth, nperseg_rude, nperseg_coarse]
                fmb_int = int(np.ceil(nperseg_two / nperseg))
                for i, vec_f in enumerate(vecs):
                    nps_rat = 2 * int(nperseg / npersegs_lim[i])
                    combined_vec = ut.replace_beginning(combined_vec, vec_f, fmb_int * nps_rat)
                    combined_vec = ut.replace_if_deviating(combined_vec, vec_f, tester=None, mindev=(std_noisy * 4 / nps_rat),
                                                           minwid=nps_rat)
                orig_resid_vec = noisy_vec - combined_vec
                nsig_sign_minwids_repl_vecs_i_min = [[d3, -1., w3, orig_vec, 0],
                                                     [d1, -1., w1, coarse_vec, 16],  # for mismatch behavior
                                                     [d1, +1., w1, coarse_vec, 16],
                                                     [d2, -1., w2, orig_vec, 0],
                                                     [d2, +1., w2, orig_vec, 0]]  # ,  # for real features
                res_vec = noisy_resid_vec
                res_repl = orig_resid_vec
                for nsig, sign, min_wid, repl_vec, i_min in nsig_sign_minwids_repl_vecs_i_min:
                    i = i_min
                    while i < len(combined_vec):
                        if sign * res_vec[i] - nsig * std_noisy > 0.:
                            i_1 = i
                            i_2 = i
                            while (i_2 < len(res_vec)) and (sign * res_vec[i_2] - nsig * std_noisy > 0):
                                i_2 += 1
                            if (i_2 - i_1) >= min_wid or (i_1 <= 2 and (i_2 - i_1) >= min_wid / 2):
                                # find the beginning and the end of the replacement region:
                                j_1 = i_1
                                while (j_1 - 1 >= 0) and (sign * res_repl[j_1 - 1] > 0):
                                    j_1 -= 1
                                j_2 = i_2
                                while (j_2 < len(res_vec)) and (sign * res_repl[j_2] > 0):
                                    j_2 += 1  # this is the last slice index, so it can be bigger
                                combined_vec[j_1:j_2] = repl_vec[j_1:j_2]
                                i = j_2 + 1
                            else:
                                i = i_2 + 1
                        else:
                            i += 1

                # Making the lowest freqs wf_fd non-decreasing

                # We do not want the zeroth freq to be an outlier.
                # We assume that the PSD should not drop towards the zero bin. If it does in reality, it is anyway
                # not well-measured and is prone to under-whitening. So we make it non-decreasing by punishing more
                # the lowest frequencies
                for j in [1, 2, 3]:  # Fix at these indices
                    i = 0
                    while combined_vec[i + j] < combined_vec[i]:
                        combined_vec[:i + j] = combined_vec[i + j]
                        i += 1
                        if i >= len(combined_vec) - j - 2:
                            break

                self.psds.append(1 / combined_vec ** 2)
                self.wf_fds_meas.append(combined_vec)
        else:
            self.psds = copy.deepcopy(self.orig_psds)
            self.wf_fds_meas = [1 / self.psds[q] ** 0.5 for q in range(len(self.psds))]

        #############################################################################################################
        # Whitening filter
        #############################################################################################################
        # compare value of the wf_fd at low and high freqs. If the low one drops too fast, we will apply the Tukey
        # window and zero the wf_fd around zero. This will help to avoid unresolved residual low-freq red noise.
        assert self.support % 2 == 0, 'self.support assumed to be even!'
        self.wf_tds = []
        self.wf_fds = []
        self.wf_fds_unwindowed = []
        for q, wpsd in enumerate(self.wf_fds_meas):
            wf_td = ifftfun(wpsd)
            meas_len = 2 * nperseg  # len(wf_td)  # because we interpolated like that
            wf_td = np.fft.ifftshift(np.fft.fftshift(wf_td) * ut.tukey(meas_len, alpha=tukey_steps_len / meas_len, sym=True))
            mid_point = len(wf_td) // 2
            result = np.zeros(self.support, dtype=dtype_res)
            result[:mid_point] = wf_td[:mid_point]
            result[-mid_point:] = wf_td[-mid_point:]

            wf_fd = fftfun(result)
            wf_fd[0] = 0.  # MAYBE THIS IS WRONG! also, the flux was already supposed to have zero mean. see below what I did.
            wf_fd = np.abs(wf_fd)  # anyway in the formula it appears twice, so the phase will get canceled
            # but maybe this exactly orig_means that we should not take this abs?
            self.wf_fds_unwindowed.append(np.array(wf_fd))  # it is unwindowed in the large-window sense of this world

            whitened_clean_flux_fd = fftfun(self.clean_fluxes[q]) * wf_fd
            first_measured_bin = int(np.ceil(self.support / nperseg))
            too_noisy = np.where(np.abs(whitened_clean_flux_fd[:first_measured_bin]) ** 2 / np.mean(np.abs(
                whitened_clean_flux_fd[len(whitened_clean_flux_fd) // 2:]) ** 2) > chi2.isf(1 / (1 * first_measured_bin), df=1))[0]
            if len(too_noisy) > 0 and not always_zero_zeroth_freq:
                len_kill = int(np.ceil(too_noisy[-1] * 1.1))
                lentuk = max(len_kill + 2, first_measured_bin + 2) * 2  # if we need to cover more, we will do that after notch
                # the multiplication by 2 is because the window is symmetric
                wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
            elif always_zero_zeroth_freq:
                lentuk = first_measured_bin * 2
                wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
            # wf_td = ifftfun(wf_fd)  # this wll be done in `self.window_wf_td()`
            self.wf_tds.append(np.array(wf_td, dtype=Dtype_r))
            self.wf_fds.append(np.array(wf_fd, dtype=Dtype_c))
        # We want to somehow make the correlation length smaller but still okay in Fourier:
        self.window_wf_td()
        self.wf_tds = np.array(self.wf_tds, dtype=Dtype_r)  # for numba need numpy.arrays

    def measure_psds_run_7(self, nperseg=Nperseg_psd, noverlap=Noverlap_psd, window=Window_psd,
                           subtract_mean=Subtract_mean_in_psd, use_rfft=Use_rfft, multi_resolution_psds=Multi_resolution_psds):
        """
        PSDS as function of quarter id.
        The PSD measuring algorithm makes the evenly sampled steps and not adjust the search for
        good pieces according to the placement of the zeros in mask. That is why
        the number of npsds is not optimal.
        :param nperseg: the length of a sample used in Welch method
        :type nperseg: int
        :param noverlap: If int: length of overlap of slices in Welch method.
            If float<1 : relative overlap to nperseg.
            Length of overlap will be int(nperseg*noverlap)
        :type noverlap: int or float<1
        :param window: window function by which the signals are multiplied to avoid boundary effects and crosstalk
            Possible options: 'hann', 'boxcar', where 'boxcar' is just normalized bow, or no window
            Default is 'hann'. It is not recommended to use 'boxcar'.
        :type window: str
        :param subtract_mean: If True, the mean of the signal is subtracted. It is recommended to do it because
            otherwise unwanted power may appear in the zero frequency
            Default is True
        :type subtract_mean: bool
        :param use_rfft: If True, the signal is treated as real, and all ffts are rffts. The length of the frequency samples
            are the nperseg//2+1.
            Default is True
        :type use_rfft: bool
        :param multi_resolution_psds:
        :type multi_resolution_psds:
        :return: creates self.psds, self.psdfreqs, self.wf_td, self.wf_fd, self.freqs, self.npsds, self.nperseg.
        :rtype:
        """
        from scipy.optimize import minimize
        from scipy.stats import median_abs_deviation as mad

        if not self.is_prepared_for_psd:
            warnings.warn("Data is not prepared for PSD. Use prepare_for_psd method.")

        if self.is_fluxes_inpainted:
            warnings.warn("Fluxes have been inpainted, new PSD will make inpainting invalid for detection.")

        if noverlap < 1:
            noverlap = int(nperseg * noverlap)
        if noverlap == -1:
            noverlap = nperseg // 2

        windowdict = {'boxcar': np.ones(nperseg) / np.linalg.norm(np.ones(nperseg)),
                      'hann': 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg)) / np.linalg.norm(
                          0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg)))}

        psds = []

        if subtract_mean:
            if use_rfft:
                for i in self.quarters:
                    clean_flux = self.clean_fluxes[i] - np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]])
                    psds.append([np.abs(rfft(windowdict[window] * clean_flux[ind1:ind1 + nperseg])
                                        ) ** 2 for ind1 in
                                 range(0, len(self.clean_fluxes[i]) - nperseg, nperseg - noverlap
                                       ) if np.sum((~self.psd_masks[i])[ind1:ind1 + nperseg]) == 0])
                self.freqs = [np.fft.rfftfreq(len(time), d=self.dt_days) for time in self.times]
                self.psdfreqs = [np.fft.rfftfreq(nperseg, d=self.dt_days) for _ in self.quarters]
            else:
                for i in self.quarters:
                    clean_flux = self.clean_fluxes[i] - np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]])
                    psds.append([np.abs(np.fft.fft(windowdict[window] * clean_flux[ind1:ind1 + nperseg])
                                        ) ** 2 for ind1 in
                                 range(0, len(self.clean_fluxes[i]) - nperseg, nperseg - noverlap
                                       ) if np.sum((~self.psd_masks[i])[ind1:ind1 + nperseg]) == 0])
                self.freqs = [np.fft.fftfreq(len(time), d=self.dt_days) for time in self.times]
                self.psdfreqs = [np.fft.fftfreq(nperseg, d=self.dt_days) for _ in self.quarters]

        else:
            if use_rfft:
                for i in self.quarters:
                    psds.append([np.abs(rfft(windowdict[window] * (self.clean_fluxes[i])[ind1:ind1 + nperseg])
                                        ) ** 2 for ind1 in
                                 range(0, len(self.clean_fluxes[i]) - nperseg, nperseg - noverlap
                                       ) if np.sum((~self.psd_masks[i])[ind1:ind1 + nperseg]) == 0])
                self.freqs = [np.fft.rfftfreq(len(time), d=self.dt_days) for time in self.times]
                self.psdfreqs = [np.fft.rfftfreq(nperseg, d=self.dt_days) for _ in self.quarters]
            else:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.fft(windowdict[window] * (self.clean_fluxes[i])[ind1:ind1 + nperseg])
                                        ) ** 2 for ind1 in
                                 range(0, len(self.clean_fluxes[i]) - nperseg, nperseg - noverlap
                                       ) if np.sum((~self.psd_masks[i])[ind1:ind1 + nperseg]) == 0])
                self.freqs = [np.fft.fftfreq(len(time), d=self.dt_days) for time in self.times]
                self.psdfreqs = [np.fft.fftfreq(nperseg, d=self.dt_days) for _ in self.quarters]

        self.npsds = np.array([len(p) for p in psds])  # number of psds used to take the average
        self.psds = [np.mean(p, axis=0) for p in psds]
        self.orig_psds = copy.deepcopy(self.psds)

        if multi_resolution_psds:
            for q in self.quarters:
                fs = np.arange(len(self.psds[q]))
                sel_ind = np.arange(len(self.psds[q]))[-len(self.psds[q]) // 3 * 2:]

                psd_high = np.array(self.orig_psds[q][sel_ind])
                fs_high = np.array(fs[sel_ind])

                not_outl = (psd_high - np.nanmedian(psd_high)) < 2 * mad(psd_high, scale='normal')

                fs_high = fs_high[not_outl]
                psd_high = psd_high[not_outl]
                p1 = np.median(np.sort(psd_high)[:3])

                # constructing fit in such a way that we avoid negative psds and increasing psds
                sol = minimize(
                    lambda x: np.sum((psd_high - (-np.abs(x[1]) * (fs_high - fs_high[-1]) + (np.abs(x[0]) + p1))) ** 2),
                    np.array([0., 0.]))
                trend = (-np.abs(sol.x[1]) * (fs - fs_high[-1]) + (np.abs(sol.x[0]) + p1))

                residuals = self.orig_psds[q] - trend
                mad_residuals = mad(residuals[sel_ind], scale='normal')
                new_psd = []
                if residuals[0] < mad_residuals * 3.:
                    new_psd.append(0.)
                else:
                    new_psd.append(residuals[0])
                for i in range(1, len(residuals) - 1):
                    if residuals[i] >= mad_residuals * 3.:
                        new_psd.append(residuals[i])
                    elif residuals[i] >= 2 * mad_residuals:
                        if (residuals[i - 1] >= mad_residuals * 2.) or (residuals[i + 1] >= mad_residuals * 2.):
                            new_psd.append(residuals[i])
                        else:
                            new_psd.append(0)
                    else:
                        new_psd.append(0)
                if residuals[-1] < mad_residuals * 3.:
                    new_psd.append(0.)
                else:
                    new_psd.append(residuals[-1])
                psd_cut = np.array(new_psd) + trend
                self.psds[q] = np.array(psd_cut)

        self.nperseg_psd = nperseg

        self.wf_tds = []
        self.wf_fds = []
        for i in self.quarters:
            if self.support is not None:
                wf_td, wf_fd = ut.psd2wf(self.psds[i], self.support, use_rfft=use_rfft, always_zero_zeroth_freq=True)
            else:
                wf_td, wf_fd = ut.psd2wf(self.psds[i], len(self.fluxes[i]), use_rfft=use_rfft, always_zero_zeroth_freq=True)
            self.wf_tds.append(wf_td)
            self.wf_fds.append(wf_fd)
        self.wf_tds = np.array(self.wf_tds, dtype=Dtype_r)  # for numba need numpy.arrays

    def window_wf_td(self, extension_coef=Wf_td_window_extension_coef):
        """
        Applies the tukey window to the whitening template in the time domain to make it smooth but preserve the mean zero
        because this will ensure that the lowest frequency gets zero in the Fourier domain whitening filter.
        :param extension_coef:
        :type extension_coef:
        :return:
        :rtype:
        """
        # We want to somehow make the correlation length smaller but still okay in Fourier:
        wf_fds_new = []
        wf_tds_new = []
        meas_len = 2 * self.nperseg_psd  # len(wf_td) measured  # because we interpolated like that
        for q in self.quarters:
            wf_td, wf_fd = ut.window_wf_td_fun(self.wf_fds[q], extension_coef=extension_coef, meas_len=meas_len)
            wf_tds_new.append(np.array(wf_td, dtype=Dtype_r))
            wf_fds_new.append(np.array(wf_fd, dtype=Dtype_c))
        self.wf_tds = np.array(wf_tds_new, dtype=Dtype_r)  # np.array([np.fft.fftshift(wf_td) for wf_td in wf_td_new])
        self.wf_fds = wf_fds_new

    def fftshift_wf_fds(self):
        self.wf_tds = np.array([np.fft.fftshift(wf_td) for wf_td in self.wf_tds], dtype=Dtype_r)

    def interpolate_fluxes(self, interpolate_clean_fluxes=True, mit_length_big_holes=4):
        """
        Interpolates all the holes in the holes in the fluxes, including big holes and end holes.
        In the end holes, the fluxes are padded by a constant value equal to the first(last) value of the array.
        Masks are not modified because measuring PSD or detecting with this interpolation makes no sense.
        """
        for q in self.quarters:
            self.fluxes[q] = ut.interpolate_over_mask(np.array(self.fluxes[q]), np.array(self.masks[q]),
                                                      interp_method='linear', ends='continue',
                                                      mitigate_hole_ends=False, endhole_method='median')
            self.fluxes[q] = ut.interpolate_over_mask(np.array(self.fluxes[q]), np.array(self.big_hole_masks[q]),
                                                      interp_method='linear', ends='continue',
                                                      mitigate_hole_ends=True, mit_length=mit_length_big_holes, endhole_method='mean')
            assert not (np.isnan(self.fluxes[q]).any()), f"Found NaNs in fluxes after interpolation in quarter {q}!"
            assert np.all(np.isfinite(self.fluxes[q])), f"Found non-finite values in fluxes after interpolation in quarter {q}!"
            if interpolate_clean_fluxes:
                # self.clean_fluxes[q] = ut.interpolate_over_mask(np.array(self.clean_fluxes[q]),
                #                                                 np.array(self.clean_masks[q]) * self.big_hole_masks[q],
                #                                                 interp_method='linear', ends='continue',
                #                                                 mitigate_hole_ends=False, endhole_method='median')
                self.clean_fluxes[q] = ut.interpolate_over_mask(np.array(self.clean_fluxes[q]),
                                                                np.array(self.clean_masks[q]),
                                                                interp_method='linear', ends='continue',
                                                                mitigate_hole_ends=False, endhole_method='median')
                self.clean_fluxes[q] = ut.interpolate_over_mask(np.array(self.clean_fluxes[q]),
                                                                np.array(self.clean_masks[q]) & self.big_hole_masks[q],
                                                                interp_method='linear', ends='continue',
                                                                mitigate_hole_ends=True, mit_length=mit_length_big_holes,
                                                                endhole_method='median')
                assert not (np.isnan(self.clean_fluxes[q]).any()), f"Found NaNs in clean_fluxes after interpolation in quarter {q}!"
                assert np.all(np.isfinite(self.clean_fluxes[q])
                              ), f"Found non-finite values in clean_fluxes after interpolation in quarter {q}!"
            self.is_holes_interpolated = True

    def subtract_low_trends(self, nperseg=None, save_low_trends=False):
        """
        Subtract trends of low frequencies which cannot be resolved by the PSD measurement
        :param nperseg: Slice length which will be used in the PSD measurement.
        :type nperseg:
        :param save_low_trends: whether to save these low trends as an attribute
        :type save_low_trends: bool
        """
        # WARNING: WE DO NOT APPLY THIS TO THE TEMPLATE. FOR SHORT TEMPLATES IT IS OK, ALTHOUGH IN PRINCIPLE, IT IS NOT CORRECT.
        if nperseg is None:
            nperseg = self.nperseg_psd

        if not self.is_holes_interpolated:
            warnings.warn("Data is not interpolated. Detrending can fail or give NaNs.")
        self.trends_low = []

        if not self.is_zero_padded:
            for q in self.quarters:
                if len(self.orig_fluxes[q]) - 1 <= nperseg * 4:
                    endlen = len(self.orig_fluxes[q]) // 4 - 1
                else:
                    endlen = nperseg
                detrended, trend_low = ut.detrend_piecewise(self.clean_fluxes[q], big_mask=self.big_hole_masks[q],
                                                            mask=self.masks[q],
                                                            nperseg=nperseg, endlen=endlen, return_trend=True)
                self.trends_low.append(trend_low)
                self.fluxes[q] = np.array(self.fluxes[q]) - trend_low

                # For the clean fluxes, subtract less trend because we want the first PSD bin to collect power around it
                detrended, trend_low = ut.detrend_piecewise(self.clean_fluxes[q], big_mask=self.big_hole_masks[q],
                                                            mask=self.masks[q], nperseg=nperseg // 2 * 3,
                                                            endlen=endlen, return_trend=True)
                self.clean_fluxes[q] = np.array(detrended)
        else:
            for q in self.quarters:
                if len(self.orig_fluxes[q]) - 1 <= nperseg * 4:
                    endlen = len(self.orig_fluxes[q]) // 4 - 1
                else:
                    endlen = nperseg
                detrended, trend_low = ut.detrend_piecewise(self.clean_fluxes[q][self.zero_padding_ends[q][0]:len(
                                                                self.clean_fluxes[q]) - self.zero_padding_ends[q][-1]],
                                                            big_mask=self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(
                                                                self.big_hole_masks[q]) - self.zero_padding_ends[q][-1]],
                                                            mask=self.masks[q][self.zero_padding_ends[q][0]:len(
                                                                self.masks[q]) - self.zero_padding_ends[q][-1]],
                                                            nperseg=nperseg, endlen=endlen, return_trend=True)
                self.trends_low.append(np.array(trend_low, dtype=Dtype_r))
                self.fluxes[q] = np.array(np.array(self.fluxes[q]) - np.concatenate(
                    (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1]))), dtype=Dtype_r)

                # For the clean fluxes, subtract less trend because we want the first PSD bin to collect power around it
                detrended, trend_low = ut.detrend_piecewise(self.clean_fluxes[q][self.zero_padding_ends[q][0]:len(
                    self.clean_fluxes[q]) - self.zero_padding_ends[q][-1]],
                                                            big_mask=self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(
                                                                self.big_hole_masks[q]) - self.zero_padding_ends[q][-1]],
                                                            mask=self.masks[q][self.zero_padding_ends[q][0]:len(
                                                                self.masks[q]) - self.zero_padding_ends[q][-1]],
                                                            nperseg=nperseg // 2 * 3, endlen=endlen, return_trend=True)

                self.clean_fluxes[q] = np.array(np.array(self.clean_fluxes[q]) - np.concatenate(
                    (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1]))), dtype=Dtype_r)
        if not save_low_trends:
            self.trends_low = None

    def notch_fluxes(self, nsig=Nsig_notch, irl_notch=Irl_notch, htol_notch=Htol_notch, minval_notch=Minval_notch):
        """
        Takes the whitened fluxes in Fourier domain, looks at the frequencies
        in all the quarters that peak more than `nsig`, designs Notch filter
        to suppress these frequencies and applies this filter to fluxes.
        Modifies self.fluxes, creates self.notch_soss

        :param nsig: minimal outlier amplitude in units of the std of the whitened flux in Fourier domain
        :type nsig: float
        :param irl_notch: Impulse response length, samples. Length in time domain where the response of
            the filter is greater than `htol`.
        :type irl_notch: float
        :param htol_notch: Tolerance parameter for the filter decay. The number considered as zero (or the end)
            of the filter in time domain.
        :type htol_notch: float
        :param minval_notch:
        :type minval_notch:
        :return:
        :rtype:
        """
        assert self.is_fluxes_whitened, "Whitened fluxes are needed for notch!"
        self.unnotched_whitened_clean_fluxes_fd = copy.deepcopy(self.whitened_clean_fluxes_fd)
        sumflux = np.sum(self.whitened_clean_fluxes_fd, axis=0)  # add all the quarters to mark repeated spurious frequencies
        sumabs = np.sum(np.abs(self.whitened_clean_fluxes_fd), axis=0)
        mask1 = np.array((np.abs(sumflux) - np.abs(np.nanmean(sumflux))) < nsig * np.nanstd(sumflux)).astype(float)
        mask2 = np.array((sumabs - np.nanmean(sumabs)) < nsig * 1.5 * np.nanstd(sumabs)).astype(float)
        inv_mask = (1 - mask1 * mask2).astype(bool)
        notch_soss = []  # make a filter for every peak to notch
        for fr in np.where(inv_mask)[0]:
            if not np.isclose(fr, 0):
                notch_soss.append(ut.notch_fun(fr, fs=self.support, irl=irl_notch, htol=htol_notch, min_val=minval_notch))

        for i in self.quarters:  # apply all the filters for all the quarters
            for sos in notch_soss:
                self.fluxes[i] = np.array(sosfiltfilt(sos, np.array(self.fluxes[i]), padtype='constant'), dtype=Dtype_r)
                self.clean_fluxes[i] = np.array(sosfiltfilt(sos, np.array(self.clean_fluxes[i]), padtype='constant'), dtype=Dtype_r)
                # very important to have zeros in the ends, because of padding.
                # padtype 'constant' will pad by the edge value, and we need to ensure it is zero
        self.notch_mask = np.array(~inv_mask)
        self.is_fluxes_whitened = False  # because these whitened fluxes are not valid anymore, we need to remeasure them
        self.notch_soss = notch_soss
        if len(notch_soss) > 0:
            self.was_notched = True
        else:
            self.was_notched = False

    def inpaint_fluxes(self):
        """
        Fills the holes in data marked by self.masks with the optimal filter.
        The filter works in a way that the impact of the inpainted region
        to the statistics will be zero.
        Modifies self.fluxes by filling the holes without respect to the
        current values in the holes.
        Warning: Whitening filters used for the inpainting have to be the same
        as ones used for detection. Do not remeasure PSD after inpainting.
        """
        self.fluxes = np.array(self.fluxes, dtype=Dtype_r)
        self.fluxes = self.inpaint_fluxes_fast(self.fluxes, self.masks, self.wf_tds)
        self.is_fluxes_inpainted = True

    def whiten_fluxes(self, whiten_clean_fluxes=False):
        """
        Whitens all the fluxes with whitening filter
        """
        assert not (np.array([np.isnan(self.fluxes[i]).any() for i in
                              self.quarters])).any(), 'there are holes in fluxes, cannot whiten!'
        self.whitened_fluxes_fd = []
        self.whitened_fluxes_td = []
        for q in self.quarters:
            whitened_flux_fd = rfft(self.fluxes[q]) * self.wf_fds[q]
            self.whitened_fluxes_fd.append(np.array(whitened_flux_fd, dtype=Dtype_c))
            self.whitened_fluxes_td.append(np.array(irfft(whitened_flux_fd), dtype=Dtype_r))

        if whiten_clean_fluxes:
            self.whitened_clean_fluxes_fd = []
            self.whitened_clean_fluxes_td = []
            for q in self.quarters:
                whitened_clean_flux_fd = rfft(self.clean_fluxes[q]) * self.wf_fds[q]
                self.whitened_clean_fluxes_fd.append(np.array(whitened_clean_flux_fd, dtype=Dtype_c))
                self.whitened_clean_fluxes_td.append(np.array(irfft(whitened_clean_flux_fd), dtype=Dtype_r))

        self.is_fluxes_whitened = True

    def initialize_templates(self, templates=None, shifts=None, template_index='all'):
        """
        Creates the template of the appropriate len by zero padding of the
        input template. If template is None, an entry from the template bank
        template_module.template_bank[template_ind] will be used. If
        template_ind=='all', the method will operate on the entire template_bank.

        Creates:
        self.initial_templates : list of numpy.array 1d
            Templates as they were in the template bank
        self.templates : list of numpy.array 1d
            Templates padded to length self.support.
        self.template_len : int
            Size of the initial templates

        :param templates: Vectors to be used as a template (time domain). Should all be
            of the same length. The working part of the template is
            supposed to be in the center (the transit peak is in the center of
            the array). The ends will be padded by zeros to match self.support.
        :type templates: list of numpy.array 1d of float
        :param shifts:
        :type shifts:
        :param template_index: Index of the template in template_module.template_bank() to be used
            if templates==None. If template_ind=='all', all the template bank will be used.
        :type template_index: int, <len(template_module.template_bank())

        :param templates:
        :type templates:
        :param shifts:
        :type shifts:
        :param template_index:
        :type template_index:
        :return:
        :rtype:
        """
        self.tb = tm.TemplateBank(templates=templates, shifts=shifts, template_index=template_index,
                                  dt=self.dt_days, support=self.support, load_essentials=True)
        # the templates will be already with shifts and padded
        self.indices_of_templates = self.tb.indices_of_templates
        self.templates_of_indices = self.tb.templates_of_indices

    def notch_templates(self):
        """
        Applies notch filter to the template. That is, suppresses some frequencies
        which were marked before in the data conditioning part. Is relevant for
        detection only if used for the data as well and if the psds were not
        remeasured after notching.
        Method produces Notches templates in time domain. After this, need to whiten the templates

        Requirements:
        self.notch_soss : the data should be notched
        self.templates : the templates should be initialized
        """
        if self.notch_soss is not None:
            for k in range(len(self.tb.templates_shifts)):  # different template indices and different subbin shifts
                for sos in self.notch_soss:
                    self.tb.templates_shifts[k] = sosfiltfilt(sos, np.array(self.tb.templates_shifts[k]), padtype='constant')
            self.tb.make_grouped_shifts_arrays()
        # Very important to have zeros in the ends, because of padding.
        # padtype 'constant' will pad by the edge value, and we need to ensure it is zero
        # That is why notching has to be done when the template is centered at the center,
        # and only then the template can be rolled.

    def get_not_interleaved_index(self, t=None, tsh=None):
        """
        Returns the index of the template `t` or the template with shift `tsh` in the array of
        non-interleaved templates. See more in the note about the template indexing.
        :param t: template index
        :type t: int
        :param tsh: shift-template index
        :type tsh: int
        :return: index of the template in the array of non-interleaved templates
        :rtype: int
        """
        if t is not None:
            res = t - self.not_interleaved_numbers[0]
        elif tsh is not None:
            res = self.templates_of_indices[tsh] - self.not_interleaved_numbers[0]
        else:
            raise ValueError('No template number or shifted template number specified')
        if res < 0:
            raise ValueError('Does not belong to not_interleaved')
        return res

    def whiten_templates(self):
        """
        Whitens all the templates  by applying the whitening filter calculated after the PSD measurement of the data.
        """
        self.whitened_templates_td = np.zeros((len(self.tb.templates_shifts), len(self.quarters), self.support), dtype=Dtype_r)
        self.whitened_templates_fd = np.zeros((len(self.tb.templates_shifts), len(self.quarters), self.fft_support),
                                              dtype=Dtype_c)
        for k in range(len(self.tb.templates_shifts)):  # different template indices
            for q in self.quarters:
                whitened_temp_fd = rfft(self.tb.templates_shifts[k]) * self.wf_fds[q]
                self.whitened_templates_fd[k][q] = whitened_temp_fd
                self.whitened_templates_td[k][q] = irfft(whitened_temp_fd)
        self.is_template_whitened = True

    def matched_filtering(self, detr_length_outliers=Detr_length_outliers,
                          nsig_outliers_stat=Nsig_outliers_stat,
                          nwind_moving_var_min=Nwind_moving_var_min,
                          ngap_moving_var_min=Ngap_moving_var_min,
                          nsamples_min_moving_var_coef=Nsamples_min_moving_var_coef):
        """
        Calculates the matched-filtering statistics of data and templates.
        Cuts the statistic from zero padding ends.
        Measures moving variance of the statistics and implements the PSD error (drift) correction.
        Makes sure the distribution of the statistics in units of SNR is standard normal.

        Requires:
        `self.whitened_fluxes_fd` : fluxes should be conditioned and whitened
        `self.whitened_templates_fd` : templates should be conditioned in the same way and whitened
        `self.masks` : need masks for std calculation, but will ignore small holes
        `self.support` : everything should be zero-padded

        Creates the following statistics:
        `self.statistics_raw` : h@C**(-1)@d. Just the matched-filtering score without denominator
        `self.stat_moving_vars` : Moving variance of statistics_raw
        `self.statistics_snr` : h@C**(-1)@d/(h@C**(-1)@h)**0.5 Statistics in units of SNR.

        :param detr_length_outliers: Detrending length used to detrend the data before calculating the outlier-free variance.
            Not really useful for the statistic.
        :type detr_length_outliers: int
        :param nsig_outliers_stat: Deviation in units of stansard deviation to be considered as an outlier in the
            outlier-free variance calculation for the statistic.
        :type nsig_outliers_stat: float
        :param nwind_moving_var_min: Minimal size of the moving window. The window size will be calculated according to the
            template width, but this is the minimal value which is required.
        :type nwind_moving_var_min: int
        :param ngap_moving_var_min: Minimal size of the gap inside the moving window to exclude the point around which
            the moving variance is taken. The gap will be related to the template width, but this is the minimal
            value which is required. Points are excluded around each bin when calculating the variance in this bin in
            order to avoid influence of these points on the variance estimation.
        :type ngap_moving_var_min: int
        :param nsamples_min_moving_var_coef: Part of the samples that were supposed to be taken which is minimal
            required for calculating moving variance. For example, 0.5 means that at least half of the requested
            samples should be not nan, not masked and not beyond the edge.
        :type nsamples_min_moving_var_coef: float

        """
        n_wh_temp = len(self.whitened_templates_td)
        self.statistics_raw = [[np.zeros(len(self.fluxes[q]), dtype=Dtype_r) for q in self.quarters]
                               for _ in range(n_wh_temp)]
        # -> corrected for moving std and then multiplied back by non-moving std
        self.stat_moving_vars = [[np.zeros(len(self.fluxes[q]), dtype=Dtype_r) for q in self.quarters]
                                 for _ in range(n_wh_temp)]
        self.statistics_snr = [[np.zeros(len(self.fluxes[q]), dtype=Dtype_r) for q in self.quarters]
                               for _ in range(n_wh_temp)]
        self.stat_tot_vars = [[Dtype_r(0.) for _ in self.quarters] for __ in range(n_wh_temp)]  # mean of moving variance
        self.stat_masks = [[np.ones(len(self.fluxes[q])).astype(bool) for q in self.quarters] for _ in
                           range(n_wh_temp)]

        assert self.is_template_whitened, 'Templates have to be whitened for matched filtering!'

        for k in range(n_wh_temp):
            for q in self.quarters:
                # Statistics in the original units:
                stat = np.real(irfft(np.conjugate(self.whitened_templates_fd[k][q]) * self.whitened_fluxes_fd[q]))
                stat = np.array(stat[self.zero_padding_ends[q][0]:len(stat) - self.zero_padding_ends[q][1]])
                # The statistic was cut historically because of memory, now there is no reason other than already written code
                var_calc = irfft(np.abs(self.whitened_templates_fd[k][q]) ** 2)[0]
                # Zero-padding ends are not a problem because they are zeros, and the normalization is fixed by irfft
                statmask = np.array(self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(
                    self.big_hole_masks[q]) - self.zero_padding_ends[q][1]], dtype=bool)
                eff_temp_wid = len(np.where(self.tb.templates_shifts[k] >= np.max(self.tb.templates_shifts[k]) * 0.1)[0])
                gap_len = 2 * eff_temp_wid  # It was set empirically for typical data and templates
                gap_len = max(ngap_moving_var_min, gap_len)
                wind_len = 30 * eff_temp_wid  # It was measured empirically for typical data and templates
                wind_len = max(wind_len, nwind_moving_var_min)
                nsamples_min = int(nsamples_min_moving_var_coef * (wind_len - gap_len))
                movvar_meas, tot_var = ut.moving_var(np.array(stat),
                                                     mask=statmask,
                                                     nwind=wind_len, ngap=gap_len,
                                                     nsamples_min=nsamples_min,
                                                     remove_outliers=True, detrend_remove_outliers=False,
                                                     detr_length_outliers=detr_length_outliers,
                                                     nsig_outliers=nsig_outliers_stat,
                                                     return_mask=False, return_total_var=True)
                movvar_meas[~statmask] = np.inf
                stat_snr = stat / movvar_meas ** 0.5

                # RE_NORMALIZING MOVING VARIANCE (PSD DRIFT) CORRECTION
                # (Separate for positive and negative)
                # var_snr = np.var(stat_snr[statmask])  # the definition of well-measured variance is that the snr should be N(0,1)
                # movvar_meas *= var_snr  # we make the variance estimate exact in order for the snr to be N(0,1)

                # Making positive values unit variance:
                # We do not do the outlier-free here assuming that the effect of the transits will be small. This
                # means that if the effect of the transits is not small, they will be impacted.
                stat_snr, movvar_meas = ut.renormalize_variance_fun(movvar_meas=movvar_meas,
                                                                    statmask=statmask,
                                                                    stat_snr=stat_snr,
                                                                    stat=stat)
                correction = movvar_meas / var_calc
                statrawcor = stat / correction
                statrawcor[~statmask] = 0.  # Very important, because inf would destroy periodicity search
                stat_var_cor = np.array(movvar_meas)
                stat_var_cor[statmask] = movvar_meas[statmask] / (correction[statmask]) ** 2
                stat_var_cor[~statmask] = 0.  # Very important, because inf would destroy periodicity search

                self.stat_tot_vars[k][q] = Dtype_r(ut.unb_var(statrawcor[statmask]))
                self.stat_moving_vars[k][q] = stat_var_cor
                self.statistics_snr[k][q] = stat_snr
                self.statistics_raw[k][q] = statrawcor
                self.stat_masks[k][q] = statmask

    def cut_zero_padding_ends(self):
        """
        Removes the ends that were padded by zeros and saves the data in the same field.
        Modifies:
        self.fluxes
        self.clean_fluxes
        self.times
        self.masks
        self.clean_masks
        self.big_hole_masks
        self.whitened_fluxes_td
        self.whitened_templates_td
        """
        if self.is_zero_padded:
            fluxes = []
            clean_fluxes = []
            times = []
            masks = []
            clean_masks = []
            big_hole_masks = []
            whitened_fluxes_td = []
            whitened_templates_td = []  # attention: they are rolled
            for q in self.quarters:
                fluxes.append(np.array(self.fluxes[q][
                                       self.zero_padding_ends[q][0]:len(self.fluxes[q]) - self.zero_padding_ends[q][
                                           -1]]))
                clean_fluxes.append(np.array(self.clean_fluxes[q][
                                             self.zero_padding_ends[q][0]:len(self.clean_fluxes[q]) - self.zero_padding_ends[q][-1]]))
                times.append(np.array(self.times[q][
                                      self.zero_padding_ends[q][0]:len(self.times[q]) - self.zero_padding_ends[q][-1]]))
                masks.append(np.array(self.masks[q][
                                      self.zero_padding_ends[q][0]:len(self.masks[q]) - self.zero_padding_ends[q][-1]]))
                clean_masks.append(np.array(self.clean_masks[q][
                                            self.zero_padding_ends[q][0]:len(self.clean_masks[q]) - self.zero_padding_ends[q][-1]]))
                big_hole_masks.append(np.array(self.big_hole_masks[q][
                                               self.zero_padding_ends[q][0]:len(self.big_hole_masks[q]) - self.zero_padding_ends[q][-1]]))
                whitened_fluxes_td.append(np.array(
                    self.whitened_fluxes_td[q][self.zero_padding_ends[q][0]:len(
                        self.whitened_fluxes_td[q]) - self.zero_padding_ends[q][-1]]))
            if self.whitened_templates_td is not None:
                for k in range(len(self.whitened_templates_td)):
                    whitened_templates_td_template = []
                    for q in self.quarters:
                        wt_td = np.roll(self.whitened_templates_td[k][q],
                                        ((len(self.whitened_templates_td[k][q]) - 1) // 2))
                        wt_td = np.array(
                            wt_td[self.zero_padding_ends[q][0]:len(wt_td) - self.zero_padding_ends[q][-1]])
                        whitened_templates_td_template.append(np.roll(wt_td, -((len(wt_td) - 1) // 2)))
                    whitened_templates_td.append(whitened_templates_td_template)
                self.whitened_templates_td = whitened_templates_td
            self.fluxes = fluxes
            self.clean_fluxes = clean_fluxes
            self.times = times
            self.masks = masks
            self.clean_masks = clean_masks
            self.big_hole_masks = big_hole_masks
            self.whitened_fluxes_td = whitened_fluxes_td
        self.is_zero_padded = False

    def time_slides(self, std_time_scrambling=Std_time_scrambling, random_state=None, random_seed=None):
        """
        ATTENTION: This function is not used for now, we use the scrambling.
        Adds random time shift to every quarter. The shift of the first quarter is zero. Shifts cannot be
        negative. The total mission duration is not preserved.
        To undo, use `self.undo_time_slides()`.
        :param std_time_scrambling: Standard deviation of the random shifts (normal distribution)
        :type std_time_scrambling: float
        :param random_state: random state of numpy.random.set_state to fix the random slides generation.
            Overrides `random_seed`
        :type random_state: tuple[str, numpy.ndarray] corresponding to numpy.random.get_state()
        :param random_seed: Sets randon seed of numpy.random. Overriden by `random_state`
        :type random_seed: int
        """
        if random_state is not None:
            np.random.set_state(random_state)
        elif random_seed is not None:
            np.random.seed(random_seed)
        else:
            np.random.seed(None)
        r = np.abs(np.random.normal(0, std_time_scrambling, len(self.times)))
        r = np.cumsum(r)
        if self.time_slide_shifts is not None:
            for i in range(len(self.times)):
                self.times[i] = np.array(self.times[i]) - self.time_slide_shifts[i] + r[i]  # we undo old time slides and do new
        else:
            for i in range(len(self.times)):
                self.times[i] = np.array(self.times[i]) + r[i]  # we do not undo old, there is no old
        self.time_slide_shifts = r

        self.tmin = np.nanmin(self.times[0])
        self.tmax = np.nanmax(self.times[0])
        for t in self.times[1:]:
            self.tmin = min(np.nanmin(t), self.tmin)
            self.tmax = max(np.nanmax(t), self.tmax)

    def undo_time_slides(self):
        """
        Attention: This function is not used for now, we use the scrambling.
        Undoes the random slides done by `self.time_slides()`.
        """
        if self.time_slide_shifts is not None:
            for i in range(len(self.times)):
                self.times[i] = np.array(self.times[i]) - self.time_slide_shifts[i]
            self.time_slide_shifts = None  # np.zeros(len(self.times))

            self.tmin = np.nanmin(self.times[0])
            self.tmax = np.nanmax(self.times[0])
            for t in self.times[1:]:
                self.tmin = min(np.nanmin(t), self.tmin)
                self.tmax = max(np.nanmax(t), self.tmax)

    def invert_statistic_after_interleaving(self, choose_random=False):
        """
        Inverts `self.statistics_interleaved` and `self.statistics_not_interleaved`, meaning multiplies by -1.
        If `choose_random` is `True`, randomly multiplies entries by -1 and 1.
        :param choose_random: Whether to multiply all by -1 or randomly multiply entries by -1 and 1.
        :type choose_random: bool
        """
        if not choose_random:
            for i in range(len(self.statistics_interleaved)):
                self.statistics_interleaved[i] = - self.statistics_interleaved[i]
            for i in range(len(self.statistics_not_interleaved)):
                self.statistics_not_interleaved[i] = - self.statistics_not_interleaved[i]
        else:
            np.random.seed(None)
            for i in range(len(self.statistics_interleaved)):
                self.statistics_interleaved[i] = np.random.choice([1., -1.]) * self.statistics_interleaved[i]
            for i in range(len(self.statistics_not_interleaved)):
                self.statistics_not_interleaved[i] = np.random.choice([1., -1.]) * self.statistics_not_interleaved[i]

    #%% Vetting SES
    # noinspection PyIncorrectDocstring
    def veto_statistics(self, apply_only_to_neg=True,
                        margin_snr_veto=Margin_snr_veto,
                        fn=Fn_fit_veto,
                        n_parts_veto_snr=N_parts_veto_snr,
                        min_bins_veto_part_time=Min_bins_veto_part_time,
                        lobe_hlen_veto_part_time=Lobe_hlen_veto_part_time,
                        n_parts_veto_fourier=N_parts_veto_fourier,
                        wind_len_overlap_fourier=Wind_len_overlap_fourier,
                        fraction_template_duration_edge=1., days_edge=None, days_hole_max=0.75,
                        fraction_template_duration_hole=1., kill_edges=False, kill_around_hole=False,
                        fraction_template_duration_minimal_neighbors=1.5,
                        delta_snr_kill_sf_minimal_neighbors=5e-2,
                        veto_extreme_statistics=True, max_abs_stat_veto=Max_abs_stat_veto,
                        threshold_fn_in_quarter_step=1.,
                        veto_positive_outliers=True, veto_tail_statistics=True,
                        use_dist_to_hole_masks_for_threshold=True, fn_close_to_hole=5e-2,
                        use_snr_dependent_threshold=True,
                        update_stat_and_veto_masks=True, zero_vetted_stats=True, renormalize_stats_variances=True,
                        save_masks_as_attributes=False):
        """
        Verify big triggers and mask ones which are non-transit-like.
        First, finds the triggers which are big enough (>`margin_snr`) to be vetted. Usually, only negative triggers are selected.
        Applies to them a series of categorical checks:
        - Whether the tigger is close to the edge of the data or a big hole
        - Whether there is a significant step in the data close to the trigger
        - Whether this trigger is too large to be of interest for the search or it originates as a tail of such trigger
        - Whether the trigger is a tail of another more significant trigger and should therefore not be considered by itself
        Then, it applies a chi-squared `veto_parts` test comparing the amplitudes of different parts of a template in time
        and Fourier domains.
        Then, applies thresholds to the worst p-value for every trigger. There is an option to apply thresholds depending
        on the distance to the nearest hole, or depending on the SNR of the trigger.
        Then, it applies the veto masks to the statistics and optionally sets the statistics in the masked places to zero.
        Finally, it renormalizes the variances of the statistics taking now into account that some values are masked,
        meaning that they are bad and probably do not belong to this distribution.
        Note: Vetting should be done before unpadding!

        Most of the parameters descriptions are to be found in the corresponding methods.
        :param update_stat_and_veto_masks: Whether to update the statistics and veto masks with zeros after vetting.
        :type update_stat_and_veto_masks: bool
        :param zero_vetted_stats: Whether to put the statistics in the vetted places to zero. Required for the periodicity search.
        :type zero_vetted_stats: bool
        :param renormalize_stats_variances: Whether to renormalize the statistics variances after vetting.
            Required to ensure the standard normal distribution
        :type renormalize_stats_variances: bool
        :param save_masks_as_attributes: Whether to save the different vetting masks as attributes of the class.
            Not recommended because they are not used anywhere and take memory. Only used for debugging and plotting.
        :type save_masks_as_attributes: bool
        """
        self.find_trigger_inds(apply_only_to_neg=apply_only_to_neg, margin_snr=margin_snr_veto)
        self.veto_statistics_by_rules(fraction_template_duration_minimal_neighbors=fraction_template_duration_minimal_neighbors,
                                      delta_snr_kill_sf_minimal_neighbors=delta_snr_kill_sf_minimal_neighbors,
                                      veto_extreme_statistics=veto_extreme_statistics, max_abs_stat_veto=max_abs_stat_veto,
                                      veto_tail_statistics=veto_tail_statistics,
                                      save_masks_as_attributes=save_masks_as_attributes,
                                      threshold_fn_in_quarter_step=threshold_fn_in_quarter_step,
                                      fraction_template_duration_edge=fraction_template_duration_edge, days_edge=days_edge,
                                      days_hole_max=days_hole_max, fraction_template_duration_hole=fraction_template_duration_hole,
                                      kill_edges=kill_edges, kill_around_hole=kill_around_hole,
                                      veto_positive_outliers=veto_positive_outliers, update_masks=update_stat_and_veto_masks)

        self.calculate_veto_chi2_p_vals(n_parts_veto_snr=n_parts_veto_snr, min_bins_veto_part_time=min_bins_veto_part_time,
                                        lobe_hlen_veto_part_time=lobe_hlen_veto_part_time, n_parts_veto_fourier=n_parts_veto_fourier,
                                        wind_len_overlap_fourier=wind_len_overlap_fourier)

        # Updating the masks with rejected low p-values:
        self.apply_thresholds_to_veto_chi2_p_values(fn=fn, use_dist_to_hole_masks_for_threshold=use_dist_to_hole_masks_for_threshold,
                                                    fn_close_to_hole=fn_close_to_hole,
                                                    use_snr_dependent_threshold=use_snr_dependent_threshold,
                                                    save_masks_as_attributes=save_masks_as_attributes,
                                                    update_stat_and_veto_masks=update_stat_and_veto_masks)

        if zero_vetted_stats:
            # Setting statistics in the masked places to zero:
            for k in range(len(self.statistics_snr)):
                for q in self.quarters:
                    self.statistics_snr[k][q][~self.stat_masks[k][q]] = 0.
                    self.statistics_raw[k][q][~self.stat_masks[k][q]] = 0.
                    self.stat_moving_vars[k][q][~self.stat_masks[k][q]] = 0.  # This is very important for the periodicity search

        if renormalize_stats_variances:
            self.renormalize_stats_variance()

    def find_trigger_inds(self, apply_only_to_neg=True, margin_snr=Margin_snr_veto):
        """
        Finds the indices of triggers (SES SNR statistics exceeding `margin_snr`). For every template and quarter.
        :param apply_only_to_neg: Whether to apply the veto only to negative statistics, which transits are
            expected to be, or also to the positive ones. The latter is needed if we want to look at the
            inverted data. It also can affetct the negative values processing if there are interactions.
        :type apply_only_to_neg: bool
        :param margin_snr: The minimal abs value of the snr from which the vetting starts to be applied
        :type margin_snr: float
        """
        self.trigger_indices_unpadded = [[[] for _ in self.quarters] for __ in range(len(self.statistics_snr))]
        self.trigger_indices_padded = [[[] for _ in self.quarters] for __ in range(len(self.statistics_snr))]

        for k in range(len(self.statistics_snr)):
            for q in self.quarters:  # over quarters
                # Finding threshold-crossing trigger indices:
                trigger_indices_quarter_unpadded = (np.abs(self.statistics_snr[k][q]) >= np.abs(margin_snr)) & self.stat_masks[k][q]
                if apply_only_to_neg:
                    trigger_indices_quarter_unpadded = (self.statistics_snr[k][q] <= 0) & trigger_indices_quarter_unpadded
                self.trigger_indices_unpadded[k][q] = np.where(trigger_indices_quarter_unpadded)[0]

        for k in range(len(self.statistics_snr)):
            for q in self.quarters:
                zpe_0 = self.zero_padding_ends[q][0]
                self.trigger_indices_padded[k][q] = self.trigger_indices_unpadded[k][q] + zpe_0

    def veto_statistics_by_rules(self, fraction_template_duration_minimal_neighbors=1.5, delta_snr_kill_sf_minimal_neighbors=5e-2,
                                 veto_extreme_statistics=True, max_abs_stat_veto=Max_abs_stat_veto, save_masks_as_attributes=False,
                                 threshold_fn_in_quarter_step=1., fraction_template_duration_edge=1., days_edge=None,
                                 days_hole_max=0.75, fraction_template_duration_hole=1., kill_edges=False, kill_around_hole=False,
                                 veto_positive_outliers=True, veto_tail_statistics=True,
                                 step_width=768, interpolate_for_step_stat=True, interp_smoothing_length_step_stat=10,
                                 update_masks=True):
        assert hasattr(self, 'trigger_indices_unpadded') and hasattr(self, 'trigger_indices_padded'), \
            'You should run `find_trigger_inds()` to identify triggers before vetting them with `veto_statistics_sharp()`!'
        self.veto_masks = [[np.ones(len(self.stat_masks[k][q])).astype(bool) for q in self.quarters] for k in
                           range(len(self.stat_masks))]
        self.veto_dist_to_holes(fraction_template_duration_edge=fraction_template_duration_edge, days_edge=days_edge,
                                days_hole_max=days_hole_max, fraction_template_duration_hole=fraction_template_duration_hole,
                                kill_edges=kill_edges, kill_around_hole=kill_around_hole)
        # analyze_kepler_quality_masks(self, remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True,
        #                              update_masks=update_masks, save_as_attributes=save_masks_as_attributes)
        self.veto_step(save_as_attributes=save_masks_as_attributes, update_masks=update_masks,
                       threshold_fn_in_quarter=threshold_fn_in_quarter_step,
                       step_width=step_width, interpolate_for_step_stat=interpolate_for_step_stat,
                       interp_smoothing_length_step_stat=interp_smoothing_length_step_stat)
        self.analyze_minimal_neighbors(fraction_template_duration=fraction_template_duration_minimal_neighbors,
                                       delta_snr_kill_sf=delta_snr_kill_sf_minimal_neighbors,
                                       veto_extreme_statistics=veto_extreme_statistics,
                                       veto_tail_statistics=veto_tail_statistics,
                                       max_abs_stat=max_abs_stat_veto, veto_positive_outliers=veto_positive_outliers,
                                       update_masks=update_masks, save_as_attributes=save_masks_as_attributes)

    def veto_dist_to_holes(self, fraction_template_duration_edge=1., days_edge=None, days_hole_max=0.75,
                           fraction_template_duration_hole=1., kill_edges=False, kill_around_hole=False):
        """
        Takes care of the fact that data around big holes and the edges of the data may be bad. It is either masked or saved
        to influence then the threshold of the chi-squared `veto_parts`.
        :param fraction_template_duration_edge: Fraction of template duration to define the time to vet near the edge of data.
        :type fraction_template_duration_edge: float
        :param days_edge: Time in days to vet near the edge of data. Overriden by `fraction_template_duration_edge`.
        :type days_edge: float or None
        :param days_hole_max: Maximal time in days to consider suspicious around the big holes.
            If `fraction_template_duration_edge` is None, is used instead. If it is not None but gives a bigger value, overrides
            this value.
        :type days_hole_max: float or None
        :param fraction_template_duration_hole: Fraction of template duration to define the time to vet near the big holes in the data.
        :type fraction_template_duration_hole: float
        :param kill_edges: Whether to mask the edges of the statistic, where data may be bad, and the template is not fully covered by data.
        :type kill_edges: bool
        :param kill_around_hole: Whether to make the veto mask false around big holes regardless of anything
        :type kill_around_hole: bool
        """
        self.distance_to_hole_masks = [[np.ones(len(self.stat_masks[k][q]), dtype=bool) for q in self.quarters] for k in
                                       range(len(self.stat_masks))]

        # ##################################################################################################
        # EXTREME POSITION: we do not trust it because data there may be bad, and veto will be poor there
        # Can be either killed, or used to change the p-value of the chi2 veto
        for k in range(len(self.tb.initial_templates_shifts)):
            for q in self.quarters:  # over quarters
                if fraction_template_duration_edge is not None:
                    templ_width = int(np.ceil(self.tb.durations[self.tb.templates_of_indices[k]] / self.dt_days))

                    wing_remove_bins = int(np.round(templ_width * fraction_template_duration_edge))
                else:
                    wing_remove_bins = int(np.round(days_edge / self.dt_days))
                self.distance_to_hole_masks[k][q][:wing_remove_bins] = False
                self.distance_to_hole_masks[k][q][-wing_remove_bins:] = False

                if kill_edges:
                    self.veto_masks[k][q][:wing_remove_bins] = False
                    self.veto_masks[k][q][-wing_remove_bins:] = False
                    self.stat_masks[k][q][:wing_remove_bins] = False
                    self.stat_masks[k][q][-wing_remove_bins:] = False

        # ##################################################################################################
        # DISTANCE TO BIG HOLES. Holes include edges, but it is possible to kill only edges and save more
        # space around all holes to use it in the vero chi2
        # Can be either killed, or used to change the p-value of the chi2 veto
        if len(self.big_hole_masks[0]) > len(self.stat_masks[0][0]):
            bhm_unpadded = [self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(self.big_hole_masks[q]) - self.zero_padding_ends[q][1]]
                            for q in self.quarters]
        else:
            bhm_unpadded = self.big_hole_masks
        distances_to_holes = ut.get_distances_to_holes(bhm_unpadded)
        for k in range(len(self.tb.initial_templates_shifts)):
            for q in self.quarters:
                if fraction_template_duration_edge is not None:
                    templ_width = int(np.ceil(self.tb.durations[self.tb.templates_of_indices[k]] / self.dt_days))
                    wing_remove_bins = int(np.round(templ_width * fraction_template_duration_hole))
                else:
                    wing_remove_bins = int(np.round(days_hole_max / self.dt_days))
                if days_hole_max is not None:
                    wing_remove_bins = max(wing_remove_bins, int(np.round(days_hole_max / self.dt_days)))

                good_dist_to_hole = (distances_to_holes[q] > wing_remove_bins)
                self.distance_to_hole_masks[k][q] &= good_dist_to_hole  # distance to hole is larger than the wing

                # Instead of killing the edges, we can use the distance to hole to change the p-value of the chi2
                if kill_around_hole:
                    self.veto_masks[k][q] &= good_dist_to_hole
                    self.stat_masks[k][q] &= good_dist_to_hole

    def veto_step(self, save_as_attributes=True, update_masks=True, threshold_fn_in_quarter=0.1,
                  step_width=768, interpolate_for_step_stat=True, interp_smoothing_length_step_stat=10):
        """
        Calculates the detection statistic for the data being explained by a step function. If the step statistic
        is too large, then the data should be masked in the region around the step, and the planet templates triggers
        there should be ignored. We note that there is an offset between where the step occurs and where the planet
        template will peak, so it is important to mask around the step occurrence.
        The step statistic is calculated on the original red flux after linearly detrending. No whitening
        filter is used. It is done because detrending the data and whitening it mitigates the original step
        and makes it hardly detectable. Mathematically, we are testing not the step in the red noise,
        but the step against planet in the red noise. We want to see that the current piece of data is explained
        better by the step than by the red noise hypothesis.
        :param save_as_attributes: Whether to save self.no_step_contamination_masks` as an attribute.
        :type save_as_attributes: bool
        :param update_masks: Whether to update the `self.veto_masks` and `self.stat_masks` with the vetted places.
        :type update_masks: bool
        :param threshold_fn_in_quarter: The p-value threshold for the step statistic
            to be considered significant. Is measured as a number of times such a score can appear in one quarter.
        :type threshold_fn_in_quarter: float
        :param step_width: Width in bins to produce the local step model. Outside of this, the model is zero. Inside,
            the model grows linearly from zero to maximum, then steps down, and then relapses linearly to zero. The step
            itself is immediate. If `step_width` is None, the step model is as long as the entire data, meaning that
            it grows and declines very slowly and steps abruptly. The edges opposite to the step itself connect
            at zero value without stepping.
        :type step_width: int or None
        :param interpolate_for_step_stat: Whether to linearly interpolate the data before calculating the step statistic.
        :type interpolate_for_step_stat: bool
        :param interp_smoothing_length_step_stat: Length of smoothing that can be used during the interpolation of
            the data before calculating the step statistic. It means that instead of linearly interpolating between the
            two values on the edges of the hole, the local means of `interp_smoothing_length_step_stat` points near
            the hole edge are calculated, and the interpolation is done between them.
        :type interp_smoothing_length_step_stat: int
        """
        from scipy.signal import find_peaks
        no_step_contamination_masks = [[np.ones(len(self.stat_masks[k][q])).astype(bool) for q in self.quarters] for k in
                                       range(len(self.stat_masks))]
        for q in self.quarters:
            msk_q = np.array(self.masks[q])
            bh_msk_q = np.array(self.big_hole_masks[q])
            # Checking zero padding:
            if len(msk_q) > len(bh_msk_q):
                msk_q = np.array(msk_q[self.zero_padding_ends[q][0]:len(msk_q) - self.zero_padding_ends[q][1]])
            elif len(msk_q) < len(bh_msk_q):
                bh_msk_q = np.array(bh_msk_q[self.zero_padding_ends[q][0]:len(bh_msk_q) - self.zero_padding_ends[q][1]])
            elif self.is_zero_padded:
                msk_q = np.array(msk_q[self.zero_padding_ends[q][0]:len(msk_q) - self.zero_padding_ends[q][1]])
                bh_msk_q = np.array(bh_msk_q[self.zero_padding_ends[q][0]:len(bh_msk_q) - self.zero_padding_ends[q][1]])
            mask = msk_q & bh_msk_q
            # -> Combining both nans of the original fluxes and the newly discovered things

            # 1) Detecting the step:
            step_stats_2 = ut.get_step_statistic(data=np.array(self.orig_fluxes[q]), mask=mask, mask_fit=None,
                                                 wf_td=None, step_width=step_width, interpolate=interpolate_for_step_stat,
                                                 interp_smoothing_length=interp_smoothing_length_step_stat) ** 2

            # threshold_stat_2 = chi2(loc=0, scale=1, df=1).isf(threshold_fn_in_quarter / (len(step_stats_2)))
            # The following is the same but faster:
            threshold_stat_2 = ut.fast_chi2_01_isf(x=threshold_fn_in_quarter / (len(step_stats_2)), df=1)
            # -> it is actually more conservative than one per data because the noise is red and we normalize it by its own variance
            # 2) Separating peaks in step statistic by finding separated local maxima or idk
            max_temp_width = int(np.ceil(tm.Durations[-1] / self.dt_days))
            step_peak_inds, _ = find_peaks(step_stats_2, height=threshold_stat_2, threshold=None, distance=max_temp_width,
                                           prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)

            # 3) For each peak, kill trigger indices around it that it influenced
            for k in range(len(self.statistics_snr)):
                temp_width = int(np.round(tm.Durations[self.templates_of_indices[k]] / self.dt_days))
                for ind in step_peak_inds:
                    no_step_contamination_masks[k][q][
                        max(0, ind - temp_width):min(len(no_step_contamination_masks[k][q]), ind + temp_width)] = False

        if save_as_attributes:
            self.no_step_contamination_masks = no_step_contamination_masks

        if update_masks:
            for q in self.quarters:
                for k in range(len(self.statistics_snr)):
                    # Updating the veto masks:
                    self.veto_masks[k][q][self.trigger_indices_unpadded[k][q]] &= no_step_contamination_masks[k][q][
                        self.trigger_indices_unpadded[k][q]]
                    self.stat_masks[k][q][self.trigger_indices_unpadded[k][q]] &= no_step_contamination_masks[k][q][
                        self.trigger_indices_unpadded[k][q]]

    def analyze_minimal_neighbors(self, fraction_template_duration=2., delta_snr_kill_sf=2e-2,
                                  veto_extreme_statistics=True, max_abs_stat=Max_abs_stat_veto,
                                  veto_tail_statistics=True,
                                  update_masks=True, save_as_attributes=True, veto_positive_outliers=True):
        """

        :param fraction_template_duration: Fraction of the template duration defining the area of impact of the
            local minimal score. The neighboring points in this area will be masked if the local minimum is below
            `delta_snr_kill_sf_minimal_neighbors` from them.
        :type fraction_template_duration: float
        :param delta_snr_kill_sf: The minimal difference in SNR from the local minimum and the surroundings in
            units of the survival function. If the local minimum is below this value, the surrounding points will be masked
            as originating from the local minimum tail.
        :type delta_snr_kill_sf: float
        :param veto_extreme_statistics: Whether to mask statistics which are too large in absolute value
        :type veto_extreme_statistics: bool
        :param max_abs_stat: Maximal absolute value of statistic in units of SNR that is allowed. Larger
            ones will be masked. This is done because we are not interested in too big transits. And big
            outliers will pollute periodicity search. So they are removed together with their local neighborhood
            in order not to leave residual contaminants.
        :type max_abs_stat: float
        :param update_masks: Whether to modify `self.stat_masks` and `self.veto_masks` attributes
        :type update_masks: bool
        :param save_as_attributes: Whether to save `self.min_neighbor_and_template_ses_snr`,
            `self.tail_stat_masks`, `self.too_large_neighbor_masks` as class attributes.
        :type save_as_attributes: bool
        :param veto_positive_outliers: Whether to also veto positive outliers, i.e. local maxima, and their areas of influence.
        :type veto_positive_outliers: bool
        """
        delta_snr_kill = ut.norm(loc=0, scale=1).isf(delta_snr_kill_sf)  # for 2% it is about 2
        # ut.get_p_value_gaussian(x=delta_snr_kill_sf, mean=0, std=2) # for 2% it is about 2
        min_neighbor_and_template_ses_snr = [[] for _ in range(len(self.quarters))]
        tail_stat_masks = [[np.ones(len(self.stat_masks[k][q])).astype(bool) for q in self.quarters] for k in range(len(self.stat_masks))]
        too_large_neighbor_masks = [[] for _ in range(len(self.quarters))]

        for q in range(len(self.quarters)):
            if self.is_zero_padded:
                msk_q = self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(self.big_hole_masks[q]) - self.zero_padding_ends[q][1]]
            else:
                msk_q = self.big_hole_masks[q]

            min_neighbor_stats_quarter = np.zeros((len(self.statistics_snr), len(self.statistics_snr[0][q])))
            if veto_positive_outliers:
                max_neighbor_stats_quarter = np.zeros((len(self.statistics_snr), len(self.statistics_snr[0][q])))
            else:
                max_neighbor_stats_quarter = []
            for k in range(len(self.statistics_snr)):
                temp = self.templates_of_indices[k]
                window_size = int(np.round(fraction_template_duration * tm.Durations[temp] / self.dt_days))

                stat_for_min_search = np.array(self.statistics_snr[k][q])
                stat_for_min_search[~msk_q] = np.nan

                min_indices = ut.sliding_nanargmin(array=stat_for_min_search, window_size=window_size)
                if veto_positive_outliers:
                    max_indices = ut.sliding_nanargmax(array=stat_for_min_search, window_size=window_size)
                else:
                    max_indices = []
                # -----
                # h_window = int(np.round(0.5 * fraction_template_duration * tm.Durations[temp] / self.dt_days))
                # min_indices = np.zeros(len(stat_for_min_search), dtype=int)
                # if veto_positive_outliers:
                #     max_indices = np.zeros(len(stat_for_min_search), dtype=int)
                # else:
                #     max_indices = []

                # for i in range(len(min_indices)):  # This takes a lot of time!
                #     ind_left = max(0, i-h_window)
                #     min_indices[i] = safe_nanargmin(stat_for_min_search[ind_left:min(len(stat_for_min_search),i+h_window)]) + ind_left
                #     if veto_positive_outliers:
                #         max_indices[i] = safe_nanargmax(stat_for_min_search[ind_left:min(len(stat_for_min_search),i+h_window)]) + ind_left

                bad_mask = (min_indices == -1)
                min_stats = stat_for_min_search[min_indices]
                min_stats[bad_mask] = np.nan
                min_neighbor_stats_quarter[k] = min_stats
                if veto_positive_outliers:
                    max_stats = stat_for_min_search[max_indices]
                    max_stats[bad_mask] = np.nan
                    max_neighbor_stats_quarter[k] = max_stats

                # self.min_neighbor_indices[k][q] = min_indices
                # self.min_neighbor_ses_snr[k][q] = min_stats
            min_neighbor_and_template_ses_snr_q = ut.safe_nanmin(min_neighbor_stats_quarter, axis=0)
            min_neighbor_and_template_ses_snr[q] = min_neighbor_and_template_ses_snr_q

            # If some of the minima are too large, we kill their area of influence:
            if veto_extreme_statistics:
                not_too_big_mask = min_neighbor_and_template_ses_snr_q > - np.abs(max_abs_stat)
                too_large_neighbor_masks[q] = not_too_big_mask

            # Updating:
            for k in range(len(self.statistics_snr)):
                tail_mask_good = (self.statistics_snr[k][q] < 0) & (
                            self.statistics_snr[k][q] <= min_neighbor_and_template_ses_snr_q + delta_snr_kill)
                # stats_quarter = np.array([self.statistics_snr[k][q] for k in range(len(self.statistics_snr))])
                # best_template_quarter = np.nanargmin(stats_quarter, axis=0)
                # best_template_snr_quarter = stats_quarter[best_template_quarter]
                tail_stat_masks[k][q] = tail_mask_good

            if veto_positive_outliers:
                max_neighbor_and_template_ses_snr_q = ut.safe_nanmax(max_neighbor_stats_quarter, axis=0)
                not_too_big_max_mask = max_neighbor_and_template_ses_snr_q < np.abs(max_abs_stat)
                too_large_neighbor_masks[q] &= not_too_big_max_mask

        if save_as_attributes:
            self.min_neighbor_and_template_ses_snr = min_neighbor_and_template_ses_snr
            self.tail_stat_masks = tail_stat_masks
            self.too_large_neighbor_masks = too_large_neighbor_masks

        # Updating:
        if update_masks:
            for q in range(len(self.quarters)):
                for k in range(len(self.statistics_snr)):
                    # Updating the veto masks:
                    if veto_tail_statistics:
                        self.veto_masks[k][q][self.trigger_indices_unpadded[k][q]] &= tail_stat_masks[k][q][self.trigger_indices_unpadded[k][q]]
                        self.stat_masks[k][q][self.trigger_indices_unpadded[k][q]] &= tail_stat_masks[k][q][self.trigger_indices_unpadded[k][q]]
                    # Also updating the too large:
                    self.veto_masks[k][q][self.trigger_indices_unpadded[k][q]] &= too_large_neighbor_masks[q][
                        self.trigger_indices_unpadded[k][q]]
                    self.stat_masks[k][q][self.trigger_indices_unpadded[k][q]] &= too_large_neighbor_masks[q][
                        self.trigger_indices_unpadded[k][q]]

    def calculate_veto_chi2_p_vals(self, n_parts_veto_snr=N_parts_veto_snr,
                                   min_bins_veto_part_time=Min_bins_veto_part_time,
                                   lobe_hlen_veto_part_time=Lobe_hlen_veto_part_time,
                                   n_parts_veto_fourier=N_parts_veto_fourier,
                                   wind_len_overlap_fourier=Wind_len_overlap_fourier,
                                   high_pas_veto_parts_fourier=False):
        """
        Calculates p-values of the chi-squared `veto_parts` that compares the amplitudes of separately fitted
        parts of every template in the time and fourier domains.
        :param n_parts_veto_snr: Maximal number of parts to split the whitened template when doing vetting
            in time domain. The parts are used to calculate the chi-square statistic of their amplitudes.
        :type n_parts_veto_snr: int
        :param min_bins_veto_part_time: Minimal size of the part used in time domain. If a part is smaller,
            the number of parts is decreased.
        :type min_bins_veto_part_time: int
        :param lobe_hlen_veto_part_time: If there is only one part because the template is too short, a chi-squared
            veto of the main lobe will be performed. THis parameter sets half-length of this lobe.
        :type lobe_hlen_veto_part_time: int
        :param n_parts_veto_fourier: maximal number of parts to split the whitened template when doing vetting
            in fourier domain
        :type n_parts_veto_fourier: int
        :param wind_len_overlap_fourier: Size of the window for the template splitting in the fourier domain.
            Is needed to avoid too long templates in the time domain. Is recommended to be smaller than the nperseg.
            The parts consistency veto in the Fourier domain is made to the overlapping
            parts because if the parts edges are too sharp in the Fourier domain, then in the time domain they
            will have an infinite support, and all tha glitches will produce impact. To restrict the size in the
            time domain, we make parts smooth and overlapping in the Fourier domain. `wind_len_overlap_fourier`
            sets the size of that overlapping window.
        :type wind_len_overlap_fourier: int
        :param high_pas_veto_parts_fourier: Whether to use the high-pass filter for the Fourier parts veto.
            Not recommended because it is not really helpful for some reason. The idea is that the data is too blue
            because detrending and inaccuracy of PSD measurement in the lower frequencies. So it makes sense to use
            high-pass filter to only check frequencies in which the noise in the whitened flux is indeed white.
            But it does not really improve the result, perhaps something else compensates it. In the post-processing
            veto of the shape though, high-passing is important.
        :type high_pas_veto_parts_fourier: bool
        """

        # ------------------------------------------------------------------------------------------------
        # Calculating and saving vato p-value for all the points!
        # ------------------------------------------------------------------------------------------------

        assert self.is_zero_padded, 'Zero padding ends must not be removed before vetting!'

        quarters = self.quarters
        stat_masks = self.stat_masks
        nperseg_psd = self.nperseg_psd
        zero_padding_ends = self.zero_padding_ends
        # ------------------------------------------------------------------------------------------------
        # ##################################################################################################

        # PREPARATION OF FLUXES, TEMPLATES, AND STATS
        # We prepare renormalized fluxes and templates to use them in the veto and not renormalize there every time:
        norms_templates = [[np.linalg.norm(self.whitened_templates_td[k][q]) for q in quarters]
                           for k in range(len(self.whitened_templates_td))]
        whitened_templates_td_renorm = [[self.whitened_templates_td[k][q] / norms_templates[k][q]
                                         for q in quarters] for k in range(len(self.whitened_templates_td))]
        whitened_templates_fd_renorm = [[self.whitened_templates_fd[k][q] / norms_templates[k][q]
                                         for q in quarters] for k in range(len(self.whitened_templates_td))]
        stds_fluxes = [np.std(self.whitened_fluxes_td[q][self.big_hole_masks[q]]) for q in quarters]
        fluxes_wh_fd_rewh = [self.whitened_fluxes_fd[q] / stds_fluxes[q] for q in quarters]

        # ##################################################################################################
        # VETTING
        # Attention: statistics and stat_mask is unpadded from the matched_filtering!
        # Veto p-vals will also be unpadded:
        self.veto_td_p_vals = [[np.ones(len(stat_masks[k][q])).astype(float) + np.nan for q in quarters] for k in
                               range(len(stat_masks))]
        self.veto_fd_p_vals = [[np.ones(len(stat_masks[k][q])).astype(float) + np.nan for q in quarters] for k in
                               range(len(stat_masks))]
        self.veto_p_vals = [[np.ones(len(stat_masks[k][q])).astype(float) + np.nan for q in quarters] for k in
                            range(len(stat_masks))]
        # ------------------------------------------------------------------------------------------------
        # VETTING PROCESS
        # Will collect trigger indices here:
        for k in range(len(self.trigger_indices_unpadded)):
            for q in quarters:  # over quarters
                # Check onlu things that are not already masked by the stat mask:
                trigger_inds_mask_to_check = stat_masks[k][q][self.trigger_indices_unpadded[k][q]]
                trigger_indices_quarter_unpadded = self.trigger_indices_unpadded[k][q][trigger_inds_mask_to_check]
                trigger_indices_quarter_padded = self.trigger_indices_padded[k][q][trigger_inds_mask_to_check]
                stat_mask_padded = np.concatenate((np.zeros(zero_padding_ends[q][0]).astype(bool), stat_masks[k][q],
                                                   np.zeros(zero_padding_ends[q][1]).astype(bool)))

                # ##################################################################################################
                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING BY SNR IN TIME DOMAIN)
                p_values_td, _ = ut.veto_parts(data_wh_fd=fluxes_wh_fd_rewh[q],
                                               template_wh_full=whitened_templates_td_renorm[k][q],
                                               indices_padded=trigger_indices_quarter_padded,
                                               mask_padded=stat_mask_padded, tlen=None,
                                               fn=Fn_fit_veto,  # this fn is not used
                                               splitting='snr', n_parts=n_parts_veto_snr,
                                               min_bins_part=min_bins_veto_part_time,
                                               nperseg_psd=nperseg_psd, lobe_hlen=lobe_hlen_veto_part_time,
                                               return_p_values=True)

                self.veto_td_p_vals[k][q][trigger_indices_quarter_unpadded] = p_values_td
                p_values_fd, _ = ut.veto_parts_fourier(data_wh_fd=fluxes_wh_fd_rewh[q],
                                                       template_wh_fd=whitened_templates_fd_renorm[k][q],
                                                       indices_padded=trigger_indices_quarter_padded,
                                                       mask_padded=stat_mask_padded, n_parts=n_parts_veto_fourier,
                                                       fn=Fn_fit_veto,  # This fn is a dummy variable here because we do thresholding after
                                                       wind_len_overlap=wind_len_overlap_fourier, nperseg_psd=nperseg_psd,
                                                       return_p_values=True, high_pass=high_pas_veto_parts_fourier)
                self.veto_fd_p_vals[k][q][trigger_indices_quarter_unpadded] = p_values_fd
                self.veto_p_vals[k][q][trigger_indices_quarter_unpadded] = np.nanmin(np.array([p_values_td, p_values_fd]), axis=0)

    def apply_thresholds_to_veto_chi2_p_values(self, fn=Fn_fit_veto, use_dist_to_hole_masks_for_threshold=True,
                                               fn_close_to_hole=5e-2, use_snr_dependent_threshold=True,
                                               save_masks_as_attributes=False,
                                               update_stat_and_veto_masks=True):
        """
        Takes the chi-squared `veto_parts` p-values and applies the threshold to them to decide which ones are too bad
        and should be masked. It is applied to the minimal p-value between the time and Fourier domains.
        The thresholds may be uniform or may depend on the distance to the big holes or on the SNR of the trigger.
        :param fn: False negative value for the chi-squared `veto_parts` p-value used in the general case.
        :type fn: float
        :param use_dist_to_hole_masks_for_threshold: Whether to use the distance to the big holes to regulate the
            threshold. If True, the threshold is `fn_close_to_hole` for the triggers close to the holes, as defined by
            `self.distance_to_hole_masks`.
        :type use_dist_to_hole_masks_for_threshold: bool
        :param fn_close_to_hole: The new false-negative value to be used instead of `fn` for the triggers close to holes
            as defined by `self.distance_to_hole_masks`. If `use_dist_to_hole_masks_for_threshold` is False, this value is not used.
        :type fn_close_to_hole: float
        :param use_snr_dependent_threshold: Whether to use the SNR-dependent threshold for the chi-squared `veto_parts`.
            If true, the threshold will be defined from the empirical function `ut.veto_threshold_snr_fun()`
        :type use_snr_dependent_threshold: bool
        :param save_masks_as_attributes: Whether to save `self.veto_p_val_masks` as an attribute.
        :type save_masks_as_attributes: bool
        :param update_stat_and_veto_masks: Whether to update the `self.stat_masks` and `self.veto_masks` with zeros in vetted places
        :type update_stat_and_veto_masks: bool
        """
        if save_masks_as_attributes:
            self.veto_p_val_masks = [[np.ones(len(self.stat_masks[k][q])).astype(bool) for q in self.quarters] for k in
                                     range(len(self.stat_masks))]
        # Updating the masks with rejected low p-values:
        for q in range(len(self.quarters)):
            for k in range(len(self.statistics_snr)):
                if (not use_dist_to_hole_masks_for_threshold) and (not use_snr_dependent_threshold):
                    veto_parts_rejection_mask_triggers = (self.veto_p_vals[k][q][self.trigger_indices_unpadded[k][q]] >= fn)
                else:
                    fns = np.ones(len(self.trigger_indices_unpadded[k][q]))
                    if use_dist_to_hole_masks_for_threshold:
                        trig_ind_unp_mask_close_to_hole = (~self.distance_to_hole_masks[k][q])[self.trigger_indices_unpadded[k][q]]
                        fns[trig_ind_unp_mask_close_to_hole] = fn_close_to_hole
                    if use_snr_dependent_threshold:
                        snrs_triggers = self.statistics_snr[k][q][self.trigger_indices_unpadded[k][q]]
                        thresholds_for_snrs = ut.veto_threshold_snr_fun(snr=snrs_triggers)
                        fns = np.minimum(fns, thresholds_for_snrs)
                    veto_parts_rejection_mask_triggers = (self.veto_p_vals[k][q][self.trigger_indices_unpadded[k][q]] >= fns)

                if update_stat_and_veto_masks:
                    self.veto_masks[k][q][self.trigger_indices_unpadded[k][q]] &= veto_parts_rejection_mask_triggers
                    self.stat_masks[k][q][self.trigger_indices_unpadded[k][q]] &= veto_parts_rejection_mask_triggers
                if save_masks_as_attributes:
                    self.veto_p_val_masks[k][q][self.trigger_indices_unpadded[k][q]] = veto_parts_rejection_mask_triggers

    def renormalize_stats_variance(self):
        for k in range(len(self.statistics_snr)):
            for q in self.quarters:
                # RE-NORMALIZING THE VARIANCE
                # The vetted triggers could have changed the true variance, now we want the cleaned stat to have
                # unit variance. Also, positive can have bad distr which I do not want to impact the negative
                # So we just repeat what was done before for moving variance correctness, but now after we updated masks with veto
                stat_snr, movvar_meas = ut.renormalize_variance_fun(
                    movvar_meas=np.array(self.stat_moving_vars[k][q]),
                    statmask=np.array(self.stat_masks[k][q]),
                    stat_snr=np.array(self.statistics_snr[k][q]),
                    stat=np.array(self.statistics_raw[k][q]))
                self.statistics_snr[k][q] = stat_snr
                self.stat_moving_vars[k][q] = movvar_meas

    #%% Old vetting

    def veto_statistics_old(self, apply_only_to_neg=True,  # ATTENTION WARNING TODO In run_12, it was False!!!
                            margin_snr=Margin_snr_veto,
                            max_abs_stat=Max_abs_stat_veto,
                            fn=Fn_fit_veto,
                            n_parts_veto_snr=N_parts_veto_snr,
                            min_bins_veto_part_time=Min_bins_veto_part_time,
                            lobe_hlen_veto_part_time=Lobe_hlen_veto_part_time,
                            n_parts_veto_fourier=N_parts_veto_fourier,
                            wind_len_overlap_fourier=Wind_len_overlap_fourier,
                            threshold_peak_sigma=Threshold_peak_sigma_veto,  # there will also be look-elsewhere
                            save_triggers_as_attribute=False,
                            high_pas_veto_parts_fourier=False):
        """
        Verify big triggers and mask ones which are non-transit-like.
        Selects from the statistic the triggers whose abs is >margin_snr, which are negative or
        both negative and positive, as defined by `apply_only_to_neg` parameter.
        Triggers inside big holes are ignored.
        Makes several tests: chi-square partial amplitude tests in time and fourier domain,
        alternative hypothesis testing against a peak.
        Should be done before unpadding!
        :param high_pas_veto_parts_fourier:
        :type high_pas_veto_parts_fourier:
        :param max_abs_stat: Maximal absolute value of statistic in units of SNr that is allowed. Larger
            ones will be masked. This is done because we are not interested in too big transits. And big
            outliers will pollute periodicity search.
        :type max_abs_stat: float
        :param apply_only_to_neg: Whether to apply the veto only to negative statistics, which transits are
            expected to be, or also to the positive ones. The latter is needed if we want to look at the
            inverted data. It also can affect the negative values processing if there are interactions.
        :type apply_only_to_neg: bool
        :param margin_snr: the minimal abs value of the snr from which the vetting starts to be applied
        :type margin_snr: float
        :param fn: false negative probability for vetoes, individually.
        :type fn: float
        :param n_parts_veto_snr: maximal number of parts to split the whitened template when doing vetting
            in time domain
        :type n_parts_veto_snr: int
        :param min_bins_veto_part_time: Minimal size of the part used in time domain. If a part is smaller,
            the number of parts is decreased.
        :type min_bins_veto_part_time: int
        :param lobe_hlen_veto_part_time: If there is only one part because the template is too short, a chi-squared
            veto of the main lobe will be performed. THis parameter sets half-length of this lobe.
        :type lobe_hlen_veto_part_time: int
        :param n_parts_veto_fourier: maximal number of parts to split the whitened template when doing vetting
            in fourier domain
        :type n_parts_veto_fourier: int
        :param wind_len_overlap_fourier: Size of the window for the template splitting in the fourier domain.
            Is needed to avoid too long templates in the time domain. Is recommended to be smaller than the nperseg.
            The parts consistency veto in the Fourier domain is made to the overlapping
            parts because if the parts edges are too sharp in the Fourier domain, then in the time domain they
            will have an infinite support, and all tha glitches will produce impact. To restrict the size in the
            time domain, we make parts smooth and overlapping in the Fourier domain. `wind_len_overlap_fourier`
            sets the size of that overlapping window.
        :type wind_len_overlap_fourier: int
        :param threshold_peak_sigma: Threshold value for the peak veto in units of sigmas of distribution.
            Peak veto makes sure it is more probable for the score to originate from the transit and not from
            a spurious peak. This veto is basically useless.
        :type threshold_peak_sigma: float
        :param save_triggers_as_attribute: whether to save the triggers with descriptions as part of the class
            attribute. If `True`, veto masks will be created.
        :type save_triggers_as_attribute: bool
        """
        assert self.is_zero_padded, 'Zero padding ends must not be removed before vetting!'
        # Attention: statistics and stat_mask is unpadded from the matched_filtering!
        # Veto masks will also be unpadded:
        self.veto_masks = [[np.ones(len(self.stat_masks[k][q])).astype(bool) for q in self.quarters] for k in
                           range(len(self.stat_masks))]
        if save_triggers_as_attribute:
            self.veto_bit_masks = [[np.zeros(len(self.veto_masks[k][q])).astype(int) for q in self.quarters]
                                   for k in range(len(self.veto_masks))]
        # veto_bit_masks code which exactly test failed. Coding is in binary format, codes are powers of 2
        # veto_bit_codes = {0: 'too_big' ,1: 'extreme_position', 2: 'peak', 3: 'parts_time', 4: 'parts_fourier'}

        # ##################################################################################################
        # MASKING TOO LARGE SCORES
        for k in range(len(self.tb.initial_templates_shifts)):
            for q in self.quarters:  # over quarters
                too_big_mask = np.abs(self.statistics_snr[k][q]) > max_abs_stat
                self.veto_masks[k][q][too_big_mask] = False  # too_big
                self.stat_masks[k][q][too_big_mask] = False  # too_big
                if save_triggers_as_attribute:
                    self.veto_bit_masks[k][q][too_big_mask] += 2 ** 0

        # ##################################################################################################
        # PREPARATION OF FLUXES, TEMPLATES, AND STATS
        # We prepare renormalized fluxes and templates to use them in the veto and not renormalize there every time:
        norms_templates = [[np.linalg.norm(self.whitened_templates_td[k][q]) for q in self.quarters]
                           for k in range(len(self.whitened_templates_td))]
        whitened_templates_td_renorm = [[self.whitened_templates_td[k][q] / norms_templates[k][q]
                                         for q in self.quarters] for k in range(len(self.whitened_templates_td))]
        whitened_templates_fd_renorm = [[self.whitened_templates_fd[k][q] / norms_templates[k][q]
                                         for q in self.quarters] for k in range(len(self.whitened_templates_td))]
        stds_fluxes = [np.std(self.whitened_fluxes_td[q][self.big_hole_masks[q]]) for q in self.quarters]
        fluxes_wh_fd_rewh = [self.whitened_fluxes_fd[q] / stds_fluxes[q] for q in self.quarters]
        # Preparing peak statistic (will be unpadded):
        stats_peak, whitened_peak_temps_td, whitened_peak_temps_fd = self.prepare_peak_statistics()

        # ##################################################################################################
        # VETTING
        # Will collect trigger indices here:
        self.veto_fd_p_vals = [[np.ones(len(self.statistics_snr[k][q])).astype(float) + np.nan for q in self.quarters] for k in
                               range(len(self.statistics_snr))]
        self.trigger_indices = [[[] for _ in self.quarters] for __ in range(len(self.whitened_templates_td))]
        for k in range(len(self.tb.initial_templates_shifts)):
            for q in self.quarters:  # over quarters
                templ_width = int(np.ceil(self.tb.durations[self.tb.templates_of_indices[k]] / self.dt_days))
                # ##################################################################################################
                # EXTREME POSITION: we do not trust it because the data mey be bad, and veto will be poor there
                self.veto_masks[k][q][:templ_width // 2] = False
                self.veto_masks[k][q][-templ_width // 2:] = False
                self.stat_masks[k][q][:templ_width // 2] = False
                self.stat_masks[k][q][-templ_width // 2:] = False
                if save_triggers_as_attribute:
                    self.veto_bit_masks[k][q][:templ_width // 2] += 2 ** 1
                    self.veto_bit_masks[k][q][-templ_width // 2:] += 2 ** 1

                # Finding threshold-crossing trigger indices:
                zpe_0 = self.zero_padding_ends[q][0]
                trigger_indices_quarter_unpadded = (np.abs(self.statistics_snr[k][q]) >= margin_snr) & self.stat_masks[k][q]
                if apply_only_to_neg:
                    trigger_indices_quarter_unpadded = (self.statistics_snr[k][q] <= 0) & trigger_indices_quarter_unpadded
                trigger_indices_quarter_unpadded = np.where(trigger_indices_quarter_unpadded)[0]
                trigger_indices_quarter_padded = trigger_indices_quarter_unpadded + zpe_0
                if save_triggers_as_attribute:
                    self.trigger_indices[k][q] = trigger_indices_quarter_unpadded

                # ##################################################################################################
                # CHECKING FOR PEAK
                # Warning: here, work with unpadded!
                # Usually, this veto does not vet anyone.
                trigger_indices_quarter_unpadded = trigger_indices_quarter_padded - zpe_0
                dur_bins = self.tb.durations[self.tb.templates_of_indices[k]] / self.dt_days
                if dur_bins > 5:
                    h_dur_check = int(self.tb.durations[self.tb.templates_of_indices[k]] / self.dt_days / 2)
                    decorr_coef = (irfft(self.whitened_templates_fd[k][q] * np.conjugate(whitened_peak_temps_fd[q]))[0] /
                                   ((whitened_peak_temps_td[q] @ whitened_peak_temps_td[q])
                                    * (self.whitened_templates_td[k][q] @ self.whitened_templates_td[k][q])) ** 0.5)  # not rolled, approx
                    max_peak_stats = np.array([np.nanmax(stats_peak[q][max(0, i - h_dur_check): i + h_dur_check])
                                               for i in range(len(stats_peak[q]))])
                    min_peak_stats = np.array([np.nanmin(stats_peak[q][max(0, i - h_dur_check): i + h_dur_check])
                                               for i in range(len(stats_peak[q]))])
                    signs = np.sign(self.statistics_snr[k][q][trigger_indices_quarter_unpadded])
                    min_or_max = np.array([max_peak_stats[ind_j] if signs[j] > 0 else min_peak_stats[ind_j]
                                           for j, ind_j in enumerate(trigger_indices_quarter_unpadded)])
                    is_goods = (signs * (self.statistics_snr[k][q][trigger_indices_quarter_unpadded] * decorr_coef
                                         - min_or_max) / (1 - decorr_coef ** 2) ** 0.5 < threshold_peak_sigma)
                    self.veto_masks[k][q][trigger_indices_quarter_unpadded[~is_goods]] = False  # peak
                    self.stat_masks[k][q][trigger_indices_quarter_unpadded[~is_goods]] = False  # peak
                    if save_triggers_as_attribute:
                        self.veto_bit_masks[k][q][trigger_indices_quarter_unpadded[~is_goods]] += 2 ** 2

                    trigger_indices_quarter_padded = trigger_indices_quarter_padded[is_goods]

                stat_mask_padded = np.concatenate((np.zeros(self.zero_padding_ends[q][0]).astype(bool), self.stat_masks[k][q],
                                                   np.zeros(self.zero_padding_ends[q][1]).astype(bool)))

                # ##################################################################################################
                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING BY SNR IN TIME DOMAIN)
                isgood_parts = ut.veto_parts(data_wh_fd=fluxes_wh_fd_rewh[q],
                                             template_wh_full=whitened_templates_td_renorm[k][q],
                                             indices_padded=trigger_indices_quarter_padded,
                                             mask_padded=stat_mask_padded, tlen=None, fn=fn, splitting='snr',
                                             n_parts=n_parts_veto_snr, min_bins_part=min_bins_veto_part_time,
                                             nperseg_psd=self.nperseg_psd, lobe_hlen=lobe_hlen_veto_part_time)
                self.veto_masks[k][q][trigger_indices_quarter_padded[~isgood_parts] - zpe_0] = False
                self.stat_masks[k][q][trigger_indices_quarter_padded[~isgood_parts] - zpe_0] = False
                if save_triggers_as_attribute:
                    self.veto_bit_masks[k][q][trigger_indices_quarter_padded[~isgood_parts] - zpe_0] += 2 ** 3
                trigger_indices_quarter_padded = trigger_indices_quarter_padded[isgood_parts]

                # ##################################################################################################
                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING IN FOURIER)
                p_values_fd, isgood_parts = ut.veto_parts_fourier(data_wh_fd=fluxes_wh_fd_rewh[q],
                                                     template_wh_fd=whitened_templates_fd_renorm[k][q],
                                                     indices_padded=trigger_indices_quarter_padded,
                                                     mask_padded=stat_mask_padded, n_parts=n_parts_veto_fourier,
                                                     fn=fn, wind_len_overlap=wind_len_overlap_fourier, nperseg_psd=self.nperseg_psd,
                                                     high_pass=high_pas_veto_parts_fourier, return_p_values=True)
                self.veto_fd_p_vals[k][q][trigger_indices_quarter_padded - zpe_0] = p_values_fd
                self.veto_masks[k][q][trigger_indices_quarter_padded[~isgood_parts] - zpe_0] = False
                self.stat_masks[k][q][trigger_indices_quarter_padded[~isgood_parts] - zpe_0] = False
                if save_triggers_as_attribute:
                    self.veto_bit_masks[k][q][trigger_indices_quarter_padded[~isgood_parts] - zpe_0] += 2 ** 4
                # trigger_indices_quarter_padded = trigger_indices_quarter_padded[isgood_parts]

        # Setting statistics in the masked places to zero:
        for k in range(len(self.statistics_snr)):
            for q in self.quarters:
                self.statistics_snr[k][q][~self.stat_masks[k][q]] = 0.
                self.statistics_raw[k][q][~self.stat_masks[k][q]] = 0.
                self.stat_moving_vars[k][q][~self.stat_masks[k][q]] = 0.  # This is very important for the periodicity search

                # RE-NORMALIZING THE VARIANCE
                # The vetted triggers could have changed the true variance, now we want the cleaned stat to have
                # unit variance. Also, positive can have bad distr which I do not want to impact the negative
                # So we just repeat what was done before for moving variance correctness, but now after we updated masks with veto
                movvar_meas = np.array(self.stat_moving_vars[k][q])
                statmask = np.array(self.stat_masks[k][q])
                stat_snr = np.array(self.statistics_snr[k][q])
                stat = np.array(self.statistics_raw[k][q])
                # Making positive values unit variance:
                indices_sign = statmask & (stat > 0)
                var_snr_indices = np.nanmean((stat_snr[indices_sign]) ** 2)
                movvar_meas[indices_sign] *= var_snr_indices
                # Making negative values unit variance:
                indices_sign = statmask & (stat < 0)
                var_snr_indices = np.nanmean((stat_snr[indices_sign]) ** 2)
                movvar_meas[indices_sign] *= var_snr_indices
                # Use this new moving variance for all the rest:
                stat_snr[statmask] = stat[statmask] / (movvar_meas[statmask]) ** 0.5  # now should be N(0,1) for sure
                self.statistics_snr[k][q] = stat_snr
                self.stat_moving_vars[k][q] = movvar_meas

        if not save_triggers_as_attribute:
            self.veto_bit_masks = None

    def prepare_peak_statistics(self):
        """
        Calculate the matched-filtering statistic of the data and the peak-like template.
        Is used in the veto to make sure the score is not just some residual peak from something.
        Usually is not important because the outlier cut is killing most of significant peaks,
        but still is kept for consistency.
        """
        peak_template = np.zeros(self.support)
        peak_template[0] = 1.
        peak_template[1] = 0.5
        peak_template[-1] = 0.5
        peak_template = rfft(peak_template)
        whitened_peak_temps_fd = [peak_template * self.wf_fds[q] for q in self.quarters]
        whitened_peak_temps_td = [irfft(whitened_peak_temps_fd[q]) for q in self.quarters]
        stats_peak = [np.real(irfft(np.conjugate(whitened_peak_temps_fd[q]) * self.whitened_fluxes_fd[q]))
                      for q in self.quarters]
        stats_peak = [stats_peak[q] / ut.outlier_free_std(stats_peak[q], nsig=3, n_iterations=2,
                                                          mask=self.big_hole_masks[q]) for q in self.quarters]
        stats_peak = [stats_peak[q][self.zero_padding_ends[q][0]:len(stats_peak[q]) - self.zero_padding_ends[q][1]]
                      for q in self.quarters]
        # -> CUT THE STATISTIC BECAUSE STATISTICS_SNR IS ALSO CUT
        return stats_peak, whitened_peak_temps_td, whitened_peak_temps_fd

    # %%
    def calculate_gaussianized_ses(self):
        """
        Calculating the single-event statistic which have the same ranking as the given statistics but follow
        exactly the Gaussian distribution. Is done through the SES units, and then the previous variances
        of the quarters are returned.
        We do it per template (which is especially important for the injections where we only have one template):
        """
        self.statistics_gaussianized = copy.deepcopy(self.statistics_raw)
        self.statistics_snr_gaussianized = copy.deepcopy(self.statistics_snr)
        for k in range(len(self.statistics_snr_gaussianized)):
            all_stat = np.concatenate([self.statistics_snr_gaussianized[k][q][self.stat_masks[k][q]] for q in range(
                len(self.statistics_snr_gaussianized[k]))])
            all_stat_c, transform = ut.score_correction(all_stat, return_function=True, two_sided_distr=True)
            for q in range(len(self.statistics_snr_gaussianized[k])):
                self.statistics_snr_gaussianized[k][q][self.stat_masks[k][q]] = transform(
                    self.statistics_snr_gaussianized[k][q][self.stat_masks[k][q]])
                self.statistics_gaussianized[k][q][self.stat_masks[k][q]] = (self.statistics_snr_gaussianized[k][q][self.stat_masks[k][q]]
                                                                             * self.stat_moving_vars[k][q][self.stat_masks[k][q]] ** 0.5)

    def calculate_ses_distr_corrections(self, n_bins=N_bins_hist_ses, snr_0_fit=Snr_0_fit_hist_ses, beg_min=Beg_min_hist_ses):
        """
        Making histograms of the distributions of SNRS squared for every template, fitting the two-gaussian model and
        finding the parameters of the tail gaussian. The first gaussian is supposed to be the standard one because the
        data was brought to units of SNR by normalizing with its variance. However, the tail can be too heavy. Usually,
        the distribution which is a sum of two gaussians fits the data. Therefore, the histogram is fitted for the
        scale anf the amplitude of the tail gaussian. We do not make likelihood optimization of the tail, we just
        fit the CDF of the bins in a stupid way. The likelihood fit with too many points will not work, so it could
        only be used in the very tail. We are not too much afraid that we will overfit large single outliers,
        we do not want to save these outliers anyway. We even cut some of them in the veto.
        Is performed separately for negative and positive values. Too big by absolute value outliers are supposed to be
        removed in the veto before.
        :param n_bins: number of bins to use in the histogram. The recommended is 50
        :type n_bins: int
        :param snr_0_fit: The smallest snr from which to start the fitting of the tail. Recommended 2.5
        :type snr_0_fit: float
        :param beg_min: Consider deviations from the distribution srarting from this SNR
        :type beg_min: float
        """
        # Attention: the too big scores were removed before with veto
        self.stat_non_gauss_scales_coefs = [[[] for __ in range(len(self.statistics_snr))],  # one is for negative,
                                            [[] for __ in range(len(self.statistics_snr))]]  # the other one is for positive
        self.log_non_gaussian_corrections = [[np.zeros_like(self.statistics_snr[k][q])
                                              for q in self.quarters] for k in range(len(self.statistics_snr))]
        for k in range(len(self.statistics_snr)):
            stats_full = [self.statistics_snr[k][q][self.stat_masks[k][q]] for q in self.quarters]
            # statistics_raw are corrected twice per variance to account for the template part,
            # they are not expected to distribute well
            stats_full = [stats_full[q] / np.std(stats_full[q]) for q in self.quarters]
            stats_full = np.concatenate(stats_full)
            stats_full_0 = np.array(stats_full)

            # For negative (it will be separate from the positive because they have different distributions
            stats_full = np.abs(np.array(stats_full[stats_full < 0]))
            best_scale, best_coef = ut.fit_two_gaussian_mix(stats_full, n_bins=n_bins, snr_0_fit=snr_0_fit, beg_min=beg_min)
            self.stat_non_gauss_scales_coefs[0][k] = [best_scale, best_coef]
            for q in self.quarters:
                indices_mask = self.statistics_snr[k][q] < 0.
                self.log_non_gaussian_corrections[k][q][indices_mask] = ut.log_non_gaussian_correction(
                    stat=np.abs(self.statistics_snr[k][q][indices_mask]), ampl=best_coef, scale=best_scale)
                # -> the minus sign is because the correction is mostly negative to punish too high scores. But since the
                # scores are multiplied by -1 because they are negative themselves, the punishing
                # additive correction should be positive
                # NO, IT IS JUST PROBABILITY, IT SHOULD NOT HAVE THESE SIGNS
                self.log_non_gaussian_corrections[k][q][self.log_non_gaussian_corrections[k][q] > 0] = 0

                # -> if the correction is trying to increase the score, we make it zero because we do not want to increase the scores
                # this can happen for small scores where the slope of the gaussian sum is smaller than the slope of the base gaussian
                # this thing may affect the overall variance of the scores!
            # For positive
            stats_full = np.array(stats_full_0)
            stats_full = np.abs(np.array(stats_full[stats_full > 0]))
            best_scale, best_coef = ut.fit_two_gaussian_mix(stats_full)
            self.stat_non_gauss_scales_coefs[1][k] = [best_scale, best_coef]
            for q in self.quarters:
                indices_mask = self.statistics_snr[k][q] > 0.
                self.log_non_gaussian_corrections[k][q][indices_mask] = ut.log_non_gaussian_correction(
                    stat=self.statistics_snr[k][q][indices_mask], ampl=best_coef, scale=best_scale)
                self.log_non_gaussian_corrections[k][q][self.log_non_gaussian_corrections[k][q] > 0] = 0

    def connect_quarters(self):
        """
        Takes data saved by quarter and connects them together to the common time axis.
        The new time array is generated, assuming that data is equally sampled.
        Then, the data quarters are appended to the data array according to their first time value.
        If there is a subbin shift in time between the quarters, the quarter is put to the
        closest tome bin
        Creates for all templates (shifts included in the same list):
        self.time_full
        self.big_hole_masks_full
        self.statistics_full
        self.stat_vars_full
        self.stat_masks_full
        self.corrections_full
        self.statistics_gaussianized_full
        """
        if self.is_zero_padded:
            self.cut_zero_padding_ends()
        self.time_full, self.big_hole_masks_full = af.connect_quarters_fun(self.times, self.big_hole_masks,
                                                                           zero_padding_ends=None,
                                                                           is_data_cut=~self.is_zero_padded,
                                                                           is_time_cut=~self.is_zero_padded,
                                                                           dt=self.dt_days, fill_gaps_by_nan=False)

        self.big_hole_masks_full = self.big_hole_masks_full.astype(bool)

        self.statistics_full = np.zeros((len(self.statistics_raw), len(self.time_full)), dtype=Dtype_r)
        self.stat_vars_full = np.zeros((len(self.statistics_raw), len(self.time_full)), dtype=Dtype_r)
        self.stat_masks_full = np.zeros((len(self.statistics_raw), len(self.time_full)), dtype=bool)
        self.corrections_full = np.zeros((len(self.statistics_raw), len(self.time_full)), dtype=Dtype_r)
        self.statistics_gaussianized_full = np.zeros((len(self.statistics_raw), len(self.time_full)), dtype=Dtype_r)

        for k in range(len(self.statistics_raw)):
            self.statistics_full[k] = af.connect_quarters_fun(self.times, self.statistics_raw[k],
                                                              zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                              is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                              fill_gaps_by_nan=False)[1]
            self.stat_masks_full[k] = (af.connect_quarters_fun(self.times, self.stat_masks[k],
                                                               zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                               is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                               fill_gaps_by_nan=False)[1]).astype(bool)
            self.stat_vars_full[k] = af.connect_quarters_fun(self.times, self.stat_moving_vars[k],
                                                             zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                             is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                             fill_gaps_by_nan=False)[1]
            if self.log_non_gaussian_corrections is not None:
                self.corrections_full[k] = af.connect_quarters_fun(self.times, self.log_non_gaussian_corrections[k],
                                                                   zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                                   is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                                   fill_gaps_by_nan=False)[1]
            if self.statistics_gaussianized is not None:
                self.statistics_gaussianized_full[k] = af.connect_quarters_fun(self.times, self.statistics_gaussianized[k],
                                                                               zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                                               is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                                               fill_gaps_by_nan=False)[1]

    def check_possible_number_of_transits(self, period, return_number=True, return_relative=False):
        """
        Roughly checks how many non-zero transits there can be in this data. Meaning transits which
        do not fall in the holes, assuming period `period`.  It is a rough check needed to filter out stars that lack
        data and will be probably rejected by veto_mes. The check is performed by roughly folding the mask and seeing
        what is the maximal stack of non-masked points. Returned values can be either the actual number of such good transits, or their
        relative fraction with respect to the total number of transits (how many layers there are in the fold).
        :param period: period in days
        :type period: float
        :param return_number: Whether to return the actual maximal number of transits
        :type return_number: bool
        :param return_relative: Whether to return the fraction of transits that should happen in this time.
        :type return_relative: bool
        """
        if hasattr(self, 'stat_masks_full') and self.stat_masks_full is not None:
            or_mask = np.bitwise_or.reduce(self.stat_masks_full, axis=0).astype(float)
            rough_folding = ut.rough_fold(array=or_mask, period_bins=int(np.round(period / self.dt_days)))
            max_n = np.nanmax(rough_folding)
            if return_relative:
                min_expected_transits = ut.rough_fold(array=np.ones(len(or_mask)), period_bins=int(np.round(period / self.dt_days)))
                max_fraction = np.max(rough_folding / min_expected_transits)
                if return_number:
                    return int(np.round(max_n)), max_fraction
                else:
                    return max_fraction
            elif return_number:
                return int(np.round(max_n))
        else:
            assert hasattr(self, 'stat_masks_interleaved') and self.stat_masks_interleaved is not None
            assert hasattr(self, 'stat_masks_not_interleaved') and self.stat_masks_not_interleaved is not None

            period_bins_interleaved = int(np.round(period / (self.dt_days / self.n_shifts)))
            or_mask_interleaved = np.bitwise_or.reduce(self.stat_masks_interleaved, axis=0).astype(float)
            rough_folding_interleaved = ut.rough_fold(array=or_mask_interleaved, period_bins=period_bins_interleaved)
            max_n_interleaved = np.nanmax(rough_folding_interleaved)

            period_bins_not_interleaved = int(np.round(period / self.dt_days))
            or_mask_not_interleaved = np.bitwise_or.reduce(self.stat_masks_not_interleaved, axis=0).astype(float)
            rough_folding_not_interleaved = ut.rough_fold(array=or_mask_not_interleaved, period_bins=period_bins_not_interleaved)
            max_n_not_interleaved = np.nanmax(rough_folding_not_interleaved)
            if return_relative:
                min_expected_transits_interleaved = ut.rough_fold(array=np.ones(len(or_mask_interleaved)),
                                                                  period_bins=period_bins_interleaved)
                max_fraction_interleaved = np.max(rough_folding_interleaved / min_expected_transits_interleaved)

                min_expected_transits_not_interleaved = ut.rough_fold(array=np.ones(len(or_mask_not_interleaved)),
                                                                      period_bins=period_bins_not_interleaved)
                max_fraction_not_interleaved = np.max(rough_folding_not_interleaved / min_expected_transits_not_interleaved)
                if return_number:
                    return (max(int(np.round(max_n_interleaved)), int(np.round(max_n_not_interleaved))),
                            max(max_fraction_interleaved, max_fraction_not_interleaved))
                else:
                    return max(max_fraction_interleaved, max_fraction_not_interleaved)
            elif return_number:
                return max(int(np.round(max_n_interleaved)), int(np.round(max_n_not_interleaved)))

    def if_has_potentially_enough_transits(self, periods_to_check,
                                           min_required_nonzero_transits_number=Min_required_nonzero_transits_number,
                                           min_required_nonzero_transits_fraction=Min_required_nonzero_transits_fraction):
        """
        Checks whether it is possible from the rough geometric considerations to get needed number of valid transits
        and their fraction with respect to the total expected number of transits for these periods.
        It is a rough check needed to filter out stars that lack data and will be probably rejected by veto_mes. The check is performed by
        roughly folding the mask and seeing what is the maximal stack of non-masked points.
        Warning: the check is only done for the specified periods, so in principle, there can be other periods in-between
        :param periods_to_check: List of periods to perform the check on them
        :type periods_to_check: list of float or numpy.ndarray of float
        :param min_required_nonzero_transits_number: Minimal number of valid transits that should be to pass the test
        :type min_required_nonzero_transits_number: int
        :param min_required_nonzero_transits_fraction: Minimal fraction of transits with respect to the total number
            of transits allowed by time, that should be to pass the test
        :type min_required_nonzero_transits_fraction: float
        :return: Whether there are enough transits, as required
        :rtype: bool
        """
        numbers = []
        fractions = []
        for per in periods_to_check:
            num, frac = self.check_possible_number_of_transits(period=per, return_number=True, return_relative=True)
            numbers.append(num)
            fractions.append(frac)
        max_number_nonzero_transits = np.max(numbers)
        max_fraction_nonzero_transits = np.max(fractions)
        if (max_number_nonzero_transits >= min_required_nonzero_transits_number
                and max_fraction_nonzero_transits >= min_required_nonzero_transits_fraction):
            return True
        else:
            return False

    def scramble_full_stats(self, max_hole=Max_hole_pieces_scrampling, max_piece=Max_piece_scrambling, min_piece=Min_piece_scrambling,
                            random_state=None, random_seed=None, regenerate_pieces=False):
        """
        Selects random pieces of the full statistics, scrambles them randomly and puts randomly onto the new time axis.
        The total duration of the mission is approximately preserved. The cutting of random pieces is done with regard to
        the big holes in the data. Every big hole will correspond to a cut. Most of the cuts are just along the big holes.
        If the remaining piece is too large, it will be cut further.

        self.tot_hole_len_scramble
        self.inds_pieces_scramble
        self.inds_pieces_in_scrambled

        :param max_hole: Maximal hole which can be ignored. Bigger holes will be cuts between pieces
        :type max_hole: int
        :param max_piece: Maximal piece. Bigger pieces will be split.
        :type max_piece: int
        :param min_piece: Minimal piece that can be obtained in the splitting.
        :type min_piece: int
        param random_state: random state of numpy.random.set_state to fix the random slides generation.
            Overrides `random_seed`
        :type random_state: tuple[str, numpy.ndarray] corresponding to numpy.random.get_state()
        :param random_seed: Sets randon seed of numpy.random. Overriden by `random_state`
        :type random_seed: int
        :param regenerate_pieces: Force creating new cutting of pieces even if pieces already exist.
        :type regenerate_pieces: bool
        """
        if random_state is not None:
            np.random.set_state(random_state)
        elif random_seed is not None:
            np.random.seed(random_seed)
        else:
            np.random.seed(None)
        random_state = np.random.get_state()
        # if self.inds_pieces_scramble is not None:
        #     self.undo_scramble_full_stats()
        # if it was never scrambled before or was unscrambled, we need to define the pieces to be shuffled:
        if (self.inds_pieces_scramble is None) or regenerate_pieces:
            if self.is_scrambled:
                self.undo_scramble_full_stats()
            inds_pieces_scramble = ut.get_pieces_indices(mask=self.big_hole_masks_full, max_hole=max_hole, max_piece=max_piece,
                                                         min_piece=min_piece, random_state=random_state)
            # Warning: if there is a hole in the beginning or the end, it may (?) be treated incorrectly and the
            # length of the scrambled data will be effectively bigger
            self.tot_hole_len_scramble = len(self.time_full) - np.sum([ip[1] - ip[0] for ip in inds_pieces_scramble])
            self.inds_pieces_scramble = np.array(inds_pieces_scramble)
            np.random.shuffle(self.inds_pieces_scramble)
        else:
            numbers_shuffle = np.arange(len(self.inds_pieces_scramble)).astype(int)
            np.random.shuffle(numbers_shuffle)
            self.inds_pieces_scramble = self.inds_pieces_scramble[numbers_shuffle]  # save to be able to unscramble
            inds_pieces_scramble = self.inds_pieces_in_scrambled[numbers_shuffle]  # to re-scramble the present scrambled pieces

        np.random.set_state(random_state)  # returning to make the state controllable
        # How many spaces between the pieces and plus one in the end, although it effectively changes the length:
        len_holes = ut.gen_uniform_piece_lengths(tot_len=self.tot_hole_len_scramble, n_pieces=len(inds_pieces_scramble) - 1)
        # `self.connect_quarters` is now destroyed. do it again to restore the original positioning
        self.big_hole_masks_full, self.inds_pieces_in_scrambled = ut.rearrange_pieces(self.big_hole_masks_full,
                                                                                      inds_pieces=inds_pieces_scramble,
                                                                                      hole_lens=len_holes, fill_hole_with=0.,
                                                                                      return_inds_target_res=True)
        self.big_hole_masks_full = self.big_hole_masks_full.astype(bool)

        for k in range(len(self.statistics_full)):
            self.statistics_full[k] = ut.rearrange_pieces(self.statistics_full[k], inds_pieces=inds_pieces_scramble,
                                                          hole_lens=len_holes, fill_hole_with=0.)
            self.stat_masks_full[k] = ut.rearrange_pieces(self.stat_masks_full[k], inds_pieces=inds_pieces_scramble,
                                                          hole_lens=len_holes, fill_hole_with=False)
            self.stat_vars_full[k] = ut.rearrange_pieces(self.stat_vars_full[k], inds_pieces=inds_pieces_scramble,
                                                         hole_lens=len_holes, fill_hole_with=0.)
            self.corrections_full[k] = ut.rearrange_pieces(self.corrections_full[k], inds_pieces=inds_pieces_scramble,
                                                           hole_lens=len_holes, fill_hole_with=0.)
            self.statistics_gaussianized_full[k] = ut.rearrange_pieces(self.statistics_gaussianized_full[k],
                                                                       inds_pieces=inds_pieces_scramble, hole_lens=len_holes,
                                                                       fill_hole_with=0.)
        self.is_scrambled = True
        self.is_shifts_interleaved = False

    def undo_scramble_full_stats(self):
        """
        Undoes what was done in `scramble_full_stats`.
        """
        self.big_hole_masks_full = ut.rearrange_pieces(self.big_hole_masks_full, inds_pieces=self.inds_pieces_in_scrambled,
                                                       hole_lens=None, fill_hole_with=0., inds_target=self.inds_pieces_scramble)
        self.big_hole_masks_full = self.big_hole_masks_full.astype(bool)
        for k in range(len(self.statistics_full)):
            self.statistics_full[k] = ut.rearrange_pieces(self.statistics_full[k], inds_pieces=self.inds_pieces_in_scrambled,
                                                          hole_lens=None, fill_hole_with=0., inds_target=self.inds_pieces_scramble)
            self.stat_masks_full[k] = ut.rearrange_pieces(self.stat_masks_full[k], inds_pieces=self.inds_pieces_in_scrambled,
                                                          hole_lens=None, fill_hole_with=False, inds_target=self.inds_pieces_scramble)
            self.stat_vars_full[k] = ut.rearrange_pieces(self.stat_vars_full[k], inds_pieces=self.inds_pieces_in_scrambled,
                                                         hole_lens=None, fill_hole_with=0., inds_target=self.inds_pieces_scramble)
            self.corrections_full[k] = ut.rearrange_pieces(self.corrections_full[k], inds_pieces=self.inds_pieces_in_scrambled,
                                                           hole_lens=None, fill_hole_with=0., inds_target=self.inds_pieces_scramble)
            self.statistics_gaussianized_full[k] = ut.rearrange_pieces(self.statistics_gaussianized_full[k],
                                                                       inds_pieces=self.inds_pieces_in_scrambled, hole_lens=None,
                                                                       fill_hole_with=0., inds_target=self.inds_pieces_scramble)
        self.inds_pieces_scramble = None
        self.inds_pieces_in_scrambled = None
        self.tot_hole_len_scramble = None
        self.is_scrambled = False
        self.is_shifts_interleaved = False

    def interleave_shifts(self, delete_fulls=True):
        """
        Takes statistics of the same template with different shifts and interleaves them in the alternating
        manner to create continuity of sliding of this template over the data. The resulting array will have the length
        equal to the sum of the length of all the arrays for different shifts.
        :param delete_fulls: Whether to delete full statistics after interleaving them
        :type delete_fulls: bool
        """
        if self.statistics_full is None:
            self.connect_quarters()

        self.interleaved_numbers = np.array([i for i in range(len(self.indices_of_templates))
                                             if ((self.indices_of_templates[i][1] - self.indices_of_templates[i][0]) >= 2)])
        self.not_interleaved_numbers = np.array([i for i in range(len(self.indices_of_templates))
                                                 if ((self.indices_of_templates[i][1] - self.indices_of_templates[i][0]) == 1)])
        # For templates with subbin shifts which get interleaved:
        if len(self.interleaved_numbers) > 0:
            shift_numbers = np.array(
                [self.indices_of_templates[i][1] - self.indices_of_templates[i][0] for i in self.interleaved_numbers])
            # Number of shifts for all the templates with shifts. need them to be the same:
            self.n_shifts = shift_numbers[0]
            # Make sure they are all the same number of shifts:
            assert ((shift_numbers / shift_numbers[0]) == np.ones(len(shift_numbers))
                    ).all(), 'Templates with shifts must have the same number of shifts!'
            self.statistics_interleaved = []
            self.stat_masks_interleaved = []
            self.stat_vars_interleaved = []
            self.corrections_interleaved = []
            self.statistics_gaussianized_interleaved = []
            for t in self.interleaved_numbers:
                self.statistics_interleaved.append(
                    (self.statistics_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
                self.stat_masks_interleaved.append(
                    (self.stat_masks_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
                self.stat_vars_interleaved.append(
                    (self.stat_vars_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
                self.corrections_interleaved.append(
                    (self.corrections_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
                self.statistics_gaussianized_interleaved.append(
                    (self.statistics_gaussianized_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
            self.statistics_interleaved = np.array(self.statistics_interleaved, dtype=Dtype_r)
            self.stat_masks_interleaved = np.array(self.stat_masks_interleaved, dtype=bool)
            self.stat_vars_interleaved = np.array(self.stat_vars_interleaved, dtype=Dtype_r)
            self.corrections_interleaved = np.array(self.corrections_interleaved, dtype=Dtype_r)
            self.statistics_gaussianized_interleaved = np.array(self.statistics_gaussianized_interleaved, dtype=Dtype_r)
            self.time_interleaved = np.arange(len(self.statistics_interleaved[0])) * self.dt_days / self.n_shifts + self.time_full[0]

        # For templates with no subbin shifts which are not interleaved:
        if len(self.not_interleaved_numbers) > 0:
            self.statistics_not_interleaved = []
            self.stat_masks_not_interleaved = []
            self.stat_vars_not_interleaved = []
            self.corrections_not_interleaved = []
            self.statistics_gaussianized_not_interleaved = []
            indices_for_not_interleaved = [self.indices_of_templates[i][0] for i in self.not_interleaved_numbers]
            self.corrections_not_interleaved = np.array(self.corrections_full[indices_for_not_interleaved], dtype=Dtype_r)
            self.statistics_not_interleaved = np.array(self.statistics_full[indices_for_not_interleaved], dtype=Dtype_r)
            self.stat_masks_not_interleaved = np.array(self.stat_masks_full[indices_for_not_interleaved], dtype=bool)
            self.stat_vars_not_interleaved = np.array(self.stat_vars_full[indices_for_not_interleaved], dtype=Dtype_r)
            self.statistics_gaussianized_not_interleaved = np.array(self.statistics_gaussianized_full[indices_for_not_interleaved],
                                                                    dtype=Dtype_r)
            self.time_not_interleaved = np.array(np.arange(len(self.statistics_not_interleaved[0])) * self.dt_days
                                                 + self.time_full[0], dtype=Dtype_r)
        self.is_shifts_interleaved = True
        if delete_fulls:
            del self.time_full
            self.time_full = None
            del self.statistics_full
            self.statistics_full = None
            del self.stat_vars_full
            self.stat_vars_full = None
            del self.stat_masks_full
            self.stat_masks_full = None
            del self.corrections_full
            self.corrections_full = None
            del self.statistics_gaussianized_full
            self.statistics_gaussianized_full = None

    def delete_irrelevant(self):
        """
        Deletes fields which were used in conditioning or single-event search
        and are not needed in the periodicity search
        """
        names_delete = [
            'trends_low',
            'clean_fluxes',
            'clean_masks',
            'whitened_clean_fluxes_fd',
            'whitened_clean_fluxes_td',
            'whitened_templates_fd',
            'whitened_templates_td',
            'notch_soss',
            'wf_td',
            'wf_fd',
            'psd_masks',
            'orig_psds',
            'wf_fds_meas',
            'wf_fds_unwindowed',
            'unnotched_whitened_clean_fluxes_fd',
            'npsds0',
            'npsds',
            'notch_mask',
            'tb',
            'stats_raw_no_veto_no_score_cor',
            'veto_masks',
            'veto_bit_masks',
            'trigger_indices_padded',
            'trigger_indices_unpadded'
        ]
        for attr_name in names_delete:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
                setattr(self, attr_name, None)

    def make_periods_array(self, p_min=None, p_max=None, dp=None, duty_cycle=Duty_cycle, t_mission=None):
        """
        Wrapper for utils.make_periods_array
        Creates the period grid that will be used for folding. The grid is built according to the duty cycle to cover
        the periodicities smoothly.

        :param p_min: Minimal period, days
        :type p_min: float, days
        :param p_max: Maximal period to consider.
        :type p_max: float, days
        :param dp: Step in the list of all the investigated periods
        :type dp: float, days
        :param duty_cycle:
        :type duty_cycle: float
        :param t_mission: total duration between the beginning and the end of the data
        :type t_mission: float, days
        :return: periods, one-dimensional period grid
        :rtype: numpy.ndarray 1d
        """
        if t_mission is None:
            t_mission = self.t_mission
        if p_min is None:
            p_min = self.pmin
        if p_max is None:
            p_max = self.pmax
        return ut.make_periods_array(p_min=p_min, p_max=p_max, dp=dp, duty_cycle=duty_cycle, t_mission=t_mission)

    def make_periods_edges(self, per_phases_per_chunk, p_min=None, p_max=None, dp=None, duty_cycle=Duty_cycle, dt=None):
        """
        Creates edges of period chunks for which the periodicity search will be performed separately.
        This is needed because of memory restrictions per core. Returns the edges of the chunks.
        :param per_phases_per_chunk: The total number of period-phase bins which should be in one chunk.
            Is set by memory estimation. Number of phases is different for every period, therefore chunks will have
            different number of periods in them.
        :type per_phases_per_chunk: int
        :param p_min: Minimal period in the search. If None, `self.pmin` is used
        :type p_min: float or None
        :param p_max: Maximal period in the search. If None, `self.pmax` is used
        :type p_max: float or None
        :param dp: Period step. If None, will be calculated from the `duty_cycle`. Usually is always None.
        :type dp: float or None
        :param duty_cycle: Maximal accumulated deviation due to the period error in the end of the mission.
            In units of days.
        :type duty_cycle: float
        :param dt: Time step of the data. If None, `self.dt_days` will be used
        :type dt: float or None
        :return: per_edges, list of pairs of periods: the beginning and the end of every chunk.
        :rtype: list[list[float, float]]
        """
        if p_min is None:
            p_min = self.pmin
        if p_max is None:
            p_max = self.pmax
        if dt is None:
            dt = self.dt_days
        return ut.make_periods_edges(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk,
                                     dp=dp, duty_cycle=duty_cycle, dt=dt)

    def make_periods_chunks(self, per_phases_per_chunk, duty_cycle, p_min=None, p_max=None, dt=None, t_mission=None):
        """
        Creates the period grid that will be used for periodicity search and splits it into chunks.
        The grid is built according to the duty cycle to cover the periodicities smoothly.
        Splitting into chunks is needed because of memory restrictions per core.
        :param per_phases_per_chunk: The total number of period-phase bins which should be in one chunk.
            Is set by memory estimation. Number of phases is different for every period, therefore chunks will have
            different number of periods in them.
        :type per_phases_per_chunk: int
        :param p_min: Minimal period in the search. If None, `self.pmin` is used
        :type p_min: float or None
        :param p_max: Maximal period in the search. If None, `self.pmax` is used
        :type p_max: float or None
        :param duty_cycle: Maximal accumulated deviation due to the period error in the end of the mission.
            In units of days.
        :type duty_cycle: float
        :param dt: Time step of the data. If None, `self.dt_days` will be used
        :type dt: float or None
        :param t_mission: Total duration of the mission in days. If None, `self.t_mission` will be used
        :type t_mission: float
        :return: list of chunks of small period grids
        :rtype: list of np.ndarray 1d
        """
        if p_min is None:
            p_min = self.pmin
        if p_max is None:
            p_max = self.pmax
        if t_mission is None:
            t_mission = self.t_mission
        if dt is None:
            dt = self.dt_days
        return ut.make_periods_chunks(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk,
                                      duty_cycle=duty_cycle, dt=dt, t_mission=t_mission)

    def sum_periods(self, p_min=None, p_max=None, dp=None, duty_cycle=Duty_cycle, periods=None, safety_bins=P_bins_overlap):
        """
        Sums the statistics with all the possible periods and initial phases. Folds the statistics and their
        variances and calculates the score of the folded planet. Folds also single-event distribution corrections
        and the gaussianized scores. The masks should be accounted for earlier. The masked statistics and variances
        should be set to zero.
        The search is done separately for the interleaved templates and the templates which do not require interleaving.
        These have different sizes and cannot be put to the same numpy.ndarray. Whereas using numpy.ndarray is
        crucial for the computational efficiency.

        Creates:
        self._phases_interleaved
        self.statistics_periodic_interleaved
        self.corrections_periodic_interleaved
        self.stats_gaussianized_periodic_interleaved
        and the same for non-interleaved

        :param p_min: Minimal period in the search. If None, `self.pmin` is used. Overriden by `periods`
        :type p_min: float or None
        :param p_max: Maximal period in the search. If None, `self.pmax` is used. Overriden by `periods`
        :type p_max: float or None
        :param dp: Period step. If None, will be calculated from the `duty_cycle`. Usually is always None.
        :type dp: float or None
        :param duty_cycle: Maximal accumulated deviation due to the period error in the end of the mission.
            In units of days. Overriden by `periods`.
        :type duty_cycle: float
        :param periods: Specifies all the periods, and then the previous parameters are not needed.
            Overrides `p_min`, `p_max`, `dp`, `duty_cycle`
        :type periods: numpy.ndarray of float
        :param safety_bins: Number of spare period bins added to the periods array for the best trigger
            not to be at the boundary of the array. This is needed because the integral score will not be full
            if the trigger is on the boundary.
        :type safety_bins: int
        """
        if duty_cycle is None:
            duty_cycle = Duty_cycle

        if periods is None:
            self.pmin = p_min
            self.pmax = p_max
            periods = self.make_periods_array(p_min=p_min, p_max=p_max, dp=dp, duty_cycle=duty_cycle, t_mission=None)
        else:
            self.pmin = periods[0]
            self.pmax = periods[-1]
        if p_max is None:
            p_max = periods[-1]

        self.period_overlap_bins = safety_bins
        if safety_bins > 0:
            periods = ut.add_safety_overlap(periods, safety_bins=safety_bins)
        self.periods = periods
        if not self.is_shifts_interleaved:
            self.interleave_shifts()

        self._phases_interleaved = []
        self._phases_not_interleaved = []
        self.statistics_periodic_not_interleaved = []
        self.statistics_periodic_interleaved = []
        self.corrections_periodic_interleaved = []
        self.corrections_periodic_not_interleaved = []
        self.stats_gaussianized_periodic_interleaved = []
        self.stats_gaussianized_periodic_not_interleaved = []
        # Attention: we use the same dt for interleaved and not? it orig_means that the phases are the same? phases are defined further.

        # interleaved
        if len(self.interleaved_numbers) > 0:
            dt = self.dt_days / self.n_shifts
            len_pad = int(np.ceil(p_max / dt))
            self._phases_interleaved = np.arange(int(np.ceil((self.periods[-1]) / dt))) * dt
            # # These phases are with respect to the beginning of the data which can be any
            # # number because sometimes we reject the zeroth quarter or do other things.
            # # Do not use these phases anywhere.
            # # Warning: we only try these phases and not something in-between. It should be ok, but still
            # statistics_by_trends = np.array(self.statistics_interleaved, dtype=Dtype_r)  # * self.trend_orig_flux_interleaved
            # # -> trends are now all normalized to be the same!
            # # Pad by zeros to avoid extra if statements in the summation:
            # statistics_by_trends = np.hstack((statistics_by_trends, np.zeros((len(statistics_by_trends), len_pad)))).astype(Dtype_r)
            # trends_sq_by_vars = np.array(self.stat_vars_interleaved, dtype=Dtype_r)  # * self.trend_orig_flux_interleaved ** 2
            # # -> trends are all normalized to be the same!
            # trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros((len(trends_sq_by_vars), len_pad)))).astype(Dtype_r)
            # corrections = np.array(self.corrections_interleaved, dtype=Dtype_r)
            # corrections = np.hstack((corrections, np.zeros((len(corrections), len_pad)))).astype(Dtype_r)
            # statistics_by_trends_gaussianized = np.array(self.statistics_gaussianized_interleaved, dtype=Dtype_r)
            # statistics_by_trends_gaussianized = np.hstack((statistics_by_trends_gaussianized,
            #                                                np.zeros((len(statistics_by_trends_gaussianized), len_pad)))).astype(Dtype_r)

            # Packing the inputs into a single array to speed up the summation:
            # inputs_array = np.vstack([statistics_by_trends, trends_sq_by_vars, corrections, statistics_by_trends_gaussianized])

            inputs_array = np.vstack([np.hstack((self.statistics_interleaved,
                                                 np.zeros((len(self.statistics_interleaved), len_pad)))).astype(Dtype_r),
                                      np.hstack((self.stat_vars_interleaved,
                                                 np.zeros((len(self.stat_vars_interleaved), len_pad)))).astype(Dtype_r),
                                      np.hstack((self.corrections_interleaved,
                                                 np.zeros((len(self.corrections_interleaved), len_pad)))).astype(Dtype_r),
                                      np.hstack((self.statistics_gaussianized_interleaved,
                                                 np.zeros((len(self.statistics_gaussianized_interleaved), len_pad)))).astype(Dtype_r)
                                      ])
            inputs_array = np.transpose(inputs_array)
            inputs_array = np.ascontiguousarray(inputs_array)

            tmin = Dtype_timings(self.tmin)
            tmax = Dtype_timings(self.tmax + self.dt_days * (1 - 1 / self.n_shifts))
            dt = Dtype_timings(dt)
            periods = np.array(periods, dtype=Dtype_timings)

            folded_results = self.sum_periods_fast(
                inputs_array, dt=dt, tmax=tmax, tmin=tmin, periods=periods)

            # Unpacking the results:
            n_templates = len(self.statistics_interleaved)
            # results_array = np.zeros((len_periods, phases_size, res_temp_size), dtype=Dtype_r)
            self.statistics_periodic_interleaved = folded_results[:, :, 0:n_templates].transpose((0, 2, 1))
            denomsq = folded_results[:, :, n_templates:2 * n_templates].transpose((0, 2, 1))
            self.corrections_periodic_interleaved = folded_results[:, :, 2 * n_templates:3 * n_templates].transpose((0, 2, 1))
            self.stats_gaussianized_periodic_interleaved = folded_results[:, :, 3 * n_templates:4 * n_templates].transpose((0, 2, 1))

            del folded_results

            # self.statistics_periodic_interleaved, denomsq, self.corrections_periodic_interleaved, \
            #     self.stats_gaussianized_periodic_interleaved = self.sum_periods_fast(
            #         statistics_by_trends, trends_sq_by_vars, corrections, statistics_by_trends_gaussianized,
            #         dt, self.tmax + self.dt_days * (1 - 1 / self.n_shifts), self.tmin, periods)
            # attention: we use the same dt for interleaved and not? it orig_means that the phase step for the interleaved could be smaller.
            # for p in range(len(self.statistics_periodic_interleaved)):
            #     for t in range(len(self.statistics_periodic_interleaved[p])):
            #         self.statistics_periodic_interleaved[p][t][denomsq[p][t] == 0] = np.nan
            #         self.statistics_periodic_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][denomsq[p][t] != 0] ** 0.5
            #         self.stats_gaussianized_periodic_interleaved[p][t][denomsq[p][t] == 0] = np.nan
            #         self.stats_gaussianized_periodic_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][denomsq[p][t] != 0] ** 0.5
            self.statistics_periodic_interleaved, self.stats_gaussianized_periodic_interleaved = self.normalize_stats_fast(
                statistics_periodic=self.statistics_periodic_interleaved,
                stats_gaussianized_periodic=self.stats_gaussianized_periodic_interleaved,
                denomsq=denomsq)

    # -----------------------------------------------------------------------------------------------------------------------
        # Not interleaved templates (which did not require subbin shifts)
        dt = self.dt_days
        len_pad = int(np.ceil(p_max / dt))
        if len(self.not_interleaved_numbers) > 0:
            self._phases_not_interleaved = np.arange(int(np.ceil((self.periods[-1]) / dt))) * dt
            # statistics_by_trends = np.array(self.statistics_not_interleaved, dtype=Dtype_r)  # * self.trend_orig_flux_not_interleaved
            # statistics_by_trends = np.hstack((statistics_by_trends, np.zeros((len(statistics_by_trends), len_pad)))).astype(Dtype_r)
            # # These phases are with respect to the beginning of the data which can be any
            # # number because sometimes we reject the zeroth quarter or do other things.
            # # Do not use these phases anywhere.
            # # Pad by zeros to avoid extra if statements in the summation:
            # trends_sq_by_vars = np.array(self.stat_vars_not_interleaved, dtype=Dtype_r)  # * self.trend_orig_flux_not_interleaved ** 2
            # trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros((len(trends_sq_by_vars), len_pad)))).astype(Dtype_r)
            # corrections = np.array(self.corrections_not_interleaved, dtype=Dtype_r)
            # corrections = np.hstack((corrections, np.zeros((len(corrections), len_pad)))).astype(Dtype_r)
            # statistics_by_trends_gaussianized = np.array(self.statistics_gaussianized_not_interleaved, dtype=Dtype_r)
            # statistics_by_trends_gaussianized = np.hstack((statistics_by_trends_gaussianized, np.zeros(
            #     (len(statistics_by_trends_gaussianized), len_pad)))).astype(Dtype_r)

            # Packing the inputs into a single array to speed up the summation:
            # inputs_array = np.vstack([statistics_by_trends, trends_sq_by_vars, corrections, statistics_by_trends_gaussianized])

            inputs_array = np.vstack([np.hstack((self.statistics_not_interleaved,
                                                 np.zeros((len(self.statistics_not_interleaved), len_pad)))).astype(Dtype_r),
                                      np.hstack((self.stat_vars_not_interleaved,
                                                 np.zeros((len(self.stat_vars_not_interleaved), len_pad)))).astype(Dtype_r),
                                      np.hstack((self.corrections_not_interleaved,
                                                 np.zeros((len(self.corrections_not_interleaved), len_pad)))).astype(Dtype_r),
                                      np.hstack((self.statistics_gaussianized_not_interleaved,
                                                 np.zeros((len(self.statistics_gaussianized_not_interleaved), len_pad)))).astype(Dtype_r)
                                      ])
            inputs_array = np.transpose(inputs_array)
            inputs_array = np.ascontiguousarray(inputs_array)

            tmin = Dtype_timings(self.tmin)
            tmax = Dtype_timings(self.tmax)
            dt = Dtype_timings(dt)
            periods = np.array(periods, dtype=Dtype_timings)

            folded_results = self.sum_periods_fast(
                inputs_array, dt=dt, tmax=tmax, tmin=tmin, periods=periods)

            # Unpacking the results:
            n_templates = len(self.statistics_not_interleaved)
            # results_array = np.zeros((len_periods, phases_size, res_temp_size), dtype=Dtype_r)
            self.statistics_periodic_not_interleaved = folded_results[:, :, 0:n_templates].transpose((0, 2, 1))
            denomsq = folded_results[:, :, n_templates:2 * n_templates].transpose((0, 2, 1))
            self.corrections_periodic_not_interleaved = folded_results[:, :, 2 * n_templates:3 * n_templates].transpose((0, 2, 1))
            self.stats_gaussianized_periodic_not_interleaved = folded_results[:, :, 3 * n_templates:4 * n_templates].transpose((0, 2, 1))

            del folded_results

            # self.statistics_periodic_not_interleaved, denomsq, self.corrections_periodic_not_interleaved, \
            #     self.stats_gaussianized_periodic_not_interleaved = self.sum_periods_fast(
            #         statistics_by_trends, trends_sq_by_vars, corrections, statistics_by_trends_gaussianized,
            #         dt, self.tmax, self.tmin, periods)
            # for p in range(len(self.statistics_periodic_not_interleaved)):
            #     for t in range(len(self.statistics_periodic_not_interleaved[p])):
            #         self.statistics_periodic_not_interleaved[p][t][denomsq[p][t] == 0] = np.nan
            #         self.statistics_periodic_not_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][denomsq[p][t] != 0] ** 0.5
            #         self.stats_gaussianized_periodic_not_interleaved[p][t][denomsq[p][t] == 0] = np.nan
            #         self.stats_gaussianized_periodic_not_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][denomsq[p][t] != 0] ** 0.5

            self.statistics_periodic_not_interleaved, self.stats_gaussianized_periodic_not_interleaved = self.normalize_stats_fast(
                statistics_periodic=self.statistics_periodic_not_interleaved,
                stats_gaussianized_periodic=self.stats_gaussianized_periodic_not_interleaved,
                denomsq=denomsq)

    def save_statistics_periodic(self, output_location=None, filename_postfix=Periodic_stats_filename_postfix, specify_periods=True):
        if output_location is None:
            output_location = self.output_location
        ftts_interleaved = self._phases_interleaved + self.tmin
        ftts_not_interleaved = self._phases_not_interleaved + self.tmin
        arr_save = np.array([self.periods, ftts_interleaved, ftts_not_interleaved,
                             self.statistics_periodic_interleaved, self.statistics_periodic_not_interleaved,
                             self.corrections_periodic_interleaved, self.corrections_periodic_not_interleaved], dtype=object)
        file_path = af.get_saving_file_path(target_id=self.target_id, output_location=output_location,
                                            filename_postfix=filename_postfix,
                                            insert_integer_periods=[self.periods[0], self.periods[-1]] if specify_periods else None,
                                            insert_after_postfix=None,
                                            file_format='npy')
        np.save(file_path, arr_save, allow_pickle=True)

    def downsample_interleaved(self, delete_periodic_interleaved=True):
        """
        Downsamples the resolution of the interleaved templates to match them in their size to the
        templates which did not require interleaving. This is needed for the template marginalization,
        where the arrays should have the same size in order to be summed.
        The downsampling is done in such a way that from every coarse-grained sell the minimal (best)
        statistic is selected.
        Preferred over `interpolate_not_interleaved` because it is less memory.
        Replaces:
        self._phases_interleaved -> self._phases
        self.statistics_periodic_interleaved -> self.statistics_periodic
        self.corrections_periodic_interleaved -> self.corrections_periodic
        self.stats_gaussianized_periodic_interleaved -> self.stats_gaussianized_periodic
        and the same for non-interleaved
        :param delete_periodic_interleaved: whether to delete the previous arrays with interleaved and
            non-interleaved templates, or no
        :type delete_periodic_interleaved: bool
        """
        n_interl = np.shape(self.statistics_periodic_interleaved)[1]
        self.statistics_periodic = np.zeros((np.shape(self.statistics_periodic_not_interleaved)[0],
                                             n_interl + np.shape(self.statistics_periodic_not_interleaved)[1],
                                             np.shape(self.statistics_periodic_not_interleaved)[2]), dtype=Dtype_r)
        self.stats_gaussianized_periodic = np.zeros_like(self.statistics_periodic, dtype=Dtype_r)
        self.corrections_periodic = np.zeros_like(self.statistics_periodic, dtype=Dtype_r)

        not_interl_inds = n_interl + np.arange(np.shape(self.statistics_periodic_not_interleaved)[1])
        # self.statistics_periodic[:, not_interl_inds, :] = np.array(self.statistics_periodic_not_interleaved, dtype=Dtype_r)
        # self.stats_gaussianized_periodic[:, not_interl_inds, :] = np.array(
        #                       self.stats_gaussianized_periodic_not_interleaved, dtype=Dtype_r)
        # self.corrections_periodic[:, not_interl_inds, :] = np.array(self.corrections_periodic_not_interleaved, dtype=Dtype_r)
        # This did not make it really faster
        self.statistics_periodic[:, not_interl_inds, :] = self.statistics_periodic_not_interleaved
        self.stats_gaussianized_periodic[:, not_interl_inds, :] = self.stats_gaussianized_periodic_not_interleaved
        self.corrections_periodic[:, not_interl_inds, :] = self.corrections_periodic_not_interleaved
        self._phases = np.array(self._phases_not_interleaved)

        n_shifts = int(np.round(len(self._phases_interleaved) / len(self._phases_not_interleaved)))
        len_not_interl = len(self._phases_not_interleaved)
        delta_len = n_shifts * len_not_interl - len(self._phases_interleaved)

        vals_downsampled, vals_interl = [self.statistics_periodic, self.statistics_periodic_interleaved]
        selecting_inds = np.zeros((np.shape(self.statistics_periodic_interleaved)[0], np.shape(self.statistics_periodic_interleaved)[1],
                                   len_not_interl)).astype(int)
        # Selecting indices which give the best value in each fine-sampled chunk
        ind_middle = int(n_shifts // 2)
        for ip in range(len(vals_interl)):  # over periods
            for k in range(len(vals_interl[ip])):
                res = np.concatenate((vals_interl[ip][k], np.ones(delta_len, dtype=Dtype_r) * vals_interl[ip][k][-1]))
                res = (res.reshape((len_not_interl, n_shifts))).transpose()
                inds_notnan = np.max((~np.isnan(res)).astype(float), axis=0).astype(bool)
                res_1 = (ind_middle * np.ones(res.shape[1])).astype(int)  # if there are only nans, take the middle
                # Attention: change here min to max if you change the definitions
                res_1[inds_notnan] = np.nanargmin(res[:, inds_notnan], axis=0)
                selecting_inds[ip][k] = np.arange(len_not_interl) * n_shifts + res_1

        for vals_downsampled, vals_interl in [[self.statistics_periodic, self.statistics_periodic_interleaved],
                                              [self.stats_gaussianized_periodic, self.stats_gaussianized_periodic_interleaved],
                                              [self.corrections_periodic, self.corrections_periodic_interleaved]]:
            for ip in range(len(vals_interl)):  # Over periods
                for k in range(len(vals_interl[ip])):  # Over interleaved templates
                    res = np.concatenate((vals_interl[ip][k], np.zeros(delta_len, dtype=Dtype_r)))
                    vals_downsampled[ip, k] = res[selecting_inds[ip][k]]

        if delete_periodic_interleaved:
            del self.statistics_periodic_interleaved
            self.statistics_periodic_interleaved = None
            del self.statistics_periodic_not_interleaved
            self.statistics_periodic_not_interleaved = None
            del self.stats_gaussianized_periodic_interleaved
            self.stats_gaussianized_periodic_interleaved = None
            del self.stats_gaussianized_periodic_not_interleaved
            self.stats_gaussianized_periodic_not_interleaved = None
            del self._phases_interleaved
            self._phases_interleaved = None
            del self._phases_not_interleaved
            self._phases_not_interleaved = None

    def interpolate_not_interleaved(self, delete_periodic_interleaved=True):
        """
        Interpolates (upsamples) the resolution of the templates which did not require interleaving to
        match them in their size to the interleaved templates. This is needed for the template marginalization,
        where the arrays should have the same size in order to be summed.
        Method `downsample_interleaved` is preferred over this method because it is less memory.
        Replaces:
        self._phases_interleaved -> self._phases
        self.statistics_periodic_interleaved -> self.statistics_periodic
        self.corrections_periodic_interleaved -> self.corrections_periodic
        self.stats_gaussianized_periodic_interleaved -> self.stats_gaussianized_periodic
        and the same for non-interleaved
        :param delete_periodic_interleaved: whether to delete the previous arrays with interleaved and
            non-interleaved templates, or no
        :type delete_periodic_interleaved: bool
        """
        n_interl = np.shape(self.statistics_periodic_interleaved)[1]
        self.statistics_periodic = np.zeros((np.shape(self.statistics_periodic_interleaved)[0],
                                             n_interl + np.shape(self.statistics_periodic_not_interleaved)[1],
                                             np.shape(self.statistics_periodic_interleaved)[2]), dtype=Dtype_r)
        self.stats_gaussianized_periodic = np.zeros_like(self.statistics_periodic, dtype=Dtype_r)
        self.corrections_periodic = np.zeros_like(self.statistics_periodic, dtype=Dtype_r)

        self.statistics_periodic[:, range(n_interl), :] = np.array(self.statistics_periodic_interleaved, dtype=Dtype_r)
        self.stats_gaussianized_periodic[:, range(n_interl), :] = np.array(self.stats_gaussianized_periodic_interleaved, dtype=Dtype_r)
        self.corrections_periodic[:, range(n_interl), :] = np.array(self.corrections_periodic_interleaved, dtype=Dtype_r)
        self._phases = np.array(self._phases_interleaved)

        for vals_interpolated, vals_not_interl in [[self.statistics_periodic, self.statistics_periodic_not_interleaved],
                                                   [self.stats_gaussianized_periodic, self.stats_gaussianized_periodic_not_interleaved],
                                                   [self.corrections_periodic, self.corrections_periodic_not_interleaved]]:

            for ip in range(len(vals_not_interl)):  # over periods
                for k in range(len(vals_not_interl[ip])):
                    res = np.interp(x=self._phases_interleaved, xp=self._phases_not_interleaved, fp=vals_not_interl[ip][k])
                    vals_interpolated[ip, n_interl + k] = res

        if delete_periodic_interleaved:
            del self.statistics_periodic_interleaved
            self.statistics_periodic_interleaved = None
            del self.statistics_periodic_not_interleaved
            self.statistics_periodic_not_interleaved = None
            del self.statistics_gaussianized_interleaved
            self.statistics_gaussianized_interleaved = None
            del self.statistics_gaussianized_not_interleaved
            self.statistics_gaussianized_not_interleaved = None
            del self.corrections_periodic_interleaved
            self.corrections_periodic_interleaved = None
            del self.corrections_periodic_not_interleaved
            self.corrections_periodic_not_interleaved = None
            del self._phases_interleaved
            self._phases_interleaved = None
            del self._phases_not_interleaved
            self._phases_not_interleaved = None

    def delete_mes(self):
        """
        Deletes the periodicity search data. Used to clear everything after the periodicity search in time
        slides and run from the beginning.
        """
        if hasattr(self, 'periods'):
            del self.periods
            self.periods = None
        if hasattr(self, '_phases_interleaved'):
            del self._phases_interleaved
            self._phases_interleaved = None
        if hasattr(self, '_phases_not_interleaved'):
            del self._phases_not_interleaved
            self._phases_not_interleaved = None
        if hasattr(self, 'selfistics_periodic_not_interleaved'):
            del self.statistics_periodic_not_interleaved
            self.statistics_periodic_not_interleaved = None
        if hasattr(self, 'statistics_periodic_interleaved'):
            del self.statistics_periodic_interleaved
            self.statistics_periodic_interleaved = None
        if hasattr(self, 'stats_gaussianized_periodic_interleaved'):
            del self.stats_gaussianized_periodic_interleaved
            self.stats_gaussianized_periodic_interleaved = None
        if hasattr(self, 'stats_gaussianized_periodic_not_interleaved'):
            del self.stats_gaussianized_periodic_not_interleaved
            self.stats_gaussianized_periodic_not_interleaved = None
        if hasattr(self, 'corrections_periodic_interleaved'):
            del self.corrections_periodic_interleaved
            self.corrections_periodic_interleaved = None
        if hasattr(self, 'corrections_periodic_not_interleaved'):
            del self.corrections_periodic_not_interleaved
            self.corrections_periodic_not_interleaved = None
        if hasattr(self, 'statistics_periodic'):
            del self.statistics_periodic
            self.statistics_periodic = None
        if hasattr(self, 'corrections_periodic'):
            del self.corrections_periodic
            self.corrections_periodic = None
        if hasattr(self, 'stats_gaussianized_periodic'):
            del self.stats_gaussianized_periodic
            self.stats_gaussianized_periodic = None
        if hasattr(self, 'stats_periodic_mar_temp'):
            del self.stats_periodic_mar_temp
            self.stats_periodic_mar_temp = None
        if hasattr(self, 'stats_gaussianized_periodic_mar_temp'):
            del self.stats_gaussianized_periodic_mar_temp
            self.stats_gaussianized_periodic_mar_temp = None
        if hasattr(self, 'stats_periodic_mar_temp_best_ph'):
            del self.stats_periodic_mar_temp_best_ph
            self.stats_periodic_mar_temp_best_ph = None
        if hasattr(self, 'stats_gaussianized_periodic_mar_temp'):
            del self.stats_gaussianized_periodic_mar_temp
            self.stats_gaussianized_periodic_mar_temp = None
        self.is_periodic_scores_corrected = False

    def calculate_periodic_scores_with_correction(self, save_uncorrected=False, force_even_if_corrected=False):
        """
        Applies the single-event statistic distribution non-gaussianity correction to the periodic scores.
        These corrections show how much more probable it is for the statistics to originate from the
        measured non-gaussian distribution rather than from the Gaussian distribution. Applying this correction
        enables the hypothesis testing to compare the planet hypothesis to the hypothesis of the non-gaussian noise.
        The correction is added to the squared statistics. This means that the statistics forget theur sign. In the
        planet search, the sign is crucial, and we are not interested in the positive deviations of the flux. Therefore,
        before applying the corrections, the signs are saved and then reapplied.
        If the correction is too large so that it completely kills the score and even makes the square snr negative,
        it is set to zero.
        Note about renormalization. We already made sure with the SES that the distribution is okay.
        Then, the periodicity summation should also be okay if the variances are correct
        :param save_uncorrected: Whether to save the statistics before applying the correction
        :type save_uncorrected: bool
        :param force_even_if_corrected:
        :type force_even_if_corrected:
        """
        zero = Dtype_r(0.)
        if (not self.is_periodic_scores_corrected) or force_even_if_corrected:
            if self.statistics_periodic is not None:
                if save_uncorrected:
                    self.uncorrected_stats_periodic = copy.deepcopy(self.statistics_periodic)
                for i in range(len(self.statistics_periodic)):  # Loop is used instead of array because of memory concerns
                    stats = self.statistics_periodic[i]
                    signs = np.sign(stats)
                    stats_squared = stats ** 2 + 2 * self.corrections_periodic[i]
                    stats_squared[stats_squared < 0] = 0.
                    self.statistics_periodic[i] = stats_squared ** 0.5 * signs
            else:
                if save_uncorrected:
                    self.uncorrected_stats_periodic = [copy.deepcopy(self.statistics_periodic_interleaved),
                                                       copy.deepcopy(self.statistics_periodic_not_interleaved)]
                for stats_periodic, cors_periodic in [[self.statistics_periodic_interleaved, self.corrections_periodic_interleaved],
                                                      [self.statistics_periodic_not_interleaved, self.corrections_periodic_not_interleaved]
                                                      ]:
                    for i in range(len(stats_periodic)):
                        stats = stats_periodic[i]
                        signs = np.sign(stats).astype(Dtype_r)
                        stats_squared = stats ** 2 + 2 * cors_periodic[i]
                        stats_squared[stats_squared < 0] = zero
                        stats_periodic[i] = stats_squared ** 0.5 * signs

        self.is_periodic_scores_corrected = True

    def marginalize_over_templates(self, m_1_3_r_s=None, delete_non_marginalized=False):
        """
        Marginalizes the periodic score over templates. Takes exponents of the snr squared, applies the template prior and takes sum.
        After this, log is taken back, so that we are again in units of snr squared and not likelihood.
        Attention! we sum only negative things, and they are squared. This does not have any N(0,1) distribution in the end.
        Only the negative things are summed because snr squared does not have any information about the sign, and we are only
        interested in the negative transits. It is conceptually the same as applying an amplitude prior which is zero where the
        inferred amplitude is positive. So, after this point, all the result includes only the negative transits.
        But the statistic after this is only positive, it is in units of SNR squared. In the end, the prior-weighted scores
        will be lower than the initial scores because the template prior is less than unity. The prior is normalized to unity.
        Creates the following arrays indexed by period and phase indices:
        self.stats_periodic_mar_temp
        self.stats_gaussianized_periodic_mar_temp
        :param m_1_3_r_s: Mass of the star in units of solar mass divided by the radius of the star to the power 1/3.
        :type m_1_3_r_s: float
        :param delete_non_marginalized: Whether to delete the statistics before marginalization
        :type delete_non_marginalized: bool
        """
        # Here, it may be important for the type to be float64 because the template prior may be small.
        if m_1_3_r_s is None:
            m_1_3_r_s = self.m_1_3_r_s
        log_template_prior, m13r_bin_edges, period_edges = tm.load_template_prior()
        m_1_3_r_s_bin = min(max(0, np.searchsorted(m13r_bin_edges, m_1_3_r_s) - 1), len(m13r_bin_edges) - 2)

        self.stats_periodic_mar_temp = np.zeros((np.shape(self.statistics_periodic)[0], np.shape(self.statistics_periodic)[2]),
                                                dtype=Dtype_r)
        self.stats_gaussianized_periodic_mar_temp = np.zeros_like(self.stats_periodic_mar_temp, dtype=Dtype_r)

        for marg_stats, stats in [[self.stats_periodic_mar_temp, self.statistics_periodic],
                                  [self.stats_gaussianized_periodic_mar_temp, self.stats_gaussianized_periodic]]:
            for ip in range(len(self.periods)):
                per_bin = min(max(0, np.searchsorted(period_edges, self.periods[ip]) - 1), len(period_edges) - 2)
                log_template_prior_bin = log_template_prior[m_1_3_r_s_bin, per_bin]
                vals_to_sum = (stats[ip]) ** 2 + np.transpose(np.array([log_template_prior_bin]))
                neg_ind_weights = ((stats[ip] < 0) & (np.abs(vals_to_sum) != np.inf)).astype(float)
                vals_to_sum[np.abs(vals_to_sum) == np.inf] = 0
                inds_notnan = np.max((~np.isnan(vals_to_sum)).astype(float), axis=0).astype(bool)
                maxvals = np.zeros(len(inds_notnan))
                maxvals[inds_notnan] = np.nanmax((vals_to_sum * neg_ind_weights)[:, inds_notnan], axis=0)
                vals_to_sum = vals_to_sum - maxvals
                summed = np.nansum(np.exp(vals_to_sum) * neg_ind_weights, axis=0)
                pos_mask = summed > 0
                summed[pos_mask] = np.log(summed[pos_mask])
                summed[pos_mask] = summed[pos_mask] + maxvals[pos_mask]  # NEW!!! maybe not needed
                summed[~inds_notnan] = np.nan
                marg_stats[ip] = np.array(summed)

        if delete_non_marginalized:
            del self.statistics_periodic
            self.statistics_periodic = None
            del self.stats_gaussianized_periodic
            self.stats_gaussianized_periodic = None

    def maximize_over_phase(self, delete_non_maximized=False):
        """
        Finds one best score per every period (maximizes over phase). This method is not used now because
        multiple-event statistic veto was introduced, therefore when one candidate was rejected, it can be that
        at the same period, there is the second-best candidate which has now to be considered.
        :param delete_non_maximized: Whether to delete the previously used non-maximized arrays.
        :type delete_non_maximized: bool
        """
        # For non-gaussianized (but with ses distribution correction) score:
        best_phase_inds = np.array([np.nanargmax(self.stats_periodic_mar_temp[pp]) for pp in range(len(self.stats_periodic_mar_temp))])
        self.best_phase_inds = best_phase_inds
        self.stats_periodic_mar_temp_best_ph = np.array([np.nanmax(self.stats_periodic_mar_temp[pp])
                                                         for pp in range(len(self.stats_periodic_mar_temp))], dtype=Dtype_r)
        phases_best_ph = np.array([self._phases[best_phase_inds[pp]] for pp in range(len(self.stats_periodic_mar_temp))], dtype=Dtype_r)
        self.stats_gaussianized_periodic_mar_temp_best_ph = np.array([self.stats_gaussianized_periodic_mar_temp[pp][best_phase_inds[pp]]
                                                                      for pp in range(len(self.stats_periodic_mar_temp))], dtype=Dtype_r)
        self.first_transit_times_best_ph = self.tmin + phases_best_ph

        # Same for gaussianized
        best_g_phase_inds = np.array([np.nanargmax(self.stats_gaussianized_periodic_mar_temp[pp])
                                      for pp in range(len(self.stats_gaussianized_periodic_mar_temp))])
        self.best_g_phase_inds = best_g_phase_inds
        self.stats_periodic_mar_temp_best_g_ph = np.array([np.nanmax(self.stats_gaussianized_periodic_mar_temp[pp])
                                                           for pp in range(len(self.stats_gaussianized_periodic_mar_temp))])
        phases_best_g_ph = np.array([self._phases[best_g_phase_inds[pp]] for pp in range(len(self.stats_gaussianized_periodic_mar_temp))])
        self.stats_gaussianized_periodic_mar_temp_best_g_ph = np.array([self.stats_gaussianized_periodic_mar_temp[pp][best_g_phase_inds[pp]]
                                                                        for pp in range(len(self.stats_gaussianized_periodic_mar_temp))],
                                                                       dtype=Dtype_r)
        self.first_transit_times_best_g_ph = self.tmin + phases_best_g_ph

        if delete_non_maximized:
            del self.stats_periodic_mar_temp
            self.stats_periodic_mar_temp = None
            del self.stats_gaussianized_periodic_mar_temp
            self.stats_gaussianized_periodic_mar_temp = None

    def get_integral_score_at_max(self, per_ind_max=None, phase_ind_max=None, gaussianized=False,
                                  dpi=D_period_integral_score, dphi=D_phase_integral_score, m_1_3_r=None,
                                  use_occurrence_rate=True, log_baseline_integral_score_rate=0):
        """
        Calculates the integral score around the best trigger. The integral score is the sum of all the statistics
        in the period-phase plane around the specified maximum. The point should really be a maximum because the
        summing indices are selected by the threshold above the background.
        The integral score is combined with the occurrence rate for this period range, so that when different
        periods are compared, it is done according to the prior. Also, integral is really integral, meaning it
        takes into account the measure: period and phase steps.
        :param per_ind_max: Index in the period grid of the maximal score
        :type per_ind_max: int
        :param phase_ind_max: Index in the phase grid of the maximal score
        :type phase_ind_max: int
        :param gaussianized: Whether to consider the gaussianized score or non-gaussianized, but distribution-corrected
        :type gaussianized: bool
        :param dphi: Size in bins of the region taken for selecting the phase indices participating in the integral score
        :type dphi: int
        :param dpi: Size in bins of the region taken for selecting the period indices participating in the integral score
        :type dpi: int
        :param m_1_3_r: Mass of the star in units of solar mass divided by the radius of the star to the power 1/3
        :type m_1_3_r: float
        :param use_occurrence_rate:
        :type use_occurrence_rate: bool
        :param log_baseline_integral_score_rate: renormalization constant which is to be subtracted from the integral score
        :type log_baseline_integral_score_rate: float
        """
        if m_1_3_r is None:
            m_1_3_r = self.m_1_3_r_s
        if log_baseline_integral_score_rate is None:
            log_baseline_integral_score_rate = 0.

        if not gaussianized:
            if per_ind_max is None:
                per_ind_max = self.best_per_index
            if phase_ind_max is None:
                phase_ind_max = self.best_phase_index
            # peak_val = self.stats_periodic_mar_temp[per_ind_max, phase_ind_max]
            per_ind_left = max(0, per_ind_max - dpi)
            phase_ind_left = max(0, phase_ind_max - dphi)
            per_ind_right = min(per_ind_max + dpi, self.stats_periodic_mar_temp.shape[0])
            phase_ind_right = min(phase_ind_max + dphi, self.stats_periodic_mar_temp.shape[1])
            vals = np.array(self.stats_periodic_mar_temp)
        else:
            if per_ind_max is None:
                per_ind_max = self.best_g_per_index
            if phase_ind_max is None:
                phase_ind_max = self.best_g_phase_index
            # peak_val = self.stats_gaussianized_periodic_mar_temp[per_ind_max, phase_ind_max]
            per_ind_left = max(0, per_ind_max - dpi)
            phase_ind_left = max(0, phase_ind_max - dphi)
            per_ind_right = min(per_ind_max + dpi, self.stats_gaussianized_periodic_mar_temp.shape[0])
            phase_ind_right = min(phase_ind_max + dphi, self.stats_gaussianized_periodic_mar_temp.shape[1])
            vals = np.array(self.stats_gaussianized_periodic_mar_temp)

        vals = vals[per_ind_left:per_ind_right][:, phase_ind_left:phase_ind_right]
        peak_coords_relative = per_ind_max - per_ind_left, phase_ind_max - phase_ind_left

        inds_sum = ut.get_indices_of_influence(vals=vals, ind_center_0=peak_coords_relative[0], ind_center_1=peak_coords_relative[1],
                                               part_peak=0.5)

        # ambient_val = np.nanmedian(vals)
        # inds_sum = (vals > ambient_val + 0.5 * (peak_val - ambient_val))
        # ambient_val = np.nanmedian(vals[~inds_sum])  # repetitive median to exclude peak
        # inds_sum = (vals >= ambient_val + 0.5 * (peak_val - ambient_val))
        # inds_sum = ut.select_connected_component(inds_sum, center=peak_coords_relative)

        dp = self.periods[max(1, per_ind_max)] - self.periods[max(0, per_ind_max - 1)]
        dftt = self._phases[max(1, phase_ind_max)] - self._phases[max(0, phase_ind_max - 1)]
        vals_to_sum = vals[inds_sum]
        vals_to_sum = vals_to_sum[np.abs(vals_to_sum) != np.inf]
        max_val = np.nanmax(vals_to_sum)
        norm_vals_to_sum = vals_to_sum - max_val
        integr_score = max_val + np.log(np.nansum(np.exp(norm_vals_to_sum))) + np.log(dp * dftt)
        # integr_score = np.log(np.nansum(np.exp(vals[inds_sum])) * dp * dftt)

        if use_occurrence_rate:
            # We use density here because we already have dp that was used in multiplication. Now we multiply the result by d_number/dp
            log_occurrence_density_per_dp_dftt = np.log(tm.get_number_of_transiting_planets_density_per_dp(
                                                        period=self.periods[per_ind_max], m_1_3_r_s=m_1_3_r, dp=1.
                                                        ) / self.periods[per_ind_max])
            integr_score = integr_score + log_occurrence_density_per_dp_dftt  # dp dftt are inside the integral
            # log_occurrences_templates_periods is normalized by the m_1_3_r to have the full astrophysical occurrence rate
            # in every m_1_3_r bin. units are rate! no probability normalization! but multiplied by probability to transit.
        integr_score = integr_score - log_baseline_integral_score_rate
        return integr_score

    def get_best_template_index(self, with_prior=True, in_gaussianized=False, ind_period=None, ind_phase=None, period=None):
        """
        Select the template which had the highest score before the marginalization over the templates was done.
        Looks at a specific position specified by `ind_period` and `ind_phase`.
        Attention: the non-marginalized scores should not have been deleted!
        :param with_prior: Whether to select best score accounting for prior or not
        :type with_prior: bool
        :param in_gaussianized: Whether to consider the gaussianized score or non-gaussianized, but distribution-corrected
        :type in_gaussianized: bool
        :param ind_period: The index in periods grid specifying the point where the best template is searched for.
            If None, the best period `self.best_per_index` will be used. Do not use None, se why in the `ind_phase` description.
        :type ind_period: int or None
        :param ind_phase: The index in phases grid specifying the point where the best template is searched for.
            If None, the best phase for the best period `self.best_phase_inds[i_p]` will be used. Attention: the best
            phases per period are generally undefined because the best trigger is set with veto. So do not use this default.
        :type ind_phase: int or None
        :param period: Period in days corresponding to `ind_period`. Is needed for the prior.
        :type period: float
        :return: best_template_index
        :rtype: int
        """
        i_p = ind_period
        i_ph = ind_phase
        per = period
        if not in_gaussianized:
            stats_periodic = self.statistics_periodic
            if ind_period is None:
                i_p = self.best_per_index
                i_ph = self.best_phase_inds[i_p]
                per = self.best_period
        else:
            stats_periodic = self.stats_gaussianized_periodic
            if ind_period is None:
                i_p = self.best_g_per_index
                i_ph = self.best_g_phase_inds[i_p]
                per = self.best_g_period
        if not with_prior:
            best_template_index = np.nanargmin([stats_periodic[i_p, tt, i_ph] for tt in range(len(stats_periodic[i_p]))])
        else:
            log_template_prior, m13r_bin_edges, period_edges = tm.load_template_prior()
            m_1_3_r_s_bin = min(max(0, np.searchsorted(m13r_bin_edges, self.m_1_3_r_s) - 1), len(m13r_bin_edges) - 2)
            per_bin = min(max(0, np.searchsorted(period_edges, per) - 1), len(period_edges) - 2)
            mask_of_non_zeros = ((~np.isclose(stats_periodic[i_p, :, i_ph], 0)) & ~np.isnan(stats_periodic[i_p, :, i_ph])).astype(bool)
            log_template_prior_bin = log_template_prior[m_1_3_r_s_bin, per_bin] * mask_of_non_zeros
            # -> if there is no stat there, we should not add a prior term

            best_template_index = np.nanargmax(np.array([(stats_periodic[i_p, tt, i_ph] * (
                    stats_periodic[i_p, tt, i_ph] < 0).astype(float)) ** 2 + log_template_prior_bin[tt]
                                                         for tt in range(len(stats_periodic[0]))]), axis=0)
        return best_template_index

    def get_summed_values(self, first_transit_time, period, template=None, minimize_downsampling=False, return_results=False):
        """
        Get the (maybe approximate) values what were used in the periodicity search for specific period, first transit time,
        and template. From the period and the first transit time, the expected transit times are calculates, and then the
        single-event arrays closest to these times are searched. Template is needed because the periodicity search summation
        was done per template. Since before the template marginalization the downsampling was done, the interleaved first
        transit time resolution is lost. And the summed values may be needed to search for in the interleaved statistics.
        Therefore, the in every downsampled element of the interleaved statistics we select the super-resolved index
        corresponding to the minimum which gives the best result which was selected during the downsampling.
        :param first_transit_time: First transit time in Kepler days (the same units as Kepler time axis is defined) for which
            the summed values are requested.
        :type first_transit_time: float
        :param period: Period in days for which the summed values are requested.
        :type period: float
        :param template: Period in days for which the summed values are requested.
        :type template: int
        :param minimize_downsampling: Whether
        :type minimize_downsampling:
        :param return_results: whether return the results (see below) or save them as a field of the class
        :type return_results: bool
        :return: summed_indices, summed_times, summed_stats_raw, summed_vars, summed_stats_raw_gauss, summed_corrections;
            See description below.
        :rtype: tuple of numpy.ndarrays 1d

        summed_indices : Indices in interleaved or non-interleaved arrays (depending on `self.is_summed_interleaved`)
            corresponding to the `self.summed_times`.
        summed_times : Times when the individual transits of the best trigger happened
        summed_stats_raw : Single-event matched-filtering scores corresponding to the `self.summed_times`
        summed_vars : Variance of the single-event matched-filtering scores corresponding to the `self.summed_times`
        summed_stats_raw_gauss : Gaussianized single-event matched-filtering scores corresponding to the `self.summed_times`
        summed_corrections : Log single-event distribution corrections corresponding to the `self.summed_times`
        """
        if template is None:
            template = self.best_template_index
        if template in self.interleaved_numbers:
            time = self.time_interleaved
            stat = self.statistics_interleaved[template]
            var = self.stat_vars_interleaved[template]
            stat_gauss = self.statistics_gaussianized_interleaved[template]
            corrections = self.corrections_interleaved[template]
            is_interleaved = True
        else:
            templ_not_interl = template - self.not_interleaved_numbers[0]
            time = self.time_not_interleaved
            stat = self.statistics_not_interleaved[templ_not_interl]
            var = self.stat_vars_not_interleaved[templ_not_interl]
            stat_gauss = self.statistics_gaussianized_not_interleaved[templ_not_interl]
            corrections = self.corrections_not_interleaved[templ_not_interl]
            is_interleaved = False
        summed_indices, expected_times = af.find_indices(time=time, first_time=first_transit_time, period=period, return_times=True)
        summed_indices = np.array(summed_indices).astype(int)
        # WARNING ATTENTION: in reality, it is impossible to find these values. The resulting score is marginalized
        # over all the templates, and this thing is per template.
        if is_interleaved and minimize_downsampling:  # try to release the consequence of downsampling
            indss = [summed_indices + i for i in range(self.n_shifts)]
            ress = []
            for si in indss:
                assert (si - len(var) < (self.n_shifts - 1)).all(), 'Indices are out of range!'
                si[si - len(var) >= 0] = len(var) - 1
                si = si.astype(int)
                msk = (~np.isinf(var[si]) & (~np.isclose(var[si], 0)))
                scal = np.sum(var[si][msk]) ** 0.5
                if scal == 0:
                    scal = np.inf
                stat_res = np.sum(stat[si][msk]) / scal
                cor_res = np.sum(corrections[si][msk])
                stat_squared = stat_res ** 2 + 2 * cor_res
                stat_squared = stat_squared * (stat_squared > 0).astype(float)  # zero if negative
                ress.append(stat_squared ** 0.5 * np.sign(stat_res))
                summed_indices = indss[np.nanargmin(ress)]
        summed_times = time[summed_indices]
        summed_stats_raw = stat[summed_indices]
        summed_vars = var[summed_indices]
        summed_stats_raw_gauss = stat_gauss[summed_indices]
        summed_corrections = corrections[summed_indices]
        if return_results:
            return summed_indices, summed_times, summed_stats_raw, summed_vars, summed_stats_raw_gauss, summed_corrections
        else:
            self.is_summed_interleaved = is_interleaved
            self.summed_indices = summed_indices
            self.summed_times = summed_times
            self.summed_stats_raw = summed_stats_raw
            self.summed_vars = summed_vars
            self.summed_stats_raw_gauss = summed_stats_raw_gauss
            self.summed_corrections = summed_corrections

    def get_best_trigger(self, apply_veto_mes=True, fn_threshold=Fn_threshold_veto_mes, max_trials=Max_mes_veto_trials,
                         in_gaussianized=False, save_veto_masks=False,
                         min_required_nonzero_transits_number=Min_required_nonzero_transits_number,
                         min_required_nonzero_transits_fraction=Min_required_nonzero_transits_fraction,
                         log_baseline_integral_score_rate=0,
                         minimal_trackable_score=Minimal_trackable_score, d_snr_2_tail=10,
                         dpi=D_period_integral_score, dphi=D_phase_integral_score):
        """
        :param apply_veto_mes:
        :type apply_veto_mes:
        :param fn_threshold:
        :type fn_threshold:
        :param max_trials:
        :type max_trials:
        :param in_gaussianized:
        :type in_gaussianized:
        :param save_veto_masks:
        :type save_veto_masks:
        :param min_required_nonzero_transits_number:
        :type min_required_nonzero_transits_number:
        :param min_required_nonzero_transits_fraction:
        :type min_required_nonzero_transits_fraction:
        :param log_baseline_integral_score_rate:
        :type log_baseline_integral_score_rate:
        :param minimal_trackable_score:
        :type minimal_trackable_score:
        :param d_snr_2_tail: 15 is 8**2 - 7**2, which is one sigma deviation down
        :type d_snr_2_tail:
        :param dpi:
        :type dpi:
        :param dphi:
        :type dphi:
        :return:
        :rtype:
        """
        """
        
        # ATTENTION. SINCE WE USE VETO OF SES, THERE WILL BE ZEROS, AND THE INTEGRAL WILL NOT BE FULL
        # THEREFORE WE IN GENERAL LOSE SNR, BOT ALSO THE DISTRIBUTION WILL HAVE AN UNKNOWN VARIANCE

        ATTENTION WARNING. The search was done on the statistics marginalized over templates. It is not well-defined then to
        check individual transits of individual templates. It is possible to define something like first marginalize then compare,
        but it is not the same. So it was decided in the end that we will only check the consistency of the best template.

        The test is checking whether some transits are zeros. In fact, it is the biggest amplitude difference between the transits,
        but it was derived to check zeroness. First, explicit zeros are checked. They can be there due to holes. And then,
        the veto is used to test statistical zeros.

        not applying score correction, we need to check whether it is more probable to have all equal transits or to have
        some equal and some zeros
        Creates:
        self.best_stat : Best score (SNR squared marginalized over templates, corrected for single-event-distribution non-gaussianity)
        self.best_per_index : Index in periods corresponding to the best score
        self.best_phase_index : Index in phases corresponding to the best score
        self.best_period : Period corresponding to the best score
        self.best_ftt : First transit time corresponding to the best score
        self.best_template_index : Index of best template at the point of the best score
        self.best_stat_gaussianized : Gaussianized statistic at the position corresponding to `self.best_stat`
        self.best_int_score : The integral over statistic with prior around the maximum
        And the same with index _g_ for the maximum over the gaussianized score.
        """
        if not in_gaussianized:
            stats_periodic_mar_temp = self.stats_periodic_mar_temp
            statistics_periodic = self.statistics_periodic
        else:
            stats_periodic_mar_temp = self.stats_gaussianized_periodic_mar_temp
            statistics_periodic = self.stats_gaussianized_periodic

        stats_periodic_mar_temp_initial = np.array(stats_periodic_mar_temp)
        # Finding period-phase bests list
        shap_x = stats_periodic_mar_temp.shape[1]
        # We will scan all the bins, including ones in the safety overlap region, because there can be large bad peaks there
        # Which will contribute to the integral score and make it too large.
        # We will avoid safety overlap regions later on
        flat_inds = ut.nanargsort(stats_periodic_mar_temp.flatten())
        sorting_inds = np.array([flat_inds // shap_x, flat_inds % shap_x]).T[::-1]
        if max_trials is None:
            max_trials = len(sorting_inds)
        if minimal_trackable_score is not None:
            stats_sorting_inds = np.array([stats_periodic_mar_temp[int(sorting_inds[index_counter][0]), int(sorting_inds[index_counter][1])]
                                           for index_counter in range(len(sorting_inds))])
            sorting_inds = sorting_inds[stats_sorting_inds >= minimal_trackable_score]
            max_trials = min(max_trials, len(sorting_inds))

        if apply_veto_mes:
            if save_veto_masks:
                veto_mes_masks = np.ones_like(stats_periodic_mar_temp).astype(bool)
            else:
                veto_mes_masks = None
            influence_matrix = np.zeros_like(stats_periodic_mar_temp, dtype=np.float32)
            assert statistics_periodic is not None, 'self.statistics_periodic needed for finding the best template'
            best_per_index = None
            best_phase_index = None
            trigger_accepted = False
            index_counter = 0
            while (not trigger_accepted) and (index_counter < max_trials):
                # and (best_stat is None or best_stat >= minimal_trackable_score or minimal_trackable_score is None):
                to_mark_influence = False
                best_per_index = int(sorting_inds[index_counter][0])
                best_phase_index = int(sorting_inds[index_counter][1])
                best_period = self.periods[best_per_index]
                best_ftt = self.tmin + self._phases[best_phase_index]
                best_stat = stats_periodic_mar_temp[best_per_index, best_phase_index]
                best_template_index = self.get_best_template_index(with_prior=True, in_gaussianized=in_gaussianized,
                                                                   ind_period=best_per_index, ind_phase=best_phase_index,
                                                                   period=best_period)

                summed_indices, summed_times, summed_stats_raw, summed_vars, summed_stats_raw_gauss, \
                    summed_corrections = self.get_summed_values(first_transit_time=best_ftt,
                                                                period=best_period, template=best_template_index,
                                                                minimize_downsampling=True, return_results=True)
                n_transits = len(summed_stats_raw)
                non_zero_mask = (~np.isclose(summed_stats_raw, 0)) & (~np.isclose(summed_vars, 0)) & (~np.isnan(summed_stats_raw)) & (
                    ~np.isinf(summed_vars))
                n_non_zeros = len(np.where(non_zero_mask)[0])
                if ((n_non_zeros < n_transits * min_required_nonzero_transits_fraction)
                        or (n_non_zeros < min_required_nonzero_transits_number)):
                    is_good = False
                    to_mark_influence = True
                elif influence_matrix[best_per_index, best_phase_index] > best_stat + d_snr_2_tail:
                    is_good = False
                    # We will not mark influence of something which is small. The influence of a larger peak was already marked
                else:
                    p_val = ut.unequal_amplitude_veto(summed_stats_raw, stat_vars=summed_vars, negative=True)
                    if p_val >= fn_threshold:
                        is_good = True
                    else:
                        is_good = False
                        to_mark_influence = True
                if not is_good:
                    if save_veto_masks:
                        veto_mes_masks[best_per_index, best_phase_index] = False
                    # WARNING: we are zeroing the statistic which did not pass the veto for
                    # this statistic not to impact the integral score

                    if to_mark_influence:
                        # Selecting the influence region of this peak and if someone is found to be just caused by the
                        # tail of this bad trigger (and if it is not big because big triggers not be not tail but actual good thing),
                        # then it will be vetted.

                        per_ind_max = best_per_index
                        phase_ind_max = best_phase_index
                        peak_val = best_stat
                        per_ind_left = max(0, per_ind_max - dpi)
                        phase_ind_left = max(0, phase_ind_max - dphi)
                        per_ind_right = min(per_ind_max + dpi, stats_periodic_mar_temp.shape[0])
                        phase_ind_right = min(phase_ind_max + dphi, stats_periodic_mar_temp.shape[1])
                        peak_coords_relative = per_ind_max - per_ind_left, phase_ind_max - phase_ind_left
                        vals_selected = stats_periodic_mar_temp_initial[per_ind_left:per_ind_right:, phase_ind_left:phase_ind_right]
                        # -> take initial because maybe connected component was destroyed by veto

                        if minimal_trackable_score is not None:
                            # Absolute difference in the snr squared between the peak and the minimal level that we bother to veto:
                            abs_diff_snr_2 = (peak_val - minimal_trackable_score) if peak_val > minimal_trackable_score else 1
                            # -> this 1 is just formal. If peak_val <= minimal_trackable_score, we are not really interested
                            part_snr_2 = None
                        else:
                            # If it is None, then select
                            abs_diff_snr_2 = None
                            part_snr_2 = 0.3
                        influence_matrix_selected = ut.get_indices_of_influence(vals_selected, ind_center_0=peak_coords_relative[0],
                                                                                ind_center_1=peak_coords_relative[1],
                                                                                part_peak=part_snr_2,
                                                                                abs_diff_val=abs_diff_snr_2)
                        # -> part_snr_2 is only used if abs_diff_snr_2 is None
                        # Updating the influence matrix with the new trigger:
                        influence_matrix[per_ind_left:per_ind_right:, phase_ind_left:phase_ind_right] = np.max([
                            influence_matrix[per_ind_left:per_ind_right:, phase_ind_left:phase_ind_right],
                            influence_matrix_selected.astype(float) * peak_val], axis=0).astype(np.float32)

                    # ###############################################################################################
                    # Zeroing to avoid contribution to the integral:
                    stats_periodic_mar_temp[best_per_index, best_phase_index] = 0.
                    # The values will be zeroed only if they are larger than the best trigger!
                # If the trigger passed the veto, then me check whether it is not in the safety overlap region
                if is_good:
                    if ((best_per_index >= self.period_overlap_bins)
                            and (best_per_index < len(stats_periodic_mar_temp) - self.period_overlap_bins)):
                        trigger_accepted = True
                index_counter += 1
            if save_veto_masks:
                if not in_gaussianized:
                    self.veto_mes_masks = veto_mes_masks
                else:
                    self.veto_g_mes_masks = veto_mes_masks
            if not trigger_accepted:
                # raise ValueError('No MES passing veto was found after ' + str(index_counter) + ' trials!')
                best_per_index, best_phase_index = np.nan, np.nan
        else:
            best_per_index, best_phase_index = sorting_inds[0]
        if not in_gaussianized:
            self.best_per_index = best_per_index
            self.best_phase_index = best_phase_index
            if (not np.isnan(best_per_index)) and (not np.isnan(best_phase_index)):
                self.best_period = self.periods[self.best_per_index]
                self.best_ftt = self.tmin + self._phases[self.best_phase_index]
                if self.statistics_periodic is not None:
                    self.best_template_index = self.get_best_template_index(with_prior=True, in_gaussianized=False,
                                                                            ind_period=self.best_per_index,
                                                                            ind_phase=self.best_phase_index, period=self.best_period)
                self.best_stat = self.stats_periodic_mar_temp[self.best_per_index, self.best_phase_index]
                self.best_stat_gaussianized = self.stats_gaussianized_periodic_mar_temp[self.best_per_index, self.best_phase_index]
                self.best_int_score = self.get_integral_score_at_max(per_ind_max=self.best_per_index,
                                                                     phase_ind_max=self.best_phase_index, gaussianized=False,
                                                                     use_occurrence_rate=True,
                                                                     log_baseline_integral_score_rate=log_baseline_integral_score_rate)
            else:
                self.best_period = np.nan
                self.best_ftt = np.nan
                self.best_template_index = np.nan
                self.best_stat = np.nan
                self.best_stat_gaussianized = np.nan
                self.best_int_score = np.nan

        else:
            self.best_g_per_index = best_per_index
            self.best_g_phase_index = best_phase_index
            if (not np.isnan(best_per_index)) and (not np.isnan(best_phase_index)):
                self.best_g_period = self.periods[self.best_g_per_index]
                if self.stats_gaussianized_periodic is not None:
                    self.best_g_template_index = self.get_best_template_index(with_prior=True, in_gaussianized=True,
                                                                              ind_period=self.best_g_per_index,
                                                                              ind_phase=self.best_g_phase_index,
                                                                              period=self.best_g_period)

                self.best_g_stat = self.stats_periodic_mar_temp[self.best_g_per_index, self.best_g_phase_index]
                self.best_g_stat_gaussianized = self.stats_gaussianized_periodic_mar_temp[self.best_g_per_index, self.best_g_phase_index]
                self.best_g_ftt = self.tmin + self._phases[self.best_g_phase_index]
                self.best_g_int_score = self.get_integral_score_at_max(per_ind_max=self.best_g_per_index,
                                                                       phase_ind_max=self.best_g_phase_index, gaussianized=True,
                                                                       use_occurrence_rate=True,
                                                                       log_baseline_integral_score_rate=log_baseline_integral_score_rate)
            else:
                self.best_g_period = np.nan
                self.best_g_template_index = np.nan
                self.best_g_stat = np.nan
                self.best_g_stat_gaussianized = np.nan
                self.best_g_ftt = np.nan
                self.best_g_int_score = np.nan

    def make_best_periodic_scores(self):
        """
        Makes the formatting of the best results that will be accessed by the `target_class`.
        """
        best_periodic_scores_data = [[np.nan, np.nan] for _ in range(Best_per_results_length)]
        for ind, val, val_gaus in [[Best_per_results_index_per, self.best_period, self.best_g_period],
                                   [Best_per_results_index_ftt, self.best_ftt, self.best_g_ftt],
                                   [Best_per_results_index_stats, self.best_stat, self.best_g_stat],
                                   [Best_per_results_index_stats_gauss, self.best_stat_gaussianized, self.best_g_stat_gaussianized],
                                   [Best_per_results_index_best_integral, self.best_int_score, self.best_g_int_score],
                                   [Best_per_results_index_template, self.best_template_index, self.best_g_template_index]]:
            best_periodic_scores_data[ind][0] = val
            best_periodic_scores_data[ind][1] = val_gaus
        best_periodic_scores_data[Best_per_results_index_per_list] = self.periods
        self.best_periodic_scores_data = best_periodic_scores_data

    def get_influence_period_indices(self, period, part_snr_2=0.5):
        """

        :param period: Period of the peak
        :type period: float
        :param part_snr_2: Define this as the end of correlation that we search around the trigger.
            That is return the indices around the given peak where snr squared drops to that part,
        :type part_snr_2:
        :return: per_ind_left, per_ind_right
        :rtype: tuple of int
        """
        d_per = self.periods[1] - self.periods[0]
        dur_max_templ = self.tb.durations[-1]
        # n_transits = int(np.round(klc.t_mission / period))
        delta_per_bins = int(np.round((period * 2 * dur_max_templ / self.t_mission * (1 - part_snr_2)) / d_per))
        close_period_ind = np.nanargmin(np.abs(self.periods - period))
        per_ind_left = max(0, close_period_ind - delta_per_bins)  # we do not divide delta_per_bins by 2 on purpose
        per_ind_right = min(len(self.periods), close_period_ind + delta_per_bins)
        return per_ind_left, per_ind_right

    def get_influence_phase_indices(self, first_transit_time, in_interleaved=True, part_snr_2=0.5):
        if in_interleaved:
            phases, stats_periodic = self._phases_interleaved, self.statistics_periodic_interleaved
        else:
            phases, stats_periodic = self._phases_not_interleaved, self.statistics_periodic_not_interleaved
        d_ftt = phases[1] - phases[0]
        dur_max_templ = self.tb.durations[-1]
        delta_ftt_bins = int(np.round((dur_max_templ * (1 - part_snr_2)) / d_ftt))
        close_phase_ind = np.nanargmin(np.abs(phases + self.tmin - first_transit_time))
        phase_ind_left = max(0, close_phase_ind - delta_ftt_bins)  # we do not divide delta_ftt_bins by 2 on purpose
        phase_ind_right = min(len(phases), close_phase_ind + delta_ftt_bins)
        return phase_ind_left, phase_ind_right


@njit(parallel=False, fastmath=False)
def sum_periods_fun(inputs_array, dt, tmax, tmin, periods):
    """
        Folds and sums the statistics with `periods` and all the possible initial phases.
        Accounts for sub-bin template shifts by folding shift-interleaved series.
        Accounts for masks by zeroing the bad statistics.

        Implementation remarks:
        - This njit-ed function needs to be initialized with the proper types, otherwise the initialization does not count
        - Do not use slices in numba! Use for loops, they are faster
        - `inputs_array` should be c-contiguous, and indexing should use it. Therefore, we
          transposed it to have the continuously repeating index last.
        - We combined all the arrays to one array and stacked the template axis and the quantity axis to make it a 2D array and
          allow for continuous indexing of the last axis.
        - parallel=False because we run on cluster
        - fastmath=False because otherwise it is slower
        - The for-loop is over periods, first_transit_times, and template*quantity. The length of time of the input array is
          almost not important for runtime
        - It is important to keep the type of timing values float 64 and
          not 32 because then the resolution of  (p / dt * n) is not achieved.
        - pre-computing grids of timings and enumerating over it does not improve the runtime. Also, zip is slower, do not use it.
        - Initializing the results array with a for loop does not modify the runtime significantly.
        - Initializing the esults array outside the function does modify the runtime, but it will be spent outside then.

        :param inputs_array: Is composed of vstacked and transposed vectors that need to be folded. Should be a c-contiguous array.
            Usually, has a shape (length_time, n_quantities*n_templates). The quantities are usually:
            - Single-events statistics multiplied by the mean flux trend, which is assumed to be
            proportional to the expected amplitude of the transits. If the statistic was already normalized by trends,
            then no need for trends;
            - Single-events statistics multiplied by the squared mean flux trend, which is assumed to be
            proportional to the expected amplitude of the transits. If the statistic was already normalized by trends,
            then no need for trends;
            - Log corrections for the single-event statistic distribution non-gaussianity. They need to be
            also folded and then applied to the folded statistics in the end;
            - Gaussianized single-events statistics multiplied by the mean flux trend, which
            is assumed to be proportional to the expected amplitude of the transits. If the statistic was already normalized by
            trends, then no need for trends.
        :type inputs_array: numpy.ndarray 2d of float32
        :param dt : Time step of equally sampled time axis. It is important to keep the type float 64 and
            not 32 because then the resolution of  (p / dt * n) is not achieved.
        :type dt : float64
        :param tmax : Maximal time in the input time axis.
        :type tmax : float64
        :param tmin : Minimal time in the input time axis.
        :type tmin : float64
        :param periods: Grid of periods in the same units as `dt`, `tmax`, `tmin` by which the statistics are to be folded.
        :type periods: numpy.ndarray 1d of float64. It is important to keep the type float 64 and
            not 32 because then the resolution of  (p / dt * n) is not achieved.
        :return: results_array, an array of shape (n_periods, n_phases, n_quantities*n_templates). Usually,
         the quantities are :
         statistics_periodic, denomsq, corrections_periodic, stats_gaussianized_periodic, see description below.
        :rtype: tuple of numpy.ndarrays 3d of float32

        Returns (all are numpy.ndarray 3d of float):
        statistics_periodic : array of folded statistics for all the periods, templates and phases. Has raw units, is to be
            normalized by sqrt of `denomsq` to get the units of SNR. Has to be corrected by `corrections_periodic` to account
            properly for the distribution.
        denomsq : array of folded statistic variances (with mean trends normalization) for all the periods, templates and phases.
            These denominators are used to bring the statistics to units of SNR.
        corrections_periodic : Folded log corrections for the single-event statistic distribution non-gaussianity.
        stats_gaussianized_periodic : array of folded gaussianized statistics for all the periods, templates and phases. Has raw
            units, is to be normalized by sqrt of `denomsq` to get the units of SNR. Has to be corrected by `corrections_periodic`
            to account properly for the distribution.
        """
    res_temp_size = len(inputs_array[0])
    phases_size = int(np.ceil(((periods[-1]) / dt)))
    len_periods = len(periods)
    results_array = np.zeros((len_periods, phases_size, res_temp_size), dtype=Dtype_r)
    t_tot = tmax - tmin
    for p_ind in range(len(periods)):
        p = periods[p_ind]
        p_over_dt = p / dt
        n_transits = int(np.ceil((t_tot / p)))
        phaselen = int(p / dt)
        for n in range(n_transits):
            time_ind_0 = int(np.round(p_over_dt * n))
            for ph_ind in range(phaselen):
                time_ind = time_ind_0 + ph_ind
                for i_res_temp in range(res_temp_size):
                    results_array[p_ind, ph_ind, i_res_temp] += inputs_array[time_ind, i_res_temp]
    return results_array


_ = sum_periods_fun(inputs_array=np.ones((192, 10), dtype=Dtype_r),
                    dt=Dtype_timings(0.02), tmax=Dtype_timings(3.), tmin=Dtype_timings(0.),
                    periods=np.arange(1., 2., 0.5).astype(Dtype_timings))


# @njit(parallel=False, fastmath=False)
# def sum_periods_fun_old(statistics_by_trends, trends_sq_by_vars, corrections, statistics_by_trends_gaussianized,
#                         dt, tmax, tmin, periods):
#     """
#     Remark: this function is slower than tha new sum_periods_fun by a factor of 3 to 4, so we do not use it starting from run_13.
#     Folds and sums the statistics with `periods` and all the possible initial phases. Accounts for
#     sub-bin template shifts, accounts for masks.
#
#     :param statistics_by_trends: Single-events statistics multiplied by the mean flux trend, which is assumed to be
#         proportional to the expected amplitude of the transits. If the statistic was already normalized by trends,
#         then no need for trends.
#     :type statistics_by_trends: numpy.ndarray 2d of float
#     :param trends_sq_by_vars: Single-events statistics multiplied by the squared mean flux trend, which is assumed to be
#         proportional to the expected amplitude of the transits. If the statistic was already normalized by trends,
#         then no need for trends.
#     :type trends_sq_by_vars: numpy.ndarray 2d of float
#     :param corrections: Log corrections for the single-event statistic distribution non-gaussianity. They need to be
#         also folded and then applied to the folded statistics in the end.
#     :type corrections: numpy.ndarray 2d of float
#     :param statistics_by_trends_gaussianized: Gaussianized single-events statistics multiplied by the mean flux trend, which
#         is assumed to be proportional to the expected amplitude of the transits. If the statistic was already normalized by
#         trends, then no need for trends.
#     :type statistics_by_trends_gaussianized: numpy.ndarray 2d of float
#     :param dt : Time step of equally sampled time axis.
#     :type dt : float
#     :param tmax : Maximal time in the input time axis.
#     :type tmax : float
#     :param tmin : Minimal time in the input time axis.
#     :type tmin : float
#     :param periods: Grid of periods in the same units as `dt`, `tmax`, `tmin` by which the statistics are to be folded.
#     :type periods: numpy.ndarray 1d of float
#     :return: statistics_periodic, denomsq, corrections_periodic, stats_gaussianized_periodic, see description below.
#     :rtype: tuple of numpy.ndarrays 3d of float
#
#     Returns (all are numpy.ndarray 3d of float):
#     statistics_periodic : array of folded statistics for all the periods, templates and phases. Has raw units, is to be
#         normalized by sqrt of `denomsq` to get the units of SNR. Has to be corrected by `corrections_periodic` to account
#         properly for the distribution.
#     denomsq : array of folded statistic variances (with mean trends normalization) for all the periods, templates and phases.
#         These denominators are used to bring the statistics to units of SNR.
#     corrections_periodic : Folded log corrections for the single-event statistic distribution non-gaussianity.
#     stats_gaussianized_periodic : array of folded gaussianized statistics for all the periods, templates and phases. Has raw
#         units, is to be normalized by sqrt of `denomsq` to get the units of SNR. Has to be corrected by `corrections_periodic`
#         to account properly for the distribution.
#     """
#     # phases_size = int(np.ceil((periods[-1]) / dt))
#     # statistics_periodic = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
#     # denomsq = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
#     # corrections_periodic = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
#     # stats_gaussianized_periodic = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
#     # for P_ind in range(len(periods)):
#     #     p = periods[P_ind]
#     #     n_transits = int(np.ceil((tmax - tmin) / p))
#     #     phaselen = int(p / dt)
#     #     for n in range(n_transits):
#     #         time_ind = int(np.round((p * n / dt)))
#     #         statistics_periodic[P_ind, :, :phaselen] += statistics_by_trends[:, time_ind:time_ind + phaselen]
#     #         denomsq[P_ind, :, :phaselen] += trends_sq_by_vars[:, time_ind:time_ind + phaselen]
#     #         corrections_periodic[P_ind, :, :phaselen] += corrections[:, time_ind:time_ind + phaselen]
#     #         stats_gaussianized_periodic[P_ind, :, :phaselen] += statistics_by_trends_gaussianized[:, time_ind:time_ind + phaselen]
#     # return statistics_periodic, denomsq, corrections_periodic, stats_gaussianized_periodic
#     phases_size = int(np.ceil((periods[-1]) / dt))
#     len_statistics_by_trends = len(statistics_by_trends)
#     len_periods = len(periods)
#     statistics_periodic = np.zeros((len_periods, len_statistics_by_trends, phases_size), dtype=Dtype_r)
#     denomsq = np.zeros((len(periods), len_statistics_by_trends, phases_size), dtype=Dtype_r)
#     corrections_periodic = np.zeros((len_periods, len_statistics_by_trends, phases_size), dtype=Dtype_r)
#     stats_gaussianized_periodic = np.zeros((len_periods, len_statistics_by_trends, phases_size), dtype=Dtype_r)
#     for p_ind in range(len_periods):
#         p = periods[p_ind]
#         n_transits = int(np.ceil((tmax - tmin) / p))
#         phaselen = int(p / dt)
#         for n in range(n_transits):
#             time_ind_0 = int(np.round((p * n / dt)))
#             for temp_ind in range(len_statistics_by_trends):
#                 for ph_ind in range(phaselen):
#                     time_ind = time_ind_0 + ph_ind
#                     statistics_periodic[p_ind, temp_ind, ph_ind] += statistics_by_trends[temp_ind, time_ind]
#                     denomsq[p_ind, temp_ind, ph_ind] += trends_sq_by_vars[temp_ind, time_ind]
#                     corrections_periodic[p_ind, temp_ind, ph_ind] += corrections[temp_ind, time_ind]
#                     stats_gaussianized_periodic[p_ind, temp_ind, ph_ind] += statistics_by_trends_gaussianized[
#                                                                                temp_ind, time_ind]
#     return statistics_periodic, denomsq, corrections_periodic, stats_gaussianized_periodic
#
#
# _ = sum_periods_fun(statistics_by_trends=np.ones((2, 192)),
#                     trends_sq_by_vars=np.ones((2, 192)),
#                     corrections=np.ones((2, 192)),
#                     statistics_by_trends_gaussianized=np.ones((2, 192)),
#                     dt=0.02, tmax=3., tmin=0., periods=np.arange(1., 2., 0.5))


@njit(parallel=False, fastmath=True)
def normalize_stats_fun(statistics_periodic, stats_gaussianized_periodic, denomsq):
    # I did not use stacking because it is too expensive to concatenate and make contiguous big arrays again,
    # and I do not want to think how to reorganize the indices
    fill_val = Dtype_r(np.nan)
    zero = Dtype_r(0.0)
    for p in range(len(statistics_periodic)):
        for t in range(len(statistics_periodic[p])):
            for ph in range(len(statistics_periodic[p][t])):
                if denomsq[p][t][ph] == zero:
                    statistics_periodic[p][t][ph] = fill_val
                    stats_gaussianized_periodic[p][t][ph] = fill_val
                else:
                    norm = np.sqrt(denomsq[p][t][ph])
                    statistics_periodic[p][t][ph] /= norm
                    stats_gaussianized_periodic[p][t][ph] /= norm
    return statistics_periodic, stats_gaussianized_periodic


_ = normalize_stats_fun(np.ones((3, 2, 4), dtype=Dtype_r), np.ones((3, 2, 4), dtype=Dtype_r), np.ones((3, 2, 4), dtype=Dtype_r))


Inpaint_rand_vec = np.abs(np.random.normal(loc=0, scale=0.001, size=8192).astype(Dtype_r))


@njit(parallel=False, fastmath=False)
def inpaint_fluxes_fun(fluxes, masks, wf_tds, rand_vec=Inpaint_rand_vec):
    length = len(fluxes[0])
    # I was afraid that wf_tds are fftshifted to the center, but since here we have the inner product, it is not important
    inpainted_fluxes = np.zeros((len(fluxes), length), dtype=Dtype_r)
    for q in range(len(fluxes)):
        # filter2cov_matrix:
        prod = np.array([np.sum(wf_tds[q] * np.roll(wf_tds[q], i)) for i in range(length)])
        c_inv = np.array([[prod[k - j] for k in range(length)] for j in range(length)], dtype=Dtype_r)
        a_t_c_inv = c_inv[~masks[q]]
        m = a_t_c_inv[:, ~masks[q]]
        m += np.identity(len(m), dtype=Dtype_r) * rand_vec[:len(m)] * Dtype_r(np.mean(np.diag(m)))
        m_inv_a_t_c_inv = np.linalg.solve(a=m, b=a_t_c_inv)  # ax = b  solution
        f = np.zeros(shape=(length, length), dtype=Dtype_r)
        f[~masks[q]] = m_inv_a_t_c_inv
        inpainted_fluxes[q] = fluxes[q] - np.dot(f, fluxes[q])
    return inpainted_fluxes


_ = inpaint_fluxes_fun(fluxes=np.array(np.random.normal(0, 1, (2, 32)), dtype=Dtype_r),
                       masks=np.random.choice([0, 1], (2, 32)).astype(bool),
                       wf_tds=np.array([np.concatenate((np.zeros(32 // 2 - 2), [-0.5, 1, 1, -0.5], np.zeros(32 // 2 - 2)))
                                        for _ in range(2)], dtype=Dtype_r))


def get_minimal_processed_folded_klc(period=None, search_periodicity=True,  delta_period=0.1, template_index='all',
                                     times=None, fluxes=None, target_id=None, get_data=False,
                                     nsig_outliers_0_neg=Nsig_outliers_0_neg, nsig_outliers_0_pos=Nsig_outliers_0_pos,
                                     data_location=None, output_location=None, cut_edge_nans=False, delete_fulls=True,
                                     allow_output_folder_creation=False, allow_data_folder_creation=False):
    """
    Usually used for not the actual Kepler Light Curves, but rather for centroids, background pixels etc.
    """
    klc = KeplerLightCurve(target_id=target_id, get_data=get_data, times=times, fluxes=fluxes, cut_edge_nans=cut_edge_nans,
                           data_location=data_location, output_location=output_location,
                           allow_data_folder_creation=allow_data_folder_creation,
                           allow_output_folder_creation=allow_output_folder_creation)
    klc.condition_data(nsig_outliers_0=nsig_outliers_0_neg, nsig_outliers_0_pos=nsig_outliers_0_pos,
                       nsig_outliers_0_neg=nsig_outliers_0_neg,
                       nsig_outliers_w=nsig_outliers_0_neg, nsig_outliers_w_pos=nsig_outliers_0_pos,
                       nsig_outliers_w_neg=nsig_outliers_0_neg)
    klc.condition_templates(template_index=template_index)
    klc.matched_filtering()
    klc.cut_zero_padding_ends()
    klc.connect_quarters()
    klc.interleave_shifts(delete_fulls=delete_fulls)
    if search_periodicity and period is not None:
        klc.sum_periods(p_min=period - delta_period, p_max=period + delta_period)
    # -> we do not marginalize, so statistics are still not in units of snr squared
    return klc


def get_fold_quickly(target_id, period, data_location=Default_data_location, output_location=Default_output_location,
                     connected_statistic_data_filename_postfix=Connected_statistic_data_filename_postfix,
                     conditioned_data_filename_postfix=Conditioned_data_filename_postfix,
                     overlap_period_bins=P_bins_overlap, allow_output_folder_creation=False):
    klc = KeplerLightCurve(target_id=target_id, get_data=True, data_location=data_location, output_location=output_location,
                           allow_output_folder_creation=allow_output_folder_creation)
    file_path_connected_data = af.get_saving_file_path(output_location=klc.output_location, target_id=klc.target_id,
                                                       filename_postfix=connected_statistic_data_filename_postfix, file_format='npy')
    file_path_conditioned_data = af.get_saving_file_path(output_location=klc.output_location, target_id=klc.target_id,
                                                         filename_postfix=conditioned_data_filename_postfix, file_format='npy')

    if hasattr(klc, 'statistics_full') and (klc.statistics_full is not None):
        pass
    elif os.path.exists(file_path_connected_data):
        klc.load_connected_stats(file_path=file_path_connected_data)
    else:
        if klc.statistics_raw is None or not hasattr(klc, 'statistics_raw'):
            if os.path.exists(file_path_conditioned_data):
                klc.load_conditioned_data(file=file_path_conditioned_data)
                klc.initialize_templates()
            else:
                klc.condition_data()
                klc.condition_templates()
            klc.detect_single_transits(gaussianize=True)
        klc.connect_quarters()
    klc.interleave_shifts()

    if not hasattr(period, '__len__'):
        periods = np.ones(1) * period
    else:
        periods = np.array(period, dtype=float)
    klc.periodicity_search(p_min=None, p_max=None, periods=periods,
                           delete_full_stats=False, force_interleave_shifts=False,
                           delete_periodic_interleaved=False,
                           save_uncorrected_periodic=True, delete_non_marginalized=False,
                           overlap_period_bins=overlap_period_bins)
    return klc
