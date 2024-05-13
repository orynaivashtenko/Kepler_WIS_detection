#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:11:32 2020

@author: orynaivashtenko
version October 2022


"""
import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import warnings
from scipy.signal import sosfiltfilt
import numba
from scipy.stats import chi2

import template_bank.template_module as tm
import auxiliary_functions as af
import utils as ut
from search_pipeline_parameters import *

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
# TODO: write note about template indexing, about vocabulary (raw score, units of snr, full stats,
# interleaved stats, periodic stats, corrections etc.
# TODO Plan:
# finish with todos
# check everything works
# check other files
# check these functions of non-vetted fluxes and all the other stuff
# maybe save more stuff to save some time
# make new time estimate and chunk splitting

# TODO: ideas and maybe update things:
# what if the template is not symmetric
# maybe use minimax of scipy.spatial.distance.cdist

# TODO discussion:
# we are saving one best thing per star. Then, we are vetting it, and it might not pass.
# But the same star could have a trigger which was slightly smaller but passing the veto.
# Maybe we need to save all the scores of one star which are higher than a certain threshold.

# When we plot folds of statistic, should these be folds with score correction or not?


class KeplerLightCurve(object):
    def __init__(self, target_id=None, times=None, fluxes=None, get_data=True,
                 data_location=Default_data_location, output_location=Default_output_location,
                 remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True,
                 m_1_3_r_s=Default_m_1_3_r_s):
        """
        Initializes a kepler light curve object. See the description in the attribute comments in the __init__.
        :param target_id: Kepler ID of the target. A 9-digit number, where some first digits may be zero.
            May be specified as integer or as string, should only contain digits.
            If get_data==True, target_id must be specified, it cannot be None.
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
            will be created if it is not there. See the descripition in naming notations.
        :type output_location: str
        :param remove_zeroth_quarter: Whether to remove quarter 0. This quarter is small and often bad,
            not intended for science.
        :type remove_zeroth_quarter: bool
        :param remove_quarter_17: Whether to remove quarter 17.
            This quarter is small, there is a hole, and there might be glitch. We remove it
        :type remove_quarter_17: bool
        :param m_1_3_r_s: Mass of the star in usits of solar mass divided by the radius of the star to
            the power 1/3. Characterizes stellar density. If not provided, will be replaced by the default.
        :type m_1_3_r_s: float
        """

        # INITIALIZING THE STAR
        if target_id is not None:
            target_id = str(target_id).rjust(9, "0")  # Convert Kepler ID to string of appropriate length
        self.output_location = output_location
        """Local path where the output will be saved"""
        self.target_id = target_id
        """Kepler ID of the target"""
        self.m_1_3_r_s = m_1_3_r_s
        """Mass of the star in usits of solar mass divided by the radius of the star to
            the power 1/3. Characterizes stellar density"""

        # ########################################################################################################
        # GETTING DATA

        self.orig_times = None
        """Initial time arrays of quarters provided by user input or downloaded from location"""
        self.orig_fluxes = None
        """Initial flux arrays of quarters provided by user input or downloaded from location"""
        if not get_data:
            if not hasattr(times[0], '__len__'):
                times = [times]
            if not hasattr(fluxes[0], '__len__'):
                fluxes = [fluxes]
            assert len(times) == len(fluxes), 'Number of time pieces is supposed to be equal to the number of light curves!'
            for i in range(len(times)):
                assert len(times[i]) == len(fluxes[i]), 'Length of fluxes[' + str(i) + '] does not match its time length!'
            self.orig_times = times
            self.orig_fluxes = fluxes
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
        """Flux arrays for quarters used for technical purposes sucn as PDS measurement. 
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
        self.big_hole_masks = [np.ones(len(self.masks[q])).astype(bool) for q in self.quarters]
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
        tmax = np.max(np.array([len(t) for t in self.orig_times]))  # maximum length of the time
        # the closest (but bigger) power of 2 to the length        Tmax += tm.L_ban
        self.support = 2 ** int(np.ceil(np.log2(tmax)))
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

        self.inpaint_fluxes_fast = numba.njit(ut.inpaint_fluxes_fun)  # it is important to use it because if shrinks ram from 4G to 1.5G
        # initialization of the numba function on some small data in order  not to
        # spend time (AND MEMORY) on big data:
        _ = self.inpaint_fluxes_fast(fluxes=np.array(np.random.normal(0, 1, (2, 32)), dtype=Dtype_r),
                                     masks=np.random.choice([0, 1], (2, 32)).astype(bool),
                                     wf_tds=np.array([np.concatenate((np.zeros(32 // 2 - 2), [-0.5, 1, 1, -0.5], np.zeros(32 // 2 - 2)))
                                                      for _ in range(2)], dtype=Dtype_r))
        self.sum_periods_fast = numba.njit(sum_periods_fun, parallel=False, fastmath=False)
        # initialization of the numba function on some small data in order  not to
        # spend time (AND MEMORY) on big data:
        _ = self.sum_periods_fast(statistics_by_trends=np.ones((2, 192)),
                                  trends_sq_by_vars=np.ones((2, 192)),
                                  corrections=np.ones((2, 192)),
                                  statistics_by_trends_gaussianized=np.ones((2, 192)),
                                  dt=0.02, tmax=3., tmin=0., periods=np.arange(1., 2., 0.5))

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
        """Number of zerros added at the beginning and the end of every quarter to match `self.support`, per quarter"""
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
        """Matched-filtering score in umits of SNR per template (subbin-shifts in a row) per quarter"""
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
        self.veto_bit_masks = None
        """Single-event quality veto passing masks of `self.statistics_raw` per template (subbin-shifts in a row) 
            per quarter for different veto tests. Binary digits code different tests."""
        self.trigger_indices = None
        """All single-event threshold-crossing indices which undervent single-enebt veto. For templates and quarters"""

        # SINGLE_EVENT STATISTIC DISTRIBUTION CORRECTION
        self.statistics_gaussianized = None
        """Gaussianized matched-filtering score per template (subbin-shifts in a row) per quarter"""
        self.statistics_snr_gaussianized = None
        """Gaussianized matched-filtering score in umits of SNR per template (subbin-shifts in a row) per quarter"""
        self.stat_non_gauss_scales_coefs = None
        """Parameters of distributions of single-event scores. Per template, one list is for negative dtatistics, 
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
        """Period-folded matched-filtering statistics, brought to units of SNR, which will not indergo 
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
        self.best_integral_score = None
        """Sum of prior-weighted statistis around the maximal `self.stats_periodic_mar_temp`"""
        self.best_phase_index = None
        """Period index corresponding to the maximal `self.stats_periodic_mar_temp`"""
        self.best_per_index = None
        """Phase index corresponding to the maximal `self.stats_periodic_mar_temp`"""
        self.best_template_index = None
        """Template index of the best (nagetive and prior-weighted if applied) template 
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
        self.best_g_integral_score = None
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
                       npsdsmin=Npsdsmin,
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
        :param nsig_outliers_0_pos: Minimal deviation from trend in the positive diraction considered
            to be outlier (in units of std)
        :type nsig_outliers_0_pos: float
        :param nsig_outliers_0_neg: Minimal deviation from trend in the negative diraction considered
            to be outlier (in units of std)
        :type nsig_outliers_0_neg: float
        :param nsig_outliers_psd:
        :type nsig_outliers_psd:
        :param nperseg_psd: Length of slice used in measuring PSD
        :type nperseg_psd: int
        :param npsdsmin: Minimal number of slices required to measure PSD with Welch method
        :type npsdsmin:  int
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
        self.measure_psds(nperseg=self.nperseg_psd, tukey_steps_len=tukey_steps_len)
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
        slices. Otherwise, the masks will not be cleaned from small holes. Therefore data in maskes
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

    def save_conditioned_data(self, file=None, path=None, filename_postfix=Default_conditioned_data_filename_postfix):
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

    def load_conditioned_data(self, file=None, filename_postfix=Default_conditioned_data_filename_postfix,
                              path=None, no_zero_padding=False):
        """
        Loads the previosly prepared conditioned data needed for the single-event search.
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
        self.is_zero_padded = True
        for q in self.quarters:
            zpe1 = int(conditioned_data[0][ind_last + 1])
            zpe2 = int(conditioned_data[0][ind_last + 2])
            self.zero_padding_ends[q][0] = zpe1
            self.zero_padding_ends[q][1] = zpe2
            lenflux = int(conditioned_data[0][ind_last + 3])
            if lenflux != self.support:
                self.is_zero_padded = False
            self.times.append(np.array(conditioned_data[1][ind_last: ind_last + lenflux]).astype(float))
            self.fluxes.append(np.array(conditioned_data[2][ind_last: ind_last + lenflux]).astype(float))
            whfltd1 = np.array(conditioned_data[3][ind_last: ind_last + lenflux]).astype(float)
            if (not self.is_zero_padded) and (not no_zero_padding):
                whfltd1 = np.concatenate((np.zeros(zpe1), whfltd1, np.zeros(zpe2)))
            self.whitened_fluxes_td.append(whfltd1)
            self.whitened_fluxes_fd.append(np.fft.rfft(whfltd1))
            wftd1 = np.array(conditioned_data[4][ind_last: ind_last + self.support]).astype(float)
            self.wf_tds.append(wftd1)
            self.wf_fds.append(np.abs(np.fft.rfft(wftd1)))
            self.big_hole_masks.append((conditioned_data[5][ind_last: ind_last + lenflux]).astype(bool))
            for k in range(len_whitened_templates_td):
                whtltd1 = np.array(conditioned_data[k + 7][ind_last:ind_last + lenflux]).astype(float)
                if (not self.is_zero_padded) and (not no_zero_padding):
                    whtltd1 = np.concatenate(
                        (whtltd1[:(len(whtltd1) + 1) // 2], np.zeros(zpe1 + zpe2), whtltd1[(len(whtltd1) + 1) // 2:]))
                self.whitened_templates_td[k].append(whtltd1)
                self.whitened_templates_fd[k].append(np.fft.rfft(whtltd1))
            ind_last += self.support
        if (not self.is_zero_padded) and (not no_zero_padding):
            for q in self.quarters:
                zerlen1 = self.zero_padding_ends[q][0]
                zerlen2 = self.zero_padding_ends[q][1]
                self.fluxes[q] = np.concatenate((np.zeros(zerlen1),
                                                 self.fluxes[q], np.zeros(zerlen2)))
                self.big_hole_masks[q] = np.concatenate((np.zeros(zerlen1),
                                                         self.big_hole_masks[q], np.zeros(zerlen2))).astype(bool)
                self.times[q] = np.concatenate((np.zeros(zerlen1) + np.nan,
                                                self.times[q], np.zeros(zerlen2) + np.nan))
                self.times[q] = ut.interpolate_time(np.array(self.times[q]), mask=None)
        self.is_zero_padded = True
        self.is_template_whitened = True

    def save_mean_psd_for_injections(self, file=None, filename_postfix=Default_psd_for_injections_filename_postfix, path=None):
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

    def save_connected_stats(self, filename_postfix=Default_connected_statistic_data_filename_postfix, path=None):
        """
        Saving all the connected stats and other fields which will be used in periodicity search or timeslides.
        :param filename_postfix: Used in the name convention "target_id + filename_postfix" in
            case `file` is not specified. The location will be `self.output_location`
        :type filename_postfix: str
        :param path: Where to save the result if `file` is not specified. if None, the
            location will be `self.output_location`
        :type path: str or None
        """
        if path is None:
            path = self.output_location
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

        file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                            filename_postfix=filename_postfix, file_format='npy')
        np.save(file_path, connected_data, allow_pickle=True)

    def save_no_veto_no_score_cor_stats(self, filename_postfix=Default_no_veto_no_score_cor_stats_filename_postfix):
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

    def load_connected_stats(self, file_path=None, filename_postfix=Default_connected_statistic_data_filename_postfix, path=None):
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
        Initializes and conditions the template bank. Most of the temlate properties and methods are inside the `TemplateBank` class,
        but notchig, whitening and similar are performed here.
        :param templates: Array of transit models used for detecton, usually produced with Batman pachageCentered at the center.
            See `template_module`
        :type templates: list numpy.ndarrays 1d of float, see `template_module`
        :param shifts: Subbin shifts of templates, see `template_module`
        :type shifts:
        :param template_index: which templates from the bank to use
        :type template_index: int or 'all'
        """
        self.initialize_templates(templates=templates, shifts=shifts, template_index=template_index)
        self.notch_templates()  # why notch is before roll, is written inside the method
        self.tb.roll_templates()
        self.whiten_templates()

    def detect_single_transits(self, nsig_outliers_stat=Nsig_outliers_stat,
                               apply_veto=True, gaussianize=Get_gaussianized_ses,
                               apply_veto_only_to_neg=False,
                               margin_snr_veto=Margin_snr_veto, fn_fit_veto=Fn_fit_veto,
                               n_parts_veto_snr=N_parts_veto_snr,
                               n_parts_veto_fourier=N_parts_veto_fourier,
                               wind_len_overlap_fourier=Wind_len_overlap_fourier,
                               threshold_peak_sigma_veto=Threshold_peak_sigma_veto,  # there will also be look-elsewhere
                               save_triggers_as_attribute=False,
                               save_no_veto_no_score_cor_stats_as_attr=False,
                               max_abs_stat_veto=Max_abs_stat_veto):
        """
        Performs the seatch of single-transit events in all the quarters if the conditioned and whitened
        data and for all the conditioned and whitened templates.
        Applies shape quality veto to the statistics. Investingates the distribution of the resulting score,
        corrects its variance, calculates correction for its non-gaussianity, calculates the gaussianized
        score which is a metric for estimating the lower limit of the likelihood.
        Please see the more detalied description inside each method.
        :param nsig_outliers_stat: Deviation in units of stansard deviation to be considered as an outlier in the
            outlier-free variance calculation for the statistic.
        :type nsig_outliers_stat: float
        :param apply_veto: Whether to apply transit shape quality veto
        :type apply_veto: bool
        :param gaussianize: Whether to calculate the gaussianized statistics, that is statistics which have the
            same ranking as the given statistics but follow exactly the Gaussian distribution.
        :type gaussianize: bool
        :param apply_veto_only_to_neg: Whether apply the shape quality veto only to the negative transits,
            or also to positive. We apply to both because they can have some interaction.
        :type apply_veto_only_to_neg: bool
        :param margin_snr_veto: The minimal statistic in units of SNR to which the transit shape quality veto
            is applied.
        :type margin_snr_veto: float
        :param fn_fit_veto: False-negative probability to veto out a good score. Is defined per individual test,
            and there are few of them, around 3.
        :type fn_fit_veto: float
        :param n_parts_veto_snr: Number of parts into which the template is split for the shape quality veto
            in which we compare these parts and make sure they are consistent to belong to the same transit.
            This is the number of parts for the case in which the template is split in the time domain into
            parts having the same SNR squared.
        :type n_parts_veto_snr: int
        :param n_parts_veto_fourier: Number of parts into which the template is split for the shape quality veto
            in which we compare these parts and make sure they are consistent to belong to the same transit.
            This is the number of parts for the case in which the template is split in the Fourier domain into
            overlapping parts having the same SNR squared.
        :type n_parts_veto_fourier: int
        :param wind_len_overlap_fourier: The parts consistency veto in the Fourier domain is made to the overlapping
            parts because if the parts edges are too sharp in the Fourier domain, then in the time domain they
            will have an infinite support, and all tha glitches will produce impact. To restrict the size in the
            time domain, we make parts smooth and overlapping in the Fourier domain. `wind_len_overlap_fourier`
            sets the size of that overlapping window
        :type wind_len_overlap_fourier: int
        :param threshold_peak_sigma_veto: Threshold value for the peak veto in units of sigmas of distribution.
            Peak veto makes sure it is more probable for the score to originate from the transit and not from
            a spurious peak. This veto is basically useless.
        :type threshold_peak_sigma_veto: float
        :param max_abs_stat_veto: Maximal absolute value of statistic in units of SNr that is allowed. Larger
            ones will be masked. This is done becaause we are not interested in too big transits. And big
            outliers will pollute periodicity search.
        :type max_abs_stat_veto: float
        :param save_triggers_as_attribute: Whether to save all the checked triggers info like position, strength,
            veto scores as an attribute of the class. Is only used for debugging.
        :type save_triggers_as_attribute: bool
        :param save_no_veto_no_score_cor_stats_as_attr: Whether to save the statistics before they underwent the
            vetting and the score distribution gaussianization. These statistics may be needed in the postprocessing.
        :type save_no_veto_no_score_cor_stats_as_attr: bool.
        """

        self.matched_filtering(nsig_outliers_stat=nsig_outliers_stat)
        if save_no_veto_no_score_cor_stats_as_attr:
            self.stats_raw_no_veto_no_score_cor = copy.deepcopy(self.statistics_raw)
        if apply_veto:
            self.veto_statistics(apply_only_to_neg=apply_veto_only_to_neg, margin_snr=margin_snr_veto, max_abs_stat=max_abs_stat_veto,
                                 n_parts_veto_snr=n_parts_veto_snr, n_parts_veto_fourier=n_parts_veto_fourier,
                                 wind_len_overlap_fourier=wind_len_overlap_fourier, fn=fn_fit_veto,
                                 threshold_peak_sigma=threshold_peak_sigma_veto, save_triggers_as_attribute=save_triggers_as_attribute)
        self.cut_zero_padding_ends()
        self.calculate_ses_distr_corrections()
        if gaussianize:
            self.calculate_gaussianized_ses()

    def periodicity_search(self, p_min=Pmin, p_max=Pmax, dp=None, duty_cycle=Duty_cycle, periods=None,
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
        :param save_means: Whether to create a new attribure self.orig_means and save the mean
            values of the fluxes there. The default is True.
        :type save_means: bool
        :param method: which function to use in calculating the average
        :type method: str, 'mean' or 'median
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
            masks = [~np.isnan(self.orig_fluxes[q]) for q in self.quarters]
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
        new mean though because we do not wnat to work with too small numbers. We note that this changes also the values of
        the PSDs etc!
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
                           detr_ends_outliers=Detr_ends_outliers, detr_endlen_outliers=Detr_endlen_outliers,
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
        :param detr_ends_outliers: See the desctiption in `detrend_mov_ave`.
        :type detr_ends_outliers: str, 'symmetric' or 'asymmetric' or 'cut'.
        :param detr_endlen_outliers: Minimal length to take from the end for determining the average in the
            first (last) point. See the function documentation.
        :type detr_endlen_outliers: int or None
        :param in_whitened_fluxes: If true, the search will be performad not in self.fluxes but in self.whitened_fluxes_td
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
                                                                detr_ends=detr_ends_outliers,
                                                                detr_endlen=detr_endlen_outliers,
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
                                                                detr_ends=detr_ends_outliers,
                                                                detr_endlen=detr_endlen_outliers,
                                                                return_symmetric_mask=True)
                self.masks[q] *= outlier_mask
                if modify_clean_masks:
                    self.clean_masks[q] *= clean_mask

    def fill_additional_outliers(self, nsig=Nsig_outliers_0, nsig_positive=Nsig_outliers_0_pos,
                           nsig_negative=Nsig_outliers_0_neg, nsig_clean=Nsig_outliers_psd,
                           detr_length_outliers=Detr_length_outliers, detr_method_outliers=Detr_method_outliers,
                           detr_endmethod_outliers=Detr_endmethod_outliers,
                           detr_ends_outliers=Detr_ends_outliers, detr_endlen_outliers=Detr_endlen_outliers,
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
        :param detr_ends_outliers: See the desctiption in `detrend_mov_ave`.
        :type detr_ends_outliers: str, 'symmetric' or 'asymmetric' or 'cut'.
        :param detr_endlen_outliers: Minimal length to take from the end for determining the average in the
            first (last) point. See the function documentation.
        :type detr_endlen_outliers: int or None
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
                                                                detr_ends=detr_ends_outliers,
                                                                detr_endlen=detr_endlen_outliers,
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
                                                                detr_ends=detr_ends_outliers,
                                                                detr_endlen=detr_endlen_outliers,
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

    def make_big_hole_masks(self, max_small_hole=Max_small_hole, exclude_individual_points=False):
        """
        Remove from the masks small holes, that is holes that can be linearly interpolated and ignored.
        :param max_small_hole: Maximal hole to be considered small, so that it can be
            interpolated linearly and ignored
        :type max_small_hole: int
        :param exclude_individual_points: If i between two holes there is one non-hole point, also make it hole
        :type exclude_individual_points: bool
        """
        self.big_hole_masks = []
        for q in self.quarters:
            big_mask = ut.clean_small_holes(np.array(self.masks[q]), max_hole=max_small_hole)
            if exclude_individual_points:
                big_mask = np.array(~ut.clean_small_holes(np.array(~big_mask), max_hole=1))
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
        :return: None
        :rtype: None
        """
        if time_edges_to_mask is not None:
            assert hasattr(time_edges_to_mask, '__len__'), 'Variable time_edges_to_mask should have type list[list[float]]!'
            assert hasattr(time_edges_to_mask[0], '__len__'), 'Variable time_edges_to_mask should have type list[list[float]]!'
            for j in range(len(time_edges_to_mask)):
                ind1 = np.nan
                ind2 = np.nan
                for q in self.quarters:
                    if (time_edges_to_mask[j][0] >= np.nanmin(self.times[q])) and (time_edges_to_mask[j][0] <= np.nanmax(self.times[q])):
                        ind1 = np.argmin(np.abs(self.times[q] - time_edges_to_mask[j][0]))
                    if (time_edges_to_mask[j][1] >= np.nanmin(self.times[q])) and (time_edges_to_mask[j][1] <= np.nanmax(self.times[q])):
                        ind2 = np.argmin(np.abs(self.times[q] - time_edges_to_mask[j][1]))
                    if (not np.isnan(ind1)) or (not np.isnan(ind2)):
                        self.masks[q][np.nanmax([0, ind1]):np.nanmin([0, len(self.times[q])])] = False
                        self.big_hole_masks[q][np.nanmax([0, ind1]):np.nanmin([0, len(self.times[q])])] = False
        else:
            if not hasattr(transit_times, '__len__'):  # then it has to be float, and we will produce array now
                min_time = np.nanmin(self.times[0])
                max_time = np.nanmax(self.times[-1])
                if transit_times is not None:  # it should be float then
                    absolute_phase = transit_times % period
                assert period is not None and absolute_phase is not None, 'At least some parameters should be not None!'
                n_min = np.ceil(min_time / period - absolute_phase / period)
                n_max = np.floor(max_time / period - absolute_phase / period)
                transit_times = np.array([absolute_phase + n * period for n in np.arange(n_min, n_max + 1)])
            if hasattr(durations, '__len__'):
                assert len(transit_times) == len(durations), 'Variables `transit_times` and `durations` should either have equal length ' \
                                                             'or `durations` should be float! '
            else:
                durations = np.ones(len(transit_times)) * durations

            for j in range(len(transit_times)):
                half_dur_bins = int(np.ceil(durations[j] / self.dt_days / 2))
                # -> we will mask this to the left and to the right from the midpoint
                for q in self.quarters:
                    if (transit_times[j] >= np.nanmin(self.times[q])) and (transit_times[j] <= np.nanmax(self.times[q])):
                        ind1 = np.nanargmin(np.abs(self.times[q] - transit_times[j]))
                        self.masks[q][max(0, ind1 - half_dur_bins): min(len(self.times[q]), ind1 + half_dur_bins + 1)] = False
                        self.big_hole_masks[q][max(0, ind1 - half_dur_bins): min(len(self.times[q]), ind1 + half_dur_bins + 1)] = False

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
                     subtract_mean=Subtract_mean_in_psd, smoothen_psds=Smoothen_psds,
                     tukey_steps_len=Tukey_steps_len_psd,
                     always_zero_zeroth_freq=Always_zero_zeroth_freq_psd,
                     feature_sizes=(1, 9, 3.5, 4, 2., 3)):
        """
        PSDS as function of quarter id.
        The PSD measuring algotithm makes the evenly sampled steps and not adjust the search for
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
        :param smoothen_psds: whether to apply multi-resolution smoothing to mitigate the noise effects
        :type smoothen_psds: bool
        :param tukey_steps_len: length taken for zeroing in the tukey windowing
        :type tukey_steps_len: int or float
        :param always_zero_zeroth_freq: whether to always put the lowest frequencies to zero
        :type always_zero_zeroth_freq: bool
        :param feature_sizes: length and depth of the deviations that willtrigger increasing nperseg, see in the code
        :type feature_sizes:
        """
        d1, w1, d2, w2, d3, w3 = feature_sizes
        if nperseg is None:
            nperseg = self.nperseg_psd

        if not self.is_prepared_for_psd:
            warnings.warn("Data is not prepared for PSD. Use prepare_for_psd method.")

        if self.is_fluxes_inpainted:
            warnings.warn("Fluxes have been inpainted, new PSD will make inpainting invalid for detection.")

        fftfun = np.fft.rfft
        ifftfun = np.fft.irfft
        freqfun = np.fft.rfftfreq
        dtype_res = np.float64

        if subtract_mean:
            clean_fluxes = [self.clean_fluxes[i] - np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]]) for i in
                            range(len(self.clean_fluxes))]
        else:
            clean_fluxes = self.clean_fluxes

        self.orig_psds, self.psdfreqs = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,
                                                      nperseg=nperseg, noverlap=noverlap, window=window,
                                                      return_freqs=True, dt=self.dt_days, return_npsds=False)

        self.freqs = [freqfun(len(time), d=self.dt_days) for time in self.times]

        #############################################################################################################
        # Smoothing
        #############################################################################################################
        if smoothen_psds:
            nperseg_two = 2 * nperseg
            nperseg_coarse = nperseg // 2
            nperseg_rude = nperseg // 4
            nperseg_smooth = nperseg // 8
            psds_two = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,  # overlap is relative!
                                     nperseg=nperseg_two, noverlap=noverlap, window=window,
                                     return_freqs=False, dt=self.dt_days, return_npsds=False)
            psds_coarse = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,  # overlap is relative!
                                        nperseg=nperseg_coarse, noverlap=noverlap, window=window,
                                        return_freqs=False, dt=self.dt_days, return_npsds=False)
            psds_smooth = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,  # overlap is relative!
                                        nperseg=nperseg_smooth, noverlap=noverlap, window=window,
                                        return_freqs=False, dt=self.dt_days, return_npsds=False)
            psds_rude = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,  # overlap is relative!
                                      nperseg=nperseg_rude, noverlap=noverlap, window=window,
                                      return_freqs=False, dt=self.dt_days, return_npsds=False)

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
                           subtract_mean=Subtract_mean_in_psd, rfft=Use_rfft, smoothen_psds=Smoothen_psds):
        """
        PSDS as function of quarter id.
        The PSD measuring algotithm makes the evenly sampled steps and not adjust the search for
        good pieces according to the placement of the zeros in mask. That is why
        the number of npsds is unoptimal.
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
        :param rfft: If True, the signal is treated as real, and all ffts are rffts. The length of the frequency samples
            are the nperseg//2+1.
            Default is True
        :type rfft: bool
        :param smoothen_psds:
        :type smoothen_psds:
        :return: creates self.psds, self.psdfreqs, self.wf_td, self.wf_fd, self.freqs, seld.npsds, self.nperseg.
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
            if rfft:
                for i in self.quarters:
                    clean_flux = self.clean_fluxes[i] - np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]])
                    psds.append([np.abs(np.fft.rfft(windowdict[window] * clean_flux[ind1:ind1 + nperseg])
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
            if rfft:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.rfft(windowdict[window] * (self.clean_fluxes[i])[ind1:ind1 + nperseg])
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

        if smoothen_psds:
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
                wf_td, wf_fd = ut.psd2wf(self.psds[i], self.support, rfft=rfft, always_zero_zeroth_freq=True)
            else:
                wf_td, wf_fd = ut.psd2wf(self.psds[i], len(self.fluxes[i]), rfft=rfft, always_zero_zeroth_freq=True)
            self.wf_tds.append(wf_td)
            self.wf_fds.append(wf_fd)
        self.wf_tds = np.array(self.wf_tds)  # for numba need numpy.arrays

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
        self.wf_tds = np.array(wf_tds_new)  # np.array([np.fft.fftshift(wf_td) for wf_td in wf_td_new])
        self.wf_fds = wf_fds_new

    def fftshift_wf_fds(self):
        self.wf_tds = np.array([np.fft.fftshift(wf_td) for wf_td in self.wf_tds])

    def interpolate_fluxes(self, interpolate_clean_fluxes=True):
        """
        Interpolates all the holes in the holes in the fluxes, including big holes and end holes.
        In the end holes, the fluxes are padded by a constant value equal to the first(last) value of the array.
        Masks are not modified because measuring PSD or detecting with this interpolation makes no sense.
        """
        for q in self.quarters:
            self.fluxes[q] = ut.interpolate_over_mask(np.array(self.fluxes[q]), np.array(self.masks[q]),
                                                      interp_method='linear', ends='continue',
                                                      mitigate_hole_ends=False, endhole_method='median')
            if interpolate_clean_fluxes:
                self.clean_fluxes[q] = ut.interpolate_over_mask(np.array(self.clean_fluxes[q]),
                                                                np.array(self.clean_masks[q]) * self.big_hole_masks[q],
                                                                interp_method='linear', ends='continue',
                                                                mitigate_hole_ends=False, endhole_method='median')
            self.is_holes_interpolated = True

    def subtract_low_trends(self, nperseg=None, save_low_trends=False):
        """
        Subtract trends of low frequencues which cannot be resolved by the PSD measurement
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
        self.fluxes = self.inpaint_fluxes_fast(self.fluxes, self.masks, self.wf_tds, dtype=Dtype_r)
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
            whitened_flux_fd = np.fft.rfft(self.fluxes[q]) * self.wf_fds[q]
            self.whitened_fluxes_fd.append(np.array(whitened_flux_fd, dtype=Dtype_c))
            self.whitened_fluxes_td.append(np.array(np.fft.irfft(whitened_flux_fd), dtype=Dtype_r))

        if whiten_clean_fluxes:
            self.whitened_clean_fluxes_fd = []
            self.whitened_clean_fluxes_td = []
            for q in self.quarters:
                whitened_clean_flux_fd = np.fft.rfft(self.clean_fluxes[q]) * self.wf_fds[q]
                self.whitened_clean_fluxes_fd.append(np.array(whitened_clean_flux_fd, dtype=Dtype_c))
                self.whitened_clean_fluxes_td.append(np.array(np.fft.irfft(whitened_clean_flux_fd), dtype=Dtype_r))

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

        :param templates: Vectors to be used as a template (time domain). Should be
            all of the same length. The working part of the template is
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

    def get_not_intelrleaved_index(self, t=None, tsh=None):
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
                whitened_temp_fd = np.fft.rfft(self.tb.templates_shifts[k]) * self.wf_fds[q]
                self.whitened_templates_fd[k][q] = whitened_temp_fd
                self.whitened_templates_td[k][q] = np.fft.irfft(whitened_temp_fd)
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
                stat = np.real(np.fft.irfft(np.conjugate(self.whitened_templates_fd[k][q]) * self.whitened_fluxes_fd[q]))
                stat = np.array(stat[self.zero_padding_ends[q][0]:len(stat) - self.zero_padding_ends[q][1]])
                # The statistic was cut historically because of memory, now there is no reason other than already written code
                var_calc = np.fft.irfft(np.abs(self.whitened_templates_fd[k][q]) ** 2)[0]
                # Zero-padding ends are not a problem because they are zeros, and the normalization is fixed by irfft
                statmask = np.array(self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(
                    self.big_hole_masks[q]) - self.zero_padding_ends[q][1]])

                eff_temp_wid = len(np.where(self.tb.templates_shifts[k] >= np.max(self.tb.templates_shifts[k]) / 2)[0])
                gap_len = 2 * eff_temp_wid  # It was set empirically for typical data and templates
                gap_len = max(ngap_moving_var_min, gap_len)
                wind_len = 30 * eff_temp_wid  # It was measured empirically for typical data and templates
                wind_len = max(wind_len, nwind_moving_var_min)
                nsamples_min = int(nsamples_min_moving_var_coef * (wind_len - gap_len))
                movvar_meas, tot_var = ut.moving_var(np.array(stat),
                                                     mask=np.array(statmask),
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
                indices_sign = statmask & (stat > 0)
                var_snr_indices = np.nanmean((stat_snr[indices_sign]) ** 2)
                movvar_meas[indices_sign] *= var_snr_indices
                # Making negative values unit variance:
                indices_sign = statmask & (stat < 0)
                var_snr_indices = np.nanmean((stat_snr[indices_sign]) ** 2)
                movvar_meas[indices_sign] *= var_snr_indices
                # Use this new moving variance for all the rest:
                stat_snr = stat / movvar_meas ** 0.5  # now should be N(0,1) for sure
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

    def invert_statistic(self, choose_random=False):
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

    def veto_statistics(self, apply_only_to_neg=False,
                        margin_snr=Margin_snr_veto,
                        max_abs_stat=Max_abs_stat_veto,
                        fn=Fn_fit_veto,
                        n_parts_veto_snr=N_parts_veto_snr,
                        min_bins_veto_part_time=Min_bins_veto_part_time,
                        lobe_hlen_veto_part_time=Lobe_hlen_veto_part_time,
                        n_parts_veto_fourier=N_parts_veto_fourier,
                        wind_len_overlap_fourier=Wind_len_overlap_fourier,
                        threshold_peak_sigma=Threshold_peak_sigma_veto,  # there will also be look-elsewhere
                        save_triggers_as_attribute=False):
        """
        Verify big triggers and mask ones which are non-transit-like.
        Selects from the statistic the triggers whose abs is >margin_snr, which are negative or
        both negative and positive, as defined by `apply_only_to_neg` parameter.
        Triggers inside big holes are ignored.
        Makes several sests: chi-square partial amplitude tests in time and fourier domain,
        alternative hypothesis testing against a peak.
        Should be done before unpadding!
        :param max_abs_stat: Maximal absolute value of statistic in units of SNr that is allowed. Larger
            ones will be masked. This is done becaause we are not interested in too big transits. And big
            outliers will pollute periodicity search.
        :type max_abs_stat: float
        :param apply_only_to_neg: Whether to apply the veto only to negative statistics, which transits are
            expected to be, or also to the positive ones. The latter is needed if we want to look at the
            inverted data. It also can affetct the negative values processing if there are interactions.
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
                    decorr_coef = (np.fft.irfft(self.whitened_templates_fd[k][q] * np.conjugate(whitened_peak_temps_fd[q]))[0] /
                                   ((whitened_peak_temps_td[q] @ whitened_peak_temps_td[q])
                                    * (self.whitened_templates_td[k][q] @ self.whitened_templates_td[k][q])) ** 0.5)  # not rolled, approx
                    max_peak_stats = np.array([np.max(stats_peak[q][max(0, i - h_dur_check): i + h_dur_check])
                                               for i in range(len(stats_peak[q]))])
                    min_peak_stats = np.array([np.min(stats_peak[q][max(0, i - h_dur_check): i + h_dur_check])
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
                isgood_parts = ut.veto_parts_fourier(data_wh_fd=fluxes_wh_fd_rewh[q],
                                                     template_wh_fd=whitened_templates_fd_renorm[k][q],
                                                     indices_padded=trigger_indices_quarter_padded,
                                                     mask_padded=stat_mask_padded, n_parts=n_parts_veto_fourier,
                                                     fn=fn, wind_len_overlap=wind_len_overlap_fourier, nperseg_psd=self.nperseg_psd)
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
        peak_template = np.fft.rfft(peak_template)
        whitened_peak_temps_fd = [peak_template * self.wf_fds[q] for q in self.quarters]
        whitened_peak_temps_td = [np.fft.irfft(whitened_peak_temps_fd[q]) for q in self.quarters]
        stats_peak = [np.real(np.fft.irfft(np.conjugate(whitened_peak_temps_fd[q]) * self.whitened_fluxes_fd[q]))
                      for q in self.quarters]
        stats_peak = [stats_peak[q] / ut.outlier_free_std(stats_peak[q], nsig=3, n_iterations=2,
                                                          mask=self.big_hole_masks[q]) for q in self.quarters]
        stats_peak = [stats_peak[q][self.zero_padding_ends[q][0]:len(stats_peak[q]) - self.zero_padding_ends[q][1]]
                      for q in self.quarters]
        # -> CUT THE STATISTIC BECAUSE STATISTICS_SNR IS ALSO CUT
        return stats_peak, whitened_peak_temps_td, whitened_peak_temps_fd

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
            self.corrections_full[k] = af.connect_quarters_fun(self.times, self.log_non_gaussian_corrections[k],
                                                               zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                               is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                               fill_gaps_by_nan=False)[1]
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
        :param periods_to_check: List of periods to perform the chack on them 
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
        the big holes in the data. Every big hole will orrespond to a cut. Most of the cuts are just along the big holes.
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
            self.inds_pieces_scramble = inds_pieces_scramble
            np.random.shuffle(self.inds_pieces_scramble)
        else:
            numbers_shuffle = np.arange(len(self.inds_pieces_scramble)).astype(int)
            np.random.shuffle(numbers_shuffle)
            self.inds_pieces_scramble = self.inds_pieces_scramble[numbers_shuffle]  # save to be able to unscramble
            inds_pieces_scramble = self.inds_pieces_in_scrambled[numbers_shuffle]  # to rescramble the present scrambled pieces

        np.random.set_state(random_state)  # returning to make the state controllable
        # How many spaces between the pieces and plus one in the end, although it effectively changes the length:
        len_holes = ut.gen_uniform_piece_lengths(tot_len=self.tot_hole_len_scramble, n_pieces=len(inds_pieces_scramble) - 1)
        # `self.connect_quarters` is now destroyed. do it again to restore the original positioning
        self.big_hole_masks_full, self.inds_pieces_in_scrambled = ut.rearrange_pieces(self.big_hole_masks_full,
                                                                                      inds_pieces=inds_pieces_scramble,
                                                                                      hole_lens=len_holes, fill_hole_with=0.,
                                                                                      return_inds_target_res=True)
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
            assert (shift_numbers / shift_numbers[0] == np.ones(len(
                shift_numbers))).all(), 'Templates with shifts must have the same number of shifts!'
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
        if hasattr(self, 'trends_low'):
            del self.trends_low
            self.trends_low = None
        if hasattr(self, 'clean_fluxes'):
            del self.clean_fluxes
            self.clean_fluxes = None
        if hasattr(self, 'clean_masks'):
            del self.clean_masks
            self.clean_masks = None
        if hasattr(self, 'whitened_clean_fluxes_fd'):
            del self.whitened_clean_fluxes_fd
            self.whitened_clean_fluxes_fd = None
        if hasattr(self, 'whitened_clean_fluxes_td'):
            del self.whitened_clean_fluxes_td
            self.whitened_clean_fluxes_td = None
        if hasattr(self, 'whitened_templates_fd'):
            del self.whitened_templates_fd
            self.whitened_templates_fd = None
        if hasattr(self, 'whitened_templates_td'):
            del self.whitened_templates_td
            self.whitened_templates_td = None
        if hasattr(self, 'notch_soss'):
            del self.notch_soss
            self.notch_soss = None
        if hasattr(self, 'wf_td'):
            del self.wf_tds
            self.wf_tds = None
        if hasattr(self, 'wf_fd'):
            del self.wf_fds
            self.wf_fds = None
        if hasattr(self, 'psd_masks'):
            del self.psd_masks
            self.psd_masks = None
        if hasattr(self, 'orig_psds'):
            del self.orig_psds
            self.orig_psds = None
        if hasattr(self, 'wf_fds_meas'):
            del self.wf_fds_meas
            self.wf_fds_meas = None
        if hasattr(self, 'wf_fds_unwindowed'):
            del self.wf_fds_unwindowed
            self.wf_fds_unwindowed = None
        if hasattr(self, 'unnotched_whitened_clean_fluxes_fd'):
            del self.unnotched_whitened_clean_fluxes_fd
            self.unnotched_whitened_clean_fluxes_fd = None
        if hasattr(self, 'npsds0'):
            del self.npsds0
            self.npsds0 = None
        if hasattr(self, 'npsds'):
            del self.npsds
            self.npsds = None
        if hasattr(self, 'notch_mask'):
            del self.notch_mask
            self.notch_mask = None
        if hasattr(self, 'tb'):
            del self.tb
            self.tb = None
        if hasattr(self, 'stats_raw_no_veto_no_score_cor'):
            del self.stats_raw_no_veto_no_score_cor
            self.stats_raw_no_veto_no_score_cor = None
        if hasattr(self, 'veto_masks'):
            del self.veto_masks
            self.veto_masks = None
        if hasattr(self, 'veto_bit_masks'):
            del self.veto_bit_masks
            self.veto_bit_masks = None
        if hasattr(self, 'trigger_indices'):
            del self.trigger_indices
            self.trigger_indices = None

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

    def sum_periods(self, p_min=Pmin, p_max=Pmax, dp=None, duty_cycle=Duty_cycle, periods=None, safety_bins=P_bins_overlap):
        """
        Sums the statistics with all the possible periods and initial phases. Folds the statistics and their
        variances and caclulates the score of the folded planet. Folds also single-event distribution corrections
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

        self.period_overlap_bins = safety_bins
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
            len_pad = int(np.ceil(Pmax / dt))
            self._phases_interleaved = np.arange(int(np.ceil((self.periods[-1]) / dt))) * dt
            # These phases are with respect to the beginning of the data which can be any
            # number because sometimes we reject the zeroth quarter or do other things.
            # Do not use these phases anywhere.
            # Warning: we only try these phases and not something in-between. It should be ok, but still
            statistics_by_trends = np.array(self.statistics_interleaved, dtype=Dtype_r)  # * self.trend_orig_flux_interleaved
            # -> trends are now all normalized to be the same!
            # Pad by zeros to avoid extra if statements in the summation:
            statistics_by_trends = np.hstack((statistics_by_trends, np.zeros((len(statistics_by_trends), len_pad))))
            trends_sq_by_vars = np.array(self.stat_vars_interleaved, dtype=Dtype_r)  # * self.trend_orig_flux_interleaved ** 2
            # -> trends are all normalized to be the same!
            trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros((len(trends_sq_by_vars), len_pad))))
            corrections = np.array(self.corrections_interleaved, dtype=Dtype_r)
            corrections = np.hstack((corrections, np.zeros((len(corrections), len_pad))))
            statistics_by_trends_gaussianized = np.array(self.statistics_gaussianized_interleaved, dtype=Dtype_r)
            statistics_by_trends_gaussianized = np.hstack((statistics_by_trends_gaussianized,
                                                           np.zeros((len(statistics_by_trends_gaussianized), len_pad))))
            self.statistics_periodic_interleaved, denomsq, self.corrections_periodic_interleaved, \
                self.stats_gaussianized_periodic_interleaved = self.sum_periods_fast(
                    statistics_by_trends, trends_sq_by_vars, corrections, statistics_by_trends_gaussianized,
                    dt, self.tmax + self.dt_days * (1 - 1 / self.n_shifts), self.tmin, periods)
            # attention: we use the same dt for interleaved and not? it orig_means that the phase step for the interleaved could be smaller.
            for p in range(len(self.statistics_periodic_interleaved)):
                for t in range(len(self.statistics_periodic_interleaved[p])):
                    self.statistics_periodic_interleaved[p][t][denomsq[p][t] == 0] = np.nan
                    self.statistics_periodic_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][denomsq[p][t] != 0] ** 0.5
                    self.stats_gaussianized_periodic_interleaved[p][t][denomsq[p][t] == 0] = np.nan
                    self.stats_gaussianized_periodic_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][denomsq[p][t] != 0] ** 0.5

        # Not interleaved templates (which did not require subbin shifts)
        dt = self.dt_days
        len_pad = int(np.ceil(Pmax / dt))
        if len(self.not_interleaved_numbers) > 0:
            self._phases_not_interleaved = np.arange(int(np.ceil((self.periods[-1]) / dt))) * dt
            statistics_by_trends = np.array(self.statistics_not_interleaved, dtype=Dtype_r)  # * self.trend_orig_flux_not_interleaved
            statistics_by_trends = np.hstack((statistics_by_trends, np.zeros((len(statistics_by_trends), len_pad))))
            # These phases are with respect to the beginning of the data which can be any
            # number because sometimes we reject the zeroth quarter or do other things.
            # Do not use these phases anywhere.
            # Pad by zeros to avoid extra if statements in the summation:
            trends_sq_by_vars = np.array(self.stat_vars_not_interleaved, dtype=Dtype_r)  # * self.trend_orig_flux_not_interleaved ** 2
            trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros((len(trends_sq_by_vars), len_pad))))
            corrections = np.array(self.corrections_not_interleaved)
            corrections = np.hstack((corrections, np.zeros((len(corrections), len_pad))))
            statistics_by_trends_gaussianized = np.array(self.statistics_gaussianized_not_interleaved, dtype=Dtype_r)
            statistics_by_trends_gaussianized = np.hstack((statistics_by_trends_gaussianized, np.zeros(
                (len(statistics_by_trends_gaussianized), len_pad))))
            self.statistics_periodic_not_interleaved, denomsq, self.corrections_periodic_not_interleaved, \
                self.stats_gaussianized_periodic_not_interleaved = self.sum_periods_fast(
                    statistics_by_trends, trends_sq_by_vars, corrections, statistics_by_trends_gaussianized,
                    dt, self.tmax, self.tmin, periods)
            for p in range(len(self.statistics_periodic_not_interleaved)):
                for t in range(len(self.statistics_periodic_not_interleaved[p])):
                    self.statistics_periodic_not_interleaved[p][t][denomsq[p][t] == 0] = np.nan
                    self.statistics_periodic_not_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][denomsq[p][t] != 0] ** 0.5
                    self.stats_gaussianized_periodic_not_interleaved[p][t][denomsq[p][t] == 0] = np.nan
                    self.stats_gaussianized_periodic_not_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][denomsq[p][t] != 0] ** 0.5

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
        self.stats_gaussianized_periodic = np.zeros_like(self.statistics_periodic)
        self.corrections_periodic = np.zeros_like(self.statistics_periodic)

        not_interl_inds = n_interl + np.arange(np.shape(self.statistics_periodic_not_interleaved)[1])
        self.statistics_periodic[:, not_interl_inds, :] = np.array(self.statistics_periodic_not_interleaved)
        self.stats_gaussianized_periodic[:, not_interl_inds, :] = np.array(self.stats_gaussianized_periodic_not_interleaved)
        self.corrections_periodic[:, not_interl_inds, :] = np.array(self.corrections_periodic_not_interleaved)
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
                res = np.concatenate((vals_interl[ip][k], np.ones(delta_len) * vals_interl[ip][k][-1]))
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
                    res = np.concatenate((vals_interl[ip][k], np.zeros(delta_len)))
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
                                             np.shape(self.statistics_periodic_interleaved)[2]))
        self.stats_gaussianized_periodic = np.zeros_like(self.statistics_periodic)
        self.corrections_periodic = np.zeros_like(self.statistics_periodic)

        self.statistics_periodic[:, range(n_interl), :] = np.array(self.statistics_periodic_interleaved)
        self.stats_gaussianized_periodic[:, range(n_interl), :] = np.array(self.stats_gaussianized_periodic_interleaved)
        self.corrections_periodic[:, range(n_interl), :] = np.array(self.corrections_periodic_interleaved)
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
                        signs = np.sign(stats)
                        stats_squared = stats ** 2 + 2 * cors_periodic[i]
                        stats_squared[stats_squared < 0] = 0.
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
        if m_1_3_r_s is None:
            m_1_3_r_s = self.m_1_3_r_s
        log_template_prior, m13r_bin_edges, period_edges = tm.load_template_prior()
        m_1_3_r_s_bin = min(max(0, np.searchsorted(m13r_bin_edges, m_1_3_r_s) - 1), len(m13r_bin_edges) - 2)

        self.stats_periodic_mar_temp = np.zeros((np.shape(self.statistics_periodic)[0], np.shape(self.statistics_periodic)[2]))
        self.stats_gaussianized_periodic_mar_temp = np.zeros_like(self.stats_periodic_mar_temp)

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
                                                         for pp in range(len(self.stats_periodic_mar_temp))])
        phases_best_ph = np.array([self._phases[best_phase_inds[pp]] for pp in range(len(self.stats_periodic_mar_temp))])
        self.stats_gaussianized_periodic_mar_temp_best_ph = np.array([self.stats_gaussianized_periodic_mar_temp[pp][best_phase_inds[pp]]
                                                                      for pp in range(len(self.stats_periodic_mar_temp))])
        self.first_transit_times_best_ph = self.tmin + phases_best_ph

        # Same for gaussianized
        best_g_phase_inds = np.array([np.nanargmax(self.stats_gaussianized_periodic_mar_temp[pp])
                                      for pp in range(len(self.stats_gaussianized_periodic_mar_temp))])
        self.best_g_phase_inds = best_g_phase_inds
        self.stats_periodic_mar_temp_best_g_ph = np.array([np.nanmax(self.stats_gaussianized_periodic_mar_temp[pp])
                                                           for pp in range(len(self.stats_gaussianized_periodic_mar_temp))])
        phases_best_g_ph = np.array([self._phases[best_g_phase_inds[pp]] for pp in range(len(self.stats_gaussianized_periodic_mar_temp))])
        self.stats_gaussianized_periodic_mar_temp_best_g_ph = np.array([self.stats_gaussianized_periodic_mar_temp[pp][best_g_phase_inds[pp]]
                                                                        for pp in range(len(self.stats_gaussianized_periodic_mar_temp))])
        self.first_transit_times_best_g_ph = self.tmin + phases_best_g_ph

        if delete_non_maximized:
            del self.stats_periodic_mar_temp
            self.stats_periodic_mar_temp = None
            del self.stats_gaussianized_periodic_mar_temp
            self.stats_gaussianized_periodic_mar_temp = None

    def get_integral_score_at_max(self, per_ind_max=None, phase_ind_max=None, gaussianized=False,
                                  dphi=D_period_integral_score, dpi=D_phase_integral_score, m_1_3_r=None):
        """
        Calculates the integral score around the best trigger. The integral score is the sum of all the statistics
        in the period-phase plane arount the specified maximum. The point should really be a maximum because the
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
        """
        _, __, log_occurrences_templates_periods, m13r_bin_edges, period_edges = tm.load_volume_prior_precomputed_terms()
        if m_1_3_r is None:
            m_1_3_r = self.m_1_3_r_s
        m_1_3_r_bin = min(max(0, np.searchsorted(m13r_bin_edges, m_1_3_r) - 1), len(m13r_bin_edges) - 2)

        if not gaussianized:
            if per_ind_max is None:
                per_ind_max = self.best_per_index
            if phase_ind_max is None:
                phase_ind_max = self.best_phase_index
            peak_val = self.stats_periodic_mar_temp[per_ind_max, phase_ind_max]
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
            peak_val = self.stats_gaussianized_periodic_mar_temp[per_ind_max, phase_ind_max]
            per_ind_left = max(0, per_ind_max - dpi)
            phase_ind_left = max(0, phase_ind_max - dphi)
            per_ind_right = min(per_ind_max + dpi, self.stats_gaussianized_periodic_mar_temp.shape[0])
            phase_ind_right = min(phase_ind_max + dphi, self.stats_gaussianized_periodic_mar_temp.shape[1])
            vals = np.array(self.stats_gaussianized_periodic_mar_temp)

        vals = vals[per_ind_left:per_ind_right][:, phase_ind_left:phase_ind_right]
        peak_coords_relative = per_ind_max - per_ind_left, phase_ind_max - phase_ind_left

        ambient_val = np.nanmedian(vals)
        inds_sum = (vals > ambient_val + 0.5 * (peak_val - ambient_val))
        ambient_val = np.nanmedian(vals[~inds_sum])  # repetitive median to exclude peak
        inds_sum = (vals >= ambient_val + 0.5 * (peak_val - ambient_val))
        inds_sum = ut.select_connected_component(inds_sum, center=peak_coords_relative)

        dp = self.periods[max(1, per_ind_max)] - self.periods[max(0, per_ind_max - 1)]
        dftt = self._phases[max(1, phase_ind_max)] - self._phases[max(0, phase_ind_max - 1)]
        vals_to_sum = vals[inds_sum]
        vals_to_sum = vals_to_sum[np.abs(vals_to_sum) != np.inf]
        max_val = np.nanmax(vals_to_sum)
        norm_vals_to_sum = vals_to_sum - max_val
        integr_score = max_val + np.log(np.nansum(np.exp(norm_vals_to_sum))) + np.log(dp * dftt)
        # integr_score = np.log(np.nansum(np.exp(vals[inds_sum])) * dp * dftt)

        per_bin = min(max(0, np.searchsorted(period_edges, self.periods[per_ind_max]) - 1), len(period_edges) - 2)
        log_occurrence = log_occurrences_templates_periods[m_1_3_r_bin, per_bin]
        log_occurrence = np.log(np.sum(np.exp(log_occurrence)))  # sum over templates
        # log_occurrences_templates_periods is normalized by the m_1_3_r to have the full astrophysical occurrence rate
        # in every m_1_3_r bin. units are rate! no probability normalization!
        return integr_score + log_occurrence

    def get_best_template_index(self, with_prior=True, in_gaussianized=False, ind_period=None, ind_phase=None, period=None):
        """
        Select the template which had the highest score before the marginalization over the templates was done.
        Looks at a specific position specified by `ind_period` and `ind_phase`.
        Attention: the non-margializeed scores should not have been deleted!
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
        was done per template. Since before the template marginalization the downsamplinf was done, the interleaved first
        transit time resolution is lost. And the summed values may be needed to search for in the interleaved statisitcs.
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
                         min_required_nonzero_transits_fraction=Min_required_nonzero_transits_fraction):
        """
        TODO: write
        # ATTENTION. SINCE WE USE VETO OF SES, THERE WILL BE ZEROS, AND THE INTEGRAL WILL NOT BE FULL
        # THEREFORE WE IN GENERAL LOSE SNR, BOT ALSO THE DISTRIBUTION WILL HAVE AN UNKNOWN VARIANCE

        ATTENTION WARNING. The search was done on the statisitcs marginalized over templates. It is not well-defined then to
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
        self.best_integral_score : The inegral over statistic with prior around the maximum
        And the same with index _g_ for the maximum over the gaussianized score.
        """
        if not in_gaussianized:
            stats_periodic_mar_temp = self.stats_periodic_mar_temp
            statistics_periodic = self.statistics_periodic
        else:
            stats_periodic_mar_temp = self.stats_gaussianized_periodic_mar_temp
            statistics_periodic = self.stats_gaussianized_periodic

        # Finding period-phase bests list
        shap_x = stats_periodic_mar_temp.shape[1]
        flat_inds = ut.nanargsort(stats_periodic_mar_temp[self.period_overlap_bins: len(
            stats_periodic_mar_temp) - self.period_overlap_bins, :].flatten())
        sorting_inds = np.array([flat_inds // shap_x + self.period_overlap_bins, flat_inds % shap_x]).T[::-1]
        if max_trials is not None:
            max_trials = min(max_trials, len(sorting_inds))
        else:
            max_trials = len(sorting_inds)

        if apply_veto_mes:
            veto_mes_masks = np.ones_like(stats_periodic_mar_temp).astype(bool)
            assert statistics_periodic is not None, 'self.statistics_periodic needed for finding the best template'
            best_per_index = None
            best_phase_index = None
            is_good = False
            index_counter = 0
            while (not is_good) and (index_counter < max_trials):
                best_per_index = sorting_inds[index_counter][0]
                best_phase_index = sorting_inds[index_counter][1]
                best_period = self.periods[best_per_index]

                best_ftt = self.tmin + self._phases[best_phase_index]
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
                else:
                    p_val = ut.unequal_amplitude_veto(summed_stats_raw, stat_vars=summed_vars, negative=True)
                    if p_val >= fn_threshold:
                        is_good = True
                    else:
                        veto_mes_masks[best_per_index, best_phase_index] = False
                        # WARNING: we are zeroing the statistic which did not pass the veto for
                        # this statistic not to impact the integral score
                        self.stats_periodic_mar_temp[best_per_index, best_phase_index] = 0.
                index_counter += 1
            if save_veto_masks:
                self.veto_mes_masks = veto_mes_masks
            if not is_good:
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
                self.best_integral_score = self.get_integral_score_at_max(per_ind_max=self.best_per_index,
                                                                          phase_ind_max=self.best_phase_index, gaussianized=False)
            else:
                self.best_period = np.nan
                self.best_ftt = np.nan
                self.best_template_index = np.nan
                self.best_stat = np.nan
                self.best_stat_gaussianized = np.nan
                self.best_integral_score = np.nan

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
                self.best_g_integral_score = self.get_integral_score_at_max(per_ind_max=self.best_g_per_index,
                                                                            phase_ind_max=self.best_g_phase_index, gaussianized=True)
            else:
                self.best_g_template_index = np.nan
                self.best_g_stat = np.nan
                self.best_g_stat_gaussianized = np.nan
                self.best_g_ftt = np.nan
                self.best_g_integral_score = np.nan

    def make_best_periodic_scores(self):
        """
        Makes the formatting of the best results that will be accessed by the `target_class`.
        """
        best_periodic_scores_data = [[np.nan, np.nan] for _ in range(Best_per_results_length)]
        for ind, val, val_gaus in [[Best_per_results_index_per, self.best_period, self.best_g_period],
                                   [Best_per_results_index_ftt, self.best_ftt, self.best_g_ftt],
                                   [Best_per_results_index_stats, self.best_stat, self.best_g_stat],
                                   [Best_per_results_index_stats_gauss, self.best_stat_gaussianized, self.best_g_stat_gaussianized],
                                   [Best_per_results_index_best_integral, self.best_integral_score, self.best_g_integral_score],
                                   [Best_per_results_index_template, self.best_template_index, self.best_g_template_index]]:
            best_periodic_scores_data[ind][0] = val
            best_periodic_scores_data[ind][1] = val_gaus
        best_periodic_scores_data[Best_per_results_index_per_list] = self.periods
        self.best_periodic_scores_data = best_periodic_scores_data


def sum_periods_fun(statistics_by_trends, trends_sq_by_vars, corrections, statistics_by_trends_gaussianized,
                    dt, tmax, tmin, periods):
    """
    Folds and sums the statistics with `periods` and all the possible initial phases. Accounts for
    subbin template shifts, accounts for masks.

    :param statistics_by_trends: Single-events statistics multiplied by the mean flux trend, which is assumed to be
        proportional to the expected amplitude of the transits. If the statistic was already normalized by trends,
        then no need for trends.
    :type statistics_by_trends: numpy.ndarray 2d of float
    :param trends_sq_by_vars: Single-events statistics multiplied by the squared mean flux trend, which is assumed to be
        proportional to the expected amplitude of the transits. If the statistic was already normalized by trends,
        then no need for trends.
    :type trends_sq_by_vars: numpy.ndarray 2d of float
    :param corrections: Log corrections for the single-event statistic distribution non-gaussianity. They need to be
        also folded and then applied to the folded statistics in the end.
    :type corrections: numpy.ndarray 2d of float
    :param statistics_by_trends_gaussianized: Gaussianized single-events statistics multiplied by the mean flux trend, which
        is assumed to be proportional to the expected amplitude of the transits. If the statistic was already normalized by
        trends, then no need for trends.
    :type statistics_by_trends_gaussianized: numpy.ndarray 2d of float
    :param dt : Time step of equally sampled time axis.
    :type dt : float
    :param tmax : Maximal time in the time axis.
    :type tmax : float
    :param tmin : Minimal time in the time axis.
    :type tmin : float
    :param periods: Grid of periods in the same units as `dt`, `tmax`, `tmin` by which the statistics are to be folded.
    :type periods: numpy.ndarray 1d of float
    :return: statistics_periodic, denomsq, corrections_periodic, stats_gaussianized_periodic, see description below.
    :rtype: tuple of numpy.ndarrays 3d of float

    Returns (all are numpy.ndarray 3d of float):
    statistics_periodic : array of folded statistics for all the periods, templates and phases. Has raw units, is to be
        normalized by sqrt of `denomsq` to get the units of SNR. Has to be corrected by `corrections_periodic` to account
        properly for the distribution.
    denomsq : array of folded statistic variances (with mean trends normalization) for all the periods, templates and phases.
        These denominators are used to bring the statistics to units of SNR.
    corrections_periodic : Folded log corrections for the single-event statistic distribution non-gaussianity.
    stats_gaussianized_periodic : array of folded guassianized statistics for all the periods, templates and phases. Has raw
        units, is to be normalized by sqrt of `denomsq` to get the units of SNR. Has to be corrected by `corrections_periodic`
        to account properly for the distribution.
    """
    phases_size = int(np.ceil((periods[-1]) / dt))
    statistics_periodic = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
    denomsq = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
    corrections_periodic = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
    stats_gaussianized_periodic = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
    for P_ind in range(len(periods)):
        p = periods[P_ind]
        n_transits = int(np.ceil((tmax - tmin) / p))
        phaselen = int(p / dt)
        for n in range(n_transits):
            time_ind = int(np.round((p * n / dt)))
            statistics_periodic[P_ind, :, :phaselen] += statistics_by_trends[:, time_ind:time_ind + phaselen]
            denomsq[P_ind, :, :phaselen] += trends_sq_by_vars[:, time_ind:time_ind + phaselen]
            corrections_periodic[P_ind, :, :phaselen] += corrections[:, time_ind:time_ind + phaselen]
            stats_gaussianized_periodic[P_ind, :, :phaselen] += statistics_by_trends_gaussianized[:, time_ind:time_ind + phaselen]
    return statistics_periodic, denomsq, corrections_periodic, stats_gaussianized_periodic
