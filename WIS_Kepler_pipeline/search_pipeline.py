#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:11:32 2020

@author: orynaivashtenko
version October 2022


"""
import numpy as np
import copy
import warnings
from scipy.signal import sosfiltfilt
import numba
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation as mad

import templates.template_module as tm
import auxiliary_functions as af
import utils as ut
from search_pipeline_parameters import *

import ssl
# noinspection PyUnresolvedReferences
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
klc.periodicity_search(Pmin=50, Pmax=52)
klc.determine_mins()
----------------------------

If conditioned data is loaded:
----------------------------
klc1 = pip.KeplerLightCurve(target_id='000757280', get_data=True, data_location='/home/orynai/Kepler_Notebooks/Data_sample_gaia_1', 
                            scramble=False, output_location='/SMALLDATA/orynai/trial_outputs')
klc1.load_conditioned_data('/SMALLDATA/orynai/trial_outputs/000757280_conditioned_data.npy')
klc1.initialize_templates()
klc1.detect_single_transits()

"""
##############################################################################

# TODO: update things:
# what if the template is not symmetric

# TODO discussion:
# we are saving one best thing per star. Then, we are vetting it, and it might not pass.
# But the same star could have a trigger which was slightly smaller but passing the veto.
# Maybe we need to save all the scores of one star which are higher than a certain threshold.

# When we plot folds of statistic, should these be folds with score correction or not?


class KeplerLightCurve(object):
    def __init__(self, target_id=None, times=None, fluxes=None, get_data=True, data_location=Default_data_location,
                 scramble=False, std_time_scrambling=Std_time_scrambling,
                 output_location=Default_output_location,
                 remove_zeroth_quarter=True, remove_quarter_17=True,
                 use_rfft=Use_rfft):  # , load_data_from_file=False
        """
        Initializes a kepler light curve object.

        Parameters:
        -------------------
        target_id : int or str
            Kepler ID of the host star. A 9-digit number, where some first digits may be zero.
            May be specified as integer or as string, should only contain digits.
            If get_data==True, target_id must be specified, it cannot be None.
        times : list of numpy.array 1d of float
            Equally-sampled time series of measurements for the quarters
        fluxes : list of numpy.array 1d of float
            List of Kepler PDC light curves. Might contain numpy.nan
        get_data : bool
            Whether to take data from some storage or to use times and fluxes. If False,
            times and fluxes will be used. If True, data will be searched for in the
            specified `data_location`. There should be a folder named as a 9-digit target_id.
            In this folder, fits filed of Kepler data should be stored.
            If such a folder is not found, it will be created. Data will be then downloaded from
            https://archive.stsci.edu/pub/kepler/lightcurves/ and placed in this folder.
        data_location : str
            Local directory in which folders with light curves can be found if they are there.
            If they are jot found, they will be downloaded.
        -------------------
        Initializes:
            self.orig_times
                Initial time arrays provided by user or downloaded from location
            self.orig_fluxes
                Initial flux arrays provided by user or downloaded from location
            self.times
                Time arrays that will be processed and modified
            self.fluxes
                Flux arrays that will be processed and modified
            self.quarters
                Array enumerating all the quarters of Kepler data. If there is no quarters
                subdivision, declare data with dingle quarter: time=[time], flux=[flux]
            self.masks : arrays of bool
                True where data is good, False where data is bad
            self.psd_masks
                Masks used for PSD measurement (some small holed may be ignored to increase slices size)
            self.big_hole_masks
                Masks masking only big holes
            self.notch_soss
                Filters to suppress outliers in frequencies
            self.dt_days
                Data timestep in units of days
            self.tmin
                Minimal time
            self.tmax
                Maximal time
            self.support
                Length of the zero-padded data (is power of 2)
            self.fft_support
                Length of the zero-padded data in Fourier (might differ from support if rfft is used)
        """
        self.output_location = output_location
        if target_id is not None:
            target_id = str(target_id).rjust(9, "0")  # Convert Kepler ID to string of appropriate length
        self.target_id = target_id
        if not get_data:
            assert len(times) == len(
                fluxes), 'Number of time pieces is supposed to be equal to the number of light curves!'
            for i in range(len(times)):
                assert len(times[i]) == len(fluxes[i]), 'Length of fluxes[' + str(
                    i) + '] does not match its time length!'
            self.orig_times = times
            self.orig_fluxes = fluxes
        else:
            assert target_id is not None, 'Please specify target ID!'
            self.orig_times, self.orig_fluxes = af.get_data_fun(target_id, location=data_location)

        if len(self.orig_times) < 1:
            raise AssertionError('There is no data to process!')

        # REMOVE ZEROTH QUARTER
        if remove_zeroth_quarter:
            if len(self.orig_times[0]) < 800:  # Make sure that the zeroth quarter is indeed the short zeroth quarter
                self.orig_times = self.orig_times[1:]
                self.orig_fluxes = self.orig_fluxes[1:]

        if len(self.orig_times) < 1:
            raise AssertionError('There is no data to process!')

        if remove_quarter_17:  # This quarter is small, there is a hola, and there might be glitch. we remove it
            if ((1570 >= np.nanmin(self.orig_times[-1])) & (
                    1570 <= np.nanmax(self.orig_times[-1]))):  # Check that Q17 is here.
                self.orig_times = self.orig_times[:-1]
                self.orig_fluxes = self.orig_fluxes[:-1]

        if len(self.orig_times) < 1:
            raise AssertionError('There is no data to process!')

        if scramble:
            np.random.seed(None)
            r = np.abs(np.random.normal(0, std_time_scrambling, len(self.orig_times)))
            r = np.cumsum(r)
            self.time_slide_shifts = r
            for i in range(len(self.orig_times)):
                self.orig_times[i] = np.array(self.orig_times[i]) + r[i]
        else:
            self.time_slide_shifts = np.zeros(len(self.orig_times))

        self.times = copy.deepcopy(self.orig_times)
        self.fluxes = copy.deepcopy(self.orig_fluxes)
        self.clean_fluxes = copy.deepcopy(self.orig_fluxes)
        self.quarters = range(len(self.fluxes))

        self.masks = [~np.isnan(self.fluxes[i]) for i in
                      self.quarters]  # mask things which need to be removed from data
        self.clean_masks = copy.deepcopy(
            self.masks)  # mask all the outliers, needed for PSD and filters, might kill transits
        self.psd_masks = copy.deepcopy(
            self.masks)  # clean_masks with maybe only big holes to have enough slices for PSD
        self.big_hole_masks = [np.ones(len(self.masks[q])).astype(bool) for q in self.quarters]
        self.stat_masks = []
        self.veto_masks = []

        self.notch_soss = []
        dts = []
        for q in self.quarters:
            dts.append(np.nanmean((self.times[q] - np.roll(self.times[q], 1))[1:]))
        self.dt_days = float(np.nanmean(dts))  # although it would be correct to weight by the lengths of the quarters
        assert self.dt_days is not None, 'dt_days should not be None!'
        self.tmin = np.nanmin(self.times[0])
        self.tmax = np.nanmax(self.times[0])
        for t in self.times:
            self.tmin = min(np.nanmin(t), self.tmin)
            self.tmax = max(np.nanmax(t), self.tmax)
        self.T_mission = self.tmax - self.tmin  # NOT CHANGED IN TIME SLIDES
        self.dt_sec = self.dt_days * 24 * 60 * 60
        self.is_fluxes_conditioned = False
        self.is_zero_padded = False
        self.is_fluxes_whitened = False
        self.is_fluxes_inpainted = False
        self.is_holes_interpolated = False
        self.is_templates_initialized = False
        self.is_prepared_for_psd = False
        self.is_template_whitened = False
        self.was_notched = None
        tmax = np.max(np.array([len(t) for t in self.orig_times]))  # maximum length of the time
        # the closest (but bigger) power of 2 to the length        Tmax += tm.L_ban
        self.support = 2 ** int(np.ceil(np.log2(tmax)))
        if use_rfft:
            self.fft_support = self.support // 2 + 1
        else:
            self.fft_support = self.support

        self.whitened_fluxes_td = np.zeros((len(self.fluxes), self.support))
        self.whitened_templates_td = None

        self.inpaint_fluxes_fast = numba.njit(ut.inpaint_fluxes_fun)
        # initialization of the numba function on some small data in order  not to
        # spend time on big data:
        _ = self.inpaint_fluxes_fast(fluxes=np.random.normal(0, 1, (2, 32)),
                                     masks=np.random.choice([0, 1], (2, 32)).astype(bool),
                                     wf_tds=np.array([np.concatenate((np.zeros(32 // 2 - 2),
                                                                      [-0.5, 1, 1, -0.5], np.zeros(32 // 2 - 2))) for _ in range(2)]))

        # numba.set_num_threads(1)
        self.sum_periods_fast = numba.njit(sum_periods_fun, parallel=False, fastmath=False)
        # initialization of the numba function on some small data in order  not to
        # spend time on big data:
        _ = self.sum_periods_fast(statistics_by_trends=np.ones((2, 192)),
                                  trends_sq_by_vars=np.ones((2, 192)), dt=0.02,
                                  tmax=3., tmin=0., periods=np.arange(1., 2., 0.5))

        self.mean_total = 0.
        len_total = 0
        for q in self.quarters:
            mean = np.nanmean(self.clean_fluxes[q])
            ll = len(self.clean_fluxes[q][~np.isnan(self.clean_fluxes[q])])
            self.mean_total += mean * ll
            len_total += ll
        self.mean_total = self.mean_total / len_total

        self.trends_low = None

        self.npsds0 = None
        self.nperseg = None
        self.freqs = None
        self.psdfreqs = None
        self.npsds = None
        self.psds = None
        self.orig_psds = None
        self.wf_td = None
        self.wf_fd = None
        self.zero_padding_ends = None

        self.notch_mask = None

        self.unnotched_wf_fd = None
        self.detrended_orig_fluxes = None
        self.means = None
        self.trends_orig_fluxes = None
        self.whitened_fluxes_fd = None
        self.whitened_fluxes_td = None
        self.whitened_clean_fluxes_fd = None
        self.whitened_clean_fluxes_td = None

        self.indices_of_templates = None
        self.templates_of_indices = None
        self.shifts = None
        self.template_len = None
        self.templates = None
        self.initial_templates = None
        self.templ_num = None
        self.templ_num_shifts = None
        self.whitened_templates_td = None
        self.whitened_templates_fd = None

        # self.statistics_raw = None
        self.statistics_snr = None
        self.statistics_raw_cor = None
        self.stat_moving_vars = None
        self.stat_vars = None
        self.stat_denoms = None
        self.stats_raw_no_veto_no_score_cor = None

        self.time_full = None
        self.trend_orig_flux_full = None
        self.statistics_full = None
        self.stat_vars_full = None
        self.stat_masks_full = None
        self.trigger_masks = None
        self.veto_bad_trigger_inds = None

        self.interleaved_numbers = None
        self.not_interleaved_numbers = None
        self.n_shifts = None

        self.statistics_interleaved = None
        self.stat_masks_interleaved = None
        self.stat_vars_interleaved = None
        self.trend_orig_flux_interleaved = None
        self.time_interleaved = None
        self.statistics_not_interleaved = None
        self.stat_masks_not_interleaved = None
        self.stat_vars_not_interleaved = None
        self.trend_orig_flux_not_interleaved = None
        self.time_not_interleaved = None

        self.Pmin = None
        self.Pmax = None
        self.periods = None
        self._phases_interleaved = None
        self.statistics_periodic_interleaved = None
        self._phases_not_interleaved = None
        self.statistics_periodic_not_interleaved = None
        self.mes_veto_masks_interleaved = None
        self.mes_veto_masks_not_interleaved = None

        self.triggers = None
        self.best_periodic_scores_data = None
        self.best_inverted_periodic_scores_data = None

        self.best_stat = None
        self.best_per = None
        self.best_temp = None
        self.best_first_transit_time = None
        self.summed_transit_times = None
        self.summed_raw_stats = None
        self.summed_trends = None
        self.summed_vars = None

        self.best_raw_stats = None  # will be defined in the non-dcore-corrected and non-vetted data
        self.best_stat_denoms = None
        self.best_trends = None
        self.best_templ_depths = None
        self.best_ampl = None
        self.best_ppm = None
        self.best_ampls = None
        self.best_ampls_stds = None
        self.best_normalized_ampls = None
        self.best_normalized_ampls_stds = None
        self.best_ppms = None
        self.best_ppms_stds = None

        self.best_stat_inv = None
        self.best_per_inv = None
        self.best_temp_inv = None
        self.best_first_transit_time_inv = None
        self.summed_transit_times_inv = None
        self.summed_raw_stats_inv = None
        self.summed_trends_inv = None
        self.summed_vars_inv = None

    def condition_data(self, nsig_outliers_0=Nsig_outliers_0, nsig_outliers_0_pos=Nsig_outliers_0_pos,
                       nsig_outliers_0_neg=Nsig_outliers_0_neg,
                       nsig_outliers_psd=Nsig_outliers_psd, nperseg_psd=Nperseg_psd, noverlap_psd=Noverlap_psd,
                       window_psd=Window_psd, npsdsmin=Npsdsmin, subtract_mean_in_psd=Subtract_mean_in_psd,
                       smoothen_psds=Smoothen_psds,
                       nsig_notch=Nsig_notch, use_rfft=Use_rfft,
                       nsig_outliers_w=Nsig_outliers_w, nsig_outliers_w_pos=Nsig_outliers_w_pos,
                       nsig_outliers_w_neg=Nsig_outliers_w_neg,
                       max_small_hole=Max_small_hole,
                       detr_length_outliers=Detr_length_outliers, detr_method_outliers=Detr_method_outliers,
                       detr_endmethod_outliers=Detr_endmethod_outliers,
                       detr_ends_outliers=Detr_ends_outliers, detr_endlen_outliers=Detr_endlen_outliers):
        """
        Preparing data for detection. Subtract mean, mask outliers, measure psds,
        padding by zeros, notching, inpainting, whitening.

        :param nsig_outliers_0: Minimal deviation from trend considered to be outlier (in units of std)
        :type nsig_outliers_0: float
        :param nsig_outliers_0_pos:
        :type nsig_outliers_0_pos:
        :param nsig_outliers_0_neg:
        :type nsig_outliers_0_neg:
        :param nsig_outliers_psd:
        :type nsig_outliers_psd:
        :param nperseg_psd: Length of slice used in measuring PSD
        :type nperseg_psd: int
        :param noverlap_psd: Length of slice overlapping when measuring PSD (if -1, will take half of `nperseg`)
        :type noverlap_psd: int
        :param window_psd: Which window will be used in PSD measurement. It is recommended to use 'hann'
        :type window_psd: : str : 'hann', 'boxcar'
        :param npsdsmin: Minimal number of slices required to measure PSD with Welch method
        :type npsdsmin:  int
        :param subtract_mean_in_psd: Whether to subtract mean when measuring PSD
        :type subtract_mean_in_psd: bool
        :param smoothen_psds:
        :type smoothen_psds:
        :param nsig_notch: Minimal deviation from trend of whitened fluxes in Fourier domain
            considered to be outlier (in units of std of whitened fluxes in Fourier space)
        :type nsig_notch: float
        :param use_rfft: Whether to use rfft (Fourier for real signals) instead of fft
        :type use_rfft: bool
        :param nsig_outliers_w: Minimal deviation from trend of whitened fluxes in real domain
           considered to be outlier (in units of their std)
        :type nsig_outliers_w: float
        :param nsig_outliers_w_pos:
        :type nsig_outliers_w_pos:
        :param nsig_outliers_w_neg:
        :type nsig_outliers_w_neg:
        :param max_small_hole:
        :type max_small_hole:
        :param detr_length_outliers:
        :type detr_length_outliers:
        :param detr_endmethod_outliers:
        :type detr_endmethod_outliers:
        :param detr_ends_outliers:
        :type detr_ends_outliers:
        :param detr_method_outliers:
        :type detr_method_outliers:
        :param detr_endlen_outliers:
        :type detr_endlen_outliers:
        :return:
        :rtype:

        Mean values of the flux are subtracted to have signal centered around zero. 
        Means are saved in self.means:
        """
        self.subtract_means(save_means=True, make_trends=True)
        self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                nsig_negative=nsig_outliers_0_neg,
                                nsig_clean=nsig_outliers_psd, detr_length_outliers=Detr_length_outliers,
                                detr_method_outliers=detr_method_outliers, detr_endmethod_outliers=detr_endmethod_outliers,
                                detr_ends_outliers=detr_ends_outliers, detr_endlen_outliers=detr_endlen_outliers,
                                in_whitened_fluxes=False, modify_clean_masks=True)
        self.subtract_means(save_means=True, make_trends=True, masks=self.masks)
        self.pad_by_zeros()
        self.make_big_hole_masks(max_small_hole=max_small_hole, exclude_individual_points=True)
        self.interpolate_fluxes()
        self.subtract_low_trends(nperseg=nperseg_psd)
        self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                nsig_negative=nsig_outliers_0_neg,
                                nsig_clean=nsig_outliers_psd, detr_length_outliers=Detr_length_outliers,
                                detr_method_outliers=detr_method_outliers, detr_endmethod_outliers=detr_endmethod_outliers,
                                detr_ends_outliers=detr_ends_outliers, detr_endlen_outliers=detr_endlen_outliers,
                                in_whitened_fluxes=False, modify_clean_masks=True)
        self.interpolate_fluxes(interpolate_clean_fluxes=False)  # need to do this again after piecewise detrending
        self.prepare_for_psd(nperseg=nperseg_psd, noverlap=noverlap_psd, npsdsmin=npsdsmin)
        self.measure_psds(nperseg=nperseg_psd, noverlap=noverlap_psd, window=window_psd,
                          subtract_mean=subtract_mean_in_psd,
                          rfft=use_rfft, smoothen_psds=smoothen_psds)
        self.whiten_fluxes(whiten_clean_fluxes=True)
        self.mask_flux_outliers(nsig=nsig_outliers_w, nsig_positive=nsig_outliers_w_pos,
                                nsig_negative=nsig_outliers_w_neg,
                                nsig_clean=nsig_outliers_psd, detr_length_outliers=Detr_length_outliers,
                                detr_method_outliers=detr_method_outliers, detr_endmethod_outliers=detr_endmethod_outliers,
                                detr_ends_outliers=detr_ends_outliers, detr_endlen_outliers=detr_endlen_outliers,
                                in_whitened_fluxes=True, modify_clean_masks=True)
        self.interpolate_fluxes()
        self.whiten_fluxes(whiten_clean_fluxes=True)
        self.notch_fluxes(nsig=nsig_notch)
        self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                nsig_negative=nsig_outliers_0_neg,
                                nsig_clean=nsig_outliers_psd,
                                detr_length_outliers=detr_length_outliers, detr_method_outliers=detr_method_outliers,
                                detr_endmethod_outliers=detr_endmethod_outliers,
                                detr_ends_outliers=detr_ends_outliers, detr_endlen_outliers=detr_endlen_outliers,
                                in_whitened_fluxes=False, modify_clean_masks=True
                                )
        self.make_big_hole_masks(max_small_hole=max_small_hole)
        self.inpaint_fluxes()
        self.whiten_fluxes(whiten_clean_fluxes=True)
        self.is_fluxes_conditioned = True

        """
        Values diverging from the large trend more than by 3 standard deviations are added to the mask 
        as False (before, mask marked Bins containing NaN). psd_masks are also modified, because we don't 
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

        # TODO: maybe define two levels of small holes and two npsds
        # TODO: maybe define better distribution of pieces in Welch

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

    def save_conditioned_data(self, path=None, filename_postfix='conditioned_data'):
        """
        Saves the essential conditioning results as .npy file that can be read
        with `load_conditioned_data` function:
            self.support
            self.zero_padding_ends
            self.times
            self.fluxes
            self.whitened_fluxes_td
            self.whitened_fluxes_fd
            self.wf_td
            self.wf_fd
            self.big_hole_masks
            self.whitened_templates_td
            self.whitened_templates_fd

        Parameters
        ----------
        :param path : path to the folder in which the data should be saved.
        :type path : str
        :param filename_postfix:
        :type filename_postfix:

        Returns
        -------
        None.


        """
        filename_postfix = filename_postfix.strip('_')
        if path is None:
            path = self.output_location

        len_data = sum([self.support for _ in self.quarters])  # is assumed to be support length times number of quarters
        conditioned_data = np.zeros((7 + self.templ_num_shifts, len_data), dtype=np.float32)
        conditioned_data[0][10] = self.support
        conditioned_data[0][11] = self.templ_num
        conditioned_data[0][12] = self.templ_num_shifts
        for tn in range(self.templ_num):
            conditioned_data[0][13 + tn] = self.indices_of_templates[tn][0]
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
            conditioned_data[4][ind_last:ind_last + self.support] = self.wf_td[q]
            conditioned_data[5][ind_last:ind_last + len(self.fluxes[q])] = self.big_hole_masks[q]
            conditioned_data[6][ind_last:ind_last + len(self.trends_orig_fluxes[q])] = self.trends_orig_fluxes[q]
            for k in range(self.templ_num_shifts):
                conditioned_data[k + 7][ind_last:ind_last + len(self.whitened_templates_td[k][q])] = \
                    self.whitened_templates_td[k][q]
            ind_last += self.support
        if self.target_id is not None:
            np.save(os.path.join(path, str(self.target_id) + '_' + filename_postfix),
                    np.array(conditioned_data, dtype=object), allow_pickle=True)
        else:
            np.save(os.path.join(path, filename_postfix), np.array(conditioned_data, dtype=object), allow_pickle=True)

    def load_conditioned_data(self, file=None, filename_postfix='conditioned_data', no_zero_padding=False):
        """
        Only works for data created by `save_conditioned_data`.

        Sets:
        self.support
        self.zero_padding_ends
        self.times
        self.fluxes
        self.whitened_fluxes_td
        self.whitened_fluxes_fd
        self.wf_td
        self.wf_fd
        self.big_hole_masks
        self.whitened_templates_td
        self.whitened_templates_fd

        :param file: path to the .npy file where the data is stored.
        :type file: str
        :param filename_postfix:
        :type filename_postfix:
        :param no_zero_padding:
        :type no_zero_padding:
        :return:
        :rtype:
        """

        filename_postfix = filename_postfix.strip('_')
        filename_postfix = filename_postfix.rstrip('.npy')
        if file is None:
            file = os.path.join(self.output_location, str(self.target_id) + '_' + filename_postfix + '.npy')
        else:
            if file[-4:] != '.npy':
                file = file + '.npy'
        conditioned_data = np.load(file, allow_pickle=True)
        self.support = int(conditioned_data[0][10])
        self.templ_num = int(conditioned_data[0][11])
        self.templ_num_shifts = int(conditioned_data[0][12])
        self.indices_of_templates = []
        for tn in range(self.templ_num - 1):
            self.indices_of_templates.append([int(conditioned_data[0][13 + tn]), int(conditioned_data[0][14 + tn])])
        self.indices_of_templates.append([int(conditioned_data[0][self.templ_num - 1]), int(self.templ_num_shifts)])

        self.zero_padding_ends = [[0, 0] for _ in self.quarters]
        self.times = []
        self.fluxes = []
        self.whitened_fluxes_td = []
        self.whitened_fluxes_fd = []
        self.wf_td = []
        self.wf_fd = []
        self.big_hole_masks = []
        self.trends_orig_fluxes = []
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
            len_orig = int(conditioned_data[0][ind_last + 4])
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
            self.wf_td.append(wftd1)
            self.wf_fd.append(np.fft.rfft(np.fft.ifftshift(wftd1)))
            self.big_hole_masks.append((conditioned_data[5][ind_last: ind_last + lenflux]).astype(bool))
            self.trends_orig_fluxes.append((conditioned_data[6][ind_last: ind_last + len_orig]).astype(float))
            for k in range(len(conditioned_data) - 7):
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

    def condition_templates(self, template_ind=Template_ind_detection, use_rfft=Use_rfft):
        """
        TODO: write this. For now, please consult descriptions of all the methods, as well as "Short user documentation"
        :param template_ind:
        :type template_ind:
        :param use_rfft: Whether to use rfft or not, meaning to use fft. The default is use_rfft=True.
        :type use_rfft:  bool
        :return:
        :rtype:
        """

        self.initialize_templates(templates=None, template_ind=template_ind, template_bank_from_file=True,
                                  templates_subbin_shift=Templates_subbin_shift)
        self.notch_templates()
        self.roll_templates()
        self.whiten_templates(use_rfft=use_rfft)

    def detect_single_transits(self, use_rfft=Use_rfft,
                               nsig_outliers_stat=Nsig_outliers_stat,
                               apply_veto=True, margin_snr=Margin_snr_veto,
                               fn=Fn_fit_veto, maxdev=Maxdev_fit_veto,
                               zero_ends_len_abs=Zero_ends_len_abs_fit_veto,
                               zero_ends_len_rel=Zero_ends_len_rel_fit_veto,
                               apply_score_correction=Apply_score_correction,
                               n_parts_veto_snr=N_parts_veto_snr, n_parts_veto_len=N_parts_veto_len):
        """
        TODO: write this. For now, please consult descriptions of all the methods, as well as "Short user documentation"
        :param use_rfft:
        :type use_rfft:
        :param nsig_outliers_stat:
        :type nsig_outliers_stat:
        :param apply_veto:
        :type apply_veto:
        :param margin_snr:
        :type margin_snr:
        :param fn:
        :type fn:
        :param maxdev:
        :type maxdev:
        :param zero_ends_len_abs:
        :type zero_ends_len_abs:
        :param zero_ends_len_rel:
        :type zero_ends_len_rel:
        :param apply_score_correction:
        :type apply_score_correction:
        :param n_parts_veto_snr:
        :type n_parts_veto_snr:
        :param n_parts_veto_len:
        :type n_parts_veto_len:
        :return:
        :rtype:
        """

        self.matched_filtering(nsig_outliers_stat=nsig_outliers_stat)
        self.cut_zero_padding_ends()
        if apply_veto:
            self.veto_statistics(margin_snr=margin_snr, fn=fn, maxdev=maxdev,
                                 rfft_psd=use_rfft, zero_ends_len_abs=zero_ends_len_abs,
                                 zero_ends_len_rel=zero_ends_len_rel,
                                 n_parts_veto_snr=n_parts_veto_snr, n_parts_veto_len=n_parts_veto_len)
        if apply_score_correction:
            self.correct_score()

    def periodicity_search(self, pmin=Pmin, pmax=Pmax, dp=None,
                           duty_cycle=Duty_cycle,
                           # threshold_mes_veto=Threshold_mes_veto, veto_threshold_for_mes=Veto_threshold_for_mes
                           ):
        """
        TODO: write this. For now, I hope that other users than me will not need this method.

        :param pmin:
        :type pmin:
        :param pmax:
        :type pmax:
        :param dp:
        :type dp:
        :param duty_cycle:
        :type duty_cycle:
        :return:
        :rtype:
        """
        # self.find_moving_averages(interpolate=True, detr_length=detr_length, detr_method=detr_method,
        #                      detr_endmethod=detr_endmethod, detr_ends=detr_ends,
        #                      detr_endlen=detr_endlen, keep_overall_level=False)
        self.connect_quarters()
        self.interleave_shifts()
        self.sum_periods(pmin=pmin, pmax=pmax, dp=dp, duty_cycle=duty_cycle)
        # self.veto_mes(mes_threshold=threshold_mes_veto, veto_threshold=veto_threshold_for_mes)
        # -> WARNING: this veto, if done after the periodicity search on the score-corrected data, will destroy the
        # distribution. Therefore, we are removing this veto from the search and will do it after the search and after
        # the time-slides and p_periodic estimation as a post-processing veto. Also, we make sure to do it on the
        # non-score-corrected data.

    def subtract_means(self, save_means=True, method='mean', make_trends=True, masks=None):
        """
        Subtracts mean value from each of self.fluxes
        :param save_means: Whether to create a new attribure self.means and save the mean
            values of the fluxes there. The default is True.
        :type save_means: bool
        :param method:
        :type method:
        :param make_trends:
        :type make_trends:
        :param masks:
        :type masks:
        :return:
        :rtype:

        Modifies
        self.fluxes
        """
        assert method == 'mean' or method == 'median', "`method` must be 'mean' or 'median'!"
        new_means = self.get_averages(masks=masks, method=method)
        if self.means is None:
            old_means = np.zeros(len(self.quarters))
        else:
            old_means = self.means
        self.fluxes = [np.array(self.fluxes[q] + old_means[q] - new_means[q]) for q in self.quarters]
        self.clean_fluxes = [np.array(self.clean_fluxes[q] + old_means[q] - new_means[q]) for q in self.quarters]
        if save_means:
            self.means = new_means
            lens = [len(self.clean_fluxes[q][~np.isnan(self.clean_fluxes[q])]) for q in self.quarters]
            self.mean_total = np.sum([self.means[q] * lens[q] for q in self.quarters]) / np.sum(lens)

        if make_trends:
            self.trends_orig_fluxes = [new_means[q] * np.ones(len(self.clean_fluxes[q])) for q in self.quarters]

        # len_total = None
        # if save_means:
        #     self.mean_total = 0.
        # len_total = 0
        # self.means = []
        # if make_trends:
        #     self.trends_orig_fluxes = []
        #
        # for q in self.quarters:
        #     mean = None
        #     if method == 'mean':
        #         mean = np.nanmean(self.clean_fluxes[q])
        #     elif method == 'median':
        #         mean = np.nanmedian(self.clean_fluxes[q])
        #     if save_means:
        #         self.means.append(mean)
        #         ll = len(self.clean_fluxes[q][~np.isnan(self.clean_fluxes[q])])
        #         self.mean_total += mean * ll
        #         len_total += ll
        #     self.fluxes[q] = np.array(self.fluxes[q] - mean)
        #     self.clean_fluxes[q] = np.array(self.clean_fluxes[q] - mean)
        #     if make_trends:
        #         self.trends_orig_fluxes.append(mean * np.ones(len(self.clean_fluxes[q])))
        # if save_means:
        #     self.mean_total = self.mean_total / len_total

    def get_averages(self, masks=None, method='mean'):
        if masks is None:
            masks = [~np.isnan(self.orig_fluxes[q]) for q in self.quarters]
        if method.lower() == 'mean':
            averages = [np.nanmean(self.orig_fluxes[q][masks[q]]) for q in self.quarters]
        elif method.lower() == 'median':
            averages = [np.nanmedian(self.orig_fluxes[q][masks[q]]) for q in self.quarters]
        else:
            raise ValueError("`method` should be 'mean' or 'median'!")
        return averages

    def mask_flux_outliers(self, nsig=Nsig_outliers_0, nsig_positive=Nsig_outliers_0_pos,
                           nsig_negative=Nsig_outliers_0_neg, nsig_clean=Nsig_outliers_psd,
                           detr_length_outliers=Detr_length_outliers, detr_method_outliers=Detr_method_outliers,
                           detr_endmethod_outliers=Detr_endmethod_outliers,
                           detr_ends_outliers=Detr_ends_outliers, detr_endlen_outliers=Detr_endlen_outliers,
                           in_whitened_fluxes=False, modify_clean_masks=True):
        """
        Detects outliers of the flux with respect to the trend. The trend is found by detrending.
        Modifies self.masks by setting False where the outliers were found.
        :param nsig: Minimal deviation from the trend considered as outlier
        :type nsig: float
        :param nsig_positive:
        :type nsig_positive:
        :param nsig_negative:
        :type nsig_negative:
        :param nsig_clean:
        :type nsig_clean:
        :param detr_length_outliers:
        :type detr_length_outliers:
        :param detr_method_outliers:
        :type detr_method_outliers:
        :param detr_endmethod_outliers:
        :type detr_endmethod_outliers:
        :param detr_ends_outliers:
        :type detr_ends_outliers:
        :param detr_endlen_outliers:
        :type detr_endlen_outliers:
        :param detr_ends_outliers:
        :type detr_ends_outliers:
        :param detr_endlen_outliers:
        :type detr_endlen_outliers:
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

    def make_big_hole_masks(self, max_small_hole=Max_small_hole, exclude_individual_points=False):
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
        :param period: Period in days of transits that should be masked. Used only if `time_edges_to_mask` is None
        and `transit_times` is None. This is useful when there are no TTVs or TDVs.
        :type period: float
        :param durations: Duration or durations in days of transits that should be masked. Used only if `time_edges_to_mask` is None.
        If `float`, then it is used as constant duration for all transits. If `list`, then its length should coincide with the length
        of `transit_times` if it is also a list. This is useful when there are TDVs.
        :type durations: float or list[float]
        :param transit_times: Middle-time of the transit or transits in days. If `float`, it is counted as middle-time of one transit,
        and all the rest are assumed to pe periodic with `period`. If `list`, will be counted as all the mid-times that should be masked.
        Should coincide in length with `durations` if it is also a list. This is useful when there are TTVs.
        Is overriden by time_edges_to_mask.
        :type transit_times: list[float] or float or ndarray
        :param absolute_phase: Phase of one of the transits with respect to the beginning of the Kepler time in days.
        Used only if `time_edges_to_mask` is None and `transit_times` is None. This is useful when there are no TTVs or TDVs.
        :type absolute_phase: float
        :param time_edges_to_mask: list of tuples marking the beginning and the end of the things that should be masked in days.
        Overriding all the other parameters which are used if it is None. This is useful when there are TTVs and TDVs.
        :type time_edges_to_mask: list[list[float]] or ndarray
        :return: None
        :rtype:
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

    def prepare_for_psd(self, nperseg=Nperseg_psd, noverlap=Noverlap_psd, npsdsmin=Npsdsmin,
                        max_small_hole=Max_small_hole):
        """
        Checks whether the number of pieces for the Welch method is enough compare to npsdsmin, and
        if no, interpolates over mask and updates the psd mask to measure psd for the first time like that.
        :param nperseg: length of slices in Welch method
        :type nperseg: int
        :param noverlap: If int: length of overlap of slices in Welch method.
                         If float<1 : relative overlap to nperseg.
                         Length of overlap will be int(nperseg*noverlap)
        :type noverlap: int or float<1
        :param npsdsmin: minimum number of slices to measure psd with Welch. If the actual number is smaller, interpolation will be made.
        :type npsdsmin: int
        :param max_small_hole:
        :type max_small_hole:
        :return: creates self.nperseg, self.big_hole_masks; modifies self.fluxes (-> interpolated fluxes), self.psd_masks
        :rtype:
        """

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

    def pad_by_zeros(self):
        """
        Pads fluxes, masks, psd_masks and times by zeros.
        The new length will be the closest power of 2 greater or equal to the longest
        flux plus the length of the template from the template bank.
        The times will be also interpolated to close the holes.

        Modifies
        -------
        self.fluxes -> pads by zeros to len support
        self.masks -> pads by zeros to len support
        self.psd_masks -> pads by zeros to len support
        self.times -> pads by zeros to len support

        Creates
        -------
        self.support
        self.zero_padding_ends

        """
        self.zero_padding_ends = []
        for i in self.quarters:
            zerlen1 = (self.support - len(self.fluxes[i])) // 2
            zerlen2 = (self.support - len(self.fluxes[i]) + 1) // 2
            self.zero_padding_ends.append((zerlen1, zerlen2))
            self.fluxes[i] = np.concatenate((np.zeros(zerlen1), self.fluxes[i], np.zeros(zerlen2)))
            self.clean_fluxes[i] = np.concatenate((np.zeros(zerlen1), self.clean_fluxes[i], np.zeros(zerlen2)))
            self.masks[i] = np.concatenate((np.zeros(zerlen1), self.masks[i], np.zeros(zerlen2))).astype(bool)
            self.big_hole_masks[i] = np.concatenate((np.zeros(zerlen1), self.big_hole_masks[i], np.zeros(zerlen2))).astype(bool)
            self.clean_masks[i] = np.concatenate((np.zeros(zerlen1),  self.clean_masks[i], np.zeros(zerlen2))).astype(bool)
            self.psd_masks[i] = np.concatenate((np.zeros(zerlen1), self.psd_masks[i], np.zeros(zerlen2))).astype(bool)
            self.times[i] = np.concatenate((np.zeros(zerlen1) + np.nan, self.times[i], np.zeros(zerlen2) + np.nan))
            # NaNs are needed for the `interpolate_time` function to treat these regions
            # as the ones to be interpolated. The self.masks[i] is not transferred to the
            # time interpolation because not all the masked flux regions are NaNs in time,
            # and we do not want to lose precision rejecting data.
            self.times[i] = ut.interpolate_time(np.array(self.times[i]), mask=None)
        self.fluxes = np.array(self.fluxes)  # need them to be numpy.arrays for numba inpainting
        self.masks = np.array(self.masks)

        self.is_zero_padded = True

    def measure_psds(self, nperseg=Nperseg_psd, noverlap=Noverlap_psd, window=Window_psd,
                     subtract_mean=Subtract_mean_in_psd, rfft=Use_rfft, smoothen_psds=True):
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
                self.freqs = [np.fft.rfftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.rfftfreq(nperseg, d=self.dt_sec) for _ in self.quarters]
            else:
                for i in self.quarters:
                    clean_flux = self.clean_fluxes[i] - np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]])
                    psds.append([np.abs(np.fft.fft(windowdict[window] * clean_flux[ind1:ind1 + nperseg])
                                        ) ** 2 for ind1 in
                                 range(0, len(self.clean_fluxes[i]) - nperseg, nperseg - noverlap
                                       ) if np.sum((~self.psd_masks[i])[ind1:ind1 + nperseg]) == 0])
                self.freqs = [np.fft.fftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.fftfreq(nperseg, d=self.dt_sec) for _ in self.quarters]

        else:
            if rfft:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.rfft(windowdict[window] * (self.clean_fluxes[i])[ind1:ind1 + nperseg])
                                        ) ** 2 for ind1 in
                                 range(0, len(self.clean_fluxes[i]) - nperseg, nperseg - noverlap
                                       ) if np.sum((~self.psd_masks[i])[ind1:ind1 + nperseg]) == 0])
                self.freqs = [np.fft.rfftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.rfftfreq(nperseg, d=self.dt_sec) for _ in self.quarters]
            else:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.fft(windowdict[window] * (self.clean_fluxes[i])[ind1:ind1 + nperseg])
                                        ) ** 2 for ind1 in
                                 range(0, len(self.clean_fluxes[i]) - nperseg, nperseg - noverlap
                                       ) if np.sum((~self.psd_masks[i])[ind1:ind1 + nperseg]) == 0])
                self.freqs = [np.fft.fftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.fftfreq(nperseg, d=self.dt_sec) for _ in self.quarters]

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
                # sol = minimize(lambda x: np.sum((psd_high-x[1]*(fs_high-x[0]))**2), [np.mean(psd_high), 0])
                # trend =  sol.x[1]*(fs - sol.x[0])

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

        self.nperseg = nperseg

        self.wf_td = []
        self.wf_fd = []
        for i in self.quarters:
            if self.support is not None:
                wf_td, wf_fd = ut.psd2wf(self.psds[i], self.support, rfft=rfft)
            else:
                wf_td, wf_fd = ut.psd2wf(self.psds[i], len(self.fluxes[i]), rfft=rfft)
            self.wf_td.append(wf_td)
            self.wf_fd.append(wf_fd)
        self.wf_td = np.array(self.wf_td)  # for numba need numpy.arrays

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

    def subtract_low_trends(self, nperseg=Nperseg_psd):
        # TODO AND WARNING: YOU DO NOT APPLY THIS TO THE TEMPLATE. FOR THE BANK WITH SHORT TEMPLATES IT IS OK, BUT STILL IT IS NOT CORRECT.
        # TODO: Maybe to implement detrend_piecewise with comparison not to std but to absolute values.
        #  TODO: maybe use minimax of scipy.spatial.distance.cdist
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
                                                            nperseg=nperseg // 2 * 3, endlen=endlen, return_trend=True)
                self.trends_low.append(trend_low)
                self.fluxes[q] = np.array(self.fluxes[q]) - trend_low
                self.clean_fluxes[q] = np.array(detrended)
        else:
            for q in self.quarters:
                if len(self.orig_fluxes[q]) - 1 <= nperseg * 4:
                    endlen = len(self.orig_fluxes[q]) // 4 - 1
                else:
                    endlen = nperseg
                detrended, trend_low = ut.detrend_piecewise(self.clean_fluxes[q][self.zero_padding_ends[q][0]:len(self.clean_fluxes[q]) -
                                                                                 self.zero_padding_ends[q][-1]],
                                                            big_mask=self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(
                                                                      self.big_hole_masks[q]) - self.zero_padding_ends[q][-1]],
                                                            mask=self.masks[q][self.zero_padding_ends[q][0]:len(self.masks[q]) -
                                                                               self.zero_padding_ends[q][-1]],
                                                            nperseg=nperseg // 2 * 3, endlen=endlen, return_trend=True)
                self.trends_low.append(trend_low)
                self.fluxes[q] = np.array(self.fluxes[q]) - np.concatenate(
                    (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1])))
                self.clean_fluxes[q] = np.array(self.clean_fluxes[q]) - np.concatenate(
                    (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1])))

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
        self.unnotched_wf_fd = copy.deepcopy(self.whitened_clean_fluxes_fd)
        sumflux = np.sum(self.whitened_clean_fluxes_fd, axis=0)  # add all the quarters to mark repeated spurious frequencies
        sumabs = np.sum(np.abs(self.whitened_clean_fluxes_fd), axis=0)
        mask1 = np.array((np.abs(sumflux) - np.abs(np.nanmean(sumflux))) < nsig * np.nanstd(sumflux)).astype(float)
        mask2 = np.array((sumabs - np.nanmean(sumabs)) < nsig * 1.5 * np.nanstd(sumabs)).astype(float)
        inv_mask = (1 - mask1 * mask2).astype(bool)
        # inv_mask = (1 - mask1).astype(bool)
        notch_soss = []  # make a filter for every peak to notch
        for fr in np.where(inv_mask)[0]:
            if not np.isclose(fr, 0):
                notch_soss.append(ut.notch_fun(fr, fs=self.support, irl=irl_notch, htol=htol_notch, min_val=minval_notch))

        for i in self.quarters:  # apply all the filters for all the quarters
            for sos in notch_soss:
                self.fluxes[i] = sosfiltfilt(sos, np.array(self.fluxes[i]), padtype='constant')
                self.clean_fluxes[i] = sosfiltfilt(sos, np.array(self.clean_fluxes[i]), padtype='constant')
                # very important to have zeros in the ends, bacause of padding.
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
        ---------
        save_inv_cov_matr : bool
            Whether to create self.Cinvs and save there the inverse covariance
            matrices. So far they are not needed further in the code, and they
            are heavy.
        Returns
        -------
        None.
            Modifies self.fluxes

        """
        self.fluxes = self.inpaint_fluxes_fast(self.fluxes, self.masks, self.wf_td)
        self.is_fluxes_inpainted = True

    def whiten_fluxes(self, whiten_clean_fluxes=False):
        """
        whitens all the fluxes with whitening filter
        """
        assert not (np.array([np.isnan(self.fluxes[i]
                                       ).any() for i in
                              self.quarters])).any(), 'there are holes in fluxes, cannot whiten!'
        self.whitened_fluxes_fd = []
        self.whitened_fluxes_td = []
        for q in self.quarters:
            whitened_flux_fd = np.fft.rfft(self.fluxes[q]) * self.wf_fd[q]
            self.whitened_fluxes_fd.append(whitened_flux_fd)
            self.whitened_fluxes_td.append(np.fft.irfft(whitened_flux_fd))

        if whiten_clean_fluxes:
            self.whitened_clean_fluxes_fd = []
            self.whitened_clean_fluxes_td = []
            for q in self.quarters:
                whitened_clean_flux_fd = np.fft.rfft(self.clean_fluxes[q]) * self.wf_fd[q]
                self.whitened_clean_fluxes_fd.append(whitened_clean_flux_fd)
                self.whitened_clean_fluxes_td.append(np.fft.irfft(whitened_clean_flux_fd))

        self.is_fluxes_whitened = True

    def find_moving_averages(self, interpolate=True, detr_length=Detr_length, detr_method=Detr_method,
                             detr_endmethod=Detr_endmethod, detr_ends=Detr_ends,
                             detr_endlen=Detr_endlen, keep_overall_level=False,
                             remove_outliers=False, nsig_outliers=Nsig_outliers_0):
        """
        Calculates the large trends of the original fluxes by taking the moving average of
        type detr_method ('mean' or 'median') of slices of size detr_length.
        Saves the trends (quarter-specific) and saves the detrended original fluxes separately.
        :param interpolate: Whether to interpolate the data in the holes containing numpy.nan.
            The interpolation will be linear, the holes in the ends of the data will
            be filled by a constant.
        :type interpolate: bool
        :param detr_length: Detrending length. Size of a slice to take the moving average
        :type detr_length: int
        :param detr_method: The type of moving average to take. Median is less sensitive to
            outliers, and mean is less sensitive to small number of samples.
            Median is preferred in the middle of the data.
        :type detr_method: str : 'mean' or 'median'
        :param detr_endmethod: The type of moving average to take in the ends of the data.
            Mean is less sensitive to small number of samples, so it is preferred
            in the ends of the data.
        :type detr_endmethod: str : 'mean' or 'median'
        :param detr_ends: How to make slices on the ends of the data. Read details in detrend_mov_ave().
            'symmetric' and 'asymmetric' preserve the size of the data,
            'cut' decreases the size of the data by detr_length.
        :type detr_ends: str : 'symmetric' or 'asymmetric' or 'cut'
        :param detr_endlen: Minimal slice taken in the end of the data. If None, will be taken
            as detr_len//4
        :type detr_endlen: int or None
        :param keep_overall_level: Whether to preserve the overall average (mean or median, defined by
            detr_method) of the data
        :type keep_overall_level: bool
        :param remove_outliers:
        :type remove_outliers:
        :param nsig_outliers:
        :type nsig_outliers:
        :return: creates self.detrended_orig_fluxes and self.trends_orig_fluxes
        :rtype:
        """
        self.detrended_orig_fluxes = []
        self.trends_orig_fluxes = []
        for q in self.quarters:
            msk = ~np.isnan(self.orig_fluxes[q])
            if interpolate:
                o_f = ut.interpolate_over_mask(np.array(self.orig_fluxes[q]), msk, interp_method='linear', ends='continue',
                                               mitigate_hole_ends=False, endhole_method='median')
                detr, trend = ut.detrend_mov_ave(np.array(o_f), length=detr_length, mask=None, method=detr_method,
                                                 endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen,
                                                 return_trend=True,
                                                 keep_overall_level=keep_overall_level, remove_outliers=remove_outliers,
                                                 nsig_outliers=nsig_outliers)
            else:
                detr, trend = ut.detrend_mov_ave(np.array(self.orig_fluxes[q]), length=detr_length, mask=msk,
                                                 method=detr_method,
                                                 endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen,
                                                 return_trend=True,
                                                 keep_overall_level=keep_overall_level, remove_outliers=remove_outliers,
                                                 nsig_outliers=nsig_outliers)
            detr[~msk] = np.nan
            self.detrended_orig_fluxes.append(detr)
            self.trends_orig_fluxes.append(trend)

    def initialize_templates(self, templates=None, shifts=None, template_ind='all',
                             templates_subbin_shift=Templates_subbin_shift,
                             template_bank_from_file=Template_bank_from_file,
                             template_bank_file=Template_bank_file,
                             shifts_template_bank_file=Shifts_template_bank_file,
                             pmin=Pmin):
        # TODO: write the template initialization in such a way that we could pass many template indices
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

        :param templates: Patterns to be used as a template (time domain). Should be
            all of the same length. The working part of the template is
            supposed to be in the center (the transit peak is in the center of
            the array). The ends will be padded by zeros to match self.support.
        :type templates: list of numpy.array 1d of float
        :param shifts:
        :type shifts:
        :param template_ind: Index of the template in template_module.template_bank() to be used
            if templates==None. If template_ind=='all', all the template bank will be used.
        :type template_ind: int, <len(template_module.template_bank())
        :param templates_subbin_shift:
        :type templates_subbin_shift:
        :param template_bank_from_file:
        :type template_bank_from_file:
        :param template_bank_file:
        :type template_bank_file:
        :param shifts_template_bank_file:
        :type shifts_template_bank_file:
        :param pmin:
        :type pmin:
        :return: creates self.initial_templates, self.templates, self.template_len
        :rtype:
        """
        # TODO: care about different shifts if needed
        if templates is not None:
            if isinstance(templates[0][0], float):
                self.initial_templates = [[t] for t in templates]
            else:
                self.initial_templates = templates
            if shifts is None:
                self.shifts = [[0.] for _ in range(len(self.initial_templates))]
            else:
                self.shifts = shifts
        elif template_bank_from_file:
            self.initial_templates = af.read_quarters_from_file(template_bank_file, dimension=3)
            self.shifts = af.read_quarters_from_file(shifts_template_bank_file, dimension=2)
            if template_ind != 'all':
                template_ind = int(template_ind)  # assert type int does not work with numpy.int64...
                self.initial_templates = np.array([self.initial_templates[template_ind]])
                self.shifts = np.array([self.shifts[template_ind]])  # Careful: only one index!!!
        elif template_ind == 'all':
            if templates_subbin_shift:
                self.initial_templates, self.shifts = tm.template_bank_subbin_shift(dt=self.dt_days,
                                                                                    dp=None, t_mission=self.T_mission,
                                                                                    pmin=pmin, n_points_int_min=50,
                                                                                    return_ordered=True, shifts_num=3,
                                                                                    return_shifts=True)
                # self.template_transit_lengths = [tm.transit_lengths(tem_shifts)
                # for tem_shifts in self.initial_templates]
            else:
                self.initial_templates = tm.template_bank(dt=self.dt_days)
                self.initial_templates = [template for template in self.initial_templates]
                self.shifts = [[0.] for _ in range(len(self.initial_templates))]
                # self.template_transit_lengths = tm.transit_lengths(self.initial_templates)
        else:
            template_ind = int(template_ind)  # assert type int does not work with numpy.int64...
            if templates_subbin_shift:
                initial_templates, shifts = tm.template_bank_subbin_shift(dt=self.dt_days,
                                                                          dp=None, t_mission=self.T_mission, pmin=Pmin,
                                                                          n_points_int_min=50, return_ordered=True,
                                                                          shifts_num=3,
                                                                          return_shifts=True)
                self.initial_templates[template_ind] = initial_templates[template_ind]
                self.shifts[template_ind] = shifts[
                    template_ind]  # Careful: only one template, still more than one index!!!
            else:
                self.initial_templates = [tm.template_bank(dt=self.dt_days)[template_ind]]
                self.shifts = [[0.] for _ in range(len(self.initial_templates))]
                # self.shifts[template_ind] = [0.]
        indlen = list(np.cumsum([len(s) for s in self.shifts]))
        # pairs of indices denoting the beginning and the end of different shifts of one template
        self.indices_of_templates = np.array(list(zip(np.concatenate(([0], indlen)), indlen)))
        self.shifts = np.hstack(self.shifts)
        self.template_len = len(self.initial_templates[0][0])
        self.templates = np.zeros((len(self.shifts), self.support))
        initial_templates_array = np.zeros((len(self.shifts), self.template_len))
        index = 0
        for template_with_shifts in self.initial_templates:
            for template in template_with_shifts:
                temp = np.concatenate((np.zeros((self.support - len(template)) // 2),
                                       template,
                                       np.zeros((self.support - len(template) + 1) // 2)))
                self.templates[index] = np.array(temp)
                initial_templates_array[index] = np.array(template)
                index += 1
        self.initial_templates = initial_templates_array
        self.templ_num = len(self.indices_of_templates)
        self.templ_num_shifts = len(self.templates)

        self.templates_of_indices = [k for k in range(len(self.indices_of_templates))
                                     for _ in range(self.indices_of_templates[k][0], self.indices_of_templates[k][1])]
        self.is_templates_initialized = True

    def notch_templates(self):
        """
        Applies notch filter to the template. That is, suppresses some frequencies
        which were marked before in the data conditioning part. Is relevant for
        detection only if used for the data as well and if the psds were not
        remeasured after notching.

        Requires
        ----------
        self.notch_soss : the data should be notched
        self.templates : the templates should be initialized

        Modifies
        ----------
        self.templates
            Notches templates in time domain. After this, need to whiten the templates
        """
        for k in range(self.templ_num_shifts):  # different template indices and different subbin shifts
            for sos in self.notch_soss:
                self.templates[k] = sosfiltfilt(sos, np.array(self.templates[k]), padtype='constant')
                # Very important to have zeros in the ends, bacause of padding.
                # Ppadtype 'constant' will pad by the edge value, and we need to ensure it is zero
                # That is why notching has to be done when the template is centered at the center,
                # and only then the template can be rolled.

    def roll_templates(self, whitened=False):
        if not whitened:
            for k in range(self.templ_num_shifts):  # different template indices
                self.templates[k] = np.roll(self.templates[k],
                                            -(len(self.templates[k]) // 2 - 1))
        else:
            for k in range(self.templ_num_shifts):  # different template indices
                for q in self.quarters:
                    self.whitened_templates_td[k][q] = np.roll(self.whitened_templates_td[k][q],
                                                               -(len(self.whitened_templates_td[k][q]) // 2 - 1))

    def whiten_templates(self, use_rfft=Use_rfft):
        """
        Whitens all the templates in self.templates by applying
        the whitening filter calculated after the PSD measurement of the data.

        Requires:
        self.wf_fd : the data should be whitened
        self.support : the data should be zero-padded

        Creates:
        self.whitened_templates_td : dict of len(templates) entries, each of length self.quarters
            Dictionary of all the whitened templates for all the quarters in time domain
        self.whitened_templates_fd : dict of len(templates) entries, each of length self.quarters
            Dictionary of all the whitened templates for all the quarters in Fourier domain

        :param use_rfft: Whether to use rfft or fft
        :type use_rfft: bool
        :return: self.whitened_templates_td, self.whitened_templates_fd
        :rtype:
        """
        assert self.is_templates_initialized, 'Templates have to be initialized!'
        self.whitened_templates_td = np.zeros((self.templ_num_shifts, len(self.quarters), self.support))
        self.whitened_templates_fd = np.zeros((self.templ_num_shifts, len(self.quarters), self.fft_support),
                                              dtype=np.complex128)
        for k in range(self.templ_num_shifts):  # different template indices
            for q in self.quarters:
                whitened_temp_fd = np.fft.rfft(self.templates[k]) * self.wf_fd[q]
                self.whitened_templates_fd[k][q] = whitened_temp_fd
                if use_rfft:
                    self.whitened_templates_td[k][q] = np.fft.irfft(whitened_temp_fd)
                else:
                    self.whitened_templates_td[k][q] = np.fft.ifft(whitened_temp_fd)
        self.is_template_whitened = True

    def matched_filtering(self, detr_length_outliers=Detr_length_outliers,
                          nsig_outliers_stat=Nsig_outliers_stat,
                          ngap_moving_var=Ngap_moving_var,
                          nsamples_min_moving_var=Nsamples_min_moving_var):
        """
        Calculates the matched-filtering statistics of data and templates.
        Cuts the statistic from zero padding ends.

        Requires:
        self.whitened_fluxes_fd : fluxes should be whitened
        self.whitened_templates_fd : templates should be whitened
        self.masks : need masks for std calculation, but will ignore small holes
        self.support : data should be zero-padded

        Creates:
        # self.statistics_raw : h@C**(-1)@d. Just the matched-filtering score without denominator
        self.stat_moving_vars : Moving variance of statistics_raw
        self.statistics_snr : h@C**(-1)@d/(h@C**(-1)@h)**0.5 Statistics in units of SNR.
        #self.statistics_abs : h@C**(-1)@d/(h@C**(-1)@h) Statistics in absolute units.
        self.stat_stds : Standard deviation of the self.statistics_raw
        self.stat_denominators : (h@C**(-1)@h)**0.5
            *where h is the template, d is the flux, C is the covariance matrix, @ is the
            inner product, and the formulae are written schematically. In reality,
            they are implemented with whitened data and fluxes.

        :param detr_length_outliers:
        :type detr_length_outliers:
        :param nsig_outliers_stat:
        :type nsig_outliers_stat:
        :param ngap_moving_var:
        :type ngap_moving_var:
        :param nsamples_min_moving_var:
        :type nsamples_min_moving_var:
        :return:
        :rtype:
        """
        # self.statistics_raw = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for _ in range(self.templ_num_shifts)]
        self.statistics_raw_cor = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for _ in range(self.templ_num_shifts)]
        # -> corrected for moving std and then multiplied back by non-moving std
        self.stat_moving_vars = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for _ in range(self.templ_num_shifts)]
        self.statistics_snr = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for _ in range(self.templ_num_shifts)]
        # self.statistics_abs = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for t in range(self.templ_num_shifts)]
        self.stat_vars = [[0. for _ in self.quarters] for __ in range(self.templ_num_shifts)]  # mean of moving variance
        self.stat_denoms = [[np.inf for _ in self.quarters] for __ in range(self.templ_num_shifts)]  # mean of denom of stat
        # TODO: think about outlier free moving variance
        self.stat_masks = [[np.ones(len(self.fluxes[q])).astype(bool) for q in self.quarters] for t in
                           range(self.templ_num_shifts)]

        assert self.is_template_whitened, 'Templates have to be whitened for matched filtering!'

        for k in range(self.templ_num_shifts):
            for q in self.quarters:
                # statistics not in snr, not in absolute units
                stat = np.real(
                    np.fft.irfft(np.conjugate(self.whitened_templates_fd[k][q]) * self.whitened_fluxes_fd[q]))
                stat = np.array(
                    stat[self.zero_padding_ends[q][0]:len(stat) - self.zero_padding_ends[q][1]])  # CUT THE STATISTIC
                # This denom will be needed for the amplitude calculation:
                stat_denom = np.fft.irfft(np.abs(self.whitened_templates_fd[k][q])**2)[0]

                statmask = np.array(self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(self.big_hole_masks[q]) -
                                                                                            self.zero_padding_ends[q][1]])
                # self.statistics_raw[k][q] = stat
                movvar, tot_var = ut.gap_var(np.array(stat), mask=np.array(statmask), ngap=ngap_moving_var,
                                             nsamples_min=nsamples_min_moving_var,
                                             remove_outliers=False, remove_outliers_light=True,
                                             detr_length_outliers=detr_length_outliers,
                                             nsig_outliers=nsig_outliers_stat, return_mask=False, return_total_var=True)
                movvar[~statmask] = np.inf
                self.stat_moving_vars[k][q] = movvar
                self.statistics_snr[k][q] = stat / movvar ** 0.5
                statrawcor = stat / movvar * np.nanmean(movvar[statmask])
                statrawcor[~statmask] = 0.
                self.statistics_raw_cor[k][q] = statrawcor
                self.stat_vars[k][q] = np.nanmean((movvar[statmask]) ** 0.5) ** 2
                self.stat_masks[k][q] = statmask
                self.stat_denoms[k][q] = stat_denom

    def cut_zero_padding_ends(self):
        """
        Removes the ends that were padded by zeros and saves the data in the same field.

        Modifies
        -------
        self.fluxes
        self.times
        self.masks
        self.whitened_fluxes_td
        self.statistics_snr
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
                    self.whitened_fluxes_td[q][
                     self.zero_padding_ends[q][0]:len(self.whitened_fluxes_td[q]) - self.zero_padding_ends[q][-1]]))
            if self.whitened_templates_td is not None:
                for k in range(self.templ_num_shifts):
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

    def time_slides(self, std_time_scrambling=Std_time_scrambling):
        np.random.seed(None)
        r = np.abs(np.random.normal(0, std_time_scrambling, len(self.times)))
        r = np.cumsum(r)
        for i in range(len(self.times)):
            self.times[i] = np.array(self.times[i]) - self.time_slide_shifts[i] + r[
                i]  # we undo old time slides and do new
        self.time_slide_shifts = r

        self.tmin = np.nanmin(self.times[0])
        self.tmax = np.nanmax(self.times[0])
        for t in self.times[1:]:
            self.tmin = min(np.nanmin(t), self.tmin)
            self.tmax = max(np.nanmax(t), self.tmax)

    def undo_time_slides(self):
        for i in range(len(self.times)):
            self.times[i] = np.array(self.times[i]) - self.time_slide_shifts[i]
        self.time_slide_shifts = np.zeros(len(self.times))

        self.tmin = np.nanmin(self.times[0])
        self.tmax = np.nanmax(self.times[0])
        for t in self.times[1:]:
            self.tmin = min(np.nanmin(t), self.tmin)
            self.tmax = max(np.nanmax(t), self.tmax)

    def invert_statistic(self, choose_random=False):
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

    def veto_statistics(self, apply_only_to_neg=False, margin_snr=Margin_snr_veto, fn=Fn_fit_veto, maxdev=Maxdev_fit_veto,
                        rfft_psd=Use_rfft, zero_ends_len_abs=Zero_ends_len_abs_fit_veto,
                        zero_ends_len_rel=Zero_ends_len_rel_fit_veto, nsig_outliers_var=Nsig_outliers_stat,
                        n_parts_veto_snr=N_parts_veto_snr, n_parts_veto_len=N_parts_veto_len,
                        inn_prod_min_veto=Inn_prod_min_veto,
                        delta_snr2_wide_templ=Delta_snr2_wide_templ_veto, threshold_to_std=Threshold_to_std_veto,
                        save_triggers_as_attribute=False):
        # TODO: decide about detrending. What to detrend (initial flux or final), to detrend or not
        # are the long transits important?
        # if the transit is near the edge and is big, it is better not to use detrending.
        """
        Verify big triggers and select only the appropriate ones.
        Selects from the statistic the triggers whose abs is >margin_snr, which are negative
        and which are not inside big holes. Cuts the data to select the appropriate slice
        around the trigger index, aligns data with the template and then calls veto_combined which:
        Checks the goodness of fit of data by the template by chi squared and maximum deviation. The comparison
        is made with respect to the best chi squared fit template to the data. The conditions for
        the fit to be good are:
            - The chi squared value is smaller than the point where the chi squared distribution with the number
              of degrees of freedom equal to the lengts of the template (truncated from extra zeros) is fn;
            - Maximum relative deviation (in units of std) is smaller than maxdev.

        Requires:
        self.statistics_snr
        self.templates
        self.stat_masks
        self.fluxes
        self.psds

        Modifies:
        self.triggers : list of the info about the tests of all the negative
                        triggers which are more significant than `margin_snr_veto`
            Data type description:
            dict with quarters as keys{
                dict of trigger indices as keys{
                    dict with templates as keys{
                        [isgood : whether the trigger is good or vetted,
                         reasons : why it is bad, which test it failed ('' if isgood),
                         [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod, inn_prod_min] : results of veto_combined
                         [score, local_threshold, total_threshold] : results of inconsistent_parts_snr
                         [score, local_threshold, total_threshold] : results of inconsistent_parts_len
                         [snr**2, snrmax**2] : results of template_width veto
                         ]}}}
        self.trigger_amplitudes :
            Adds to dict list of len(#quarters) of numpy.arrays 1d,
            corresponfing to self.triggers.
            Amplitudes in absolute units (quarter-specific) determined from fitting
        self.veto_masks :
            Adds to dict list of len(#quarters) of numpy.arrays 1d of len support.
            Masks in statistic space. If False, the statistic in this index has high absolute
            value > margin_snr, byt was vetoed as bad.

        :param margin_snr: The minimal abs(snr) that is considered as a possible transit candidate. Values
            greater that are checked by sign (only negative are accepted) and shape
            (only good-fitting the template are accepted).
        :type margin_snr: float
        :param fn: False negative value of the chi-squared distribution with the number of degrees of freedom equal to the
            length of the truncated template. Note that this distribution is without accounting for multiple-comparison
            effect (we are fitting the best amplitude for the template, to the real distribution of the scores
            will be shrunk closer to zero)
        :type fn: float
        :param maxdev: Maximum allowed absolute deviation of the data from the best-fit template.
        :type maxdev: float
        :param rfft_psd:
        :type rfft_psd:
        :param zero_ends_len_abs: length of zeros to take at the end of the template from each side in addition
            to the non-zero region of the template. Is needed to make sure that the shape
            of the templite is intdeed fulfilled. Is dominant over zero_end_len_rel
        :type zero_ends_len_abs: int or None
        :param zero_ends_len_rel: If zero_end_len_abs is None, is used to calculate new zero_end_len_abs
            as [(length of nonzero template)*zero_end_len_rel]. This amount of
            Bins is used from each side.
        :type zero_ends_len_rel: float
        :param nsig_outliers_var:
        :type nsig_outliers_var:
        :param n_parts_veto_snr:
        :type n_parts_veto_snr:
        :param n_parts_veto_len:
        :type n_parts_veto_len:
        :param inn_prod_min_veto:
        :type inn_prod_min_veto:
        :param delta_snr2_wide_templ:
        :type delta_snr2_wide_templ:
        :param threshold_to_std:
        :type threshold_to_std:
        :return: modifies self.triggers, self.trigger_amplitudes, self.veto_masks
        :rtype:
        """

        # TODO: when one veto triggered, do not do other vetos, this point will be masked in any case (do it after you debug)
        assert not self.is_zero_padded, 'Zero padding ends have to be removed before vetting!'
        self.stats_raw_no_veto_no_score_cor = copy.deepcopy(self.statistics_snr)

        triggers = [{} for _ in self.quarters]
        for k in range(self.templ_num_shifts):
            template = self.initial_templates[k]
            halflen = len(template) // 2
            for q in self.quarters:  # over quarters
                statistic = np.array(self.statistics_snr[k][q])
                mask = np.array(self.stat_masks[k][q])  # allow small holes to be in the transit
                # vetoing bad-fitting (negative-amplitude) template triggers:
                trigger_indices_quarter = None
                if apply_only_to_neg:
                    trigger_indices_quarter = np.where((statistic <= 0) & (np.abs(statistic) >= margin_snr) & mask)[0]
                else:
                    trigger_indices_quarter = np.where((np.abs(statistic) >= margin_snr) & mask)[0]
                data_cut = None
                template_cut = None
                for i in trigger_indices_quarter:
                    to_veto_combined = True  # to process to put data to the vetoing function
                    if (i - halflen >= 0) and (i + len(template) - halflen <= len(self.fluxes[q])):
                        data_cut = np.array(self.fluxes[q][i - halflen:i + len(template) - halflen])
                        template_cut = np.array(template)
                    elif (i - halflen >= 0) and np.isclose(template[halflen + len(self.fluxes[q]) - i], 0):
                        data_cut = np.array(self.fluxes[q][i - halflen:])
                        template_cut = np.array(template[:len(data_cut)])
                    elif (i + len(template) - halflen <= len(self.fluxes[q])) and np.isclose(template[halflen - i], 0):
                        data_cut = np.array(self.fluxes[q][:i + len(template) - halflen])
                        template_cut = np.array(template[len(template) - len(data_cut):])
                    elif (len(template) > len(self.fluxes[q]) and np.isclose(
                            template[halflen + len(self.fluxes[q]) - i], 0) and np.isclose(template[halflen - i], 0)):
                        data_cut = np.array(self.fluxes[q])
                        template_cut = np.array(template[halflen - i:halflen - i + len(data_cut)])
                    else:
                        if i not in triggers[q].keys():
                            triggers[q][i] = {k: [False, 'extreme_position ', [], [], [], []]}
                        else:
                            triggers[q][i][k] = [False, 'extreme_position ', [], [], [], []]
                        to_veto_combined = False  # Already marked inappropriate, will not be treated with veto_combined
                    if to_veto_combined:
                        # TODO: remind myself why the chi2 veto is done on non-whitened fluxes
                        isgood, scores, reason = ut.veto_combined(data=data_cut, template=template_cut,
                                                                  fn=fn, maxdev_std=maxdev, maxdev_ampl=None,
                                                                  zero_ends_len_abs=zero_ends_len_abs,
                                                                  zero_ends_len_rel=zero_ends_len_rel,
                                                                  return_amplitude=False, return_reason=True,
                                                                  inn_prod_min=inn_prod_min_veto, return_scores=True)
                        if i not in triggers[q].keys():
                            if isgood:
                                triggers[q][i] = {k: [isgood, '', scores, [], [], []]}
                            else:
                                triggers[q][i] = {k: [isgood, reason, scores, [], [], []]}
                        else:
                            if isgood:
                                triggers[q][i][k] = [isgood, '', scores, [], [], []]
                            else:
                                triggers[q][i][k] = [isgood, reason, scores, [], [], []]

                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING BY SNR)
                isgood_parts, scores, local_th, total_th = ut.veto_parts(np.array(self.whitened_fluxes_td[q]),
                                                                         np.array(self.whitened_templates_td[k][q]),
                                                                         indices=trigger_indices_quarter,
                                                                         mask=np.array(self.big_hole_masks[q]), tlen=None,
                                                                         fn=fn,
                                                                         n_parts=n_parts_veto_snr,
                                                                         nsig_outliers_var=nsig_outliers_var,
                                                                         splitting='snr',
                                                                         return_amplitudes=False, return_scores=True)
                for j in range(len(trigger_indices_quarter)):
                    triggers[q][trigger_indices_quarter[j]][k][0] &= isgood_parts[j]
                    if not isgood_parts[j]:
                        triggers[q][trigger_indices_quarter[j]][k][1] += 'inconsistent_parts_snr '
                    triggers[q][trigger_indices_quarter[j]][k][3] = [scores[j], local_th, total_th]

                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING BY LENGTH)
                isgood_parts, scores, local_th, total_th = ut.veto_parts(np.array(self.whitened_fluxes_td[q]),
                                                                         np.array(self.whitened_templates_td[k][q]),
                                                                         indices=trigger_indices_quarter,
                                                                         mask=np.array(self.big_hole_masks[q]), tlen=None,
                                                                         fn=fn,
                                                                         n_parts=n_parts_veto_len,
                                                                         nsig_outliers_var=nsig_outliers_var,
                                                                         splitting='len',
                                                                         return_amplitudes=False, return_scores=True)
                for j in range(len(trigger_indices_quarter)):
                    triggers[q][trigger_indices_quarter[j]][k][0] &= isgood_parts[j]
                    if not isgood_parts[j]:
                        triggers[q][trigger_indices_quarter[j]][k][1] += 'inconsistent_parts_len '
                    triggers[q][trigger_indices_quarter[j]][k][4] = [scores[j], local_th, total_th]

        # VETO OF NARROW TEMPLATES TRIGGERING ON WIDER TRANSITS
        for q in self.quarters:
            for ind in triggers[q].keys():
                snrs = []
                for k in triggers[q][ind].keys():
                    snrs.append(self.statistics_snr[k][q][ind])
                timax = np.argmax(np.abs(snrs))  # snr is negative, but this will find also positive things
                tmax = list(triggers[q][ind].keys())[timax]
                snrmax2 = snrs[timax] ** 2
                for ik in range(len(snrs)):
                    k = list(triggers[q][ind].keys())[ik]
                    if (k < tmax) and (snrs[ik] ** 2 < snrmax2 - delta_snr2_wide_templ):  # TODO: better check for coincidence between templates
                        is_good_temp = False
                    else:
                        is_good_temp = True
                    iot = self.indices_of_templates[self.templates_of_indices[k]]
                    for ts in np.arange(iot[0], iot[1]):
                        if ts not in triggers[q][ind].keys():
                            if not is_good_temp:
                                triggers[q][ind][ts] = [is_good_temp, 'template_width ', [], [], [],
                                                             [snrs[ik] ** 2, snrmax2]]
                            else:
                                triggers[q][ind][ts] = [is_good_temp, '', [], [], [], [snrs[ik] ** 2, snrmax2]]
                        else:
                            triggers[q][ind][ts][0] &= is_good_temp
                            if not is_good_temp:
                                if 'template_width' not in triggers[q][ind][ts][1]:
                                    triggers[q][ind][ts][1] += 'template_width '
                            triggers[q][ind][ts][5] = [snrs[ik] ** 2, snrmax2]

        self.veto_masks = [[np.ones(len(self.stat_masks[k][q])) for q in self.quarters] for k in
                           range(len(self.stat_masks))]
        self.trigger_masks = [[np.ones(len(self.stat_masks[k][q])) * np.nan for q in self.quarters] for k in
                              range(len(self.stat_masks))]
        self.veto_bad_trigger_inds = [[[[], []] for _ in self.quarters] for __ in range(self.templ_num_shifts)]

        for q in self.quarters:
            for ind in triggers[q].keys():
                for k in triggers[q][ind].keys():
                    self.trigger_masks[k][q][ind] = triggers[q][ind][k][0]
                    if not triggers[q][ind][k][0]:
                        self.veto_bad_trigger_inds[k][q][0].append(ind)
                        self.veto_bad_trigger_inds[k][q][1].append(triggers[q][ind][k][1])

        for k in range(len(self.veto_bad_trigger_inds)):
            for q in self.quarters:
                self.veto_masks[k][q][self.veto_bad_trigger_inds[k][q][0]] = False
                self.stat_masks[k][q][self.veto_bad_trigger_inds[k][q][0]] = False

        for k in range(len(self.veto_bad_trigger_inds)):
            for q in self.quarters:
                self.statistics_snr[k][q][~self.stat_masks[k][q]] = 0.
                self.statistics_raw_cor[k][q][~self.stat_masks[k][q]] = 0.

        if save_triggers_as_attribute:
            self.triggers = triggers

    def correct_score(self):
        # TODO: think how zeros in the masked region influence the period-summed distribution
        all_stat = np.array([])
        for k in range(len(self.statistics_snr)):
            for q in range(len(self.statistics_snr[k])):
                all_stat = np.concatenate((all_stat, self.statistics_snr[k][q][self.stat_masks[k][q]]))
        #  all_stat_c, transform = ut.score_correction(all_stat, return_function=True, negative_values=True)
        all_stat_c, transform = ut.score_correction(all_stat, return_function=True, two_sided_distr=True)
        # -> we do not use one-sided distr or do????????
        for k in range(len(self.statistics_snr)):
            for q in range(len(self.statistics_snr[k])):
                self.statistics_snr[k][q][self.stat_masks[k][q]] = transform(
                    self.statistics_snr[k][q][self.stat_masks[k][q]])
                self.statistics_raw_cor[k][q][self.stat_masks[k][q]] = (self.statistics_snr[k][q][self.stat_masks[k][q]] /
                                                                        self.stat_moving_vars[k][q][self.stat_masks[k][q]] ** 0.5
                                                                        * np.nanmean(self.stat_moving_vars[k][q][self.stat_masks[k][q]]))

    def connect_quarters(self):
        # TODO: is it legite to connect quarters with such a shift? ( what??)
        """
        Takes data saved by quarter and connects them together to the common time axis.
        The new time array is generated, assuming that data is equally sampled.
        Then, the data quarters are appended to the data array according to their first time value.
        If there is a subbin shift in time between the quarters, the quarter is put to the
        closest tome bin

        Creates
        -------
        self.statistics_full : for all the templates and the shifts
        self.stat_masks_full : for all the templates and the shifts
        self.stat_vars_full : for all the templates and the shifts
        """
        self.time_full, self.trend_orig_flux_full = af.connect_quarters_fun(self.times, self.trends_orig_fluxes,
                                                                            zero_padding_ends=None,
                                                                            is_data_cut=~self.is_zero_padded,
                                                                            is_time_cut=~self.is_zero_padded,
                                                                            dt=self.dt_days, fill_gaps_by_nan=False)
        self.statistics_full = np.zeros((self.templ_num_shifts, len(self.time_full)))
        self.stat_vars_full = np.zeros((self.templ_num_shifts, len(self.time_full)))
        self.stat_masks_full = np.zeros((self.templ_num_shifts, len(self.time_full))).astype(bool)
        vars_quarters = [[np.ones(len(self.statistics_raw_cor[k][q])) * self.stat_vars[k][q]
                          for q in self.quarters] for k in range(self.templ_num_shifts)]

        for k in range(self.templ_num_shifts):
            self.statistics_full[k] = af.connect_quarters_fun(self.times, self.statistics_raw_cor[k],
                                                              zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                              is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                              fill_gaps_by_nan=False)[1]
            self.stat_masks_full[k] = (af.connect_quarters_fun(self.times, self.stat_masks[k],
                                                               zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                               is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                               fill_gaps_by_nan=False)[1]).astype(bool)
            self.stat_vars_full[k] = af.connect_quarters_fun(self.times, vars_quarters[k],
                                                             zero_padding_ends=None, is_data_cut=~self.is_zero_padded,
                                                             is_time_cut=~self.is_zero_padded, dt=self.dt_days,
                                                             fill_gaps_by_nan=False)[1]

    def interleave_shifts(self):
        """

        """
        self.interleaved_numbers = np.array([i for i in range(len(self.indices_of_templates))
                                             if ((self.indices_of_templates[i][1] - self.indices_of_templates[i][0]) >= 2)])
        self.not_interleaved_numbers = np.array([i for i in range(len(self.indices_of_templates))
                                                 if ((self.indices_of_templates[i][1] - self.indices_of_templates[i][0]) == 1)])
        # for interleaved:
        if len(self.interleaved_numbers) > 0:
            shift_numbers = np.array(
                [self.indices_of_templates[i][1] - self.indices_of_templates[i][0] for i in self.interleaved_numbers])
            # number of shifts for all the templates with shifts. need them to be the same
            self.n_shifts = shift_numbers[0]
            # Make sure they are all the same number of shifts:
            assert (shift_numbers / shift_numbers[0] == np.ones(len(
                shift_numbers))).all(), 'Templates with shifts must have the same number of shifts!'
            self.statistics_interleaved = []
            self.stat_masks_interleaved = []
            self.stat_vars_interleaved = []
            for t in self.interleaved_numbers:
                self.statistics_interleaved.append(
                    (self.statistics_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
                self.stat_masks_interleaved.append(
                    (self.stat_masks_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
                self.stat_vars_interleaved.append(
                    (self.stat_vars_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
            self.statistics_interleaved = np.array(self.statistics_interleaved)
            self.stat_masks_interleaved = np.array(self.stat_masks_interleaved).astype(bool)
            self.stat_vars_interleaved = np.array(self.stat_vars_interleaved)
            self.trend_orig_flux_interleaved = np.vstack((
                self.trend_orig_flux_full, self.trend_orig_flux_full,
                self.trend_orig_flux_full)).T.flatten()
            self.time_interleaved = np.arange(len(self.trend_orig_flux_interleaved)) * self.dt_days / self.n_shifts + self.time_full[0]

        # for not interleaved:
        if len(self.not_interleaved_numbers) > 0:
            self.statistics_not_interleaved = []
            self.stat_masks_not_interleaved = []
            self.stat_vars_not_interleaved = []
            indices_for_not_interleaved = [self.indices_of_templates[i][0] for i in self.not_interleaved_numbers]
            self.statistics_not_interleaved = np.array(self.statistics_full[indices_for_not_interleaved])
            self.stat_masks_not_interleaved = np.array(self.stat_masks_full[indices_for_not_interleaved])
            self.stat_vars_not_interleaved = np.array(self.stat_vars_full[indices_for_not_interleaved])
            self.trend_orig_flux_not_interleaved = np.array(self.trend_orig_flux_full)
            self.time_not_interleaved = np.arange(len(self.trend_orig_flux_not_interleaved)) * self.dt_days + self.time_full[0]

        del self.time_full
        del self.statistics_full
        del self.stat_vars_full
        del self.stat_masks_full
        del self.trend_orig_flux_full

    def delete_irrelevant(self):
        if hasattr(self, 'clean_fluxes'):
            del self.clean_fluxes
        if hasattr(self, 'clean_masks'):
            del self.clean_masks
        if hasattr(self, 'whitened_clean_fluxes_fd'):
            del self.whitened_clean_fluxes_fd
        if hasattr(self, 'whitened_clean_fluxes_td'):
            del self.whitened_clean_fluxes_td
        if hasattr(self, 'whitened_templates_fd'):
            del self.whitened_templates_fd
        if hasattr(self, 'whitened_templates_td'):
            del self.whitened_templates_td
        if hasattr(self, 'initial_templates'):
            del self.initial_templates
        # del self.statistics_abs #not used anymore
        if hasattr(self, 'stat_moving_vars'):
            del self.stat_moving_vars
        if hasattr(self, 'notch_soss'):
            del self.notch_soss
        if hasattr(self, 'wf_td'):
            del self.wf_td
        if hasattr(self, 'wf_fd'):
            del self.wf_fd
        # del self.statistics_full #will be deleted later
        # del self.time_full #will be deleted later
        if hasattr(self, 'psd_masks'):
            del self.psd_masks
        # del self.zero_padding_ends
        if hasattr(self, 'notch_mask'):
            del self.notch_mask
        if hasattr(self, 'templates'):
            del self.templates

        if hasattr(self, 'npsds0'):
            del self.npsds0
        if hasattr(self, 'npsds'):
            del self.npsds

    def sum_periods(self, pmin=Pmin, pmax=Pmax, dp=None, duty_cycle=Duty_cycle):
        """
        Sums the statistics with all the possible periods and initial phases. Accounts for
        subbin template shifts, accounts for masks.

        Creates:
        self.statistics_periodic,
        self.statistics_periodic_counts,
        self.statistics_phases,
        self.statistics_periodic_lengths_filled,
        self.statistics_periodic_relative_amplitudes

        :param pmin: Minimal period, days
        :type pmin: float
        :param pmax: Maximal period to consider.
        :type pmax: float
        :param dp: Step in the list of all the investigated periods
        :type dp: float
        :param duty_cycle:
        :type duty_cycle:
        :return:
        :rtype:
        """
        """
        

        Parameters
        ----------
        statistics_interleaved,  statistics_not_interleaved: numpy.array 2d of float
            Raw (not SNR units, not abs units) corrected (for moving variance)
            statistics for all the templates. Quarters connected
        stat_masks_full : numpy.array 2d of bool
            Masks for `statistics_interleaved`,  `statistics_not_interleaved`  .
        stat_vars_full : numpy.array 2d of float
            Variances of statistics in the corresponding quarter. Arrays for each
            template (with shifts) of arrays of variances which are constant inside
            pieces corresponding to one quarter.
        trend_orig_flux_full : numpy.array 1d of float
            Trend of the data (quarters connected).
        shifts : numpy.array 1d of float
            Template subbin shifts corresponding to each entry in `self.templates`.
        indices_of_templates : numpy.array 2d of int
            Lists of lists of two indices denoting the beginning and the end of entries
            corresponding to different shifts ofone template in `self.templates`.
        dt : float
            Time step of equally sampled time axis.
        tmax : float
            Maximal time in the time axis.
        tmin : float
            Minimal time in the time axis.
        """

        self.Pmin = pmin
        self.Pmax = pmax
        per = pmin
        periods = [per]
        if dp is None:
            while per <= pmax:
                dp = duty_cycle * per / self.T_mission
                per += dp
                periods.append(per)
        else:
            while per <= pmax:
                per += dp
                periods.append(per)
        periods = np.array(periods)
        self.periods = periods
        self._phases_interleaved = []
        self._phases_not_interleaved = []
        self.statistics_periodic_not_interleaved = []
        self.statistics_periodic_interleaved = []
        # attention: we use the same dt for interleaved and not? it means that the phases are the same? phases are defined further.

        # interleaved
        if len(self.interleaved_numbers) > 0:
            self._phases_interleaved = np.arange(int(np.ceil((self.periods[-1]) / (self.dt_days / self.n_shifts)))) * (
                    self.dt_days / self.n_shifts)
            # Warning: we only try these phases and not something in-between. It should be ok, but still
            statistics_by_trends = (self.statistics_interleaved * self.trend_orig_flux_interleaved)
            statistics_by_trends = np.hstack((statistics_by_trends, np.zeros(
                (len(statistics_by_trends), int(np.ceil(Pmax / (self.dt_days / self.n_shifts)))))))
            # These phases are with respect to the beginning of the data which can be any
            # number because sometimes we reject the zeroth quarter or do other things.
            # Do not use these phases anywhere.
            # Pad by zeros to avoid extra if statements in the summation:
            trends_sq_by_vars = self.stat_vars_interleaved * self.trend_orig_flux_interleaved ** 2
            trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros(
                (len(trends_sq_by_vars), int(np.ceil(Pmax / (self.dt_days / self.n_shifts)))))))

            self.statistics_periodic_interleaved, denomsq = self.sum_periods_fast(
                                                              statistics_by_trends, trends_sq_by_vars, self.dt_days / self.n_shifts,
                                                              self.tmax + self.dt_days * (1 - 1 / self.n_shifts), self.tmin,periods)
            # attention: we use the same dt for interleaved and not? it means that the phase step for the interleaved could be smaller.
            for p in range(len(self.statistics_periodic_interleaved)):
                for t in range(len(self.statistics_periodic_interleaved[p])):
                    self.statistics_periodic_interleaved[p][t][denomsq[p][t] == 0] = np.nan
                    self.statistics_periodic_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][
                                                                                          denomsq[p][t] != 0] ** 0.5

        # not interleaved
        if len(self.not_interleaved_numbers) > 0:
            self._phases_not_interleaved = np.arange(int(np.ceil((self.periods[-1]) / self.dt_days))) * self.dt_days
            statistics_by_trends = (self.statistics_not_interleaved * self.trend_orig_flux_not_interleaved)
            statistics_by_trends = np.hstack((statistics_by_trends, np.zeros(
                (len(statistics_by_trends), int(np.ceil(Pmax / self.dt_days))))))
            # These phases are with respect to the beginning of the data which can be any
            # number because sometimes we reject the zeroth quarter or do other things.
            # Do not use these phases anywhere.
            # Pad by zeros to avoid extra if statements in the summation:
            trends_sq_by_vars = self.stat_vars_not_interleaved * self.trend_orig_flux_not_interleaved ** 2
            trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros(
                (len(trends_sq_by_vars), int(np.ceil(Pmax / self.dt_days))))))

            self.statistics_periodic_not_interleaved, denomsq = self.sum_periods_fast(
                statistics_by_trends, trends_sq_by_vars, self.dt_days, self.tmax, self.tmin, periods)
            for p in range(len(self.statistics_periodic_not_interleaved)):
                for t in range(len(self.statistics_periodic_not_interleaved[p])):
                    self.statistics_periodic_not_interleaved[p][t][denomsq[p][t] == 0] = np.nan
                    self.statistics_periodic_not_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][
                                                                                              denomsq[p][t] != 0] ** 0.5

    def veto_mes(self, mes_threshold=Threshold_mes_veto, veto_threshold=Veto_threshold_for_mes):
        # WARNING: this veto, if done after the periodicity search on the score-corrected data, will destroy the
        # distribution. Therefore, we are removing this veto from the search and will do it after the search and after
        # the time-slides and p_periodic estimation as a post-processing veto. Also, we make sure to do it on the
        # non-score-corrected data.
        if len(self.interleaved_numbers) > 0:
            self.mes_veto_masks_interleaved = np.ones(np.shape(self.statistics_periodic_interleaved)).astype(bool)
            indices = np.array(np.where(self.statistics_periodic_interleaved < -np.abs(mes_threshold))).T
            for pi, ti, phi in indices:
                expected_transit_times = [self.tmin + self._phases_interleaved[phi] + i * self.periods[pi]
                                          for i in range(
                        int(np.ceil((self.tmax - self.tmin - self._phases_interleaved[phi]) / self.periods[pi])))]

                summed_transit_points = [np.argmin(np.abs(self.time_interleaved - expected_transit_times[i])) for i in
                                         range(len(expected_transit_times))]
                summed_stats = np.array(self.statistics_interleaved[ti][summed_transit_points])  # raw
                summed_trends = np.array(self.trend_orig_flux_interleaved[summed_transit_points])
                summed_vars = np.array(self.stat_vars_interleaved[ti][summed_transit_points])
                amplitude_vars = []
                amplitudes_normalized = []
                for si in range(len(summed_vars)):
                    if not np.isclose(summed_stats[si], 0):
                        amplitude_vars.append(1 / summed_vars[si])
                        amplitudes_normalized.append(
                            summed_stats[si] / summed_vars[si] / summed_trends[si] * self.mean_total)
                    else:
                        amplitude_vars.append(np.inf)
                        amplitudes_normalized.append(0.)
                veto_score = ut.unequal_amplitude_veto(values=-np.array(amplitudes_normalized), variances=amplitude_vars,
                                                       A_all=None, indmax=None, A_1=None)
                if veto_score > veto_threshold:
                    self.statistics_periodic_interleaved[pi][ti][phi] = 0.
                    self.mes_veto_masks_interleaved[pi][ti][phi] = False

        if len(self.not_interleaved_numbers) > 0:
            self.mes_veto_masks_not_interleaved = np.ones(np.shape(self.statistics_periodic_not_interleaved)).astype(
                bool)
            indices = np.array(np.where(self.statistics_periodic_not_interleaved < -np.abs(mes_threshold))).T
            for pi, ti, phi in indices:
                expected_transit_times = [self.tmin + self._phases_not_interleaved[phi] + i * self.periods[pi]
                                          for i in range(
                        int(np.ceil((self.tmax - self.tmin - self._phases_not_interleaved[phi]) / self.periods[pi])))]

                summed_transit_points = [np.argmin(np.abs(self.time_not_interleaved - expected_transit_times[i])) for i
                                         in range(len(expected_transit_times))]
                summed_stats = np.array(self.statistics_not_interleaved[ti][summed_transit_points])  # raw
                summed_trends = np.array(self.trend_orig_flux_not_interleaved[summed_transit_points])
                summed_vars = np.array(self.stat_vars_not_interleaved[ti][summed_transit_points])
                amplitude_vars = []
                amplitudes_normalized = []
                for si in range(len(summed_vars)):
                    if (not np.isclose(summed_stats[si], 0) and not np.isinf(summed_vars[si])):
                        amplitude_vars.append(1 / summed_vars[si])
                        amplitudes_normalized.append(
                            summed_stats[si] / summed_vars[si] / summed_trends[si] * self.mean_total)
                    else:
                        amplitude_vars.append(np.inf)
                        amplitudes_normalized.append(0.)
                if len(np.array(amplitudes_normalized)) < 3:
                    veto_score = np.inf
                else:
                    veto_score = ut.unequal_amplitude_veto(values=-np.array(amplitudes_normalized),
                                                        variances=amplitude_vars, A_all=None, indmax=None, A_1=None)
                if veto_score > veto_threshold:
                    self.statistics_periodic_not_interleaved[pi][ti][phi] = 0.
                    self.mes_veto_masks_not_interleaved[pi][ti][phi] = False

    def delete_mes(self):
        del self.periods
        del self._phases_interleaved
        del self._phases_not_interleaved
        del self.statistics_periodic_not_interleaved
        del self.statistics_periodic_interleaved

    def save_best_candidates(self, n_per_temp=N_per_temp_saving, ngap_moving_var=Ngap_moving_var, path=None,
                             save_to_file=True, save_inverted=True):
        """
        Determines `n_per_temp` best scores for each template. For each period,
        finds one best phase corresponding to min (negative) statistic. Then, selects
        `n_per_temp` best periods for every template. For them, saves statistic values,
        periods, phases, summed values (see the description of returned results).

        Modifies:
        self.best_periodic_scores_data : multi-level non-uniform nested list
            A structure saving info about `n_per_temp` best scores per each template:
        period_indices_selected : list of lists of len `n_per_temp`
            `n_per_temp` period indices of best statistics for each template
        periods_selected : list of lists of len `n_per_temp`
            `n_per_temp` periods of best statistics for each template
        phase_indices_selected : list of lists of len `n_per_temp`
            best phase indices for the `n_per_temp` periods (one oer period)
            corresponding to best statistics for each template
        phases_selected : list of lists of len `n_per_temp`
            best phases for the `n_per_temp` periods (one oer period)
            corresponding to best statistics for each template
        statistic_values_selected : list of lists of len `n_per_temp`
            `n_per_temp` best statistins (with no repeating periods) for each template
        summed_transit_times_selected : list of lists of len `n_per_temp` of lists
            times used for summation to get statistic_values_selected
        summed_stats_raw_selected : list of lists of len `n_per_temp` of lists
            raw statistics used for summation to get statistic_values_selected
        summed_trends_selected : list of lists of len `n_per_temp` of lists
            trend values used for summation to get statistic_values_selected
        summed_vars_selected : list of lists of len `n_per_temp` of lists
            variances used for summation to get statistic_values_selected

        :param n_per_temp: How many best scores per template to save. The default is 10.
        :type n_per_temp: int
        :param ngap_moving_var:
        :type ngap_moving_var:
        :param path:
        :type path:
        :param save_to_file:
        :type save_to_file:
        :param save_inverted:
        :type save_inverted:
        :return:
        :rtype:
        """
        # Warning: saved things are score-corrected. In order to get non-score-corrected values for amplitude estimations etc,
        # use dedicated methods
        if path is None:
            path = self.output_location
        period_indices_selected = []
        periods_selected = []
        phase_indices_selected = []  # relative phase indices wrt the local beginning of data (quarters may be cut)
        phases_selected = []  # phase wrt the beginning of Kepler time
        statistic_values_selected = []
        summed_transit_times_selected = []
        summed_stats_raw_selected = []
        summed_trends_selected = []
        summed_vars_selected = []
        if len(self.statistics_periodic_interleaved) > 0:
            for ti in range(len(self.statistics_periodic_interleaved[0])):  # over templates
                phase_ind_min_per = []
                stats_min_per = []
                for pi in range(len(self.statistics_periodic_interleaved)):  # over periods
                    ph_i_min = np.nanargmin(self.statistics_periodic_interleaved[pi][ti])
                    phase_ind_min_per.append(ph_i_min)
                    stats_min_per.append(self.statistics_periodic_interleaved[pi][ti][ph_i_min])
                stat_best_per_inds = np.argsort(stats_min_per)[:n_per_temp]
                per_sel = self.periods[stat_best_per_inds]
                phase_ind_sel = np.array(phase_ind_min_per)[stat_best_per_inds]
                phases_sel = np.array(self._phases_interleaved)[phase_ind_sel]  # relative phases wrt the beginning of this data
                phases_abs = (np.array(self.time_interleaved)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time

                summed_transit_times_per = []
                summed_stats_raw_per = []
                summed_trends_per = []
                summed_vars_per = []
                for j in range(len(stat_best_per_inds)):
                    pi = stat_best_per_inds[j]
                    period = per_sel[j]
                    phase = phases_sel[j]
                    expected_transit_times = [self.tmin + phase + i * period  # i am not sure tmin does not miss some nan in the beginning
                                              for i in range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
                    summed_transit_points = [np.nanargmin(np.abs(self.time_interleaved - expected_transit_times[i])) for
                                             i in range(len(expected_transit_times))]
                    summed_transit_times_per.append(self.time_interleaved[summed_transit_points])
                    summed_stats_raw_per.append(self.statistics_interleaved[ti][summed_transit_points])
                    summed_trends_per.append(self.trend_orig_flux_interleaved[summed_transit_points])
                    summed_vars_per.append(self.stat_vars_interleaved[ti][summed_transit_points])
                period_indices_selected.append(stat_best_per_inds)
                periods_selected.append(per_sel)
                phase_indices_selected.append(phase_ind_sel)
                statistic_values_selected.append(np.array(stats_min_per)[stat_best_per_inds])
                phases_selected.append(phases_abs)
                summed_transit_times_selected.append(summed_transit_times_per)
                summed_stats_raw_selected.append(summed_stats_raw_per)
                summed_trends_selected.append(summed_trends_per)
                summed_vars_selected.append(summed_vars_per)
        if len(self.statistics_periodic_not_interleaved) > 0:
            for ti in range(len(self.statistics_periodic_not_interleaved[0])):  # over templates
                phase_ind_min_per = []
                stats_min_per = []
                for pi in range(len(self.statistics_periodic_not_interleaved)):  # over periods
                    ph_i_min = np.nanargmin(self.statistics_periodic_not_interleaved[pi][ti])
                    phase_ind_min_per.append(ph_i_min)
                    stats_min_per.append(self.statistics_periodic_not_interleaved[pi][ti][ph_i_min])
                stat_best_per_inds = np.argsort(stats_min_per)[:n_per_temp]
                per_sel = self.periods[stat_best_per_inds]
                phase_ind_sel = np.array(phase_ind_min_per)[stat_best_per_inds]
                phases_sel = np.array(self._phases_not_interleaved)[phase_ind_sel]  # relative phases wrt the beginning of this data
                phases_abs = (np.array(self.time_not_interleaved)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time

                summed_transit_times_per = []
                summed_stats_raw_per = []
                summed_trends_per = []
                summed_vars_per = []
                for j in range(len(stat_best_per_inds)):
                    pi = stat_best_per_inds[j]
                    period = per_sel[j]
                    phase = phases_sel[j]
                    expected_transit_times = [self.tmin + phase + i * period
                                              for i in range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
                    summed_transit_points = [np.nanargmin(np.abs(self.time_not_interleaved - expected_transit_times[i]))
                                             for i in range(len(expected_transit_times))]
                    summed_transit_times_per.append(self.time_not_interleaved[summed_transit_points])
                    summed_stats_raw_per.append(self.statistics_not_interleaved[ti][summed_transit_points])
                    summed_trends_per.append(self.trend_orig_flux_not_interleaved[summed_transit_points])
                    summed_vars_per.append(self.stat_vars_not_interleaved[ti][summed_transit_points])
                period_indices_selected.append(stat_best_per_inds)
                periods_selected.append(per_sel)
                phase_indices_selected.append(phase_ind_sel)
                statistic_values_selected.append(np.array(stats_min_per)[stat_best_per_inds])
                phases_selected.append(phases_abs)
                summed_transit_times_selected.append(summed_transit_times_per)
                summed_stats_raw_selected.append(summed_stats_raw_per)
                summed_trends_selected.append(summed_trends_per)
                summed_vars_selected.append(summed_vars_per)

        times_quarters = [(np.nanmin(self.times[q]), np.nanmax(self.times[q])) for q in
                          self.quarters]  # attention: there should be no zero-padding
        self.best_periodic_scores_data = [[] for _ in range(Search_results_length)]
        self.best_periodic_scores_data[Search_results_index_per] = periods_selected
        self.best_periodic_scores_data[Search_results_index_stat] = statistic_values_selected
        self.best_periodic_scores_data[Search_results_index_sum_tr_times] = summed_transit_times_selected
        self.best_periodic_scores_data[Search_results_index_sum_stats] = summed_stats_raw_selected
        self.best_periodic_scores_data[Search_results_index_sum_trends] = summed_trends_selected
        self.best_periodic_scores_data[Search_results_index_sum_vars] = summed_vars_selected
        self.best_periodic_scores_data[Search_results_index_shifts] = [self.time_slide_shifts, times_quarters]
        self.best_periodic_scores_data[Search_results_index_per_list] = self.periods

        # TODO: we are not using this saving (because it is producing too many files), but it is hardcoded:
        if save_to_file:
            np.save(os.path.join(path, str(self.target_id) + '_best_periodic_scores_P=' + str(self.Pmin) + '_' + str(
                self.Pmax)), np.array(self.best_periodic_scores_data, dtype=object), allow_pickle=True)

        if save_inverted:
            period_indices_selected = []
            periods_selected = []
            phase_indices_selected = []
            phases_selected = []
            statistic_values_selected = []
            summed_transit_times_selected = []
            summed_stats_raw_selected = []
            summed_trends_selected = []
            summed_vars_selected = []
            if len(self.statistics_periodic_interleaved) > 0:
                for ti in range(len(self.statistics_periodic_interleaved[0])):  # over templates
                    phase_ind_max_per = []
                    stats_max_per = []
                    for pi in range(len(self.statistics_periodic_interleaved)):  # over periods
                        ph_i_max = np.nanargmax(self.statistics_periodic_interleaved[pi][ti])
                        phase_ind_max_per.append(ph_i_max)
                        stats_max_per.append(self.statistics_periodic_interleaved[pi][ti][ph_i_max])
                    stat_best_per_inds = (np.argsort(stats_max_per)[::-1])[:n_per_temp]
                    per_sel = self.periods[stat_best_per_inds]
                    phase_ind_sel = np.array(phase_ind_max_per)[stat_best_per_inds]
                    phases_sel = np.array(self._phases_interleaved)[phase_ind_sel]
                    phases_abs = (np.array(self.time_interleaved)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time

                    summed_transit_times_per = []
                    summed_stats_raw_per = []
                    summed_trends_per = []
                    summed_vars_per = []
                    for j in range(len(stat_best_per_inds)):
                        pi = stat_best_per_inds[j]
                        period = per_sel[j]
                        phase = phases_sel[j]
                        expected_transit_times = [self.tmin + phase + i * period
                                                  for i in
                                                  range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
                        summed_transit_points = [np.nanargmin(np.abs(self.time_interleaved - expected_transit_times[i]))
                                                 for i in range(len(expected_transit_times))]
                        summed_transit_times_per.append(self.time_interleaved[summed_transit_points])
                        summed_stats_raw_per.append(self.statistics_interleaved[ti][summed_transit_points])
                        summed_trends_per.append(self.trend_orig_flux_interleaved[summed_transit_points])
                        summed_vars_per.append(self.stat_vars_interleaved[ti][summed_transit_points])
                    period_indices_selected.append(stat_best_per_inds)
                    periods_selected.append(per_sel)
                    phase_indices_selected.append(phase_ind_sel)
                    statistic_values_selected.append(np.array(stats_max_per)[stat_best_per_inds])
                    phases_selected.append(phases_abs)
                    summed_transit_times_selected.append(summed_transit_times_per)
                    summed_stats_raw_selected.append(summed_stats_raw_per)
                    summed_trends_selected.append(summed_trends_per)
                    summed_vars_selected.append(summed_vars_per)
            if len(self.statistics_periodic_not_interleaved) > 0:
                for ti in range(len(self.statistics_periodic_not_interleaved[0])):  # over templates
                    phase_ind_max_per = []
                    stats_max_per = []
                    for pi in range(len(self.statistics_periodic_not_interleaved)):  # over periods
                        ph_i_max = np.nanargmax(self.statistics_periodic_not_interleaved[pi][ti])
                        phase_ind_max_per.append(ph_i_max)
                        stats_max_per.append(self.statistics_periodic_not_interleaved[pi][ti][ph_i_max])
                    stat_best_per_inds = (np.argsort(stats_max_per)[::-1])[:n_per_temp]
                    per_sel = self.periods[stat_best_per_inds]
                    phase_ind_sel = np.array(phase_ind_max_per)[stat_best_per_inds]
                    phases_sel = np.array(self._phases_not_interleaved)[phase_ind_sel]
                    phases_abs = (np.array(self.time_not_interleaved)[phase_ind_sel]) % per_sel  # phases wrt the beginning of Kepler time

                    summed_transit_times_per = []
                    summed_stats_raw_per = []
                    summed_trends_per = []
                    summed_vars_per = []
                    for j in range(len(stat_best_per_inds)):
                        pi = stat_best_per_inds[j]
                        period = per_sel[j]
                        phase = phases_sel[j]
                        expected_transit_times = [self.tmin + phase + i * period
                                                  for i in
                                                  range(int(np.ceil((self.tmax - self.tmin - phase) / period)))]
                        summed_transit_points = [
                            np.nanargmin(np.abs(self.time_not_interleaved - expected_transit_times[i])) for i in
                            range(len(expected_transit_times))]
                        summed_transit_times_per.append(self.time_not_interleaved[summed_transit_points])
                        summed_stats_raw_per.append(self.statistics_not_interleaved[ti][summed_transit_points])
                        summed_trends_per.append(self.trend_orig_flux_not_interleaved[summed_transit_points])
                        summed_vars_per.append(self.stat_vars_not_interleaved[ti][summed_transit_points])
                    period_indices_selected.append(stat_best_per_inds)
                    periods_selected.append(per_sel)
                    phase_indices_selected.append(phase_ind_sel)
                    statistic_values_selected.append(np.array(stats_max_per)[stat_best_per_inds])
                    phases_selected.append(phases_abs)
                    summed_transit_times_selected.append(summed_transit_times_per)
                    summed_stats_raw_selected.append(summed_stats_raw_per)
                    summed_trends_selected.append(summed_trends_per)
                    summed_vars_selected.append(summed_vars_per)

            times_quarters = [(np.nanmin(self.times[q]), np.nanmax(self.times[q])) for q in
                              self.quarters]  # attention: there should be no zero-padding

            self.best_inverted_periodic_scores_data = [[] for _ in range(Search_results_length)]
            self.best_inverted_periodic_scores_data[Search_results_index_per] = periods_selected
            self.best_inverted_periodic_scores_data[Search_results_index_stat] = statistic_values_selected
            self.best_inverted_periodic_scores_data[Search_results_index_sum_tr_times] = summed_transit_times_selected
            self.best_inverted_periodic_scores_data[Search_results_index_sum_stats] = summed_stats_raw_selected
            self.best_inverted_periodic_scores_data[Search_results_index_sum_trends] = summed_trends_selected
            self.best_inverted_periodic_scores_data[Search_results_index_sum_vars] = summed_vars_selected
            self.best_inverted_periodic_scores_data[Search_results_index_shifts] = [self.time_slide_shifts, times_quarters]
            self.best_inverted_periodic_scores_data[Search_results_index_per_list] = self.periods
            if save_to_file:
                np.save(os.path.join(path, str(self.target_id) + '_best_inverted_periodic_scores_P=' + str(
                    self.Pmin) + '_' + str(self.Pmax)), np.array(self.best_periodic_scores_data, dtype=object),
                        allow_pickle=True)

    def get_best_trigger(self):
        self.best_stat = 0
        best_candidates = self.best_periodic_scores_data
        argm = np.argmin(best_candidates[Search_results_index_stat])
        ti, i = argm // len(best_candidates[Search_results_index_stat][0]), argm % len(best_candidates[Search_results_index_stat][0])
        # the structure used to be (25,10), where 10 was just 10 best per each template
        smin = best_candidates[Search_results_index_stat][ti][i]  # minimal statistic
        if smin < self.best_stat:
            self.best_stat = smin
            self.best_per = best_candidates[Search_results_index_per][ti][i]
            self.best_temp = ti
            self.best_first_transit_time = best_candidates[Search_results_index_sum_tr_times][ti][i][0]
            self.summed_transit_times = best_candidates[Search_results_index_sum_tr_times][ti][i]
            self.summed_raw_stats = np.array(best_candidates[Search_results_index_sum_stats][ti][i])
            self.summed_trends = np.array(best_candidates[Search_results_index_sum_trends][ti][i])
            self.summed_vars = np.array(best_candidates[Search_results_index_sum_vars][ti][i])
            self.summed_vars[self.summed_vars == 0] = np.nan

        self.best_stat_inv = 0
        best_candidates = self.best_inverted_periodic_scores_data
        argm = np.argmax(best_candidates[Search_results_index_stat])  # we took inverted best scores collection, they are positive
        ti, i = argm // len(best_candidates[Search_results_index_stat][0]), argm % len(best_candidates[Search_results_index_stat][0])
        # the structure used to be (25,10), where 10 was just 10 best per each template
        smax = best_candidates[Search_results_index_stat][ti][i]  # maximal statistic
        if smax > self.best_stat_inv:
            self.best_stat_inv = smax
            self.best_per_inv = best_candidates[Search_results_index_per][ti][i]
            self.best_temp_inv = ti
            self.best_first_transit_time_inv = best_candidates[Search_results_index_sum_tr_times][ti][i][0]
            self.summed_transit_times_inv = best_candidates[Search_results_index_sum_tr_times][ti][i]
            self.summed_raw_stats_inv = np.array(best_candidates[Search_results_index_sum_stats][ti][i])
            self.summed_trends_inv = np.array(best_candidates[Search_results_index_sum_trends][ti][i])
            self.summed_vars_inv = np.array(best_candidates[Search_results_index_sum_vars][ti][i])
            self.summed_vars_inv[self.summed_vars_inv == 0] = np.nan

    def get_values_at_transit_times(self, in_non_corrected=True, transit_times=None, template_ind=None):
        if in_non_corrected:
            assert self.stats_raw_no_veto_no_score_cor is not None
            stats_raw = self.stats_raw_no_veto_no_score_cor
        else:
            stats_raw = self.statistics_raw_cor
        if transit_times is None:
            transit_times = self.summed_transit_times
        if template_ind is None:
            template_ind = self.best_temp

        best_temps_sh = np.arange(self.indices_of_templates[template_ind][0], self.indices_of_templates[template_ind][-1])
        best_raw_stats = []
        best_stat_denoms = []
        best_trends = []
        best_templ_depths = []
        for i, tt in enumerate(transit_times):
            best_stat = 0.
            best_time_ind = None
            best_temp_ind = None
            best_quarter = None
            for q in self.quarters:
                if self.times[q][~np.isnan(self.times[q])][0] <= tt <= self.times[q][~np.isnan(self.times[q])][-1]:
                    ind_tr = np.nanargmin(np.abs(self.times[q] - tt))
                    for k in best_temps_sh:
                        ti0 = max(0, ind_tr - 1)
                        ti1 = min(ind_tr + 1, len(stats_raw[k][q]))
                        rel_shift = np.nanargmin(stats_raw[k][q][ti0:ti1])
                        time_index = ti0 + rel_shift
                        stat_here = stats_raw[k][q][time_index]
                        if stat_here < best_stat:
                            best_stat = stat_here
                            best_time_ind = time_index
                            best_temp_ind = k
                            best_quarter = q
            if best_time_ind is not None:  # it can be None if the transit is between the quarters or something like that
                best_raw_stats.append(stats_raw[best_temp_ind][best_quarter][best_time_ind])
                best_stat_denoms.append(self.stat_denoms[best_temp_ind][best_quarter])
                best_trends.append(self.trends_orig_fluxes[best_quarter][best_time_ind])
                best_templ_depths.append(np.max(np.abs(self.templates[best_temp_ind])))
            else:
                best_raw_stats.append(np.nan)
                best_stat_denoms.append(np.nan)
                best_trends.append(np.nan)
                best_templ_depths.append(np.nan)

        self.best_raw_stats = np.array(best_raw_stats)
        self.best_stat_denoms = np.array(best_stat_denoms)
        self.best_trends = np.array(best_trends)
        self.best_templ_depths = np.array(best_templ_depths)

    def measure_depths(self):
        if self.best_raw_stats is None:
            if self.summed_transit_times is None:
                self.get_best_trigger()
            self.get_values_at_transit_times(in_non_corrected=True,
                                             transit_times=self.summed_transit_times, template_ind=self.best_temp)
        self.best_ampl, self.best_ppm, self.best_ampls, self.best_ampls_stds, self.best_normalized_ampls, \
            self.best_normalized_ampls_stds, self.best_ppms, self.best_ppms_stds = \
            af.estimate_ampls_ppms_from_stats(summed_raw_stats=self.best_raw_stats, summed_trends=self.best_trends,
                                              summed_stat_denoms=self.best_stat_denoms,
                                              templ_max=self.best_templ_depths, trends_stds=None)


def sum_periods_fun(statistics_by_trends, trends_sq_by_vars, dt, tmax, tmin, periods):
    """
    Sums the statistics with all the possible periods and initial phases. Accounts for
    subbin template shifts, accounts for masks.
    Does not return

    Parameters
    ----------
    statistics_by_trends : numpy.array 2d of float
    trends_sq_by_vars : numpy.array 2d of float
    dt : float
        Time step of equally sampled time axis.
    tmax : float
        Maximal time in the time axis.
    tmin : float
        Minimal time in the time axis.

    Returns
    -------
    statistics_periodic : numpy.array 3d
        Sum of the statistics for the entire transit chain. For all the templates
        with shifts, for all the periods and starting phases.
    statistics_periodic_counts : numpy.array 3d
        Number of transits summed in each entry. For all the templates
        with shifts, for all the periods and starting phases.
    phases : numpy.array 3d
        Starting phases corresponding to `statistics_periodic`. For all the
        templates with shifts, for all the periods.
    lengths_filled : numpy.array 2d
        Number of phases corresponding to `statistics_periodic`. For all the
        templates with shifts, for all the periods. For different periods,
        the length of the array of all the possible starting phases has different
        length. These are the lengths. The remaining entries in `statistics_periodic`
        and `statistics_periodic_counts` are zero.
    relative_amplitudes : numpy.array 4d
        SNRs of individual transits in absolute units divided by the trend level
        in this place. For all the templates, with shifts, for all the periods
        and starting phases.

    """
    phases_size = int(np.ceil((periods[-1]) / dt))
    statistics_periodic = np.zeros((len(periods), len(statistics_by_trends), phases_size))
    denomsq = np.zeros((len(periods), len(statistics_by_trends), phases_size))
    for P_ind in range(len(periods)):
        p = periods[P_ind]
        n_transits = int(np.ceil((tmax - tmin) / p))
        phaselen = int(p / dt)
        for n in range(n_transits):
            time_ind = int(np.round((p * n / dt)))
            statistics_periodic[P_ind, :, :phaselen] += statistics_by_trends[:, time_ind:time_ind + phaselen]
            denomsq[P_ind, :, :phaselen] += trends_sq_by_vars[:, time_ind:time_ind + phaselen]
    return statistics_periodic, denomsq


def variables_from_file(filename):
    """
    Opens the file with a list of variable names and values. Recognizes the data
    types of the values. Creates variables with corresponding names and assigns
    them the values. Format of the file (example):
        dummy_coefficient = 1.
        curse='hann'

        best_number=13
        is_life_great = True

    Parameters
    ----------
    filename : str
        Name (full path) of the file from which the variables should be read.
    Returns
    -------
    None.

    Creates
    -------
    The variables and assigns values
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    def isint(string):
        assert type(string) == str, 'Input must be string!'
        if string.isdecimal():
            return True
        elif string[0] == '-' and (string[1:]).isdecimal():
            return True
        else:
            return False

    def issimplenum(string):
        if isint(string):
            return True
        elif string.count('.') == 1 and isint(string.replace('.', '')):
            return True
        else:
            return False

    def isfloat(string):
        string = string.lower()
        if issimplenum(string):
            return True
        elif ((string.count('e') == 1 and issimplenum(string.replace('e', '')))
              or (string.count('e-') == 1 and issimplenum(string.replace('e-', '')))
              or (string.count('e+') == 1 and issimplenum(string.replace('e+', '')))):
            return True
        else:
            return False

    for line in lines:
        if '=' in line:
            text = line.split('#')[0]  # remove comments
            text = text.split('\\')[0]  # remove endlines
            name, val = text.split('=')[0:2]
            name = name.strip()
            val = val.strip()
            if (name != '') and (val != ''):
                if val == 'None':
                    val = None
                elif val == 'True':
                    val = True
                elif val == 'False':
                    val = False
                elif isint(val):
                    val = int(val)
                elif isfloat(val):
                    val = float(val)
                else:
                    assert (((val[0] == '"') and (val[-1] == '"')) or ((val[0] == "'") and (val[-1] == "'"))), \
                             'Undefined type of value: not int, bool, float, str or None! ( line "' + str(line) + '" )'
                    val = val[1:-1]

                assert ((65 <= ord(name[0]) <= 90) or (97 <= ord(name[0]) <= 122) or name[
                    0] == '_'), 'Incorrect variable name specified!'
                if len(name) > 1:
                    for e in name[1:]:
                        assert ((48 <= ord(e) <= 57) or (65 <= ord(e) <= 90) or (
                                97 <= ord(e) <= 122) or e == '_'), 'Incorrect variable name specified!'
                globals()[name] = val

