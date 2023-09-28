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
# from scipy.optimize import minimize
# from scipy.stats import median_abs_deviation as mad
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
# TODO: write documentation
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
    def __init__(self, target_id=None, times=None, fluxes=None, get_data=True, data_location=Default_data_location,
                 output_location=Default_output_location,
                 remove_zeroth_quarter=True, remove_quarter_17=True):  # , load_data_from_file=False
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

        self.times = copy.deepcopy(self.orig_times)
        self.time_slide_shifts = None  # np.zeros(len(self.orig_times))
        self.fluxes = copy.deepcopy(self.orig_fluxes)
        self.clean_fluxes = copy.deepcopy(self.orig_fluxes)
        self.quarters = range(len(self.fluxes))

        self.masks = [~np.isnan(self.fluxes[i]) for i in
                      self.quarters]  # mask things which need to be removed from data
        self.clean_masks = copy.deepcopy(self.masks)  # mask all the outliers, needed for PSD and filters, might kill transits
        self.psd_masks = copy.deepcopy(self.masks)  # clean_masks with maybe only big holes to have enough slices for PSD
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
        self.is_prepared_for_psd = False
        self.is_template_whitened = False
        self.was_notched = None
        self.nperseg_psd = None
        tmax = np.max(np.array([len(t) for t in self.orig_times]))  # maximum length of the time
        # the closest (but bigger) power of 2 to the length        Tmax += tm.L_ban
        self.support = 2 ** int(np.ceil(np.log2(tmax)))
        self.fft_support = self.support // 2 + 1  # use_rfft

        self.whitened_fluxes_td = np.zeros((len(self.fluxes), self.support))
        self.whitened_templates_td = None

        self.inpaint_fluxes_fast = numba.njit(ut.inpaint_fluxes_fun)  # it is important to use it because if shrinks ram from 4G to 1.5G
        # initialization of the numba function on some small data in order  not to
        # spend time on big data:
        _ = self.inpaint_fluxes_fast(fluxes=np.array(np.random.normal(0, 1, (2, 32)), dtype=Dtype_r),
                                     masks=np.random.choice([0, 1], (2, 32)).astype(bool),
                                     wf_tds=np.array([np.concatenate((np.zeros(32 // 2 - 2), [-0.5, 1, 1, -0.5], np.zeros(32 // 2 - 2)))
                                                      for _ in range(2)], dtype=Dtype_r))

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
        self.freqs = None
        self.psdfreqs = None
        self.npsds = None
        self.psds = None
        self.orig_psds = None
        self.wf_fds_meas = None
        self.wf_tds = None
        self.wf_fds = None
        self.wf_fds_unwindowed = None
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

        self.tb = None  # template_bank
        self.whitened_templates_td = None
        self.whitened_templates_fd = None
        self.indices_of_templates = None
        self.templates_of_indices = None

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
        self.big_hole_masks_full = None  # used only in the statistic shuffling
        self.veto_bad_trigger_inds = None

        self.inds_pieces_scramble = None
        self.tot_hole_len_scramble = None
        self.inds_pieces_in_scrambled = None

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
                       nsig_notch=Nsig_notch,
                       nsig_outliers_w=Nsig_outliers_w, nsig_outliers_w_pos=Nsig_outliers_w_pos,
                       nsig_outliers_w_neg=Nsig_outliers_w_neg,
                       max_small_hole=Max_small_hole,
                       detr_length_outliers=Detr_length_outliers, detr_method_outliers=Detr_method_outliers,
                       detr_endmethod_outliers=Detr_endmethod_outliers,
                       detr_ends_outliers=Detr_ends_outliers, detr_endlen_outliers=Detr_endlen_outliers,
                       tukey_steps_len=Tukey_steps_len_psd,
                       conditional_zero_zeroth_freq=Conditional_zero_zeroth_freq_wf,
                       always_zero_zeroth_freq=Always_zero_zeroth_freq_wf):
        """
        Preparing data for detection. Subtract mean, mask outliers, measure psds,
        padding by zeros, notching, inpainting, whitening.

        See the description of the non-described parameters in the docstrings of the corresponding functions.
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
        :param tukey_steps_len:
        :type tukey_steps_len:
        :param conditional_zero_zeroth_freq:
        :type conditional_zero_zeroth_freq:
        :param always_zero_zeroth_freq:
        :type always_zero_zeroth_freq:
        :return:
        :rtype:

        Mean values of the flux are subtracted to have signal centered around zero. 
        Means are saved in self.means:
        """
        self.nperseg_psd = nperseg_psd
        self.subtract_means(save_means=True, make_trends=True)
        self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                nsig_negative=nsig_outliers_0_neg,
                                nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                detr_method_outliers=detr_method_outliers, detr_endmethod_outliers=detr_endmethod_outliers,
                                detr_ends_outliers=detr_ends_outliers, detr_endlen_outliers=detr_endlen_outliers,
                                in_whitened_fluxes=False, modify_clean_masks=True)
        self.subtract_means(save_means=True, make_trends=True, masks=self.masks)
        self.pad_by_zeros()
        self.make_big_hole_masks(max_small_hole=max_small_hole, exclude_individual_points=True)
        self.interpolate_fluxes()
        self.subtract_low_trends(nperseg=self.nperseg_psd)
        self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                nsig_negative=nsig_outliers_0_neg,
                                nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                detr_method_outliers=detr_method_outliers, detr_endmethod_outliers=detr_endmethod_outliers,
                                detr_ends_outliers=detr_ends_outliers, detr_endlen_outliers=detr_endlen_outliers,
                                in_whitened_fluxes=False, modify_clean_masks=True)
        self.interpolate_fluxes(interpolate_clean_fluxes=True) # interpolate_clean_fluxes=False)  # need to do this again after piecewise detrending
        self.prepare_for_psd(nperseg=self.nperseg_psd, noverlap=noverlap_psd, npsdsmin=npsdsmin)
        self.measure_psds(nperseg=self.nperseg_psd, noverlap=noverlap_psd, window=window_psd,
                          subtract_mean=subtract_mean_in_psd, smoothen_psds=smoothen_psds,
                          tukey_steps_len=tukey_steps_len, conditional_zero_zeroth_freq=conditional_zero_zeroth_freq,
                          always_zero_zeroth_freq=always_zero_zeroth_freq)
        self.whiten_fluxes(whiten_clean_fluxes=True)
        self.mask_flux_outliers(nsig=nsig_outliers_w, nsig_positive=nsig_outliers_w_pos,
                                nsig_negative=nsig_outliers_w_neg,
                                nsig_clean=nsig_outliers_psd, detr_length_outliers=detr_length_outliers,
                                detr_method_outliers=detr_method_outliers, detr_endmethod_outliers=detr_endmethod_outliers,
                                detr_ends_outliers=detr_ends_outliers, detr_endlen_outliers=detr_endlen_outliers,
                                in_whitened_fluxes=True, modify_clean_masks=True)
        self.interpolate_fluxes()
        self.whiten_fluxes(whiten_clean_fluxes=True)
        self.notch_fluxes(nsig=nsig_notch)
        #self.whiten_fluxes(whiten_clean_fluxes=True)
        self.check_low_freq_whitening_residuals(resid_len_permission=3)
        self.fftshift_wf_fds()
        self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos,
                                nsig_negative=nsig_outliers_0_neg,
                                nsig_clean=nsig_outliers_psd,
                                detr_length_outliers=detr_length_outliers, detr_method_outliers=detr_method_outliers,
                                detr_endmethod_outliers=detr_endmethod_outliers,
                                detr_ends_outliers=detr_ends_outliers, detr_endlen_outliers=detr_endlen_outliers,
                                in_whitened_fluxes=False, modify_clean_masks=True)
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

    def save_conditioned_data(self, file=None, path=None, filename_postfix='conditioned_data'):
        """
        Saves the essential conditioning results as .npy file that can be read
        with `load_conditioned_data` function:
            self.support
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
        # conditioned_data[0][11] = self.tb.num_templates
        # conditioned_data[0][12] = len(self.whitened_templates_td
        # for tn in range(len(self.whitened_templates_td):
        #     conditioned_data[0][13 + tn] = self.tb.indices_of_templates[tn][0]
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
            conditioned_data[6][ind_last:ind_last + len(self.trends_orig_fluxes[q])] = self.trends_orig_fluxes[q]
            for k in range(len(self.whitened_templates_td)):
                conditioned_data[k + 7][ind_last:ind_last + len(self.whitened_templates_td[k][q])] = self.whitened_templates_td[k][q]
            ind_last += self.support
        if self.target_id is not None:
            file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                                filename_postfix=filename_postfix, file_format='npy')
            np.save(file_path, np.array(conditioned_data, dtype=object), allow_pickle=True)
        else:
            np.save(os.path.join(path, filename_postfix), np.array(conditioned_data, dtype=object), allow_pickle=True)

    def load_conditioned_data(self, file=None, filename_postfix=Default_conditioned_data_filename_postfix,
                              path=None, no_zero_padding=False):
        """
        Only works for data created by `save_conditioned_data`.

        Sets:
        self.support
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

        :param file: path to the .npy file where the data is stored.
        :type file: str
        :param filename_postfix:
        :type filename_postfix:
        :param no_zero_padding:
        :type no_zero_padding:
        :return:
        :rtype:
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
        # self.tb.num_templates = int(conditioned_data[0][11])
        # self.tb.num_templates_shifts = int(conditioned_data[0][12])
        # self.tb.indices_of_templates = []
        # for tn in range(self.tb.num_templates - 1):
        #    self.tb.indices_of_templates.append([int(conditioned_data[0][13 + tn]), int(conditioned_data[0][14 + tn])])
        # self.tb.indices_of_templates.append([int(conditioned_data[0][self.tb.num_templates - 1]), int(self.tb.num_templates_shifts)])

        self.zero_padding_ends = [[0, 0] for _ in self.quarters]
        self.times = []
        self.fluxes = []
        self.whitened_fluxes_td = []
        self.whitened_fluxes_fd = []
        self.wf_tds = []
        self.wf_fds = []
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
            self.wf_tds.append(wftd1)
            self.wf_fds.append(np.fft.rfft(np.fft.ifftshift(wftd1)))
            self.big_hole_masks.append((conditioned_data[5][ind_last: ind_last + lenflux]).astype(bool))
            self.trends_orig_fluxes.append((conditioned_data[6][ind_last: ind_last + len_orig]).astype(float))
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

    def save_connected_stats(self, filename_postfix=Default_connected_statistic_data_filename_postfix, path=None):
        if path is None:
            path = self.output_location
        num_temp_shifts = len(self.statistics_full)
        len_one = len(self.time_full)
        connected_data = np.zeros((3 + num_temp_shifts * 3, len_one + 1), dtype=Dtype_r)

        # connected_data[0, 0] = num_temp_shifts
        connected_data[0, 1:] = self.time_full
        connected_data[1, 1:] = self.trend_orig_flux_full
        connected_data[2, 1:] = self.big_hole_masks_full.astype(Dtype_r)
        line_num_0 = 3
        for kk in range(num_temp_shifts):
            connected_data[line_num_0 + kk, 0] = self.templates_of_indices[kk]
            connected_data[line_num_0 + kk, 1:] = self.statistics_full[kk]
            connected_data[line_num_0 + 2 * num_temp_shifts + kk, 1:] = self.stat_masks_full[kk].astype(float)
            connected_data[line_num_0 + 2 * num_temp_shifts + kk, 0] = self.templates_of_indices[kk]
            connected_data[line_num_0 + num_temp_shifts + kk, 1:] = self.stat_vars_full[kk]
            connected_data[line_num_0 + num_temp_shifts + kk, 0] = self.templates_of_indices[kk]

        # vars_quarters = [[np.array(np.ones(len(self.statistics_raw_cor[k][q])) * self.stat_vars[k][q], dtype=Dtype_r)
        #                   for q in self.quarters] for k in range(self.tb.num_templates_shifts)]
        file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                            filename_postfix=filename_postfix, file_format='npy')
        np.save(file_path, connected_data, allow_pickle=True)

    def save_no_veto_no_score_cor_stats(self, filename_postfix=Default_no_veto_no_score_cor_stats_filename_postfix):
        file_path = af.get_saving_file_path(output_location=self.output_location, target_id=self.target_id,
                                            filename_postfix=filename_postfix, file_format='npy')
        np.save(file_path, np.array(self.stats_raw_no_veto_no_score_cor, dtype=object), allow_pickle=True)

    def load_connected_stats(self, file_path=None, filename_postfix=Default_connected_statistic_data_filename_postfix, path=None):
        if file_path is None:
            if path is None:
                path = self.output_location
            file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                                filename_postfix=filename_postfix, file_format='npy')
        connected_data = np.load(file_path, allow_pickle=True)

        num_temp_shifts = (len(connected_data) - 3) // 3  # int(connected_data[0, 0])
        self.time_full = connected_data[0, 1:]
        self.trend_orig_flux_full = connected_data[1, 1:]
        self.big_hole_masks_full = (connected_data[2, 1:]).astype(bool)
        templates_of_indices = []
        line_num_0 = 3

        self.statistics_full = np.zeros((num_temp_shifts, len(self.time_full)), dtype=Dtype_r)
        self.stat_vars_full = np.zeros((num_temp_shifts, len(self.time_full)), dtype=Dtype_r)
        self.stat_masks_full = np.zeros((num_temp_shifts, len(self.time_full)), dtype=bool)

        for kk in range(num_temp_shifts):
            templates_of_indices.append(connected_data[line_num_0 + kk, 0])
            self.statistics_full[kk] = connected_data[line_num_0 + kk, 1:]
            self.stat_masks_full[kk] = (connected_data[line_num_0 + 2 * num_temp_shifts + kk, 1:]).astype(bool)
            self.stat_vars_full[kk] = connected_data[line_num_0 + num_temp_shifts + kk, 1:]

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
        # vars_quarters = [[np.array(np.ones(len(self.statistics_raw_cor[k][q])) * self.stat_vars[k][q], dtype=Dtype_r)
        #                   for q in self.quarters] for k in range(self.tb.num_templates_shifts)]

    def select_connected_stats_indices(self, indices):
        #inds_templs_to_use = np.arange(tm.Indices_of_templates[self.best_temp][0], tm.Indices_of_templates[self.best_temp][-1])
        indices_with_shifts = [i for ind in indices
                               for i in range(tm.Indices_of_templates[ind][0], tm.Indices_of_templates[ind][1])]
        self.indices_of_templates = np.array(self.indices_of_templates)[indices]
        self.statistics_full = self.statistics_full[indices_with_shifts]
        self.stat_vars_full = self.stat_vars_full[indices_with_shifts]
        self.stat_masks_full = self.stat_masks_full[indices_with_shifts]
        indices_of_templates = []
        index_last = 0
        for i in range(len(self.indices_of_templates)):
            indices_current = np.array(self.indices_of_templates[i])
            indices_current = indices_current - indices_current[0] + index_last
            index_last = indices_current[1]
            indices_of_templates.append(list(indices_current))
        self.indices_of_templates = np.array(indices_of_templates)
        self.templates_of_indices = np.array([k for k in range(len(self.indices_of_templates))
                                              for _ in range(self.indices_of_templates[k][0], self.indices_of_templates[k][1])])

    def condition_templates(self, templates=None, shifts=None, template_index='all'):
        """
        Initializes and conditions the template bank. Most of the temlate properties and methods are inside the `TemplateBank` class,
        but notchig, whitening and similar are performed here.
        :param templates: Array of transit models used for detecton, usually produced with Batman pachageCentered at the center
        :type templates:
        :param shifts:
        :type shifts:
        :param template_index:
        :type template_index:
        :return:
        :rtype:
        """
        """
        The documentaton is not written yet. Try consulting descriptions of all the methods
        :param template_index:
        :type template_index:
        :return:
        :rtype:
        """
        self.initialize_templates(templates=templates, shifts=shifts, template_index=template_index)
        self.notch_templates()  # why notch is before roll, is written inside the method
        self.tb.roll_templates()
        self.whiten_templates()

    def detect_single_transits(self, nsig_outliers_stat=Nsig_outliers_stat,
                               apply_veto=True, apply_score_correction=Apply_score_correction,
                               apply_veto_only_to_neg=False,
                               margin_snr_veto=Margin_snr_veto,
                               fn_step_veto=Fn_step_veto, fn_fit_veto=Fn_fit_veto,
                               min_len_step_checking_veto=Min_len_step_checking_veto,
                               max_len_step_checking_veto=Max_len_step_checking_veto,
                               n_parts_veto_snr=N_parts_veto_snr,
                               n_parts_veto_fourier=N_parts_veto_fourier,
                               wind_len_overlap_fourier=Wind_len_overlap_fourier,
                               delta_snr_wide_templ=Delta_snr_wide_templ_veto,
                               threshold_peak_sigma_veto=Threshold_peak_sigma_veto,  # there will also be look-elsewhere
                               save_triggers_as_attribute=False):
        """

        :param nsig_outliers_stat: Deviation in units of stansard deviation to be considered as an outlier in the
            outlier-free variance calculation for the statistic.
        :type nsig_outliers_stat:
        :param apply_veto:
        :type apply_veto:
        :param apply_score_correction:
        :type apply_score_correction:
        :param apply_veto_only_to_neg:
        :type apply_veto_only_to_neg:
        :param margin_snr_veto:
        :type margin_snr_veto:
        :param fn_step_veto:
        :type fn_step_veto:
        :param fn_fit_veto:
        :type fn_fit_veto:
        :param min_len_step_checking_veto:
        :type min_len_step_checking_veto:
        :param max_len_step_checking_veto:
        :type max_len_step_checking_veto:
        :param n_parts_veto_snr:
        :type n_parts_veto_snr:
        :param n_parts_veto_fourier:
        :type n_parts_veto_fourier:
        :param wind_len_overlap_fourier:
        :type wind_len_overlap_fourier:
        :param delta_snr_wide_templ:
        :type delta_snr_wide_templ:
        :param threshold_peak_sigma_veto:
        :type threshold_peak_sigma_veto:
        :param save_triggers_as_attribute:
        :type save_triggers_as_attribute:
        :return:
        :rtype:
        """

        self.matched_filtering(nsig_outliers_stat=nsig_outliers_stat)
        if apply_veto:
            self.veto_statistics(apply_only_to_neg=apply_veto_only_to_neg,
                                 margin_snr=margin_snr_veto,
                                 fn_step=fn_step_veto, fn=fn_fit_veto,
                                 min_len_step_checking=min_len_step_checking_veto,
                                 max_len_step_checking=max_len_step_checking_veto,
                                 n_parts_veto_snr=n_parts_veto_snr,
                                 n_parts_veto_fourier=n_parts_veto_fourier,
                                 wind_len_overlap_fourier=wind_len_overlap_fourier,
                                 delta_snr_wide_templ=delta_snr_wide_templ,
                                 threshold_peak_sigma=threshold_peak_sigma_veto,  # there will also be look-elsewhere
                                 save_triggers_as_attribute=save_triggers_as_attribute)
        self.cut_zero_padding_ends()
        if apply_score_correction:
            self.correct_score()

    def periodicity_search(self, pmin=Pmin, pmax=Pmax, dp=None,
                           duty_cycle=Duty_cycle, delete_full_stats=True,
                           # threshold_mes_veto=Threshold_mes_veto, veto_threshold_for_mes=Veto_threshold_for_mes
                           ):
        """
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
        self.interleave_shifts(delete_fulls=delete_full_stats)
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

    def prepare_for_psd(self, nperseg=None, noverlap=Noverlap_psd, npsdsmin=Npsdsmin,
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
        :return: creates self.big_hole_masks; modifies self.fluxes (-> interpolated fluxes), self.psd_masks
        :rtype:
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
                     subtract_mean=Subtract_mean_in_psd, smoothen_psds=True,
                     tukey_steps_len=16, conditional_zero_zeroth_freq=True, always_zero_zeroth_freq=False,
                     zero_to_high_freq_thresh_wf_ratio=0.6):
        """
        PSDS as function of quarter id.
        The PSD measuring algotithm makes the evenly sampled steps and not adjust the search for
        good pieces according to the placement of the zeros in mask.

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
        :param smoothen_psds: whether to apply multi-resolution smoothing to mitigate the noise effects
        :type smoothen_psds: bool
        :param tukey_steps_len: length taken for zeroing in the tukey windowing
        :type tukey_steps_len: int or float
        :param conditional_zero_zeroth_freq: Whether to put the low frequencies to zero if they deviate too much
        :type conditional_zero_zeroth_freq: bool
        :param always_zero_zeroth_freq: whether to always put the lowest frequencies to zero
        :type always_zero_zeroth_freq: bool
        :param zero_to_high_freq_thresh_wf_ratio : the ratio between the whitening filter in the Fourier domain above which
            the filter is considered to be quickly decaying. Then, Tukey window will be applied to the lowest frequencies.
            It is done because the lowest frequencies often have orders of magnitude spike of PSD which is not resolved by Welch.
        :type zero_to_high_freq_thresh_wf_ratio : float
        :return: creates self.psds, self.psdfreqs, self.wf_tds, self.wf_fds, self.freqs, seld.npsds.
        :rtype:
        """
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

        # qq = -1
        # clean_fluxes = [self.clean_fluxes[qq]]

        self.orig_psds, self.psdfreqs = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,
                                                      nperseg=nperseg, noverlap=noverlap, window=window,
                                                      return_freqs=True, dt=self.dt_sec, return_npsds=False)

        self.freqs = [freqfun(len(time), d=self.dt_sec) for time in self.times]

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
                                     return_freqs=False, dt=self.dt_sec, return_npsds=False)
            psds_coarse = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,  # overlap is relative!
                                        nperseg=nperseg_coarse, noverlap=noverlap, window=window,
                                        return_freqs=False, dt=self.dt_sec, return_npsds=False)
            psds_smooth = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,  # overlap is relative!
                                        nperseg=nperseg_smooth, noverlap=noverlap, window=window,
                                        return_freqs=False, dt=self.dt_sec, return_npsds=False)
            psds_rude = ut.welch_psds(clean_fluxes=clean_fluxes, psd_masks=self.psd_masks,  # overlap is relative!
                                      nperseg=nperseg_rude, noverlap=noverlap, window=window,
                                      return_freqs=False, dt=self.dt_sec, return_npsds=False)

            self.psdfreqs = [freqfun(nperseg_two, d=self.dt_sec) for _ in range(len(self.orig_psds))]
            self.psds = []
            self.wf_fds_meas = []

            for q in range(len(self.orig_psds)):
                noisy_vec = 1 / psds_two[q] ** 0.5
                mean_vec = ut.sinc_interp(1 / psds_smooth[q] ** 0.5, nperseg_two, tukey_steps_len=tukey_steps_len)
                noisy_resid_vec = noisy_vec - mean_vec
                orig_vec = ut.sinc_interp(1 / self.orig_psds[q] ** 0.5, nperseg_two, tukey_steps_len=tukey_steps_len)
                orig_resid_vec = orig_vec - mean_vec
                coarse_vec = ut.sinc_interp(1 / psds_coarse[q] ** 0.5, nperseg_two, tukey_steps_len=tukey_steps_len)
                coarse_resid_vec = coarse_vec - mean_vec
                rude_vec = ut.sinc_interp(1 / psds_rude[q] ** 0.5, nperseg_two, tukey_steps_len=tukey_steps_len)
                rude_resid_vec = rude_vec - mean_vec

                std_noisy = ut.outlier_free_std(noisy_resid_vec[len(noisy_resid_vec) // 2:], nsig=3, n_iterations=2)

                nsig_sign_minwids_res_repl_vecs_res_repl = [[0.6, +1., 20, coarse_resid_vec, coarse_vec, coarse_resid_vec],  # low f bend
                                                            [0.5, +1., 20, rude_resid_vec, coarse_vec, coarse_resid_vec],  # low f bend
                                                            [0.6, -1., 20, coarse_resid_vec, coarse_vec, coarse_resid_vec],  # low f bend
                                                            [1, +1., 10, coarse_resid_vec, coarse_vec, coarse_resid_vec],  # mismatch behav
                                                            [1, -1., 10, coarse_resid_vec, coarse_vec, coarse_resid_vec],  # low freq bend
                                                            [2, -1., 2, coarse_resid_vec, coarse_vec, coarse_resid_vec],  # low freq bend
                                                            [1, -1., 10, orig_resid_vec, orig_vec, orig_resid_vec],  # for mismatch behavior
                                                            [1, +1., 10, orig_resid_vec, orig_vec, orig_resid_vec],  # for mismatch behavior
                                                            [2.5, -1., 4, orig_resid_vec, orig_vec, orig_resid_vec],  # for real features
                                                            [2.5, +1., 4, orig_resid_vec, orig_vec, orig_resid_vec]]  # for lowest freqs
                combined_vec = np.array(mean_vec)
                for nsig, sign, min_wid, res_vec, repl_vec, res_repl in nsig_sign_minwids_res_repl_vecs_res_repl:
                    i = 0
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
                ########################################################################################################################
                # Special treatment for the lowest freqs
                ########################################################################################################################

                # 1. Using better resolution if drops fast

                # Compare value of the wf_fd at low and high freqs. If the low one drops too fast, we will use the
                # more exact measurement to avoid unresolved residual low-freq red noise.
                low_bin = 4  # below this, we take the low frequencies
                high_bin = len(orig_vec) - 20  # From here on is what we compare to
                if ((conditional_zero_zeroth_freq and (
                        np.mean(np.abs(orig_vec[0:3])) / np.mean(np.abs(orig_vec[high_bin:])) < zero_to_high_freq_thresh_wf_ratio))
                        or always_zero_zeroth_freq):
                    i = low_bin
                    while orig_vec[i] < combined_vec[i] and i < len(combined_vec):
                        i += 1
                    combined_vec[:i] = orig_vec[:i]

                # # We will check the residuals of the whitened flux when making the whitening filter (23.09.2023)
                # 2. Checking the whitening residuals

                # Same thing of replacing the low freqs by the exact measurement, but by looking at the residuals (chi2)
                # (I am by far not sure all this is really needed)
                std_noisy_high = np.std(noisy_resid_vec[len(noisy_resid_vec) // 2:])
                low_freq_bin = 8
                low_resid_2_sum = np.sum((noisy_resid_vec[:low_freq_bin] / std_noisy_high) ** 2)
                if low_resid_2_sum >= chi2.isf(low_freq_bin / len(noisy_resid_vec), df=low_freq_bin - 1):
                    j_2 = low_freq_bin
                    sign_1 = np.sign(orig_vec[j_2] - combined_vec[j_2])
                    sign_2 = np.sign(orig_vec[j_2 + 1] - combined_vec[j_2 + 1])
                    while (not np.isclose(sign_1, 0.)) and (np.sign(sign_1 * sign_2) != -1.) and j_2 < len(combined_vec) - 1:
                        j_2 += 1
                        sign_1 = np.sign(orig_vec[j_2] - combined_vec[j_2])
                        sign_2 = np.sign(orig_vec[j_2 + 1] - combined_vec[j_2 + 1])
                    combined_vec[:j_2] = orig_vec[:j_2]

                # 3. Making the lowest freqs wf_fd non-decreasing

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
            # but maybe this exactly means that we should not take this abs?
            self.wf_fds_unwindowed.append(np.array(wf_fd))  # it is unwindowed in the large-window sense of this world

            whitened_clean_flux_fd = fftfun(self.clean_fluxes[q]) * wf_fd
            first_measured_bin = int(np.ceil(self.support / nperseg))
            too_noisy = np.where(np.abs(whitened_clean_flux_fd[:first_measured_bin]) ** 2 / np.mean(np.abs(
                whitened_clean_flux_fd[len(whitened_clean_flux_fd) // 2:]) ** 2) > chi2.isf(1 / (2 * first_measured_bin), df=1))[0]
            if len(too_noisy) > 0 and not always_zero_zeroth_freq:
                len_kill = int(np.ceil(too_noisy[-1] * 1.1))
                lentuk = min(len_kill + 2, first_measured_bin + 2) * 2  # if we need to cover more, we will do that after notch
                # the multiplication by 2 is because the window is symmetric
                wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
            elif always_zero_zeroth_freq:
                lentuk = first_measured_bin * 2
                wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
            self.wf_tds.append(np.array(wf_td, dtype=Dtype_r))
            self.wf_fds.append(np.array(wf_fd, dtype=Dtype_c))
        # We want to somehow make the correlation length smaller but still okay in Fourier:
        self.window_wf_td()
        # # The following comment was indented
        # wf_td = ifftfun(wf_fd)
        # wind = np.zeros(len(wf_td))
        # len_to_save = meas_len  # it is 2 nperseg
        # extension_coef = 2.5
        # tuk = ut.tukey(int(2 * extension_coef * len_to_save), alpha=1 - 1 / extension_coef, sym=True)
        # wind[:len(tuk) // 2] = tuk[-len(tuk) // 2:]
        # wind[-len(tuk) // 2:] = tuk[:len(tuk) // 2]
        # wf_td = wf_td * wind
        # wf_td -= np.mean(wf_td) * wind / np.mean(wind)
        # wf_fd = np.abs(fftfun(wf_td))
        # wf_td = np.fft.fftshift(ifftfun(wf_fd))  # we need wf_fd for inpainting
        # self.wf_tds.append(np.array(wf_td, dtype=Dtype_r))
        # self.wf_fds.append(np.array(wf_fd, dtype=Dtype_c))
        self.wf_tds = np.array(self.wf_tds, dtype=Dtype_r)  # for numba need numpy.arrays

    def check_low_freq_whitening_residuals(self, resid_len_permission=3, wf_td_window_extension_coef=Wf_td_window_extension_coef):
        wf_fds_new = []
        wf_tds_new = []
        for q in self.quarters:
            whitened_clean_flux_fd = np.fft.rfft(self.clean_fluxes[q]) * self.wf_fds[q]
            first_measured_bin = int(np.ceil(self.support / self.nperseg_psd))
            too_noisy = np.where(np.abs(whitened_clean_flux_fd[:4 * first_measured_bin]) ** 2 / np.mean(np.abs(
                whitened_clean_flux_fd[len(whitened_clean_flux_fd) // 2:]) ** 2) > chi2.isf(1 / (4 * first_measured_bin), df=1))[0]
            if len(too_noisy) > resid_len_permission:
                len_kill = int(np.ceil(too_noisy[-1] * 1.1))
                lentuk = len_kill  # * 2  # cover the outlier and a bit more to kill the outlier
                wf_fd_new = np.array(self.wf_fds_unwindowed[q])
                while len(too_noisy) > resid_len_permission:
                    wf_fd_new = np.array(self.wf_fds_unwindowed[q])
                    wf_fd_new[:lentuk // 2] = wf_fd_new[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
                    whitened_clean_flux_fd = np.fft.rfft(self.clean_fluxes[q]) * wf_fd_new
                    too_noisy = np.where(np.abs(whitened_clean_flux_fd[:4 * first_measured_bin]) ** 2 / np.mean(np.abs(
                        whitened_clean_flux_fd[len(whitened_clean_flux_fd) // 2:]) ** 2) > chi2.isf(1 / (4 * first_measured_bin), df=1))[0]
                    lentuk += resid_len_permission
                wf_td_new, wf_fd_new = ut.window_wf_td_fun(wf_fd_new, extension_coef=wf_td_window_extension_coef)
                wf_fds_new.append(wf_fd_new)
                wf_tds_new.append(wf_td_new)
            # if len(too_noisy) > resid_len_permission:
            #     len_kill = int(np.ceil(too_noisy[-1] * 1.1))
            #     lentuk = len_kill  # * 2  # cover the outlier and a bit more to kill the outlier
            #     wf_fd_new = np.array(self.wf_fds_unwindowed[q])
            #     wf_fd_new[:lentuk // 2] = wf_fd_new[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
            #     whitened_clean_flux_fd = np.fft.rfft(self.clean_fluxes[q]) * wf_fd_new
            #     too_noisy = np.where(np.abs(whitened_clean_flux_fd[:4 * first_measured_bin]) ** 2 / np.mean(np.abs(
            #         whitened_clean_flux_fd[len(whitened_clean_flux_fd) // 2:]) ** 2) > chi2.isf(1 / (4 * first_measured_bin), df=1))[0]
            #     while len(too_noisy) > resid_len_permission:
            #         lentuk += resid_len_permission
            #         wf_fd_new = np.array(self.wf_fds_unwindowed[q])
            #         wf_fd_new[:lentuk // 2] = wf_fd_new[:lentuk // 2] * ut.tukey(lentuk, alpha=1, sym=True)[:lentuk // 2]
            #         whitened_clean_flux_fd = np.fft.rfft(self.clean_fluxes[q]) * wf_fd_new
            #         too_noisy = np.where(np.abs(whitened_clean_flux_fd[:4 * first_measured_bin]) ** 2 / np.mean(np.abs(
            #             whitened_clean_flux_fd[len(whitened_clean_flux_fd) // 2:]) ** 2) > chi2.isf(1 / (4 * first_measured_bin), df=1))[0]
            #     wf_td_new, wf_fd_new = ut.window_wf_td_fun(wf_fd_new, extension_coef=wf_td_window_extension_coef)
            #     wf_fds_new.append(wf_fd_new)
            #     wf_tds_new.append(wf_td_new)
            else:
                wf_fds_new.append(self.wf_fds[q])
                wf_tds_new.append(self.wf_tds[q])
        self.wf_fds = wf_fds_new
        self.wf_tds = wf_tds_new  #np.array([np.fft.irfft(wf_fds_new[q]) for q in self.quarters])

    def window_wf_td(self, extension_coef=Wf_td_window_extension_coef):
        """
        applies the tukey window to the whitening template in the time domain to make it smooth but preserve the mean zero
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

    def subtract_low_trends(self, nperseg=None):
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
                detrended, trend_low = ut.detrend_piecewise(self.clean_fluxes[q][self.zero_padding_ends[q][0]:len(
                                                            self.clean_fluxes[q]) - self.zero_padding_ends[q][-1]],
                                                            big_mask=self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(
                                                                self.big_hole_masks[q]) - self.zero_padding_ends[q][-1]],
                                                            mask=self.masks[q][self.zero_padding_ends[q][0]:len(
                                                                self.masks[q]) - self.zero_padding_ends[q][-1]],
                                                            nperseg=nperseg // 2 * 3, endlen=endlen, return_trend=True)
                self.trends_low.append(np.array(trend_low, dtype=Dtype_r))
                self.fluxes[q] = np.array(np.array(self.fluxes[q]) - np.concatenate(
                    (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1]))), dtype=Dtype_r)
                self.clean_fluxes[q] = np.array(np.array(self.clean_fluxes[q]) - np.concatenate(
                    (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1]))), dtype=Dtype_r)

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
                self.fluxes[i] = np.array(sosfiltfilt(sos, np.array(self.fluxes[i]), padtype='constant'), dtype=Dtype_r)
                self.clean_fluxes[i] = np.array(sosfiltfilt(sos, np.array(self.clean_fluxes[i]), padtype='constant'), dtype=Dtype_r)
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
        self.fluxes = np.array(self.fluxes, dtype=Dtype_r)
        self.fluxes = self.inpaint_fluxes_fast(self.fluxes, self.masks, self.wf_tds, dtype=Dtype_r)
        self.is_fluxes_inpainted = True

    def whiten_fluxes(self, whiten_clean_fluxes=False):
        """
        whitens all the fluxes with whitening filter
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

        Requires
        ----------
        self.notch_soss : the data should be notched
        self.templates : the templates should be initialized

        Modifies
        ----------
        self.templates
            Notches templates in time domain. After this, need to whiten the templates
        """
        if self.notch_soss is not None:
            for k in range(len(self.tb.templates_shifts)):  # different template indices and different subbin shifts
                for sos in self.notch_soss:
                    self.tb.templates_shifts[k] = sosfiltfilt(sos, np.array(self.tb.templates_shifts[k]), padtype='constant')
            self.tb.make_grouped_shifts_arrays()
        # Very important to have zeros in the ends, bacause of padding.
        # Ppadtype 'constant' will pad by the edge value, and we need to ensure it is zero
        # That is why notching has to be done when the template is centered at the center,
        # and only then the template can be rolled.

    def whiten_templates(self):
        """
        Whitens all the templates in self.templates by applying
        the whitening filter calculated after the PSD measurement of the data.

        Requires:
        self.wf_fds : the data should be whitened
        self.support : the data should be zero-padded

        Creates:
        self.whitened_templates_td : dict of len(templates) entries, each of length self.quarters
            Dictionary of all the whitened templates for all the quarters in time domain
        self.whitened_templates_fd : dict of len(templates) entries, each of length self.quarters
            Dictionary of all the whitened templates for all the quarters in Fourier domain

        :return: self.whitened_templates_td, self.whitened_templates_fd
        :rtype:
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

        :param detr_length_outliers: Detrending length used to detrend the data before calculating the outlier-free variance.
            Not really useful for the statistic.
        :type detr_length_outliers: float
        :param nsig_outliers_stat: Deviation in units of stansard deviation to be considered as an outlier in the
            outlier-free variance calculation for the statistic.
        :type nsig_outliers_stat: float
        :param ngap_moving_var:
        :type ngap_moving_var:
        :param nsamples_min_moving_var:
        :type nsamples_min_moving_var:
        :return:
        :rtype:
        """
        n_wh_temp = len(self.whitened_templates_td)
        # self.statistics_raw = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for _ in range(n_wh_temp)]
        self.statistics_raw_cor = [[np.zeros(len(self.fluxes[q]), dtype=Dtype_r) for q in self.quarters]
                                   for _ in range(n_wh_temp)]
        # -> corrected for moving std and then multiplied back by non-moving std
        self.stat_moving_vars = [[np.zeros(len(self.fluxes[q]), dtype=Dtype_r) for q in self.quarters]
                                 for _ in range(n_wh_temp)]
        self.statistics_snr = [[np.zeros(len(self.fluxes[q]), dtype=Dtype_r) for q in self.quarters]
                               for _ in range(n_wh_temp)]
        # self.statistics_abs = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for t in range(n_wh_temp)]
        self.stat_vars = [[Dtype_r(0.) for _ in self.quarters] for __ in range(n_wh_temp)]  # mean of moving variance
        self.stat_denoms = [[Dtype_r(np.inf) for _ in self.quarters] for __ in range(n_wh_temp)]  # mean of den stat
        self.stat_masks = [[np.ones(len(self.fluxes[q])).astype(bool) for q in self.quarters] for _ in
                           range(n_wh_temp)]

        assert self.is_template_whitened, 'Templates have to be whitened for matched filtering!'

        for k in range(n_wh_temp):
            for q in self.quarters:
                # statistics not in snr, not in absolute units
                stat = np.real(
                    np.fft.irfft(np.conjugate(self.whitened_templates_fd[k][q]) * self.whitened_fluxes_fd[q]))
                stat = np.array(
                    stat[self.zero_padding_ends[q][0]:len(stat) - self.zero_padding_ends[q][1]])  # CUT THE STATISTIC BECAUSE MEMORY
                # This denom will be needed for the amplitude calculation:
                stat_denom = np.fft.irfft(np.abs(self.whitened_templates_fd[k][q]) ** 2)[0]

                statmask = np.array(self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(
                    self.big_hole_masks[q]) - self.zero_padding_ends[q][1]])
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

    def time_slides(self, std_time_scrambling=Std_time_scrambling):
        """
        This function is not used for now, we use the scrambling
        :param std_time_scrambling:
        :type std_time_scrambling:
        :return:
        :rtype:
        """
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
        This function is not used for now, we use the scrambling
        :return:
        :rtype:
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
                        fn_step=Fn_step_veto, fn=Fn_fit_veto,
                        min_len_step_checking=Min_len_step_checking_veto,
                        max_len_step_checking=Max_len_step_checking_veto,
                        n_parts_veto_snr=N_parts_veto_snr,
                        n_parts_veto_fourier=N_parts_veto_fourier,
                        wind_len_overlap_fourier=Wind_len_overlap_fourier,
                        delta_snr_wide_templ=Delta_snr_wide_templ_veto,
                        threshold_peak_sigma=Threshold_peak_sigma_veto,  # there will also be look-elsewhere
                        save_triggers_as_attribute=False):
        """
        Verify big triggers and mask ones which are non-transit-like.
        Selects from the statistic the triggers whose abs is >margin_snr, which are negative or
        both negative and positive, as defined by `apply_only_to_neg` parameter.
        Triggers inside big holes are ignored.
        Makes several rests: chi-square partial amplitude tests in time and fourier domain,
        alternative hypothesis testing against step and peak, masking out narrower transits if wider are
        sufficiently more significant.
        Should be done before unpadding!
        :param apply_only_to_neg: Whether to apply the veto only to negative statistics, which transits are expected to be, or
            also to the positive ones. The latter is needed if we want to look at the inverted data. It also can affetct the negative
            values processing due to masking and comparison.
        :type apply_only_to_neg: bool
        :param margin_snr: the minimal abs value of the snr from which the vetting starts to be applied
        :type margin_snr: float
        :param fn_step: false negative probability for the test against step
        :type fn_step: float
        :param fn: false negative probability for vetoes except step
        :type fn: float
        :param min_len_step_checking: minimal number of points participating in the mean calculation for the step veto
        :type min_len_step_checking: float, bins
        :param max_len_step_checking: maximal and desired number of points participating in the mean calculation for the step veto
        :type max_len_step_checking: float, bins
        :param n_parts_veto_snr: maximal number of parts to split the whitened template when doing vetting in time domain
        :type n_parts_veto_snr: int
        :param n_parts_veto_fourier: maximal number of parts to split the whitened template when doing vetting in fourier domain
        :type n_parts_veto_fourier: int
        :param wind_len_overlap_fourier: size of the window for the template splitting in the fourier domain. is needed to avoid too
            long templates in the time domain. Is recommended to be smaller than the nperseg
        :type wind_len_overlap_fourier: int
        :param delta_snr_wide_templ: the difference in the statistic in units of snr between a given template and a wider template
            for the given template to be masked as insignificant
        :type delta_snr_wide_templ: float
        :param threshold_peak_sigma: threshold value for the peak veto
        :type threshold_peak_sigma: float
        :param save_triggers_as_attribute: whether to save the triggers with descriptions as part of the class attribute. If
            `True`, veto masks will be created.
        :type save_triggers_as_attribute: bool
        :return:
        :rtype:
        """
        assert self.is_zero_padded, 'Zero padding ends must not be removed before vetting!'
        self.stats_raw_no_veto_no_score_cor = copy.deepcopy(self.statistics_raw_cor)
        self.veto_masks = [[np.ones(len(self.stat_masks[k][q])).astype(bool) for q in self.quarters] for k in
                           range(len(self.stat_masks))]
        if save_triggers_as_attribute:
            self.veto_bad_trigger_inds = [[{} for _ in self.quarters] for __ in range(len(self.whitened_templates_td))]
            self.trigger_indices = [[[] for _ in self.quarters] for __ in range(len(self.whitened_templates_td))]

        # we prepare here renormalized fluxes and templates to use them in the veto and not renormalize there every time:
        norms_templates = [[np.linalg.norm(self.whitened_templates_td[k][q]) for q in self.quarters]
                           for k in range(len(self.whitened_templates_td))]
        whitened_templates_td_renorm = [[self.whitened_templates_td[k][q] / norms_templates[k][q]
                                         for q in self.quarters] for k in range(len(self.whitened_templates_td))]
        whitened_templates_fd_renorm = [[self.whitened_templates_fd[k][q] / norms_templates[k][q]
                                         for q in self.quarters] for k in range(len(self.whitened_templates_td))]
        stds_fluxes = [np.std(self.whitened_fluxes_td[q][self.big_hole_masks[q]]) for q in self.quarters]
        fluxes_wh_fd_rewh = [self.whitened_fluxes_fd[q] / stds_fluxes[q] for q in self.quarters]

        # find the best templates to use them in the template width veto (warning: unpadded!)
        templs_of_max_snr = [np.argmax(np.array([self.statistics_snr[k][q] for k in range(len(self.statistics_snr))]), axis=0)
                             for q in self.quarters]
        templs_of_min_snr = [np.argmax(np.array([self.statistics_snr[k][q] for k in range(len(self.statistics_snr))]), axis=0)
                             for q in self.quarters]

        stats_peak = []
        whitened_peak_temps_fd = []
        whitened_peak_temps_td = []
        # this will not be important for the normal search because peaks will be filtered in the template_width veto.
        # but for the injections, it may be more important. In the injections, there is no template_width because we only use one
        # template. It is bad because the injections are not exactly as the main search, but this peak veto will make it
        # at least a bit closer
        if self.tb.durations[-1] / self.dt_days > 3:  # if there will be some templates requiring peak veto
            peak_template = np.zeros_like(self.whitened_templates_td[0][0])
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

        for k in range(len(self.tb.initial_templates_shifts)):
            template = copy.deepcopy(self.tb.initial_templates_shifts[k])  # needed only for step veto
            for q in self.quarters:  # over quarters
                stat_mask_padded = np.concatenate((np.zeros(self.zero_padding_ends[q][0]).astype(bool), self.stat_masks[k][q],
                                                   np.zeros(self.zero_padding_ends[q][1]).astype(bool)))
                # -> allow small holes to be in the transit
                statistic_snr_padded = np.concatenate((np.zeros(self.zero_padding_ends[q][0]), self.statistics_snr[k][q],
                                                       np.zeros(self.zero_padding_ends[q][1])))
                if apply_only_to_neg:
                    trigger_indices_quarter = np.where((statistic_snr_padded <= 0) & (np.abs(statistic_snr_padded) >= margin_snr
                                                                                      ) & stat_mask_padded)[0]
                    # -> if the index was already fixed to be bad before, it will not be used anymore.
                else:
                    trigger_indices_quarter = np.where((np.abs(statistic_snr_padded) >= margin_snr) & stat_mask_padded)[0]

                trigger_indices_quarter_mask = np.ones(len(trigger_indices_quarter)).astype(bool)
                if save_triggers_as_attribute:
                    self.trigger_indices[k][q] = trigger_indices_quarter

                # CHECKING FOR STEP
                for i_ind, ind in enumerate(trigger_indices_quarter):
                    jump_to_left = len(template) // 2 - np.nonzero(template)[0][0]
                    jump_to_right = len(template) // 2 - np.nonzero(template[::-1])[0][0]
                    if (ind - jump_to_left >= min_len_step_checking) and (
                            ind + jump_to_right < len(self.fluxes[q]) - max_len_step_checking):
                        ind_left = max(0, ind - jump_to_left - max_len_step_checking)
                        ind_right = min(ind + jump_to_right + max_len_step_checking + 1, len(self.fluxes[q]))
                        indices_center_exclude = np.arange(ind - jump_to_left, ind + jump_to_right + 1) - ind_left
                        data_to_test = np.array(self.fluxes[q][ind_left:ind_right])
                        mask_to_test = np.array(self.big_hole_masks[q][ind_left:ind_right])
                        p_value_step = ut.veto_step(data=data_to_test, indices_center_exclude=indices_center_exclude,
                                                    mask=mask_to_test, min_len_to_process=min_len_step_checking)
                        if p_value_step is not None:  # it will be none if all the points are masked
                            if p_value_step < fn_step:
                                self.veto_masks[k][q][ind - self.zero_padding_ends[q][0]] = False  # step
                                self.stat_masks[k][q][ind - self.zero_padding_ends[q][0]] = False  # step
                                trigger_indices_quarter_mask[i_ind] = False
                                if save_triggers_as_attribute:
                                    if ind not in self.veto_bad_trigger_inds[k][q]:
                                        self.veto_bad_trigger_inds[k][q][ind] = 'step '
                                    else:
                                        self.veto_bad_trigger_inds[k][q][ind] = self.veto_bad_trigger_inds[k][q][ind] + 'step '
                    else:
                        self.veto_masks[k][q][ind - self.zero_padding_ends[q][0]] = False  # extreme_position
                        self.stat_masks[k][q][ind - self.zero_padding_ends[q][0]] = False  # extreme_position
                        trigger_indices_quarter_mask[i_ind] = False
                        if save_triggers_as_attribute:
                            if ind not in self.veto_bad_trigger_inds[k][q]:
                                self.veto_bad_trigger_inds[k][q][ind] = 'extreme_position '
                            else:
                                self.veto_bad_trigger_inds[k][q][ind] = self.veto_bad_trigger_inds[k][q][ind] + 'extreme_position '

                trigger_indices_quarter = trigger_indices_quarter[trigger_indices_quarter_mask]
                # If one test is false, we will not to run the next one.
                trigger_indices_quarter_mask = np.ones(len(trigger_indices_quarter)).astype(bool)

                # VETO OF NARROW TEMPLATES TRIGGERING ON WIDER TRANSITS
                # everywhere_good_masks = [np.prod([(self.veto_masks[k][q].astype(Dtype_r))
                for i_ind, ind in enumerate(trigger_indices_quarter):
                    is_good = True
                    snr_k = statistic_snr_padded[ind]  # self.statistics_snr[k][q][ind]
                    if snr_k > 0:
                        temp_max = templs_of_max_snr[q][ind - self.zero_padding_ends[q][0]]
                        snr_max = self.statistics_snr[temp_max][q][ind - self.zero_padding_ends[q][0]]
                        if ((self.tb.durations[self.tb.templates_of_indices[k]] < self.tb.durations[self.tb.templates_of_indices[temp_max]])
                                and (snr_k < snr_max - delta_snr_wide_templ)):
                            is_good = False
                    else:
                        temp_min = templs_of_min_snr[q][ind - self.zero_padding_ends[q][0]]
                        snr_min = self.statistics_snr[temp_min][q][ind - self.zero_padding_ends[q][0]]
                        if ((self.tb.durations[self.tb.templates_of_indices[k]] < self.tb.durations[self.tb.templates_of_indices[temp_min]])
                                and (snr_k > snr_min + delta_snr_wide_templ)):
                            is_good = False
                    if not is_good:
                        # indices are of padded, and masks are unpadded to save memory:
                        self.veto_masks[k][q][ind - self.zero_padding_ends[q][0]] = False  # template_width
                        self.stat_masks[k][q][ind - self.zero_padding_ends[q][0]] = False  # template_width
                        trigger_indices_quarter_mask[i_ind] = False
                        if save_triggers_as_attribute:
                            if ind not in self.veto_bad_trigger_inds[k][q]:
                                self.veto_bad_trigger_inds[k][q][ind] = 'template_width '
                            else:
                                self.veto_bad_trigger_inds[k][q][ind] = self.veto_bad_trigger_inds[k][q][ind] + 'template_width '
                trigger_indices_quarter = trigger_indices_quarter[trigger_indices_quarter_mask]

                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING BY SNR IN TIME DOMAIN)
                isgood_parts = ut.veto_parts(data_wh_fd=fluxes_wh_fd_rewh[q],
                                             template_wh_full=whitened_templates_td_renorm[k][q],
                                             indices_padded=trigger_indices_quarter,
                                             mask_padded=stat_mask_padded, tlen=None, hlen_if_short=5, fn=fn,
                                             n_parts=n_parts_veto_snr,
                                             nperseg_psd=self.nperseg_psd, splitting='snr', lobe_hlen=5)

                self.veto_masks[k][q][trigger_indices_quarter[~isgood_parts] - self.zero_padding_ends[q][0]] = False
                self.stat_masks[k][q][trigger_indices_quarter[~isgood_parts] - self.zero_padding_ends[q][0]] = False
                # -> inconsistent_parts_time
                if save_triggers_as_attribute:
                    for j, ind_j in enumerate(trigger_indices_quarter):
                        if not isgood_parts[j]:
                            if ind_j not in self.veto_bad_trigger_inds[k][q]:
                                self.veto_bad_trigger_inds[k][q][ind_j] = 'inconsistent_parts_time '
                            else:
                                self.veto_bad_trigger_inds[k][q][ind_j] = (self.veto_bad_trigger_inds[k][q][j]
                                                                           + 'inconsistent_parts_time ')
                trigger_indices_quarter = trigger_indices_quarter[isgood_parts]

                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING IN FOURIER)
                isgood_parts = ut.veto_parts_fourier(data_wh_fd=fluxes_wh_fd_rewh[q],
                                                     template_wh_fd=whitened_templates_fd_renorm[k][q],
                                                     indices_padded=trigger_indices_quarter,
                                                     mask_padded=stat_mask_padded, n_parts=n_parts_veto_fourier,
                                                     fn=fn, wind_len_overlap=wind_len_overlap_fourier)
                self.veto_masks[k][q][trigger_indices_quarter[~isgood_parts] - self.zero_padding_ends[q][0]] = False
                self.stat_masks[k][q][trigger_indices_quarter[~isgood_parts] - self.zero_padding_ends[q][0]] = False
                # -> inconsistent_parts_fourier
                if save_triggers_as_attribute:
                    for j, ind_j in enumerate(trigger_indices_quarter):
                        if not isgood_parts[j]:
                            if ind_j not in self.veto_bad_trigger_inds[k][q]:
                                self.veto_bad_trigger_inds[k][q][ind_j] = 'inconsistent_parts_fourier '
                            else:
                                self.veto_bad_trigger_inds[k][q][ind_j] = (self.veto_bad_trigger_inds[k][q][j]
                                                                           + 'inconsistent_parts_fourier ')
                trigger_indices_quarter = trigger_indices_quarter[isgood_parts]

                # # CHECKING FOR PEAK (can only work for the full bank)
                # warning: here, work with unpadded!
                trig_ind_quart_unpad = trigger_indices_quarter - self.zero_padding_ends[q][0]
                dur_bins = self.tb.durations[self.tb.templates_of_indices[k]] / self.dt_days
                if dur_bins > 5:
                    h_dur_check = int(self.tb.durations[self.tb.templates_of_indices[k]] / self.dt_days / 2)
                    decorr_coef = (np.fft.irfft(self.whitened_templates_fd[k][q] * np.conjugate(whitened_peak_temps_fd[q]))[0] /
                                   ((whitened_peak_temps_td[q] @ whitened_peak_temps_td[q])
                                   * (self.whitened_templates_td[k][q] @ self.whitened_templates_td[k][q]))**0.5)  # not rolled, approx
                    max_peak_stats = np.array([np.max(stats_peak[q][max(0, i - h_dur_check): i + h_dur_check])
                                               for i in range(len(stats_peak[q]))])
                    min_peak_stats = np.array([np.min(stats_peak[q][max(0, i - h_dur_check): i + h_dur_check])
                                               for i in range(len(stats_peak[q]))])
                    signs = np.sign(self.statistics_snr[k][q][trig_ind_quart_unpad])
                    min_or_max = np.array([max_peak_stats[ind_j] if signs[j] > 0 else min_peak_stats[ind_j]
                                           for j, ind_j in enumerate(trig_ind_quart_unpad)])
                    is_goods = (signs * (self.statistics_snr[k][q][trig_ind_quart_unpad] * decorr_coef
                                - min_or_max) / (1 - decorr_coef**2)**0.5 < threshold_peak_sigma)
                    self.veto_masks[k][q][trig_ind_quart_unpad[~is_goods]] = False  # peak
                    self.stat_masks[k][q][trig_ind_quart_unpad[~is_goods]] = False  # peak
                    if save_triggers_as_attribute:
                        for j, ind_j in enumerate(trigger_indices_quarter):
                            if not is_goods[j]:
                                if ind_j not in self.veto_bad_trigger_inds[k][q]:
                                    self.veto_bad_trigger_inds[k][q][ind_j] = 'peak_test '
                                else:
                                    self.veto_bad_trigger_inds[k][q][ind_j] = (self.veto_bad_trigger_inds[k][q][j]
                                                                               + 'peak_test ')
                    # trigger_indices_quarter = trigger_indices_quarter[is_goods]

        # Setting statistics in the masked places to zero:
        for k in range(len(self.statistics_snr)):
            for q in self.quarters:
                self.statistics_snr[k][q][~self.stat_masks[k][q]] = 0.
                self.statistics_raw_cor[k][q][~self.stat_masks[k][q]] = 0.

    def correct_score(self):
        # TODO: think how zeros in the masked region influence the period-summed distribution
        # all_stat = np.array([])
        # for k in range(len(self.statistics_snr)):
        #     for q in range(len(self.statistics_snr[k])):
        #         all_stat = np.concatenate((all_stat, self.statistics_snr[k][q][self.stat_masks[k][q]]))
        # #  all_stat_c, transform = ut.score_correction(all_stat, return_function=True, negative_values=True)
        # all_stat_c, transform = ut.score_correction(all_stat, return_function=True, two_sided_distr=True)
        # # -> we do not use one-sided distr or do????????
        # for k in range(len(self.statistics_snr)):
        #     for q in range(len(self.statistics_snr[k])):
        #         self.statistics_snr[k][q][self.stat_masks[k][q]] = transform(
        #             self.statistics_snr[k][q][self.stat_masks[k][q]])
        #         self.statistics_raw_cor[k][q][self.stat_masks[k][q]] = (self.statistics_snr[k][q][self.stat_masks[k][q]] /
        #                                                                 self.stat_moving_vars[k][q][self.stat_masks[k][q]] ** 0.5
        #                                                                 * np.nanmean(self.stat_moving_vars[k][q][self.stat_masks[k][q]]))

        # Now (27/07/2023), do it per template (which is especially important for the injections where we only have one template):
        for k in range(len(self.statistics_snr)):
            all_stat = np.concatenate([self.statistics_snr[k][q][self.stat_masks[k][q]] for q in range(len(self.statistics_snr[k]))])
            all_stat_c, transform = ut.score_correction(all_stat, return_function=True, two_sided_distr=True)
            for q in range(len(self.statistics_snr[k])):
                self.statistics_snr[k][q][self.stat_masks[k][q]] = transform(
                    self.statistics_snr[k][q][self.stat_masks[k][q]])
                self.statistics_raw_cor[k][q][self.stat_masks[k][q]] = (self.statistics_snr[k][q][self.stat_masks[k][q]] /
                                                                        self.stat_moving_vars[k][q][self.stat_masks[k][q]] ** 0.5
                                                                        * np.nanmean(self.stat_moving_vars[k][q][self.stat_masks[k][q]]))

    def connect_quarters(self):
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
        self.big_hole_masks_full = (af.connect_quarters_fun(self.times, self.big_hole_masks,
                                                            zero_padding_ends=None,
                                                            is_data_cut=~self.is_zero_padded,
                                                            is_time_cut=~self.is_zero_padded,
                                                            dt=self.dt_days, fill_gaps_by_nan=False)[1]).astype(bool)

        self.statistics_full = np.zeros((len(self.statistics_raw_cor), len(self.time_full)), dtype=Dtype_r)
        self.stat_vars_full = np.zeros((len(self.statistics_raw_cor), len(self.time_full)), dtype=Dtype_r)
        self.stat_masks_full = np.zeros((len(self.statistics_raw_cor), len(self.time_full)), dtype=bool)
        vars_quarters = [[np.array(np.ones(len(self.statistics_raw_cor[k][q])) * self.stat_vars[k][q], dtype=Dtype_r)
                          for q in self.quarters] for k in range(len(self.statistics_raw_cor))]

        for k in range(len(self.statistics_raw_cor)):
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

    def scramble_full_stats(self, max_hole=Max_hole_pieces_scrampling, max_piece=Max_piece_scrambling, min_piece=Min_piece_scrambling):
        # if self.inds_pieces_scramble is not None:
        #     self.undo_scramble_full_stats()
        # if it was never scrambled before or was unscrambled, we need to define the pieces to be shuffled:
        if self.inds_pieces_scramble is None:
            inds_pieces_scramble = ut.get_pieces_indices(mask=self.big_hole_masks_full, max_hole=max_hole, max_piece=max_piece,
                                                         min_piece=min_piece)
            # warning: if there is a hole in the beginning or the end, it may (?) be treated incorrectly and the
            # length of the scrambled data will be effectively bigger
            self.tot_hole_len_scramble = len(self.time_full) - np.sum([ip[1] - ip[0] for ip in inds_pieces_scramble])
            self.inds_pieces_scramble = inds_pieces_scramble
            np.random.shuffle(self.inds_pieces_scramble)
        else:
            numbers_shuffle = np.arange(len(self.inds_pieces_scramble)).astype(int)
            np.random.shuffle(numbers_shuffle)
            self.inds_pieces_scramble = self.inds_pieces_scramble[numbers_shuffle]  # save to be able to unscramble
            inds_pieces_scramble = self.inds_pieces_in_scrambled[numbers_shuffle]  # to rescramble the present scrambled pieces

        # how many spaces between the pieces and plus one in the end, although it effectively changes the length ->
        len_holes = ut.gen_uniform_piece_lengths(tot_len=self.tot_hole_len_scramble, n_pieces=len(inds_pieces_scramble) - 1)
        # "connect_quarters" is now destroyed. do it again to restore the original positioning
        self.trend_orig_flux_full, self.inds_pieces_in_scrambled = ut.rearrange_pieces(self.trend_orig_flux_full,
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

    def undo_scramble_full_stats(self):
        self.trend_orig_flux_full = ut.rearrange_pieces(self.trend_orig_flux_full, inds_pieces=self.inds_pieces_in_scrambled,
                                                        hole_lens=None, fill_hole_with=0., inds_target=self.inds_pieces_scramble)
        for k in range(len(self.statistics_full)):
            self.statistics_full[k] = ut.rearrange_pieces(self.statistics_full[k], inds_pieces=self.inds_pieces_in_scrambled,
                                                          hole_lens=None, fill_hole_with=0., inds_target=self.inds_pieces_scramble)
            self.stat_masks_full[k] = ut.rearrange_pieces(self.stat_masks_full[k], inds_pieces=self.inds_pieces_in_scrambled,
                                                          hole_lens=None, fill_hole_with=False, inds_target=self.inds_pieces_scramble)
            self.stat_vars_full[k] = ut.rearrange_pieces(self.stat_vars_full[k], inds_pieces=self.inds_pieces_in_scrambled,
                                                         hole_lens=None, fill_hole_with=0., inds_target=self.inds_pieces_scramble)
        self.inds_pieces_scramble = None
        self.inds_pieces_in_scrambled = None
        self.tot_hole_len_scramble = None

    def interleave_shifts(self, delete_fulls=True):
        """
        Takes statistics of the same template with different shifts and interleaves them in the alternating
        manner to create continuity of sliding of this template over the data. The resulting array will have the length
        equal to the sum of the length of all the arrays for different shifts.
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
            self.statistics_interleaved = np.array(self.statistics_interleaved, dtype=Dtype_r)
            self.stat_masks_interleaved = np.array(self.stat_masks_interleaved, dtype=bool)
            self.stat_vars_interleaved = np.array(self.stat_vars_interleaved, dtype=Dtype_r)
            self.trend_orig_flux_interleaved = np.array(np.vstack((
                self.trend_orig_flux_full, self.trend_orig_flux_full,
                self.trend_orig_flux_full)).T.flatten(), dtype=Dtype_r)
            self.time_interleaved = np.arange(len(self.trend_orig_flux_interleaved)) * self.dt_days / self.n_shifts + self.time_full[0]

        # for not interleaved:
        if len(self.not_interleaved_numbers) > 0:
            self.statistics_not_interleaved = []
            self.stat_masks_not_interleaved = []
            self.stat_vars_not_interleaved = []
            indices_for_not_interleaved = [self.indices_of_templates[i][0] for i in self.not_interleaved_numbers]
            self.statistics_not_interleaved = np.array(self.statistics_full[indices_for_not_interleaved], dtype=Dtype_r)
            self.stat_masks_not_interleaved = np.array(self.stat_masks_full[indices_for_not_interleaved], dtype=bool)
            self.stat_vars_not_interleaved = np.array(self.stat_vars_full[indices_for_not_interleaved], dtype=Dtype_r)
            self.trend_orig_flux_not_interleaved = np.array(self.trend_orig_flux_full, dtype=Dtype_r)
            self.time_not_interleaved = np.array(np.arange(len(self.trend_orig_flux_not_interleaved)) * self.dt_days
                                                 + self.time_full[0], dtype=Dtype_r)
        if delete_fulls:
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
            del self.wf_tds
        if hasattr(self, 'wf_fd'):
            del self.wf_fds
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

        if hasattr(self, 'whitened_clean_fluxes_fd'):
            del self.whitened_clean_fluxes_fd

        if hasattr(self, 'triggers'):
            del self.triggers
        if hasattr(self, 'stats_raw_no_veto_no_score_cor'):
            del self.stats_raw_no_veto_no_score_cor
        if hasattr(self, 'veto_bad_trigger_inds'):
            del self.veto_bad_trigger_inds

    def make_periods_array(self, pmin=Pmin, pmax=Pmax, dp=None, duty_cycle=Duty_cycle, t_mission=None):
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
        :type pmin: float, days
        :param pmax: Maximal period to consider.
        :type pmax: float, days
        :param dp: Step in the list of all the investigated periods
        :type dp: float, days
        :param duty_cycle:
        :type duty_cycle: float
        :param t_mission: total duration between the beginning and the end of the data
        :type t_mission: float, days
        :return:
        :rtype:
        """
        if t_mission is None:
            t_mission = self.T_mission
        per = pmin
        periods = [per]
        if dp is None:
            while per <= pmax:
                dp = duty_cycle * per / t_mission
                per += dp
                periods.append(per)
        else:
            while per <= pmax:
                per += dp
                periods.append(per)
        periods = np.array(periods)
        return periods

    def sum_periods(self, pmin=Pmin, pmax=Pmax, dp=None, duty_cycle=Duty_cycle, periods=None):
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
        :param periods: Alternatively, you can specify all the periods, and then the previous parameters are not needed.
        :type periods: ndarray
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
        if periods is None:
            self.Pmin = pmin
            self.Pmax = pmax
            periods = self.make_periods_array(pmin=pmin, pmax=pmax, dp=dp, duty_cycle=duty_cycle, t_mission=None)
            self.periods = periods
        else:
            self.Pmin = periods[0]
            self.Pmax = periods[-1]
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
            statistics_by_trends = np.array(self.statistics_interleaved * self.trend_orig_flux_interleaved, dtype=Dtype_r)
            statistics_by_trends = np.hstack((statistics_by_trends, np.zeros(
                (len(statistics_by_trends), int(np.ceil(Pmax / (self.dt_days / self.n_shifts)))))))
            # These phases are with respect to the beginning of the data which can be any
            # number because sometimes we reject the zeroth quarter or do other things.
            # Do not use these phases anywhere.
            # Pad by zeros to avoid extra if statements in the summation:
            trends_sq_by_vars = np.array(self.stat_vars_interleaved * self.trend_orig_flux_interleaved ** 2, dtype=Dtype_r)
            trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros(
                (len(trends_sq_by_vars), int(np.ceil(Pmax / (self.dt_days / self.n_shifts)))))))

            self.statistics_periodic_interleaved, denomsq = self.sum_periods_fast(
                statistics_by_trends, trends_sq_by_vars, self.dt_days / self.n_shifts, self.tmax + self.dt_days * (
                        1 - 1 / self.n_shifts), self.tmin, periods)
            # attention: we use the same dt for interleaved and not? it means that the phase step for the interleaved could be smaller.
            for p in range(len(self.statistics_periodic_interleaved)):
                for t in range(len(self.statistics_periodic_interleaved[p])):
                    self.statistics_periodic_interleaved[p][t][denomsq[p][t] == 0] = np.nan
                    self.statistics_periodic_interleaved[p][t][denomsq[p][t] != 0] /= denomsq[p][t][
                                                                                          denomsq[p][t] != 0] ** 0.5

        # not interleaved
        if len(self.not_interleaved_numbers) > 0:
            self._phases_not_interleaved = np.arange(int(np.ceil((self.periods[-1]) / self.dt_days))) * self.dt_days
            statistics_by_trends = np.array(self.statistics_not_interleaved * self.trend_orig_flux_not_interleaved, dtype=Dtype_r)
            statistics_by_trends = np.hstack((statistics_by_trends, np.zeros(
                (len(statistics_by_trends), int(np.ceil(Pmax / self.dt_days))))))
            # These phases are with respect to the beginning of the data which can be any
            # number because sometimes we reject the zeroth quarter or do other things.
            # Do not use these phases anywhere.
            # Pad by zeros to avoid extra if statements in the summation:
            trends_sq_by_vars = np.array(self.stat_vars_not_interleaved * self.trend_orig_flux_not_interleaved ** 2, dtype=Dtype_r)
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
                    if not np.isclose(summed_stats[si], 0) and not np.isinf(summed_vars[si]):
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

    def save_best_candidates(self, n_per_temp=N_per_temp_saving, path=None,
                             save_to_file=True, save_inverted=True,
                             filename_postfix_period=Default_best_candidates_filename_postfix_period,
                             filename_postfix_inverted_period=Default_inverted_best_candidates_filename_postfix_period):
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
        # period_indices_selected = []
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
                # period_indices_selected.append(stat_best_per_inds)
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
                # period_indices_selected.append(stat_best_per_inds)
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
        if self.inds_pieces_scramble is not None:
            self.best_periodic_scores_data[Search_results_index_slides] = [self.inds_pieces_scramble, self.inds_pieces_in_scrambled]
        elif self.time_slide_shifts is not None:
            self.best_periodic_scores_data[Search_results_index_slides] = [self.time_slide_shifts, times_quarters]
        self.best_periodic_scores_data[Search_results_index_per_list] = self.periods

        if save_to_file:
            filename_postfix = filename_postfix_period + str(self.Pmin) + '_' + str(self.Pmax)
            file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                                filename_postfix=filename_postfix, file_format='npy')

            np.save(file_path, np.array(self.best_periodic_scores_data, dtype=object), allow_pickle=True)

        if save_inverted:
            # period_indices_selected = []
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
                    # period_indices_selected.append(stat_best_per_inds)
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
                    # period_indices_selected.append(stat_best_per_inds)
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
            # warning: you updated time slides, but the indices of the shuffled pieces are not saved here because you did not need them yet.

            self.best_inverted_periodic_scores_data = [[] for _ in range(Search_results_length)]
            self.best_inverted_periodic_scores_data[Search_results_index_per] = periods_selected
            self.best_inverted_periodic_scores_data[Search_results_index_stat] = statistic_values_selected
            self.best_inverted_periodic_scores_data[Search_results_index_sum_tr_times] = summed_transit_times_selected
            self.best_inverted_periodic_scores_data[Search_results_index_sum_stats] = summed_stats_raw_selected
            self.best_inverted_periodic_scores_data[Search_results_index_sum_trends] = summed_trends_selected
            self.best_inverted_periodic_scores_data[Search_results_index_sum_vars] = summed_vars_selected
            if self.inds_pieces_scramble is not None:
                self.best_inverted_periodic_scores_data[Search_results_index_slides] = [
                                                                                self.inds_pieces_scramble, self.inds_pieces_in_scrambled]
            elif self.time_slide_shifts is not None:
                self.best_inverted_periodic_scores_data[Search_results_index_slides] = [self.time_slide_shifts, times_quarters]
            self.best_inverted_periodic_scores_data[Search_results_index_per_list] = self.periods
            if save_to_file:
                filename_postfix = filename_postfix_inverted_period + str(self.Pmin) + '_' + str(self.Pmax)
                file_path = af.get_saving_file_path(output_location=path, target_id=self.target_id,
                                                    filename_postfix=filename_postfix, file_format='npy')
                np.save(file_path, np.array(self.best_periodic_scores_data, dtype=object), allow_pickle=True)

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
        """
        Get UNCORRECTED VALUES!!!
        :param in_non_corrected:
        :type in_non_corrected:
        :param transit_times:
        :type transit_times:
        :param template_ind:
        :type template_ind:
        :return:
        :rtype:
        """
        # TODO correct this!
        if in_non_corrected:
            assert self.stats_raw_no_veto_no_score_cor is not None
            stats_raw = self.stats_raw_no_veto_no_score_cor
        else:
            stats_raw = self.statistics_raw_cor
        if transit_times is None:
            transit_times = self.summed_transit_times
        if template_ind is None:
            template_ind = self.best_temp

        best_temps_sh = np.arange(self.tb.indices_of_templates[template_ind][0], self.tb.indices_of_templates[template_ind][-1])
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
                # TODO make sure about shifted/not
                best_raw_stats.append(stats_raw[best_temp_ind][best_quarter][best_time_ind])
                best_stat_denoms.append(self.stat_denoms[best_temp_ind][best_quarter])
                best_trends.append(self.trends_orig_fluxes[best_quarter][best_time_ind])
                best_templ_depths.append(np.max(np.abs(self.tb.templates_shifts[best_temp_ind])))
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
        # TODO DOOOOOOOOOOOOOO
        # TODO: MAKE SURE IT WAS NOT USED BECUASE IT USES WRONG STAT
        # TODO: get rid of the redundancies, because there is a similar function in target_class!
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

    :param statistics_by_trends :
    :type statistics_by_trends : ndarray 2d of float
    :param trends_sq_by_vars :
    :type trends_sq_by_vars : ndarray 2d of float
    :param dt : Time step of equally sampled time axis.
    :type dt : float
    :param tmax : Maximal time in the time axis.
    :type tmax : float
    :param tmin : Minimal time in the time axis.
    :type tmin : float
    :param periods:
    :type periods: ndarray 1d

    Produces:
    statistics_periodic : ndarray 3d
        Sum of the statistics for the entire transit chain. For all the templates
        with shifts, for all the periods and starting phases.
    statistics_periodic_counts : ndarray 3d
        Number of transits summed in each entry. For all the templates
        with shifts, for all the periods and starting phases.
    phases : numpy.array 3d
        Starting phases corresponding to `statistics_periodic`. For all the
        templates with shifts, for all the periods.
    lengths_filled : ndarray 2d
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
    statistics_periodic = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
    denomsq = np.zeros((len(periods), len(statistics_by_trends), phases_size), dtype=Dtype_r)
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
