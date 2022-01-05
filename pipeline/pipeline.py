#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:11:32 2020

@author: orynaivashtenko
version December 2021


"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import wget
import requests
import copy
import sys
import warnings
from scipy.signal import iirnotch
from scipy.signal import tf2sos
from scipy.signal import tf2zpk
from scipy.signal import sosfiltfilt
import importlib
import numba
import time

#PARAMETERS INITIALIZATION

#Masking:
nsig_outliers_0 = 5
nsig_outliers_0_pos = None #it is possible to mask positive and negatve outliers differently
nsig_outliers_0_neg = None
nsig_outliers_psd = None
max_small_hole = 3
nsig_outliers_w = 5
nsig_outliers_w_pos = None
nsig_outliers_w_neg = None
nsig_outliers_stat = 4.
#Masking. Detrending in outliers detection:
detr_length_outliers = 128
detr_method_outliers = 'median'
detr_endmethod_outliers = 'mean'
detr_ends_outliers = 'symmetric'
detr_endlen_outliers = None

#Measuring PSD:
nperseg_psd = 128
noverlap_psd = 0.5
window_psd = 'hann'
npsdsmin = 40
use_rfft = True
subtract_mean_in_psd = True

#Notch:
nsig_notch = 4.
irl_notch = 600.
htol_notch = 1e-5
minval_notch = 0

#Detrending original fluxes to find the trends:
detr_length = 800 #in order not to spoil even the largest transit
detr_method = 'median'
detr_endmethod = 'mean'
detr_ends = 'symmetric'
detr_endlen = detr_length//4
detr_return_trend_ = False
detr_keep_overall_level = False

#Templates and detection
template_bank_from_file = True
templates_subbin_shift = True
template_ind_detection = 'all'
fn_fit_veto=0.001
maxdev_fit_veto=4.0
zero_ends_len_abs_fit_veto=3
zero_ends_len_rel_fit_veto=0.1
Pmin=100 #days

#Processing the statistic
nperseg_moving_var = 512
#->make sure it is smallet than period
ngap_moving_var = 128
margin_snr_veto = 4.

duty_cycle = 0.1

Pmin = 70
Pmax = 400

track_process = False

data_location = '/Users/orynaivashtenko/Work/Kepler_Project/Data/llc_trial_2'
output_location = '/Users/orynaivashtenko/Work/Kepler_Project/Objects_output'
#TODO: write a function to save something there

# location = os.getcwd()
location = os.path.dirname(__file__)
sys.path.append(location)
parameters_file = os.path.join(location, 'pipeline_parameters.txt')
#PARAMETERS CAN BE SPECIFIED IN THE FILE, AND THEY WILL BE READ FROM IT
#variables_from_file(parameters_file)
template_module_location = os.path.join(location,'template_module')
template_bank_file = os.path.join(location, 'template_bank.txt')
shifts_template_bank_file = os.path.join(location, 'shifts_template_bank.txt')
import template_module as tm
importlib.reload(tm)

#TODO: update usege of these parameters in the code


class KeplerLightCurve(object):  
    def __init__(self, target_id=None, times=[], fluxes=[], get_data=True, data_location=data_location): #, load_data_from_file=False
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
            specified data_location. There should be a folder named as a 9-digit target_id.
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
                Filters to suppress oultiers in frequencies
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
        
        if target_id is not None:
            target_id = str(target_id).rjust(9, "0") #convert Kepler ID to string of appropriate length
        
        self.target_id = target_id
        if not get_data: 
            assert len(times)==len(fluxes), 'Number of time pieces is supposed to be equal to the number of light curves!'
            for i in range(len(times)):
                assert len(times[i])==len(fluxes[i]), 'Length of fluxes['+str(i)+'] does not match its time length!'
            self.orig_times = times 
            self.orig_fluxes = fluxes
        else:
            assert target_id is not None, 'Please specify target ID!'
            self.orig_times, self.orig_fluxes = get_data_fun(target_id, location=data_location)
        self.times = copy.deepcopy(self.orig_times)
        self.fluxes = copy.deepcopy(self.orig_fluxes)
        self.psd_fluxes = None #if we are working with very deep transits, we want to mask them for psd measuring but not to mask them completely
        self.quarters = range(len(self.fluxes))
        self.masks = [~np.isnan(self.fluxes[i]) for i in self.quarters]
        self.psd_masks = copy.deepcopy(self.masks)
        self.big_hole_masks = [np.ones(len(self.masks[q])).astype(bool) for q in self.quarters]
        self.notch_soss = []
        dts = []
        for q in self.quarters:
            dts.append(np.nanmean((self.times[q]-np.roll(self.times[q],1))[1:]))
        self.dt_days = np.nanmean(dts) #although it would be correct to weight by the lengths of the quarters
        self.tmin = np.nanmin(self.times[0])
        self.tmax = np.nanmax(self.times[0])
        for t in self.times[1:]:
            self.tmin = min(np.nanmin(t), self.tmin)
            self.tmax = max(np.nanmax(t), self.tmax)
        self.dt_sec = self.dt_days*24*60*60
        self.is_fluxes_conditioned=False
        self.is_zero_padded=False
        self.is_fluxes_whitened=False
        self.is_fluxes_inpainted = False
        self.is_holes_interpolated = False
        self.is_templates_initialized = False
        self.is_prepared_for_psd = False
        self.is_template_whitened = np.zeros(len(tm.template_bank())).astype(bool)
        self.is_statistics_zero_cut = False
        Tmax = np.max(np.array([len(t) for t in self.orig_times])) #maximum length of the time
        # the closest (but bigger) power of 2 to the length        Tmax += tm.L_ban
        self.support = 2**int(np.ceil(np.log2(Tmax))) 
        if use_rfft:
            self.fft_support = self.support//2+1
        else:
            self.fft_support = self.support
        
        
        self.inpaint_fluxes_fast = numba.njit(inpaint_fluxes_fun)
        # initialization of the numba function on some small data in order  not to
        # spend time on big data:
        _ = self.inpaint_fluxes_fast(fluxes = np.random.normal(0,1,(2,32)),
                                      masks = np.random.choice([0,1],(2,32)).astype(bool),
                                      wf_tds = np.array([np.concatenate((np.zeros(32//2-2),
                                        [-0.5,1,1,-0.5],np.zeros(32//2-2))) for i in range(2)]))
        self.sum_periods_fast = numba.njit(sum_periods_fun, parallel=False, fastmath=False)
        # initialization of the numba function on some small data in order  not to
        # spend time on big data:        
        _ = self.sum_periods_fast(statistics_by_trends=np.ones((2,192)), 
                                   trends_sq_by_vars=np.ones((2,192)) , dt=0.02, 
                                   tmax=3., tmin=0., periods=np.arange(1., 2., 0.5))
        

    
    def condition_data(self, nsig_outliers=nsig_outliers_0, nsig_outliers_0_pos=nsig_outliers_0_pos, nsig_outliers_0_neg=nsig_outliers_0_neg,
                       nsig_outliers_psd=nsig_outliers_psd, nperseg_psd=nperseg_psd, noverlap_psd=noverlap_psd, 
                       window_psd=window_psd, npsdsmin=npsdsmin, subtract_mean_in_psd=subtract_mean_in_psd, 
                       nsig_notch=nsig_notch, use_rfft=use_rfft, 
                       nsig_outliers_w=nsig_outliers_w, nsig_outliers_w_pos=nsig_outliers_w_pos, nsig_outliers_w_neg=nsig_outliers_w_neg,
                       max_small_hole=max_small_hole):
        """
        Preparing data for detection. Subtract mean, mask outliers, measure psds,
        padding by zeros, notching, inpainting, whitening.

        Parameters
        ----------
        nsig_outliers : float
            Minimal deviation from trend considered to be outlier (in units of std)
        nperseg_psd : int
            Length of slice used in measuring PSD
        noverlap_psd : int
            Length of slice overlapping when measuring PSD (if -1, will take half of `nperseg`)
        window_psd : str : 'hann', 'boxcar'
            Which window will be used in PSD measurement. It is recommended to use 'hann'
        npsdsmin : int
            Minimal number of slices required to measure PSD with Welch method
        subtract_mean_in_psd : bool
            Whether to subtract mean when measuring PSD
        nsig_notch : float
            Minimal deviation from trend of whitened fluxes in Fourier domain 
            considered to be outlier (in units of std of whitened fluxes in Fourier space)
        nsig_outliers_w : float
           Minimal deviation from trend of whitened fluxes in real domain 
           considered to be outlier (in units of their std)
        use_rfft : bool
            Whether to use rfft (Fourier for real signals) instead of fft

        Modifies
        -------
        TODO: write this. For now, please consult the "Short user documentation"

        """
        
        """
        Mean values of the flux are subtracted to have signal centered around zero. 
        Means are saved in self.means:
        """
        if not track_process:
            self.subtract_means(save_means=True)
            self.mask_flux_outliers(nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg, 
                                    nsig_psd=nsig_outliers_psd, modify_psd_masks=True)
            self.pad_by_zeros() 
            self.interpolate_fluxes()
            self.prepare_for_psd(nperseg=nperseg_psd, noverlap=noverlap_psd, 
                                 npsdsmin=npsdsmin, interp_method='linear', nsig_outliers_psd=nsig_outliers_psd)
            self.measure_psds(nperseg=nperseg_psd, noverlap=noverlap_psd, window=window_psd, subtract_mean=subtract_mean_in_psd, rfft=use_rfft)
            self.whiten_fluxes()
            self.mask_flux_outliers(nsig_outliers_w, nsig_positive=nsig_outliers_w_pos, nsig_negative=nsig_outliers_w_neg, in_whitened_fluxes=True, modify_psd_masks=False)
            self.interpolate_fluxes()
            self.whiten_fluxes()
            self.notch_fluxes(nsig=nsig_notch)
            self.mask_flux_outliers(nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg, modify_psd_masks=False,
                                    make_big_hole_masks=True, max_small_hole=max_small_hole)
            self.inpaint_fluxes() 
            self.whiten_fluxes()
            self.is_fluxes_conditioned=True
            
        #TODO: update the tracked functions
        else:    
            t1 = time.time()
            t0 = t1
            print('Subtracting means...', end=' ')
            self.subtract_means(save_means=True)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Values diverging from the large trend more than by 3 standard deviations are added to the mask 
            as False (before, mask marked bins containing NaN). psd_masks are also modified, because we don't 
            want outliers to take part in PSD measurement:
            """
            print('Masking outliers...', end=' ')
            self.mask_flux_outliers(nsig_outliers_0, modify_psd_masks=True)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Zero padding is needed to apply the whitening filter to avoid crosstalk between ends. 
            Applying filters in Fourier domain is equivalent to circular convolution, therefore 
            the ends of the data will influence each other which is unphysical. As well, at this step 
            the time array is interpolated and extended on the zero-padded region.
            """
            # zero padding is needed to apply the whitening filter to avoid crosstalk between ends.
            # When padding by zeros is done before the interpolation, in the interpolation the padded zeros become non-zeros, 
            # and due to this the overall trends will not probuce  outliers inside the data at the ends.
            # REM: the floating mean is subtracted with whitening
            print('Padding by zeros...', end=' ')
            self.pad_by_zeros() 
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Will fill all the bad data masked by mask with linear interpolation. Interpolation 
            is required for whitening, notching and other filters because filtering is making linear 
            combinations of points in the kernel, and adding bad data or NaN there will spoil all 
            the data. These points still remain masked for the PSD measurement if there is enough 
            slices. Otherwise, the masks will not be cleaned from small holes. Therefore data in maskes
            points will be used only if necessary. Remark: when padding by zeros is done before the 
            interpolation, in the interpolation the padded zeros become constant values equal to the 
            last edge data value, and due to this outliers will not be produced in the data at the ends:
            """
            # Interpolation is required for whitening, notching etc.
            # But in the preparation for PSD, the masks will not be cleaned if not needed,
            # therefore data in holes will be used only if necessary
            print('Interpolating fluxes...', end=' ')
            self.interpolate_fluxes()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            In the preparation for PSD, the code is making sure that there is enough slices (npsdsmin) 
            to ensure the needed precision in PSD measurement. If there is not enough, 
            clean_small_holes() is used to remove small holes from the PSD mask. They are not 
            significant, so we easily can take interpolated values in them:
            """
            #TODO: maybe define two levels of small holes and two npsds
            #TODO: maybe define better distribution of pieces in Welch
            print('Preparing fpr PSD...', end=' ')
            self.prepare_for_psd(nperseg=nperseg_psd, noverlap=noverlap_psd, 
                                 npsdsmin=npsdsmin, interp_method='linear')
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Use Welch method to measure Power Spectral Density. Split data into multiple slices 
            of length nperseg_psd, calculate their abs squared of Fourier transform, and average. 
            The resulting array will have length nperseg_psd, therefore for making whitening filters, 
            (psd)^{-0.5} is converted to time domain and padded by zeros (finite response filter). 
            PSD and whitening filters are quarter-specific:
            """
            print('Measuring PSD...', end=' ')
            self.measure_psds(nperseg=nperseg_psd, noverlap=noverlap_psd, window=window_psd, subtract_mean=subtract_mean_in_psd, rfft=use_rfft)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Whitening filters are applied to the fluxes to create whitened fluxes. (Remark: the 
            floating mean is subtracted with whitening):
            """
            print('Whitening fluxes...', end=' ')
            self.whiten_fluxes()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Masking outliers in whitened fluxes. Effectively we are making detection in the whitened
            data with whitened template, so outliers are unwanted there:
            """
            print('Masking outliers...', end=' ')
            self.mask_flux_outliers(nsig_outliers_w, in_whitened_fluxes=True, modify_psd_masks=True)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Interpolation to cover also new outliers. Note that outliers were masked in the whitened 
            flux, but interpolation is made in the red flux. It is possible to make interpolation in 
            the whitened flux and then redden it, but the inverse whitening filter is unstable, so 
            the red flux changes significantly. As well, in most cases interpolation in the red flux 
            indeed fixes the outliers because the corresponding point in the red flux has the biggest 
            impact on the outlier:
            """
            # Before notch need to interpolate over mask because holes will not allow to do whitening which is needed in notch,
            # and need to freshly whiten the fluxes because need whitened_fd for notch
            print('Interploating fluxes...', end=' ')
            self.interpolate_fluxes()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Whiten the interpolated fluxes again before applying the Notch filter because in it, 
            the whitened fluxes in Fourier domain are needed:
            """
            print('Whitening fluxes...', end=' ')
            self.whiten_fluxes()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            In whitened_fluxes_fd, common peaks in all the quarters are searched, and Notch filters 
            are designed to suppressed them. They are applied to the fluxes:
            """
            print('Notching fluxes...', end=' ')
            self.notch_fluxes(nsig=nsig_notch)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            #There is an option to remeasure PSD after notch. Usually, it leaves wider notched gaps in the whitened flux,
            # but removes remnant excess in power around peaks.
            """
            After notching, some outliers can appear, they are also added to the mask: 
            """
            print('Masking outliers...', end=' ')
            self.mask_flux_outliers(nsig_outliers_0, modify_psd_masks=True)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            All the bad points ever masked are filled using the optimal inpainting filter which zeroes 
            the impact of these point to the statistic. Note: PSD should not be remeasured after 
            inpainting because the same C^{-1} has to be used for inpainting and for detection: 
            """
            print('Inpainting...', end=' ')
            self.inpaint_fluxes() 
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            These fluxes will be used for detection:
            """
            print('Whitening...', end=' ')
            self.whiten_fluxes()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Total time: ', t2-t0, ' sec')
            self.is_fluxes_conditioned=True
            _ = os.system('clear')


    def detect_single_transits(self, template_ind=template_ind_detection, use_rfft=use_rfft,
                         nsig_outliers=nsig_outliers_0, margin_snr=margin_snr_veto, 
                         fn=fn_fit_veto, maxdev=maxdev_fit_veto,
                         zero_ends_len_abs=zero_ends_len_abs_fit_veto,
                         zero_ends_len_rel=zero_ends_len_rel_fit_veto):
        """
        TODO: write this. For now, please consult descriptions of all the methods, as well as "Short user documentation"

        Parameters
        ----------
        template_ind : TYPE, optional
            DESCRIPTION. The default is template_ind_detection.
        use_rfft : TYPE, optional
            DESCRIPTION. The default is use_rfft.
        nsig_outliers : TYPE, optional
            DESCRIPTION. The default is nsig_outliers_0.
        margin_snr : TYPE, optional
            DESCRIPTION. The default is margin_snr_veto.
        fn : TYPE, optional
            DESCRIPTION. The default is fn_fit_veto.
        maxdev : TYPE, optional
            DESCRIPTION. The default is maxdev_fit_veto.
        zero_ends_len_abs : TYPE, optional
            DESCRIPTION. The default is zero_ends_len_abs_fit_veto.
        zero_ends_len_rel : TYPE, optional
            DESCRIPTION. The default is zero_ends_len_rel_fit_veto.

        Returns
        -------
        None.

        """
        if not track_process:
            self.initialize_templates(templates=None, template_ind=template_ind, template_bank_from_file=True,
                                      templates_subbin_shift=templates_subbin_shift)
            self.notch_all_templates()
            self.roll_templates()
            self.whiten_all_templates(use_rfft=use_rfft)
            self.matched_filtering(template_ind=template_ind)
            self.cut_zero_padding_ends()
            self.veto_statistics(margin_snr=margin_snr, template_ind=template_ind,
                                 fn=fn, maxdev=maxdev,
                             rfft_psd=use_rfft, zero_ends_len_abs=zero_ends_len_abs,
                             zero_ends_len_rel=zero_ends_len_rel)
          
        else:    
            t1 = time.time()
            t0 = t1
            print('Initializing templates...', end=' ')
            self.initialize_templates(templates=None, template_ind=template_ind, template_bank_from_file=True,
                                      templates_subbin_shift=templates_subbin_shift)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Notching the templates...', end=' ')
            self.notch_all_templates()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Whitening the templates...', end=' ')
            self.roll_templates()
            self.whiten_all_templates(use_rfft=use_rfft)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Matched filtering...', end=' ')
            self.matched_filtering(template_ind=template_ind)
            self.cut_zero_padding_ends()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Vetoing the statistics...', end=' ')
            self.veto_statistics(margin_snr=margin_snr, template_ind=template_ind,
                                 fn=fn, maxdev=maxdev,
                             rfft_psd=use_rfft, zero_ends_len_abs=zero_ends_len_abs,
                             zero_ends_len_rel=zero_ends_len_rel)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Total time: ', t2-t0, ' sec')
            _ = os.system('clear')


    def periodicity_search(self, Pmin=Pmin, Pmax=Pmax, dP=None):
        """
        TODO: write this. For now, I hope that other users than me will not need this method.

        Parameters
        ----------
        Pmin : TYPE, optional
            DESCRIPTION. The default is Pmin.
        Pmax : TYPE, optional
            DESCRIPTION. The default is Pmax.
        dP : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if not track_process:
            self.find_moving_averages(interpolate=True, detr_length=detr_length, detr_method=detr_method, 
                                 detr_endmethod=detr_endmethod, detr_ends=detr_ends, 
                                 detr_endlen=detr_endlen, keep_overall_level=False)
            self.connect_quarters()
            self.interleave_shifts()
            self.sum_periods(Pmin=Pmin, Pmax=Pmax, duty_cycle=duty_cycle)
            
        else:
            t1 = time.time()
            t0 = t1
            print('Detrending original fluxes...', end=' ')
            self.find_moving_averages(interpolate=True, detr_length=detr_length, detr_method=detr_method, 
                                 detr_endmethod=detr_endmethod, detr_ends=detr_ends, 
                                 detr_endlen=detr_endlen, keep_overall_level=False)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Connecting quarters...', end=' ')
            self.connect_quarters()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Interwining shifts...', end=' ')
            self.interleave_shifts()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Summing over periods...', end=' ')
            self.sum_periods(Pmin=Pmin, Pmax=Pmax, duty_cycle=duty_cycle)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Total time: ', t2-t0, ' sec')
            _ = os.system('clear')
        

#TODO: make sure you remember that function calls on np.arrays create no copy

    def subtract_means(self, save_means=True):
        """
        Subtracts mean value from each of self.fluxes
        
        Parameters
        ----------
        save_means : bool
            Whether to create a new attribure self.means and save the mean
            values of the fluxes there. The default is True.

        Modifies
        -------
        self.fluxes

        """
        if save_means:
            self.means = []
        for i in self.quarters: 
            mean = np.nanmean(self.fluxes[i])
            if save_means:
                self.means.append(mean)
            self.fluxes[i] = np.array(self.fluxes[i]-mean)

    
    def mask_flux_outliers(self, nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg, 
                           nsig_psd=nsig_outliers_psd,
                           in_whitened_fluxes=False, modify_psd_masks=True, 
                           make_big_hole_masks=False, max_small_hole=max_small_hole):
        """
        Detects outliers of the flux with respect to the trend. The trend is found by detrending.
        Modifies self.masks by setting False where the outliers were found.
        
        Parameters
        ----------
        nsig : float
            Minimal deviation from the trend considered as outlier
        in_whitened_fluxes : bool
            If true, the search will be performad not in self.fluxes but
            in self.whitened_fluxes_td
        modify_psd_masks : bool
            If True, self.psd_masks will be modified setting False to the 
            outliers positions. If False, only self.masks will be modified

        Modifies
        -------
        self.masks
        self.psd_masks

        """
        if not in_whitened_fluxes:
            for i in self.quarters: 
                flux=np.array(self.fluxes[i])
                flux[np.where(~self.masks[i])[0]] = np.nanmean(self.fluxes[i])
                outlier_mask = mask_outliers_fun(flux, mask=np.array(self.masks[i]), nsig=nsig, 
                                                 nsig_positive=nsig_positive, nsig_negative=nsig_negative,
                                                 detr_length=detr_length_outliers,
                                                 detr_method=detr_method_outliers, detr_endmethod=detr_endmethod_outliers,
                                                 detr_ends=detr_ends_outliers, detr_endlen=detr_endlen_outliers)
                self.masks[i] *=  outlier_mask
                if modify_psd_masks:
                    if nsig_psd is not None:
                        outlier_psd_mask = mask_outliers_fun(flux, mask=np.array(self.masks[i]), nsig=nsig_psd, 
                                                 nsig_positive=None, nsig_negative=None,
                                                 detr_length=detr_length_outliers,
                                                 detr_method=detr_method_outliers, detr_endmethod=detr_endmethod_outliers,
                                                 detr_ends=detr_ends_outliers, detr_endlen=detr_endlen_outliers)
                        self.psd_masks[i] *=  outlier_psd_mask
                    else:
                        self.psd_masks[i] *=  outlier_mask
        else:
            if not self.is_fluxes_whitened:
                self.whiten_fluxes()
            for i in self.quarters: 
                wflux=copy.deepcopy(self.whitened_fluxes_td[i])
                #need to calculate mean and variance in not masked region, otherwise they will be too small:
                indices = np.where(self.masks[i])[0]
                meanwflux = np.mean(wflux[indices])
                stdwflux = np.std(wflux[indices])
                if nsig_positive is None:
                    nsig_positive = nsig
                if nsig_negative is None:
                    nsig_negative = nsig
                w_outlier_mask = ((wflux-meanwflux) < (nsig_positive * stdwflux))
                w_outlier_mask *= ((wflux-meanwflux) > (- nsig_negative * stdwflux))
                self.masks[i] *=  w_outlier_mask
                if modify_psd_masks:
                    if nsig_psd is None:
                        self.psd_masks[i] *=  w_outlier_mask
                    else:
                        self.psd_masks[i] *= (np.abs(wflux-meanwflux) < nsig_psd * stdwflux)
        # if make_big_hole_masks:
            # self.big_hole_masks = []
            # for q in self.quarters:
            #     self.big_hole_masks.append(clean_small_holes(np.array(self.masks[q]), max_hole=max_small_hole))
                
    
    #TODO: mask_planet
    def mask_planet(self, planet_ephemeris_params):
        # import template_module
        # flux_par
        pass
    
    def prepare_for_psd(self, nperseg=nperseg_psd, noverlap=noverlap_psd, npsdsmin=npsdsmin, interp_method='linear', 
                        nsig_outliers_psd=nsig_outliers_psd, max_small_hole=max_small_hole):
        """
        Checks whether the number of pieces for the Welch method is enough compare to npsdsmin, and 
        if no, interpolates over mask and updates the psd mask to measure psd for the first time like that.

        Parameters
        ----------
        nperseg : int
            length of slices in Welch method
        noverlap : int or float<1
            If int: length of overlap of slices in Welch method. 
            If float<1 : relative overlap to nperseg.
            Length of overlap will be int(nperseg*noverlap) 
        npsdsmin : int
            minimum number of slices to measure psd with Welch. If the actual number is smaller, interpolation will be made.
            
        Creates    
        ----------
        self.nperseg
        self.big_hole_masks
        
        Modifies
        ----------
        self.fluxes (-> interpolated fluxes)
        self.psd_masks

        """
        if nsig_outliers_psd is not None:
            self.psd_fluxes = copy.deepcopy(self.fluxes)
            for i in self.quarters:
                self.psd_fluxes[i] = interpolate_over_mask(self.psd_fluxes[i], self.psd_masks[i], interp_method='linear', ends='continue',
                                                                mitigate_hole_ends=False, endhole_method='median')
            
        elif not self.is_holes_interpolated:
            warnings.warn("Data is not interpolated. Cleaning small holes may cause NaN in the fluxes.")
        if noverlap<1:
            noverlap = int(nperseg*noverlap)
        if noverlap==-1: #an old piece of code for the case is we still use notation of noverlap=-1
            noverlap=nperseg//2
        # Calculating the number of pieces available for psd measurement. If this number is too low, the interpolation
        # over holes will be done first
        self.npsds0 = np.zeros(len(self.fluxes))
        for i in self.quarters:
            mask_big = clean_small_holes(np.array(self.psd_masks[i]), max_hole=max_small_hole) #mask of only big and edge holes
            for ind1 in range(0,len(self.fluxes[i]) - nperseg, nperseg - noverlap):
                if np.sum((~self.psd_masks[i])[ind1 : ind1 + nperseg])==0:
                    self.npsds0[i] += 1
            if self.npsds0[i]<npsdsmin: #if the number of psd samples is insufficient, first try to interpolate small holes
                npsds1 = 0
                #new number of psd samples will be:
                for ind1 in range(0,len(self.fluxes[i]) - nperseg, nperseg - noverlap):
                    if np.sum((~mask_big)[ind1 : ind1 + nperseg])==0:
                        npsds1 += 1
                if npsds1>=npsdsmin: #if it is sufficient, go with it
                    self.psd_masks[i]=mask_big
                else:#if the new number is still not sufficient, then interpolate all:
                    if self.is_zero_padded:
                        self.psd_masks[i] = np.concatenate((np.zeros((self.zero_padding_ends[i])[0]), 
                                            np.ones(len(self.orig_fluxes[i])),
                                            np.zeros((self.zero_padding_ends[i])[1]))).astype(bool)
                    else:
                        self.psd_masks[i]=np.ones(len(self.psd_masks[i])).astype(bool)                       
                #in psd_masks masks, inner holes are interpolated to increase the number of psd samples
        self.is_prepared_for_psd = True
                    
    
    def pad_by_zeros(self, use_rfft=use_rfft):
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
                zerlen1 = (self.support-len(self.fluxes[i]))//2
                zerlen2 = (self.support-len(self.fluxes[i])+1)//2
                self.zero_padding_ends.append((zerlen1, zerlen2))
                self.fluxes[i] = np.concatenate((np.zeros(zerlen1), 
                                                  self.fluxes[i], np.zeros(zerlen2) ))
                self.masks[i] = np.concatenate((np.zeros(zerlen1), 
                                                  self.masks[i], np.zeros(zerlen2))).astype(bool)
                self.psd_masks[i] = np.concatenate((np.zeros(zerlen1), 
                                                  self.psd_masks[i], np.zeros(zerlen2))).astype(bool)
                # self.big_hole_masks[i] = np.concatenate((np.zeros(zerlen1), 
                #                                   self.big_hole_masks[i], np.zeros(zerlen2))).astype(bool)
                self.times[i] = np.concatenate((np.zeros(zerlen1)+np.nan, 
                                                  self.times[i], np.zeros(zerlen2)+np.nan ))
                # NaNs are needed for the 'interpolate_time" function to treat these regions
                # as the ones to be interpolated. The self.masks[i] is not transferred to the
                # time interpolation because not all the masked flux regions are NaNs in time
                # and we do not want to lose precision rejecting data.
                self.times[i] = interpolate_time(np.array(self.times[i]), mask=None)
            self.fluxes = np.array(self.fluxes) #need them to be numpy.arrays for numba
            self.masks = np.array(self.masks)
                
            self.is_zero_padded = True
    
    
    def measure_psds(self, nperseg=nperseg_psd, noverlap=noverlap_psd, window=window_psd, subtract_mean=subtract_mean_in_psd, rfft=use_rfft):
        
        """
        PSDS as function of quarter id.
        The PSD measuring algotithm makes the evenly sampled steps and not adjust the search for
        good pieces according to the placement of the zeros in mask. That is why
        the number of npsds is unoptimal.
        
        Parameters
        -------------------
        nperseg : int
        the length of a sample used in Welch method
        Default is 64
        
        noverlap : int or float<1
                If int: length of overlap of slices in Welch method. 
                If float<1 : relative overlap to nperseg.
                Length of overlap will be int(nperseg*noverlap) 
            
        window : string
            window function by which the signals are multiplied to avoid boundary effects and crosstalk
            Possible options: 'hann', 'boxcar', where 'boxcar' is just normalized bow, or no window
            Default is 'hann'. It is not recommended to use 'boxcar'.
            
        subtract_mean : bool
            If True, the mean of the signal is subtracted. It is recommended to do it because
            otherwise unwanted power may appear in the zero frequency
            Default is True
        
        rfft : bool
            If True, the signal is treated as real, and all ffts are rffts. The length of the frequency samples
            are the nperseg//2+1.
            Default is True
        
        Creates
        ----------------
        self.psds
        self.psdfreqs
        self.wf_td
        self.wf_fd
        self.freqs
        seld.npsds
        self.nperseg
        """
        if not self.is_prepared_for_psd:
            warnings.warn("Data is not prepared for PSD. Use prepare_for_psd method.")
            
        if self.psd_fluxes is None:
            self.psd_fluxes = copy.deepcopy(self.fluxes)
            
        if self.is_fluxes_inpainted:
            warnings.warn("Fluxes have been inpainted, new PSD will make inpainting invalid for detection.")
        
        if noverlap<1:
            noverlap = int(nperseg * noverlap)
        if noverlap==-1:
            noverlap=nperseg//2
            
        windowdict = {'boxcar' : np.ones(nperseg)/np.linalg.norm(np.ones(nperseg)), 
            'hann' : 0.5 * (1-np.cos(2*np.pi*np.arange(nperseg)/(nperseg)))/np.linalg.norm(
                0.5 * (1-np.cos(2*np.pi*np.arange(nperseg)/(nperseg))))}
        
        psds = []
        
        if subtract_mean:            
            if rfft:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.rfft(windowdict[window]*(self.psd_fluxes[i]-np.nanmean(self.psd_fluxes[i]))[ind1:ind1+nperseg])
                        )**2 for ind1 in range(0,len(self.psd_fluxes[i])-nperseg,nperseg-noverlap
                                ) if np.sum((~self.psd_masks[i])[ind1:ind1+nperseg])==0])
                self.freqs = [np.fft.rfftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.rfftfreq(nperseg, d=self.dt_sec) for i in self.quarters]
            else:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.fft(windowdict[window]*(self.psd_fluxes[i]-np.nanmean(self.psd_fluxes[i]))[ind1:ind1+nperseg])
                        )**2 for ind1 in range(0,len(self.psd_fluxes[i])-nperseg,nperseg-noverlap
                                ) if np.sum((~self.psd_masks[i])[ind1:ind1+nperseg])==0])
                self.freqs = [np.fft.fftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.fftfreq(nperseg, d=self.dt_sec) for i in self.quarters]
        
        else:
            if rfft:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.rfft(windowdict[window]*(self.psd_fluxes[i])[ind1:ind1+nperseg])
                        )**2 for ind1 in range(0,len(self.psd_fluxes[i])-nperseg,nperseg-noverlap
                                ) if np.sum((~self.psd_masks[i])[ind1:ind1+nperseg])==0])
                self.freqs = [np.fft.rfftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.rfftfreq(nperseg, d=self.dt_sec) for i in self.quarters]
            else:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.fft(windowdict[window]*(self.psd_fluxes[i])[ind1:ind1+nperseg])
                        )**2 for ind1 in range(0,len(self.psd_fluxes[i])-nperseg,nperseg-noverlap
                                ) if np.sum((~self.psd_masks[i])[ind1:ind1+nperseg])==0])
                self.freqs = [np.fft.fftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.fftfreq(nperseg, d=self.dt_sec) for i in self.quarters]
        
        
        self.npsds = np.array([len(p) for p in psds]) #number of psds used to take the average
        self.psds = [np.mean(p, axis=0) for p in psds]
        # self.psds = [np.interp(self.freqs[i], psdfreqs, self.psds[i]) for i in self.quarters]
        self.nperseg = nperseg
        
        self.wf_td = []
        self.wf_fd = []
        for i in self.quarters:
            if self.support is not None:
                wf_td, wf_fd = psd2wf(self.psds[i], self.support, rfft=rfft)
            else:
                wf_td, wf_fd = psd2wf(self.psds[i], len(self.fluxes[i]), rfft=rfft)
            self.wf_td.append(wf_td)
            self.wf_fd.append(wf_fd)
        self.wf_td = np.array(self.wf_td) #for numba need numpy.arrays
    

    def interpolate_fluxes(self):
        '''
        Interpolates all the holes in the holes in the fluxes, including big holes and end holes.
        In the end holes, the fluxes are padded by a constant value equal to the first(last) value of the array.
        Masks are not modified because measuring PSD or detecting with this interpolation makes no sense.
        '''
        for i in self.quarters:
            self.fluxes[i] = interpolate_over_mask(np.array(self.fluxes[i]), self.masks[i], interp_method='linear', ends='continue',
                                                            mitigate_hole_ends=False, endhole_method='median')
            self.is_holes_interpolated = True
        
    
    def notch_fluxes(self, nsig=nsig_notch, irl_notch=irl_notch, htol_notch=htol_notch, minval_notch=minval_notch):
        """
        Takes the whitened fluxes in Fourier domain, looks at the frequencies
        in all the quarters that peak more than `nsig`, designs Notch filter
        to suppress these frequencies and applies this filter to fluxes.

        Parameters
        ----------
        nsig : float
            minimal outlier amplitude in units of the std of the whitened flux in Fourier domain
        irl_notch : float
            Impulse response length, samples. Length in time domain where the response of 
            the filter is greater than `htol`.
        htol_notch : float
            Tolerance parameter for the filter decay. The number considered as zero (or the end) 
            of the filter in time domain.

        Returns
        -------
        None.
            Modifies self.fluxes, creates self.notch_soss
        """
        if not self.is_fluxes_whitened:
            self.whiten_fluxes()
        # sumflux=np.sum(self.whitened_fluxes_fd, axis=0) #add all the quarters to mark repeated spurious frequencies
        # mask1 = (np.abs(sumflux-np.nanmean(sumflux)) < nsig * np.nanstd(sumflux)).astype(float)
        # sumflux=np.sum(np.abs(self.whitened_fluxes_fd), axis=0)
        # mask2 = (np.abs(sumflux-np.nanmean(sumflux)) < nsig * np.nanstd(sumflux)).astype(float)
        sumflux=np.sum(self.whitened_fluxes_fd, axis=0) #add all the quarters to mark repeated spurious frequencies
        sumabs=np.sum(np.abs(self.whitened_fluxes_fd), axis=0)
        mask1 = ((np.abs(sumflux)-np.abs(np.nanmean(sumflux))) < nsig * np.nanstd(sumflux)).astype(float)
        mask2 = ((sumabs-np.nanmean(sumabs)) < nsig * np.nanstd(sumabs)).astype(float)
        inv_mask = (1 - mask1 * mask2).astype(bool)
        notch_soss=[] #make a filter for every peak to notch
        for fr in np.where(inv_mask)[0]:
            if not np.isclose(fr,0):
                notch_soss.append(notch_fun(fr, fs=self.support, irl=irl_notch, htol=htol_notch, min_val=minval_notch))
        
        for i in self.quarters: #apply all the filters for all the quarters
            for sos in notch_soss:
                self.fluxes[i]=sosfiltfilt(sos, np.array(self.fluxes[i]), padtype='constant') 
                # very important to have zeros in the ends, bacause of padding.
                # padtype 'constant' will pad by the edge value, and we need to ensure it is zero
        self.notch_mask = np.array(~inv_mask)        
        self.is_fluxes_whitened = False #because these whitened fluxes are not valid anymore, we need to remeasure them
        self.notch_soss = notch_soss
         

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
        if not self.is_fluxes_whitened:
            self.whiten_fluxes()
        # if np.array([np.isnan(self.fluxes[q]).any() for q in self.quarters]).any():
        #     self.interpolate_fluxes()
        self.fluxes = self.inpaint_fluxes_fast(self.fluxes, self.masks, self.wf_td)
        self.is_fluxes_inpainted = True
    
    
    def whiten_fluxes(self):
        """
        whitens all the fluxes with whitening filter
        """
        assert not (np.array([np.isnan(self.fluxes[i]
                        ).any() for i in self.quarters])).any(), 'there are holes in fluxes, cannot whiten!'
        
        self.whitened_fluxes_fd = []
        self.whitened_fluxes_td = []
        for i in self.quarters:
            whitened_flux_fd = np.fft.rfft(self.fluxes[i])*self.wf_fd[i]
            self.whitened_fluxes_fd.append(whitened_flux_fd)
            self.whitened_fluxes_td.append(np.fft.irfft(whitened_flux_fd))
        self.is_fluxes_whitened = True
    
    
    
    def detrend_fluxes(self, interpolate=True, detr_length=detr_length, detr_method=detr_method, 
                             detr_endmethod=detr_endmethod, detr_ends=detr_ends, 
                             detr_endlen=detr_endlen, keep_overall_level=False):
        """
        Calculates the large trends of the fluxes by taking the moving average of
        type detr_method ('mean' or 'median') of slices of size detr_length.
        Saves the trends (quarter-specific) and saves the detrended fluxes separately.

        Parameters
        ----------
        interpolate : bool
            Whether to interpolate the data in the holes containing numpy.nan.
            The interpolation will be linear, the holes in the ends of the data will 
            be filled by a constant.
        detr_length : int
            Detrending length. Size of a slice to take the moving average
        detr_method : str : 'mean' or 'median'
            The type of moving average to take. Median is less sensitive to
            outliers, and mean is less sensitive to small number of samples.
            Median is preferred in the middle of the data.
        detr_endmethod : str : 'mean' or 'median'
            The type of moving average to take in the ends of the data.
            Mean is less sensitive to small number of samples, so it is preferred
            in the ends of the data.
        detr_ends : str : 'symmetric' or 'asymmetric' or 'cut'
            How to make slices on the ends of the data. Read details in detrend_amat().
            'symmetric' and 'asymmetric' preserve the size of the data,
            'cut' decreases the size of the data by detr_length.
        detr_endlen : int or None
            Minimal slice taken in the end of the data. If None, will be taken 
            as detr_len//4
        keep_overall_level : bool
            Whether to preserve the overall average (mean or median, defined by 
            detr_method) of the data
        Returns
        -------
        None.
            Creates self.detrended_fluxes and self.trends_fluxes

        """
        self.detrended_fluxes = []
        self.trends_fluxes = []
        for q in self.quarters:
            if interpolate:
                msk = ~np.isnan(self.fluxes[q])
                flx = interpolate_over_mask(np.array(self.fluxes[q]), msk, interp_method='linear', ends='continue',
                                                            mitigate_hole_ends=False, endhole_method='median')
                detr, trend = detrend_amat(np.array(flx), length=detr_length, mask=msk, method=detr_method, 
                             endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen, return_trend=True, keep_overall_level=False)
            else:
                detr, trend = detrend_amat(np.array(self.fluxes[q]), length=detr_length, mask=None, method=detr_method, 
                             endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen, return_trend=True, keep_overall_level=False)
            self.detrended_fluxes.append(detr)
            self.trends_fluxes.append(trend)
    
    
    def find_moving_averages(self, interpolate=True, detr_length=detr_length, detr_method=detr_method, 
                             detr_endmethod=detr_endmethod, detr_ends=detr_ends, 
                             detr_endlen=detr_endlen, keep_overall_level=False,
                             remove_outliers=False, nsig_outliers=nsig_outliers_0):
        """
        Calculates the large trends of the original fluxes by taking the moving average of
        type detr_method ('mean' or 'median') of slices of size detr_length.
        Saves the trends (quarter-specific) and saves the detrended original fluxes separately.

        Parameters
        ----------
        interpolate : bool
            Whether to interpolate the data in the holes containing numpy.nan.
            The interpolation will be linear, the holes in the ends of the data will 
            be filled by a constant.
        detr_length : int
            Detrending length. Size of a slice to take the moving average
        detr_method : str : 'mean' or 'median'
            The type of moving average to take. Median is less sensitive to
            outliers, and mean is less sensitive to small number of samples.
            Median is preferred in the middle of the data.
        detr_endmethod : str : 'mean' or 'median'
            The type of moving average to take in the ends of the data.
            Mean is less sensitive to small number of samples, so it is preferred
            in the ends of the data.
        detr_ends : str : 'symmetric' or 'asymmetric' or 'cut'
            How to make slices on the ends of the data. Read details in detrend_amat().
            'symmetric' and 'asymmetric' preserve the size of the data,
            'cut' decreases the size of the data by detr_length.
        detr_endlen : int or None
            Minimal slice taken in the end of the data. If None, will be taken 
            as detr_len//4
        keep_overall_level : bool
            Whether to preserve the overall average (mean or median, defined by 
            detr_method) of the data
        Returns
        -------
        None.
            Creates self.detrended_orig_fluxes and self.trends_orig_fluxes

        """        
        self.detrended_orig_fluxes = []
        self.trends_orig_fluxes = []
        for q in self.quarters:
            if interpolate:
                msk = ~np.isnan(self.orig_fluxes[q])
                o_f = interpolate_over_mask(np.array(self.orig_fluxes[q]), msk, interp_method='linear', ends='continue',
                                                            mitigate_hole_ends=False, endhole_method='median')
                detr, trend = detrend_amat(np.array(o_f), length=detr_length, mask=None, method=detr_method, 
                             endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen, return_trend=True, 
                             keep_overall_level=False, remove_outliers=remove_outliers, nsig_outliers=nsig_outliers)
            else:
                detr, trend = detrend_amat(np.array(self.orig_fluxes[q]), length=detr_length, mask=None, method=detr_method, 
                             endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen, return_trend=True, 
                             keep_overall_level=False, remove_outliers=remove_outliers, nsig_outliers=nsig_outliers)
            self.detrended_orig_fluxes.append(detr)
            self.trends_orig_fluxes.append(trend)
            
            
    
    def initialize_templates(self, templates=None, template_ind='all', 
                             templates_subbin_shift=templates_subbin_shift,
                             template_bank_from_file=template_bank_from_file):
        """
        Creates the template of the appropriate len by zero padding of the
        input template. If template is None, an entry from the template bank
        template_module.template_bank[template_ind] will be used. If
        template_ind=='all', the method will operate on the entire template_bank.

        Parameters
        ----------
        templates : list of numpy.array 1d of float
            Patterns to be used as a template (time domain). Should be
            all of the same length. The working part of the template is 
            supposed to be in the center (the transit peak is in the center of
            the array). The ends will be padded by zeros to match self.support. 
        template_ind : int, <len(template_module.template_bank())
            Index of the template in template_module.template_bank() to be used
            if templates==None. If template_ind=='all', all the template bank will be used.

        Creates
        -------
        self.initial_templates : list of numpy.array 1d
            Templates as they were in the template bank
        self.templates : list of numpy.array 1d
            Templates padded to length self.support.
        self.template_len : int
            Size of the initial templates

        """
        #TODO: care about different shifts if needed
        if templates is not None:
            self.initial_templates = templates
        elif template_bank_from_file:
            self.initial_templates = read_quarters_from_file(template_bank_file, dimension=3)
            self.shifts = read_quarters_from_file(shifts_template_bank_file, dimension=2)
        elif template_ind=='all':
            if templates_subbin_shift:
                self.initial_templates, self.shifts = tm.template_bank_subbin_shift(dt=self.dt_days,
                                          dP=None, T_mission=self.tmax-self.tmin, Pmin=Pmin,
                                          n_points_int_min=50, return_ordered=True, shifts_num=3,
                                          return_shifts=True)
                # self.template_transit_lengths = [tm.transit_lengths(tem_shifts) 
                                                 # for tem_shifts in self.initial_templates]
            else:
                self.initial_templates = tm.template_bank(dt=self.dt_days)
                self.initial_templates = [template for template in self.initial_templates]
                self.shifts = [[0.] for i in range(len(self.initial_templates))]
                # self.template_transit_lengths = tm.transit_lengths(self.initial_templates)
        else:
            assert type(template_ind)==int, "The template_ind should be int or 'all'!"
            if templates_subbin_shift:
                initial_templates, shifts = tm.template_bank_subbin_shift(dt=self.dt_days,
                                          dP=None, T_mission=self.tmax-self.tmin, Pmin=Pmin,
                                          n_points_int_min=50, return_ordered=True, shifts_num=3,
                                          return_shifts=True)
                self.initial_templates[template_ind] = initial_templates[template_ind]
                self.shifts[template_ind] = shifts[template_ind]
            else:
                self.initial_templates = [tm.template_bank(dt=self.dt_days)[template_ind]]
                self.shifts[template_ind] = [0.]
        indlen = list(np.cumsum([len(s) for s in self.shifts]))
        #pairs of indices denoting the beginning and the end of different shifts of one template 
        self.indices_of_templates = np.array(list(zip(np.concatenate(([0],indlen)), indlen))) 
        self.shifts = np.hstack(self.shifts)
        self.template_len = len(self.initial_templates[0][0])
        self.templates = np.zeros((len(self.shifts), self.support))
        initial_templates_array = np.zeros((len(self.shifts), self.template_len))
        index = 0
        for template_with_shifts in self.initial_templates:
            for template in template_with_shifts:
                temp = np.concatenate((np.zeros((self.support-len(template))//2),
                                        template,
                                       np.zeros((self.support-len(template)+1)//2)))
                self.templates[index] = np.array(temp)
                initial_templates_array[index] = np.array(template)
                index += 1
        self.initial_templates = initial_templates_array
        self.is_templates_initialized = True
      
    
    def notch_template(self, template=None, template_ind=20):
        """
        Applies notch filter to the template. That is, suppresses some frequencies
        which were marked before in the data conditioning part. Is relevant for 
        detection only if used for the data as well and if the psds were not
        remeasured after notching. If template==None takes the 
        template_module.template_bank()[template_ind], prepares it (adjusting
        length to self.support and shifting) and whitens. 

        Parameters
        ----------
        template : numpy.array 1d of float
            The template to be whitened. If not specified,
            the template_module.template_bank[template_number] will be used. 
        template_ind : int
            Index of the template from template_module.template_bank() to be
            whitened. Will be used only if template==None.
        
        Requires
        ----------
        self.notch_soss : the data should be notched
        self.support : the data should be zero-padded
        
        Returns
        ----------
        template : numpy.array 1d of float
            Notched template in time domain.

        """
        if template is None:
            template = tm.template_bank(dt=self.dt_days)[template_ind]
        if len(template)!=self.support:
            temp = np.zeros(self.support)
            temp[:len(template)] = template
            temp = np.roll(temp,-len(template)//2) 
            template = np.array(temp)
        for sos in self.notch_soss:
            template=sosfiltfilt(sos, np.array(template), padtype='constant') 
            # Very important to have zeros in the ends, bacause of padding.
            # Padtype 'constant' will pad by the edge value, and we need to ensure it is zero)
            # That is why notching has to be done when the template is centered at the center,
            # and only then the template can be rolled.
        return template
    
    
    def notch_all_templates(self):
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
        for k in range(len(self.templates)): #different template indices and different subbin shifts
            for sos in self.notch_soss:
                self.templates[k] = sosfiltfilt(sos, np.array(self.templates[k]), padtype='constant') 
                # Very important to have zeros in the ends, bacause of padding.
                # Ppadtype 'constant' will pad by the edge value, and we need to ensure it is zero
                # That is why notching has to be done when the template is centered at the center,
                # and only then the template can be rolled.

    
    def roll_templates(self):
        for k in range(len(self.templates)): #different template indices
           self.templates[k] = np.roll(self.templates[k], 
                                        -(len(self.templates[k])//2-1))    


    def whiten_template(self, template=None, template_ind=20, use_rfft=use_rfft):
        """
        Whitens one template specified in the template parameter by applying
        the whitening filter calculated after the PSD measurement of the data.
        If template==None takes from the 
        template_module.template_bank()[template_ind], prepares it (adjusting
        length to self.support and shifting) and whitens.    

        Parameters
        ----------
        template : numpy.array 1d of float
            The template to be whitened
        template_ind : int, < len(template_module.template_bank())
            Index of the template from template_module.template_bank() to be
            whitened. Will be used only if template==None.
        use_rfft : bool
            Whether to use rfft or to use fft
            
        Requires
        ----------
        self.wf_fd : the data should be whitened
        self.support : the data should be zero-padded
        
        Returns
        ----------
        whitened_template_td : numpy.array 1d of float
            Whitened template in time domain.
        whitened_template_fd : numpy.array 1d of float
            Whitened template in Fourier domain.
        """
        if template is None:
            template = tm.template_bank(dt=self.dt_days)[template_ind]
        if len(template)!=self.support:
            temp = np.zeros(self.support)
            temp[:len(template)] = template
            temp = np.roll(temp,-len(template)//2) 
            template = np.array(temp)
        whitened_template_fd = []
        whitened_template_td = []
        if use_rfft:
            for i in self.quarters:
                whitened_temp_fd = np.fft.rfft(template)*self.wf_fd[i]
                whitened_template_fd.append(whitened_temp_fd)
                whitened_template_td.append(np.fft.irfft(whitened_temp_fd))
        else:
            for i in self.quarters:
                whitened_temp_fd = np.fft.fft(template)*self.wf_fd[i]
                whitened_template_fd.append(whitened_temp_fd)
                whitened_template_td.append(np.fft.ifft(whitened_temp_fd))
        return whitened_template_td, whitened_template_fd
    
    
    def whiten_all_templates(self, use_rfft=use_rfft):
        """
        Whitens all the templates in self.templates by applying
        the whitening filter calculated after the PSD measurement of the data.
        
        Parameters
        ----------
        use_rfft : bool
            Whether to use rfft or fft

        Requires
        ----------
        self.wf_fd : the data should be whitened
        self.support : the data should be zero-padded
        
        Creates
        ----------
        self.whitened_templates_td : dict of len(templates) entries, each of length self.quarters
            Dictionary of all the whitened templates for all the quarters in time domain
        self.whitened_templates_fd : dict of len(templates) entries, each of length self.quarters
            Dictionary of all the whitened templates for all the quarters in Fourier domain
        """
        if not self.is_templates_initialized:
            self.initialize_templates(template_ind='all')
        self.whitened_templates_td = np.zeros((len(self.templates), len(self.quarters), self.support))
        self.whitened_templates_fd = np.zeros((len(self.templates), len(self.quarters), self.fft_support), dtype=np.complex128)
        for k in range(len(self.templates)): #different template indices
            for q in self.quarters:
                whitened_temp_fd = np.fft.rfft(self.templates[k])*self.wf_fd[q]
                self.whitened_templates_fd[k][q] = whitened_temp_fd
                if use_rfft:
                    self.whitened_templates_td[k][q] = np.fft.irfft(whitened_temp_fd)
                else:
                    self.whitened_templates_td[k][q] = np.fft.ifft(whitened_temp_fd)
        self.is_template_whitened[:]=True
            
        
    def matched_filtering(self, template_ind='all', max_small_hole=max_small_hole,  
                          detr_length_outliers=detr_length_outliers, 
                          nsig_outliers_stat=nsig_outliers_stat):
        """
        Calculates the matched-filtering statistics of data and templates.
        Cuts the statistic from zero padding ends.
        
        Parameters
        ----------
        template_ind : int or 'all'
            The index of the template from self.whitened_templates_td to use.
            If 'all', will be done on all the templates  
        nsig_outliers : float
            Minimal deviation considered as outlier in calculation of outlier-free
            std of the statistic
            
        Requires
        ---------
        self.whitened_fluxes_fd : fluxes should be whitened
        self.whitened_templates_fd : templates should be whitened
        self.masks : need masks for std calculation, but will ignore small holes
        self.support : data should be zero-padded
        
        Creates
        ---------
        self.statistics_raw : h@C**(-1)@d. Just the matched-filtering score without denominator
        self.stat_moving_vars : Moving variance of statistics_raw
        self.statistics_snr : h@C**(-1)@d/(h@C**(-1)@h)**0.5 Statistics in units of SNR.
        self.statistics_abs : h@C**(-1)@d/(h@C**(-1)@h) Statistics in absolute units.
        self.stat_stds : Standard deviation of the self.statistics_raw
        self.stat_denominators : (h@C**(-1)@h)**0.5
            *where h is the template, d is the flux, C is the covariance matrix, @ is the 
            inner product, and the formulae are written schematically. In reality,
            they are implemented with whitened data and fluxes.
        """
        self.big_hole_masks = []
        for q in self.quarters:
            bigmask = np.array(self.masks[q][
                    self.zero_padding_ends[q][0]:len(self.masks[q])-self.zero_padding_ends[q][1]])
            self.big_hole_masks.append(clean_small_holes(np.array(bigmask), max_hole=max_small_hole).astype(bool))
                
        if template_ind=='all':
            self.orig_time_lengths = np.array([len(self.orig_fluxes[q]) for q in self.quarters])
            max_time_len = np.max(self.orig_time_lengths)
            self.statistics_raw = np.zeros((len(self.templates),len(self.quarters),max_time_len))
            self.statistics_raw_cor = np.zeros((len(self.templates),len(self.quarters),max_time_len)) #corrected for moving std and then multiplied back by non-moving std
            self.stat_moving_vars = np.zeros((len(self.templates),len(self.quarters),max_time_len))
            self.statistics_snr = np.zeros((len(self.templates),len(self.quarters),max_time_len))
            self.statistics_abs = np.zeros((len(self.templates),len(self.quarters),max_time_len))
            self.stat_vars = np.zeros((len(self.templates),len(self.quarters))) #mean of moving variance
            #TODO: think about outlier free moving variance
            self.stat_masks = np.zeros((len(self.templates),len(self.quarters),max_time_len)).astype(bool)
            
            if not self.is_template_whitened.all():
                self.whiten_all_templates()
            
            for k in range(len(self.whitened_templates_fd)):
                for q in self.quarters:
                    #statistics not in snr, not in absolute units
                    stat = np.real(np.fft.irfft(np.conjugate(self.whitened_templates_fd[k][q])*self.whitened_fluxes_fd[q]))
                    stat = np.array(stat[self.zero_padding_ends[q][0]:len(stat)-self.zero_padding_ends[q][1]]) #CUT THE STATISTIC
                    statmask = np.array(self.big_hole_masks[q])
                    self.statistics_raw[k][q][:self.orig_time_lengths[q]] = stat       
                    #TODO: outliers in moving variance
                    movvar = moving_var(np.array(stat), mask=statmask, nperseg=nperseg_moving_var, ngap=ngap_moving_var,
                                                 remove_outliers=True, detr_length_outliers=detr_length_outliers,
                                                 nsig_outliers=nsig_outliers_stat, return_mask=False)
                    movvar[~statmask] = np.inf
                    #TODO: think about the masked regions. Big hole mask.
                    self.stat_moving_vars[k][q][:self.orig_time_lengths[q]] = movvar
                    self.statistics_snr[k][q][:self.orig_time_lengths[q]] = stat/movvar**0.5
                    statrawcor = stat/movvar*np.nanmean(movvar[statmask])
                    statrawcor[~statmask] = 0.
                    self.statistics_raw_cor[k][q][:self.orig_time_lengths[q]] = statrawcor
                    self.stat_vars[k][q] = np.nanmean((movvar[statmask])**0.5)**2
                    self.statistics_abs[k][q][:self.orig_time_lengths[q]] = stat/movvar
                    self.stat_masks[k][q][:self.orig_time_lengths[q]] = statmask

        else:
            assert type(template_ind)==int, "Template index `template_ind` must be int or 'all'!"
            if not self.is_template_whitened[template_ind]:
                self.whiten_all_templates()
            k = template_ind
            for q in self.quarters:
                #statistics not in snr, not in absolute units
                stat = (np.fft.irfft(np.conjugate(self.whitened_templates_fd[k][q])*self.whitened_fluxes_fd[q]))
                stat = np.array(stat[self.zero_padding_ends[q][0]:len(stat)-self.zero_padding_ends[q][1]]) #CUT THE STATISTIC
                statmask = np.array(self.masks[q][
                    self.zero_padding_ends[q][0]:len(self.masks[q])-self.zero_padding_ends[q][1]])
                statmask = clean_small_holes(np.array(statmask), max_hole=max_small_hole)
                self.statistics_raw[k][q][:self.orig_time_lengths[q]] = stat       
                #TODO: outliers in moving variance
                movvar = moving_var(np.array(stat), mask=statmask,  nperseg=nperseg_moving_var, ngap=ngap_moving_var,
                                    remove_outliers=True, detr_length_outliers=detr_length_outliers,
                                                 nsig_outliers=nsig_outliers_stat, return_mask=False)
                movvar[statmask] = np.inf
                self.stat_moving_vars[k][q][:self.orig_time_lengths[q]] = movvar
                self.statistics_snr[k][q][:self.orig_time_lengths[q]] = stat/movvar**0.5
                statrawcor = stat/movvar*np.nanmean(movvar[statmask])
                statrawcor[~statmask] = 0.
                self.statistics_raw_cor[k][q][:self.orig_time_lengths[q]] = statrawcor
                self.stat_vars[k][q] = np.nanmean((movvar[statmask])**0.5)**2
                self.statistics_abs[k][q][:self.orig_time_lengths[q]] = stat/movvar
                self.stat_masks[k][q][:self.orig_time_lengths[q]] = statmask
    
    
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
            times = []
            masks = []
            stat_masks = []
            # big_hole_masks = []
            whitened_fluxes_td = []
            for q in self.quarters:
                fluxes.append(np.array(self.fluxes[q][
                    self.zero_padding_ends[q][0]:len(self.fluxes[q])-self.zero_padding_ends[q][-1]]))
                times.append(np.array(self.times[q][
                    self.zero_padding_ends[q][0]:len(self.times[q])-self.zero_padding_ends[q][-1]]))
                masks.append(np.array(self.masks[q][
                    self.zero_padding_ends[q][0]:len(self.masks[q])-self.zero_padding_ends[q][-1]]))
                # big_hole_masks.append(np.array(self.big_hole_masks[q][
                #     self.zero_padding_ends[q][0]:len(self.big_hole_masks[q])-self.zero_padding_ends[q][-1]]))
                whitened_fluxes_td.append(np.array(
                    self.whitened_fluxes_td[q][
                        self.zero_padding_ends[q][0]:len(self.whitened_fluxes_td[q])-self.zero_padding_ends[q][-1]]))
                # self.detrended_fluxes[q] = np.array(self.detrended_fluxes[q][
                    #self.zero_padding_ends[q][0]:len(self.detrended_fluxes[q])-self.zero_padding_ends[q][-1]])
                # self.trends_fluxes[q] = np.array(self.trends_fluxes[q][
                    #self.zero_padding_ends[q][0]:len(self.trends_fluxes[q])-self.zero_padding_ends[q][-1]])
            self.fluxes = fluxes
            self.times = times
            self.masks = masks
            # self.big_hole_masks = big_hole_masks
            self.whitened_fluxes_td = whitened_fluxes_td
            self.is_zero_padded = False
        if not self.is_statistics_zero_cut:
            statistics_snr = []
            statistics_raw_cor = []
            statistics_masks = []
            big_hole_masks = []
            for q in self.quarters:
                big_hole_masks.append(self.big_hole_masks[q][:self.orig_time_lengths[q]])
            self.big_hole_masks = big_hole_masks
            for k in range(len(self.statistics_snr)):
                stat_snr = []
                stat_raw_cor = []
                stat_masks = []
                for q in range(len(self.statistics_snr[k])):
                    stat_snr.append(self.statistics_snr[k][q][:self.orig_time_lengths[q]])
                    stat_raw_cor.append(self.statistics_raw_cor[k][q][:self.orig_time_lengths[q]])
                    stat_masks.append(self.stat_masks[k][q][:self.orig_time_lengths[q]])
                statistics_snr.append(stat_snr)
                statistics_raw_cor.append(stat_raw_cor)
                statistics_masks.append(stat_masks)
            self.statistics_snr = statistics_snr
            self.stat_masks = statistics_masks
            self.statistics_raw_cor = statistics_raw_cor
        self.is_statistics_zero_cut = True

            
    def veto_statistics(self, margin_snr=margin_snr_veto, template_ind=template_ind_detection,
                         fn=fn_fit_veto, maxdev=maxdev_fit_veto,
                     rfft_psd=use_rfft, zero_ends_len_abs=zero_ends_len_abs_fit_veto,
                     zero_ends_len_rel=zero_ends_len_rel_fit_veto):
        #TODO: decide about detrending. What to detrend (initial flux or final), to detrend or not
        #are the long transits important?
        #if the transit is near the edge and is big, it is better not to use detrending.
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
        Parameters
        --------------
        margin_snr : float
            The minimal abs(snr) that is considered as a tpossible ransit candidate. Values
            greater that are checked by sign (only negative are accepted) and shape
            (only good-fitting the template are accepted).        
        template_ind : int, (< len(self.templates)) or 'all'
            The template number from the template list that will be used.
            If 'all', all the templates will be used.
        fn : float
            False negative value of the chi-squared distribution with the number of degrees of freedom equal to the
            length of the truncated template. Note that this distribution is without accounting for multiple-comparison 
            effect (we are fitting the best amplitude for the template, to the real distribution of the scores
            will be shrunk closer to zero)
        maxdev : float
            Maximum allowed absolute deviation of the data from the best-fit template.
        zero_end_len_abs : int or None
            length of zeros to take at the end of the template from each side in addition
            to the non-zero region of the template. Is needed to make sure that the shape
            of the templite is intdeed fulfilled. Is dominant over zero_end_len_rel
        zero_end_len_rel : float
            If zero_end_len_abs is None, is used to calculate new zero_end_len_abs
            as [(length of nonzero template)*zero_end_len_rel]. This amount of
            bins is used from each side.
        
        Requires
        --------------
        self.statistics_snr
        self.templates
        self.stat_masks
        self.fluxes
        self.psds
            
        Modifies
        --------------
        self.trigger_indices : 
            Adds to dict list of len(#quarters) of numpy.arrays 1d.
            Indices at each quarter where the verified significant triggers happened.                                 
        self.trigger_amplitudes : 
            Adds to dict list of len(#quarters) of numpy.arrays 1d, 
            corresponfing to self.trigger_indices.
            Amplitudes in absolute units (quarter-specific) determined from fitting
        self.veto_masks : 
            Adds to dict list of len(#quarters) of numpy.arrays 1d of len support.
            Masks in statistic space. If False, the statistic in this index has high absolute
            value > margin_snr, byt was vetoed as bad.                           
        """   
        
        if template_ind=='all':
            temp_indices = np.arange(len(self.templates))
        else:
            assert type(temp_indices)==int, "temp_indices is assumed to be int or 'all'!"
            temp_indices = [template_ind]
            
        if self.is_zero_padded:
            self.cut_zero_padding_ends()
        
        self.trigger_indices = [[[] for q in self.quarters] for k in range(len(self.templates))]
        self.trigger_amplitudes = [[[] for q in self.quarters] for k in range(len(self.templates))]
        self.veto_masks = [[np.ones(len(self.stat_masks[k][q])).astype(bool) for q in self.quarters] for k in range(len(self.stat_masks))]
        self.veto_bad_trigger_inds = [[[] for q in self.quarters] for k in range(len(self.templates))]
        for k in temp_indices:
            template = self.initial_templates[k]
            halflen=len(template)//2
            for q in self.quarters: #over quarters
                trigger_indices_quarter = []
                trigger_amplitudes_quarter = []
                statistic = np.array(self.statistics_snr[k][q])
                mask = np.array(self.stat_masks[k][q]) #allow small holes to be in the transit
                veto_mask = np.ones(len(statistic)).astype(bool)
                
                #masking the positive-amplitude triggers:
                positive_trigger_indices = np.where((statistic>0)&(np.abs(statistic)>=margin_snr)&(mask))[0]
                self.stat_masks[k][q][positive_trigger_indices] = False
                
                #vetoing bad-fitting (negative-amplitude) template triggers:
                trigger_indices = np.where((statistic<=0)&(np.abs(statistic)>=margin_snr)&(mask)&(veto_mask))[0]
                veto_bad_trigger_ind = [[],[],[]]
                for i in trigger_indices:
                    to_veto_combined = True #to process to put data to the vetoing function
                    if ((i-halflen>=0) and (i+len(template)-halflen<=len(self.fluxes[q]))):
                        data_cut = np.array(self.fluxes[q][i-halflen:i+len(template)-halflen])
                        template_cut = np.array(template)
                    elif ((i-halflen>=0) and np.isclose(template[halflen + len(self.fluxes[q])-i],0)):
                        data_cut = np.array(self.fluxes[q][i-halflen:])
                        template_cut = np.array(template[:len(data_cut)])
                    elif ((i+len(template)-halflen<=len(self.fluxes[q])) and np.isclose(template[halflen-i],0)):
                        data_cut = np.array(self.fluxes[q][:i+len(template)-halflen])
                        template_cut = np.array(template[len(template)-len(data_cut):])
                    elif (len(template)>len(self.fluxes[q]) and np.isclose(template[halflen + len(self.fluxes[q])-i],0) and np.isclose(template[halflen-i],0)):
                        data_cut = np.array(self.fluxes[q])
                        template_cut = np.array(template[halflen-i:halflen-i+len(data_cut)])
                    else:
                        veto_bad_trigger_ind[0].append(i)
                        veto_bad_trigger_ind[1].append('extreme_position ')
                        veto_bad_trigger_ind[2].append(np.nan)
                        to_veto_combined = False #Already marked inappropriate, will not be treated with veto_combined
                    if to_veto_combined:
                        isgood, ampl, reason = veto_combined(data=data_cut, template=template_cut, 
                                fn=fn, maxdev_std=maxdev, maxdev_ampl=None,
                                zero_ends_len_abs=zero_ends_len_abs, zero_ends_len_rel=zero_ends_len_rel, 
                                return_amplitude=True, return_reason=True)
                        if isgood:
                            trigger_indices_quarter.append(i)
                            trigger_amplitudes_quarter.append(ampl)
                        else:
                            veto_bad_trigger_ind[0].append(i)
                            veto_bad_trigger_ind[1].append(reason)
                            veto_bad_trigger_ind[2].append(ampl)
                veto_mask[veto_bad_trigger_ind[0]] = False
                self.veto_masks[k][q] = veto_mask
                self.stat_masks[k][q] *= veto_mask
                self.trigger_indices[k][q] = copy.deepcopy(trigger_indices_quarter)
                self.trigger_amplitudes[k][q] = copy.deepcopy(trigger_amplitudes_quarter)
                self.veto_bad_trigger_inds[k][q] = copy.deepcopy(veto_bad_trigger_ind)
                self.statistics_snr[k][q][~self.stat_masks[k][q]] = 0. 
                self.statistics_raw_cor[k][q][~self.stat_masks[k][q]] = 0. #ATTENTION! WE DO THIS FOR FAST SUMMATION IN FUTURE TO AVOID EXTRA IF
                
            
                
                
    
    def connect_quarters(self):
        # TODO: is it legite to connect quarters with such a shift?
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
        self.time_full, self.masks_full = connect_quarters_fun(self.times, self.masks,
                             zero_padding_ends=None, is_data_cut=False, is_time_cut=False, dt=self.dt_days, fill_gaps_by_nan=False)
        self.time_full, self.trend_orig_flux_full = connect_quarters_fun(self.times, self.trends_orig_fluxes,
                             zero_padding_ends=None, is_data_cut=False, is_time_cut=False, dt=self.dt_days, fill_gaps_by_nan=False)
        self.statistics_full = np.zeros((len(self.templates), len(self.time_full)))
        self.stat_vars_full = np.zeros((len(self.templates), len(self.time_full)))
        self.stat_masks_full = np.zeros((len(self.templates), len(self.time_full))).astype(bool)
        vars_quarters = [[np.ones(len(self.statistics_raw_cor[k][q]))*self.stat_vars[k][q]
                           for q in self.quarters] for k in range(len(self.templates))]
        
        for k in range(len(self.templates)):
            self.statistics_full[k] = connect_quarters_fun(self.times, self.statistics_raw_cor[k],
                     zero_padding_ends=None, is_data_cut=True, is_time_cut=True, dt=self.dt_days, fill_gaps_by_nan=False)[1]
            self.stat_masks_full[k] = (connect_quarters_fun(self.times, self.stat_masks[k],
                     zero_padding_ends=None, is_data_cut=True, is_time_cut=True, dt=self.dt_days, fill_gaps_by_nan=False)[1]).astype(bool)
            self.stat_vars_full[k] = connect_quarters_fun(self.times, vars_quarters[k],
                     zero_padding_ends=None, is_data_cut=True, is_time_cut=True, dt=self.dt_days, fill_gaps_by_nan=False)[1]

        
    
    def interleave_shifts(self):
        """

        """
        self.interleaved_numbers = np.array([i for i in range(len(self.indices_of_templates)) 
                                  if ((self.indices_of_templates[i][1]-self.indices_of_templates[i][0])>=2)])
        self.not_interleaved_numbers = np.array([i for i in range(len(self.indices_of_templates)) 
                                  if ((self.indices_of_templates[i][1]-self.indices_of_templates[i][0])==1)])
        # self.not_interleaved_numbers = np.array([self.indices_of_templates[i][0] for i in range(len(self.indices_of_templates)) 
        #                           if ((self.indices_of_templates[i][1]-self.indices_of_templates[i][0])==1)])
        
        #for interleaved:
        shift_numbers = np.array([self.indices_of_templates[i][1]-self.indices_of_templates[i][0] for i in self.interleaved_numbers])
            #number of shifts for all the templates with shifts. need them to be the same
        self.n_shifts = shift_numbers[0]
        assert (shift_numbers/shift_numbers[0]==np.ones(len(shift_numbers))).all(), 'Templates with shifts must have the same number of shifts!'  #make sure they are all the same number of shifts
        self.statistics_interleaved = []
        self.stat_masks_interleaved = []
        self.stat_vars_interleaved = []
        for t in self.interleaved_numbers:
            self.statistics_interleaved.append((self.statistics_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
            self.stat_masks_interleaved.append((self.stat_masks_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
            self.stat_vars_interleaved.append((self.stat_vars_full[self.indices_of_templates[t][0]:self.indices_of_templates[t][1]]).T.flatten())
        self.statistics_interleaved = np.array(self.statistics_interleaved)   
        self.stat_masks_interleaved = np.array(self.stat_masks_interleaved).astype(bool)
        self.stat_vars_interleaved = np.array(self.stat_vars_interleaved)
        self.trend_orig_flux_interleaved = np.vstack((
            self.trend_orig_flux_full, self.trend_orig_flux_full,
            self.trend_orig_flux_full)).T.flatten()
        self.time_interleaved = np.arange(len(self.trend_orig_flux_interleaved))*self.dt_days/self.n_shifts+self.time_full[0]
        
        #for not interleaved:
        self.statistics_not_interleaved = []
        self.stat_masks_not_interleaved = []
        self.stat_vars_not_interleaved = []
        indices_for_not_interleaved = [self.indices_of_templates[i][0] for i in self.not_interleaved_numbers]
        self.statistics_not_interleaved = np.array(self.statistics_full[indices_for_not_interleaved])
        self.stat_masks_not_interleaved = np.array(self.stat_masks_full[indices_for_not_interleaved])
        self.stat_vars_not_interleaved = np.array(self.stat_vars_full[indices_for_not_interleaved])
        self.trend_orig_flux_not_interleaved = np.array(self.trend_orig_flux_full)
        self.time_not_interleaved = np.arange(len(self.trend_orig_flux_not_interleaved))*self.dt_days+self.time_full[0]
        
    
        
            
        
    def sum_periods(self, Pmin=Pmin, Pmax=Pmax, duty_cycle=duty_cycle):
        """
        Sums the statistics with all the possible periods and initial phases. Accounts for 
        subbin template shifts, accounts for masks.
        Does not return 
    
        Parameters
        ----------
        statistics_full : numpy.array 2d of float
            Raw (not SNR units, not abs units) corrected (for moving variance)
            statistics for all the templates. Quarters connected
        stat_masks_full : numpy.array 2d of bool
            Masks for `statistics_full`  .
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
        dP : float
            Step in the list of all the inveastigated periods
        Pmin : float
            Minimal period to consider.
        Pmax : float
            Maximal period to consider.
    
        Parameters
        ----------
        dP : float
            Period step, days
        Pmin : float
            Minimal period, days
        Pmax : float
            maximal period, days
    
        Creates
        -------
        self.statistics_periodic, 
        self.statistics_periodic_counts, 
        self.statistics_phases, 
        self.statistics_periodic_lengths_filled, 
        self.statistics_periodic_relative_amplitudes
    
        """
        
        Per = Pmin
        periods = [Per]
        while Per<= Pmax:
            dP = duty_cycle * Per / (self.tmax - self.tmin)
            Per += dP
            periods.append(Per) 
        periods = np.array(periods)
        self.periods = periods
        self.phases_interleaved = np.arange(int(np.ceil((self.periods[-1])/(self.dt_days/self.n_shifts))))*(self.dt_days/self.n_shifts)
        self.phases_not_interleaved = np.arange(int(np.ceil((self.periods[-1])/self.dt_days)))*self.dt_days
        # self.phases = np.arange(int(np.ceil((self.periods[-1])/self.dt_days)))*self.dt_days
        #attention: we use the same dt for interleaved and not? it means that the phases are the same
        
        #interleaved
        statistics_by_trends = (self.statistics_interleaved * self.trend_orig_flux_interleaved)
        statistics_by_trends = np.hstack((statistics_by_trends, np.zeros(
                        (len(statistics_by_trends),int(np.ceil(Pmax/self.dt_days/self.n_shifts))))))
        #pad by zeros to avoid extra if statements in the summation:       
        trends_sq_by_vars = self.stat_vars_interleaved * self.trend_orig_flux_interleaved**2
        trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros(
                        (len(trends_sq_by_vars),int(np.ceil(Pmax/self.dt_days/self.n_shifts))))))
        
        self.statistics_periodic_interleaved, denomsq = self.sum_periods_fast(
                        statistics_by_trends, trends_sq_by_vars, self.dt_days/self.n_shifts, self.tmax+self.dt_days*(1-1/self.n_shifts), self.tmin, periods)
        #attention: we use the same dt for interleaved and not? it means that the phase step for the interleaved could be smaller.
        for p in range(len(self.statistics_periodic_interleaved)):
            for t in range(len(self.statistics_periodic_interleaved[p])):
                self.statistics_periodic_interleaved[p][t][denomsq[p][t]==0]=np.nan
                self.statistics_periodic_interleaved[p][t][denomsq[p][t]!=0] /= denomsq[p][t][denomsq[p][t]!=0]**0.5
        
        
        #not interleaved
        statistics_by_trends = (self.statistics_not_interleaved * self.trend_orig_flux_not_interleaved)
        statistics_by_trends = np.hstack((statistics_by_trends, np.zeros(
                        (len(statistics_by_trends),int(np.ceil(Pmax/self.dt_days))))))
        #pad by zeros to avoid extra if statements in the summation:       
        trends_sq_by_vars = self.stat_vars_not_interleaved * self.trend_orig_flux_not_interleaved**2
        trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros(
                        (len(trends_sq_by_vars),int(np.ceil(Pmax/self.dt_days))))))
                
        self.statistics_periodic_not_interleaved, denomsq = self.sum_periods_fast(
                        statistics_by_trends, trends_sq_by_vars, self.dt_days, self.tmax, self.tmin, periods)
        for p in range(len(self.statistics_periodic_not_interleaved)):
            for t in range(len(self.statistics_periodic_not_interleaved[p])):
                self.statistics_periodic_not_interleaved[p][t][denomsq[p][t]==0]=np.nan
                self.statistics_periodic_not_interleaved[p][t][denomsq[p][t]!=0] /= denomsq[p][t][denomsq[p][t]!=0]**0.5
        
     
        
    def determine_mins(self, ngap_moving_var=ngap_moving_var):
        pind_min_interleaved = np.nanargmin( # argmin over periods
                                    [np.nanmin(# min over templates
                                        [np.nanmin(# min over phases
                                            self.statistics_periodic_interleaved[pi][ti]) 
                                        for ti in range(len(self.statistics_periodic_interleaved[pi])) ])
                                    for pi in range(len(self.statistics_periodic_interleaved)) ])
        tempind_min_interleaved = np.nanargmin( # argmin over templates
                                     [np.nanmin( # min over phases
                                         self.statistics_periodic_interleaved[pind_min_interleaved][ti]) #in the best period
                                     for ti in range(len(self.statistics_periodic_interleaved[pind_min_interleaved])) ])
        phasind_min_interleaved = np.nanargmin( # argmin over phases
                                    self.statistics_periodic_interleaved[pind_min_interleaved][tempind_min_interleaved])
    
        pind_min_not_interleaved = np.nanargmin( # argmin over periods
                                    [np.nanmin(# min over templates
                                        [np.nanmin(# min over phases
                                            self.statistics_periodic_not_interleaved[pi][ti]) 
                                        for ti in range(len(self.statistics_periodic_not_interleaved[pi])) ])
                                    for pi in range(len(self.statistics_periodic_not_interleaved)) ])
        tempind_min_not_interleaved = np.nanargmin( # argmin over templates
                                     [np.nanmin( # min over phases
                                         self.statistics_periodic_not_interleaved[pind_min_not_interleaved][ti]) #in the best period
                                     for ti in range(len(self.statistics_periodic_not_interleaved[pind_min_not_interleaved])) ])
        phasind_min_not_interleaved = np.nanargmin( # argmin over phases
                                    self.statistics_periodic_not_interleaved[pind_min_not_interleaved][tempind_min_not_interleaved])
        
        pind_incorrect = np.nanargmin( # argmin over periods
                                    [np.nanmin(# min over templates
                                        [np.nanmin(# min over phases
                                            np.abs(self.statistics_periodic_not_interleaved[pi][ti])) 
                                        for ti in range(len(self.statistics_periodic_not_interleaved[pi])) ])
                                    for pi in range(len(self.statistics_periodic_not_interleaved)) ])
        tempind_incorrect = np.nanargmin( # argmin over templates
                                     [np.nanmin( # min over phases
                                         np.abs(self.statistics_periodic_not_interleaved[pind_min_not_interleaved][ti])) #in the best period
                                     for ti in range(len(self.statistics_periodic_not_interleaved[pind_min_not_interleaved])) ])
        
        if np.nanmin(self.statistics_periodic_interleaved)<np.nanmin(self.statistics_periodic_not_interleaved):
            self.is_min_interleaved = True
            self.snr_measured = np.nanmin(self.statistics_periodic_interleaved)
            self.pind_measured = pind_min_interleaved
            self.phind_measured = phasind_min_interleaved
            self.ph_measured = self.phases_interleaved[phasind_min_interleaved]
            self.tempind_min_interleaved = tempind_min_interleaved
            self.tempind_measured = self.interleaved_numbers[tempind_min_interleaved]
            self.std_around_min = outlier_free_std(np.concatenate((
                self.statistics_periodic_interleaved[self.pind_measured][tempind_min_interleaved][:self.phind_measured - ngap_moving_var//2 * self.n_shifts],
                self.statistics_periodic_interleaved[self.pind_measured][tempind_min_interleaved][self.phind_measured + ngap_moving_var//2 * self.n_shifts:])))
        else:
            self.is_min_interleaved = False
            self.snr_measured = np.nanmin(self.statistics_periodic_not_interleaved)
            self.pind_measured = pind_min_not_interleaved
            self.phind_measured = phasind_min_not_interleaved
            self.ph_measured = self.phases_not_interleaved[phasind_min_not_interleaved]
            self.tempind_min_not_interleaved = tempind_min_not_interleaved
            self.tempind_measured = self.not_interleaved_numbers[tempind_min_not_interleaved]
            self.std_around_min = outlier_free_std(np.concatenate((
                self.statistics_periodic_not_interleaved[self.pind_measured][tempind_min_not_interleaved][:self.phind_measured - ngap_moving_var//2],
                self.statistics_periodic_not_interleaved[self.pind_measured][tempind_min_not_interleaved][self.phind_measured + ngap_moving_var//2:])))
            
        self.period_measured = self.periods[self.pind_measured]
        self.std_free_place = outlier_free_std(self.statistics_periodic_not_interleaved[pind_incorrect][tempind_incorrect])
        
        
        
        expected_transit_times = [self.tmin + self.ph_measured + i * self.period_measured 
                                   for i in range(int(np.ceil((self.tmax - self.tmin - self.ph_measured)/self.period_measured) ))]
        self.expected_transit_times = expected_transit_times
        if self.is_min_interleaved:
            summed_transit_points = [np.argmin(np.abs(self.time_interleaved - expected_transit_times[i])) for i in range(len(expected_transit_times))  ]
            self.summed_transit_times = np.array(self.time_interleaved[summed_transit_points])
            summed_stats = np.array(self.statistics_interleaved[tempind_min_interleaved][summed_transit_points])
            summed_trends = np.array(self.trend_orig_flux_interleaved[summed_transit_points])
            summed_vars = np.array(self.stat_vars_interleaved[tempind_min_interleaved][summed_transit_points])
        else:
            summed_transit_points = [np.argmin(np.abs(self.time_not_interleaved - expected_transit_times[i])) for i in range(len(expected_transit_times))  ]
            self.summed_transit_times = np.array(self.time_not_interleaved[summed_transit_points])
            summed_stats = np.array(self.statistics_not_interleaved[tempind_min_not_interleaved][summed_transit_points])
            summed_trends = np.array(self.trend_orig_flux_not_interleaved[summed_transit_points])
            summed_vars = np.array(self.stat_vars_not_interleaved[tempind_min_not_interleaved][summed_transit_points])
        denom = np.sum(summed_vars * summed_trends**2)**0.5
        self.summed_transit_statistics = (summed_stats * summed_trends)/denom
        self.origrnal_transit_indices = [[] for q in self.quarters]
        for t in self.expected_transit_times:
            qmin = 0
            imin = 0
            minval = np.abs(t - self.times[0][0])
            for q in self.quarters:
                if (((t - self.times[q][0])>=0) and (np.abs(t - self.times[q][0])<minval)):
                    qmin = q
                    minval = np.abs(t - self.times[q][0])
            minval = t - self.times[qmin][0]
            for i in range(len(self.times[qmin])):
                if np.abs(t - self.times[qmin][i])<minval:
                    imin = i
                    minval = np.abs(t - self.times[qmin][i])
            self.origrnal_transit_indices[qmin].append(imin)
                
        
        
    #plotting methods
    def plot_results(self, 
                     plot_all=None,
                     original_data=None,
                     psds=None,
                     whitened_fluxes_td=None,
                     whitened_fluxes_fd=None,
                     single_event_statistic_snr_without_mask=None,
                     single_event_statistic_snr=None,
                     fluxes_with_veto=None,
                     histogram_single_event=None,
                     summed_statistic_minimal_of_phase=None,
                     summed_statistic_of_phase=None,
                     histogram_summed_statistic=None,
                     minimal_statistic_periods=None,
                     temp_num=8, period=None, per_ind=0, nbins_hist=25, 
                     figsize=None, edgelist_hist=None, edgemin_hist=None, edgemax_hist=None
                     ):
        if figsize is None:
            width=12
            height=5
        else:
            width=figsize[0]
            height=figsize[1]
        
        if period is None:
            period = self.periods[per_ind]
        else:
            per_ind = np.argmin(np.abs(self.periods-period))
            period = self.periods[per_ind]
        
        if plot_all==True:
            if original_data is None:
                original_data = True
            if psds is None:
                psds = True
            if whitened_fluxes_td is None:
                whitened_fluxes_td = True
            if whitened_fluxes_fd is None:
                whitened_fluxes_fd = True
            if single_event_statistic_snr_without_mask is None:
                single_event_statistic_snr_without_mask = True
            if single_event_statistic_snr is None:
                single_event_statistic_snr = True
            if fluxes_with_veto is None:
                fluxes_with_veto = True
            if histogram_single_event is None:
                histogram_single_event = True
            if summed_statistic_minimal_of_phase is None:
                summed_statistic_minimal_of_phase = True
            if summed_statistic_of_phase is None:
                summed_statistic_of_phase = True
            if histogram_summed_statistic is None:
                histogram_summed_statistic = True
            if minimal_statistic_periods is None:
                minimal_statistic_periods = True
        elif plot_all==False:
            if original_data is None:
                original_data = False
            if psds is None:
                psds = False
            if whitened_fluxes_td is None:
                whitened_fluxes_td = False
            if whitened_fluxes_fd is None:
                whitened_fluxes_fd = False
            if single_event_statistic_snr_without_mask is None:
                single_event_statistic_snr_without_mask = False
            if single_event_statistic_snr is None:
                single_event_statistic_snr = False
            if fluxes_with_veto is None:
                fluxes_with_veto = False
            if histogram_single_event is None:
                histogram_single_event = False
            if summed_statistic_minimal_of_phase is None:
                summed_statistic_minimal_of_phase = False
            if summed_statistic_of_phase is None:
                summed_statistic_of_phase = False
            if histogram_summed_statistic is None:
                histogram_summed_statistic = False
            if minimal_statistic_periods is None:
                minimal_statistic_periods = False
        elif plot_all==None:
            if original_data is None:
                original_data = True
            if psds is None:
                psds = True
            if whitened_fluxes_td is None:
                whitened_fluxes_td = True
            if whitened_fluxes_fd is None:
                whitened_fluxes_fd = True
            if single_event_statistic_snr_without_mask is None:
                single_event_statistic_snr_without_mask = False
            if single_event_statistic_snr is None:
                single_event_statistic_snr = True
            if fluxes_with_veto is None:
                fluxes_with_veto = True
            if histogram_single_event is None:
                histogram_single_event = True
            if summed_statistic_minimal_of_phase is None:
                summed_statistic_minimal_of_phase = True
            if summed_statistic_of_phase is None:
                summed_statistic_of_phase = False
            if histogram_summed_statistic is None:
                histogram_summed_statistic = True
            if minimal_statistic_periods is None:
                minimal_statistic_periods = True
            
        
        if original_data:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.orig_times[q], self.orig_fluxes[q])
            plt.title('Original unprocessed PDS lightcurves')
            plt.xlabel('time, days')
            plt.ylabel('flux')
            plt.show()
        
        if psds:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.psdfreqs[q], self.psds[q])
            plt.title('Measured PSDS of all the quarters')
            plt.xlabel('frequency, Hz')
            plt.ylabel('PSD')
            plt.show()
        
        if whitened_fluxes_td:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.times[q], self.whitened_fluxes_td[q])
            plt.title('Whitened fluxes in time domain')
            plt.xlabel('time, days')
            plt.ylabel('whitened flux')
            plt.show()
        
        if whitened_fluxes_fd:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.freqs[q], np.real(self.whitened_fluxes_fd[q]))
            plt.title('Whitened fluxes in Fourier domain')
            plt.xlabel('frequency, Hz')
            plt.ylabel('whitened flux')
            plt.show()
        
        if single_event_statistic_snr_without_mask:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.times[q], self.statistics_snr[temp_num][q])
            plt.title('Single-event statistic in units of SNR for template #'+str(temp_num))
            plt.xlabel('time, days')
            plt.ylabel('statistic [SNR]')
            plt.show()
        
        if single_event_statistic_snr:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                if q%2==0:
                    clr = (0.2, 0.5,0.8)
                else:
                    clr = (0.2, 0.7,0.4)
                clr
                plt.plot(self.times[q], self.statistics_snr[temp_num][q], color=clr)
            #plot what was masked in statistic masks
            label_made=False
            for q in self.quarters:
                if not label_made:
                    if (~self.stat_masks[temp_num][q]).any():
                        label_made=True
                        plt.plot(self.times[q][~self.stat_masks[temp_num][q]], self.statistics_snr[temp_num][q][~self.stat_masks[temp_num][q]],
                                 'o', color=(0.7,0.5,0), label='masked statistic', markersize=3)
                else:
                    plt.plot(self.times[q][~self.stat_masks[temp_num][q]], self.statistics_snr[temp_num][q][~self.stat_masks[temp_num][q]],
                             'o', color=(0.7,0.5,0), markersize=3)
            #plot what from this was masked in conditioning
            label_made=False
            for q in self.quarters:
                if not label_made:
                    if (~self.stat_masks[temp_num][q]).any():
                        label_made=True
                        plt.plot(self.times[q][~self.big_hole_masks[q]], self.statistics_snr[temp_num][q][~self.big_hole_masks[q]],
                                 'o', color='k', label='masked in conditioning')
                else:
                    plt.plot(self.times[q][~self.big_hole_masks[q]], self.statistics_snr[temp_num][q][~self.big_hole_masks[q]],
                             'o', color='k')
            
            times_with_transits = np.array([])
            stat_with_transits = np.array([])
            for q in self.quarters:
                times_with_transits = np.concatenate((times_with_transits, self.times[q][self.origrnal_transit_indices[q]]))
                stat_with_transits = np.concatenate((stat_with_transits, self.statistics_snr[temp_num][q][self.origrnal_transit_indices[q]]))
            plt.plot(times_with_transits, stat_with_transits, 'o', color=(0.8,0,0), label='transits', markersize=4)
            
            plt.title('Single-event statistic in units of SNR for template #'+str(temp_num))
            plt.xlabel('time, days')
            plt.ylabel('statistic [SNR]')
            plt.legend()
            plt.show()
            
        if fluxes_with_veto:
            plt.figure(figsize=(width, height))
            labeled_chi_squared_veto = False
            labeled_maximal_deviation_std_veto = False
            labeled_maximal_deviation_amplitude_veto = False
            labeled_extreme_position_veto = False
            labeled=False
            for q in self.quarters:
                plt.plot(self.times[q], self.fluxes[q], color='k', linewidth=2)
                if not labeled:
                    plt.plot(self.times[q][~self.big_hole_masks[q]], self.fluxes[q][~self.big_hole_masks[q]], 'o', 
                             color=(0,0.5,1), markersize=5, label='big inpainted holes')
                    labeled=True
                else:
                    plt.plot(self.times[q][~self.big_hole_masks[q]], self.fluxes[q][~self.big_hole_masks[q]], 'o', 
                             color=(0,0.5,1), markersize=5)
            for q in self.quarters:
                chi_squared_veto = []
                maximal_deviation_std_veto = []
                maximal_deviation_amplitude_veto = []
                extreme_position_veto = []
                for ind in range(len(self.veto_bad_trigger_inds[temp_num][q][0])):
                    if self.veto_bad_trigger_inds[temp_num][q] != [[], [], []]:
                        if 'chi_squared' in self.veto_bad_trigger_inds[temp_num][q][1][ind]:
                            chi_squared_veto.append(self.veto_bad_trigger_inds[temp_num][q][0][ind])
                        elif 'maximal_deviation_std' in self.veto_bad_trigger_inds[temp_num][q][1][ind]:    
                            maximal_deviation_std_veto.append(self.veto_bad_trigger_inds[temp_num][q][0][ind])
                        elif 'maximal_deviation_amplitude' in self.veto_bad_trigger_inds[temp_num][q][1][ind]:        
                            maximal_deviation_amplitude_veto.append(self.veto_bad_trigger_inds[temp_num][q][0][ind])
                        elif 'extreme_position' in self.veto_bad_trigger_inds[temp_num][q][1][ind]:           
                            extreme_position_veto.append(self.veto_bad_trigger_inds[temp_num][q][0][ind])
                if len(chi_squared_veto)>0:
                    if not labeled_chi_squared_veto:
                        plt.plot(self.times[q][chi_squared_veto], self.fluxes[q][chi_squared_veto], 'o', color='r', markersize=4, label='chi squared veto')
                        labeled_chi_squared_veto = True
                    else:
                        plt.plot(self.times[q][chi_squared_veto], self.fluxes[q][chi_squared_veto], 'o', color='r', markersize=4)
                if len(maximal_deviation_std_veto)>0:
                    if not labeled_maximal_deviation_std_veto:
                        plt.plot(self.times[q][maximal_deviation_std_veto], self.fluxes[q][maximal_deviation_std_veto], 'o', color=(1,0.5,0), markersize=4, label='maximal deviation std veto')
                        labeled_maximal_deviation_std_veto = True
                    else:
                        plt.plot(self.times[q][maximal_deviation_std_veto], self.fluxes[q][maximal_deviation_std_veto], 'o', color=(1,0.5,0), markersize=4)
                if len(maximal_deviation_amplitude_veto)>0:
                    if not labeled_maximal_deviation_amplitude_veto :
                        plt.plot(self.times[q][maximal_deviation_amplitude_veto], self.fluxes[q][maximal_deviation_amplitude_veto], 'o', color=(0.6,0.4,0), markersize=4, label='maximal deviation amplitude veto')
                        labeled_maximal_deviation_amplitude_veto = True
                    else:
                        plt.plot(self.times[q][maximal_deviation_amplitude_veto], self.fluxes[q][maximal_deviation_amplitude_veto], 'o', color=(0.6,0.4,0), markersize=4)
                if len(extreme_position_veto)>0:
                    if not labeled_extreme_position_veto:
                        plt.plot(self.times[q][extreme_position_veto], self.fluxes[q][extreme_position_veto], 'o', color=(0.7,0,1), markersize=4, label='extreme position veto')
                        labeled_extreme_position_veto = True
                    else:
                        plt.plot(self.times[q][extreme_position_veto], self.fluxes[q][extreme_position_veto], 'o', color=(0.7,0,1), markersize=4)
            times_with_transits = np.array([])
            fluxes_with_transits = np.array([])
            for q in self.quarters:
                times_with_transits = np.concatenate((times_with_transits, self.times[q][self.origrnal_transit_indices[q]]))
                fluxes_with_transits = np.concatenate((fluxes_with_transits, self.fluxes[q][self.origrnal_transit_indices[q]]))
            plt.plot(times_with_transits, fluxes_with_transits, 'o', color=(0,0.7,0), markersize=3, label='transits')
            plt.title('Conditioned fluxes in time domain with vetos for template #'+str(temp_num))
            plt.xlabel('time, days')
            plt.ylabel('flux')
            plt.legend(fontsize=(plt.rcParams['font.size']-3) )
            plt.show()
            
        if histogram_single_event:
            datstat = np.array([])
            for q in self.quarters:
                datstat = np.concatenate((datstat, self.statistics_snr[temp_num][q][self.stat_masks[temp_num][q]]))
            histogram(datstat, n=nbins_hist, edgelist=edgelist_hist, edgemin=edgemin_hist, edgemax=edgemax_hist, xscale='square', yscale='log',
                  density=False, normalize=True, ylogshift=True, logzeros='zero', 
                  return_results=False, plot=True, new_figure=True, ax=None, figsize=(width/3*2,height), show=True,
                  xlabel='statistic squared', ylabel='log counts', 
                  title='Histogram of statistic in units of snr (uncorrected) for all the quarters for template #'+str(temp_num), 
                  xlim=None, ylim=None,
                  color='b', linestyle='solid', marker='', linewidth=None, alpha=0.9, label=None, 
                  loc_legend=None, plot_type='bar')
            
        if summed_statistic_minimal_of_phase:
            plt.figure(figsize=(width, height))
            if self.is_min_interleaved:
                plt.plot(self.phases_interleaved, self.statistics_periodic_interleaved[self.pind_measured][self.tempind_min_interleaved])
                plt.plot([self.phases_interleaved[self.phind_measured]], 
                         [self.statistics_periodic_interleaved[self.pind_measured][self.tempind_min_interleaved][self.phind_measured]], 'o',
                         label='minimal value determined')
            else:
                plt.plot(self.phases_not_interleaved, self.statistics_periodic_not_interleaved[self.pind_measured][self.tempind_min_not_interleaved])
                plt.plot([self.phases_not_interleaved[self.phind_measured]], 
                         [self.statistics_periodic_not_interleaved[self.pind_measured][self.tempind_min_not_interleaved][self.phind_measured]], 'o',
                         label='minimal value determined')
            plt.title('Summed statistic as function of phase for minimal parameters: period='
                      +str(np.round(self.periods[self.pind_measured],2))+' days, template #'+str(self.tempind_measured))
            plt.xlabel('initial phase, days')
            plt.ylabel('statistic, [SNR]')
            plt.legend()
            plt.show()
            
        if summed_statistic_of_phase:
            plt.figure(figsize=(width, height))
            if temp_num in self.interleaved_numbers:
                plt.plot(self.phases_interleaved, self.statistics_periodic_interleaved[per_ind][temp_num])
            else:
                plt.plot(self.phases_not_interleaved, self.statistics_periodic_not_interleaved[per_ind][temp_num])
            plt.title('Summed statistic as function of phase for period='+str(period)+' days, template #'+str(temp_num))
            plt.xlabel('initial phase, days')
            plt.ylabel('statistic, [SNR]')
            plt.show()
            
        if histogram_summed_statistic:
            if temp_num in self.interleaved_numbers:
                datstat = self.statistics_periodic_interleaved[per_ind][temp_num]
            else:
                datstat = self.statistics_periodic_not_interleaved[per_ind][temp_num]
            histogram(datstat[~np.isnan(datstat)], n=nbins_hist, edgelist=edgelist_hist, edgemin=edgemin_hist, edgemax=edgemax_hist, xscale='square', yscale='log',
                  density=False, normalize=True, ylogshift=True, logzeros='zero', 
                  return_results=False, plot=True, new_figure=True, ax=None, figsize=(width/3*2,height), show=True,
                  xlabel='statistic squared', ylabel='log counts', 
                  title='Histogram of summed statistic for all phases for period='+str(temp_num), 
                  xlim=None, ylim=None,
                  color='b', linestyle='solid', marker='', linewidth=None, alpha=0.9, label=None, 
                  loc_legend=None, plot_type='bar') 
            
        if minimal_statistic_periods:
            mins = []
            for p in range(len(self.periods)):
                mins.append(np.nanmin(self.statistics_periodic_not_interleaved[p]))
            plt.figure(figsize=(width, height))
            plt.plot(self.periods, mins)
            plt.title('Minimum of summed statistic as function of period')
            plt.xlabel('periods, days')
            plt.ylabel('statistic, [SNR]')
            plt.show()
                

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
    dP : float
        Step in the list of all the inveastigated periods
    Pmin : float
        Minimal period to consider.
    Pmax : float
        Maximal period to consider.

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
    phases_size = int(np.ceil((periods[-1])/dt)) 
    statistics_periodic = np.zeros((len(periods), len(statistics_by_trends),phases_size))
    denomsq = np.zeros((len(periods), len(statistics_by_trends),phases_size))
    for P_ind in range(len(periods)):
        P = periods[P_ind]
        n_transits = int(np.ceil((tmax-tmin)/P))
        phaselen = int(P/dt)
        for n in range(n_transits):      
            time_ind = int(np.round((P*n/dt)))
            statistics_periodic[P_ind, :,:phaselen] += statistics_by_trends[:,time_ind:time_ind+phaselen]
            denomsq[P_ind, :,:phaselen] += trends_sq_by_vars[:,time_ind:time_ind+phaselen]
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
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    def isint(string):
        assert type(string)==str, 'Input must be string!'
        if string.isdecimal():
            return True
        elif string[0]=='-' and (string[1:]).isdecimal():
            return True
        else:
            return False
    def issimplenum(string):
        if isint(string):
            return True
        elif string.count('.')==1 and isint(string.replace('.', '')):
            return True
        else:
            return False
    def isfloat(string):
        string = string.lower()
        if issimplenum(string):
            return True
        elif ((string.count('e')==1 and issimplenum(string.replace('e', '')))
              or (string.count('e-')==1 and issimplenum(string.replace('e-', '')))
              or (string.count('e+')==1 and issimplenum(string.replace('e+', '')))):
            return True
        else:
            return False
    for line in lines:
        if '=' in line:
            text = line.split('#')[0] #remove comments
            text = text.split('\\')[0] #remove endlines
            name, val = text.split('=')[0:2]
            name = name.strip()
            val = val.strip()
            if ((name!='') and (val!='')):
                if val=='None':
                    val=None
                elif val=='True':
                    val=True
                elif val=='False':
                    val=False
                elif isint(val):
                    val = int(val)
                elif isfloat(val):
                    val = float(val)
                else:
                    assert (((val[0]=='"') and (val[-1]=='"')) 
                            or ((val[0]=="'") and (val[-1]=="'"))), 'Undefined type of value: not int, bool, float, str or None! ( line "'+str(line)+'" )'
                    val = val[1:-1]
                    
                assert ((65<=ord(name[0])<=90) or (97<=ord(name[0])<=122) or name[0]=='_') , 'Incorrect variable name specified!'
                if len(name)>1:
                    for e in name[1:]:
                        assert ((48<=ord(e)<=57) or (65<=ord(e)<=90) or (97<=ord(e)<=122) or e=='_') , 'Incorrect variable name specified!'
                globals()[name] = val
                
                
def get_data_fun(target_id, location=data_location):
    """
    Search default directory for prev downloaded data. if not found, download and put there.
    
    Parameters
    -----------
    target_id : int or str
        Kepler ID of the observed star.
        A 9-digit string (first digits can be zero) or integer. May be less than 9
        digits, then will be zero-padded (example: '757280' -> '000757280')
    location : str
        Location (full path) of the data in the local storage. Data will be searched for 
        in `location`. There should be a folder named as a 9-digit target_id.
        In this folder, fits filed of Kepler data should be stored. 
        If such a folder is not found, it will be created. Data will be then downloaded from
        https://archive.stsci.edu/pub/kepler/lightcurves/ and placed in this folder.
    
    Returns
    ----------
    times, fluxes : lists of numpy.arrays 1d (may contain numpy.nan)
    """
    target_id = target_id.rjust(9, "0")
    folder=os.path.join(location,target_id)
    
    if target_id not in os.listdir(location):
        try:
            os.mkdir(folder)
        except OSError:
                warnings.warn("Creation of the directory failed")
                
        webdir = 'https://archive.stsci.edu/pub/kepler/lightcurves/'
        # webdir = 'https://archive.stsci.edu/missions/kepler/lightcurves/'
        webfolder = webdir + target_id[:4] + '/' + target_id + '/'
        r = requests.get(webfolder)   
        lines=[line for line in (r.text).splitlines() if 'llc.fits' in line]
        urls = [webfolder + line[line.index('href="kplr'+target_id)+6 : line.index('.fits"')+5] for line in lines]
        for url in urls:
            wget.download(url, folder)
    
    times=[]
    fluxes=[]
    files = [f for f in os.listdir(folder)[:] if 'llc.fits' in f]
    if len(files)<17:
            warnings.warn('Number of files is not equal to the total number quarters. Available '+str(len(files)) + ' files')
    for f in files:
        filc = fits.open(os.path.join(folder,f))
        times.append(filc[1].data['TIME'])
        fluxes.append(filc[1].data['PDCSAP_FLUX'])
        filc.close()
    
    firstindices = []
    for tim in times:
        firstindices.append(np.argmax(~np.isnan(tim))) #argmax is for finding the first instance of not nan
    i0 = np.max(firstindices) #find the biggest initial indes to call all times by it and make sure there is no nans
    # sort them by increasing time because the files in the directory are sorted at random:
    pairs = list(zip(times,fluxes))
    pairs.sort(key=lambda p: p[0][i0])
    times, fluxes = list(map(list,zip(*pairs)))
    return times, fluxes


def unb_var(arr):
    """
    Calculates the unbiased variance estimate (numpy.var is biased!).
    Is related to the biased estimation as
    var_u = N/(N-1) * var_b,
    where N is the size of the sample.
    Parameters
    ----------
    arr : numpy.array 1d or list of float, may contain numpy.nan
        Sample of numbers to calculate their variance
    
    Returns
    ---------
    float
        Unbiased variance estimate of arr
    """
    arr = np.array(arr[~np.isnan(arr)])
    return 1 / (len(arr)-1) * np.sum(np.abs(arr - np.mean(arr))**2)
        
        
def moving_var(signal, mask=None, nperseg=nperseg_moving_var, ngap=ngap_moving_var,
               remove_outliers=False, detr_length_outliers=detr_length_outliers,
               nsig_outliers=nsig_outliers_stat, return_mask=False):
    """
    Calculates the moving variance of the signal. At every point, takes the window of length
    nperseg symmetrically from the two sides, excludes from the middle the window of length
    ngap and calculates the unbiased variance of the remaining. It is needed to
    avoid the impact of the entries correlated with the considered point to the 
    variance estimation. For example, if there is a transit in this point, the 
    variance is taken around the transit, so its presence does not alter the measurement.
    At the ends of the singal, the data is taken only from one side of the 
    considered point up to the available length.
    
    Parameters
    ----------
    signal : numpy.array 1d of float, NO NANS
        Data from which the moving variance is taken.
    nperseg : int
        Length of the segment to calculate local variance for each bin
    ngap : int
        Length of the gap to exclude around each bin when calculating the
        variance in this bin in order to avoid influence of these
        points on the variance estimation. 
    Returns
    -------
    mov_var : numpy.array 1d of float, len(signal)
        Moving variance estimate at each point of the signal
    """
    signal = np.array(signal)
    mask = np.array(mask)
    if mask is None:
        mask = np.ones(len(signal)).astype(bool)
        
    if remove_outliers:
        # data1 = detrend_amat(np.array(signal), length=detr_length_outliers, mask=mask, 
        #                          method='median', endmethod='mean', ends='symmetric', endlen=None, 
        #                          return_trend=False, keep_overall_level=False, remove_outliers=True, nsig_outliers=nsig_outliers)
        # outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        # mask *= outlier_mask
        outlier_mask = np.abs(signal - np.nanmean(signal[mask])) < nsig_outliers * np.nanstd(signal[mask])
        mask *= outlier_mask
        outlier_mask = np.abs(signal - np.nanmean(signal[mask])) < nsig_outliers * np.nanstd(signal[mask])
        mask *= outlier_mask
        
        
    if nperseg>len(signal)/2:
        nperseg = len(signal)//2 + 1
        warnings.warn('Not enough data to calculate moving variance.')
        if ngap>=nperseg//2:
            ngap=nperseg//2
    signal[~mask] = 0.
    sum1 = np.cumsum(signal)
    sum1 = np.concatenate(([0], sum1))
    sum2 = np.cumsum(signal**2)
    sum2 = np.concatenate(([0], sum2))
    summ = np.cumsum(mask)
    summ = np.concatenate(([0], summ))
    mov_var = []
    for i in np.arange(0,ngap//2):
        counts = summ[i+nperseg//2]-summ[i+ngap//2]
        if counts==0:
            warnings.warn('COUNTS=0!. 1st')
        mov_var.append(((sum2[i+nperseg//2]-sum2[i+ngap//2])/counts-((sum1[i+nperseg//2]-sum1[i+ngap//2])/counts)**2)
                       *counts/(counts-1))
    for i in np.arange(ngap//2,nperseg//2):
        counts = summ[i-ngap//2]-summ[0]+summ[i+nperseg//2]-summ[i+ngap//2]
        if counts==0:
            warnings.warn('COUNTS=0!. 2nd')
        mov_var.append(((sum2[i-ngap//2]-sum2[0]+sum2[i+nperseg//2]-sum2[i+ngap//2])/counts
                       -((sum1[i-ngap//2]-sum1[0]+sum1[i+nperseg//2]-sum1[i+ngap//2])/counts)**2)
                       *counts/(counts-1))
    for i in range(nperseg//2,len(signal)-nperseg//2):
        counts = summ[i+nperseg//2]-summ[i+ngap//2]+summ[i-ngap//2]-summ[i-nperseg//2]
        if counts==0:
            warnings.warn('COUNTS=0!. 3rd')
        mov_var.append(((sum2[i+nperseg//2]-sum2[i+ngap//2]+sum2[i-ngap//2]-sum2[i-nperseg//2])/counts
                      -(((sum1[i+nperseg//2]-sum1[i+ngap//2]+sum1[i-ngap//2]-sum1[i-nperseg//2])/counts)**2))
                       *counts/(counts-1))
    for i in np.arange(len(signal)-nperseg//2, len(signal)-ngap//2):
        counts = summ[i-ngap//2]-summ[i-nperseg//2]+summ[-1]-summ[i+ngap//2] 
        if counts==0:
            warnings.warn('COUNTS=0!. 4th')
        mov_var.append(((sum2[i-ngap//2]-sum2[i-nperseg//2]+sum2[-1]-sum2[i+ngap//2])/counts
                     -((sum1[i-ngap//2]-sum1[i-nperseg//2]+sum1[-1]-sum1[i+ngap//2])/counts)**2)
                       *counts/(counts-1))
    for i in np.arange(len(signal)-ngap//2, len(signal)):
        counts = summ[i-ngap//2]-summ[i-nperseg//2]
        if counts==0:
            warnings.warn('COUNTS=0!. 5th')
        mov_var.append(((sum2[i-ngap//2]-sum2[i-nperseg//2])/counts
                       -((sum1[i-ngap//2]-sum1[i-nperseg//2])/counts)**2)
                       *counts/(counts-1))
    if not return_mask:
        return np.abs(np.array(mov_var)) #because small fluctuations around zero can occur  
    else:
        return np.abs(np.array(mov_var)), mask
    
        

def outlier_free_std(data, nsig=3, mask=None, detrend=False, detr_len=128):
    """
    Calculates the std of data without impact of outliers and excluding masked regions

    Parameters
    ----------
    data : np.array 1d of float
        The data, may contain nans.
    nsig : float
        Minimum amplitude in numbers of sigma (std) that is considered an outlier
    mask : np.array 1d of bool
        mask of bad values (False is bad, True is good). If None, will be
        replaced by measured mask of nans.
    detrend : bool
        If True, data will be detrended before measuring std
    detr_len : int
        If detrend, data will be detrended with this detrending length

    Returns
    -------
    float
        std of the data
    """
    if mask is None:
        mask =~np.isnan(data)
    if detrend:
        data1 = detrend_amat(np.array(data), length=detr_len, mask=mask, method='median', endmethod='mean', ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False)
    else:
        data1 = np.array(data)
    outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig * np.nanstd(data1[mask])
    return np.nanstd(data[outlier_mask*mask])        


def psd2wf(psd, support=None, rfft=True):
    """
    'Power spectrum density to whitening filter'.
    Generates a whitening filter of the needed length (support) from the PSD
    (Power spectral density). A whitening filter of the length of the data is 
    needed in order to be applied to the red-noise data and turn it into the 
    white noise for then using in detection together with whitened template.
    
    Parameters
    ----------
    psd : numpy.array 1d
        PSD in full fft frequencies (if rfft==False) or in rfft frequencies (if rfft==True)
        
    support : int
        length of the needed filter
        if None, take the length corresponding to the PSD
    
    rfft : bool
        If True, the signal is treated as real, and all ffts are rffts. The length 
        of the frequency samples are the nperseg//2+1, where nperseg is the length
        in the rela spece corresponding to this PSD
        Default is True
    Returns
    -------
    wf_td : numpy.array 1d
        whitening filter in time domain
    wf_fd : numpy.array 1d
        whitening filter in Fourier domain.
    """
    if rfft:
        if support is None:
            support=2*(len(psd)-1)
        assert support%2==0, 'support assumed to be even!'
        wpsd = copy.deepcopy(psd**(-0.5))
        wpsd[0]=0. #if you don't do this, you will have a complete disaster after notch!
        wf_td = np.fft.irfft(wpsd)
        mid_point = len(wf_td)//2
        result = np.zeros(support)
        result[:mid_point] = wf_td[:mid_point]
        result[-mid_point:] = wf_td[-mid_point:]
        wf_td = np.fft.fftshift(result)
        wf_fd = np.fft.rfft(result)
    else:
        if support=='none':
            support=len(psd)
        assert support%2==0, 'support assumed to be even!'
        wpsd = copy.deepcopy(psd**(-0.5))
        wpsd[0]=0.
        wf_td = np.fft.ifft(wpsd)
        mid_point = len(wf_td)//2
        result = np.zeros(support, dtype=np.complex128)
        result[:mid_point] = wf_td[:mid_point]
        result[-mid_point:] = wf_td[-mid_point:]
        wf_td = np.fft.fftshift(result)
        wf_fd = np.fft.fft(result)
    return wf_td, wf_fd


def filter2cov_matrix(vec, pad_by_zeros=False, trim_zeros=True):
    """
    Make a covariance matrix for the statistically shift-invariant data
    in the time domain from the covariance vector in time domain (vec).
    If vec is the time domain whitening filter, the result will be 
    the INVERSE covariance matrix.
    
    Parameters
    ----------
    vec : np.array
        A filter which will be convolved with itself in an
        outer product way to make a covariance matrix
    pad_by_zeros : bool, optional
        If true, pads the vector by zeros of the same size 
        in order to omit calculation of covariance with bins from the other side. 
    trim_zeros : bool, optional
        Trim after padding the omitted zeros. The default is False.
    Returns
    -------
    np.array two-dimensional
        The covariance matrix (for the time domain). 
        If vec is the time domain whitening filter, the result will be 
        the INVERSE covariance matrix.
    """
    L=len(vec)    
    if pad_by_zeros:
        vec = np.concatenate((vec, np.zeros(L)))
        L=len(vec)
    prod = np.array([np.sum(vec*np.roll(vec,i)) for i in range(L)])
    C = np.array([ [prod[k-j] for k in range(L)] for j in range(L) ])
    if pad_by_zeros and trim_zeros:
        assert L%4==0, 'The length of the vector has to be even.'
        return C[L//4:-L//4,L//4:-L//4]
    return C


def inpaint_fluxes_fun(fluxes, masks, wf_tds):   
    L = len(fluxes[0])
    inpainted_fluxes = np.zeros((len(fluxes), L))
    for q in range(len(fluxes)):
        # filter2cov_matrix:
        prod = np.array([np.sum(wf_tds[q]*np.roll(wf_tds[q],i)) for i in range(L)])
        Cinv = np.array([ [prod[k-j] for k in range(L)] for j in range(L) ])
        
        AtCinv = Cinv[~masks[q]]
        Minv = np.linalg.inv(AtCinv[:,~masks[q]])
        F = np.zeros((L, L))
        F[~masks[q]] = np.dot(Minv, AtCinv)
        inpainted_fluxes[q] = fluxes[q] - np.dot(F, fluxes[q])
    return inpainted_fluxes


def interpolate_over_mask(flux, mask=None, interp_method='linear', ends='continue', mitigate_hole_ends=False, endhole_method='median', mit_length=10):
    """
    Takes the data and the mask of holes, interpolates the holes. If the mask is not
    provided, finds numpy.nan in the data and interpolates them.
    
    Parameters
    ----------
    flux : np.array 1d of floats
        data to be interpolated, may contain nans
    mask : np.array 1d of bool
        False entries mean there is a hole to be interpolated. No matter what 
        is the value of data at this point.
    interp_method : str : 'mean' or 'median' or 'iterative_fourier' or ' linear' (default). 
        How to fill the holes. By mean of the data, by median of the data,
        by doing Fourier transform many times, cuting high frequencies,
        going back and substituting to the holes,
        or by linearly interpolating over the holes.
    ends : str : 'continue' or 'zeros' or 'keep'.
        What to do to the ends of the data in linear method.
        'continue' is to take the first(last) value and fill by it all the rest.
        'zeros' meens to replace by zeros.
        'keep' means to leave what was in the original data (maybe nans)
    mitigate_hole_ends : bool
        If true, for the linear, mean and median interpolation, not the ends of the holes will be taken,
        but the average of points near the hole
    endhole_method : str : 'mean' or 'median'(default)
        The type of average for mitigate_hole_ends.
    mit_length : int. Default is 10
        The length of the slice for the averaging in the mitigate_hole_ends
    Returns
    -------
    numpy.array 1d of float
        Interpolated flux
    """
    
    def nan_average_filt(ar, kernel_size=3, method='mean'):
        """
        Return moving average of the data with the kernel_size.
        The average is either mean or median, defined by method.
        
        Parameters
        ----------
        ar : np.array
            the input array to filter.
        kernel_size : int, optional
            the size of subarray around each point to find the mean, should be odd. The default is 3.
        method : str, 'mean' or 'median'
            the method of how to compute the average
        Returns
        -------
        arfilt : np.array
            the array where each bin is the mean over kernel_size neighbors with
            ignoring nans (unless all the subarray is nans)
        """
        arfilt = np.array([ar[0]])
        if method=='mean':
            for i in np.arange(1,kernel_size//2):
                arfilt = np.append(arfilt, np.nanmean(ar[0: 2*i+1]))
            for i in np.arange(kernel_size//2, len(ar)-kernel_size//2-1):
                arfilt = np.append(arfilt, np.nanmean(ar[i-kernel_size//2: i+kernel_size//2+1]))
            for i in np.arange(len(ar)-kernel_size//2-1,len(ar)):
                arfilt = np.append(arfilt, np.nanmean(ar[2*i-1-len(ar):]))
        elif method=='median':
            for i in np.arange(1,kernel_size//2):
                arfilt = np.append(arfilt, np.nanmedian(ar[0: 2*i+1]))
            for i in np.arange(kernel_size//2, len(ar)-kernel_size//2-1):
                arfilt = np.append(arfilt, np.nanmedian(ar[i-kernel_size//2: i+kernel_size//2+1]))
            for i in np.arange(len(ar)-kernel_size//2-1,len(ar)):
                arfilt = np.append(arfilt, np.nanmedian(ar[2*i-1-len(ar):]))
        else:
            raise AssertionError('Undefined averaging method specified!')
        arfilt[-1]=ar[-1]
        return arfilt

    if mask is None:
        mask = ~np.isnan(flux)
    else:
        mask = np.array(mask).astype(bool)
    if interp_method=='mean' or interp_method=='median':
        ind1 = (np.where(mask)[0])[0] 
        ind2 = (np.where(mask)[0])[-1]
        #if there is a hole in the beginning or end, it cannot be interpolated
        mask1 = np.array(mask[ind1:ind2+1])
        begholes = np.where(mask1*np.roll(~mask1,-1))[0]+1
        endholes = np.where(mask1*np.roll(~mask1,1))[0]
        lenholes = endholes-begholes
        kernel_size = np.max(lenholes)+4
        indholes = np.where(~mask1)[0]
        flux[indholes]=nan_average_filt(flux, kernel_size=kernel_size, method=interp_method)[indholes]
    elif interp_method=='iterative_fourier':
        nrep = 6
        flux1 = np.array(flux)
        indholes = np.where(~ mask)[0]
        flux1[indholes] = 0
        for i in range(nrep):
            fflux1 = np.fft.fft(flux1)
            fflux1[len(flux1)//10 + 4**i : -(len(flux1)//10 + 4**i)]=0
            flux1[indholes]=(np.fft.ifft(fflux1))[indholes]
        flux=flux1
    elif interp_method=='linear':
        ind1 = (np.where(mask)[0])[0] 
        ind2 = (np.where(mask)[0])[-1]
        #if there is a hole in the beginning or end, it cannot be interpolated
        flux1 = np.array(flux[ind1:ind2+1])
        mask1 = np.array(mask[ind1:ind2+1])
        begholes = np.where(mask1*np.roll(~mask1,-1))[0]
        endholes = np.where(mask1*np.roll(~mask1,1))[0]
        if not mitigate_hole_ends:
            for j in range(len(begholes)):
                flux1[begholes[j]+1:endholes[j]]=np.interp(
                    np.arange(begholes[j], endholes[j]+1),np.array([begholes[j], endholes[j]]),
                    np.array([flux1[begholes[j]],flux1[endholes[j]]]))[1:-1]
        else:
            if endhole_method=='median':
                for j in range(len(begholes)):
                    i1 = max(begholes[j]-mit_length+1,0)
                    i2 = min(endholes[j]+mit_length,len(flux1))
                    flux1[begholes[j]+1:endholes[j]]=np.interp(
                        np.arange(begholes[j], endholes[j]+1),np.array([begholes[j], endholes[j]]),
                        np.array([np.nanmedian((flux1[i1:begholes[j]+1])[np.where(mask1[i1:begholes[j]+1])[0]]),
                                  np.nanmedian((flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            elif endhole_method=='mean':
                for j in range(len(begholes)):
                    i1 = max(begholes[j]-mit_length+1,0)
                    i2 = min(endholes[j]+mit_length,len(flux1))
                    flux1[begholes[j]+1:endholes[j]]=np.interp(
                        np.arange(begholes[j], endholes[j]+1),np.array([begholes[j], endholes[j]]),
                        np.array([np.nanmean((flux1[i1:begholes[j]+1])[np.where(mask1[i1:begholes[j]+1])[0]]),
                                  np.nanmean((flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            else:
                raise AssertionError('Incorrect endhole_method specified!')
        flux[ind1:ind2+1]=flux1
    if interp_method=='linear' or interp_method=='mean' or interp_method=='median':
        if ends == 'continue':
            if ind1 !=0:
                flux[0:ind1]=flux[ind1]
            if ind2 != len(flux)-1:
                flux[ind2:]=flux[ind2]
        elif ends == 'zeros':
            if ind1 !=0:
                flux[0:ind1]=0
            if ind2 != len(flux)-1:
                flux[ind2:]=0
        elif ends == 'keep':
            pass
        else:
            raise AssertionError('Unlefined ends parameter specified!')
    return flux


def clean_small_holes(mask, max_hole=max_small_hole):
    """
    Deletes from the mask holes of size <=max_small_hole,
    replacing False with True.
    
    Parameters
    ----------
    mask : numpy.array 1d of bool
        True is good data, False is bad data
    max_hole : int
        Maximum size of hole which is considered to be small
        
    Returns
    -------
    mask : np.array 1d of bool
        Mask in which small holes are removed, i.e. made True
    """
    mask = np.array(mask).astype(bool)
    ind1 = (np.where(mask)[0])[0] 
    ind2 = (np.where(mask)[0])[-1]
    mask1 = np.concatenate((np.ones(ind1),np.array(mask[ind1:ind2+1]), np.ones(len(mask)-ind2-1))).astype(bool)
    begholes = np.where(mask1*np.roll(~mask1,-1))[0]+1
    endholes = np.where(mask1*np.roll(~mask1,1))[0]
    lenholes = endholes-begholes
    small_holes = np.where(lenholes<=max_hole)[0]
    for i in range(len(small_holes)):
        mask[begholes[small_holes[i]]:endholes[small_holes[i]]]=True  #mask with only big holes left
    return mask

    
def interpolate_time(time, mask=None):
    """
    Takes time (which is expected to be evenly sampled) and linearly interpolates
    the holes which are marked with mask. Is different from the simple linear
    interpolation because also treats the ends of the data.
    
    Parameters
    ----------
    time : numpy.array 1d of float
        Equally-sampled time array with possible nans in holes.
    mask : numpy.array 1d of bool
        True is good data, False is bad data.   
    
    Returns
    -------
    time2 : numpy.array 1d of float
        Interpolated time. Equally sampled, no NaNs.
    """
    time2 = copy.copy(time)
    if mask is None:
        mask = ~np.isnan(time2)
    dt = np.abs(np.nanmean((time2-np.roll(time2,1))[1:]))
    ind1 = (np.where(mask)[0])[0] 
    ind2 = (np.where(mask)[0])[-1]
    #if there is a hole in the beginning or end, it will be interpolated later
    time1 = np.array(time2[ind1:ind2+1])
    mask1 = np.array(mask[ind1:ind2+1])
    begholes = np.where(mask1*np.roll(~mask1,-1))[0]
    endholes = np.where(mask1*np.roll(~mask1,1))[0]
    for j in range(len(begholes)):
        time1[begholes[j]+1:endholes[j]]=np.interp(
            np.arange(begholes[j], endholes[j]+1),np.array([begholes[j], endholes[j]]),
            np.array([time1[begholes[j]],time1[endholes[j]]]))[1:-1]
    time2[ind1:ind2+1]=time1
    if ind1 !=0:
        time2[0:ind1]=(-dt)*np.arange(1,ind1+1)[::-1]+time1[0]
    if ind2 != len(time2)-1:
        time2[ind2+1:]=dt*np.arange(1,len(time2)-ind2)+time1[-1]
    return time2

    

def detrend_amat(data, length=128, mask=None, method='median', endmethod='mean', ends='symmetric', endlen=None, 
                 return_trend=True, keep_overall_level=False, remove_outliers=False, nsig_outliers=nsig_outliers_0,
                 detr_length_outliers=detr_length_outliers):
    """
    Amateur detrending function. Subtracting the moving mean or median from the data.
    
    Parameters
    --------------------------
    data : np.array 1d or list. Data sample from which the moving average has to be subtracted
    length : int. The length of sample to calculate moving average. At a given point, 
                    half length to the left and to the rifgt are taken.
    method : str. Kind of moving value to subtract: 'median' or 'mean'
                    The default is 'median', because it is less sensitive to outliers.
    endmethod: str. Kind of moving value to subtract at the ends of the data (see below): 'median' or 'mean'.
                    The default is 'mean', because it is less sensitive to small
                    amount of data samples available in the ends
    ends : str. How to treat the points closer than length/2 to the ends of the data:
                'symmetric' :  endlen is length/4
                               1) if point<endlen, take endlen samples from the right, and up to zero from the left
                               2) if endlen<point<half length, takethe same number of points from left and right
                               3) if point>half length, take usual half length from both sides
                               for the second end, the same in opposite direction
                'asymmetric' : endlen is length/16
                               1) if point<half length, take up to zero from the left, and the length to the 
                                    right linearly increases from endlen for point=0 to length/2 at point=length/2
                               2) if point>half length, take usual half length from both sides
                               for the second end, the same in opposite direction
                'cut' : not to calculate the result at this point. The size of the output will be len(data)-length
    endlen : int, default is None. Minimal length to take from the end for determining the average in the
        first (last) point. If None, will be assigned depending on the method (see description of ends)
    return_trend : bool. Whether to return the trend in addition to the detrended data
    keep_overall_level : bool. If true, the total mean or median will be added to the data (that is preserved)
    
    Returns
    --------------------------
    if not return_trend: data with subtracted mean, np.array
    if return_trend: data with subtracted mean, np.array; trend (moving mean), np.array
    """
    assert ((method=='mean') or (method=='median')), 'Incorrect method specified'
    data = np.array(data)
    mask = np.array(mask)
    if mask is None:
        mask=~np.isnan(data)
        
    if remove_outliers:
        data1 = detrend_amat(np.array(data), length=detr_length_outliers, mask=mask, method='median', endmethod='mean', 
                                 ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False, remove_outliers=False)
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask
    
    if ends=='asymmetric':
        trend = []
        res = []
        if endlen is None:
            l4 = int(np.ceil(length/16))
        else:
            l4 = endlen
        l2 = int(np.ceil(length/2))
        if endmethod=='median':
            for i in np.arange(0, l2):
                if (mask[0:i+int(l4 + (l2-l4)/l2*i)]).any():
                    m = np.nanmedian((data[0:i+int(l4 + (l2-l4)/l2*i)]
                                      )[mask[0:i+int(l4 + (l2-l4)/l2*i)]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:i+int(l4 + (l2-l4)/l2*i)])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod=='mean':
            for i in np.arange(0, l2):
                if (mask[0:i+int(l4 + (l2-l4)/l2*i)]).any():
                    m = np.nanmean((data[0:i+int(l4 + (l2-l4)/l2*i)]
                                    )[mask[0:i+int(l4 + (l2-l4)/l2*i)]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:i+int(l4 + (l2-l4)/l2*i)])
                    trend.append(m)
                    res.append(data[i] - m)
        if method=='median':
            for i in np.arange(l2,len(data)-l2):
                if (mask[i-l2:i+l2]).any():
                    m = np.nanmedian((data[i-l2:i+l2])[mask[i-l2:i+l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i-l2:i+l2])
                    trend.append(m)
                    res.append(data[i] - m)
        elif method=='mean':
            for i in np.arange(l2,len(data)-l2):
                if (mask[i-l2:i+l2]).any():
                    m = np.nanmean((data[i-l2:i+l2])[mask[i-l2:i+l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i-l2:i+l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if endmethod=='median':
            for i in np.arange(len(data)-l2, len(data)):
                if (mask[i-int(l4 + (l2-l4)/l2*(len(data)-i)):]).any():
                    m = np.nanmedian((data[i-int(l4 + (l2-l4)/l2*(len(data)-i)):]
                                      )[mask[i-int(l4 + (l2-l4)/l2*(len(data)-i)):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i-int(l4 + (l2-l4)/l2*(len(data)-i)):])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod=='mean':
            for i in np.arange(len(data)-l2, len(data)):
                if (mask[i-int(l4 + (l2-l4)/l2*(len(data)-i)):]).any():
                    m = np.nanmean((data[i-int(l4 + (l2-l4)/l2*(len(data)-i)):]
                                    )[mask[i-int(l4 + (l2-l4)/l2*(len(data)-i)):]])
                    trend.append(m)
                    res.append(data[i] - m) 
                else:
                    m = np.nanmean(data[i-int(l4 + (l2-l4)/l2*(len(data)-i)):])
                    trend.append(m)
                    res.append(data[i] - m)
    elif ends=='symmetric':
        trend = []
        res = []
        if endlen is None:
            l4 = int(np.ceil(length/4))
        else:
            l4 = endlen
        l2 = int(np.ceil(length/2))
        if endmethod=='median':
            for i in np.arange(0, l4):
                if (mask[0:i+l4]).any():
                    m = np.nanmedian((data[0:i+l4])[mask[0:i+l4]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:i+l4])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(l4, l2):
                if (mask[0:2*i]).any():
                    m = np.nanmedian((data[0:2*i])[mask[0:2*i]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:2*i])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod=='mean':
            for i in np.arange(0, l4):
                if (mask[0:i+l4]).any():
                    m = np.nanmean((data[0:i+l4])[mask[0:i+l4]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:i+l4])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(l4, l2):
                if (mask[0:2*i]).any():
                    m = np.nanmean((data[0:2*i])[mask[0:2*i]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:2*i])
                    trend.append(m)
                    res.append(data[i] - m)
        if method=='median':
            for i in np.arange(l2,len(data)-l2):
                if (mask[i-l2:i+l2]).any():
                    m = np.nanmedian((data[i-l2:i+l2])[mask[i-l2:i+l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    # trend.append(np.nan)
                    # res.append(np.nan)
                    m = np.nanmedian(data[i-l2:i+l2])
                    trend.append(m)
                    res.append(data[i] - m)
        elif method=='mean':
            for i in np.arange(l2,len(data)-l2):
                if (mask[i-l2:i+l2]).any():
                    m = np.nanmean((data[i-l2:i+l2])[mask[i-l2:i+l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i-l2:i+l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if endmethod=='median':
            for i in np.arange(len(data)-l2, len(data)-l4):
                if (mask[2*i-len(data):]).any():
                    m = np.nanmedian((data[2*i-len(data):])[mask[2*i-len(data):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[2*i-len(data):])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(len(data)-l4, len(data)):
                if (mask[i-l4:]).any():
                    m = np.nanmedian((data[i-l4:])[mask[i-l4:]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i-l4:])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod=='mean':
            for i in np.arange(len(data)-l2, len(data)-l4):
                if (mask[2*i-len(data):]).any():
                    m = np.nanmean((data[2*i-len(data):])[mask[2*i-len(data):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[2*i-len(data):])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(len(data)-l4, len(data)):
                if (mask[i-l4:]).any():
                    m = np.nanmean((data[i-l4:])[mask[i-l4:]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i-l4:])
                    trend.append(m)
                    res.append(data[i] - m)
    elif ends=='cut':
        trend = []
        res = []
        l2 = int(np.ceil(length/2))
        if method=='median':
            for i in np.arange(l2,len(data)-l2):
                if (mask[i-l2:i+l2]).any():
                    m = np.nanmedian((data[i-l2:i+l2])[mask[i-l2:i+l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i-l2:i+l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if method=='mean':
            for i in np.arange(l2,len(data)-l2):
                if (mask[i-l2:i+l2]).any():
                    m = np.nanmean((data[i-l2:i+l2])[mask[i-l2:i+l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i-l2:i+l2])
                    trend.append(m)
                    res.append(data[i] - m)
    else:
        raise AssertionError ('Incorrect way to treat ends specified!')
    res = np.array(res)    
    if keep_overall_level:
        if method=='median':
            res = res - np.nanmedian(res) + np.nanmedian(data[mask])
        if method=='mean':
            res = res - np.nanmean(res) + np.nanmean(data[mask])
    if return_trend:
        return res, trend
    else:
        return res


def notch_fun(f0, fs=6.28, df=None, irl=None, htol=1e-4, min_val=0, min_db=None, out='sos'):
    """
    Design second-order IIR notch digital filter from specified parameters which
    are different from ones accepted by scipy.iirnotch.
    A notch filter is a band-stop filter with a narrow bandwidth (high quality factor).
    It rejects a narrow frequency band and leaves the rest of the spectrum little changed. 
    Parameters
    ----------
    f0 : float
        The frequency that we want to notch, Hz.
    fs : float
        Sampling frequency. The [0,pi] in the discrete frequency 
        corresponds to [0, fs/2] in the continuous frequency.
        The default is 2 pi, corresponding to w in [0, pi].
    df : float
        Frequency width of the 3 dB band.
        The frequency response at df/2 is 3 dB or 1/2.
        The default is None, will be replaced by f0/30 
    irl : float
        Impulse response length, samples.Length in time domain where the response
        of the filter is greater than htol. Parameter is dominant over df.
        If not specified, df be used to define the filter.
    htol : float
        Tolerance parameter for the filter decay. The number considered 
        as zero (or the end) of the filter in time domain.
    min_val : float
        The value that the amplitude takes in minimum for the notched frequency
    min_db : float or None
        The value that the amplitude takes in minimum for the notched frequency in db.
        The dominant parameter is min_val.
    out : string
        Type of output.
        The default is 'sos'. Others are 'tf' (transfer function) and 'zpk',  zeros, poles, and gain
        
    Returns
    -------
    if out=='sos':
        sos : second order section representation of the filter
        (See the example of application in KeplerLightCurve.notch_fluxes())
    elif out=='zpk':
        z, p, k : zeros, poles, gain representation of the filter
    elif out=='tf': (transfer function)
        b, a : numerator and denominator polynomial coefficients of the IIR filter
    """
    assert f0 <= fs/2, 'The specified frequency is out of sampling range'
    # To use scipy.iirnotch we have to convert from out parameters to one accepted by it
    w0 = 2* np.pi * f0/fs # Convert from physical frequency to the digital frequency
    if irl is not None: 
        D = htol**(1 / irl) # amplitude of the pole (close to unity)
        dw = 2*np.arctan(1/D-1) #digital frequency width defined from it
        Q = w0/dw/2 #Quality factor, dimensionless parameter that characterizes 
                    #notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw
                    #(for some reason, it works correctly with factor of 1/2 )  
    else: #if the impulse response length is not specified, use df
        if df is not None:
            Q = f0/df/2 #iQuality factor. In reality, Q gives w0/dw/2 
                        #for some reason, it works correctly with factor of 1/2
        else:
            Q = 30 #the default quality factor if no defining parameter is specified
    b, a = iirnotch(f0, Q, fs=fs) #numerator and denominator polynomials of the IIR filter
    #if we want to notch the frequency not to zero, but to some minimal value,
    #we beed to shift and renormalize the numerator
    if min_val!=0: 
        b = b*(1-min_val)+min_val*a
    elif min_db is not None:
        min_val=10**(min_db/10)
        b = b*(1-min_val)+min_val*a
    if out=='sos':
        sos = tf2sos(b, a) #convert to second order sections representation
        return sos
    elif out=='ztk':
        z, p, k = tf2zpk(b, a) #convert to zeros, poles, gain representation
        return z, p, k
    elif out=='tf': #transfer function representation
        return b, a
    else:
        raise AssertionError('Invalid output type specified!')
        

def mask_outliers_fun(data, mask=None, nsig=3., nsig_positive=None, nsig_negative=None,
                           detr_length=128, detr_method='median', detr_endmethod='mean', detr_ends='symmetric',
                       detr_endlen=None):
    """
    Masks entries which deviate from the general trend more than nsig standard deviations.
    Detrending is made to find the general trend.
    
    Parameters
    ----------
    data : numpy array 1d of float
    mask : numpy.array 1d of bool
        True is good data, False is bad data. 
        If None, will be measured to mask NaN.
    nsig : float 
        Maximum deviation from the trend in units of std of data (more extreme
        values will be marked as outliers). Is dominated by nsig_positive
        and nsig_negative.
    nsig_positive : float 
        Maximum positive deviation from the trend in units of std of data (more extreme
        values will be marked as outliers). Dominates over nsig. If is None,
        will be replaced by nsig.
    nsig_negative : float 
        Maximum positive deviation from the trend in units of std of data (more extreme
        values will be marked as outliers). Dominates over nsig. If is None,
        will be replaced by nsig.
    detr_length : int. Default is 128
        Detrending length.
    detr_method : None or str : 'mean' or 'median'
        The type of moving average using to define the trend. The default is 'median',
        because it is less sensitive to outliers. If None, data will not be detrended.
    detr_endmethod : str, 'mean' or 'median'
        The type of moving average using to define the trend at the ends of the
        data. The default is 'mean', because it is less sensitive to small
        amount of data samples available in the ends
    detr_ends : str, 'symmetric' or 'asymmetric' or 'cut'. 
        See the desctiption in detrend_amat. The default is 'symmetric'.
    detr_endlen : int. Default is None
        Minimal length to take feom the end for determining the average in the
        first (last) point. If None, will then be redefined 
        depending on the detr_endmethod
    detr_return_trend : bool. Default is False.
        Whether to return the trend as well or just the detrended data. 
    detr_keep_overall_level : bool. Default is False.
        Whether to keep the overall average the same as initially.
    Returns
    -------
    outlier_mask : np.array 1d of bool.
        True where data is good, False where there are outliers.
    """
    
    if mask is None:
        mask=np.ones(len(data)).astype(bool)
    if (nsig_positive is None and nsig_negative is None):
        indices = np.where(mask)[0]
        if detr_method is None:
            outlier_mask = np.abs(data-np.nanmean(data[indices]))<nsig * np.nanstd(data[indices])
        else:
            detr = detrend_amat(data, length=detr_length, mask=mask, method=detr_method, endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen, 
                                return_trend=False, keep_overall_level=False)
            outlier_mask = np.abs(detr-np.nanmean(detr[indices]))<nsig * np.nanstd(detr[indices])
    else:
        if ((nsig_negative is None) and (nsig is not None)):
            nsig_negative = nsig
        elif ((nsig_positive is None) and (nsig is not None)):
            nsig_positive = nsig
        indices = np.where(mask)[0]
        if detr_method is None:
            if nsig_positive is not None:
                outlier_mask_p = data-np.nanmean(data[indices]) < nsig_positive * np.nanstd(data[indices])
            else:
                outlier_mask_p = np.ones(len(data)).astype(bool)
            if nsig_negative is not None:
                outlier_mask_n = data-np.nanmean(data[indices]) >- nsig_negative * np.nanstd(data[indices])
            else:
                outlier_mask_n = np.ones(len(data)).astype(bool)
            outlier_mask = outlier_mask_p * outlier_mask_n
        else:
            detr = detrend_amat(data, length=detr_length, mask=mask, method=detr_method, endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen, 
                                return_trend=False, keep_overall_level=detr_keep_overall_level)
            if nsig_positive is not None:
                outlier_mask_p = detr-np.nanmean(detr[indices]) < nsig_positive * np.nanstd(detr[indices])
            else:
                outlier_mask_p = np.ones(len(data)).astype(bool)
            if nsig_negative is not None:
                outlier_mask_n = detr-np.nanmean(detr[indices]) >- nsig_negative * np.nanstd(detr[indices])
            else:
                outlier_mask_n = np.ones(len(data)).astype(bool)
            outlier_mask = outlier_mask_p * outlier_mask_n
    return outlier_mask  


def veto_combined(data, template, fn=0.01, maxdev_std=4., maxdev_ampl=None, 
                  zero_ends_len_abs=3, zero_ends_len_rel=0.1, 
                  return_amplitude=True, return_reason=True):
    """
    Checks the goodness of fit of data by the template by chi squared and maximum deviation. 
    Fits the best amplitude for the template. Performs a chi squared test with this 
    template and the maximum deviation test. The conditions for the fit to be good are:
        - The chi squared value is smaller than the point where the chi squared distribution with the number 
          of degrees of freedom equal to the lengts of the template (truncated from extra zeros) is fn 
        - Maximum relative deviation (in units of std) is smaller than maxdev
    If both tests are satisfied, returns True. Otherwise False. If return_amplitude,
    the amplitude from the fitting is also returned in absolute units.
    Attention: NO NANS are allowed.
    
    Parameters
    -----------------
    data : numpy.array 1d of float
        Data to be compared with the template
    template : numpy.array 1d of float
        Template (one-transit, positive-valued, not fft-shifted) to which the data
        should be compared (indices correspond to data indices). The template will be truncated
        to cut extra zeros (one extra zero from each side will be left to show that the template ended
        and reduce the score for peaks that continue beyond the template shape).
    fn : float. 
        False negative probability for chi squared test with known amplitude.
    maxdev_std : float. 
        Relative (in units of std) maximum deviation for the maximum deviation test.
    maxdev_ampl : float. 
        Relative (in units of maximal value of fitted signal) maximum deviation 
        for the maximum deviation test.
    zero_end_len_abs : int or None
        Length of zeros to take at the end of the template from each side in addition
        to the non-zero region of the template. Is needed to make sure that the shape
        of the templite is intdeed fulfilled. Is dominant over zero_end_len_rel.
    zero_end_len_rel : float
        If zero_end_len_abs is None, is used to calculate new zero_end_len_abs
        as [(length of nonzero template)*zero_end_len_rel]. This amount of
        bins is used from each side.  
    return_amplitude : bool
        If True, the fitted amplitude of the transit will be returned together with veto.
    return_reason : bool
        If True, and if the transit is vetoed, the reason why it was vetoed will be returned.
    Returs
    -----------------
    If return_amplitude:
        If both chi2 and maxdev tests are satisfied, returns (True, amplitude).
        Otherwise (False, amplitude)
    Else:
        If both chi2 and maxdev tests are satisfied, returns True. Otherwise False
    """
    from scipy.stats import chi2
    from scipy.optimize import minimize
    result= True
    reason = '' #if the signal will be vetoed, the reason will be reported
    inds = np.where(~np.isclose(template, 0))[0]#[[0,-1]]
    assert len(data) - len(inds) >= 4
    if len(data[0:inds[0]])+len(data[inds[-1]+1:]) != len(data)-len(inds):
        warnings.warn('Warning: template indices are not together, template is disconnected')
        
    #subtract linear trend
    inds_outside = np.array([i for i in range(len(data)) if i not in inds])
    data_outside = np.array(data[inds_outside])
    lin_sol = minimize(lambda x: np.sum((data_outside-x[0]-inds_outside*x[1])**2), x0=[0.,0.]) #find linear trend
    data = data - lin_sol.x[0] - lin_sol.x[1] * np.arange(len(data))
    data_left = data[np.max([0,inds[0]-20]):inds[0]]
    data_rignt = data[inds[-1]:np.min([inds[-1]+10, len(data)])]
    if np.abs(np.mean(data_left) - np.mean(data_rignt)) >= 2*((np.var(data_left)*len(data_left) 
                                + np.var(data_rignt)*len(data_rignt))/((len(data_left)+len(data_rignt))))**0.5:
        reason += 'step '
        result = False #check that it is not a step instead of transit
    data_outside = np.array(data[inds_outside])
    std_dat = np.std(data_outside)
    
    if zero_ends_len_abs is None:
        zero_ends_len_abs = int(np.round(len(inds)*zero_ends_len_rel))
    assert zero_ends_len_abs>=1, 'Incorrect zero_end_len_abs='+str(zero_ends_len_abs)+'! Sould be >=1'
    spare_ind_left = 0
    spare_ind_right = 0
    if inds[0]>0: #if possible, include some zeros in the ends where the template is zero to ensure that the shape goes to zero
        spare_ind_left = min(zero_ends_len_abs,inds[0])
        inds = np.concatenate((np.arange(inds[0]-spare_ind_left,inds[0]), inds))
    if inds[-1]<len(template)-1: 
        spare_ind_right = min(zero_ends_len_abs,len(template)-1-inds[-1])
        inds = np.concatenate((inds, np.arange(inds[-1]+1,inds[-1]+1+spare_ind_right)))
        
    temp = np.array(template[inds])
    dat = np.array(data[inds])
    if (spare_ind_left>0) and (spare_ind_right>0):
        dat = dat - np.mean(np.concatenate((dat[0:spare_ind_left], dat[len(dat)-spare_ind_right:]))) #although we detrended, we subtract local median
    elif spare_ind_left>0:
        dat = dat - np.mean(dat[0:spare_ind_left]) #although we detrended, we subtract local median
    elif spare_ind_right>0:
        dat = dat - np.mean(dat[len(dat)-spare_ind_right:]) #although we detrended, we subtract local median   
     
    df = len(temp)
    x0 = chi2.ppf(1-fn, df)
    sol = minimize(lambda x: np.sum((dat-temp*x)**2), np.max(np.abs(dat))*np.sign(np.max(np.abs(dat))))
    ampl=sol.x[0]
    xi2 = sol.fun/std_dat**2
    noise = dat-temp*ampl

    maxdev_std_dat = np.max(np.abs(noise)/std_dat)
    maxdev_ampl_dat = np.max(np.abs(noise))/np.max(np.abs(temp*ampl))
        
    if xi2>x0:
        reason += 'chi_squared '
        result = False
    if maxdev_std_dat>maxdev_std:
        reason += 'maximal_deviation_std '
        result = False
    if ((maxdev_ampl is not None) and (maxdev_ampl_dat>maxdev_ampl)):
        reason += 'maximal_deviation_amplitude '
        result = False
        
    if return_amplitude:
        if return_reason:
            return result, ampl, reason
        else:
            return result, ampl
    else:
        if return_reason:
            return result, reason
        else:
            return result
            

def connect_quarters_fun(time_quarters, data_quarters, zero_padding_ends=None, is_data_cut=False,
                     is_time_cut=False, dt=None, fill_gaps_by_nan=False):
    """
    Takes list of time arrays for each quarter and list of data arrays for aech quarter. 
    Cuts zero padding ends. Generates new time array from the smallest up to the biggest 
    time of time_quarters. Time quarters are assumed to be equally sampled with the same 
    time step dt (will be measured if not given), and quarters are assumed not to be 
    shifted with respect to each other by a non-integer number of bins. If they
    are, they will be rounded to the closest integer bins of dt. 
    Data quarters are added to the positions of the resulting array corresponding
    to the first time bin of their quarter. The gaps are filled by zeros or NaNs

    Parameters
    ----------
    time_quarters : list of numpy.arrays 1d
        Time by quarters. Should be equally sampled with the same 
        time step `dt` (will be measured if not given), and quarters are assumed 
        not to be shifted with respect to each other by a non-integer number 
        of bins. If they are, they will be rounded to the closest integer bins 
        of dt. Times cut from zero padding ends should not overlap.
    data_quarters : list of numpy.arrays 1d
        Data by qharters corresponding to `time_quarters`.
    zero_padding_ends : list of tuples (int, int), >0, or None
        Lengths of padded pieces from the beginning and from the end to be cut.
    is_data_cut : bool
        Whether the `data_quarters` were cut from `zero_padding_ends` before.
    is_time_cut : bool
        Whether the `time_quarters` were cut from `zero_padding_ends` before.
    dt : float or None
        Time step of sampling in `time_quarters`, should be the same in all
        the quarters. If None, will be measured.
    fill_gaps_by_nan : bool
        Whether to fill the gaps between the `data_quarters` by NaNs. If 
        False, will be filled by zeros.

    Returns
    -------
    time_connected :  numpy.array 1d
        New time sample covering all the time_quarters (with cut `zero_padding_ends`)
    data_connected : numpy.array 1d
        Connected `data_quarters` according to their time position with gaps
        filled by zeros or NaNs.

    """
    if zero_padding_ends is None:
        zero_padding_ends = [(0,0)]*len(time_quarters)
    for i in range(len(time_quarters)):
        if np.isnan(time_quarters[i]).any():
            time_quarters[i] = interpolate_time(np.array(time_quarters[i]))
    if dt is None:
        dts = []
        for j in range(len(time_quarters)):
            dts.append(np.abs(np.nanmean((time_quarters[j]-np.roll(time_quarters[j],1))[1:])))  
        dt = np.nanmean(dts)
    if not is_time_cut:
        t1 = time_quarters[0][zero_padding_ends[0][0]]
        t2 = time_quarters[-1][len(time_quarters[-1])-1-zero_padding_ends[-1][-1]]
    else:
        t1 = time_quarters[0][0]
        t2 = time_quarters[-1][-1]
    time_connected = np.arange(t1, t2+dt, dt)
    data_connected = np.zeros(len(time_connected))
    if fill_gaps_by_nan:
        data_connected = data_connected*np.nan
    for q in range(len(data_quarters)):
        
        if not is_data_cut:
            data_q = np.array(data_quarters[q][zero_padding_ends[q][0]:len(data_quarters[q])-zero_padding_ends[q][-1]])
        else:
            data_q = np.array(data_quarters[q])
        if not is_time_cut:
            time_ind = int(np.round((time_quarters[q][zero_padding_ends[q][0]] - t1)/dt))
        else:
            time_ind = int(np.round((time_quarters[q][0] - t1)/dt))
        data_connected[time_ind:time_ind+len(data_q)] = np.array(data_q)
    return time_connected, data_connected    


def write_quarters_to_file(name, var=None, location=None):
    """
    Prints array (list or numpy.array of float) variable var to file named name in the location location.
    If the var is not given, the name of the variable is the same as name

    Parameters
    ----------
    name : str
        Name of the file ('.txt' is not needed). If var is None, it is also 
        the name of the variable to be printed to the file.
    var : list or array
        Variable to be printed to the file. If var is None, name is the name 
        of the variable to be printed to the file.
    location : str
        Location where to create the file. If None, the current local directory
        will be used

    Returns
    -------
    None.

    """
    if name[-4:]=='.txt':
        name = name[-4:]
    if var==None:
        var = globals()[name]
    if location is None:
        location = os.getcwd() 
        fileloc = os.path.join(location, name)+'.txt'
    else:
        fileloc = name +'.txt'
    file = open(fileloc, 'w')
    print(*var, file=file)
    file.close()

    
def read_quarters_from_file(name, location=None, dimension=2, to_array=True):
    """
    Reads an array (numpy.array or list of float, maximum 2d) printed to the txt file 
    named name and returns the output in form of list.

    Parameters
    ----------
     name : str
        Name of the file ('.txt' is not needed). 
    location : str
        Location where to create the file. If None, the current local directory
        will be used

    Returns
    -------
    res : list
        List (possibly with nested lists) of data read from the file.

    """
    if location is None:
        location = os.getcwd() 
        fileloc = os.path.join(location, name)
    else:
        if name[-4:]!='.txt':
            fileloc = name +'.txt'
    file = open(fileloc, 'r')
    res = file.read()
    if dimension<3:
        res = res.replace('array', '')
        res = res.replace('(', '')
        res = res.replace(')', '')
        res = res.replace(',', ' ')
        res = (res.split('] ['))
        for i in range(len(res)):
            res[i] = res[i].replace(']','')
            res[i] = res[i].replace('[','')
            res[i] = res[i].replace('\n','')
            res[i] = list(map(float,res[i].split()))
        if len(res)==1:
            res = res[0]
    elif dimension==3:
        res = res.replace('array', '')
        res = res.replace('(', '')
        res = res.replace(')', '')
        res = res.replace(',', ' ')
        res = res.replace('  ', ' ') #important to do in because 
        #otherwise ']  [' with two spaces can appear (which we can't see)
        res = (res.split(']] [['))
        for j in range(len(res)):
            res[j] = (res[j].split('] ['))
            for i in range(len(res[j])):
                res[j][i] = res[j][i].replace(']','')
                res[j][i] = res[j][i].replace('[','')
                res[j][i] = res[j][i].replace('\n','')
                if to_array:
                    res[j][i] = np.array(list(map(float,res[j][i].split())))
                else:
                    res[j][i] = list(map(float,res[j][i].split()))
        file.close()
    return res


def plot_many(list_times, list_values=None, first_index=0, prefix='quarter', 
              legend=True, xlabel='', ylabel='',title='', separate=False, first_quarter=0):
    """
    Plots a list of values as function of a list of arguments. If they overlap, can 
    separate them from each other. Is designed to plot different quarters of
    Kepler data. The legend can be created depicting the numeration of the quarters.

    Parameters
    ----------
    list_times : list of numpy.arrays or lists
        List of arrays of arguments. If list_values is None, will be used as a 
        list of values to plot.
    list_values : list of numpy.arrays or lists, optional
        List of arrays of values to plot. If None, the list_times will be
        used as list of arrays of values to plot.
    first_index : int
        The index from which to start the slice of data to be plotted. Will be
        used for all the arrays in the list of data. Is relevant for example to
        plot in the rfft space without displaying the zeroth sample.
    prefix : str
        Prefix for the caption in the legend for all the entries in the data list.
    legend : bool
        Whether to add the legend to the plot.
    xlabel : str
        Caption of the x-axis of the plot.
    ylabel : str
        Caption of the y-axis of the plot.
    title : str
        Title of the plot.
    separate : bool
        If true, the plots will be shifted in y-direction in order not to overlap.
    first_quarter : int
        Index of the first quarter of data in the list (needed only for captions)

    Returns
    -------
    None.

    """
    if list_values is not None:    
        if separate==False:
            sep = np.zeros(len(list_times))
        elif separate==True:
            sep=[0]
            for i in range(len(list_times)-1):
                sep.append(sep[-1]+np.max(list_values[i])-np.min(list_values[i+1]))
        else:
            sep = [i*separate for i in range(len(list_times))]
        for i in range(len(list_times)):
            plt.plot(np.real((list_times[i])[first_index:]), 
                     np.real(sep[i] +(list_values[i])[first_index:]), label=f'{prefix}:{i+first_quarter}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    else:    
        if separate==False:
            sep = np.zeros(len(list_times))
        elif separate==True:
            sep=[0]
            for i in range(len(list_times)-1):
                sep.append(sep[-1]+np.max(list_times[i])-np.min(list_times[i+1]))
        else:
            sep = [i*separate for i in range(len(list_times))]
        for i in range(len(list_times)):
            plt.plot(np.real(sep[i]+(list_times[i])[first_index:]), label=f'{prefix}:{i+first_quarter}')
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


def histogram(data, n=25, edgelist=None, edgemin=None, edgemax=None, xscale=None, yscale=None,
              density=False, normalize=True, ylogshift=True, logzeros='zero', 
              return_results=False, plot=True, new_figure=True, ax=None, figsize=(9,6), show=False,
              xlabel=None, ylabel=None, title=None, xlim=None, ylim=None,
              color='b', linestyle='solid', marker='', linewidth=None, alpha=0.9, label=None, 
              loc_legend=None, plot_type='plot',
              ):
    """
    xscale : str, None, 'square', 'log', 
    yscale : str, None, 'square', 'log'
    plot_type: str, 'bar', 'plot'
    logzeros : what to do with log(counts) where counts==0. 
        Options: 'zero' : put to zero
                 'nan' : put to np.nan
                  None : to do nothing, they will give -np.inf
    
    """    
    if xscale=='square' or xscale=='log':
        data = np.abs(data)
        
    if edgelist is None:
        if edgemin is None:
            edgemin = np.min(data)
        if edgemax is None:
            edgemax = np.max(data)
        edgelist = [edgemin+(edgemax-edgemin)/n * i for i in range(n+1)]
    
    counts, bins = np.histogram(data, bins=edgelist, density=density)
    bins=np.delete(bins,-1)+(bins[1]-bins[0])/2
    
    if xscale=='square':
        bins = np.abs(np.array(bins))**2
    elif xscale=='log':
        bins = np.log(np.abs(np.array(bins)))
    
    if (not density) and normalize:
        counts = counts/np.sum(counts)
    
    if yscale=='square':
        counts = counts**2
    elif yscale=='log':
        if logzeros=='zero':
            indzeros = np.where(counts==0)[0]
            logcounts = np.zeros(len(counts))
            logcounts[counts!=0] = np.log(counts[counts!=0])
            logcounts[counts==0] = 0.
            counts = logcounts
        elif logzeros=='nan':
            logcounts = np.zeros(len(counts))
            logcounts[counts!=0] = np.log(counts[counts!=0])
            logcounts[counts==0] = np.nan
            counts = logcounts
        elif logzeros is None:
            counts = np.log(counts)
        else:
            raise AssertionError('Undefined logzeros parameter specified!')
        if ylogshift:
            counts = counts - np.nanmin(counts[counts!=-np.inf])
            if logzeros=='zero':
                counts[indzeros] = 0

    if plot:
        if new_figure:
            plt.figure(figsize=figsize)
        if ax is None:
            ax=plt.gca()

        if plot_type=='plot':
            ax.plot(bins, counts ,color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, alpha=alpha, label=label)
        elif plot_type=='bar':
            # binwids = (edgelist[-1]-edgelist[0])/(len(edgelist)-1)
            # ax.bar(bins, counts, width=binwids ,color=color, alpha=alpha, label=label)
            # plt.step(bins, counts, where='mid', color=color, alpha=alpha, label=label)
            plt.fill_between(bins, counts, step="mid", color=color, alpha=alpha)

        if label is not None:
            if loc_legend is not None:
                ax.legend(loc=loc_legend)
            else:
                ax.legend()
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()
            
    if return_results:
        return bins, counts 
    
    
