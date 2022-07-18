#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:11:32 2020

@author: orynaivashtenko
version January 2022


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
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation as mad
from scipy.stats import chi2
from scipy.special import erfinv
from scipy.interpolate import interp1d

#from scipy.signal import savgol_filter

##############################################################################
"""
Quick example of usage:
----------------------------
klc = pip.KeplerLightCurve(target_id='000757280', get_data=True, data_location='/home/orynai/Kepler_Notebooks/Data_sample_gaia_1', scramble=False,
                          output_location='/SMALLDATA/orynai/trial_outputs')
klc.condition_data()
klc.condition_templates()
klc.detect_single_transits()
klc.periodicity_search(Pmin=50, Pmax=52)
klc.determine_mins()
----------------------------

If conditioned data is loaded:
----------------------------
klc1 = pip.KeplerLightCurve(target_id='000757280', get_data=True, data_location='/home/orynai/Kepler_Notebooks/Data_sample_gaia_1', scramble=False,
                          output_location='/SMALLDATA/orynai/trial_outputs')
klc1.load_conditioned_data('/SMALLDATA/orynai/trial_outputs/000757280_conditioned_data.npy')
klc1.initialize_templates()
klc1.detect_single_transits()

"""
##############################################################################


#PARAMETERS INITIALIZATION

#Masking:
nsig_outliers_0 = 5.
nsig_outliers_0_pos = 4. #it is possible to mask positive and negatve outliers differently
nsig_outliers_0_neg = 5.
nsig_outliers_psd = 4
max_small_hole = 3
nsig_outliers_w = 5
nsig_outliers_w_pos = 5.
nsig_outliers_w_neg = 5.
nsig_outliers_stat = 3.
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
smoothen_psds = True

#Notch:
nsig_notch = 5.
irl_notch = 3000.
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
margin_snr_veto = 3.
fn_fit_veto=0.01
maxdev_fit_veto=4.0
zero_ends_len_abs_fit_veto = None # this means that zero_ends_len_rel_fit_veto will be used
zero_ends_len_rel_fit_veto = 1 # this means that it will be equal to the 1 transit (template) duration
threshold_mes_veto = 5.
veto_threshold_for_mes = 2.05 #false alarm 5%
n_parts_veto_snr = 4
n_parts_veto_len = 6
inn_prod_min_veto = 0.2
delta_snr2_wide_templ_veto = 4. #should correspond to 2 sigma
threshold_to_std_veto = 35**0.5
time_chunk_minimization_size = 20

apply_score_correction = True

#Processing the statistic
nperseg_moving_var = 512
#->make sure it is smallet than period
ngap_moving_var = 192
nsamples_min_moving_var = 128

duty_cycle = 0.1

Pmin = 30  #days
Pmax = 500

std_time_scrambling = 6. #days

n_per_temp_saving = 10 #how many best candidates per template to save

track_process = False

#data_location = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data')
#if 'Outputs_of_pipeline' not in os.listdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))):
#    try:
#        os.mkdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Outputs_of_pipeline'))
#    except OSError:
#                warnings.warn("Creation of the output directory failed")
        
default_output_location = os.getcwd()#os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Outputs_of_pipeline')
default_data_location = os.getcwd()

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
#TODO: think about things:
    # not full number of quarters
    # what if the template is not symmetric


class KeplerLightCurve(object):  
    def __init__(self, target_id=None, times=[], fluxes=[], get_data=True, data_location=default_data_location, 
                 scramble=False, std_time_scrambling=std_time_scrambling, 
                 output_location=default_output_location,
                 remove_zeroth_quarter=True): #, load_data_from_file=False
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
        self.output_location = output_location
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
        
        #REMOVE ZEROTH QUARTER
        if remove_zeroth_quarter:
            self.orig_times = self.orig_times[1:]
            self.orig_fluxes = self.orig_fluxes[1:]
        
        if scramble:
            np.random.seed()
            r = np.abs(np.random.normal(0,std_time_scrambling,len(self.orig_times)))
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
        
        
        self.masks = [~np.isnan(self.fluxes[i]) for i in self.quarters] #mask things which need to be removed from data
        self.clean_masks = copy.deepcopy(self.masks) #mask all the oultiers, needed for PSD and filters, might kill transits
        self.psd_masks = copy.deepcopy(self.masks) #clean_masks with maybe only big holes to have enough slices for PSD
        self.big_hole_masks = [np.ones(len(self.masks[q])).astype(bool) for q in self.quarters]
        self.stat_masks = []
        self.veto_masks = []
        
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
        self.T_mission = self.tmax - self.tmin #NOT CHANGED IN TIME SLIDES
        self.dt_sec = self.dt_days*24*60*60
        self.is_fluxes_conditioned=False
        self.is_zero_padded=False
        self.is_fluxes_whitened=False
        self.is_fluxes_inpainted = False
        self.is_holes_interpolated = False
        self.is_templates_initialized = False
        self.is_prepared_for_psd = False
        self.is_template_whitened = False
        Tmax = np.max(np.array([len(t) for t in self.orig_times])) #maximum length of the time
        # the closest (but bigger) power of 2 to the length        Tmax += tm.L_ban
        self.support = 2**int(np.ceil(np.log2(Tmax))) 
        if use_rfft:
            self.fft_support = self.support//2+1
        else:
            self.fft_support = self.support
        
        self.whitened_fluxes_td = np.zeros((len(self.fluxes), self.support))
        self.whitened_templates_td = None
        
        self.inpaint_fluxes_fast = numba.njit(inpaint_fluxes_fun)
        # initialization of the numba function on some small data in order  not to
        # spend time on big data:
        _ = self.inpaint_fluxes_fast(fluxes=np.random.normal(0,1,(2,32)),
                                     masks=np.random.choice([0,1],(2,32)).astype(bool),
                                     wf_tds=np.array([np.concatenate((np.zeros(32//2-2),
                                        [-0.5,1,1,-0.5],np.zeros(32//2-2))) for i in range(2)]))
        
        #numba.set_num_threads(1)
        self.sum_periods_fast = numba.njit(sum_periods_fun, parallel=False, fastmath=False)
        # initialization of the numba function on some small data in order  not to
        # spend time on big data:        
        _ = self.sum_periods_fast(statistics_by_trends=np.ones((2, 192)),
                                  trends_sq_by_vars=np.ones((2, 192)) , dt=0.02,
                                  tmax=3., tmin=0., periods=np.arange(1., 2., 0.5))
        

    def condition_data(self, nsig_outliers_0=nsig_outliers_0, nsig_outliers_0_pos=nsig_outliers_0_pos, nsig_outliers_0_neg=nsig_outliers_0_neg,
                       nsig_outliers_psd=nsig_outliers_psd, nperseg_psd=nperseg_psd, noverlap_psd=noverlap_psd, 
                       window_psd=window_psd, npsdsmin=npsdsmin, subtract_mean_in_psd=subtract_mean_in_psd, 
                       nsig_notch=nsig_notch, use_rfft=use_rfft, 
                       nsig_outliers_w=nsig_outliers_w, nsig_outliers_w_pos=nsig_outliers_w_pos, nsig_outliers_w_neg=nsig_outliers_w_neg,
                       max_small_hole=max_small_hole, track_process=track_process):
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
            self.subtract_means(save_means=True, make_trends=True)
            self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg, 
                                    nsig_clean=nsig_outliers_psd, in_whitened_fluxes=False, modify_clean_masks=True)
            self.pad_by_zeros()
            self.make_big_hole_masks(max_small_hole=max_small_hole, exclude_individual_points=True)
            self.interpolate_fluxes()
            self.subtract_low_trends(nperseg=nperseg_psd)
            self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg, 
                                    nsig_clean=nsig_outliers_psd, in_whitened_fluxes=False, modify_clean_masks=True)
            self.interpolate_fluxes(interpolate_clean_fluxes=False) #need to do this again after piecewise detrending
            self.prepare_for_psd(nperseg=nperseg_psd, noverlap=noverlap_psd, npsdsmin=npsdsmin)
            self.measure_psds(nperseg=nperseg_psd, noverlap=noverlap_psd, window=window_psd, subtract_mean=subtract_mean_in_psd, 
                              rfft=use_rfft, smoothen_psds=smoothen_psds)
            self.whiten_fluxes(whiten_clean_fluxes=True)
            self.mask_flux_outliers(nsig=nsig_outliers_w, nsig_positive=nsig_outliers_w_pos, nsig_negative=nsig_outliers_w_neg, 
                                    nsig_clean=nsig_outliers_psd, in_whitened_fluxes=True, modify_clean_masks=True)
            self.interpolate_fluxes()
            self.whiten_fluxes(whiten_clean_fluxes=True)
            self.notch_fluxes(nsig=nsig_notch)
            self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg,
                                    nsig_clean=nsig_outliers_psd, in_whitened_fluxes=False, modify_clean_masks=True)
            self.make_big_hole_masks(max_small_hole=max_small_hole)
            self.inpaint_fluxes() 
            self.whiten_fluxes(whiten_clean_fluxes=True)
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
            self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg, 
                                    nsig_clean=nsig_outliers_psd, in_whitened_fluxes=False, modify_clean_masks=True)
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
            self.prepare_for_psd(nperseg=nperseg_psd, noverlap=noverlap_psd, npsdsmin=npsdsmin)
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
            self.whiten_fluxes(whiten_clean_fluxes=True)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            Masking outliers in whitened fluxes. Effectively we are making detection in the whitened
            data with whitened template, so outliers are unwanted there:
            """
            print('Masking outliers...', end=' ')
            self.mask_flux_outliers(nsig=nsig_outliers_w, nsig_positive=nsig_outliers_w_pos, nsig_negative=nsig_outliers_w_neg, 
                                    nsig_clean=nsig_outliers_psd, in_whitened_fluxes=True, modify_clean_masks=True)
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
            self.whiten_fluxes(whiten_clean_fluxes=True)
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
            self.mask_flux_outliers(nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg,
                                    nsig_clean=nsig_outliers_psd, in_whitened_fluxes=False, modify_clean_masks=True)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            All the bad points ever masked are filled using the optimal inpainting filter which zeroes 
            the impact of these point to the statistic. Note: PSD should not be remeasured after 
            inpainting because the same C^{-1} has to be used for inpainting and for detection: 
            """
            print('Inpainting...', end=' ')
            self.make_big_hole_masks(max_small_hole=max_small_hole)
            self.inpaint_fluxes() 
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            """
            These fluxes will be used for detection:
            """
            print('Whitening...', end=' ')
            self.whiten_fluxes(whiten_clean_fluxes=True) 
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Total time: ', t2-t0, ' sec')
            self.is_fluxes_conditioned=True
            _ = os.system('clear')

    
    def save_conditioned_data(self, path=None):
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
        path : str
            path to the folder in which the data should be saved.

        Returns
        -------
        None.

        """
        if path is None:
            path=self.output_location
            
        len_data = sum([self.support for q in self.quarters]) #is assumed to be support length times number of quarters
        conditioned_data = np.zeros((7+self.templ_num_shifts, len_data), dtype=np.float32)
        conditioned_data[0][10] = self.support
        conditioned_data[0][11] = self.templ_num
        conditioned_data[0][12] = self.templ_num_shifts
        for tn in range(self.templ_num):
            conditioned_data[0][13+tn] = self.indices_of_templates[tn][0]
        ind_last = 0
        for q in self.quarters:
            conditioned_data[0][ind_last + 1] = self.zero_padding_ends[q][0]
            conditioned_data[0][ind_last + 2] = self.zero_padding_ends[q][1]
            conditioned_data[0][ind_last + 3] = len(self.whitened_fluxes_td[q]) #this depends on whether data is cut from zeros or not
            conditioned_data[0][ind_last + 4] = len(self.orig_fluxes[q])
            conditioned_data[1][ind_last:ind_last+len(self.fluxes[q])] = self.times[q]
            conditioned_data[2][ind_last:ind_last+len(self.fluxes[q])] = self.fluxes[q]
            conditioned_data[3][ind_last:ind_last+len(self.whitened_fluxes_td[q])] = self.whitened_fluxes_td[q]
            conditioned_data[4][ind_last:ind_last+self.support] = self.wf_td[q]
            conditioned_data[5][ind_last:ind_last+len(self.fluxes[q])] = self.big_hole_masks[q]
            conditioned_data[6][ind_last:ind_last+len(self.trends_orig_fluxes[q])] = self.trends_orig_fluxes[q]
            for k in range(self.templ_num_shifts):
                conditioned_data[k+7][ind_last:ind_last+len(self.whitened_templates_td[k][q])] = self.whitened_templates_td[k][q]
            ind_last += self.support
        if self.target_id is not None:
            np.save(os.path.join(path, str(self.target_id)+'_conditioned_data'), np.array(conditioned_data, dtype=object), allow_pickle=True)
        else:
            np.save(os.path.join(path,'conditioned_data'), np.array(conditioned_data, dtype=object), allow_pickle=True)
    

    def load_conditioned_data(self, filename, no_zero_padding=False):
        """
        Only works for data created by `save_conditioned_data`. 

        Parameters
        ----------
        filename : str
            path to the .npy file where the data is stored.

        Sets
        -------
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

        """
        if filename is None:
            filename = os.path.join(self.output_location, str(self.target_id)+'_conditioned_data.npy')
        else:
            assert filename[-4:]=='.npy', 'file should have .npy type!'
        conditioned_data = np.load(filename, allow_pickle=True)
        self.support = int(conditioned_data[0][10])
        self.templ_num = int(conditioned_data[0][11])
        self.templ_num_shifts = int(conditioned_data[0][12])
        self.indices_of_templates = []
        for tn in range(self.templ_num-1):
            self.indices_of_templates.append([conditioned_data[0][13+tn], conditioned_data[0][14+tn]])
        self.indices_of_templates.append([conditioned_data[0][self.templ_num-1], self.templ_num_shifts])
        
        self.zero_padding_ends = [[0,0] for q in self.quarters]
        self.times = []
        self.fluxes = []
        self.whitened_fluxes_td = []
        self.whitened_fluxes_fd = []
        self.wf_td = []
        self.wf_fd = []
        self.big_hole_masks = []
        self.trends_orig_fluxes = []
        self.whitened_templates_td = [[] for k in range(len(conditioned_data)-7)]
        self.whitened_templates_fd = [[] for k in range(len(conditioned_data)-7)]
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
            self.times.append(np.array(conditioned_data[1][ind_last : ind_last + lenflux]).astype(float))
            self.fluxes.append(np.array(conditioned_data[2][ind_last : ind_last + lenflux]).astype(float))
            whfltd1 = np.array(conditioned_data[3][ind_last : ind_last + lenflux]).astype(float)
            if ((not self.is_zero_padded) and (not no_zero_padding)):
                whfltd1 = np.concatenate((np.zeros(zpe1), whfltd1, np.zeros(zpe2)))
            self.whitened_fluxes_td.append(whfltd1)
            self.whitened_fluxes_fd.append(np.fft.rfft(whfltd1))
            wftd1 = np.array(conditioned_data[4][ind_last : ind_last + self.support]).astype(float)
            self.wf_td.append(wftd1)
            self.wf_fd.append(np.fft.rfft(np.fft.ifftshift(wftd1)))
            self.big_hole_masks.append((conditioned_data[5][ind_last : ind_last + lenflux]).astype(bool))
            self.trends_orig_fluxes.append((conditioned_data[6][ind_last : ind_last + len_orig]).astype(float))
            for k in range(len(conditioned_data)-7):
                whtltd1 = np.array(conditioned_data[k+7][ind_last:ind_last+lenflux]).astype(float)
                if ((not self.is_zero_padded) and (not no_zero_padding)):
                    whtltd1 = np.concatenate((whtltd1[:(len(whtltd1)+1)//2], np.zeros(zpe1+zpe2), whtltd1[(len(whtltd1)+1)//2:])) 
                self.whitened_templates_td[k].append(whtltd1)
                self.whitened_templates_fd[k].append(np.fft.rfft(whtltd1))
            ind_last += self.support
        if ((not self.is_zero_padded) and (not no_zero_padding)):
            for q in self.quarters:
                zerlen1 = self.zero_padding_ends[q][0]
                zerlen2 = self.zero_padding_ends[q][1]
                self.fluxes[q] = np.concatenate((np.zeros(zerlen1), 
                                                  self.fluxes[q], np.zeros(zerlen2) ))
                self.big_hole_masks[q] = np.concatenate((np.zeros(zerlen1), 
                                                  self.big_hole_masks[q], np.zeros(zerlen2))).astype(bool)
                self.times[q] = np.concatenate((np.zeros(zerlen1)+np.nan, 
                                                  self.times[q], np.zeros(zerlen2)+np.nan ))
                self.times[q] = interpolate_time(np.array(self.times[q]), mask=None)
        self.is_zero_padded = True
        self.is_template_whitened = True
    

    def condition_templates(self, template_ind=template_ind_detection, use_rfft=use_rfft):
        """
        TODO: write this. For now, please consult descriptions of all the methods, as well as "Short user documentation"

        Parameters
        ----------
        use_rfft : TYPE, optional
            DESCRIPTION. The default is use_rfft.
        nsig_outliers : TYPE, optional
            DESCRIPTION. The default is nsig_outliers_0.

        Returns
        -------
        None.

        """
        if not track_process:
            self.initialize_templates(templates=None, template_ind=template_ind, template_bank_from_file=True,
                                      templates_subbin_shift=templates_subbin_shift)
            self.notch_templates()
            self.roll_templates()
            self.whiten_templates(use_rfft=use_rfft)
          
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
            self.notch_templates()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Whitening the templates...', end=' ')
            self.roll_templates()
            self.whiten_templates(use_rfft=use_rfft)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            print('Total time: ', t2-t0, ' sec')
            _ = os.system('clear')    


    def detect_single_transits(self, template_ind=template_ind_detection, use_rfft=use_rfft,
                         nsig_outliers_stat=nsig_outliers_stat, margin_snr=margin_snr_veto, 
                         fn=fn_fit_veto, maxdev=maxdev_fit_veto,
                         zero_ends_len_abs=zero_ends_len_abs_fit_veto,
                         zero_ends_len_rel=zero_ends_len_rel_fit_veto, track_process=track_process,
                         apply_score_correction=apply_score_correction):
        """
        TODO: write this. For now, please consult descriptions of all the methods, as well as "Short user documentation"

        Parameters
        ----------
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
            self.matched_filtering(nsig_outliers_stat=nsig_outliers_stat)
            self.cut_zero_padding_ends()
            self.veto_statistics(margin_snr=margin_snr, fn=fn, maxdev=maxdev,
                             rfft_psd=use_rfft, zero_ends_len_abs=zero_ends_len_abs,
                             zero_ends_len_rel=zero_ends_len_rel, 
                             n_parts_veto_snr=n_parts_veto_snr, n_parts_veto_len=n_parts_veto_len)
            if apply_score_correction:
                self.correct_score()
          
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
            self.notch_templates()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Whitening the templates...', end=' ')
            self.roll_templates()
            self.whiten_templates(use_rfft=use_rfft)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Matched filtering...', end=' ')
            self.matched_filtering(nsig_outliers_stat=nsig_outliers_stat)
            self.cut_zero_padding_ends()
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Vetoing the statistics...', end=' ')
            self.veto_statistics(margin_snr=margin_snr, fn=fn, maxdev=maxdev,
                             rfft_psd=use_rfft, zero_ends_len_abs=zero_ends_len_abs,
                             zero_ends_len_rel=zero_ends_len_rel)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Total time: ', t2-t0, ' sec')
            _ = os.system('clear')    


    def periodicity_search(self, Pmin=Pmin, Pmax=Pmax, dP=None, track_process=track_process):
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
            # self.find_moving_averages(interpolate=True, detr_length=detr_length, detr_method=detr_method, 
            #                      detr_endmethod=detr_endmethod, detr_ends=detr_ends, 
            #                      detr_endlen=detr_endlen, keep_overall_level=False)
            self.connect_quarters()
            self.interleave_shifts()
            self.sum_periods(Pmin=Pmin, Pmax=Pmax, dP=dP, duty_cycle=duty_cycle)
            self.veto_mes(mes_threshold=threshold_mes_veto, veto_threshold=veto_threshold_for_mes)
            
        else:
            t1 = time.time()
            t0 = t1
            # print('Detrending original fluxes...', end=' ')
            # self.find_moving_averages(interpolate=True, detr_length=detr_length, detr_method=detr_method, 
            #                      detr_endmethod=detr_endmethod, detr_ends=detr_ends, 
            #                      detr_endlen=detr_endlen, keep_overall_level=False)
            # t2 = time.time()
            # print(np.round(t2-t1,2), 'sec. Done')
            # t1 = t2
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
            self.sum_periods(Pmin=Pmin, dP=dP, Pmax=Pmax, duty_cycle=duty_cycle)
            t2 = time.time()
            print(np.round(t2-t1,2), 'sec. Done')
            t1 = t2
            print('Total time: ', t2-t0, ' sec')
            _ = os.system('clear')
        

#TODO: make sure you remember that function calls on np.arrays create no copy

    def subtract_means(self, save_means=True, method='mean', make_trends=True):
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
        assert method=='mean' or method=='median', "`method` must be 'mean' or 'median'!"
        
        if save_means:
            self.mean_total = 0.
            len_total = 0
            self.means = []
        if make_trends:
            self.trends_orig_fluxes = []
        for q in self.quarters: 
            if method=='mean':
                mean = np.nanmean(self.clean_fluxes[q])
            elif method=='median':
                mean = np.nanmedian(self.clean_fluxes[q])
            if save_means:
                self.means.append(mean)
                l = len(self.clean_fluxes[q][~np.isnan(self.clean_fluxes[q])])
                self.mean_total += mean * l
                len_total += l
            self.fluxes[q] = np.array(self.fluxes[q]-mean)
            self.clean_fluxes[q] = np.array(self.clean_fluxes[q]-mean)
            if make_trends:
                self.trends_orig_fluxes.append(mean * np.ones(len(self.clean_fluxes[q])))
        if save_means:
            self.mean_total = self.mean_total/len_total

    
    def mask_flux_outliers(self, nsig=nsig_outliers_0, nsig_positive=nsig_outliers_0_pos, 
                           nsig_negative=nsig_outliers_0_neg, nsig_clean=nsig_outliers_psd, 
                           detr_length_outliers=detr_length_outliers, detr_method_outliers=detr_method_outliers,
                           detr_endmethod_outliers=detr_endmethod_outliers, 
                           in_whitened_fluxes=False, modify_clean_masks=True):
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
        modify_clean_masks : bool
            If True, self.psd_masks will be modified setting False to the 
            outliers positions. If False, only self.masks will be modified

        Modifies
        -------
        self.masks
        self.psd_masks

        """
        if not in_whitened_fluxes:
            for q in self.quarters: 
                outlier_mask, clean_mask = mask_outliers_fun(np.array(self.fluxes[q]), mask=np.array(self.clean_masks[q]), 
                                                 nsig0=nsig, nsig1=nsig_clean,
                                                 nsig_positive=nsig_positive, nsig_negative=nsig_negative,
                                                 detr_length=detr_length_outliers,
                                                 detr_method=detr_method_outliers, detr_endmethod=detr_endmethod_outliers,
                                                 detr_ends=detr_ends_outliers, detr_endlen=detr_endlen_outliers,
                                                 return_symmetric_mask=True)
                self.masks[q] *=  outlier_mask
                if modify_clean_masks:
                    self.clean_masks[q] *=  clean_mask
        else:
            assert self.is_fluxes_whitened, 'Cannot mask outliers in whitened fluxes because fluxes are nbnot whitened!'
                
            for q in self.quarters: 
                outlier_mask, clean_mask = mask_outliers_fun(np.array(self.whitened_fluxes_td[q]), mask=np.array(self.clean_masks[q]), 
                                                 nsig0=nsig, nsig1=nsig_clean,
                                                 nsig_positive=nsig_positive, nsig_negative=nsig_negative,
                                                 detr_length=detr_length_outliers,
                                                 detr_method=None, detr_endmethod=detr_endmethod_outliers,
                                                 detr_ends=detr_ends_outliers, detr_endlen=detr_endlen_outliers,
                                                 return_symmetric_mask=True)
                self.masks[q] *=  outlier_mask
                if modify_clean_masks:
                    self.clean_masks[q] *= clean_mask
     
    
    def make_big_hole_masks(self, max_small_hole=max_small_hole, exclude_individual_points=False):
        self.big_hole_masks = []
        for q in self.quarters:
            big_mask = clean_small_holes(np.array(self.masks[q]), max_hole=max_small_hole)
            if exclude_individual_points:
                big_mask = np.array(~clean_small_holes(np.array(~big_mask), max_hole=1)) 
                self.masks[q] *= big_mask
                self.clean_masks[q] *= big_mask
            self.big_hole_masks.append(big_mask)
                
            
    #TODO: mask_planet
    def mask_planet(self, planet_ephemeris_params):
        # import template_module
        # flux_par
        pass
    
    def prepare_for_psd(self, nperseg=nperseg_psd, noverlap=noverlap_psd, npsdsmin=npsdsmin,
                        max_small_hole=max_small_hole):
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
        
        self.psd_masks = copy.deepcopy(self.clean_masks)    
        if not self.is_holes_interpolated:
            warnings.warn("Data is not interpolated. Cleaning small holes may cause NaN in the fluxes.")
        if noverlap<1:
            noverlap = int(nperseg*noverlap)
        if noverlap==-1: #an old piece of code for the case is we still use notation of noverlap=-1
            noverlap=nperseg//2
        # Calculating the number of pieces available for psd measurement. If this number is too low, the interpolation
        # over holes will be done first
        self.npsds0 = np.zeros(len(self.fluxes))
        for i in self.quarters:
            for ind1 in range(0,len(self.fluxes[i]) - nperseg, nperseg - noverlap):
                if np.sum((~self.psd_masks[i])[ind1 : ind1 + nperseg])==0:
                    self.npsds0[i] += 1
            if self.npsds0[i]<npsdsmin: #if the number of psd samples is insufficient, first try to interpolate small holes
                mask_big = clean_small_holes(np.array(self.psd_masks[i]), max_hole=max_small_hole) #mask of only big and edge holes
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
                self.clean_fluxes[i] = np.concatenate((np.zeros(zerlen1), 
                                                  self.clean_fluxes[i], np.zeros(zerlen2) ))
                self.masks[i] = np.concatenate((np.zeros(zerlen1), 
                                                  self.masks[i], np.zeros(zerlen2))).astype(bool)
                self.big_hole_masks[i] = np.concatenate((np.zeros(zerlen1), 
                                                  self.big_hole_masks[i], np.zeros(zerlen2))).astype(bool)
                self.clean_masks[i] = np.concatenate((np.zeros(zerlen1), 
                                                  self.clean_masks[i], np.zeros(zerlen2))).astype(bool)
                self.psd_masks[i] = np.concatenate((np.zeros(zerlen1), 
                                                  self.psd_masks[i], np.zeros(zerlen2))).astype(bool)
                self.times[i] = np.concatenate((np.zeros(zerlen1)+np.nan, 
                                                  self.times[i], np.zeros(zerlen2)+np.nan ))
                # NaNs are needed for the 'interpolate_time" function to treat these regions
                # as the ones to be interpolated. The self.masks[i] is not transferred to the
                # time interpolation because not all the masked flux regions are NaNs in time
                # and we do not want to lose precision rejecting data.
                self.times[i] = interpolate_time(np.array(self.times[i]), mask=None)
            self.fluxes = np.array(self.fluxes) #need them to be numpy.arrays for numba inpainting
            self.masks = np.array(self.masks)
                
            self.is_zero_padded = True
    
    
    def measure_psds(self, nperseg=nperseg_psd, noverlap=noverlap_psd, window=window_psd, 
                     subtract_mean=subtract_mean_in_psd, rfft=use_rfft, smoothen_psds=True):
        
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
                    clean_flux = self.clean_fluxes[i]-np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]])
                    psds.append([np.abs(np.fft.rfft(windowdict[window]*(clean_flux)[ind1:ind1+nperseg])
                        )**2 for ind1 in range(0,len(self.clean_fluxes[i])-nperseg,nperseg-noverlap
                                ) if np.sum((~self.psd_masks[i])[ind1:ind1+nperseg])==0])
                self.freqs = [np.fft.rfftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.rfftfreq(nperseg, d=self.dt_sec) for i in self.quarters]
            else:
                for i in self.quarters:
                    clean_flux = self.clean_fluxes[i]-np.nanmean((self.clean_fluxes[i])[self.psd_masks[i]])
                    psds.append([np.abs(np.fft.fft(windowdict[window]*(clean_flux)[ind1:ind1+nperseg])
                        )**2 for ind1 in range(0,len(self.clean_fluxes[i])-nperseg,nperseg-noverlap
                                ) if np.sum((~self.psd_masks[i])[ind1:ind1+nperseg])==0])
                self.freqs = [np.fft.fftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.fftfreq(nperseg, d=self.dt_sec) for i in self.quarters]
        
        else:
            if rfft:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.rfft(windowdict[window]*(self.clean_fluxes[i])[ind1:ind1+nperseg])
                        )**2 for ind1 in range(0,len(self.clean_fluxes[i])-nperseg,nperseg-noverlap
                                ) if np.sum((~self.psd_masks[i])[ind1:ind1+nperseg])==0])
                self.freqs = [np.fft.rfftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.rfftfreq(nperseg, d=self.dt_sec) for i in self.quarters]
            else:
                for i in self.quarters:
                    psds.append([np.abs(np.fft.fft(windowdict[window]*(self.clean_fluxes[i])[ind1:ind1+nperseg])
                        )**2 for ind1 in range(0,len(self.clean_fluxes[i])-nperseg,nperseg-noverlap
                                ) if np.sum((~self.psd_masks[i])[ind1:ind1+nperseg])==0])
                self.freqs = [np.fft.fftfreq(len(time), d=self.dt_sec) for time in self.times]
                self.psdfreqs = [np.fft.fftfreq(nperseg, d=self.dt_sec) for i in self.quarters]
        
        self.npsds = np.array([len(p) for p in psds]) #number of psds used to take the average
        self.psds = [np.mean(p, axis=0) for p in psds]
        self.orig_psds = copy.deepcopy(self.psds)
        
        if smoothen_psds:
            for q in self.quarters:
                fs = np.arange(len(self.psds[q]))
                sel_ind = np.arange(len(self.psds[q]))[-len(self.psds[q])//3*2:]
                
                
                psd_high = np.array(self.orig_psds[q][sel_ind])
                fs_high = np.array(fs[sel_ind])
                
                not_outl = (psd_high-np.nanmedian(psd_high))<2*mad(psd_high, scale='normal')
                
                fs_high = fs_high[not_outl]
                psd_high = psd_high[not_outl]
                p1 = np.median(np.sort(psd_high)[:3])
                # sol = minimize(lambda x: np.sum((psd_high-x[1]*(fs_high-x[0]))**2), [np.mean(psd_high), 0])
                # trend =  sol.x[1]*(fs - sol.x[0])
                
                # constructing fit in such a way that we avoid negative psds and increasing psds
                sol = minimize(lambda x: np.sum((psd_high - ( -np.abs(x[1])*(fs_high-fs_high[-1]) + (np.abs(x[0])+p1))  )**2), [0., 0.])
                trend = ( -np.abs(sol.x[1])*(fs-fs_high[-1]) + (np.abs(sol.x[0])+p1) ) 
                
                residuals = self.orig_psds[q] - trend
                mad_residuals = mad(residuals[sel_ind], scale='normal')
                new_psd = []
                if residuals[0]<mad_residuals*3.:
                    new_psd.append(0.)
                else:
                    new_psd.append(residuals[0])
                for i in range(1,len(residuals)-1):
                    if residuals[i]>=mad_residuals*3.:
                        new_psd.append(residuals[i])
                    elif residuals[i]>=2*mad_residuals:
                        if (residuals[i-1]>=mad_residuals*2.) or (residuals[i+1]>=mad_residuals*2.):
                            new_psd.append(residuals[i])
                        else:
                            new_psd.append(0)
                    else:
                        new_psd.append(0)
                if residuals[-1]<mad_residuals*3.:
                    new_psd.append(0.)
                else:
                    new_psd.append(residuals[-1])
                psd_cut = np.array(new_psd)+trend

                # fs = np.arange(len(self.psds[q]))
                # sel_ind = np.arange(len(self.psds[q]))[-len(self.psds[q])//3*2:]
                
                # psd_high = np.array(self.psds[q][sel_ind])
                # fs_high = np.array(fs[sel_ind])
            
                # sol = minimize(lambda x: np.sum((psd_high-x[1]*(fs_high-x[0]))**2), [np.mean(psd_high), 0])
                # trend =  sol.x[1]*(fs - sol.x[0])
                # residuals = self.psds[q] - trend
                
                # # psd_high = np.array(psds[q][sel_ind])
                # # fs_high = np.array(fs[sel_ind])
                
                # # sol = minimize(lambda x: np.sum((psd_high - (x[1]**2/(fs_high+5) + x[0]))**2), [np.mean(psd_high), 0.01])
                # # trend = (sol.x[1]**2/(fs +5) + sol.x[0])
                # # residuals = psds[q] - trend
                
                # std_residuals = np.std(residuals[sel_ind])
                # residuals[residuals<std_residuals*3.]=0.
                # psd_cut = residuals + trend
                # # psd_cut = savgol_filter(psd_cut, window_length=5, polyorder=2) #creates artifacts, even negative
                self.psds[q] = np.array(psd_cut )
            
                
                
        
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


    def interpolate_fluxes(self, interpolate_clean_fluxes=True):
        '''
        Interpolates all the holes in the holes in the fluxes, including big holes and end holes.
        In the end holes, the fluxes are padded by a constant value equal to the first(last) value of the array.
        Masks are not modified because measuring PSD or detecting with this interpolation makes no sense.
        '''
        for q in self.quarters:
            self.fluxes[q] = interpolate_over_mask(np.array(self.fluxes[q]), np.array(self.masks[q]), interp_method='linear', ends='continue',
                                                            mitigate_hole_ends=False, endhole_method='median')
            if interpolate_clean_fluxes:
                self.clean_fluxes[q] = interpolate_over_mask(np.array(self.clean_fluxes[q]), np.array(self.clean_masks[q])*self.big_hole_masks[q], interp_method='linear', ends='continue',
                                                            mitigate_hole_ends=False, endhole_method='median')
            self.is_holes_interpolated = True
        
    
    # def subtract_low_trends(self, nperseg=nperseg_psd):
    #     assert self.is_holes_interpolated, 'The holes should be interpolated before detrending!'
    #     self.trends_low = []
    #     if not self.is_zero_padded:
    #         for q in self.quarters:
    #             if len(self.clean_fluxes[q])-1<=nperseg*4:
    #                 endlen = len(self.clean_fluxes[q])//4-1
    #             else:
    #                 endlen = nperseg*2
    #             trend_low = trend_fft(self.clean_fluxes[q], nperseg=nperseg, endlen=endlen)
    #             self.trends_low.append(trend_low)
    #             self.fluxes[q] = np.array(self.fluxes[q]) - trend_low
    #             self.clean_fluxes[q] = np.array(self.clean_fluxes[q]) - trend_low
    #     else:
    #         for q in self.quarters:
    #             if len(self.clean_fluxes[q][
    #                 self.zero_padding_ends[q][0]:len(self.clean_fluxes[q])-self.zero_padding_ends[q][-1]])-1<=nperseg*4:
    #                 endlen = len(self.clean_fluxes[q][
    #                 self.zero_padding_ends[q][0]:len(self.clean_fluxes[q])-self.zero_padding_ends[q][-1]])//4-1
    #             else:
    #                 endlen = nperseg*2
    #             trend_low = trend_fft(self.clean_fluxes[q][
    #                 self.zero_padding_ends[q][0]:len(self.clean_fluxes[q])-self.zero_padding_ends[q][-1]], nperseg=nperseg, endlen=endlen)
    #             self.trends_low.append(trend_low)
    #             self.fluxes[q] = np.array(self.fluxes[q]) - np.concatenate(
    #                 (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1])))
    #             self.clean_fluxes[q] = np.array(self.clean_fluxes[q]) - np.concatenate(
    #                 (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1])))
    
    
    def subtract_low_trends(self, nperseg=nperseg_psd):
        #TODO: Maybe to implement detrend_piecewise with comparison not to std but to absolute values. maybe use minimax of scipy.spatial.distance.cdist
        if not self.is_holes_interpolated:
            warnings.warn("Data is not interpolated. Detrending canfail or give NaNs.")
        self.trends_low = []
        
        if not self.is_zero_padded:
            for q in self.quarters:
                if len(self.orig_fluxes[q])-1<=nperseg*4:
                    endlen = len(self.orig_fluxes[q])//4-1
                else:
                    endlen = nperseg
                detrended, trend_low = detrend_piecewise(self.clean_fluxes[q], big_mask=self.big_hole_masks[q], mask=self.masks[q], 
                                                         nperseg=nperseg//2*3, endlen=endlen, return_trend=True)
                self.trends_low.append(trend_low)
                self.fluxes[q] = np.array(self.fluxes[q]) - trend_low
                self.clean_fluxes[q] = np.array(detrended)
        else:
            for q in self.quarters:
                if len(self.orig_fluxes[q])-1<=nperseg*4:
                    endlen = len(self.orig_fluxes[q])//4-1
                else:
                    endlen = nperseg
                detrended, trend_low = detrend_piecewise(self.clean_fluxes[q][
                    self.zero_padding_ends[q][0]:len(self.clean_fluxes[q])-self.zero_padding_ends[q][-1]], 
                    big_mask=self.big_hole_masks[q][
                    self.zero_padding_ends[q][0]:len(self.big_hole_masks[q])-self.zero_padding_ends[q][-1]],
                    mask=self.masks[q][
                    self.zero_padding_ends[q][0]:len(self.masks[q])-self.zero_padding_ends[q][-1]],
                    nperseg=nperseg//2*3, endlen=endlen, return_trend=True)
                self.trends_low.append(trend_low)
                self.fluxes[q] = np.array(self.fluxes[q]) - np.concatenate(
                    (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1])))
                self.clean_fluxes[q] = np.array(self.clean_fluxes[q]) - np.concatenate(
                    (np.zeros(self.zero_padding_ends[q][0]), trend_low, np.zeros(self.zero_padding_ends[q][1])))
                
        
    
    
    
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
        assert self.is_fluxes_whitened, "Whitened fluxes are needed for notch!"
        self.unnotched_wf_fd = copy.deepcopy(self.whitened_clean_fluxes_fd)
        sumflux=np.sum(self.whitened_clean_fluxes_fd, axis=0) #add all the quarters to mark repeated spurious frequencies
        sumabs=np.sum(np.abs(self.whitened_clean_fluxes_fd), axis=0)
        mask1 = ((np.abs(sumflux)-np.abs(np.nanmean(sumflux))) < nsig * np.nanstd(sumflux)).astype(float)
        mask2 = ((sumabs-np.nanmean(sumabs)) < nsig * 1.5 * np.nanstd(sumabs)).astype(float)
        inv_mask = (1 - mask1 * mask2).astype(bool)
        # inv_mask = (1 - mask1).astype(bool)
        notch_soss=[] #make a filter for every peak to notch
        for fr in np.where(inv_mask)[0]:
            if not np.isclose(fr,0):
                notch_soss.append(notch_fun(fr, fs=self.support, irl=irl_notch, htol=htol_notch, min_val=minval_notch))
        
        for i in self.quarters: #apply all the filters for all the quarters
            for sos in notch_soss:
                self.fluxes[i]=sosfiltfilt(sos, np.array(self.fluxes[i]), padtype='constant') 
                self.clean_fluxes[i]=sosfiltfilt(sos, np.array(self.clean_fluxes[i]), padtype='constant') 
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
        self.fluxes = self.inpaint_fluxes_fast(self.fluxes, self.masks, self.wf_td)
        self.is_fluxes_inpainted = True
    
    
    def whiten_fluxes(self, whiten_clean_fluxes=False):
        """
        whitens all the fluxes with whitening filter
        """
        assert not (np.array([np.isnan(self.fluxes[i]
                        ).any() for i in self.quarters])).any(), 'there are holes in fluxes, cannot whiten!'
        self.whitened_fluxes_fd = []
        self.whitened_fluxes_td = []
        for q in self.quarters:
            whitened_flux_fd = np.fft.rfft(self.fluxes[q])*self.wf_fd[q]
            self.whitened_fluxes_fd.append(whitened_flux_fd)
            self.whitened_fluxes_td.append(np.fft.irfft(whitened_flux_fd))
            
        if whiten_clean_fluxes:
            self.whitened_clean_fluxes_fd = []
            self.whitened_clean_fluxes_td = []
            for q in self.quarters:
                whitened_clean_flux_fd = np.fft.rfft(self.clean_fluxes[q])*self.wf_fd[q]
                self.whitened_clean_fluxes_fd.append(whitened_clean_flux_fd)
                self.whitened_clean_fluxes_td.append(np.fft.irfft(whitened_clean_flux_fd))
            
        self.is_fluxes_whitened = True
    
    
    
    
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
            How to make slices on the ends of the data. Read details in detrend_mov_ave().
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
            msk = ~np.isnan(self.orig_fluxes[q])
            if interpolate:
                o_f = interpolate_over_mask(np.array(self.orig_fluxes[q]), msk, interp_method='linear', ends='continue',
                                                            mitigate_hole_ends=False, endhole_method='median')
                detr, trend = detrend_mov_ave(np.array(o_f), length=detr_length, mask=None, method=detr_method, 
                             endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen, return_trend=True, 
                             keep_overall_level=False, remove_outliers=remove_outliers, nsig_outliers=nsig_outliers)
            else:
                detr, trend = detrend_mov_ave(np.array(self.orig_fluxes[q]), length=detr_length, mask=msk, method=detr_method, 
                             endmethod=detr_endmethod, ends=detr_ends, endlen=detr_endlen, return_trend=True, 
                             keep_overall_level=False, remove_outliers=remove_outliers, nsig_outliers=nsig_outliers)
            detr[~msk] = np.nan
            self.detrended_orig_fluxes.append(detr)
            self.trends_orig_fluxes.append(trend)
     
            
    
    def initialize_templates(self, templates=None, shifts=None, template_ind='all', 
                             templates_subbin_shift=templates_subbin_shift,
                             template_bank_from_file=template_bank_from_file, 
                             template_bank_file=template_bank_file):
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
            if isinstance(templates[0][0], float):
                self.initial_templates = [[t] for t in templates]
            else:
                self.initial_templates = templates
            if shifts is None:
                self.shifts = [[0.] for i in range(len(self.initial_templates))]
            else:
                self.shifts = shifts
        elif template_bank_from_file:
            self.initial_templates = read_quarters_from_file(template_bank_file, dimension=3)
            self.shifts = read_quarters_from_file(shifts_template_bank_file, dimension=2)
        elif template_ind=='all':
            if templates_subbin_shift:
                self.initial_templates, self.shifts = tm.template_bank_subbin_shift(dt=self.dt_days,
                                          dP=None, T_mission=self.T_mission, Pmin=Pmin,
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
                                          dP=None, T_mission=self.T_mission, Pmin=Pmin,
                                          n_points_int_min=50, return_ordered=True, shifts_num=3,
                                          return_shifts=True)
                self.initial_templates[template_ind] = initial_templates[template_ind]
                self.shifts[template_ind] = shifts[template_ind]
            else:
                self.initial_templates = [tm.template_bank(dt=self.dt_days)[template_ind]]
                self.shifts = [[0.] for i in range(len(self.initial_templates))]
                #self.shifts[template_ind] = [0.]
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
        self.templ_num = len(self.indices_of_templates)
        self.templ_num_shifts = len(self.templates)
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
        for k in range(self.templ_num_shifts): #different template indices and different subbin shifts
            for sos in self.notch_soss:
                self.templates[k] = sosfiltfilt(sos, np.array(self.templates[k]), padtype='constant') 
                # Very important to have zeros in the ends, bacause of padding.
                # Ppadtype 'constant' will pad by the edge value, and we need to ensure it is zero
                # That is why notching has to be done when the template is centered at the center,
                # and only then the template can be rolled.

    
    def roll_templates(self, whitened=False):
        if not whitened:
            for k in range(self.templ_num_shifts): #different template indices
                self.templates[k] = np.roll(self.templates[k], 
                                            -(len(self.templates[k])//2-1))    
        else:
            for k in range(self.templ_num_shifts): #different template indices
                for q in self.quarters:
                    self.whitened_templates_td[k][q] = np.roll(self.whitened_templates_td[k][q], 
                                            -(len(self.whitened_templates_td[k][q])//2-1))
            
    
    
    def whiten_templates(self, use_rfft=use_rfft):
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
        assert self.is_templates_initialized, 'Templates have to be initialized!'
        self.whitened_templates_td = np.zeros((self.templ_num_shifts, len(self.quarters), self.support))
        self.whitened_templates_fd = np.zeros((self.templ_num_shifts, len(self.quarters), self.fft_support), dtype=np.complex128)
        for k in range(self.templ_num_shifts): #different template indices
            for q in self.quarters:
                whitened_temp_fd = np.fft.rfft(self.templates[k])*self.wf_fd[q]
                self.whitened_templates_fd[k][q] = whitened_temp_fd
                if use_rfft:
                    self.whitened_templates_td[k][q] = np.fft.irfft(whitened_temp_fd)
                else:
                    self.whitened_templates_td[k][q] = np.fft.ifft(whitened_temp_fd)
        self.is_template_whitened=True
            
        
    def matched_filtering(self, detr_length_outliers=detr_length_outliers, 
                          nsig_outliers_stat=nsig_outliers_stat, 
                          nperseg_moving_var=nperseg_moving_var, ngap_moving_var=ngap_moving_var, 
                          nsamples_min_moving_var=nsamples_min_moving_var):
        """
        Calculates the matched-filtering statistics of data and templates.
        Cuts the statistic from zero padding ends.
        
        Parameters
        ----------
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
        #self.statistics_abs : h@C**(-1)@d/(h@C**(-1)@h) Statistics in absolute units.
        self.stat_stds : Standard deviation of the self.statistics_raw
        self.stat_denominators : (h@C**(-1)@h)**0.5
            *where h is the template, d is the flux, C is the covariance matrix, @ is the 
            inner product, and the formulae are written schematically. In reality,
            they are implemented with whitened data and fluxes.
        """                
        self.statistics_raw = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for t in range(self.templ_num_shifts)]
        self.statistics_raw_cor = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for t in range(self.templ_num_shifts)] #corrected for moving std and then multiplied back by non-moving std
        self.stat_moving_vars = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for t in range(self.templ_num_shifts)]
        self.statistics_snr = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for t in range(self.templ_num_shifts)]
        #self.statistics_abs = [[np.zeros(len(self.fluxes[q])) for q in self.quarters] for t in range(self.templ_num_shifts)]
        self.stat_vars = [[0. for q in self.quarters] for t in range(self.templ_num_shifts)] #mean of moving variance
        #TODO: think about outlier free moving variance
        self.stat_masks = [[np.ones(len(self.fluxes[q])).astype(bool) for q in self.quarters] for t in range(self.templ_num_shifts)]
        
        assert self.is_template_whitened, 'Templates have to be whitened for matched filtering!'
        
        for k in range(self.templ_num_shifts):
            for q in self.quarters:
                #statistics not in snr, not in absolute units
                stat = np.real(np.fft.irfft(np.conjugate(self.whitened_templates_fd[k][q])*self.whitened_fluxes_fd[q]))
                stat = np.array(stat[self.zero_padding_ends[q][0]:len(stat)-self.zero_padding_ends[q][1]]) #CUT THE STATISTIC
                statmask = np.array(self.big_hole_masks[q][self.zero_padding_ends[q][0]:len(self.big_hole_masks[q])-self.zero_padding_ends[q][1]])
                self.statistics_raw[k][q] = stat       
                # movvar = moving_var(np.array(stat), mask=np.array(statmask), nperseg=nperseg_moving_var ,ngap=ngap_moving_var, nsamples_min=nsamples_min_moving_var,
                #                              remove_outliers=False, remove_outliers_light=True, detr_length_outliers=detr_length_outliers,
                #                              nsig_outliers=nsig_outliers_stat, return_mask=False)
                movvar, tot_var = gap_var(np.array(stat), mask=np.array(statmask), ngap=ngap_moving_var, nsamples_min=nsamples_min_moving_var,
                                              remove_outliers=False, remove_outliers_light=True, detr_length_outliers=detr_length_outliers,
                                              nsig_outliers=nsig_outliers_stat, return_mask=False, return_total_var=True)
                movvar[~statmask] = np.inf
                self.stat_moving_vars[k][q] = movvar
                self.statistics_snr[k][q] = stat/movvar**0.5
                # statrawcor = stat/movvar*tot_var
                statrawcor = stat/movvar*np.nanmean(movvar[statmask])
                statrawcor[~statmask] = 0.
                self.statistics_raw_cor[k][q] = statrawcor
                # self.stat_vars[k][q] = tot_var
                self.stat_vars[k][q] = np.nanmean((movvar[statmask])**0.5)**2
                #self.statistics_abs[k][q] = stat/movvar
                self.stat_masks[k][q] = statmask
    
    
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
            whitened_templates_td = [] #attention: they are rolled
            for q in self.quarters:
                fluxes.append(np.array(self.fluxes[q][
                    self.zero_padding_ends[q][0]:len(self.fluxes[q])-self.zero_padding_ends[q][-1]]))
                clean_fluxes.append(np.array(self.clean_fluxes[q][
                    self.zero_padding_ends[q][0]:len(self.clean_fluxes[q])-self.zero_padding_ends[q][-1]]))
                times.append(np.array(self.times[q][
                    self.zero_padding_ends[q][0]:len(self.times[q])-self.zero_padding_ends[q][-1]]))
                masks.append(np.array(self.masks[q][
                    self.zero_padding_ends[q][0]:len(self.masks[q])-self.zero_padding_ends[q][-1]]))
                clean_masks.append(np.array(self.clean_masks[q][
                    self.zero_padding_ends[q][0]:len(self.clean_masks[q])-self.zero_padding_ends[q][-1]]))
                big_hole_masks.append(np.array(self.big_hole_masks[q][
                    self.zero_padding_ends[q][0]:len(self.big_hole_masks[q])-self.zero_padding_ends[q][-1]]))
                whitened_fluxes_td.append(np.array(
                    self.whitened_fluxes_td[q][
                        self.zero_padding_ends[q][0]:len(self.whitened_fluxes_td[q])-self.zero_padding_ends[q][-1]]))
                # self.detrended_fluxes[q] = np.array(self.detrended_fluxes[q][
                    #self.zero_padding_ends[q][0]:len(self.detrended_fluxes[q])-self.zero_padding_ends[q][-1]])
                # self.trends_fluxes[q] = np.array(self.trends_fluxes[q][
                    #self.zero_padding_ends[q][0]:len(self.trends_fluxes[q])-self.zero_padding_ends[q][-1]])
            if self.whitened_templates_td is not None:
                for k in range(self.templ_num_shifts):
                    whitened_templates_td_template = []
                    for q in self.quarters:
                        wt_td = np.roll(self.whitened_templates_td[k][q], 
                                                      ((len(self.whitened_templates_td[k][q])-1)//2))  
                        wt_td = np.array(
                            wt_td[self.zero_padding_ends[q][0]:len(wt_td) - self.zero_padding_ends[q][-1]])
                        whitened_templates_td_template.append(np.roll(wt_td, -((len(wt_td)-1)//2)))
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
    
            
    def time_slides(self, std_time_scrambling=std_time_scrambling):
        np.random.seed()
        r = np.abs(np.random.normal(0,std_time_scrambling,len(self.times)))
        r = np.cumsum(r)
        for i in range(len(self.times)):
            self.times[i] = np.array(self.times[i]) - self.time_slide_shifts[i] + r[i] #we undo old time slides and do new
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
    
    
    def minimize_over_chunks(self, time_chunk_size=time_chunk_minimization_size):
        """
        Finds the minimal (best, because snr is negative by convention) running statistics
        over subbin shifter templates and over time chunks of length `time_chunk_size`.
        This is done for all the different templates and all the quarters.

        Parameters
        ----------
        time_chunk_size : TYPE, optional
            DESCRIPTION. The default is time_chunk_minimization_size.

        Returns
        -------
        None.

        """
        self.indices_of_min = [[[] for q in self.quarters] for tempnum in range(len(self.indices_of_templates))]
        self.quasi_indep_snr = [[[] for q in self.quarters] for tempnum in range(len(self.indices_of_templates))]
        for tempnum in range(len(self.indices_of_templates)):
            for q in self.quarters:
                for timind in np.arange(0, len(self.statistics_snr[self.indices_of_templates[tempnum][0]][q])-time_chunk_size, time_chunk_size):
                    #TODO:INCLUDE MASK
                    self.indices_of_min[tempnum][q].append(np.argmin([np.min(self.statistics_snr[ti][q][timind:timind+time_chunk_size]) 
                                    for ti in np.arange(self.indices_of_templates[tempnum][0], self.indices_of_templates[tempnum][1])]))
                    self.quasi_indep_snr[tempnum][q].append(np.min([np.min(self.statistics_snr[ti][q][timind:timind+time_chunk_size]) 
                                    for ti in np.arange(self.indices_of_templates[tempnum][0], self.indices_of_templates[tempnum][1])]))
    
            
    def veto_stat_outliers(self, threshold_to_std=threshold_to_std_veto):
        #template_widths = [len(np.where(self.initial_templates[tt]>0)[0]) for tt in range(len(self.initial_templates))]
        whitened_template_widths = [[len(np.where(np.abs(self.whitened_templates_td[tt][q])>0.5*np.abs(np.min(self.whitened_templates_td[tt][q])))[0]) 
                        for q in self.quarters] for tt in range(len(self.initial_templates))]
        for tt in range(len(self.statistics_raw_cor)):
            for q in range(len(self.statistics_raw_cor[tt])):
                half_wid = (whitened_template_widths[tt][q]+1)//2
                var_stat = np.var(self.statistics_raw_cor[tt][q][self.stat_masks[tt][q]])
                outlier_mask = self.statistics_raw_cor[tt][q]**2<3**2*var_stat # for variance measurement only
                var_stat = np.var(self.statistics_raw_cor[tt][q][outlier_mask & self.stat_masks[tt][q]])
                outlier_indices = np.where(self.statistics_raw_cor[tt][q]**2>threshold_to_std**2*var_stat)[0] # the actual measurement
                
                length = len(self.stat_masks[tt][q])
                for ind in outlier_indices:
                    self.stat_masks[tt][q][max(0,ind-half_wid): min(ind+half_wid, length)] = False
        
                                
    def veto_around_min(self):
        #take moving chunks of some like template width.
        #if the min is masked, mask everything.
        whitened_template_widths = [[len(np.where(np.abs(self.whitened_templates_td[tt][q])>0.5*np.abs(np.min(self.whitened_templates_td[tt][q])))[0]) 
                        for q in self.quarters] for tt in range(len(self.initial_templates))]
        for tt in range(len(self.statistics_raw_cor)):
            for q in range(len(self.statistics_raw_cor[tt])):
                wid = whitened_template_widths[tt][q]
                half_wid = (whitened_template_widths[tt][q]+1)//2
                for ind in np.arange(0, len(self.statistics_raw_cor[tt][q])-half_wid, half_wid):
                    indmin = ind + np.nanargmin(self.statistics_raw_cor[tt][q][ind : ind+wid])
                    if not self.stat_masks[tt][q][indmin]:
                        if self.statistics_raw_cor[tt][q][indmin] == np.nanmin(self.statistics_raw_cor[tt][q][
                                    max(0, indmin-half_wid) : min(indmin+half_wid, len(self.statistics_raw_cor[tt][q]))]):
                                self.stat_masks[tt][q][max(0, indmin-half_wid) : min(indmin+half_wid, len(self.statistics_raw_cor[tt][q]))] = False
                    
                        
    def veto_statistics(self, in_maximized=False, margin_snr=margin_snr_veto, fn=fn_fit_veto, maxdev=maxdev_fit_veto,
                        rfft_psd=use_rfft, zero_ends_len_abs=zero_ends_len_abs_fit_veto,
                        zero_ends_len_rel=zero_ends_len_rel_fit_veto, nsig_outliers_var=nsig_outliers_stat,
                        n_parts_veto_snr=n_parts_veto_snr, n_parts_veto_len=n_parts_veto_len, inn_prod_min_veto=inn_prod_min_veto,
                        delta_snr2_wide_templ=delta_snr2_wide_templ_veto, threshold_to_std=threshold_to_std_veto):
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
        """   
            
        assert not self.is_zero_padded, 'Zero padding ends have to be removed bevore vetting!'
        # TODO: when one veto triggered, do not do other vetos, this point will be masked in any case (do it after you debug)
        assert not self.is_zero_padded, 'Zero padding ends have to be removed bevore vetting!'
        self.templates_of_indices = [k for k in range(len(self.indices_of_templates)) for i in 
                                      range(self.indices_of_templates[k][0], self.indices_of_templates[k][1])] #opposite to self.indices_of_templates
        
        self.triggers = [{} for q in self.quarters]
        # self.trigger_amplitudes = [[[] for q in self.quarters] for k in range(self.templ_num_shifts)]
        # self.veto_masks = [[np.ones(len(self.stat_masks[k][q])).astype(bool) for q in self.quarters] for k in range(len(self.stat_masks))]
        for k in range(self.templ_num_shifts):
            template = self.initial_templates[k]
            halflen=len(template)//2
            for q in self.quarters: #over quarters
                # trigger_amplitudes_quarter_0 = []
                statistic = np.array(self.statistics_snr[k][q])
                mask = np.array(self.stat_masks[k][q]) #allow small holes to be in the transit
                # veto_mask = np.ones(len(statistic)).astype(bool)
    
                ## masking the positive-amplitude triggers:
                # positive_trigger_indices = np.where((statistic>0)&(np.abs(statistic)>=margin_snr)&(mask))[0]
                # self.stat_masks[k][q][positive_trigger_indices] = False
    
                # vetoing bad-fitting (negative-amplitude) template triggers:
                if not in_maximized:
                    trigger_indices_quarter = np.where((statistic<=0)&(np.abs(statistic)>=margin_snr)&(mask))[0]
                else:
                    pass
                    #TODO: WRITE THIS (first write the maximization function correctly with argmin)
                    #trigger_indices_quarter = np.array(self.indices_of_min[k][q])
                    
                for i in trigger_indices_quarter:
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
                        if i not in self.triggers[q].keys():
                            self.triggers[q][i] = {k:[False, 'extreme_position ', [], [], [], []]}
                        else:
                            self.triggers[q][i][k] = [False, 'extreme_position ', [], [], [], []]
                        to_veto_combined = False #Already marked inappropriate, will not be treated with veto_combined
                    if to_veto_combined:
                        isgood, scores, reason = veto_combined(data=data_cut, template=template_cut, 
                                fn=fn, maxdev_std=maxdev, maxdev_ampl=None,
                                zero_ends_len_abs=zero_ends_len_abs, zero_ends_len_rel=zero_ends_len_rel, 
                                return_amplitude=False, return_reason=True, inn_prod_min=inn_prod_min_veto, return_scores=True)
                        if i not in self.triggers[q].keys():
                            if isgood:
                                self.triggers[q][i] = {k:[isgood, '', scores, [], [], []]}
                            else:
                                self.triggers[q][i] = {k:[isgood, reason, scores, [], [], []]}
                        else:
                            if isgood:
                                self.triggers[q][i][k] = [isgood, '', scores, [], [], []]
                            else:
                                self.triggers[q][i][k] = [isgood, reason, scores, [], [], []]
    
                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING BY SNR)
    
                isgood_parts, scores, local_th, total_th = veto_parts(np.array(self.whitened_fluxes_td[q]), np.array(self.whitened_templates_td[k][q]),
                                          indices=trigger_indices_quarter, mask=np.array(self.big_hole_masks[q]), tlen=None, fn=fn,
                                          n_parts=n_parts_veto_snr, nsig_outliers_var=nsig_outliers_var, splitting='snr',
                                          return_amplitudes=False, return_scores=True)
                for j in range(len(trigger_indices_quarter)):
                    self.triggers[q][trigger_indices_quarter[j]][k][0] &= isgood_parts[j]
                    if not isgood_parts[j]:
                        self.triggers[q][trigger_indices_quarter[j]][k][1] += 'inconsistent_parts_snr '
                    self.triggers[q][trigger_indices_quarter[j]][k][3] = [scores[j], local_th, total_th]
    
    
                # VETO OF INCONSISTENT PARTS OF TEMPLATE FITTING (SPLITTING BY LENGTH)
                isgood_parts, scores, local_th, total_th = veto_parts(np.array(self.whitened_fluxes_td[q]), np.array(self.whitened_templates_td[k][q]),
                                          indices=trigger_indices_quarter, mask=np.array(self.big_hole_masks[q]), tlen=None, fn=fn,
                                          n_parts=n_parts_veto_len, nsig_outliers_var=nsig_outliers_var,  splitting='len',
                                          return_amplitudes=False, return_scores=True)
                for j in range(len(trigger_indices_quarter)):
                    self.triggers[q][trigger_indices_quarter[j]][k][0] &= isgood_parts[j]
                    if not isgood_parts[j]:
                        self.triggers[q][trigger_indices_quarter[j]][k][1] += 'inconsistent_parts_len '
                    self.triggers[q][trigger_indices_quarter[j]][k][4] = [scores[j], local_th, total_th]
    
        # VETO OF NARROW TEMPLATES TRIGGERING ON WIDER TRANSITS
    
        for q in self.quarters:
            for ind in self.triggers[q].keys():
                snrs = []
                for k in self.triggers[q][ind].keys():
                    snrs.append(self.statistics_snr[k][q][ind])
                timax = np.argmax(np.abs(snrs)) #snr is negative
                tmax = list(self.triggers[q][ind].keys())[timax]
                snrmax2 = snrs[timax]**2
                for ik in range(len(snrs)):
                    k = list(self.triggers[q][ind].keys())[ik]
                    if (k < tmax) and (snrs[ik]**2<snrmax2-delta_snr2_wide_templ):
                        is_good_temp = False
                    else:
                        is_good_temp = True
                    iot = self.indices_of_templates[self.templates_of_indices[k]]
                    for ts in np.arange(iot[0], iot[1]):
                        if ts not in self.triggers[q][ind].keys():
                            if not is_good_temp:
                                self.triggers[q][ind][ts] = [is_good_temp, 'template_width ', [], [], [], [snrs[ik]**2, snrmax2]]
                            else:
                                self.triggers[q][ind][ts] = [is_good_temp, '', [], [], [], [snrs[ik]**2, snrmax2]]
                        else:
                            self.triggers[q][ind][ts][0] &= is_good_temp
                            if not is_good_temp:
                                if 'template_width' not in self.triggers[q][ind][ts][1]:
                                    self.triggers[q][ind][ts][1] += 'template_width '
                            self.triggers[q][ind][ts][5] = [snrs[ik]**2, snrmax2]
        
        self.veto_masks = [[np.ones(len(self.stat_masks[k][q])) for q in self.quarters] for k in range(len(self.stat_masks))]
        self.trigger_masks = [[np.ones(len(self.stat_masks[k][q]))*np.nan for q in self.quarters] for k in range(len(self.stat_masks))]
        self.veto_bad_trigger_inds = [[ [[], []] for q in self.quarters] for k in range(self.templ_num_shifts)]
        
        for q in self.quarters:
            for ind in self.triggers[q].keys():
                for k in self.triggers[q][ind].keys():
                    self.trigger_masks[k][q][ind] = self.triggers[q][ind][k][0]
                    if not self.triggers[q][ind][k][0]:
                        self.veto_bad_trigger_inds[k][q][0].append(ind)
                        self.veto_bad_trigger_inds[k][q][1].append(self.triggers[q][ind][k][1])
                        
        for k in range(len(self.veto_bad_trigger_inds)):
            for q in self.quarters:
                self.veto_masks[k][q][self.veto_bad_trigger_inds[k][q][0]] = False
                self.stat_masks[k][q][self.veto_bad_trigger_inds[k][q][0]] = False
                
        #self.veto_stat_outliers(threshold_to_std=threshold_to_std)
        #self.veto_around_min()
        
        for k in range(len(self.veto_bad_trigger_inds)):
            for q in self.quarters:
                self.statistics_snr[k][q][~self.stat_masks[k][q]] = 0. 
                self.statistics_raw_cor[k][q][~self.stat_masks[k][q]] = 0.
        
                
            
    
    
    def correct_score(self):
        self.statistics_snr_no_rank = copy.deepcopy(self.statistics_snr)
        all_stat = np.array([])
        for k in range(len(self.statistics_snr)):
            for q in range(len(self.statistics_snr[k])):
                all_stat = np.concatenate((all_stat, self.statistics_snr[k][q][self.stat_masks[k][q]]))
        all_stat_c, transform = score_correction(all_stat, return_function=True, negative_values=True)
        for k in range(len(self.statistics_snr)):
            for q in range(len(self.statistics_snr[k])):
                self.statistics_snr[k][q][self.stat_masks[k][q]] = transform(self.statistics_snr[k][q][self.stat_masks[k][q]])
                self.statistics_raw_cor[k][q][self.stat_masks[k][q]] = (self.statistics_snr[k][q][
                    self.stat_masks[k][q]]/self.stat_moving_vars[k][q][self.stat_masks[k][q]] **0.5 
                    * np.nanmean(self.stat_moving_vars[k][q][self.stat_masks[k][q]]))
            
    
    def save_ses_quarters(self, path=None):
        """
        Saves the essential SES results that will be used in the periodicity search
        as .npy file that can be read with `load_ses_quarters` funtion:
            self.times
            self.trends_orig_fluxes
            self.statistics_raw_cor
            self.stat_masks
            self.stat_vars
            
        Parameters
        ----------
        path : str
            path to the folder in which the data should be saved.

        Returns
        -------
        None.

        """
        if path is None:
            path=self.output_location
            
        len_data = sum([self.support for q in self.quarters]) #is assumed to be support length times number of quarters
        ses_quarter_data = np.zeros((4+3*self.templ_num_shifts, len_data), dtype=np.float32)
        ses_quarter_data[0][0] = self.support
        ses_quarter_data[0][1] = self.templ_num
        ses_quarter_data[0][2] = self.templ_num_shifts
        ses_quarter_data[0][3] = self.dt_days
        ses_quarter_data[0][4] = len(self.quarters)
        for tn in range(self.templ_num):
            ses_quarter_data[0][5+tn] = self.indices_of_templates[tn][0]
        ind_last = 0
        for q in self.quarters:
            ses_quarter_data[1][ind_last + 0] = self.zero_padding_ends[q][0]
            ses_quarter_data[1][ind_last + 1] = self.zero_padding_ends[q][1]
            ses_quarter_data[1][ind_last + 2] = len(self.statistics_raw_cor[0][q]) #this depends on whether data is cut from zeros or not
            ses_quarter_data[1][ind_last + 3] = len(self.times[q])
            for tn in range(self.templ_num_shifts):
                ses_quarter_data[1][ind_last + 4 + tn] = self.stat_vars[tn][q]
            ses_quarter_data[2][ind_last:ind_last+len(self.times[q])] = self.times[q]
            ses_quarter_data[3][ind_last:ind_last+len(self.times[q])] = self.trends_orig_fluxes[q]
            row_pos = 4
            for k in range(self.templ_num_shifts):
                ses_quarter_data[k+row_pos][ind_last:ind_last+len(self.statistics_raw_cor[k][q])] = self.statistics_raw_cor[k][q]
            row_pos += self.templ_num_shifts
            for k in range(self.templ_num_shifts):
                ses_quarter_data[k+row_pos][ind_last:ind_last+len(self.statistics_raw_cor[k][q])] = self.stat_masks[k][q]
            row_pos += self.templ_num_shifts
            for k in range(self.templ_num_shifts):
                ses_quarter_data[k+row_pos][ind_last:ind_last+len(self.statistics_raw_cor[k][q])] = self.statistics_snr_no_rank[k][q]
            ind_last += self.support
        if self.target_id is not None:
            np.save(os.path.join(path, str(self.target_id)+'_ses_quarter_data'), np.array(ses_quarter_data, dtype=object), allow_pickle=True)
        else:
            np.save(os.path.join(path,'_ses_quarter_data'), np.array(ses_quarter_data, dtype=object), allow_pickle=True)
    
    
    def load_ses_quarters(self, filename=None):
        """
        Only works for data created by `save_ses_quarters`. Maybe contains assertion bugs.

        Parameters
        ----------
        filename : str
            path to the .npy file where the data is stored.

        Sets
        -------
        self.times
        self.trends_orig_fluxes
        self.statistics_raw_cor
        self.stat_masks
        self.stat_vars
        self.dt_days

        """
        if filename is None:
            filename = os.path.join(self.output_location, str(self.target_id)+'_ses_quarter_data.npy')
        else:    
            assert filename[-4:]=='.npy', 'file should have .npy type!'
        ses_quarter_data = np.load(filename, allow_pickle=True)
        self.support = ses_quarter_data[0][0]
        self.templ_num = ses_quarter_data[0][1]
        self.templ_num_shifts = ses_quarter_data[0][2]
        self.dt_days = ses_quarter_data[0][3]
        self.quarters = range(int(ses_quarter_data[0][4]))
        self.indices_of_templates = []
        for tn in range(self.templ_num-1):
            self.indices_of_templates.append([ses_quarter_data[0][5+tn], ses_quarter_data[0][6+tn]])
        self.indices_of_templates.append([ses_quarter_data[0][6+self.templ_num-1], self.templ_num_shifts])
        ind_last = 0
        self.zero_padding_ends = [[0,0] for q in self.quarters]
        self.whitened_fluxes_td = []
        self.times = []
        self.stat_vars = [[] for k in range(self.templ_num_shifts)]
        self.statistics_raw_cor = [[] for k in range(self.templ_num_shifts)]
        self.stat_masks = [[] for k in range(self.templ_num_shifts)]
        self.statistics_snr_no_rank = [[] for k in range(self.templ_num_shifts)]
        for q in self.quarters:
            self.zero_padding_ends[q][0] = ses_quarter_data[1][ind_last + 0]
            self.zero_padding_ends[q][1] = ses_quarter_data[1][ind_last + 1]
            len_time = ses_quarter_data[1][ind_last + 2]
            len_stat = ses_quarter_data[1][ind_last + 3]
            for tn in range(self.templ_num_shifts):
                self.stat_vars[tn].append(ses_quarter_data[1][ind_last + 4 + tn])
            self.times.append(ses_quarter_data[2][ind_last:ind_last+len_time])
            self.trends_orig_fluxes.append(ses_quarter_data[3][ind_last:ind_last+len_time])
            row_pos = 4
            for k in range(self.templ_num_shifts):
                self.statistics_raw_cor[k].append(ses_quarter_data[k+row_pos][ind_last:ind_last+len_stat])
            row_pos += self.templ_num_shifts
            for k in range(self.templ_num_shifts):
                self.stat_masks[k].append(ses_quarter_data[k+row_pos][ind_last:ind_last+len_stat])
            row_pos += self.templ_num_shifts
            for k in range(self.templ_num_shifts):
                self.statistics_snr_no_rank[k].append(ses_quarter_data[k+row_pos][ind_last:ind_last+len_stat])
            ind_last += self.support
    
    
    
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
        #self.time_full, self.masks_full = connect_quarters_fun(self.times, self.masks,
        #                     zero_padding_ends=None, is_data_cut=~self.is_zero_padded, is_time_cut=~self.is_zero_padded, dt=self.dt_days, fill_gaps_by_nan=False)
        self.time_full, self.trend_orig_flux_full = connect_quarters_fun(self.times, self.trends_orig_fluxes,
                             zero_padding_ends=None, is_data_cut=~self.is_zero_padded, is_time_cut=~self.is_zero_padded, dt=self.dt_days, fill_gaps_by_nan=False)
        self.statistics_full = np.zeros((self.templ_num_shifts, len(self.time_full)))
        self.stat_vars_full = np.zeros((self.templ_num_shifts, len(self.time_full)))
        self.stat_masks_full = np.zeros((self.templ_num_shifts, len(self.time_full))).astype(bool)
        vars_quarters = [[np.ones(len(self.statistics_raw_cor[k][q]))*self.stat_vars[k][q]
                           for q in self.quarters] for k in range(self.templ_num_shifts)]
        
        for k in range(self.templ_num_shifts):
            self.statistics_full[k] = connect_quarters_fun(self.times, self.statistics_raw_cor[k],
                     zero_padding_ends=None, is_data_cut=~self.is_zero_padded, is_time_cut=~self.is_zero_padded, dt=self.dt_days, fill_gaps_by_nan=False)[1]
            self.stat_masks_full[k] = (connect_quarters_fun(self.times, self.stat_masks[k],
                     zero_padding_ends=None, is_data_cut=~self.is_zero_padded, is_time_cut=~self.is_zero_padded, dt=self.dt_days, fill_gaps_by_nan=False)[1]).astype(bool)
            self.stat_vars_full[k] = connect_quarters_fun(self.times, vars_quarters[k],
                     zero_padding_ends=None, is_data_cut=~self.is_zero_padded, is_time_cut=~self.is_zero_padded, dt=self.dt_days, fill_gaps_by_nan=False)[1]

        
    
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
        #del self.statistics_abs #not used anymore
        if hasattr(self, 'stat_moving_vars'):
            del self.stat_moving_vars
        if hasattr(self, 'notch_soss'):
            del self.notch_soss
        if hasattr(self, 'wf_td'):
            del self.wf_td
        if hasattr(self, 'wf_fd'):
            del self.wf_fd
        #del self.statistics_full #will be deleted later
        #del self.time_full #will be deleted later
        if hasattr(self, 'psd_masks'):
            del self.psd_masks
        #del self.zero_padding_ends
        if hasattr(self, 'notch_mask'):
            del self.notch_mask
        if hasattr(self, 'templates'):
            del self.templates
        
        if hasattr(self, 'npsds0'):
            del self.npsds0
        if hasattr(self, 'npsds'):
            del self.npsds
        
        # del self.is_fluxes_conditioned
        # del self.is_zero_padded
        # del self.is_fluxes_whitened
        # del self.is_fluxes_inpainted 
        # del self.is_holes_interpolated 
        # del self.is_templates_initialized 
        # del self.is_prepared_for_psd 
        # del self.is_template_whitened 
        
        
        
    def sum_periods(self, Pmin=Pmin, Pmax=Pmax, dP=None, duty_cycle=duty_cycle):
        """
        Sums the statistics with all the possible periods and initial phases. Accounts for 
        subbin template shifts, accounts for masks.
        Does not return 
    
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
        self.Pmin = Pmin
        self.Pmax = Pmax
        Per = Pmin
        periods = [Per]
        if dP is None:
            while Per<= Pmax:
                dP = duty_cycle * Per / self.T_mission
                Per += dP
                periods.append(Per)
        else:
            while Per<= Pmax:
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
                        (len(statistics_by_trends),int(np.ceil(Pmax/(self.dt_days/self.n_shifts))) ))))
        #pad by zeros to avoid extra if statements in the summation:       
        trends_sq_by_vars = self.stat_vars_interleaved * self.trend_orig_flux_interleaved**2
        trends_sq_by_vars = np.hstack((trends_sq_by_vars, np.zeros(
                        (len(trends_sq_by_vars),int(np.ceil(Pmax/(self.dt_days/self.n_shifts))) ))))
        
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
        
    
    def veto_mes(self, mes_threshold=threshold_mes_veto, veto_threshold=veto_threshold_for_mes):
        self.mes_veto_masks_interleaved = np.ones(np.shape(self.statistics_periodic_interleaved)).astype(bool)
        indices = np.array(np.where(self.statistics_periodic_interleaved<-np.abs(mes_threshold))).T
        for pi, ti, phi in indices:
            expected_transit_times = [self.tmin + self.phases_interleaved[phi] + i * self.periods[pi] 
                       for i in range(int(np.ceil((self.tmax - self.tmin - self.phases_interleaved[phi])/self.periods[pi]) ))]

            summed_transit_points = [np.argmin(np.abs(self.time_interleaved - expected_transit_times[i])) for i in range(len(expected_transit_times))  ]
            summed_stats = np.array(self.statistics_interleaved[ti][summed_transit_points]) #raw
            summed_trends = np.array(self.trend_orig_flux_interleaved[summed_transit_points])
            summed_vars = np.array(self.stat_vars_interleaved[ti][summed_transit_points])
            amplitude_vars = []
            amplitudes_normalized = []
            for si in range(len(summed_vars)):
                if not np.isclose(summed_stats[si], 0):
                    amplitude_vars.append(1/summed_vars[si])
                    amplitudes_normalized.append(summed_stats[si]/summed_vars[si]/summed_trends[si]*self.mean_total)
                else:
                    amplitude_vars.append(np.inf)
                    amplitudes_normalized.append(0.)
            veto_score = unequal_amplitude_veto(values=-np.array(amplitudes_normalized), variances=amplitude_vars, A_all=None, indmax=None, A_1=None)
            if veto_score > veto_threshold:
                self.statistics_periodic_interleaved[pi][ti][phi] = 0.
                self.mes_veto_masks_interleaved[pi][ti][phi] = False
                
        self.mes_veto_masks_not_interleaved = np.ones(np.shape(self.statistics_periodic_not_interleaved)).astype(bool)
        indices = np.array(np.where(self.statistics_periodic_not_interleaved<-np.abs(mes_threshold))).T
        for pi, ti, phi in indices:
            expected_transit_times = [self.tmin + self.phases_not_interleaved[phi] + i * self.periods[pi] 
                       for i in range(int(np.ceil((self.tmax - self.tmin - self.phases_not_interleaved[phi])/self.periods[pi]) ))]

            summed_transit_points = [np.argmin(np.abs(self.time_not_interleaved - expected_transit_times[i])) for i in range(len(expected_transit_times))  ]
            summed_stats = np.array(self.statistics_not_interleaved[ti][summed_transit_points]) #raw
            summed_trends = np.array(self.trend_orig_flux_not_interleaved[summed_transit_points])
            summed_vars = np.array(self.stat_vars_not_interleaved[ti][summed_transit_points])
            amplitude_vars = []
            amplitudes_normalized = []
            for si in range(len(summed_vars)):
                if (not np.isclose(summed_stats[si], 0) and not np.isinf(summed_vars[si])):
                    amplitude_vars.append(1/summed_vars[si])
                    amplitudes_normalized.append(summed_stats[si]/summed_vars[si]/summed_trends[si]*self.mean_total)
                else:
                    amplitude_vars.append(np.inf)
                    amplitudes_normalized.append(0.)
            if len(np.array(amplitudes_normalized))<3:
                veto_score = np.inf
            else:
                veto_score = unequal_amplitude_veto(values=-np.array(amplitudes_normalized), variances=amplitude_vars, A_all=None, indmax=None, A_1=None)
            if veto_score > veto_threshold:
                self.statistics_periodic_not_interleaved[pi][ti][phi] = 0.
                self.mes_veto_masks_not_interleaved[pi][ti][phi] = False
                            
        # for pi in range(len(self.statistics_periodic_interleaved)):
        #     for ti in range(len(self.statistics_periodic_interleaved[pi])):
        #         for phi in range(len(self.statistics_periodic_interleaved[pi][ti])):
        #             if self.statistics_periodic_interleaved[pi][ti][phi] < -np.abs(mes_threshold):
        #                 expected_transit_times = [self.tmin + self.phases_interleaved[phi] + i * self.periods[pi] 
        #                            for i in range(int(np.ceil((self.tmax - self.tmin - self.phases_interleaved[phi])/self.periods[pi]) ))]
    
        #                 summed_transit_points = [np.argmin(np.abs(self.time_interleaved - expected_transit_times[i])) for i in range(len(expected_transit_times))  ]
        #                 summed_stats = np.array(self.statistics_interleaved[ti][summed_transit_points]) #raw
        #                 summed_trends = np.array(self.trend_orig_flux_interleaved[summed_transit_points])
        #                 summed_vars = np.array(self.stat_vars_interleaved[ti][summed_transit_points])
        #                 amplitude_vars = []
        #                 amplitudes_normalized = []
        #                 for si in range(len(summed_vars)):
        #                     if not np.isclose(summed_stats[si], 0):
        #                         amplitude_vars.append(1/summed_vars[si])
        #                         amplitudes_normalized.append(summed_stats[si]/summed_vars[si]/summed_trends[si]*self.mean_total)
        #                     else:
        #                         amplitude_vars.append(np.inf)
        #                         amplitudes_normalized.append(0.)
        #                 veto_score = unequal_amplitude_veto(values=-np.array(amplitudes_normalized), variances=amplitude_vars, A_all=None, indmax=None, A_1=None)
        #                 if veto_score > veto_threshold:
        #                     self.statistics_periodic_interleaved[pi][ti][phi] = 0.
        #                     self.mes_veto_masks_interleaved[pi][ti][phi] = False
    
        
                            
    def delete_mes(self):
        del self.periods
        del self.phases_interleaved
        del self.phases_not_interleaved
        del self.statistics_periodic_not_interleaved
        del self.statistics_periodic_interleaved
        #del self.mes_veto_masks_interleaved
        #del self.mes_veto_masks_not_interleaved
     
    
    def determine_mins(self, ngap_moving_var=ngap_moving_var):
        """
        Finds the best value of statistics and saves the corresponding parameters.

        Parameters
        ----------
        ngap_moving_var : TYPE, optional
            DESCRIPTION. The default is ngap_moving_var.

        Returns
        -------
        

        """
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
        self.summed_transit_indices = copy.deepcopy(summed_transit_points)
        self.summed_transit_statistics_raw = summed_stats
        self.summed_transit_statistics_variances = copy.deepcopy(summed_vars)
        self.summed_transit_statistics_trends = copy.deepcopy(summed_trends)
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
            
    
    def save_best_candidates(self, n_per_temp=n_per_temp_saving, ngap_moving_var=ngap_moving_var, path=None, save_to_file=True):
        """
        Determines `n_per_temp` best scores for each template. For each period,
        finds one best phase corresponding to min (negative) statistic. Then, selects
        `n_per_temp` best periods for every template. For them, saves statistic values,
        periods, phases, summed values (see the description of returned results).

        Parameters
        ----------
        n_per_temp : int
            How many best scores per template to save. The default is 10.
        ngap_moving_var : TYPE, optional
            DESCRIPTION. The default is ngap_moving_var.

        Modifies
        -------
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

        """
        if path is None:
            path=self.output_location
        period_indices_selected = []
        periods_selected = []    
        phase_indices_selected = []
        phases_selected = []
        statistic_values_selected = []
        summed_transit_times_selected = []
        summed_stats_raw_selected = []
        summed_trends_selected = []
        summed_vars_selected = []
        for ti in range(len(self.statistics_periodic_interleaved[0])): #over templates
            phase_ind_min_per = []
            stats_min_per = []
            for pi in range(len(self.statistics_periodic_interleaved)): #over periods
                ph_i_min = np.nanargmin(self.statistics_periodic_interleaved[pi][ti])
                phase_ind_min_per.append(ph_i_min)
                stats_min_per.append(self.statistics_periodic_interleaved[pi][ti][ph_i_min])
            stat_best_per_inds = np.argsort(stats_min_per)[:n_per_temp]
            per_sel = self.periods[stat_best_per_inds]
            phase_ind_sel = np.array(phase_ind_min_per)[stat_best_per_inds]
            phases_sel = np.array(self.phases_interleaved)[phase_ind_sel]
            
            summed_transit_times_per = []
            summed_stats_raw_per = []
            summed_trends_per = []
            summed_vars_per = []
            for j in range(len(stat_best_per_inds)):
                pi = stat_best_per_inds[j]
                period = per_sel[j]
                phase = phases_sel[j]
                expected_transit_times = [self.tmin + phase + i * period 
                                               for i in range(int(np.ceil((self.tmax - self.tmin - phase)/period) ))]            
                summed_transit_points = [np.nanargmin(np.abs(self.time_interleaved - expected_transit_times[i])) for i in range(len(expected_transit_times))  ]
                summed_transit_times_per.append(self.time_interleaved[summed_transit_points])
                summed_stats_raw_per.append(self.statistics_interleaved[ti][summed_transit_points])
                summed_trends_per.append(self.trend_orig_flux_interleaved[summed_transit_points])
                summed_vars_per.append(self.stat_vars_interleaved[ti][summed_transit_points])
                #denom = np.sum(summed_vars * summed_trends**2)**0.5
                #summed_transit_statistics = (summed_stats * summed_trends)/denom
                # stds_around_min_per.append(outlier_free_std(np.concatenate((
                    # self.statistics_periodic_interleaved[pi][ti][:min(ph_i_sel - ngap_moving_var//2 * self.n_shifts, 0)],
                    # self.statistics_periodic_interleaved[pi][ti][min(ph_i_sel + ngap_moving_var//2 * self.n_shifts, len(self.statistics_periodic_interleaved[pi][ti])):]))))
            period_indices_selected.append(stat_best_per_inds)
            periods_selected.append(per_sel)
            phase_indices_selected.append(phase_ind_sel)    
            statistic_values_selected.append(np.array(stats_min_per)[stat_best_per_inds])
            phases_selected.append(phases_sel)
            summed_transit_times_selected.append(summed_transit_times_per)
            summed_stats_raw_selected.append(summed_stats_raw_per)
            summed_trends_selected.append(summed_trends_per)
            summed_vars_selected.append(summed_vars_per)
        for ti in range(len(self.statistics_periodic_not_interleaved[0])): #over templates
            phase_ind_min_per = []
            stats_min_per = []
            for pi in range(len(self.statistics_periodic_not_interleaved)): #over periods
                ph_i_min = np.nanargmin(self.statistics_periodic_not_interleaved[pi][ti])
                phase_ind_min_per.append(ph_i_min)
                stats_min_per.append(self.statistics_periodic_not_interleaved[pi][ti][ph_i_min])
            stat_best_per_inds = np.argsort(stats_min_per)[:n_per_temp]
            per_sel = self.periods[stat_best_per_inds]
            phase_ind_sel = np.array(phase_ind_min_per)[stat_best_per_inds]
            phases_sel = np.array(self.phases_not_interleaved)[phase_ind_sel]
            
            summed_transit_times_per = []
            summed_stats_raw_per = []
            summed_trends_per = []
            summed_vars_per = []
            for j in range(len(stat_best_per_inds)):
                pi = stat_best_per_inds[j]
                period = per_sel[j]
                phase = phases_sel[j]
                expected_transit_times = [self.tmin + phase + i * period 
                                               for i in range(int(np.ceil((self.tmax - self.tmin - phase)/period) ))]            
                summed_transit_points = [np.nanargmin(np.abs(self.time_not_interleaved - expected_transit_times[i])) for i in range(len(expected_transit_times))  ]
                summed_transit_times_per.append(self.time_not_interleaved[summed_transit_points])
                summed_stats_raw_per.append(self.statistics_not_interleaved[ti][summed_transit_points])
                summed_trends_per.append(self.trend_orig_flux_not_interleaved[summed_transit_points])
                summed_vars_per.append(self.stat_vars_not_interleaved[ti][summed_transit_points])
            period_indices_selected.append(stat_best_per_inds)
            periods_selected.append(per_sel)
            phase_indices_selected.append(phase_ind_sel)    
            statistic_values_selected.append(np.array(stats_min_per)[stat_best_per_inds])
            phases_selected.append(phases_sel)
            summed_transit_times_selected.append(summed_transit_times_per)
            summed_stats_raw_selected.append(summed_stats_raw_per)
            summed_trends_selected.append(summed_trends_per)
            summed_vars_selected.append(summed_vars_per)
    
        times_quarters = [(np.nanmin(self.times[q]), np.nanmax(self.times[q])) for q in self.quarters] #attention: there should be no zero-padding
        self.best_periodic_scores_data = []
        self.best_periodic_scores_data.append(period_indices_selected)
        self.best_periodic_scores_data.append(periods_selected)
        self.best_periodic_scores_data.append(phase_indices_selected)
        self.best_periodic_scores_data.append(phases_selected)
        self.best_periodic_scores_data.append(statistic_values_selected)
        self.best_periodic_scores_data.append(summed_transit_times_selected)
        self.best_periodic_scores_data.append(summed_stats_raw_selected)
        self.best_periodic_scores_data.append(summed_trends_selected)
        self.best_periodic_scores_data.append(summed_vars_selected)
        self.best_periodic_scores_data.append([self.time_slide_shifts, times_quarters])
    #     self.origrnal_transit_indices = [[] for q in self.quarters]
    #     for t in self.expected_transit_times:
    #         qmin = 0
    #         imin = 0
    #         minval = np.abs(t - self.times[0][0])
    #         for q in self.quarters:
    #             if (((t - self.times[q][0])>=0) and (np.abs(t - self.times[q][0])<minval)):
    #                 qmin = q
    #                 minval = np.abs(t - self.times[q][0])
    #         minval = t - self.times[qmin][0]
    #         for i in range(len(self.times[qmin])):
    #             if np.abs(t - self.times[qmin][i])<minval:
    #                 imin = i
    #                 minval = np.abs(t - self.times[qmin][i])
    #         self.origrnal_transit_indices[qmin].append(imin)
        if save_to_file:
            np.save(os.path.join(path, str(self.target_id)+'_best_periodic_scores_P='+str(self.Pmin)+'_'+str(self.Pmax)), np.array(self.best_periodic_scores_data, dtype=object), allow_pickle=True)
    
    
    def save_trigger_info(self, path=None):
        if path is None:
            path=self.output_location
        np.save(os.path.join(path, str(self.target_id)+'_trigger_info'), np.array(self.triggers, dtype=object), allow_pickle=True)
    
    
    def save_all_results(self, path=None):
        if path is None:
            path=self.output_location
        self.save_conditioned_data(path)
        self.save_trigger_info(path)
        self.save_best_candidates(n_per_temp=10, path=path)
    
    
    def return_results(self, return_outside=True, path=None, filename=None):
        """
        For only one best periodic score, but detailled. For many scores, see `save_best_candidates`.
        As well, see `save_all_results` for bigger saving.
        
        Returns
        -------
        results.

        """
        if path is None:
            path=self.output_location
        
        temp_to_check = np.arange(self.indices_of_templates[self.tempind_measured][0], 
                                  self.indices_of_templates[self.tempind_measured][-1])
        veto_info = [[] for i in range(len(self.summed_transit_statistics))]
        ind_veto_info = 0
        for q in self.quarters:
            if len(self.origrnal_transit_indices[q])>0:
                for ind in self.origrnal_transit_indices[q]:
                    for temp in temp_to_check:
                        if ind in self.veto_bad_trigger_inds[temp][q][0]:
                            position = np.where(np.array(self.veto_bad_trigger_inds[temp][q][0])==ind)[0][0]
                            veto_info[ind_veto_info].append(self.veto_bad_trigger_inds[temp][q][1][position])
                    ind_veto_info += 1
                    
        if self.is_min_interleaved:
            phas_result, stat_per_result = self.phases_interleaved, self.statistics_periodic_interleaved[self.pind_measured][self.tempind_min_interleaved]
        else:
            phas_result, stat_per_result = self.phases_not_interleaved, self.statistics_periodic_not_interleaved[self.pind_measured][self.tempind_min_not_interleaved]
        
        self.results = (self.target_id,
                        self.snr_measured, 
                        self.period_measured,
                        self.ph_measured,
                        self.tempind_measured,
                        self.std_around_min,
                        self.summed_transit_statistics,
                        self.summed_transit_statistics_raw ,
                        self.summed_transit_statistics_variances,
                        self.summed_transit_statistics_trends,
                        veto_info,
                        phas_result,
                        stat_per_result)
        
        if filename is None:
            filename0 = 'klc'+str(self.target_id)+'_results'
            filename=filename0
            num = 0
            while (filename in os.listdir(path)) or (filename+'.txt' in os.listdir(path)):
                num += 1
                filename = filename0 + '_' +str(num)
        if filename[-4:]!='.txt':
            filename = filename + '.txt'
        
        with open(os.path.join(path, filename), 'w') as output_file:
            for r in self.results:
                if type(r)==list or type(r)==np.ndarray:
                    print(*r, file=output_file, end='\n#\n')
                else:
                    print(r, file=output_file, end='\n#\n')
        
        if return_outside:        
            return self.results
    
    
    def print_veto_profile(self, quarters, indices=None, times=None, temp_inds=12, path=None, filename=None):
        """
        quarters : int or list of int or str 'all' or None
            None is when time is specified and quarter has to be found    
        indices : int or list of int or str 'all'
        
        time : float or list of float or str 'all'
        
        temp_inds : int or list of int or str 'all'
        
        file : str or None
        """
        if path is None:
            path=self.output_location
        if quarters == 'all':
            quarters = self.quarterss
        elif isinstance(quarters, int):
            quarters = [quarters]
        else:
            assert hasattr(quarters, "__len__"), 'Invalid quarters value specified!'
            if hasattr(indices, "__len__") and len(indices)>1:
                assert (len(quarters)==len(indices)), 'Undefined way to match indices to quarters!'
            elif hasattr(times, "__len__") and len(times)>1:
                assert (len(quarters)==len(times)), 'Undefined way to match times to quarters!'
            
        if isinstance(temp_inds, int):
            temp_inds = [temp_inds]
        
        if isinstance(indices, int):
            indices = [indices]
            
        if isinstance(times, float):
            times = [times]
            
        if indices is None:
            indices = []
            if quarters is not None and len(quarters)==1:
                for tim in times:
                    ind1 = np.argmin(np.abs(self.times[quarters[0]]-tim))
                    if np.abs(self.times[quarters[0]][ind1]-tim) > 2 * self.dt_days:
                        warnings.warn('indices not in this quarters!')
                    else:
                        indices.append(ind1)
            else:
                quarters = []
                indices = []
                for tim in times:
                    timedifmins = []
                    indmins = []
                    for q in self.quarters:
                        ind1 = np.argmin(np.abs(self.times[q]-tim))
                        indmins.append(ind1)
                        timedifmins.append(np.abs(self.times[q][ind1]-tim))
                    qindmin = np.argmin(timedifmins)
                    quarters.append(self.quarters[qindmin])
                    indices.append(indmins[qindmin])
        
        if len(quarters)==1:
            quarters = np.array([quarters[0]]*len(indices))
        
        if filename is None:
            filename0 = 'veto_klc'+str(self.target_id)
            filename=filename0
            num = 0
            while (filename in os.listdir(path)) or (filename+'.txt' in os.listdir(path)):
                num += 1
                filename = filename0 + '_' +str(num)
        if filename[-4:]!='.txt':
            filename = filename + '.txt'
        
        with open(os.path.join(path, filename), 'w') as file:
            for i in range(len(indices)):
                q = quarters[i]
                index = indices[i]
                if temp_inds=='all':
                    temp_inds = list(self.triggers[q][index].keys())
                for j in range(len(temp_inds)):
                    temp_ind = temp_inds[j] 
                    print('quarter:', q, file=file)
                    print('index:', index, file=file)
                    print('template index:', temp_ind, file=file)
                    if index not in self.triggers[q].keys():
                        print('not a trigger')
                        return None
                    if temp_ind not in self.triggers[q][index].keys():
                        print('not a trigger')
                        return None
                    print('whether passed vetting:', self.triggers[q][index][temp_ind][0], file=file)
                    if not self.triggers[q][index][temp_ind][0]:
                        print('reasons why:', self.triggers[q][index][temp_ind][1], file=file)
                    if len(self.triggers[q][index][temp_ind][2])==8:
                        print('chi squared:', self.triggers[q][index][temp_ind][2][0], file=file)
                        print('threshold chi squared:', self.triggers[q][index][temp_ind][2][1], file=file)
                        print('max dev std:', self.triggers[q][index][temp_ind][2][2], file=file)
                        print('threshold max dev std:', self.triggers[q][index][temp_ind][2][3], file=file)
                        print('max dev ampl:', self.triggers[q][index][temp_ind][2][4], file=file)
                        print('threshold max dev ampl:', self.triggers[q][index][temp_ind][2][5], file=file)
                        print('inner product abs:', np.abs(self.triggers[q][index][temp_ind][2][6]), file=file)
                        print('threshold inner product abs:', self.triggers[q][index][temp_ind][2][7], file=file)
                    if len(self.triggers[q][index][temp_ind][3])!=0:
                        print('chi2 parts snr local:', *self.triggers[q][index][temp_ind][3][0][:-1], file=file)
                        print('threshold chi2 parts snr local:', self.triggers[q][index][temp_ind][3][1], file=file)
                        print('chi2 parts snr total:', self.triggers[q][index][temp_ind][3][0][-1], file=file)
                        print('threshold chi2 parts snr total:', self.triggers[q][index][temp_ind][3][2], file=file)
                    if len(self.triggers[q][index][temp_ind][4])!=0:
                        print('chi2 parts len local:', *self.triggers[q][index][temp_ind][4][0][:-1], file=file)
                        print('threshold chi2 parts len local:', self.triggers[q][index][temp_ind][3][1], file=file)
                        print('chi2 parts len total:', self.triggers[q][index][temp_ind][4][0][-1], file=file)
                        print('threshold chi2 parts len total:', self.triggers[q][index][temp_ind][3][2], file=file)
                    if len(self.triggers[q][index][temp_ind][5])!=0:
                        print('snr2 of this trigger:', self.triggers[q][index][temp_ind][5][0], file=file)
                        print('snr2 of the best template:', self.triggers[q][index][temp_ind][5][1], file=file)
                    print('', file=file)
            
    
    #plotting methods
    def plot_results(self, 
                     plot_all=None,
                     original_data=None,
                     psds=None,
                     whitened_fluxes_td=None,
                     whitened_fluxes_fd=None,
                     single_event_statistic_snr_without_mask=None,
                     single_event_statistic_before_summation=None,
                     single_event_statistic_snr=None,
                     fluxes_with_veto=None,
                     histogram_single_event=None,
                     summed_statistic_minimal_of_phase=None,
                     summed_statistic_of_phase=None,
                     histogram_summed_statistic=None,
                     minimal_statistic_periods=None,
                     before_summation=False,
                     temp_num=None, period=None, per_ind=None, nbins_hist=25, 
                     figsize=None, edgelist_hist=None, edgemin_hist=None, edgemax_hist=None
                     ):
        if figsize is None:
            width=12
            height=5
        else:
            width=figsize[0]
            height=figsize[1]
        
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
            if single_event_statistic_before_summation is None:
                single_event_statistic_before_summation = True
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
            if single_event_statistic_before_summation is None:
                single_event_statistic_before_summation = False
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
            if single_event_statistic_before_summation is None:
                single_event_statistic_before_summation = True
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
        
        if before_summation:
            if plot_all==True:
                original_data = True
                psds = True
                whitened_fluxes_td = True
                whitened_fluxes_fd = True
                single_event_statistic_snr_without_mask = False
                single_event_statistic_before_summation = False
                single_event_statistic_snr = True
                fluxes_with_veto = True
                histogram_single_event = True
                summed_statistic_minimal_of_phase = False
                summed_statistic_of_phase = False
                histogram_summed_statistic = False
                minimal_statistic_periods = False
            
            if temp_num is None:
                temp_num = 11
        else:
            if period is None:
                if per_ind is not None:
                    period = self.periods[per_ind]
                else:
                    per_ind = self.pind_measured
                    period = self.periods[self.pind_measured]
            else:
                per_ind = np.argmin(np.abs(self.periods-period))
                period = self.periods[per_ind]
                
            if temp_num is None:
                temp_num = self.tempind_measured
            
        
        if original_data:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.orig_times[q], self.orig_fluxes[q])
            plt.title('Original unprocessed PDS lightcurves')
            plt.xlabel('time, days')
            plt.ylabel('flux')
            plt.tight_layout()
            plt.show()
        
        if psds:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.psdfreqs[q], np.log(np.abs(self.psds[q])))
            plt.title('Measured PSDS of all the quarters')
            plt.xlabel('frequency, Hz')
            plt.ylabel('log[PSD]')
            plt.tight_layout()
            plt.show()
        
        if whitened_fluxes_td:
            plt.figure(figsize=(width, height))
            labeled = False
            for q in self.quarters:
                plt.plot(self.times[q], self.whitened_fluxes_td[q])
                if not labeled:
                    plt.plot(self.times[q][~self.big_hole_masks[q]], self.whitened_fluxes_td[q][~self.big_hole_masks[q]], 
                             'o', color='k', label='big masked holes')
                    labeled = True
                else:
                    plt.plot(self.times[q][~self.big_hole_masks[q]], self.whitened_fluxes_td[q][~self.big_hole_masks[q]], 
                             'o', color='k')
            plt.title('Whitened fluxes in time domain')
            plt.xlabel('time, days')
            plt.ylabel('whitened flux')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        if whitened_fluxes_fd:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.freqs[q], np.real(self.whitened_fluxes_fd[q]))
            plt.title('Whitened fluxes in Fourier domain')
            plt.xlabel('frequency, Hz')
            plt.ylabel('whitened flux')
            plt.tight_layout()
            plt.show()
        
        if single_event_statistic_snr_without_mask:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                plt.plot(self.times[q], self.statistics_snr[self.indices_of_templates[temp_num][0]][q])
            plt.title('Single-event statistic in units of SNR for template #'+str(temp_num))
            plt.xlabel('time, days')
            plt.ylabel('statistic [SNR]')
            plt.tight_layout()
            plt.show()
        
        if single_event_statistic_snr:
            plt.figure(figsize=(width, height))
            for q in self.quarters:
                if q%2==0:
                    clr = (0.2, 0.5,0.8)
                else:
                    clr = (0.2, 0.7,0.4)
                clr
                plt.plot(self.times[q], self.statistics_snr[self.indices_of_templates[temp_num][0]][q], color=clr)
            #plot what was masked in statistic masks
            label_made=False
            for q in self.quarters:
                if not label_made:
                    if (~self.stat_masks[self.indices_of_templates[temp_num][0]][q]).any():
                        label_made=True
                        plt.plot(self.times[q][~self.stat_masks[self.indices_of_templates[temp_num][0]][q]], 
                                 self.statistics_snr[self.indices_of_templates[temp_num][0]][q][~self.stat_masks[self.indices_of_templates[temp_num][0]][q]],
                                 'o', color=(0.8,0.5,0), label='masked statistic', markersize=5)
                else:
                    plt.plot(self.times[q][~self.stat_masks[self.indices_of_templates[temp_num][0]][q]], 
                             self.statistics_snr[self.indices_of_templates[temp_num][0]][q][~self.stat_masks[self.indices_of_templates[temp_num][0]][q]],
                             'o', color=(0.8,0.5,0), markersize=5)
            #plot what from this was masked in conditioning
            label_made=False
            for q in self.quarters:
                if not label_made:
                    if (~self.stat_masks[self.indices_of_templates[temp_num][0]][q]).any():
                        label_made=True
                        plt.plot(self.times[q][~self.big_hole_masks[q]], self.statistics_snr[self.indices_of_templates[temp_num][0]][q][~self.big_hole_masks[q]],
                                 'o', color='k', label='masked in conditioning', markersize=3)
                else:
                    plt.plot(self.times[q][~self.big_hole_masks[q]], self.statistics_snr[self.indices_of_templates[temp_num][0]][q][~self.big_hole_masks[q]],
                             'o', color='k', markersize=3)
            if not before_summation:
                times_with_transits = np.array([])
                stat_with_transits = np.array([])
                for q in self.quarters:
                    times_with_transits = np.concatenate((times_with_transits, self.times[q][self.origrnal_transit_indices[q]]))
                    stat_with_transits = np.concatenate((stat_with_transits, self.statistics_snr[self.indices_of_templates[temp_num][0]][q][self.origrnal_transit_indices[q]]))
                plt.plot(times_with_transits, stat_with_transits, 'o', color=(0.8,0,0), label='transits', markersize=4)
            
            plt.title('Single-event statistic in units of SNR for template #'+str(temp_num))
            plt.xlabel('time, days')
            plt.ylabel('statistic [SNR]')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        if single_event_statistic_before_summation:
            plt.figure(figsize=(width, height))
            if self.is_min_interleaved:
                plt.plot(self.time_interleaved, self.statistics_interleaved[self.tempind_min_interleaved], linewidth=2)  
                plt.plot(self.time_interleaved[self.summed_transit_indices], self.statistics_interleaved[self.tempind_min_interleaved][self.summed_transit_indices],'o',
                         label='summed values')
            else:
                plt.plot(self.time_not_interleaved, self.statistics_not_interleaved[self.tempind_min_not_interleaved], linewidth=2)  
                plt.plot(self.time_not_interleaved[self.summed_transit_indices], self.statistics_not_interleaved[self.tempind_min_not_interleaved][self.summed_transit_indices],'o',
                         label='summed values')
            plt.title('Single-event statistic in initial units used for summation. template #'+str(temp_num))
            plt.xlabel('time, days')
            plt.ylabel('statistic raw corrected for moving variance')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            
            
        if fluxes_with_veto:
            plt.figure(figsize=(width, height))
            labeled_chi_squared_veto = False
            labeled_maximal_deviation_std_veto = False
            labeled_maximal_deviation_amplitude_veto = False
            labeled_extreme_position_veto = False
            labeled_parts_len_veto = False
            labeled_parts_snr_veto = False
            labeled_inner_product_veto = False
            labeled_template_width_veto = False
            labeled=False
            colors=[(0,0,0), (0.3,0.3,0.3)]
            for q in self.quarters:
                plt.plot(self.times[q], self.fluxes[q], color=colors[q%2], linewidth=2)
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
                parts_len_veto = []
                parts_snr_veto = []
                inner_product_veto = []
                template_width_veto = []
                tin = self.indices_of_templates[temp_num][0]
                for ind in range(len(self.veto_bad_trigger_inds[tin][q][0])):
                    if self.veto_bad_trigger_inds[tin][q] != [[], [], []]:
                        if 'chi_squared' in self.veto_bad_trigger_inds[tin][q][1][ind]:
                            chi_squared_veto.append(self.veto_bad_trigger_inds[tin][q][0][ind])
                        if 'maximal_deviation_std' in self.veto_bad_trigger_inds[tin][q][1][ind]:    
                            maximal_deviation_std_veto.append(self.veto_bad_trigger_inds[tin][q][0][ind])
                        if 'maximal_deviation_amplitude' in self.veto_bad_trigger_inds[tin][q][1][ind]:        
                            maximal_deviation_amplitude_veto.append(self.veto_bad_trigger_inds[tin][q][0][ind])
                        if 'extreme_position' in self.veto_bad_trigger_inds[tin][q][1][ind]:           
                            extreme_position_veto.append(self.veto_bad_trigger_inds[tin][q][0][ind])
                        if 'inconsistent_parts_len' in self.veto_bad_trigger_inds[tin][q][1][ind]:           
                            parts_len_veto.append(self.veto_bad_trigger_inds[tin][q][0][ind])
                        if 'inconsistent_parts_snr' in self.veto_bad_trigger_inds[tin][q][1][ind]:           
                            parts_snr_veto.append(self.veto_bad_trigger_inds[tin][q][0][ind])
                        if 'inner_product' in self.veto_bad_trigger_inds[tin][q][1][ind]:           
                            inner_product_veto.append(self.veto_bad_trigger_inds[tin][q][0][ind])
                        if 'template_width' in self.veto_bad_trigger_inds[tin][q][1][ind]:           
                            template_width_veto.append(self.veto_bad_trigger_inds[tin][q][0][ind])
                if len(chi_squared_veto)>0:
                    if not labeled_chi_squared_veto:
                        plt.plot(self.times[q][chi_squared_veto], self.fluxes[q][chi_squared_veto], 'o', color='r', markersize=5, label='chi squared veto')
                        labeled_chi_squared_veto = True
                    else:
                        plt.plot(self.times[q][chi_squared_veto], self.fluxes[q][chi_squared_veto], 'o', color='r', markersize=5)
                if len(maximal_deviation_std_veto)>0:
                    if not labeled_maximal_deviation_std_veto:
                        plt.plot(self.times[q][maximal_deviation_std_veto], self.fluxes[q][maximal_deviation_std_veto], 'o', color=(1,0.5,0), markersize=5, label='maximal deviation std veto')
                        labeled_maximal_deviation_std_veto = True
                    else:
                        plt.plot(self.times[q][maximal_deviation_std_veto], self.fluxes[q][maximal_deviation_std_veto], 'o', color=(1,0.5,0), markersize=5)
                if len(maximal_deviation_amplitude_veto)>0:
                    if not labeled_maximal_deviation_amplitude_veto :
                        plt.plot(self.times[q][maximal_deviation_amplitude_veto], self.fluxes[q][maximal_deviation_amplitude_veto], 'o', color=(0.6,0.4,0), markersize=5, label='maximal deviation amplitude veto')
                        labeled_maximal_deviation_amplitude_veto = True
                    else:
                        plt.plot(self.times[q][maximal_deviation_amplitude_veto], self.fluxes[q][maximal_deviation_amplitude_veto], 'o', color=(0.6,0.4,0), markersize=5)
                if len(extreme_position_veto)>0:
                    if not labeled_extreme_position_veto:
                        plt.plot(self.times[q][extreme_position_veto], self.fluxes[q][extreme_position_veto], 'o', color=(0.4,0.3,1), markersize=5, label='extreme position veto')
                        labeled_extreme_position_veto = True
                    else:
                        plt.plot(self.times[q][extreme_position_veto], self.fluxes[q][extreme_position_veto], 'o', color=(0.4,0.3,1), markersize=5)
                if len(parts_len_veto)>0:
                    if not labeled_parts_len_veto:
                        plt.plot(self.times[q][parts_len_veto], self.fluxes[q][parts_len_veto], 'o', color=(0.7,0,1), markersize=5, label='inconsistent parts len')
                        labeled_parts_len_veto = True
                    else:
                        plt.plot(self.times[q][parts_len_veto], self.fluxes[q][parts_len_veto], 'o', color=(0.7,0,1), markersize=5)
                if len(parts_snr_veto)>0:
                    if not labeled_parts_snr_veto:
                        plt.plot(self.times[q][parts_snr_veto], self.fluxes[q][parts_snr_veto], 'o', color=(0.9,0,1), markersize=5, label='inconsistent parts snr')
                        labeled_parts_snr_veto = True
                    else:
                        plt.plot(self.times[q][parts_snr_veto], self.fluxes[q][parts_snr_veto], 'o', color=(0.9,0,1), markersize=5)
                
                if len(inner_product_veto)>0:
                    if not labeled_inner_product_veto:
                        plt.plot(self.times[q][inner_product_veto], self.fluxes[q][inner_product_veto], 'o', color=(0.6,0.6,0.6), markersize=3, label='inner product veto')
                        labeled_inner_product_veto = True
                    else:
                        plt.plot(self.times[q][inner_product_veto], self.fluxes[q][inner_product_veto], 'o', color=(0.6,0.6,0.6), markersize=3)
                if len(template_width_veto)>0:
                    if not labeled_template_width_veto:
                        plt.plot(self.times[q][template_width_veto], self.fluxes[q][template_width_veto], 'o', color=(0,0.9,0), markersize=3, label='template width veto')
                        labeled_template_width_veto = True
                    else:
                        plt.plot(self.times[q][template_width_veto], self.fluxes[q][template_width_veto], 'o', color=(0,0.9,0), markersize=3)
            if not before_summation:
                times_with_transits = np.array([])
                fluxes_with_transits = np.array([])
                for q in self.quarters:
                    times_with_transits = np.concatenate((times_with_transits, self.times[q][self.origrnal_transit_indices[q]]))
                    fluxes_with_transits = np.concatenate((fluxes_with_transits, self.fluxes[q][self.origrnal_transit_indices[q]]))
                plt.plot(times_with_transits, fluxes_with_transits, 'o', color=(0.3,1,0.3), markersize=3, label='transits')
            plt.title('Conditioned fluxes in time domain with vetos for template #'+str(temp_num))
            plt.xlabel('time, days')
            plt.ylabel('flux')
            plt.legend(fontsize=(plt.rcParams['font.size']-3) )
            plt.tight_layout()
            plt.show()
            
        if histogram_single_event:
            tin = self.indices_of_templates[temp_num][0]
            datstat = np.array([])
            for q in self.quarters:
                datstat = np.concatenate((datstat, self.statistics_snr[tin][q][self.stat_masks[tin][q]]))
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
            plt.tight_layout()
            plt.show()
            
        if summed_statistic_of_phase:
            plt.figure(figsize=(width, height))
            if temp_num in self.interleaved_numbers:
                plt.plot(self.phases_interleaved, self.statistics_periodic_interleaved[per_ind][temp_num])
            else:
                temp_num_ind = np.where(self.not_interleaved_numbers==temp_num)[0][0]
                plt.plot(self.phases_not_interleaved, self.statistics_periodic_not_interleaved[per_ind][temp_num_ind])
            plt.title('Summed statistic as function of phase for period='+str(period)+' days, template #'+str(temp_num))
            plt.xlabel('initial phase, days')
            plt.ylabel('statistic, [SNR]')
            plt.tight_layout()
            plt.show()
            
        if histogram_summed_statistic:
            if temp_num in self.interleaved_numbers:
                temp_num_ind = np.where(self.interleaved_numbers==temp_num)[0][0]
                datstat = self.statistics_periodic_interleaved[per_ind][temp_num_ind]
            else:
                temp_num_ind = np.where(self.not_interleaved_numbers==temp_num)[0][0]
                datstat = self.statistics_periodic_not_interleaved[per_ind][temp_num_ind]
            histogram(datstat[~np.isnan(datstat)], n=nbins_hist, edgelist=edgelist_hist, edgemin=edgemin_hist, edgemax=edgemax_hist, xscale='square', yscale='log',
                  density=False, normalize=True, ylogshift=True, logzeros='zero', 
                  return_results=False, plot=True, new_figure=True, ax=None, figsize=(width/3*2,height), show=True,
                  xlabel='statistic squared', ylabel='log counts', 
                  title='Histogram of summed statistic for all phases for period='+str(np.round(period, 2))+' days, template #'+str(temp_num), 
                  xlim=None, ylim=None,
                  color='b', linestyle='solid', marker='', linewidth=None, alpha=0.9, label=None, 
                  loc_legend=None, plot_type='bar') 
            
        if minimal_statistic_periods:
            mins = []
            for p in range(len(self.periods)):
                mins.append(min(np.nanmin(self.statistics_periodic_interleaved[p]), np.nanmin(self.statistics_periodic_not_interleaved[p])))
            plt.figure(figsize=(width, height))
            plt.plot(self.periods, mins)
            plt.title('Minimum of summed statistic as function of period')
            plt.xlabel('periods, days')
            plt.ylabel('statistic, [SNR]')
            plt.tight_layout()
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
    with open(filename, 'r') as f:
        lines = f.readlines()
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
                
                
def get_data_fun(target_id, location=default_data_location):
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
        filc =  fits.open(os.path.join(folder,f))
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
               remove_outliers=False, remove_outliers_light=False, detr_length_outliers=detr_length_outliers,
               nsig_outliers=nsig_outliers_stat, return_mask=False, nsamples_min=nsamples_min_moving_var,
               adjustable_slices=False):
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
    
    if mask is None:
        mask = ~np.isnan(signal)
    else:    
        mask = np.array(mask)
        assert len(signal)==len(mask), 'Data and mask have to be of the same length!'
    
        
    if remove_outliers:
        data1 = detrend_mov_ave(np.array(signal), length=detr_length_outliers, mask=mask, 
                                 method='median', endmethod='mean', ends='symmetric', endlen=None, 
                                 return_trend=False, keep_overall_level=False, remove_outliers=True, nsig_outliers=nsig_outliers)
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask
    elif remove_outliers_light:
        outlier_mask = np.abs(signal - np.nanmean(signal[mask])) < nsig_outliers * np.nanstd(signal[mask])
        mask *= outlier_mask
        outlier_mask = np.abs(signal - np.nanmean(signal[mask])) < nsig_outliers * np.nanstd(signal[mask])
        mask *= outlier_mask
    
    all_done = False
    if (nperseg>len(signal)/2) or (nsamples_min*4>len(np.where(mask)[0])):
        #TODO: write this normally, not draft for pipeline
        if adjustable_slices:
            if len(np.where(mask)[0])/2>nsamples_min*2:
                nperseg = len(signal)//2 + 1
                #warnings.warn('Not enough data to calculate moving variance.')
            else:
                mov_var = np.ones(len(signal))*unb_var(signal[mask])
            if ngap>=nperseg//2:
                ngap=nperseg//2
        else:
            mov_var = np.ones(len(signal))*unb_var(signal[mask])
            all_done=True
            
    if not all_done:
        signal[~mask] = 0.
        sum1 = np.nancumsum(signal)
        sum1 = np.concatenate(([0], sum1))
        sum2 = np.nancumsum(signal**2)
        sum2 = np.concatenate(([0], sum2))
        summ = np.nancumsum(mask)
        summ = np.concatenate(([0], summ))
        mov_var = []
        for i in np.arange(0,ngap//2):
            counts = summ[i+nperseg//2]-summ[i+ngap//2]
            if counts<nsamples_min:
                nperseg1 = nperseg + (nsamples_min-counts)*2
                counts = summ[i+nperseg1//2]-summ[i+ngap//2]
                mov_var.append(((sum2[i+nperseg1//2]-sum2[i+ngap//2])/counts-((sum1[i+nperseg1//2]-sum1[i+ngap//2])/counts)**2)
                           *counts/(counts-1))
            else:
                mov_var.append(((sum2[i+nperseg//2]-sum2[i+ngap//2])/counts-((sum1[i+nperseg//2]-sum1[i+ngap//2])/counts)**2)
                           *counts/(counts-1))
        for i in np.arange(ngap//2,nperseg//2):
            counts = summ[i-ngap//2]-summ[0]+summ[i+nperseg//2]-summ[i+ngap//2]
            if counts<nsamples_min:
                nperseg1 = nperseg + (nsamples_min-counts)*2
                counts = summ[i-ngap//2]-summ[0]+summ[i+nperseg1//2]-summ[i+ngap//2]
                mov_var.append(((sum2[i-ngap//2]-sum2[0]+sum2[i+nperseg1//2]-sum2[i+ngap//2])/counts
                           -((sum1[i-ngap//2]-sum1[0]+sum1[i+nperseg1//2]-sum1[i+ngap//2])/counts)**2)
                           *counts/(counts-1))
            else:
                mov_var.append(((sum2[i-ngap//2]-sum2[0]+sum2[i+nperseg//2]-sum2[i+ngap//2])/counts
                           -((sum1[i-ngap//2]-sum1[0]+sum1[i+nperseg//2]-sum1[i+ngap//2])/counts)**2)
                           *counts/(counts-1))
        for i in range(nperseg//2,len(signal)-nperseg//2):
            counts = summ[i+nperseg//2]-summ[i+ngap//2]+summ[i-ngap//2]-summ[i-nperseg//2]
            if counts<nsamples_min:
                nperseg1 = nperseg + (nsamples_min-counts)
                counts = summ[i+nperseg1//2]-summ[i+ngap//2]+summ[i-ngap//2]-summ[i-nperseg1//2]
                mov_var.append(((sum2[i+nperseg1//2]-sum2[i+ngap//2]+sum2[i-ngap//2]-sum2[i-nperseg1//2])/counts
                          -(((sum1[i+nperseg1//2]-sum1[i+ngap//2]+sum1[i-ngap//2]-sum1[i-nperseg1//2])/counts)**2))
                           *counts/(counts-1))
            else:
                mov_var.append(((sum2[i+nperseg//2]-sum2[i+ngap//2]+sum2[i-ngap//2]-sum2[i-nperseg//2])/counts
                          -(((sum1[i+nperseg//2]-sum1[i+ngap//2]+sum1[i-ngap//2]-sum1[i-nperseg//2])/counts)**2))
                           *counts/(counts-1))
        for i in np.arange(len(signal)-nperseg//2, len(signal)-ngap//2):
            counts = summ[i-ngap//2]-summ[i-nperseg//2]+summ[-1]-summ[i+ngap//2] 
            if counts<nsamples_min:
                nperseg1 = nperseg + (nsamples_min-counts)*2
                counts = summ[i-ngap//2]-summ[i-nperseg1//2]+summ[-1]-summ[i+ngap//2] 
                mov_var.append(((sum2[i-ngap//2]-sum2[i-nperseg1//2]+sum2[-1]-sum2[i+ngap//2])/counts
                         -((sum1[i-ngap//2]-sum1[i-nperseg1//2]+sum1[-1]-sum1[i+ngap//2])/counts)**2)
                           *counts/(counts-1))
            else:
                mov_var.append(((sum2[i-ngap//2]-sum2[i-nperseg//2]+sum2[-1]-sum2[i+ngap//2])/counts
                         -((sum1[i-ngap//2]-sum1[i-nperseg//2]+sum1[-1]-sum1[i+ngap//2])/counts)**2)
                           *counts/(counts-1))
        for i in np.arange(len(signal)-ngap//2, len(signal)):
            counts = summ[i-ngap//2]-summ[i-nperseg//2]
            if counts<nsamples_min:
                nperseg1 = nperseg + (nsamples_min-counts)*2
                counts = summ[i-ngap//2]-summ[i-nperseg1//2]
                mov_var.append(((sum2[i-ngap//2]-sum2[i-nperseg1//2])/counts
                           -((sum1[i-ngap//2]-sum1[i-nperseg1//2])/counts)**2)
                           *counts/(counts-1))
            else:
                mov_var.append(((sum2[i-ngap//2]-sum2[i-nperseg//2])/counts
                           -((sum1[i-ngap//2]-sum1[i-nperseg//2])/counts)**2)
                           *counts/(counts-1))
    if not return_mask:
        return np.abs(np.array(mov_var)) #because small fluctuations around zero can occur  
    else:
        return np.abs(np.array(mov_var)), mask
    
  
def gap_var(signal, mask=None, ngap=ngap_moving_var,
               remove_outliers=False, remove_outliers_light=False, detr_length_outliers=detr_length_outliers,
               nsig_outliers=nsig_outliers_stat, return_mask=False, nsamples_min=nsamples_min_moving_var,
               adjustable_slices=False, return_total_var=False):
    """
    Calculates the moving variance of the signal taking for every point all the 
    values that are outside `ngap` and obeying the mask. If needed, removes outliers. 
    At every point,  excludes around it symmetric window of length `ngap` and 
    calculates the unbiased variance of the remaining non-masked signal. It is needed to
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
    
    if mask is None:
        mask = ~np.isnan(signal)
    else:    
        mask = np.array(mask)
        assert len(signal)==len(mask), 'Data and mask have to be of the same length!'
    
        
    if remove_outliers:
        data1 = detrend_mov_ave(np.array(signal), length=detr_length_outliers, mask=mask, 
                                 method='median', endmethod='mean', ends='symmetric', endlen=None, 
                                 return_trend=False, keep_overall_level=False, remove_outliers=True, nsig_outliers=nsig_outliers)
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask
    elif remove_outliers_light:
        outlier_mask = np.abs(signal - np.nanmean(signal[mask])) < nsig_outliers * np.nanstd(signal[mask])
        mask *= outlier_mask
        outlier_mask = np.abs(signal - np.nanmean(signal[mask])) < nsig_outliers * np.nanstd(signal[mask])
        mask *= outlier_mask
    
    total_var = unb_var(signal[mask])
    
    # return np.ones(len(signal)) * total_var, total_var

    signal[~mask] = 0.
    sum1 = np.nancumsum(signal)
    sum1 = np.concatenate(([0], sum1))
    sum2 = np.nancumsum(signal**2)
    sum2 = np.concatenate(([0], sum2))
    summ = np.nancumsum(mask)
    summ = np.concatenate(([0], summ))
    mov_var = []
    
    for i in np.arange(0,ngap//2):
        counts = summ[-1]-summ[i+ngap//2]
        if counts<nsamples_min:
            mov_var.append(total_var)
        else:
            mov_var.append(((sum2[-1]-sum2[i+ngap//2])/counts-((sum1[-1]-sum1[i+ngap//2])/counts)**2)
                        *counts/(counts-1))

    for i in range(ngap//2,len(signal)-ngap//2,):
        counts = summ[-1]-summ[i+ngap//2]+summ[i-ngap//2]
        if counts<nsamples_min:
            mov_var.append(total_var)
        else:
            mov_var.append(((sum2[-1]-sum2[i+ngap//2]+sum2[i-ngap//2])/counts
                      -(((sum1[-1]-sum1[i+ngap//2]+sum1[i-ngap//2])/counts)**2))
                        *counts/(counts-1))
            
    for i in np.arange(len(signal)-ngap//2, len(signal)):
        counts = summ[i-ngap//2]
        if counts<nsamples_min:
            mov_var.append(total_var)
        else:
            mov_var.append(((sum2[i-ngap//2])/counts
                        -((sum1[i-ngap//2])/counts)**2)
                        *counts/(counts-1))
    if not return_total_var:
        if not return_mask:
            return np.abs(np.array(mov_var)) #because small fluctuations around zero can occur  
        else:
            return np.abs(np.array(mov_var)), mask
    else:
        if not return_mask:
            return np.abs(np.array(mov_var)), total_var #because small fluctuations around zero can occur  
        else:
            return np.abs(np.array(mov_var)), mask, total_var
      

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
        data1 = detrend_mov_ave(np.array(data), length=detr_len, mask=mask, method='median', endmethod='mean', ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False)
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


def tanhstep(x, center=0., hwhm=1., start_from_zero=False):
    if type(x)==list or type(x)==np.ndarray:
        res = (1. + np.tanh((np.array(x)-center)/hwhm * 1/2*np.log(3)))/2.
        if  start_from_zero:
            m = np.nanmax(res)
            res -= np.nanmin(res)
            res *= m/np.nanmax(res)
        return res
    else:
        return (1. + np.tanh((np.array(x)-center)/hwhm * 1/2*np.log(3)))/2.


def trend_fft(data, nperseg, endlen=None, mask=None, data_low=None, fit_order='cubic'):
    """
    Calculates low-frequency trend of the `data` by zeroing its high-frequency
    components (correspondinh to wavelength >`nperseg`) in the Fourier transform. 
    After that, treats the ends of the data (size of the end is `endlen`)  
    by fitting quadratic polynomials to them in order to avoid bad behaviour
    of the trend from Fourier components which tend to repeat oscillatory behaviour
    in the ends and do not follow the data.

    Parameters
    ----------
    data : numpy.array or list of float
        Data vector to be detrended
    nperseg : int
        Wavelength below which the frequencies should be included in the trend.
    endlen : int
        Length of the data in the ends to be fitted by quadratic polynomial
    mask : array of bool
        Masking good data points with `True`, bad data points with `False`.
    data_low : numpy.ndarray or None
        The low-frequency trend. If None, it is found from the zeroing Fourier components,
        and it is the usual scenario. In rare cases, the required procedure is only to 
        fix the ends with the fit. In this case, data_low can be specified. 
        The length should be the same as the length of `data`.
    fit_order : str : 'quadratic' or 'cubic'
    Returns
    -------
    data_low : numpy.array or list of float
        Smooth low-frequency trend of the `data`

    """
    assert fit_order=='quadratic' or fit_order=='cubic', 'Undefined `fit_order` specified!'
    
    data = np.array(data)
    time = np.arange(len(data))
    nanmask = ~np.isnan(data)
    if not (nanmask).all():
        data_i = interpolate_over_mask(data, mask=nanmask)
    else:
        data_i = data
    if mask is None:
        mask = nanmask
    if len(data)==1:
        warnings.warn('WHY YOU GIVE ME DATA OF LENGTH 1 TO FIT???')
    if (len(data)<=nperseg) or (len(data)//2<=endlen):
        #parabolic fit or the entire data
        if fit_order=='quadratic':
            msk1 = np.array(mask).astype(bool)
            dat1 = (data_i)[msk1]
            tim1 = (time)[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1)//2]
            x10 = (tim1[-1]+tim1[0])/2
            x20 = ((dat1[0]-cent)/(tim1[0]-x10)**2 + (dat1[-1]-cent)/(tim1[-1]-x10)**2)/2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2]*(tim1-x[1])**2 + x[0]))**2), 
             [x00, x10 , x20])
            data_low = (sol.x[2]*(np.arange(len(data))-sol.x[1])**2 + sol.x[0])
        else:
            msk1 = np.array(mask).astype(bool)
            dat1 = (data_i)[msk1]
            tim1 = (time)[msk1]
            tim10 = tim1[0]
            tim1 = tim1 - tim10
            x00 = (dat1[0] * tim1[-1] - dat1[-1] * tim1[0]) / (tim1[-1] - tim1[0])
            x10 = (dat1[-1] - dat1[0]) / (tim1[-1] - tim1[0])
            x20 = 0.
            x30 = 0.
            sol = minimize(lambda x: np.nansum((dat1 - (x[3]*tim1**3 + x[2]*tim1**2 + x[1]*tim1 + x[0]) )**2), [x00, x10 , x20, x30])
            data_low = (sol.x[3]*(np.arange(len(data)))**3
                                 + sol.x[2]*(np.arange(len(data)))**2 
                                 + sol.x[1]*(np.arange(len(data))) + sol.x[0] )
    
    else:
        if endlen is None:
            endlen = nperseg
        assert endlen<len(data)//2, '`endlen` is too big!'
        
        if data_low is None:
            data_rf = np.fft.rfft(np.array(data_i))
            data_rf[int(len(data_i)/nperseg):]=0.
            data_low = np.real(np.fft.irfft(data_rf, n=len(data_i)))
        else:
            assert len(data_low)==len(data), 'The lengths of `data_low` and `data` must be equal!'
        
        w1 = tanhstep(np.arange(endlen), center=endlen/4*3, hwhm=endlen/16, start_from_zero=True)
        w2 = 1-w1
        w3 = w1[::-1]
        w4 = w2[::-1]
        data_low[:endlen] *= w1
        data_low[-endlen:] *= w3

        # fit the beginning
        if fit_order=='quadratic':
            msk1 = np.array(mask[:endlen]).astype(bool)
            dat1 = (data_i[:endlen])[msk1]
            tim1 = (time[:endlen])[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1)//2]
            x10 = (tim1[-1]+tim1[0])/2
            x20 = ((dat1[0]-cent)/(tim1[0]-x10)**2 + (dat1[-1]-cent)/(tim1[-1]-x10)**2)/2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2]*(tim1-x[1])**2 + x[0]))**2), 
             [x00, x10 , x20])
            data_low[:endlen] += (sol.x[2]*(np.arange(endlen)-sol.x[1])**2 + sol.x[0]) * w2
        else:
            msk1 = np.array(mask[:endlen]).astype(bool)
            dat1 = (data_i[:endlen])[msk1]
            tim1 = (time[:endlen])[msk1]
            tim10 = tim1[0]
            tim1 = tim1 - tim10
            x00 = (dat1[0] * tim1[-1] - dat1[-1] * tim1[0]) / (tim1[-1] - tim1[0])
            x10 = (dat1[-1] - dat1[0]) / (tim1[-1] - tim1[0])
            x20 = 0.
            x30 = 0.
            sol = minimize(lambda x: np.nansum((dat1 - (x[3]*tim1**3 + x[2]*tim1**2 + x[1]*tim1 + x[0]) )**2), [x00, x10 , x20, x30])
            data_low[:endlen] += (sol.x[3]*(np.arange(endlen))**3
                                 + sol.x[2]*(np.arange(endlen))**2 
                                 + sol.x[1]*(np.arange(endlen)) + sol.x[0] ) * w2
        
        # fit the end:
        if fit_order=='quadratic':
            msk1 = np.array(mask[len(data_i)-endlen:]).astype(bool)
            dat1 = (data[len(data)-endlen:])[msk1]
            tim1 = (time[len(data)-endlen:])[msk1]        
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1)//2]
            x10 = (tim1[-1]+tim1[0])/2
            x20 = ((dat1[0]-cent)/(tim1[0]-x10)**2 + (dat1[-1]-cent)/(tim1[-1]-x10)**2)/2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2]*(tim1-x[1])**2 + x[0]))**2), 
             [x00, x10 , x20])
            data_low[-endlen:] += (sol.x[2]*(np.arange(endlen)+time[len(data)-endlen]-sol.x[1])**2 + sol.x[0]) * w4
        else:
            msk1 = np.array(mask[len(data_i)-endlen:]).astype(bool)
            dat1 = (data[len(data)-endlen:])[msk1]
            tim1 = (time[len(data)-endlen:])[msk1]
            tim10 = tim1[0]
            tim1 = tim1 - tim10
            x00 = (dat1[0] * tim1[-1] - dat1[-1] * tim1[0]) / (tim1[-1] - tim1[0])
            x10 = (dat1[-1] - dat1[0]) / (tim1[-1] - tim1[0])
            x20 = 0.
            x30 = 0.
            sol = minimize(lambda x: np.nansum((dat1 - (x[3]*tim1**3 + x[2]*tim1**2 + x[1]*tim1 + x[0]) )**2), [x00, x10 , x20, x30])
            data_low[-endlen:] += (sol.x[3]*(np.arange(endlen)+time[len(data)-endlen]-tim10)**3
                                 + sol.x[2]*(np.arange(endlen)+time[len(data)-endlen]-tim10)**2 
                                 + sol.x[1]*(np.arange(endlen)+time[len(data)-endlen]-tim10) + sol.x[0] ) * w4
    
    return data_low
    

def detrend_piecewise(data, big_mask=None, mask=None, nperseg=None, endlen=None, nsig_step=1.5, 
                      checklen=64, residual_threshold_step_in_sigma=2., 
                      slice_for_residual_step=5, return_trend=False):
    """
    Detrends the data using `trend_fft` function in data which can contain
    several pieces separated by a step. Different pieces are separated by holes in
    `mask`. If the difference between the ends of the two subsequent pieces 
    exceeds high-frequency standard deviation of the data multiplied by `nsig_step`,
    then the pieces are detrended separately.
    Attention: trend will contain nans in the places of mask.

    Parameters
    ----------
    data : numpy.array or list of float
        Data vector to be detrended
    nperseg : int
        Wavelength below which the frequencies should be included in the trend.
    endlen : int
        Length of the data in the ends to be fitted by quadratic polynomial
    mask : array of bool
        Masking good data points with `True`, bad data points with `False`.
    nsig_step : float
        Minimal step between the edges of t wo subsequent pieces so that they 
        will be detrended separately (in units of high-frequency std of `data`)
    checklen : int
        Length of gap extension data from the edges used to calculate the 
        residuals to determine where are the steps.
    residual_threshold_step_in_sigma : float
        In the end, the trend is verified for having steps around holes in 
        the residuals. If the residuals before and after the hole differ more than
        by `residual_threshold_step_in_sigma` sigma, the piece will be cut, and 
        its ends will be fitted separately.
    slice_for_residual_step : int
        Length of slice used to calculate the median values around the hole
        which are needed to identify steps after the first trend attempt.
    return_trend : bool
        Whether to return trend in addition to detrended data

    Returns
    -------
    if return_trend:
        detrended, trend : numpy.arrays
        Detrended data and trend
    else:
        detrended : numpy.array
        `detrended` is detrended data containing zeros in the bad places of masks.
        `trend` is the low-frequency trend of the data containinhg nans
        in the bad places of masks

    """
    threshold_step_size_std = residual_threshold_step_in_sigma # redefining names
    medlen = slice_for_residual_step
    
    data = np.array(data) # copying and going to numpy
    
    nanmask = ~np.isnan(data)
    if not (nanmask).all(): # interpolating data if it contains nans
        data_i = interpolate_over_mask(data, mask=nanmask) #will need for fft
    else:
        data_i = data
               
    if mask is None:
        mask = nanmask
    else:
        mask = np.array(mask).astype(bool)
        mask *= nanmask
        
    if big_mask is None: # big mask shows potential jumps. If there is no big mask, no jumps will be checked for.
        big_mask = clean_small_holes(mask, max_hole=5) #np.ones(len(data)).astype(bool)
    else:
        big_mask =np.array(big_mask).astype(bool)
        
    #remove cases when there is only one unmasked point surrounded by masked because these pieces are hard to detrend
    (big_mask[1:-1])[(big_mask & ~np.roll(big_mask,1) & ~np.roll(big_mask,-1))[1:-1]] = False 
    
    if nperseg is None:
        nperseg=len(data)//10
    if endlen is None:
            endlen = nperseg
    
    detrended = np.zeros(len(data)) # will collect here the result
    
    ind_i = (np.where(mask)[0])[0] # start where there is the first valid point
    ind_f = (np.where(mask)[0])[-1]+1 # end with the last valid point
    mask0 = np.array(big_mask[ind_i:ind_f]) # mask of the working region
    endpieces = np.where((~mask0)*np.roll(mask0,1))[0] + ind_i # ends of pieces split by holes
    endpieces = np.concatenate((endpieces, [ind_f]))
    begpieces = np.where(mask0*np.roll(~mask0,1))[0] + ind_i # beginnings of pieces split by holes
    begpieces = np.concatenate(([ind_i], begpieces))
    
    fdat = np.abs(np.fft.rfft(data_i))
    std_fast_0 = (np.mean((fdat[-len(data)//4:])**2)/(len(data)))**0.5 #standard deviation of fase modes
    std_fast = std_fast_0+np.max(fdat)/20000 #I don't remember why exactly this was made but it worked
    
    # now we will select the bad pieces which have to be detrended separately:
    beg_big_pieces = [ind_i]
    end_big_pieces = []
    for i in range(1,len(begpieces)):
        i1 = max(endpieces[i-1]-checklen, begpieces[i-1]) # if the piece before the hole is too small, take it all
        i2 = min(begpieces[i]+checklen, endpieces[i]) # if the piece after the hole is too small, take it all
        mask1 = big_mask[i1 : i2]
        dat1 = (data_i[i1 : i2])[mask1]
        tim1 = np.arange(len(mask1))[mask1]
                
        x00 = np.nanmean(dat1) # define initial fit parameters
        cent = dat1[len(dat1)//2]
        x10 = (tim1[-1]+tim1[0])/2
        x20 = ((dat1[0]-cent)/(tim1[0]-x10)**2 + (dat1[-1]-cent)/(tim1[-1]-x10)**2)/2
        sol = minimize(lambda x: np.nansum((dat1 - (x[2]*(tim1-x[1])**2 + x[0]))**2), [x00, x10 , x20])
        # tr_end = sol.x[2]*(tim1-sol.x[1])**2 + sol.x[0]
        mres = (sol.fun/len(dat1))**0.5 # mean residual
    
        if mres > nsig_step * std_fast: #if the fit does not fit the two pieces together, detrend them separately
            end_big_pieces.append(endpieces[i-1])
            beg_big_pieces.append(begpieces[i])
    end_big_pieces.append(ind_f)
    
    # make one detrending and check it is ok inside the pieces:
    for i in range(len(beg_big_pieces)):
        piece = np.array(data_i[beg_big_pieces[i]:end_big_pieces[i]])
        mask_p = mask[beg_big_pieces[i]:end_big_pieces[i]]
        big_mask_p = big_mask[beg_big_pieces[i]:end_big_pieces[i]]
        trend_local = trend_fft(piece, nperseg=nperseg, endlen=endlen, mask=mask_p) 
        # TODO: make here something to avoid repeating detrending if all is fine
        begholes = np.where(big_mask_p*np.roll(~big_mask_p,-1))[0]+1 #define begholes inside the piece
        endholes = np.where(big_mask_p*np.roll(~big_mask_p,1))[0]
        
        beg_small_pieces = [0]
        end_small_pieces = []
        for j in range(len(begholes)):
            if (    (len(np.where(~np.isnan(piece[begholes[j]-medlen:begholes[j]]))[0])>=3)
                and (len(np.where(~np.isnan(piece[endholes[j]:endholes[j]+medlen]))[0])>=3)
                and ((np.abs(np.nanmedian(piece[begholes[j]-medlen:begholes[j]] 
                                  - trend_local[begholes[j]-medlen:begholes[j]])
                    -np.nanmedian(piece[endholes[j]:endholes[j]+medlen] 
                                  - trend_local[endholes[j]:endholes[j]+medlen])
                             )/ std_fast) > threshold_step_size_std)):
                #inds_step.append((beg_big_pieces[i]+begholes[j], beg_big_pieces[i]+endholes[j]))
                end_small_pieces.append(begholes[j])
                beg_small_pieces.append(endholes[j])
        end_small_pieces.append(len(piece))
        if len(end_small_pieces)==1:
            detrended[beg_big_pieces[i] : end_big_pieces[i]] = (
                np.array(data[beg_big_pieces[i]:end_big_pieces[i]])-trend_local)
        else:
            for k in range(len(beg_small_pieces)):
                small_piece = np.array(piece[beg_small_pieces[k]:end_small_pieces[k]])
                mask_s = np.array(mask_p[beg_small_pieces[k]:end_small_pieces[k]])
                trend_s_0 = np.array(trend_local[beg_small_pieces[k]:end_small_pieces[k]])
                trend_s = trend_fft(small_piece, nperseg=nperseg, endlen=endlen, mask=mask_s, data_low=trend_s_0) 
                detrended[beg_big_pieces[i]+beg_small_pieces[k]:beg_big_pieces[i]+end_small_pieces[k]] = (
                    np.array(data[beg_big_pieces[i]+beg_small_pieces[k]:beg_big_pieces[i]+end_small_pieces[k]])-trend_s)
                #trend[beg_big_pieces[i]:end_big_pieces[i]] = trend_local
    
    trend = np.array(data_i-detrended)
    if return_trend:
        return detrended, trend
    else:
        return detrended
    
    
    
def detrend_mov_ave(data, length=128, mask=None, method='median', endmethod='mean', ends='symmetric', endlen=None, 
                 return_trend=True, keep_overall_level=False, remove_outliers=False, nsig_outliers=nsig_outliers_0,
                 detr_length_outliers=detr_length_outliers, remove_outliers_light=False):
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
    
    if mask is None:
        mask=~np.isnan(data)
    
    mask = np.array(mask)
    assert len(data)==len(mask), 'Data and mask must be of the same length!'
    
    if remove_outliers:
        data1 = detrend_mov_ave(np.array(data), length=detr_length_outliers, mask=mask, method='median', endmethod='mean', 
                                 ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False, remove_outliers=False)
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask
        
    elif remove_outliers_light:
        outlier_mask = np.abs(data - np.nanmean(data[mask])) < nsig_outliers * np.nanstd(data[mask])
        mask *= outlier_mask
    
    if ends=='asymmetric':
        trend = []
        res = []
        if endlen is None:
            l4 = int(np.ceil(length/16))
        elif endlen>int(length/2):
            l4 = int(length/2)
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
        elif endlen>int(length/2):
            l4 = int(length/2)
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
        

def mask_outliers_fun(data, mask=None, nsig0=nsig_outliers_0, nsig1=nsig_outliers_psd, 
                      nsig_positive=nsig_outliers_0_pos, nsig_negative=nsig_outliers_0_neg,
                      detr_length=detr_length_outliers, detr_method='median', detr_endmethod='mean', detr_ends='symmetric',
                      detr_endlen=None, return_symmetric_mask=False):
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
        See the desctiption in detrend_mov_ave. The default is 'symmetric'.
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
    
    
    data0 = np.array(data)
    nanmask=~np.isnan(data)
    data0[~nanmask] = np.nanmean(data0)
        
    if mask is None:
        mask=nanmask    
        
    if nsig0 is None:
        if nsig_negative is not None:
            if nsig_positive is not None:
                nsig0 = min(nsig_negative, nsig_positive, nsig1)
            else: 
                nsig0 = nsig_negative
        elif nsig_positive is not None:
            nsig0 = nsig_positive
        else:
            nsig0 = nsig1
            
    if nsig1 is None:
        nsig1 = nsig0
    if nsig_positive is None:
        nsig_positive = nsig0
    if nsig_negative is None:
        nsig_negative = nsig0
    
    assert ((nsig_positive is not None) and (nsig_negative is not None)), '`nsig` is not specified correctly!'
    
    mask1 = copy.deepcopy(mask)
    
    if detr_method is not None:
        data1 = detrend_mov_ave(np.array(data0), length=detr_length, mask=mask1, method=detr_method, endmethod=detr_endmethod, 
                                     ends=detr_ends, endlen=detr_endlen, return_trend=False, keep_overall_level=False, remove_outliers=False)
    else:
        data1 = data0

    outlier_mask = np.abs(data1 - np.nanmean(data1[mask1])) < nsig0 * np.nanstd(data1[mask1])
    mask1 *= outlier_mask #first time to remove big outliers that spoil the std
    
    outlier_mask_p = (data1 - np.nanmean(data1[mask1])) < nsig_positive * np.nanstd(data1[mask1])
    outlier_mask_n = (data1 - np.nanmean(data1[mask1])) > - nsig_negative * np.nanstd(data1[mask1])
    if not return_symmetric_mask:
        return outlier_mask_p * outlier_mask_n
    else: 
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask1])) < nsig1 * np.nanstd(data1[mask1])
        return outlier_mask_p * outlier_mask_n, outlier_mask



def veto_combined(data, template, fn=0.01, maxdev_std=4., maxdev_ampl=None, 
                  zero_ends_len_abs=3, zero_ends_len_rel=0.1, 
                  return_amplitude=True, return_reason=True, inn_prod_min=inn_prod_min_veto,
                  return_scores=False):
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
    is_good= True
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
        is_good = False #check that it is not a step instead of transit
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
    
    inn_prod = dat @ temp / ((dat@dat)*(temp@temp))**0.5
    if np.abs(inn_prod)<inn_prod_min:
        reason += 'inner_product '
        is_good = False
        
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
        is_good = False
    if maxdev_std_dat>maxdev_std:
        reason += 'maximal_deviation_std '
        is_good = False
    if ((maxdev_ampl is not None) and (maxdev_ampl_dat>maxdev_ampl)):
        reason += 'maximal_deviation_amplitude '
        is_good = False
     
    if return_scores:
        if return_amplitude:
            if return_reason:
                return is_good, [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod, inn_prod_min], ampl, reason
            else:
                return is_good, [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod, inn_prod_min], ampl
        else:
            if return_reason:
                return is_good, [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod, inn_prod_min], reason
            else:
                return is_good, [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod, inn_prod_min]
    else:
        if return_amplitude:
            if return_reason:
                return is_good, ampl, reason
            else:
                return is_good, ampl
        else:
            if return_reason:
                return is_good, reason
            else:
                return is_good



def veto_parts(data_wh_full, template_wh_full, indices, mask=None, tlen=None, 
               fn=0.01, n_parts=4, nsig_outliers_var=3., nperseg_psd=nperseg_psd,
               return_amplitudes=False, return_scores=False, splitting='snr'):
    """
    Parameters:
    ---------------------------
    data_wh_full : numpy.array of float
        Whitened data for the entire quarter with cut zero padding ends.
    template_wh_full : 
        Whitened template rolled in such a way that the center of the transit is at zero index
    indices : array-like 1d of bool
        The position where the transit center is located
    tlen : int or None
        Length of the non-zero part of the template (usually set by the whitening filter, can be
        also affected by the notch filter). If None, will be measured.
    fn : float
        False negative rate that is needed to ve achieved in the veto. Calculated from chi2 distribution
    n_parts : int
        Number of parts to which the template should be split (expecting to have the same SNR in them)
    nsig_outliers_var : float
        deviation from the mean in units of STD to be considered as outlier in statistic when measuring variance
    return_scores : bool
        Whether to return also the chi2 scores or only the veto result
    splitting : str : 'snr' or 'len'
        According to what to split the template: equal parts in snr or in length.
        
    Returns
    ---------------------------
    
    If return_scores:
        If return_amplitudes:
            is_goods : list of bool
                Veto result. If True, then the signal is good.
            amplitudes : list of float,
                Estimated amplitudes of transith at every index.
            chi2_scores : list of lists of float
                Values of chi square for every index for all the parts ans the total (the last value)
        Else:
            is_goods, chi2_scores
    Else:
        If return_amplitudes:
            is_goods, amplitudes
        Eles:
            is_goods
    """   
    assert (splitting=='snr' or splitting=='len') , "`splitting` must be 'snr' or 'len'!"
    data_wh_full = np.array(data_wh_full)
    template_wh_full = np.array(template_wh_full)
    if tlen is None:
        temp_smthnd = np.abs(template_wh_full)
        temp_smthnd = (temp_smthnd + np.roll(temp_smthnd,1)+np.roll(temp_smthnd,-1))/3 # averaging the rolled template making it smooth
        inds_small = np.where((temp_smthnd/np.max(temp_smthnd)<0.001))[0] #choosing where the tamplate is close to zero
        #TODO: maybe change 0.001 to 0.0001 in furhter runs!
        inds_small = inds_small[np.where((inds_small>nperseg_psd//4)&(inds_small<=len(template_wh_full)-nperseg_psd//4))[0]] #if suddenly there is zero in the beginning, ignore
        tlen = inds_small[0] + (len(temp_smthnd) - inds_small[-1]) #taking parts where the template is not close to zero
        # this weird procedure is needed because we are dealing with the whitened template, and its shape can be complex, so at some point
        # it can reash zero, but it does not mean that it is the end of the transit. We are smoothening it, so that if it is zero only in one point,
        # the smoothened one will be nonzero. 
        # then, we are setting a threshold and selecting everything non-zero
        # note the rolling! the transit center should be in the zeroth index!
    
    if mask is None:
        mask = ~np.isnan(data_wh_full).astype(bool)
        
    local_th = chi2.ppf(1-fn/(n_parts+1), df=1) # local threshold, chi2 of parts will be compared to it
    total_th = chi2.ppf(1-fn/(n_parts+1), df=4) # total threshold, sum of chi2 of all parts will be compared to it
    
    template_cut_centered = np.roll(template_wh_full, tlen//2)[:tlen] # put the transit center in the center 
    stat_for_var = np.convolve(template_cut_centered,data_wh_full, mode='valid') # make a small sample of statistic to estimate the variance
    stat_for_var = stat_for_var[(np.abs(stat_for_var)<nsig_outliers_var*np.nanstd(stat_for_var))&mask[tlen//2:len(mask)-((tlen-1)//2)]]
    # quarter std correction so that the SNR of matched filtering the template is in units.
    var_stat_global = np.nanvar(stat_for_var)
    if splitting=='snr':
        chi2_temp_indices = [0] + list(np.searchsorted(np.cumsum(template_cut_centered**2) / np.dot(template_cut_centered,template_cut_centered),
                                                   (np.arange(n_parts)/n_parts)[1:])) + [len(template_cut_centered)]  
    else:
        chi2_temp_indices = list((len(template_cut_centered) * np.arange(n_parts)/n_parts).astype(int)) + [len(template_cut_centered)]
    chi2_temp_indices = list(set(chi2_temp_indices)) #remove repetitions
    chi2_temp_indices.sort()
    n_parts = len(chi2_temp_indices)-1
    while ((chi2_temp_indices-np.roll(chi2_temp_indices, 1))[1:]<4).any(): # want every piece to be at least 4 bins
        n_parts -= 1
        if splitting=='snr':
            chi2_temp_indices = [0] + list(np.searchsorted(np.cumsum(template_cut_centered**2) / np.dot(template_cut_centered,template_cut_centered),
                                                   (np.arange(n_parts)/n_parts)[1:])) + [len(template_cut_centered)]  
        else:
            chi2_temp_indices = list((len(template_cut_centered) * np.arange(n_parts)/n_parts).astype(int)) + [len(template_cut_centered)]  
    
    if len(chi2_temp_indices)<=2:       
        if return_scores:
            if return_amplitudes:
                return np.ones(len(indices)).astype(bool), np.ones(len(indices))*np.nan, np.ones(len(indices))*np.nan, local_th, total_th
            else:
                return np.ones(len(indices)).astype(bool), np.ones(len(indices))*np.nan, local_th, total_th
        else:
            if return_amplitudes:
                return np.ones(len(indices)).astype(bool), np.ones(len(indices))*np.nan
            else:
                return np.ones(len(indices)).astype(bool)
            
    n_parts = len(chi2_temp_indices)-1
    templates_wh_local = []
    vars_stat_local = []
    actual_Vs = []
    for i in range(1,len(chi2_temp_indices)):
        i1 = chi2_temp_indices[i-1]
        i2 = chi2_temp_indices[i]
        template_wh_local = np.array(template_cut_centered[i1:i2])
        stat_for_var = np.convolve(template_wh_local, data_wh_full, mode='valid')
        stat_for_var = stat_for_var[(np.abs(stat_for_var)<nsig_outliers_var*np.nanstd(stat_for_var))&mask[
                                                                len(template_wh_local)//2 : len(mask)-((len(template_wh_local)-1)//2)]]
        #quarter std correction so that the SNR of matched filtering the template is in units.
        var_stat_local = np.nanvar(stat_for_var)
        templates_wh_local.append(template_wh_local)
        vars_stat_local.append(var_stat_local)
        actual_Vs.append(1/var_stat_local - 1/var_stat_global)
    
    is_goods = []
    chi2_scores = []
    amplitudes = []
    for index in indices:
        if (index >= tlen//2) and (index < len(data_wh_full) - (tlen+1)//2):
            data_wh = data_wh_full[index-tlen//2:index+(tlen+1)//2]
            A_global = np.dot(template_cut_centered,data_wh) / var_stat_global #global amplitude
            amplitudes.append(A_global)
            chi2_total = 0
            is_good = True
            chi2_scores_index = []
            for i in range(len(chi2_temp_indices)-1):
                i1 = chi2_temp_indices[i]
                i2 = chi2_temp_indices[i+1]
                template_wh_local = np.array(template_cut_centered[i1:i2])
                A_local = np.dot(template_wh_local,data_wh[i1:i2]) / vars_stat_local[i]#np.dot(template_wh[i1:i2],template_wh[i1:i2])
                chi2_local = (A_local - A_global)**2 / (actual_Vs[i])
                chi2_scores_index.append(chi2_local)
                if chi2_local > local_th:
                    is_good = False
                chi2_total += chi2_local
            chi2_scores_index.append(chi2_total)
            if chi2_total > total_th:
                is_good = False
        
        elif (index >= tlen//2) or (index < len(data_wh_full) - (tlen+1)//2):
            if index < len(data_wh_full) - (tlen+1)//2: # it is in the beginning
                data_wh = data_wh_full[:index+(tlen+1)//2]
                template_not_full = template_cut_centered[:index+(tlen+1)//2]
            else: # it is in the end
                data_wh = data_wh_full[index-tlen//2:]
                template_not_full = template_cut_centered[len(template_cut_centered) - (len(data_wh_full) - index) - tlen//2:]
            stat_for_var = np.convolve(template_not_full,data_wh_full, mode='valid')
            stat_for_var = stat_for_var[(np.abs(stat_for_var)<nsig_outliers_var*np.nanstd(stat_for_var))&mask[
                                     len(template_not_full)//2:len(mask)-((len(template_not_full)-1)//2)]]
            #quarter std correction so that the SNR of matched filtering the template is in units.
            var_stat_global_not_full = np.nanvar(stat_for_var)
            A_global_not_full = np.dot(template_not_full,data_wh) / var_stat_global_not_full #global amplitude
            amplitudes.append(A_global_not_full)
            n_parts_not_full = int(np.round(n_parts / tlen * len(template_not_full) ))
            if splitting=='snr':
                chi2_temp_indices_not_full = [0] + list(np.searchsorted(np.cumsum(template_not_full**2) / var_stat_global_not_full,
                                                   (np.arange(n_parts_not_full)/n_parts_not_full)[1:])) + [len(template_not_full)]  
            else:
                chi2_temp_indices_not_full = list((len(template_not_full) * np.arange(n_parts_not_full)/n_parts_not_full).astype(int)) + [len(template_not_full)]
            chi2_temp_indices_not_full = list(set(chi2_temp_indices_not_full)) #remove repetitions
            chi2_temp_indices_not_full.sort()
            n_parts_not_full = len(chi2_temp_indices_not_full)-1
            
            while ((chi2_temp_indices_not_full-np.roll(chi2_temp_indices_not_full, 1))[1:]<4).any():
                n_parts_not_full -= 1
                if splitting=='snr':
                    chi2_temp_indices_not_full = [0] + list(np.searchsorted(np.cumsum(template_not_full**2) / np.dot(template_not_full,template_not_full),
                                                           (np.arange(n_parts_not_full)/n_parts_not_full)[1:])) + [len(template_not_full)] 
                else:
                    chi2_temp_indices_not_full = list((len(template_not_full) * np.arange(n_parts_not_full)/n_parts_not_full).astype(int)) + [len(template_not_full)]

            if n_parts_not_full<2:# or ((chi2_temp_indices_not_full-np.roll(chi2_temp_indices_not_full, 1))[1:]<3).any():
                is_good = True
                chi2_scores_index = [np.nan, np.nan]
            else:
                chi2_total = 0
                is_good = True
                chi2_scores_index = []
                for i in range(1,len(chi2_temp_indices_not_full)):
                    i1 = chi2_temp_indices_not_full[i-1]
                    i2 = chi2_temp_indices_not_full[i]
                    template_wh_local = np.array(template_not_full[i1:i2])
                    stat_for_var = np.convolve(template_wh_local, data_wh_full, mode='valid')
                    stat_for_var = stat_for_var[(np.abs(stat_for_var)<nsig_outliers_var*np.nanstd(stat_for_var))&mask[
                                                                            len(template_wh_local)//2:len(mask)-((len(template_wh_local)-1)//2)]]
                    #quarter std correction so that the SNR of matched filtering the template is in units.
                    var_stat_local = np.nanvar(stat_for_var)
                    actual_V = 1/var_stat_local - 1/var_stat_global_not_full
                    A_local = np.dot(template_wh_local,data_wh[i1:i2]) / var_stat_local#np.dot(template_wh[i1:i2],template_wh[i1:i2])
                    chi2_local = (A_local - A_global_not_full)**2 / (actual_V)
                    chi2_scores_index.append(chi2_local)
                    if chi2_local > local_th:
                        is_good = False
                    chi2_total += chi2_local
                chi2_scores_index.append(chi2_total)
                if chi2_total > total_th:
                    is_good = False
        else:
            raise ValueError('Invalid template size!')    
        chi2_scores.append(chi2_scores_index)   
        is_goods.append(is_good)
        
    if return_scores:
        if return_amplitudes:
            return is_goods, amplitudes, chi2_scores, local_th, total_th
        else:
            return is_goods, chi2_scores, local_th, total_th
    else:
        if return_amplitudes:
            return is_goods, amplitudes
        else:
            return is_goods


def score_correction(scores_i, return_function=False, negative_values=False):
    scores_i = np.array(scores_i)
    argsorted = np.argsort(scores_i)
    argargsorted = np.argsort(argsorted)
    sorted_scores = scores_i[argsorted]
    if negative_values is None:
        if (scores_i<0).any():
            negative_values = True
        else:
            negative_values = False
    if negative_values==True:
        scores_c = 2**0.5 * erfinv(2*(np.arange(len(sorted_scores))+0.5)/len(sorted_scores) - 1)
    elif negative_values==False:
        scores_c = 2**0.5 * erfinv((np.arange(len(sorted_scores))+0.5)/len(sorted_scores))
    if return_function:
        # indices = ~np.isinf(scores_c)
        # funct = interp1d(sorted_scores[indices], scores_c[indices])
        funct = interp1d(sorted_scores, scores_c)
    if not return_function:
        return scores_c[argargsorted]
    else:
        return scores_c[argargsorted], funct
            

def unequal_amplitude_veto(values, variances=None, A_all=None, indmax=None, A_1=None, units_of_snr=True):
    values = np.array(values)
    if variances is None:
        variances = np.ones(len(values))
    else:
        variances = np.array(variances)
    if A_all is None:
        A_all = np.sum(values/variances) / np.sum(1/variances)
    if indmax is None:
        indmax = np.argmax(values)
    if A_1 is None:
        A_1 = values[indmax]
    
    expected_mean_H_all = sum(A_all**2/variances) - A_1 * A_all / variances[indmax]
    expected_variance_H_all = np.abs(sum(A_all**2/variances) - 2 * A_1 * A_all / variances[indmax] + A_1**2 / variances[indmax])
    statistic = -(A_all * sum(values/variances) - A_1 * values[indmax]/variances[indmax] - expected_mean_H_all)
    if np.isclose(expected_variance_H_all,0):
        statistic = np.inf
    elif units_of_snr:
        statistic = statistic/expected_variance_H_all**0.5
    else:
        statistic = statistic/expected_variance_H_all
    return statistic


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
    with open(fileloc, 'w') as file:
        print(*var, file=file)

    
def read_quarters_from_file(name, location='', dimension=2, to_array=True):
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
        fileloc = name
        
    if name[-4:]!='.txt':
        fileloc = name +'.txt'
        
    with open(fileloc, 'r') as file:
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
    #TODO: fix scale in the log counts 
    data = np.array(data[~np.isnan(data)])
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
            sh = np.nanmin(counts[counts!=-np.inf]) - (np.nanmax(counts[counts!=-np.inf])-np.nanmin(counts[counts!=-np.inf]))/20
            counts = counts - sh
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
    
    
