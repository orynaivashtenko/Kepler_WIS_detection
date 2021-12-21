#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 12:16:38 2021

@author: orynaivashtenko
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import requests
import wget
from astropy.io import fits
import copy
from scipy.stats import chi2
from scipy.optimize import minimize
sys.path.append(os.getcwd())
import template_module as tm




def unb_var(arr): #unbiased variance estimate (numpy.var is biased!)
    arr1 = np.array(arr[~np.isnan(arr)])
    if len(arr1)==1:
        return 0
    else:
        return 1 / (len(arr1)-1) * np.sum(np.abs(arr1 - np.mean(arr1))**2)


def MAD(arr): #unbiased variance estimate (numpy.var is biased!)
    arr1 = np.array(arr[~np.isnan(arr)])
    return np.median(np.abs(arr1 - np.median(arr1)))

def inhmedian(arr, choose_min=True):
    """
    Inherent median. Returns the value from this array closest to the mean.
    If there are two such:
    If choose_min, the smaller closest, else - the larger closest
    Also does not return non-integer values if there is no
    exact median. For example, in [1,2,3,4] will return not 2.5, but 2 (if choose_min==True)
    if choose_min==False, will return 3. Also does not always convert everything to float
    ------
    choose_min : in case of ambiguity, return nte minimum value from two
    """
    arr1 = np.sort(np.array(arr[~np.isnan(arr)]))
    if len(arr1)%2!=0:
        return arr1[(len(arr1)-1)//2]
    else:
        a1 = arr1[(len(arr1)-1)//2]
        a2 = arr1[len(arr1)//2]
        m = np.mean(arr)
        if np.abs(a1-m) < np.abs(a2-m):
            return a1
        elif np.abs(a1-m) > np.abs(a2-m):
            return a2
        else:
            if choose_min:
                return a1
            else:
                return a2

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
    binwid = (edgelist[-1]-edgelist[0])/(len(edgelist)-1)
    
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
            ax.bar(bins, counts, binwid ,color=color, alpha=alpha, label=label)

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
        
    
def measure_psd(x, mask=None, nperseg=64, noverlap=-1, window='hann', dt=1.,
                      subtract_mean=True, rfft=True):
    """
    Function for measuring psd of the signal with Welch method. 
    
    Parameters:
    ------------------
    x : np.ndarray of 1d, float
        signal in time domain
        
    mask, mask_y : np.ndarray 1d, bool or float zeros and ones
        mask being zeros where the signal have nans or other unwanted points
        By default (when None) are replaced by True array of length of the signal
        
    nperseg : int
        the length of a sample used in Welch method
        Default is 64
        
    noverlap : int
        length of the intersection of the Welch samples
        Default -1 is replaced by half of the nperseg
        
    window : string
        window function by which the signals are multiplied to avoid boundary effects and crosstalk
        Possible options: 'hann', 'boxcar', where 'boxcar' is just normalized bow, or no window
        Default is 'hann'. It is not recommended to use 'boxcar'.
    
    dt : float
        Time interval between the two subsequent (evenly spaced) entries
        Default is 1.
        
    subtract_mean : bool
        If True, the mean of the signal is subtracted. It is recommended to do it because
        otherwise unwanted power may appear in the zero frequency
        Default is True
    
    rfft : bool
        If True, the signal is treated as real, and all ffts are rffts. The length of the frequency samples
        are the nperseg//2+1.
        Default is True
    
    Returns:
    -----------
    freqs, psd
    """
    lenx = len(x)
    if mask is None:
        mask = np.ones(lenx).astype(bool)
    elif type(mask) is not bool:
        mask = mask.astype(bool)
    
    if noverlap==-1:
        noverlap=nperseg//2
        #to normalize ones or not to normalize?... YES(checked with scipy)
    windowdict = {'boxcar' : np.ones(nperseg)/np.linalg.norm(np.ones(nperseg)), 
        'hann' : 0.5 * (1-np.cos(2*np.pi*np.arange(nperseg)/(nperseg)))/np.linalg.norm(
            0.5 * (1-np.cos(2*np.pi*np.arange(nperseg)/(nperseg))))}
    
    if subtract_mean:
        x = x-np.nanmean(x)
        
    if rfft:
        #measuring psds for x:
        psds = [np.abs(np.fft.rfft(windowdict[window]*x[ind1:ind1+nperseg])
                )**2 for ind1 in range(0,len(x)-nperseg,nperseg-noverlap
                        ) if np.sum((~mask)[ind1:ind1+nperseg])==0]
        psd = np.mean(np.array(psds), axis=0)
        freqs = np.fft.rfftfreq(nperseg, d=dt)
    
    else:
        #measuring psds for x:
        psds = [np.abs(np.fft.fft(windowdict[window]*x[ind1:ind1+nperseg])
                )**2 for ind1 in range(0,len(x)-nperseg,nperseg-noverlap
                        ) if np.sum((~mask)[ind1:ind1+nperseg])==0]
        psd = np.mean(np.array(psds), axis=0)
        freqs = np.fft.fftfreq(nperseg, d=dt)
    return freqs, psd


def psd2wf(psd, support=None, rfft=True):
    """
    Generates a whitening filter of the needed length (support) from the PSD
    
    Parameters
    ----------
    psd : np.array
        
    support : int
        length of the needed filter
        if None, take the length of the psd
    
    rfft : bool
        If True, the signal is treated as real, and all ffts are rffts. The length of the frequency samples
        are the nperseg//2+1.
        Default is True

    Returns
    -------
    wf_td : np.array 1d
        whitening filter in time domain
    wf_fd : np.array 1d
        whitening filter in Fourier domain.

    """
    if rfft:
        if support is None:
            support=2*(len(psd)-1)

        assert support%2==0, 'support assumed to be even!'
        wpsd = copy.deepcopy(psd**(-0.5))
        wpsd[0]=0.

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


def detrend_amat(data, length=128, mask=None, method='median', endmethod='mean',
                 ends='symmetric', endlen=None, return_trend=True, keep_overall_level=False):
    """
    Subtracting the moving mean or median from the data sample
    --------------------------
    data : np.array 1d or list. Data sample from which the moving average has to be subtracted
    length : int. The length of sample to calculate moving average. At a given point, 
                    half length to the left and to the rifgt are taken.
    method : str. Kind of moving value to subtract: 'median' or 'mean'
    endmethod: str. Kind of moving value to subtract at the ends of the data (see below): 'median' or 'mean'
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
    endlen : int, default is None. If None, will be assigned depending on the method (see description of ends)
    return_trend : bool. Whether to return the trend in addition to the detrended data
    keep_overall_level : bool. If true, the total mean or median will be added to the data (that is preserved)
    --------------------------
    Returns: 
    if not return_trend: data with subtracted mean, np.array
    if return_trend: data with subtracted mean, np.array; trens (moving mean), np.array
    """
    if (method!='median'):
        for i in range(max(len(method), len('median'))):
            print(method[i], 'median'[i], method[i]=='median'[i])
    assert ((method=='mean') or (method=='median')), 'Incorrect method specified'
    data = np.array(data)
    if mask is None:
        mask=~np.isnan(data)
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
    


def interpolate_over_mask(data, mask=None, interp_method='linear', ends='continue', mitigate_hole_ends=False, endhole_method='median', mit_length=10):
    """
    Takes the data and the mask of holes, interpolates the holes. If the mask is not
    provided, finds nans in the data and interpolates them.
    
    Parameters
    ----------
    data : np.array 1d of floats
        data to be interpolated, may contain nans
    mask : np.array 1d of bool
        False entries mean there is a hole to be interpolated.
    interp_method : str
        Can be 'mean', 'median', 'iterative_fourier',' linear'(default)
        How to fill the holes. By mean of the data, by median of the data,
        by doing Fourier transform many times, cuting high frequencies,
        going back and substituting to the holes,
        or by linearly interpolating over the holes.
    ends : str
        Can be 'continue', 'zeros', 'keep'
        What to do to the ends of the data in linear method.
        'continue' is to take the first(last) value and fill by it all the rest.
        'zeros' meens to replace by zeros.
        'keep' means to leave what was in the original data (maybe nans)
    mitigate_hole_ends : bool
        If true, for the linear, mean and median interpolation, not the ends of the holes will be taken,
        but the average od points near the hole
    endhole_method : str : 'mean' or 'median'(default)
        The type of average for mitigate_hole_ends.
    mit_length : int. Default is 10
        The length of the slice for the averaging in the mitigate_hole_ends

    Returns
    -------
    interpolated data, 1d np.array of floats. 
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
        mask = ~np.isnan(data)
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
        data[indholes]=nan_average_filt(data, kernel_size=kernel_size, method=interp_method)[indholes]
    elif interp_method=='iterative_fourier':
        nrep = 6
        data1 = np.array(data)
        indholes = np.where(~ mask)[0]
        data1[indholes] = 0
        for i in range(nrep):
            fdata1 = np.fft.fft(data1)
            fdata1[len(data1)//10 + 4**i : -(len(data1)//10 + 4**i)]=0
            data1[indholes]=(np.fft.ifft(fdata1))[indholes]
        data=data1
    elif interp_method=='linear':
        ind1 = (np.where(mask)[0])[0] 
        ind2 = (np.where(mask)[0])[-1]
        #if there is a hole in the beginning or end, it cannot be interpolated
        data1 = np.array(data[ind1:ind2+1])
        mask1 = np.array(mask[ind1:ind2+1])
        begholes = np.where(mask1*np.roll(~mask1,-1))[0]
        endholes = np.where(mask1*np.roll(~mask1,1))[0]
        if not mitigate_hole_ends:
            for j in range(len(begholes)):
                data1[begholes[j]+1:endholes[j]]=np.interp(
                    np.arange(begholes[j], endholes[j]+1),np.array([begholes[j], endholes[j]]),
                    np.array([data1[begholes[j]],data1[endholes[j]]]))[1:-1]
        else:
            if endhole_method=='median':
                for j in range(len(begholes)):
                    i1 = max(begholes[j]-mit_length+1,0)
                    i2 = min(endholes[j]+mit_length,len(data1))
                    data1[begholes[j]+1:endholes[j]]=np.interp(
                        np.arange(begholes[j], endholes[j]+1),np.array([begholes[j], endholes[j]]),
                        np.array([np.nanmedian((data1[i1:begholes[j]+1])[np.where(mask1[i1:begholes[j]+1])[0]]),
                                  np.nanmedian((data1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            elif endhole_method=='mean':
                for j in range(len(begholes)):
                    i1 = max(begholes[j]-mit_length+1,0)
                    i2 = min(endholes[j]+mit_length,len(data1))
                    data1[begholes[j]+1:endholes[j]]=np.interp(
                        np.arange(begholes[j], endholes[j]+1),np.array([begholes[j], endholes[j]]),
                        np.array([np.nanmean((data1[i1:begholes[j]+1])[np.where(mask1[i1:begholes[j]+1])[0]]),
                                  np.nanmean((data1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            else:
                raise AssertionError('Incorrect endhole_method specified!')
        data[ind1:ind2+1]=data1
    if interp_method=='linear' or interp_method=='mean' or interp_method=='median':
        if ends == 'continue':
            if ind1 !=0:
                data[0:ind1]=data[ind1]
            if ind2 != len(data)-1:
                data[ind2:]=data[ind2]
        elif ends == 'zeros':
            if ind1 !=0:
                data[0:ind1]=0
            if ind2 != len(data)-1:
                data[ind2:]=0
        elif ends == 'keep':
            pass
        else:
            raise AssertionError('Unlefined ends parameter specified!')
    return data


def is_bad_fft(fftdat, amp_max=500, rfft=True):
    """
    
    If there are too narrow peaks that cannot be eliminated with measured psd, the star is bad
    ----------------------
    fftdat : np.array
    amp_peak : float, in units of sigma_tail limit amplitude of the peak (above it the peak is bad)
    width_peak : width of what is called a peak
    edge_per : limit value of periods allowed for peaks (if lower or equal, the peak is bad)
    dt : time stap of the data
    nperseg : length (in samples) of piece for psd measurement in Welch
    rfft : 
    """
    
    fftdat = np.array(np.abs(fftdat))
    if rfft:
        tail_var = np.nanmedian(fftdat[len(fftdat)//2:])
        flag = False
        for i in range(0, len(fftdat)-2):
            if ((fftdat[i+1]/fftdat[i]>2.) and (fftdat[i+1]/fftdat[i+2]>2.) and (fftdat[i+1]/tail_var)>amp_max*(len(fftdat)/4/(i+len(fftdat)/4)-0.18)/0.8):
                flag = True
                # this amp_max*(...) is the function cheching that the peak is too high. The function (...) is a decaying powet law. The peaks
                # decay at big frequencies, therefore the threshold for a too big peak is lower.
                break
        return flag
    else:
        raise AssertionError('Please finish the code for rfft=False !')


        
def veto_combined(data, template, fn=0.01, maxdev=2.8):
    """
    Fits the best amplitude for the template. Performs a chi squared testwith this template and the maximum deviation test.
    If both tests are satisfied, returns True. Otherwise False
    -----------------
    fn : float. False negative probability for chi squaredtest with known amplitude
    maxdev : float. Maximum deviation in absolute units for the maximum deviation test
    htol_temp : float. 
    -----------------
    If both chi2 and maxdev tests are satisfied, returns True. Otherwise False
    """
    inds = np.where(~np.isclose(template, 0))[0]#[[0,-1]]
    if inds[0]-1>=0: #if possible, include one index where the template is zero to ensure that the shape goes to zero
        inds = np.concatenate(([inds[0]-1], inds))
    if inds[-1]+1<len(template): #if possible, include one index where the template is zero to ensure that the shape goes to zero
        inds = np.concatenate((inds, [inds[-1]+1]))
    temp = np.array(template[inds])
    dat = np.array(data[inds])
    df = len(temp)
    x0 = chi2.ppf(1-fn, df)
    sol = minimize(lambda x: np.sum((dat-temp*x)**2), np.max(np.abs(dat))*np.sign(np.max(np.abs(dat))))
    maxdev_sq_comp = np.max((dat-temp*sol.x)**2)
    if sol.fun<=x0 and maxdev_sq_comp<=maxdev**2:
        return True
    else:
        return False
    
def veto_the_trigger(data, index, template=None, template_number=20, fn=0.01, maxdev=2.8):
    """
    Checks the goodness of fit of data by the template by chi squared and maximum deviation. The comparison
    is made with respect to the best chi squared fit of the template to the data. The conditions for
    the fit to be good are:
        - The chi squared value is smaller than the point where the chi squared distribution with the number 
          of degrees of freedom equal to the lengts of the template (truncated from extra zeros) is fn 
        - Maximum absolute deviation is smaller than maxdev
    --------------
    data : np.array 1d
        Data to be vetoed.
    index : int
        Index of the position of the peak of the template in the data where the matched-filtering triggered.
    template : np.array 1d.
        Template (one-transit, positive-valued, not fft-shifted) to fit to the data. If not specified,
        the template_module.template_bank[template_number] will be used. The template will be truncated
        to cut extra zeros (one extra zero from each side will be left to show that the template ended
        and reduce the score for peaks that continue beyond the template shape).
    template_number : int, <len(template_module.template_bank())
        The template number from the template bank that will be used in case when the template is not specified.
    fn : float
        False negative value of the chi-squared distribution with the number of degrees of freedom equal to the
        length of the truncated template. Note that this distribution is without accounting for multiple-comparison 
        effect (we are fitting the best amplitude for the template, to the real distribution of the scores
        will be shrinked closer to zero)
    maxdev : float
        Maximum allowed absolute deviation of the data from the best-fit template.
    
    --------------
    Returns:
    True if the trigger is good
    False if the trigger is bad
    
    """
    if template is None:
        template = tm.template_bank()[template_number]
    inds = np.where(~np.isclose(template, 0))[0]
    temp = np.array(template[inds[0]-2: inds[-1]+3]) #need these indices to have one zero point from each side to make sure the shape ended
    halflen=len(temp)//2
    data_cut = np.array(data[index-halflen:index+len(template)-halflen])
    return veto_combined(data_cut, temp, fn=fn, maxdev=maxdev)


def moving_var(signal, nperseg=256, ngap=64):
    """
    Calculates the moving variance of the signal. At every point, takes the window of length
    nperseg symmetrically from the two sides, excludes from the middle the window of length
    ngap and calculates the unbiased variance of the remaining. 
    At the ends of the singal, the data is taken only from one side up to the available length.
    Exclusion of window ngap from the middle is done to estimate the variance around this point and
    mitigate the impact of this point
    """
    mov_var = []
    sum1 = np.cumsum(signal)
    sum1 = np.concatenate(([0], sum1))
    sum2 = np.cumsum(signal**2)
    sum2 = np.concatenate(([0], sum2))
    for i in np.arange(0,ngap//2):
        mov_var.append(((sum2[i+nperseg//2]-sum2[i+ngap//2])/(nperseg//2-ngap//2)-((sum1[i+nperseg//2]-sum1[i+ngap//2])/(nperseg//2-ngap//2))**2)
                       *(nperseg//2-ngap//2)/(nperseg//2-ngap//2-1))
    for i in np.arange(ngap//2,nperseg//2):
        mov_var.append(((sum2[i-ngap//2]-sum2[0]+sum2[i+nperseg//2]-sum2[i+ngap//2])/(i-ngap//2+nperseg//2-ngap//2)
                       -((sum1[i-ngap//2]-sum1[0]+sum1[i+nperseg//2]-sum1[i+ngap//2])/(i-ngap//2+nperseg//2-ngap//2))**2)
                       *(i-ngap//2+nperseg//2-ngap//2)/(i-ngap//2+nperseg//2-ngap//2-1))
    for i in range(nperseg//2,len(signal)-nperseg//2):
        mov_var.append(((sum2[i+nperseg//2]-sum2[i+ngap//2]+sum2[i-ngap//2]-sum2[i-nperseg//2])/(nperseg-ngap)
                      -(((sum1[i+nperseg//2]-sum1[i+ngap//2]+sum1[i-ngap//2]-sum1[i-nperseg//2])/(nperseg-ngap))**2))
                       *(nperseg-ngap)/(nperseg-ngap-1))
    for i in np.arange(len(signal)-nperseg//2, len(signal)-ngap//2):
        mov_var.append(((sum2[i-ngap//2]-sum2[i-nperseg//2]+sum2[-1]-sum2[i+ngap//2])/(len(signal)-i-ngap//2+nperseg//2-ngap//2)
                     -((sum1[i-ngap//2]-sum1[i-nperseg//2]+sum1[-1]-sum1[i+ngap//2])/(len(signal)-i-ngap//2+nperseg//2-ngap//2))**2)
                       *(len(signal)-i-ngap//2+nperseg//2-ngap//2)/(len(signal)-i-ngap//2+nperseg//2-ngap//2-1))
    for i in np.arange(len(signal)-ngap//2, len(signal)):
        mov_var.append(((sum2[i-ngap//2]-sum2[i-nperseg//2])/(nperseg//2-ngap//2)
                       -((sum1[i-ngap//2]-sum1[i-nperseg//2])/(nperseg//2-ngap//2))**2)
                       *(nperseg//2-ngap//2)/(nperseg//2-ngap//2-1))
    return np.array(mov_var)


def hms2deg(hh,mm,ss):
    """
    to go from hours, minutes, seconds to degrees
    """
    return(hh*60*60+mm*60+ss)/(24*60*60)*360


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
        

def outlier_free_average(data, nsig=3, method='mean', mask=None, detrend=False, detr_len=128):
    if mask==None:
        mask = mask=~np.isnan(data)
    if detrend:
        data1 = detrend_amat(np.array(data), length=detr_len, mask=mask, method='median', endmethod='mean', ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False)
    else:
        data1 = np.array(data)
    outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig * np.nanstd(data1[mask])
    if method=='mean':
        return(np.nanmean(data[outlier_mask*mask]))
    elif method=='median':
        return(np.nanmedian(data[outlier_mask*mask]))
    else:
        raise AssertionError('Undefined method specified!')


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
    std of the data, floar

    """
    if mask==None:
        mask = mask=~np.isnan(data)
    if detrend:
        data1 = detrend_amat(np.array(data), length=detr_len, mask=mask, method='median', endmethod='mean', ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False)
    else:
        data1 = np.array(data)
    outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig * np.nanstd(data1[mask])
    return np.nanstd(data[outlier_mask*mask])


def mask_outliers_fun(data, mask=None, nsig=3., nsig_positive=None, nsig_negative=None,
                           detr_length=128, detr_method='median', detr_endmethod='mean', detr_ends='symmetric',
                       detr_endlen=None, detr_return_trend=False, detr_keep_overall_level=False):
    """
    Masks entries which deviate from the general trend more than nsig standard deviations.
    Detrending is made to find the general trend.
    ----------
    data : numpy array 1d of float
    mask : numpy.array 1d of bool
        True is good data, False is bad data. 
        If None, will be measured to mask nans
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
    detr_method : str, 'mean' or 'median'
        The type of moving average using to define the trend. The default is 'median',
        because it is less sensitive to outliers.
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
                                return_trend=detr_return_trend, keep_overall_level=detr_keep_overall_level)
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
                                return_trend=detr_return_trend, keep_overall_level=detr_keep_overall_level)
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


def integ_interp(x1, x0, y0):
    """
    Interpolates the signal (making it more coarse-grained) using integration.
    Each output point is the mean of the input array located at a distance less than
    half-bin of the coarse array.
    -----------------------
    x1 : np.array or list or float. Output arguments array
    If array or list, has to be equally spaced.
        If int, will be treated as step dx1 in the new array.
        Number of points will be chosen automatically to cover the input.
        Integration will be performed in such way that these points are 
        in the middle of integration domain
    x0 : np.array or list. Input arguments array
        Has to be equally spaced
    y0 : np.array or list. Input function values array
    """
    if type(x1)!=int:
        assert type(x1)==np.ndarray, 'Incorrect type of the new x (x1) array!'
        assert np.isclose(np.std((x1-np.roll(x1,-1))[:-1]),0), 'New array of arguments (x1) has to be equally spaced!'
    if type(x1)==np.ndarray or type(x1)==list:
        dx1 = x1[1] - x1[0]
        dx0 = x0[1] - x0[0]
        assert x0[0] <= x1[0] - dx1/2, 'Sample too short at the beginning!'
        assert x0[-1] >= x1[-1] + dx1/2, 'Sample too short at the end!'
        assert dx0 <= dx1, 'Data points not fine enough to integrate!'
    else: 
        dx1 = x1
        assert dx0 <= dx1, 'Data points not fine enough to integrate!'
        x1 = np.arange(x0[0] + dx1/2,x0[-1] - dx1/2, dx1)
    xint = np.append(x1, x1[-1] + dx1) - dx1/2 #array used for integration (shifted by half-bin and extended)
    y1 = np.zeros(len(x1))*np.nan
    for i in range(len(x1)):
        i1 = np.where(x0>xint[i])[0][0]  # ATTENTION: BOUNDARY POINTS WILL BE IGNORED 
        i2 = np.where(x0<xint[i+1])[0][-1]
        y1[i] = np.nanmean(y0[i1:i2+1])
    return x1, y1


def get_data(target_id, data_location, return_sap=False): #FOR ALL QUARTERS
    #TODO: add bad_fft verification
    """
    Search default directory for prev downloaded data. if not found, download and put there.
    Return the times, fluxes lists
    target_id - a Kepler id as a string (example - '757280')
    """
    target_id = str(target_id).rjust(9, "0")
    folder=os.path.join(data_location,target_id)
    
    if target_id not in os.listdir(data_location):
        try:
            os.mkdir(folder)
        except OSError:
                print ("Creation of the directory failed")
                
        webdir = 'https://archive.stsci.edu/pub/kepler/lightcurves/'
        # webdir = 'https://archive.stsci.edu/missions/kepler/lightcurves/'
        webfolder = webdir + target_id[:4] + '/' + target_id + '/'
        r = requests.get(webfolder)   
        lines=[line for line in (r.text).splitlines() if 'llc.fits' in line]
        urls = [webfolder + line[line.index('href="kplr'+target_id)+6 : line.index('.fits"')+5] for line in lines]
        if len(urls)!=18:
            print('warning: number of files is not equal to the number of all quarters')
            print('downloaded '+str(len(urls)) + ' files')
        for url in urls:
            wget.download(url, folder)
    
    times=[]
    fluxes=[]
    if return_sap:
        saps=[]
    files = [f for f in os.listdir(folder)[:] if 'llc.fits' in f]
    if len(files)!=18:
            print('warning: number of files is not equal to the number of all quarters')
            print('available '+str(len(files)) + ' files')
    for f in files:
        filc = fits.open(os.path.join(folder,f))
        times.append(filc[1].data['TIME'])
        fluxes.append(filc[1].data['PDCSAP_FLUX'])
        if return_sap:
            saps.append(filc[1].data['SAP_FLUX'])
        filc.close()
    
    firstindices = []
    for time in times:
        firstindices.append(np.argmax(~np.isnan(time))) #argmax is for finding the first instance of not nan
    i0 = np.max(firstindices) #find the biggest initial indes to call all times by it and make sure there is no nans
    # sort them by increasing time because the files in the directory are sorted at random:
    if return_sap:
        pairs = list(zip(times,fluxes,saps))
        pairs.sort(key=lambda p: p[0][i0])
        times, fluxes, saps = list(map(list,zip(*pairs)))
        return times, fluxes, saps
    else:    
        pairs = list(zip(times,fluxes))
        pairs.sort(key=lambda p: p[0][i0])
        times, fluxes = list(map(list,zip(*pairs)))
        return times, fluxes
    
    
def interpolate_small_holes(flux, mask, max_hole=5 , interp_method='linear'):
    if mask.dtype != np.dtype('bool'):
        mask = np.array(mask).astype(bool)
    ind1 = (np.where(mask)[0])[0] 
    ind2 = (np.where(mask)[0])[-1]
    mask1 = np.concatenate((np.ones(ind1),np.array(mask[ind1:ind2+1]), np.ones(len(mask)-ind2-1))).astype(bool)
    begholes = np.where(mask1*np.roll(~mask1,-1))[0]+1
    endholes = np.where(mask1*np.roll(~mask1,1))[0]
    lenholes = endholes-begholes
    big_holes = np.where(lenholes>max_hole)[0]
    for i in range(len(big_holes)):
        mask1[begholes[big_holes[i]]:endholes[big_holes[i]]]=True
    flux = interpolate_over_mask(np.array(flux), np.array(mask1), interp_method=interp_method, mitigate_hole_ends=False)
    mask_big = np.array(mask | (~mask1)) #mask with only big holes left
    return flux, mask_big


def inpaint(data, mask, wf_td):
    """
    To inpaint only one vector (in order not to wait too long if we just need a check)

    Parameters
    ----------
    data : np.array 1d float
        the flux.
    mask : np.array 1d bool
        ones where holes.
    wf_td : np.array 1d
        whitening template in the time domain.

    Returns
    -------
    np.array 1d
        inpainted template.

    """
    ## Here is the original code, below is its simplified (faster) version
    ## The code below runs for ~40s, the code above ~24s
    # Nd = len(mask) #length of the data
    # ind = np.where(~mask)[0]
    # data[ind]=0
    # A = []
    # for i in ind:
    #     z = np.zeros(Nd)
    #     z[i]=1
    #     A.append(z)
    # A = np.array(A).T # matrix of vectors with ones in holes,
    #                   # size Nd*Nh (vertical rectangle), where Nh is number of holes
    
    # Cinv = filter2cov_matrix(wf_td) #C^-1 made from the whitened filter
    # AtCinv = A.T @ Cinv
    # Minv = np.linalg.inv(AtCinv @ A)
    # F = A @ Minv @ AtCinv #NOTE THAT IN THE ORIGINAL NOTATIONS THERE IS ALSO 1-ALL THIS
    # return data - F @ data
    
    if np.isnan(data).any():
        data[np.where(np.isnan(data))[0]]=np.nanmean(data) #this does not matter, these values 
        # do not influence when multiplied by filter. this is just to avoid nans
    Cinv = filter2cov_matrix(wf_td) #C^-1 made from the whitened filter
    AtCinv = np.array(Cinv[~mask])
    Minv = np.linalg.inv(AtCinv[:,~mask])
    F = np.zeros((len(mask),len(mask)))
    F[~mask] = np.array(Minv @ AtCinv)
    return data - (F@data)


def filter2cov_matrix(vec, pad_by_zeros=False, trim_zeros=True):
    """
    make a covariance matrix in the time domain from the
    whitening filter in time domain(vec)
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
        the covariance matrix (for the time domain). 
        if vec is the time domain whitening filter, it will be the INVERSE covariance matriz

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


def pad_by_zeros(data, support=None, minimal_extension=0, return_zero_padding_ends=False):
    """
    Pads data by zeros from the two sides.
    
    Parameters
    -------------------
    data : array 1d
    support : int or None
        the new length of padded data. If None, The new length will be the closest power of 2 greater or equal to len(data)+minimal_extension
    minimal_extension : int
        If `support` is None, the new length of padded data is len(data)+minimal_extension
    return_zero_padding_ends : bool
        Whether to return the tuple of number of zeros padded from each side
    
    Returns
    -------------------
    If return_zero_padding_ends:
        data_padded, (zerlen1, zerlen2) 
    Else:
        data_padded
    """
    if support is None:
        # the closest (but bigger) power of 2 to the length  
        support = 2**int(np.ceil(np.log2(len(data)+minimal_extension))) 
        
    zerlen1 = (support-len(data))//2
    zerlen2 = (support-len(data)+1)//2
    data_padded = np.concatenate((np.zeros(zerlen1), data, np.zeros(zerlen2) ))
    if return_zero_padding_ends:
        return data_padded, (zerlen1, zerlen2)
    else:
        return data_padded
