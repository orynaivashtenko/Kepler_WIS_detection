#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import copy
import warnings

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import chi2, norm, binom, poisson
from scipy.special import erfc, erfinv, ndtr, gammaincc, gammainccinv
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, tf2sos, tf2zpk
from scipy.signal.windows import tukey

from search_pipeline_parameters import *
# from numba import float32
# -> I forgot why I did this, and there may be implications...

if Use_pyfftw:
    from pyfftw.interfaces.numpy_fft import rfft as rfft
    from pyfftw.interfaces.numpy_fft import irfft as irfft
    import pyfftw.interfaces.cache
    pyfftw.interfaces.cache.enable()
    pyfftw.forget_wisdom()
else:
    rfft = np.fft.rfft
    irfft = np.fft.irfft


def isgood(vec):
    return (~np.isnan(vec)) & np.isfinite(vec)


def standard_normal_pdf(x):
    """Is faster than scipy.norm.pdf"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def safe_nanargmin(x, axis=0):
    return safe_nanfun(x=x, axis=axis, fun='argmin')


def safe_nanargmax(x, axis=0):
    return safe_nanfun(x=x, axis=axis, fun='argmax')


def safe_nanmin(x, axis=0):
    return safe_nanfun(x=x, axis=axis, fun='min')


def safe_nanmax(x, axis=0):
    return safe_nanfun(x=x, axis=axis, fun='max')


def safe_nanfun(x, axis=0, fun='min'):
    if fun == 'min':
        fun = np.nanmin
        repl = np.nan
    elif fun == 'max':
        fun = np.nanmax
        repl = np.nan
    elif fun == 'argmin':
        fun = np.nanargmin
        repl = -1
    elif fun == 'argmax':
        fun = np.nanargmax
        repl = -1
    else:
        raise ValueError('Function `fun` not recognized!')
    if hasattr(x[0], '__len__'):
        x = np.array(x)
        inds_notnan = np.max((~np.isnan(x)).astype(float), axis=axis).astype(bool)
        res = np.array(x, dtype=float)
        res[np.isnan(res)] = np.inf
        res = fun(res, axis=axis)
        res[~inds_notnan] = repl
    else:
        if np.sum(~np.isnan(x)) == 0:
            res = repl
        else:
            res = fun(x)
    return res


def nanargsort(array):
    array = np.array(array)
    notnan_mask = ~np.isnan(array)
    argsort = np.argsort(array[notnan_mask])
    return np.arange(len(array))[notnan_mask][argsort]


def nansort(array):
    array = np.array(array)
    notnan_mask = ~np.isnan(array)
    return np.sort(array[notnan_mask])


def sliding_nanargmin(array, window_size):
    return sliding_nanargfun(array=array, window_size=window_size, fun='argmin')


def sliding_nanargmax(array, window_size):
    return sliding_nanargfun(array=array, window_size=window_size, fun='argmax')


def sliding_nanargfun(array, window_size, fun='nanargmin'):
    """
    Sliding window nanargmin. Is faster than a for loop for some window sizes. Uses a 2D array construction
    :param array: 1D numpy array
    :param window_size: size of the sliding window
    :param fun: function to apply to the sliding windows, can be 'min', 'max', 'argmin', 'argmax'
    :return: indices of the minimums in the sliding windows
    """
    if fun == 'argmin':
        fun = np.nanargmin
        repl = -1
    elif fun == 'argmax':
        fun = np.nanargmax
        repl = -1
    else:
        raise ValueError('Function `fun` not recognized!')

    window_size = int(window_size)
    stacked_array = np.full((window_size, len(array)), np.nan)
    shifts = np.arange(-((window_size+1) // 2), window_size // 2, 1, dtype=np.int64) + 1
    for i, shift in enumerate(shifts):
        ind_0 = max(0, shift)
        ind_1 = min(len(array), len(array) + shift)
        stacked_array[window_size-i-1, ind_0:ind_1] = np.roll(array, shift)[ind_0:ind_1]

    not_nan_mask = np.sum((np.isnan(stacked_array)).astype(int), axis=0) != window_size
    res_array = np.full(len(array), repl).astype(int)
    res_array[not_nan_mask] = fun(stacked_array[:, not_nan_mask], axis=0)
    res_array += np.arange(len(res_array)) - (window_size // 2)
    return res_array


def safe_div(numerators, denominators, fill_value=0.):
    numerators = np.array(numerators)
    denominators = np.array(denominators)

    non_zero_mask = ~np.isclose(denominators, 0)
    numerators[non_zero_mask] = numerators[non_zero_mask] / denominators[non_zero_mask]
    numerators[~non_zero_mask] = fill_value
    return numerators


def dec_to_bin(num, return_list=False, return_str=False):
    """
    From decimal to binary
    :param num:
    :type num:
    :param return_list:
    :type return_list:
    :param return_str:
    :type return_str:
    :return:
    :rtype:
    """
    if not hasattr(num, '__len__'):
        num = int(num)
        if num == 0:
            bits = [0]
        else:
            max_ord = int(np.log2(num))
            bits = []
            for i in range(max_ord + 1):
                bits.append(num % 2)
                num = num // 2
        if return_list:
            return bits  # bits[0] is 2**0
        elif return_str:
            return ''.join([str(b) for b in bits[::-1]])
        else:
            return int(''.join([str(b) for b in bits[::-1]]))
    else:
        return dec_to_bin_vec(num=num, return_list=return_list, return_str=return_str)


def dec_to_bin_vec(num, return_list=True, return_str=False):
    """
    From decimal to binary for vectors
    :param num:
    :type num:
    :param return_list:
    :type return_list:
    :param return_str:
    :type return_str:
    :return:
    :rtype:
    """
    if not hasattr(num, '__len__'):
        num = np.array([num])
    num = num.astype(int)
    if np.max(num) > 0:
        max_ord = int(np.log2(np.max(num)))
        bits = np.zeros((max_ord + 1, len(num)), dtype=int)
        for i in range(max_ord + 1):
            bits[i] = (num % 2)
            num = num // 2
        bits = bits
    else:
        return np.zeros(len(num))

    if return_list:
        return bits  # bits[0] is 2**0
    else:
        str_bits = [''.join(list((b.astype(int)).astype(str))) for b in (bits[::-1]).T]
        if return_str:
            return str_bits
        else:
            return np.array(str_bits).astype(int)


def get_specific_bit_from_dec(x_dec, bit):
    x_res = x_dec % (2**(bit+1))
    x_res = x_res // (2**bit)
    return x_res


def get_unique_indices(data_list):
    data_list_unique = []
    inds_unique = []
    for i_dat, dat in enumerate(data_list):
        if dat not in data_list_unique:
            data_list_unique.append(dat)
            inds_unique.append(i_dat)
    return inds_unique, data_list_unique


def get_uniform_random_sequence(low, high, n):
    res = np.linspace(low, high, n, endpoint=False)
    if n > 1:
        dx = res[1] - res[0]
    else:
        dx = high - low
    res = res + np.random.uniform(low=0, high=dx, size=n)
    return res


def cos_vec(h1, h2, wf_fd=None, freqmask=None):
    """
    Calculate cosine angle between two vectors h1, h2, which are given in the time domain
    :param h1: First non-whitened vector in the time domain
    :type h1: ndarray(float)
    :param h2: Second non-whitened vector in the time domain, same length as `h1`
    :type h2: ndarray(float)
    :param wf_fd: whitening filter in the Fourier domain
    :type wf_fd: ndarray(float) of length consistent with rfft of `h1` and `h2`
    :param freqmask: Frequency-domain mask of which frequencies should participate in the product
    :type freqmask: ndarray(bool) of length consistent with rfft of `h1` and `h2`
    :return:
    :rtype: float
    """
    if freqmask is None:
        freqmask = np.ones(len(h1) // 2 + 1)  # .astype(float)
    else:
        freqmask = freqmask.astype(float)
    if wf_fd is None:
        wf_fd = np.ones(len(freqmask))
    # Warning! freq mask is not participating in the normalization!
    norm_1 = np.sum(np.abs(irfft(rfft(h1, norm='ortho') * wf_fd * freqmask, norm='ortho', n=len(h1))) ** 2)
    norm_2 = np.sum(np.abs(irfft(rfft(h2, norm='ortho') * wf_fd * freqmask, norm='ortho', n=len(h2))) ** 2)
    return (np.sum(irfft(rfft(h1, norm='ortho') * wf_fd * freqmask, norm='ortho', n=len(h1))
                   * irfft(rfft(h2, norm='ortho') * wf_fd * freqmask, norm='ortho', n=len(h2)))
            / (norm_1 ** 0.5 * norm_2 ** 0.5))


def moving_mean(data, nwind, ngap=0, mask=None, nsamples_min=32,
                remove_outliers=False, nsig_outliers=6,
                return_mask=False, return_total_mean=False,
                replace_ends=False, endmethod='mean', replace_len=None, endlen=None):
    """
    Function which is faster than detrend_moving_ave, but less sophisticated
    Calculates the moving mean of the signal. At every point, takes the window of length
    `nwind` symmetrically from the two sides, excludes from the middle the window of length
    `ngap` and calculates the mean or median of the remaining. It is needed to
    avoid the impact of the entries correlated with the considered point to the
    variance estimation. For example, if there is a transit in this point, the
    variance is taken around the transit, so its presence does not alter the measurement.
    At the ends of the signal, the data is taken only from one side of the
    considered point up to the available length.
    :param data: The array from which the moving variance should be calculated.
    :type data: numpy.ndarray of float, NO NANS
    :param nwind: Size of the moving window
    :type nwind: int
    :param ngap: Size of the gap inside the moving window to exclude the point around which the variance is taken.
        Points are excluded around each bin when calculating the variance in this bin in order to avoid influence of these
        points on the variance estimation.
    :type ngap: int, should be <`nwind`
    :param mask: Array with `True` where `signal` is food for processing and `False` where it is bad
    :type mask: None or numpy.ndarray (of bool)
    :param nsamples_min: Minimal number of samples in the window to take the variance
    :type nsamples_min: int
    :param remove_outliers: Whether to remove outliers before calculating the moving variance
    :type remove_outliers: bool
    :param nsig_outliers: Threshold in units of sigma if `remove_outliers` is True
    :type nsig_outliers: float
    :param return_mask: whether to return the mask
    :type return_mask: bool
    :param return_total_mean: whether to return the total scalar variance of `signal`
    :type return_total_mean: bool
    :param replace_ends:
    :type replace_ends:
    :param replace_len:
    :type replace_len:
    :param endmethod:
    :type endmethod:
    :param endlen:
    :type endlen:
    :return: mov_var Moving variance estimate at each point of the signal
    :rtype: numpy.ndarray 1d of float, len(`signal`)
    """
    data = np.array(data)

    if mask is None:
        mask = isgood(data)
    else:
        mask = np.array(mask).astype(bool)
        mask &= isgood(data)
    mask &= np.isfinite(data)

    if remove_outliers:
        data_minus_mov_ave = data - moving_mean(data=data, nwind=nwind, ngap=ngap, mask=mask, nsamples_min=nsamples_min,
                                                remove_outliers=False, nsig_outliers=nsig_outliers, return_mask=False,
                                                return_total_mean=False)
        outlier_mask = np.abs(data_minus_mov_ave) < nsig_outliers * np.nanstd(data_minus_mov_ave[mask])
        mask &= outlier_mask

    tot_mean = np.nanmean(data[mask])
    data[~mask] = 0.
    sum1 = np.nancumsum(data * mask.astype(float))
    sum1 = np.concatenate(([0], sum1[:-1]))
    counts = np.nancumsum(mask)
    counts = np.concatenate(([0], counts[:-1]))

    hnw = nwind // 2
    hng = ngap // 2
    vecs = [sum1, counts]
    for i in range(len(vecs)):
        vec = vecs[i]
        temp_vec_w = np.roll(vec, -hnw - 1)
        temp_vec_w[len(temp_vec_w) - hnw - 1:] = vec[-1]
        temp_vec_g = np.roll(vec, -hng - 1)
        temp_vec_g[len(temp_vec_g) - hng - 1:] = vec[-1]
        right_vec = temp_vec_w - temp_vec_g

        temp_vec_w = np.roll(vec, hnw)
        temp_vec_w[:hnw] = vec[0]
        temp_vec_g = np.roll(vec, hng)
        temp_vec_g[:hng] = vec[0]
        left_vec = temp_vec_g - temp_vec_w
        vecs[i] = right_vec + left_vec
    sum1, counts = vecs
    mask_counts = (counts >= max(nsamples_min, 2))
    counts = counts.astype(float)  # for the next row
    counts[~mask_counts] = np.nan  # only for division, to avoid dividing by zero is there is some
    mov_mean = sum1 / counts
    mov_mean[~mask_counts] = tot_mean

    if replace_ends:
        if replace_len is None:
            len_repl = int(np.ceil(nwind // 2))
        else:
            len_repl = replace_len
        len_calc = 2 * len_repl
        if endmethod == 'mean':
            if endlen is None:
                endlen = -1
            endlen = int(endlen)
            mov_mean[:len_repl] = moving_mean_edge_fast(data[:len_calc],
                                                        left_mask_int=np.array(mask[:len_calc], dtype=int),
                                                        endlen=endlen)[:len_repl]
            mov_mean[-len_repl:] = moving_mean_edge_fast(data[-len_calc:][::-1],
                                                         left_mask_int=np.array(mask[-len_calc:][::-1], dtype=int),
                                                         endlen=endlen)[:len_repl][::-1]
        else:
            mov_mean[:len_repl] = moving_average_edge(data[:len_calc], left_mask=mask[:len_calc],
                                                      endmethod=endmethod, ends='symmetric', endlen=endlen)[:len_repl]
            mov_mean[-len_repl:] = moving_average_edge(data[-len_calc:][::-1], left_mask=mask[-len_calc:][::-1],
                                                       endmethod=endmethod, ends='symmetric', endlen=endlen)[:len_repl][::-1]

    if not return_total_mean:
        if not return_mask:
            return mov_mean
        else:
            return mov_mean, mask
    else:
        if not return_mask:
            return mov_mean, tot_mean
        else:
            return mov_mean, mask, tot_mean


@njit(parallel=False, fastmath=False)
def moving_mean_edge_fast(left_data, left_mask_int, endlen=-1):
    """
    Substitute for the ends of calculating moving average. If this is applied throughout, it will be too slow.
    :param left_data: Data sample from which the moving average has to be subtracted
    :type left_data: np.array 1d of float. no nans!
    :param left_mask_int:
    :type left_mask_int: numpy.ndarray of int

    :param endlen: Minimal length to take from the end for determining the average in the
        first (last) point. If None, will be assigned depending on the method (see description of ends)
    :type endlen: int: if -1, will be measured
    :return: if not return_trend: left_data with subtracted mean, np.array
             if return_trend: left_data with subtracted mean, np.array; trend (moving mean), np.array
    :rtype:
    """
    ldat = len(left_data)
    trend = np.full(ldat, np.nan)

    if endlen == -1:
        endlen = int(np.ceil(ldat / 4))  # use ceil to avoid zero length
    if endlen > int(ldat / 2):
        endlen = int(ldat / 2)
    if endlen == 0:
        endlen = 1

    for i in range(0, endlen):
        i1 = i + endlen
        i1 = min(i1, ldat)
        summed = 0.
        counts = 0
        for j in range(i1):
            summed += left_data[j] * left_mask_int[j]
            counts += left_mask_int[j]
        if counts > 0:
            m = summed / counts
        else:
            m = np.mean(left_data[:i1])
        trend[i] = m
    for i in range(endlen, ldat):
        i1 = min(2 * i, ldat)
        summed = 0.
        counts = 0
        for j in range(i1):
            summed += left_data[j] * left_mask_int[j]
            counts += left_mask_int[j]
        if counts > 0:
            m = summed / counts
        else:
            m = np.mean(left_data[:i1])
        trend[i] = m
    return trend


# _ = moving_mean_edge_fast(left_data=np.arange(10, dtype=np.float64),
#                           left_mask_int=np.arange(10, dtype=int), endlen=-1)
# _ = moving_mean_edge_fast(left_data=np.arange(10, dtype=np.float32),
#                           left_mask_int=np.arange(10, dtype=int), endlen=-1)

# We initialize the full thing instead of just the edge thing because for some reason, it does not work as initialization then
_ = moving_mean(data=np.arange(20, dtype=np.float64), nwind=4, ngap=0, mask=None, nsamples_min=1,
                remove_outliers=False, nsig_outliers=6,
                return_mask=False, return_total_mean=False,
                replace_ends=True, endmethod='mean', replace_len=4, endlen=None)

_ = moving_mean(data=np.arange(20, dtype=np.float32), nwind=4, ngap=0, mask=None, nsamples_min=1,
                remove_outliers=False, nsig_outliers=6,
                return_mask=False, return_total_mean=False,
                replace_ends=True, endmethod='mean', replace_len=4, endlen=None)


def moving_average_edge(left_data, left_mask=None, endmethod='mean', ends='symmetric', endlen=None):
    """
    Substitute for the ends of calculating moving average. If this is applied throughout, it will be too slow.
    :param left_data: Data sample from which the moving average has to be subtracted
    :type left_data: np.array 1d or list
    :param left_mask:
    :type left_mask:
    :param endmethod: Kind of moving value to subtract at the ends of the left_data (see below): 'median' or 'mean'.
                    The default is 'mean', because it is less sensitive to small
                    amount of left_data samples available in the ends.
    :type endmethod: str
    :param ends: How to treat the points closer than length/2 to the ends of the left_data:
                'symmetric' :  endlen is length/4
                               1) if point<endlen, take endlen samples from the right, and up to zero from the left
                               2) if endlen<point<half-length, take the same number of points from left and right
                               3) if point>half-length, take usual half-length from both sides
                               for the second end, the same in opposite direction
                'asymmetric' : endlen is length/16
                               1) if point<half-length, take up to zero from the left, and the length to the
                                    right linearly increases from endlen for point=0 to length/2 at point=length/2
                               2) if point>half-length, take usual half-length from both sides
                               for the second end, the same in opposite direction
    :type ends: str
    :param endlen: Minimal length to take from the end for determining the average in the
        first (last) point. If None, will be assigned depending on the method (see description of ends)
    :type endlen: int or None
    :return: if not return_trend: left_data with subtracted mean, np.array
             if return_trend: left_data with subtracted mean, np.array; trend (moving mean), np.array
    :rtype:
    """
    assert ((endmethod == 'mean') or (endmethod == 'median')), 'Incorrect method specified'
    left_data = np.array(left_data)

    if left_mask is None:
        left_mask = isgood(left_data)
    else:
        left_mask = np.array(left_mask) & isgood(left_data)

    assert len(left_data) == len(left_mask), 'Data and left_mask must be of the same length!'
    ldat = len(left_data)

    trend = np.ones(len(left_data)) * np.nan
    if ends == 'asymmetric':
        f1 = (ldat - endlen) / ldat
        if endlen is None:
            endlen = int(np.ceil(ldat / 16))
        if endlen > ldat:
            endlen = ldat
        if endmethod == 'median':
            for i in np.arange(0, ldat):
                i1 = i + int(endlen + f1 * i)
                if (left_mask[0:i1]).any():
                    m = np.median((left_data[0:i1])[left_mask[0:i1]])
                else:
                    m = np.median(left_data[0:i1])
                trend[i] = m
        elif endmethod == 'mean':
            for i in np.arange(0, ldat):
                i1 = i + int(endlen + f1 * i)
                if (left_mask[0:i1]).any():
                    m = np.mean((left_data[0:i1])[left_mask[0:i1]])
                else:
                    m = np.mean(left_data[0:i1])
                trend[i] = m

    elif ends == 'symmetric':
        if endlen is None:
            endlen = int(np.ceil(ldat / 4))
        if endlen > ldat:
            endlen = ldat
        if endmethod == 'median':
            for i in np.arange(0, endlen):
                i1 = i + endlen
                if (left_mask[0:i1]).any():
                    m = np.median((left_data[0:i1])[left_mask[0:i1]])
                else:
                    m = np.median(left_data[0:i1])
                trend[i] = m
            for i in np.arange(endlen, ldat):
                i1 = 2 * i
                if (left_mask[0:i1]).any():
                    m = np.median((left_data[0:i1])[left_mask[0:i1]])
                else:
                    m = np.median(left_data[0:i1])
                trend[i] = m
        elif endmethod == 'mean':
            for i in np.arange(0, endlen):
                i1 = i + endlen
                if (left_mask[0:i1]).any():
                    m = np.mean((left_data[0:i1])[left_mask[0:i1]])
                else:
                    m = np.mean(left_data[0:i1])
                trend[i] = m
            for i in np.arange(endlen, ldat):
                i1 = 2 * i
                if (left_mask[0:i1]).any():
                    m = np.mean((left_data[0:i1])[left_mask[0:i1]])
                else:
                    m = np.mean(left_data[0:i1])
                trend[i] = m
    else:
        raise AssertionError('Incorrect way to treat ends specified!')
    return trend


def left_right_moving_mean_and_var(data, nwind, ngap=0, mask=None, nsamples_min=None,
                                   remove_outliers=False, nsig_outliers=6,
                                   return_mask=False, return_total_mean_var=False):
    """
    Function which is faster than detrend_moving_ave, but less sophisticated
    Calculates the moving mean of the signal. At every point, takes the window of length
    `nwind` symmetrically from the two sides, excludes from the middle the window of length
    `ngap` and calculates the mean or median of the remaining. It is needed to
    avoid the impact of the entries correlated with the considered point to the
    variance estimation. For example, if there is a transit in this point, the
    variance is taken around the transit, so its presence does not alter the measurement.
    At the ends of the data, the data is taken only from one side of the
    considered point up to the available length.
    :param data: The array from which the moving variance should be calculated.
    :type data: numpy.ndarray of float, NO NANS
    :param nwind: Size of the moving window
    :type nwind: int
    :param ngap: Size of the gap inside the moving window to exclude the point around which the variance is taken.
        Points are excluded around each bin when calculating the variance in this bin in order to avoid influence of these
        points on the variance estimation.
    :type ngap: int, should be <`nwind`
    :param mask: Array with `True` where `signal` is food for processing and `False` where it is bad
    :type mask: None or numpy.ndarray (of bool)
    :param nsamples_min: Minimal number of samples in the window to take the variance
    :type nsamples_min: int
    :param remove_outliers: Whether to remove outliers before calculating the moving variance
    :type remove_outliers: bool
    :param nsig_outliers: Threshold in units of sigma if `remove_outliers` is True
    :type nsig_outliers: float
    :param return_mask: whether to return the mask
    :type return_mask: bool
    :param return_total_mean_var: whether to return the total scalar variance of `signal`
    :type return_total_mean_var: bool
    :return: mov_var Moving variance estimate at each point of the signal
    :rtype: numpy.ndarray 1d of float, len(`signal`)
    """
    if nsamples_min is None:
        nsamples_min = nwind // 2

    data = np.array(data)

    if mask is None:
        mask = isgood(data)
    else:
        mask = np.array(mask).astype(bool)
        mask &= isgood(data)
        assert len(data) == len(mask), 'Data and mask have to be of the same length!'

    if remove_outliers:
        data_minus_mov_ave = data - moving_mean(data=data, nwind=nwind, ngap=ngap, mask=mask, nsamples_min=nsamples_min,
                                                remove_outliers=False, nsig_outliers=nsig_outliers, return_mask=False,
                                                return_total_mean=False)
        outlier_mask = np.abs(data_minus_mov_ave) < nsig_outliers * np.nanstd(data_minus_mov_ave[mask])
        mask &= outlier_mask
        expected_power_part = get_expected_power_after_outlier_removal(nsig_outliers=nsig_outliers)
    else:
        expected_power_part = 1.

    tot_mean = np.nanmean(data[mask])
    tot_var = unb_var(data[mask])
    data[~mask] = 0.
    sum_left = np.nancumsum(data * mask.astype(float))
    sum_left = np.concatenate(([0], sum_left[:-1]))
    sum_right = np.nancumsum(data[::-1] * mask[::-1].astype(float))
    sum_right = np.concatenate(([0], sum_right[:-1]))
    counts_left = np.nancumsum(mask)
    counts_left = np.concatenate(([0], counts_left[:-1]))
    counts_right = np.nancumsum(mask[::-1])
    counts_right = np.concatenate(([0], counts_right[:-1]))
    sum2_left = np.nancumsum((data * mask.astype(float)) ** 2)
    sum2_left = np.concatenate(([0], sum2_left[:-1]))
    sum2_right = np.nancumsum((data[::-1] * mask[::-1].astype(float)) ** 2)
    sum2_right = np.concatenate(([0], sum2_right[:-1]))

    # Left and right processing while right is reversed
    vecs = [sum_left, counts_left, sum2_left, sum_right, counts_right, sum2_right]
    for i in range(len(vecs)):
        vec = vecs[i]
        temp_vec_w = np.roll(vec, nwind)
        temp_vec_w[:nwind] = vec[0]  # fills for rolling over
        temp_vec_g = np.roll(vec, ngap)
        temp_vec_g[:ngap] = vec[0]
        left_vec = temp_vec_g - temp_vec_w
        # See another moving something function if you want to use both left and right sides
        vecs[i] = left_vec

    means_left_right = []
    vars_left_right = []
    for inds in [[0, 1, 2], [3, 4, 5]]:
        sum_side, counts_side, sum2_side = vecs[inds[0]], vecs[inds[1]], vecs[inds[2]]
        mask_counts_side = (counts_side >= max(nsamples_min, 2)) | (np.arange(len(counts_side)) <= nwind)
        # -> but for edges, will have a different thing
        counts_side = counts_side.astype(float)  # for the next row
        counts_side[~mask_counts_side] = np.nan  # only for division, to avoid dividing by zero is there is some
        mov_mean_side = sum_side / counts_side
        mov_mean_side[~mask_counts_side] = tot_mean
        if nsamples_min > 2:
            mov_mean_side[:nsamples_min] = mov_mean_side[nsamples_min]
        means_left_right.append(mov_mean_side)

        mov_var_side = (sum2_side / counts_side - (sum_side / counts_side) ** 2) * counts_side / (counts_side - 1)
        mov_var_side = mov_var_side / expected_power_part
        mov_var_side[~mask_counts_side] = tot_var
        if nsamples_min > 2:
            mov_var_side[:nsamples_min] = mov_var_side[nsamples_min]
        vars_left_right.append(mov_var_side)

    means_left_right[1] = means_left_right[1][::-1]
    vars_left_right[1] = vars_left_right[1][::-1]

    if not return_total_mean_var:
        if not return_mask:
            return means_left_right[0], means_left_right[1], vars_left_right[0], vars_left_right[1]
        else:
            return means_left_right[0], means_left_right[1], vars_left_right[0], vars_left_right[1], mask
    else:
        if not return_mask:
            return means_left_right[0], means_left_right[1], vars_left_right[0], vars_left_right[1], tot_mean, tot_var
        else:
            return means_left_right[0], means_left_right[1], vars_left_right[0], vars_left_right[1], mask, tot_mean, tot_var


def unb_var(arr):
    """
    Calculates the unbiased variance estimate (numpy.var is biased!).
    Is related to the biased estimation as
    var_u = N/(N-1) * var_b,
    where N is the size of the sample.
    :param arr: Sample of numbers to calculate their variance
    :type arr: numpy.ndarray 1d or list of float, may contain numpy.nan.
    :return: Unbiased variance estimate of arr
    :rtype: float
    """
    arr = np.array(arr[isgood(arr)])
    return 1 / (len(arr) - 1) * np.sum(np.abs(arr - np.mean(arr)) ** 2)


def renormalize_variance_fun(movvar_meas, statmask, stat_snr, stat, nsig_outliers=None):
    # RE-NORMALIZING THE VARIANCE
    # The vetted triggers could have changed the true variance, now we want the cleaned stat to have
    # unit variance. Also, positive can have bad distr which I do not want to impact the negative
    # So we just repeat what was done before for moving variance correctness, but now after we updated masks with veto
    if nsig_outliers is not None:
        expected_power = get_expected_power_after_outlier_removal(nsig_outliers=nsig_outliers)
    else:
        expected_power = 1.
    # Making positive values unit variance:
    indices_sign = statmask & (stat > 0)
    var_snr_indices = np.nanmean((stat_snr[indices_sign]) ** 2)
    if nsig_outliers is not None:
        msk_outliers = (stat_snr[indices_sign]) ** 2 <= var_snr_indices * nsig_outliers ** 2
        var_snr_indices = np.nanmean((stat_snr[indices_sign][msk_outliers]) ** 2)
        var_snr_indices = var_snr_indices / expected_power
    movvar_meas[indices_sign] *= var_snr_indices
    # Making negative values unit variance:
    indices_sign = statmask & (stat < 0)
    var_snr_indices = np.nanmean((stat_snr[indices_sign]) ** 2)
    if nsig_outliers is not None:
        msk_outliers = (stat_snr[indices_sign]) ** 2 <= var_snr_indices * nsig_outliers ** 2
        var_snr_indices = np.nanmean((stat_snr[indices_sign][msk_outliers]) ** 2)
        var_snr_indices = var_snr_indices / expected_power
    movvar_meas[indices_sign] *= var_snr_indices
    # Use this new moving variance for all the rest:
    stat_snr[statmask] = stat[statmask] / (movvar_meas[statmask]) ** 0.5  # now should be N(0,1) for sure
    return stat_snr, movvar_meas


def moving_std_naive(data, window_len, mask=None, gap_len=0):
    """NAIVE AND SLOW. USE IT ONLY FOR DEBUGGING OF MOVING_VAR"""
    if mask is None:
        mask = isgood(data)
    d = window_len // 2
    if gap_len == 0:
        return [unb_var(data[max(0, i - d): min(i + d + 1, len(data))][
                               mask[max(0, i - d): min(i + d + 1, len(data))]]) ** 0.5
                if len(np.where(mask[max(0, i - d): min(i + d + 1, len(data))])[0]) > 1 else np.inf for i in range(len(data))]
    g = gap_len // 2

    return [unb_var(np.concatenate((data[max(0, i - d): max(0, i - g)], data[min(i + g + 1, len(data)): min(i + d + 1, len(data))]))[
                           np.concatenate(
                               (mask[max(0, i - d): max(0, i - g)], mask[min(i + g + 1, len(data)): min(i + d + 1, len(data))]))]) ** 0.5
            if len(
        np.where(np.concatenate((mask[max(0, i - d): max(0, i - g)], mask[min(i + g + 1, len(data)): min(i + d + 1, len(data))])))[0]) > 1
            else np.inf for i in range(len(data))]


def moving_var_maybe_faster(signal, nwind, ngap=0, mask=None, nsamples_min=32,
                            remove_outliers=False, nsig_outliers=6,
                            detrend_remove_outliers=False, detr_length_outliers=512,
                            return_mask=False, return_total_var=False):
    """
    Calculates the moving variance of the signal. At every point, takes the window of length
    `nwind` symmetrically from the two sides, excludes from the middle the window of length
    `ngap` and calculates the unbiased variance of the remaining. It is needed to
    avoid the impact of the entries correlated with the considered point to the
    variance estimation. For example, if there is a transit in this point, the
    variance is taken around the transit, so its presence does not alter the measurement.
    At the ends of the singal, the data is taken only from one side of the
    considered point up to the available length.
    :param signal: The array from which the moving variance should be calculated.
    :type signal: numpy.ndarray of float, NO NANS
    :param nwind: Size of the moving window
    :type nwind: int
    :param ngap: Size of the gap inside the moving window to exclude the point around which the variance is taken.
        Points are excluded around each bin when calculating the variance in this bin in order to avoid influence of these
        points on the variance estimation.
    :type ngap: int, should be <`nwind`
    :param mask: Array with `True` where `signal` is food for processing and `False` where it is bad
    :type mask: None or numpy.ndarray (of bool)
    :param nsamples_min: Minimal number of samples in the window to take the variance
    :type nsamples_min: int
    :param remove_outliers: Whether to remove outliers before calculating the moving variance
    :type remove_outliers: bool
    :param nsig_outliers: Threshold in units of sigma if `remove_outliers` is True
    :type nsig_outliers: float
    :param detrend_remove_outliers: Whether to detrend the signal before removing the outliers from it
    :type detrend_remove_outliers: bool
    :param detr_length_outliers: Detrending length to use if `detrend_remove_outliers` is true
    :type detr_length_outliers: int
    :param return_mask: whether to return the mask
    :type return_mask: bool
    :param return_total_var: whether to return the total scalar variance of `signal`
    :type return_total_var: bool
    :return: mov_var Moving variance estimate at each point of the signal
    :rtype: numpy.ndarray 1d of float, len(`signal`)
    """
    if mask is None:
        mask = isgood(signal)
    else:
        mask = np.array(mask).astype(bool)
        assert len(signal) == len(mask), 'Data and mask have to be of the same length!'
    signal_masked = signal * mask.astype(float)

    if remove_outliers:
        tot_var, mask = outlier_free_std(data=np.array(signal), nsig=nsig_outliers, n_iterations=2, mask=mask,
                                         detrend=detrend_remove_outliers, detr_len=detr_length_outliers,
                                         renormalize=True, return_var=True, return_mask=True)
        # When outliers are removed, the variance is expected to be smaller in any case, therefore we will use
        # the renormalization to make in unbiased. In the `outlier_free_std` function, it was already done.
        expected_power_part = get_expected_power_after_outlier_removal(nsig_outliers=nsig_outliers)
    else:
        tot_var = unb_var(signal[mask])
        expected_power_part = 1.

    signal[~mask] = 0.
    sum1 = np.nancumsum(np.insert(signal_masked, 0, 0.))[:-1]
    sum2 = np.nancumsum(np.insert(signal_masked ** 2, 0, 0.))[:-1]
    counts = np.nancumsum(np.insert(mask.astype(float), 0, 0.))[:-1]

    n = len(signal)
    hnw = nwind // 2
    hng = ngap // 2

    left = np.maximum(0, np.arange(n) - hnw)
    midl = np.maximum(0, np.arange(n) - hng)
    midr = np.minimum(n - 1, np.arange(n) + hng + 1)
    righ = np.minimum(n - 1, np.arange(n) + hnw + 1)

    sum1 = sum1[midl] - sum1[left] + sum1[righ] - sum1[midr]
    sum2 = sum2[midl] - sum2[left] + sum2[righ] - sum2[midr]
    counts = counts[midl] - counts[left] + counts[righ] - counts[midr]

    mask_counts = (counts >= max(nsamples_min, 2))
    counts = counts.astype(float)  # for the next row
    counts[~mask_counts] = np.nan  # only for division, to avoid dividing by zero is there is some
    mov_var = (sum2 / counts - (sum1 / counts) ** 2) * counts / (counts - 1)
    mov_var = mov_var / expected_power_part
    mov_var[~mask_counts] = tot_var
    if not return_total_var:
        if not return_mask:
            return mov_var
        else:
            return mov_var, mask
    else:
        if not return_mask:
            return mov_var, tot_var
        else:
            return mov_var, mask, tot_var


def moving_var(signal, nwind, ngap=0, mask=None, nsamples_min=32,
               remove_outliers=False, nsig_outliers=6,
               detrend_remove_outliers=False, detr_length_outliers=512,
               return_mask=False, return_total_var=False):
    """
    Calculates the moving variance of the signal. At every point, takes the window of length
    `nwind` symmetrically from the two sides, excludes from the middle the window of length
    `ngap` and calculates the unbiased variance of the remaining. It is needed to
    avoid the impact of the entries correlated with the considered point to the
    variance estimation. For example, if there is a transit in this point, the
    variance is taken around the transit, so its presence does not alter the measurement.
    At the ends of the singal, the data is taken only from one side of the
    considered point up to the available length.
    :param signal: The array from which the moving variance should be calculated.
    :type signal: numpy.ndarray of float, NO NANS
    :param nwind: Size of the moving window
    :type nwind: int
    :param ngap: Size of the gap inside the moving window to exclude the point around which the variance is taken.
        Points are excluded around each bin when calculating the variance in this bin in order to avoid influence of these
        points on the variance estimation.
    :type ngap: int, should be <`nwind`
    :param mask: Array with `True` where `signal` is food for processing and `False` where it is bad
    :type mask: None or numpy.ndarray (of bool)
    :param nsamples_min: Minimal number of samples in the window to take the variance
    :type nsamples_min: int
    :param remove_outliers: Whether to remove outliers before calculating the moving variance
    :type remove_outliers: bool
    :param nsig_outliers: Threshold in units of sigma if `remove_outliers` is True
    :type nsig_outliers: float
    :param detrend_remove_outliers: Whether to detrend the signal before removing the outliers from it
    :type detrend_remove_outliers: bool
    :param detr_length_outliers: Detrending length to use if `detrend_remove_outliers` is true
    :type detr_length_outliers: int
    :param return_mask: whether to return the mask
    :type return_mask: bool
    :param return_total_var: whether to return the total scalar variance of `signal`
    :type return_total_var: bool
    :return: mov_var Moving variance estimate at each point of the signal
    :rtype: numpy.ndarray 1d of float, len(`signal`)
    """
    signal = np.array(signal)

    if mask is None:
        mask = isgood(signal)
    else:
        mask = np.array(mask).astype(bool)
        mask &= isgood(signal)
        assert len(signal) == len(mask), 'Data and mask have to be of the same length!'

    if remove_outliers:
        # if detrend_remove_outliers:
        #     data1 = detrend_mov_ave(np.array(signal), length=detr_length_outliers, mask=mask,
        #                             method='median', endmethod='mean', ends='symmetric', endlen=None,
        #                             return_trend=False, keep_overall_level=False, remove_outliers=True,
        #                             nsig_outliers=nsig_outliers)
        # else:
        #     data1 = signal

        tot_var, mask = outlier_free_std(data=np.array(signal), nsig=nsig_outliers, n_iterations=2, mask=mask,
                                         detrend=detrend_remove_outliers, detr_len=detr_length_outliers,
                                         renormalize=True, return_var=True, return_mask=True)
        # When outliers are removed, the variance is expected to be smaller in any case, therefore we will use
        # the renormalization to make in unbiased. In the `outlier_free_std` function, it was already done.
        expected_power_part = get_expected_power_after_outlier_removal(nsig_outliers=nsig_outliers)
    else:
        tot_var = unb_var(signal[mask])
        expected_power_part = 1.

        # outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        # mask *= outlier_mask
        # # renormalize the variance to be unbiased
        # var_calc = np.nanvar(data1[mask])
        # expected_power = norm(loc=0, scale=1).expect(func=lambda x: x ** 2, lb=-nsig_outliers, ub=nsig_outliers, conditional=False)
        # var_calc = var_calc / expected_power
        #
        # outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * var_calc**0.5
        # mask *= outlier_mask

    signal[~mask] = 0.
    sum1 = np.nancumsum(signal * mask.astype(float))
    sum1 = np.concatenate(([0], sum1[:-1]))
    sum2 = np.nancumsum((signal * mask.astype(float)) ** 2)
    sum2 = np.concatenate(([0], sum2[:-1]))
    counts = np.nancumsum(mask)
    counts = np.concatenate(([0], counts[:-1]))
    # mov_var = np.ones(len(signal)) * np.inf

    hnw = nwind // 2
    hng = ngap // 2
    vecs = [sum2, sum1, counts]
    for i in range(len(vecs)):
        vec = vecs[i]
        temp_vec_w = np.roll(vec, -hnw - 1)
        temp_vec_w[len(temp_vec_w) - hnw - 1:] = vec[-1]
        temp_vec_g = np.roll(vec, -hng - 1)
        temp_vec_g[len(temp_vec_g) - hng - 1:] = vec[-1]
        right_vec = temp_vec_w - temp_vec_g

        temp_vec_w = np.roll(vec, hnw)
        temp_vec_w[:hnw] = vec[0]
        temp_vec_g = np.roll(vec, hng)
        temp_vec_g[:hng] = vec[0]
        left_vec = temp_vec_g - temp_vec_w
        vecs[i] = right_vec + left_vec
    sum2, sum1, counts = vecs
    mask_counts = (counts >= max(nsamples_min, 2))
    counts = counts.astype(float)  # for the next row
    counts[~mask_counts] = np.nan  # only for division, to avoid dividing by zero is there is some
    mov_var = (sum2 / counts - (sum1 / counts) ** 2) * counts / (counts - 1)
    mov_var = mov_var / expected_power_part
    mov_var[~mask_counts] = tot_var
    if not return_total_var:
        if not return_mask:
            return mov_var
        else:
            return mov_var, mask
    else:
        if not return_mask:
            return mov_var, tot_var
        else:
            return mov_var, mask, tot_var


def get_p_value_gaussian(x, mean, std):
    """
    Calculates the integral from x to infinity of the
    Gaussian with `mean` and `std`
    :param x:
    :type x: float
    :param mean:
    :type mean: float
    :param std:
    :type std: float
    :return:
    :rtype: float
    """
    return 0.5 * erfc((x - mean) / (2 ** 0.5 * std))


def fast_chi2_01_sf(x, df):
    """
    Faster than chi2(loc=0, scale=1).sf(x, df). For very long arrays, it is similar. For small arrays and
    single number, it is much faster
    :param x: the value for which the survival function is calculated
    :type x: float or numpy.ndarray of float
    :param df: number of degrees of freedom
    :type df: int
    :return:
    :rtype: same as `x`
    """
    if not hasattr(x, '__iter__'):
        if x <= 0:
            return 1.
        else:
            return gammaincc(df / 2.0, x / 2.0)
    else:
        res = gammaincc(df / 2.0, x / 2.0)
        res[x <= 0] = 1.
        return res


def fast_chi2_01_isf(x, df):
    return 2. * gammainccinv(df / 2.0, x)


def get_expected_power_after_outlier_removal(nsig_outliers):
    # conditional: bool, optional. If True, the integral is corrected by the conditional probability of
    # the integration interval. The return value is the expectation of the function, conditional on being
    # in the given interval. Default is False.
    # This is what we want to compute:
    # return norm(loc=0, scale=1).expect(func=lambda x: x ** 2, lb=-nsig_outliers, ub=nsig_outliers, conditional=True)
    # However, it is slow. Therefore, we compute the following equivalent:
    # res = norm.cdf(nsig) - nsig * norm.pdf(nsig) - (norm.cdf(-nsig) - (-nsig) * norm.pdf(-nsig))
    # normalization = norm.cdf(nsig) - norm.cdf(-nsig)
    # ndtr is like norm.cdf, but faster
    # standard_normal_pdf is like norm.pdf, but faster
    cdf_diff = 2 * ndtr(nsig_outliers) - 1  # this is also the normalization to make conditional=True
    res = cdf_diff - 2 * nsig_outliers * standard_normal_pdf(nsig_outliers)
    return res / cdf_diff


def outlier_free_std(data, nsig=3, n_iterations=1, mask=None, detrend=False, detr_len=128,
                     renormalize=True, return_var=False, return_mask=False):
    """
    Calculates the std of data without impact of outliers and excluding masked regions
    :param data: The data, may contain nans.
    :type data: np.array 1d of float
    :param nsig: Minimum amplitude in numbers of sigma (std) that is considered an outlier
    :type nsig: float
    :param n_iterations:
    :type n_iterations:
    :param mask: mask of bad values (False is bad, True is good). If None, will be
        replaced by measured mask of nans.
    :type mask: np.array 1d of bool
    :param detrend: If True, data will be detrended before measuring std
    :type detrend: bool
    :param detr_len: If detrend, data will be detrended with this detrending length
    :type detr_len: int
    :param renormalize: Account for the fact that when the outliers are cut, the expected result is smaller,
        so that it needs to be renormalized to be an unbiased variance estimate
    :type renormalize: bool
    :return: std of the data
    :rtype: float
    :param return_var:
    :type return_var:
    :param return_mask:
    :type return_mask:
    """
    if mask is None:
        mask = isgood(data)
    else:
        mask = np.array(mask, dtype=bool) & isgood(data)
    if detrend:
        data1 = data - moving_mean(data=np.array(data), nwind=detr_len, ngap=0, mask=mask, nsamples_min=32,
                                   remove_outliers=True, nsig_outliers=nsig,
                                   return_mask=False, return_total_mean=False,
                                   replace_ends=True, endmethod='mean')
        # data1 = detrend_mov_ave(np.array(data), length=detr_len, mask=mask, method='median', endmethod='mean',
        #                         ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False,
        #                         remove_outliers=True, nsig_outliers=nsig)
    else:
        data1 = np.array(data)

    if renormalize:
        expected_power = get_expected_power_after_outlier_removal(nsig_outliers=nsig)
    else:
        expected_power = 1.

    if np.isnan(data1[mask]).any() or (not np.all(np.isfinite(data1[mask]))) or len(data1[mask]) < 3:
        raise ValueError('Not enough data to calculate the std!')

    var_calc = np.nanvar(data1[mask]) * len(data1[mask]) / (len(data1[mask]) - 1)
    for i in range(n_iterations):
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig * var_calc ** 0.5
        mask = outlier_mask * mask
        var_calc = np.nanvar(data1[mask]) * len(data1[mask]) / (len(data1[mask]) - 1)
        if renormalize:
            var_calc = var_calc / expected_power
    if not return_mask:
        if not return_var:
            return var_calc**0.5
        else:
            return var_calc
    else:
        if not return_var:
            return var_calc**0.5, mask
        else:
            return var_calc, mask


def get_mean_and_std_of_mean_two_sigmas_model(data, stds=None, mask=None):
    """
    Calculates the standard deviation of the mean of the data with the following model. We assume that the
    data points data[i] come from the model with distribution normal(mu, (stds[i]**2+sigma_global**2)**0.5).
    The stds are known, and sigma_global is not known. The goal is to estimate mu and provide the error of this
    estimate. This function provides the error of this estimate. It finds it from calculating the estimator
    for sigma_global.
    PROBLEM !!!: for some reason, we do not use the given stds.
    :param data:
    :type data:
    :param stds:
    :type stds:
    :param mask:
    :type mask:
    :return:
    :rtype:
    """
    if mask is None:
        mask = isgood(data)
    # if not hasattr(stds, '__len__'):
    #     stds = np.ones(len(data)) * stds
    data = np.array(data)[mask]
    variances = (np.array(stds)[mask]) ** 2
    # std_p is std of the underlying population which may have an additional spread to the measurements
    # mean_p is estimate of the mean of the underlying population

    if len(data) == 1:
        var_p = 0
    else:
        var_p = np.var(data) * len(data) / (len(data) - 1) - np.mean(variances)
    if var_p < 0:
        var_p = 0
    msk = (variances + var_p) > 0
    if np.sum(msk.astype(float)) == 0:
        mean_p = np.nan
        var_mean_p = np.nan
    else:
        mean_p = (np.sum((data / (variances + var_p))[msk])) / (np.sum((1 / (variances + var_p))[msk]))
        var_mean_p = 1 / np.sum((1 / (variances + var_p))[msk])
    return mean_p, var_mean_p**0.5, var_p**0.5

    # # vars = (np.array(stds)**2)[mask]
    # n_samples = len(data)
    #
    # sum_data = np.sum(data)
    # sum_data_2 = np.sum(data ** 2)
    # var_of_mean_est = 1 / (n_samples * (n_samples - 1)) * (sum_data_2 - 1 / n_samples * sum_data ** 2)
    # return np.sqrt(var_of_mean_est)


def get_poisson_at_least(rate, min_number=1):
    """
    Calculates the poisson probability of having at least `min_number` events with the given `rate`.
    For example, we need the probability that one or more events will happen, then `min_number`=1,
    and the result is 1 - poisson(rate).pmf(0)
    :param rate: rate parameter in the poisson process. Mean number of events per unit time
    :type rate: float
    :param min_number: minimal number of events that we want to occur
    :type min_number: int
    :return: probability of having at least `min_number` events
    :rtype: float
    """
    p_small = 0
    for n in range(min_number):
        p_small += poisson(rate).pmf(n)
    return 1 - p_small


def get_average_dt(times):
    """
    take list of time arrays and get the average time step.
    It would be correct to weigh them by lenght, but we do not do it
    :param times:
    :type times: list of numpy.ndarray of float
    :return:
    :rtype: float
    """
    dts = []
    for q in range(len(times)):
        dts.append(np.nanmean((times[q] - np.roll(times[q], 1))[1:]))
    return float(np.nanmean(dts))  # although it would be correct to weight by the lengths of the quarters


def get_average_mean(fluxes):
    """
    Find the average (weighted by lengths) mean of a list of arrays
    :param fluxes: list of arrays of which the mean is to be found
    :type fluxes: list of numpy.ndarray of float
    :return: average mean
    :rtype: float
    """
    mean_total = 0.
    len_total = 0
    for q in range(len(fluxes)):
        mean = np.nanmean(fluxes[q])
        ll = len(fluxes[q][isgood(fluxes[q])])
        mean_total += mean * ll
        len_total += ll
    mean_total = mean_total / len_total
    return mean_total


def remove_end_nans(fluxes):
    for q in range(len(fluxes)):
        non_nans = np.where(~np.isnan(fluxes[q]))[0]
        fluxes[q] = fluxes[q][non_nans[0]: non_nans[-1] + 1]
    return fluxes


def window_data(data, alpha=0.1):
    """
    Apply Tukey window that smoothly zeroes the ends of the arrays.
    :param data: The vector or the list of vectors to be windowed.
    :type data: numpy.ndarray 1d or list of numpy.ndarray 1d
    :param alpha: smoothness parameter of the window
    :type alpha: float
    :return: data, the resulting windowed vector or vectors
    :rtype: same as `data`
    """
    if hasattr(data[0], '__len__'):
        for i in range(len(data)):
            data[i] = data * tukey(len(data), alpha=alpha, sym=True)


def pad_by_zeros_fun(data, support=None, minimal_extension=0, in_the_center=False, return_zero_padding_ends=False):
    """
    Pads data by zeros from the two sides.
    :param data:
    :type data: numpy.array 1d
    :param support: the new length of padded data. If None, The new length will be the closest power of 2 greater
        or equal to len(data)+minimal_extension
    :type support: int or None
    :param minimal_extension: If `support` is None, the new length of padded data is len(data)+minimal_extension
    :type minimal_extension: int
    :param in_the_center:
    :type in_the_center:
    :param return_zero_padding_ends: Whether to return the tuple of number of zeros padded from each side
    :type return_zero_padding_ends: bool
    :return: If return_zero_padding_ends:
                 data_padded, (zerlen1, zerlen2)
             Else:
                 data_padded
    :rtype:
    """
    if support is None:
        # the closest (but bigger) power of 2 to the length
        support = 2 ** int(np.ceil(np.log2(len(data) + minimal_extension)))
    if not in_the_center:
        zerlen1 = (support - len(data)) // 2
        zerlen2 = (support - len(data) + 1) // 2
        data_padded = np.concatenate((np.zeros(zerlen1), data, np.zeros(zerlen2)))
        if return_zero_padding_ends:
            return data_padded, (zerlen1, zerlen2)
        else:
            return data_padded
    else:
        data_padded = np.concatenate((data[:(len(data) + 1) // 2], np.zeros(support - len(data)), data[(len(data) + 1) // 2:]))
        return data_padded


def find_mean_with_outlier_removal(samples_list_per_paramerer, distr, outlier_sf=1e-2, n_repetitions=2,
                                   return_masks=False):
    """
    :param samples_list_per_paramerer: Outer index is different parameters (different means), inner index is different samples
        of the same distribution
    :type samples_list_per_paramerer:
    :param distr: str or distribution, function similar to scipy.stats distributions. should have `expect` and `isf` methods
    :type distr: str or distribution object from scipy.stats or similar
    :param outlier_sf:
    :type outlier_sf:
    :param n_repetitions:
    :type n_repetitions:
    :param return_masks:
    :type return_masks:
    :return:
    :rtype:
    """
    means = np.mean(samples_list_per_paramerer, axis=-1)
    masks = None
    if (not isinstance(distr, str)) or (distr != 'chi2_1'):
        unitless_boundary = distr.isf(outlier_sf)
        expected_val_truncated = distr.expect(func=lambda x: x, lb=0, ub=unitless_boundary, conditional=True)
    else:
        unitless_boundary = fast_chi2_01_isf(x=outlier_sf, df=1)
        squ = np.sqrt(unitless_boundary)
        cumul = ndtr(squ)
        # took the integral for chi2 distribution with 1 degree of freedom analytically
        expected_val_truncated = (2 * cumul - 2 * standard_normal_pdf(squ) * squ - 1) / (2 * cumul - 1)
    for _ in range(n_repetitions):
        means_temporary = np.array(means)
        upper_boundaries = means_temporary * unitless_boundary
        masks = [samples_list_per_paramerer[i] <= upper_boundary for (i, upper_boundary) in enumerate(upper_boundaries)]
        # we normalized all, so expected_val_full = 1., therefore we do not multiply by it
        means = [np.mean(samples_list_per_paramerer[i][mask]) / expected_val_truncated for (i, mask) in enumerate(masks)]
    means = np.array(means)
    if not return_masks:
        return means
    else:
        return means, masks


def psd_mean_with_outlier_removal(power_list, outlier_sf=1e-2, n_repetitions=2,
                                  rfft_edge_bins=True, return_masks=False):
    """
    Calculates the mean of slice powers `power_list` to estimate the psd with rejecting outliers. Assumes that the power
    should have exponential distribution with the scale parameter equal to the PSD at a given frequency. Measures iteratively
    `n_repetitions` times the PSD as the mean at a given frequency and removes the outliers set by exceeding a survival
    function of `outlier_sf` for an exponential distribution. After the iterations, calculates the mean with the final mask,
    and corrects the estimate for the fact that part of the distribution was truncated.
    :param power_list: List of power slices, each one being a vector for different frequencies
    :type power_list: list of numpy.ndarray 1d of float, where the length of the list is the number of slices
    :param outlier_sf: Survival function value to determine the outlier rejection threshold. Corresponds to exponential distribution
    :type outlier_sf: float
    :param n_repetitions: Number of iterations of rejecting outliers and re-estimating the mean
    :type n_repetitions: int
    :param rfft_edge_bins: Use the distribution of the power for rfft of a normal vector. For all the rfft frequencies except
        for the zeroth and the last, it is 'expon', for the zeroth and the last it is 'chi2' which is chi-squared
        distribution with 1 degree of freedom
    :type rfft_edge_bins: bool
    :param return_masks: Whether to return the masks of the non-outliers used in the mean calculation
    :type return_masks: bool
    :return: means, masks or only means.
    :rtype: means: numpy.ndarray 1d of float, masks: list of numpy.ndarray 1d of bool
    """
    from scipy.stats import expon, chi2
    power_list_per_freq = np.array(np.array(power_list).T)

    if return_masks:
        if rfft_edge_bins:
            power_list_per_freq_expon = np.array(power_list_per_freq[1:-1])
            means_expon, masks_expon = find_mean_with_outlier_removal(samples_list_per_paramerer=power_list_per_freq_expon,
                                                                      distr=expon(loc=0, scale=1),
                                                                      outlier_sf=outlier_sf, n_repetitions=n_repetitions,
                                                                      return_masks=True)
            power_list_per_freq_chi2 = np.array([power_list_per_freq[0], power_list_per_freq[-1]])
            means_chi2, masks_chi2 = find_mean_with_outlier_removal(samples_list_per_paramerer=power_list_per_freq_chi2,
                                                                    distr='chi2_1',  # chi2(loc=0, scale=1, df=1),
                                                                    outlier_sf=outlier_sf, n_repetitions=n_repetitions,
                                                                    return_masks=True)
            means_final = np.concatenate(([means_chi2[0]], means_expon, [means_chi2[-1]]))
            masks_final = [masks_chi2[0]] + masks_expon + [masks_chi2[-1]]
        else:
            means_final, masks_final = find_mean_with_outlier_removal(samples_list_per_paramerer=power_list_per_freq,
                                                                      distr=expon(loc=0, scale=1),
                                                                      outlier_sf=outlier_sf, n_repetitions=n_repetitions,
                                                                      return_masks=True)
        return means_final, masks_final
    else:
        if rfft_edge_bins:
            power_list_per_freq_expon = np.array(power_list_per_freq[1:-1])
            means_expon = find_mean_with_outlier_removal(samples_list_per_paramerer=power_list_per_freq_expon,
                                                         distr=expon(loc=0, scale=1),
                                                         outlier_sf=outlier_sf, n_repetitions=n_repetitions,
                                                         return_masks=False)
            power_list_per_freq_chi2 = np.array([power_list_per_freq[0], power_list_per_freq[-1]])
            means_chi2 = find_mean_with_outlier_removal(samples_list_per_paramerer=power_list_per_freq_chi2,
                                                        distr=chi2(loc=0, scale=1, df=1),
                                                        outlier_sf=outlier_sf, n_repetitions=n_repetitions,
                                                        return_masks=False)
            means_final = np.concatenate(([means_chi2[0]], means_expon, [means_chi2[-1]]))
        else:
            means_final = find_mean_with_outlier_removal(samples_list_per_paramerer=power_list_per_freq,
                                                         distr=expon(loc=0, scale=1),
                                                         outlier_sf=outlier_sf, n_repetitions=n_repetitions,
                                                         return_masks=False)
        return means_final

    # means = np.mean(power_list, axis=0)
    # masks = None
    # unitless_boundary = distr.isf(outlier_sf)
    # expected_val_truncated = distr.expect(func=lambda x: x, lb=0, ub=unitless_boundary, conditional=True)
    # for _ in range(n_repetitions):
    #     means_temporary = np.array(means)
    #     upper_boundaries = means_temporary * unitless_boundary
    #     masks = [power_list_per_freq[i] <= upper_boundary for (i, upper_boundary) in enumerate(upper_boundaries)]
    #     # we normalized all, so expected_val_full = 1., therefore we do not multiply by it
    #     means = [np.mean(power_list_per_freq[i][mask]) / expected_val_truncated for (i, mask) in enumerate(masks)]
    # means = np.array(means)


def welch_psds(clean_fluxes, psd_masks, nperseg, noverlap, window, use_rfft=True, reject_outliers=False, outlier_sf=1e-2,
               n_repetitions_outliers=4, return_freqs=False, dt=1., return_npsds=False, return_slices=False):
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
    if noverlap < 1:
        noverlap = int(nperseg * noverlap)
    if noverlap == -1:
        noverlap = nperseg // 2

    windowdict = {'boxcar': np.ones(nperseg) / np.linalg.norm(np.ones(nperseg)),
                  'hann': 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg)) / np.linalg.norm(
                      0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg)))}
    if use_rfft:
        fftfun = rfft
        freqfun = np.fft.rfftfreq
    else:
        fftfun = np.fft.fft
        freqfun = np.fft.fftfreq

    psds = []
    for i in range(len(clean_fluxes)):
        psds.append([np.abs(fftfun(windowdict[window] * clean_fluxes[i][ind1:ind1 + nperseg])) ** 2
                     for ind1 in range(0, len(clean_fluxes[i]) - nperseg, nperseg - noverlap
                                       ) if np.sum((~psd_masks[i])[ind1:ind1 + nperseg]) == 0])
    npsds = np.array([len(p) for p in psds])  # number of psds used to take the average
    returns = []
    returns_count = 0
    if return_slices:
        slices = copy.deepcopy(psds)
        returns.append(slices)
        returns_count += 1

    if reject_outliers:
        psds = [psd_mean_with_outlier_removal(p, outlier_sf=outlier_sf, n_repetitions=n_repetitions_outliers,
                                              rfft_edge_bins=True, return_masks=False) for p in psds]
    else:
        psds = [np.mean(p, axis=0) for p in psds]
    returns.append(psds)
    returns_count += 1

    if return_freqs:
        returns.append([freqfun(nperseg, d=dt) for _ in range(len(psds))])
        returns_count += 1
    if return_npsds:
        returns.append(npsds)
        returns_count += 1

    if returns_count == 1:
        return returns[0]
    else:
        return returns


def window_wf_td_fun(wf_fd,  extension_coef=2.5, meas_len=256):
    """
    applies the tukey window to one whitening template in the time domain to make it smooth but preserve the mean zero
    because this will ensure that the lowest frequency gets zero in the Fourier domain whitening filter.
    :param wf_fd:
    :type wf_fd:
    :param extension_coef:
    :type extension_coef:
    :param meas_len:
    :type meas_len:
    :return:
    :rtype:
        """
    wf_td = irfft(wf_fd)
    wind = np.zeros(len(wf_td))
    len_to_save = meas_len  # it is 2 nperseg
    tuk = tukey(int(2 * extension_coef * len_to_save), alpha=1 - 1 / extension_coef, sym=True)
    wind[:len(tuk) // 2] = tuk[-len(tuk) // 2:]
    wind[-len(tuk) // 2:] = tuk[:len(tuk) // 2]
    wf_td = wf_td * wind
    wf_td -= np.mean(wf_td) * wind / np.mean(wind)
    wf_fd = np.abs(rfft(wf_td))
    wf_td = irfft(wf_fd)  # we need wf_fd for inpainting
    return wf_td, wf_fd


def psd2wf(psd, support=None, use_rfft=True, tukey_steps_len=0,
           conditional_zero_zeroth_freq=True, always_zero_zeroth_freq=False):
    """
    Generates a whitening filter of the needed length (support) from the PSD.
    Makes sinc-extrapolation, applies tuckey window to the end and to the low frequencies if they drop too quickly
    :param psd: measured PSD in coarse resolution
    :type psd: ndarray
    :param support: the desired length of the extrapolated filter (time domain). If None, take the length corresponding to the psd
    :type support: int
    :param use_rfft: whether to use rfft (or just fft). If true, the signal is treated as real,
                 and all ffts are rffts. The length of the frequency samples are the nperseg//2+1.
    :type use_rfft: bool
    :param tukey_steps_len: length taken for zeroing in the tukey windowing
    :type tukey_steps_len: int or float
    :param conditional_zero_zeroth_freq: whether to put the low frequencies to zero if they deviate too much
    :type conditional_zero_zeroth_freq: bool
    :param always_zero_zeroth_freq: whether to always put the lowest frequencies to zero
    :type always_zero_zeroth_freq: bool
    :return: wf_td : whitening filter in time domain; wf_fd : whitening filter in Fourier domain.
    :rtype: tuple(ndarray, ndarray)
    """
    wf_fd_meas = 1 / np.array(psd)**0.5
    was_vector = False
    if not hasattr(wf_fd_meas[0], '__len__'):
        wf_fd_meas = np.array([wf_fd_meas])
        was_vector = True
    if use_rfft:
        ifftfun = irfft
        fftfun = rfft
        dtype_res = np.float64
        if support is None:
            support = 2 * (len(wf_fd_meas) - 1)
    else:
        if support == 'none':
            support = len(wf_fd_meas)
        ifftfun = np.fft.ifft
        fftfun = np.fft.fft
        dtype_res = np.complex128

    assert support % 2 == 0, 'self.support assumed to be even!'
    wf_tds = []
    wf_fds = []
    for wpsd in wf_fd_meas:
        wf_td = ifftfun(wpsd)
        meas_len = len(wf_td)
        wf_td = np.fft.ifftshift(np.fft.fftshift(wf_td) * tukey(meas_len, alpha=tukey_steps_len / meas_len, sym=True))
        mid_point = len(wf_td) // 2
        result = np.zeros(support, dtype=dtype_res)
        result[:mid_point] = wf_td[:mid_point]
        result[-mid_point:] = wf_td[-mid_point:]

        wf_td = np.array(result)  # np.fft.fftshift(result)
        wf_fd = np.abs(fftfun(result))  # It is complex, which is weird, but what can we do
        if ((conditional_zero_zeroth_freq and (
                np.mean(np.abs(wf_fd[0:len(wf_fd) // 50])) / np.mean(np.abs(wf_fd[5 * len(wf_fd) // 10:7 * len(wf_fd) // 10])) < 0.3))
                or always_zero_zeroth_freq):
            first_measured_bin = int(np.ceil(support / meas_len))
            # wf_fd[:first_measured_bin] = 0.
            lentuk = 2 * first_measured_bin
            wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * tukey(lentuk, alpha=3 * first_measured_bin / lentuk,
                                                              sym=True)[:lentuk // 2]
            wf_td = ifftfun(wf_fd)  # np.fft.fftshift(ifftfun(wf_fd))
        wf_tds.append(wf_td)
        wf_fds.append(wf_fd)

    if was_vector:
        wf_tds = wf_tds[0]
        wf_fds = wf_fds[0]

    return wf_tds, wf_fds


def whiten_vector(vector, wf_fd):
    support = 2 * (len(wf_fd) - 1)
    vector = np.roll(vector, -np.argmax(vector))
    vector = np.concatenate((vector[:len(vector) // 2], np.zeros(support - len(vector)),
                             vector[len(vector) // 2:]))
    vector = irfft(rfft(vector) * wf_fd)
    return vector


def filter2cov_matrix(vec, pad_by_zeros=False, trim_zeros=True):
    """
    Make a covariance matrix for the statistically shift-invariant data
    in the time domain from the covariance vector in time domain (vec).
    If vec is the time domain whitening filter, the result will be
    the INVERSE covariance matrix.
    :param vec: A filter which will be convolved with itself in an
        outer product way to make a covariance matrix
    :type vec: numpy.ndarray 1d
    :param pad_by_zeros: If true, pads the vector by zeros of the same size
        in order to omit calculation of covariance with Bins from the other side.
    :type pad_by_zeros: bool
    :param trim_zeros: Trim after padding the omitted zeros. The default is False.
    :type trim_zeros: bool
    :return: The covariance matrix (for the time domain).
        If vec is the time domain whitening filter, the result will be
        the INVERSE covariance matrix.
    :rtype: numpy.ndarray 2d
    """
    length = len(vec)
    if pad_by_zeros:
        vec = np.concatenate((vec, np.zeros(length)))
        length = len(vec)
    prod = np.array([np.sum(vec * np.roll(vec, i)) for i in range(length)])
    cov_matr = np.array([[prod[k - j] for k in range(length)] for j in range(length)])
    if pad_by_zeros and trim_zeros:
        assert length % 4 == 0, 'The length of the vector has to be even.'
        return cov_matr[length // 4:-length // 4, length // 4:-length // 4]
    return cov_matr


# Inpaint_rand_vec = np.abs(np.random.normal(loc=0, scale=0.001, size=8192).astype(Dtype_r))
#
#
# def inpaint_fluxes_fun(fluxes, masks, wf_tds, dtype=np.float32, rand_vec=Inpaint_rand_vec):
#     length = len(fluxes[0])
#     # I was afraid that wf_tds are fftshifted to the center, but since here we have the inner product, it is not important
#     inpainted_fluxes = np.zeros((len(fluxes), length), dtype=dtype)
#     for q in range(len(fluxes)):
#         # filter2cov_matrix:
#         prod = np.array([np.sum(wf_tds[q] * np.roll(wf_tds[q], i)) for i in range(length)])
#         c_inv = np.array([[prod[k - j] for k in range(length)] for j in range(length)], dtype=dtype)
#         a_t_c_inv = c_inv[~masks[q]]
#         m = a_t_c_inv[:, ~masks[q]]
#         m += np.identity(len(m), dtype=dtype) * rand_vec[:len(m)] * float32(np.mean(np.diag(m)))
#         m_inv_a_t_c_inv = np.linalg.solve(a=m, b=a_t_c_inv)  # ax = b  solution
#         f = np.zeros(shape=(length, length), dtype=dtype)
#         f[~masks[q]] = m_inv_a_t_c_inv
#         inpainted_fluxes[q] = fluxes[q] - np.dot(f, fluxes[q])
#     return inpainted_fluxes


def nan_average_filt(ar, kernel_size=3, method='mean'):
    """
    Very slow and bas for the edges. Use miving_mean instead, or detrend_mov_ave in the worst case.
    Return moving average of the data with the kernel_size.
    The average is either mean or median, defined by method.
    :param ar: The input array to filter.
    :type ar: numpy.ndarray
    :param kernel_size: The size of subarray around each point to find the mean, should be odd. The default is 3.
    :type kernel_size: int
    :param method: The method of how to compute the average
    :type method: str, 'mean' or 'median'
    :return: The array where each bin is the mean over kernel_size neighbors with
        ignoring nans (unless all the subarray is nans)
    :rtype: numpy.ndarray
    """
    arfilt = np.array([ar[0]])
    if method == 'mean':
        for i in np.arange(1, kernel_size // 2):
            arfilt = np.append(arfilt, np.nanmean(ar[0: 2 * i + 1]))
        for i in np.arange(kernel_size // 2, len(ar) - kernel_size // 2 - 1):
            arfilt = np.append(arfilt, np.nanmean(ar[i - kernel_size // 2: i + kernel_size // 2 + 1]))
        for i in np.arange(len(ar) - kernel_size // 2 - 1, len(ar)):
            arfilt = np.append(arfilt, np.nanmean(ar[2 * i - 1 - len(ar):]))
    elif method == 'median':
        for i in np.arange(1, kernel_size // 2):
            arfilt = np.append(arfilt, np.nanmedian(ar[0: 2 * i + 1]))
        for i in np.arange(kernel_size // 2, len(ar) - kernel_size // 2 - 1):
            arfilt = np.append(arfilt, np.nanmedian(ar[i - kernel_size // 2: i + kernel_size // 2 + 1]))
        for i in np.arange(len(ar) - kernel_size // 2 - 1, len(ar)):
            arfilt = np.append(arfilt, np.nanmedian(ar[2 * i - 1 - len(ar):]))
    else:
        raise AssertionError('Undefined averaging method specified!')
    arfilt[-1] = ar[-1]
    return arfilt


def interpolate_over_mask(flux, mask=None, interp_method='linear', ends='continue', mitigate_hole_ends=False,
                          endhole_method='median', mit_length=10):
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
        'zeros' means to replace by zeros.
        'keep' orig_means to leave what was in the original data (maybe nans)
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
    if mask is None:
        mask = isgood(flux)
    else:
        mask = (np.array(mask).astype(bool)) & isgood(flux)

    ind1 = None
    ind2 = None
    if interp_method == 'mean' or interp_method == 'median':
        ind1 = (np.where(mask)[0])[0]
        ind2 = (np.where(mask)[0])[-1]
        # if there is a hole in the beginning or end, it cannot be interpolated
        mask1 = np.array(mask[ind1:ind2 + 1])
        begholes = np.where(mask1 * np.roll(~mask1, -1))[0] + 1
        endholes = np.where(mask1 * np.roll(~mask1, 1))[0]
        lenholes = endholes - begholes
        kernel_size = np.max(lenholes) + 4
        indholes = np.where(~mask1)[0]
        flux[indholes] = moving_mean(data=flux, nwind=kernel_size, ngap=0, mask=mask1, nsamples_min=kernel_size//4,
                                     remove_outliers=False, nsig_outliers=6, return_mask=False, return_total_mean=False)
    elif interp_method == 'iterative_fourier':
        nrep = 6
        flux1 = np.array(flux)
        indholes = np.where(~ mask)[0]
        flux1[indholes] = 0
        for i in range(nrep):
            fflux1 = np.fft.fft(flux1)
            fflux1[len(flux1) // 10 + 4 ** i: -(len(flux1) // 10 + 4 ** i)] = 0
            flux1[indholes] = (np.fft.ifft(fflux1))[indholes]
        flux = flux1
    elif interp_method == 'linear':
        ind1 = (np.where(mask)[0])[0]
        ind2 = (np.where(mask)[0])[-1]
        # if there is a hole in the beginning or end, it cannot be interpolated
        flux1 = np.array(flux[ind1:ind2 + 1])
        mask1 = np.array(mask[ind1:ind2 + 1])
        begholes = np.where(mask1 * np.roll(~mask1, -1))[0]
        endholes = np.where(mask1 * np.roll(~mask1, 1))[0]
        if not mitigate_hole_ends:
            for j in range(len(begholes)):
                flux1[begholes[j] + 1:endholes[j]] = np.interp(
                    np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
                    np.array([flux1[begholes[j]], flux1[endholes[j]]]))[1:-1]
        else:
            if endhole_method == 'median':
                av_fun = np.median
            elif endhole_method == 'mean':
                av_fun = np.mean
            else:
                raise AssertionError('Incorrect endhole_method specified!')
            for j in range(len(begholes)):
                i1 = max(begholes[j] - mit_length + 1, 0)
                i2 = min(endholes[j] + mit_length, len(flux1))

                slice1 = (flux1[i1:begholes[j] + 1])[np.where(mask1[i1:begholes[j] + 1])[0]]
                slice2 = (flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]]
                if (len(slice1) > 0) and (len(slice2) > 0):
                    flux1[begholes[j] + 1:endholes[j]] = np.interp(
                        np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
                        np.array([av_fun(slice1), av_fun(slice2)]))[1:-1]
                else:
                    raise ValueError('Interpolation failed!')
            # # Same but without a check, which is redundant but I was trying to fix a bug:
            # if endhole_method == 'median':
            #     for j in range(len(begholes)):
            #         i1 = max(begholes[j] - mit_length + 1, 0)
            #         i2 = min(endholes[j] + mit_length, len(flux1))
            #         flux1[begholes[j] + 1:endholes[j]] = np.interp(
            #             np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
            #             np.array([np.nanmedian((flux1[i1:begholes[j] + 1])[np.where(mask1[i1:begholes[j] + 1])[0]]),
            #                       np.nanmedian((flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            # elif endhole_method == 'mean':
            #     for j in range(len(begholes)):
            #         i1 = max(begholes[j] - mit_length + 1, 0)
            #         i2 = min(endholes[j] + mit_length, len(flux1))
            #         flux1[begholes[j] + 1:endholes[j]] = np.interp(
            #             np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
            #             np.array([np.nanmean((flux1[i1:begholes[j] + 1])[np.where(mask1[i1:begholes[j] + 1])[0]]),
            #                       np.nanmean((flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            # else:
            #     raise AssertionError('Incorrect endhole_method specified!')
        flux[ind1:ind2 + 1] = flux1
    if interp_method == 'linear' or interp_method == 'mean' or interp_method == 'median':
        if ends == 'continue':
            if ind1 != 0:
                flux[0:ind1] = flux[ind1]
            if ind2 != len(flux) - 1:
                flux[ind2:] = flux[ind2]
        elif ends == 'zeros':
            if ind1 != 0:
                flux[0:ind1] = 0
            if ind2 != len(flux) - 1:
                flux[ind2:] = 0
        elif ends == 'keep':
            pass
        else:
            raise AssertionError('Undefined ends parameter specified!')
    return flux


def sinc_interp(f_vec, support, tukey_steps_len=16):
    """
    Works for real-fft! returns real vector!
    """
    if tukey_steps_len < 1:
        tukey_steps_len = support * tukey_steps_len
    t_vec = irfft(f_vec)
    meas_len = len(t_vec)
    t_vec = np.fft.ifftshift(np.fft.fftshift(t_vec) * tukey(meas_len, alpha=tukey_steps_len / meas_len, sym=True))
    mid_point = len(t_vec) // 2
    result = np.zeros(support, dtype=np.float64)  # f_vec.dtype)
    result[:mid_point] = t_vec[:mid_point]
    result[-mid_point:] = t_vec[-mid_point:]
    return np.abs(rfft(result))


def tanhstep(x, center=0., hwhm=1., start_from_zero=False):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        res = (1. + np.tanh((np.array(x) - center) / hwhm * 1 / 2 * np.log(3))) / 2.
        if start_from_zero:
            m = np.nanmax(res)
            res -= np.nanmin(res)
            res *= m / np.nanmax(res)
        return res
    else:
        return (1. + np.tanh((np.array(x) - center) / hwhm * 1 / 2 * np.log(3))) / 2.


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
    data_low : numpy.array or list of float3
        Smooth low-frequency trend of the `data`

    """
    assert fit_order == 'quadratic' or fit_order == 'cubic', 'Undefined `fit_order` specified!'

    data = np.array(data)
    time = np.arange(len(data))
    nanmask = isgood(data)
    if not nanmask.all():
        data_i = interpolate_over_mask(data, mask=nanmask)
    else:
        data_i = data
    if mask is None:
        mask = nanmask
    if len(data) == 1:
        warnings.warn('WHY YOU GIVE ME DATA OF LENGTH 1 TO FIT???')
    if (len(data) <= nperseg) or (len(data) // 2 <= endlen):
        # parabolic fit or the entire data
        if fit_order == 'quadratic':
            msk1 = np.array(mask).astype(bool)
            dat1 = data_i[msk1]
            tim1 = time[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1) // 2]
            x10 = (tim1[-1] + tim1[0]) / 2
            x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2),
                           np.array([x00, x10, x20]))
            data_low = (sol.x[2] * (np.arange(len(data)) - sol.x[1]) ** 2 + sol.x[0])
        else:
            msk1 = np.array(mask).astype(bool)
            dat1 = data_i[msk1]
            tim1 = time[msk1]
            tim10 = tim1[0]
            tim1 = tim1 - tim10
            x00 = (dat1[0] * tim1[-1] - dat1[-1] * tim1[0]) / (tim1[-1] - tim1[0])
            x10 = (dat1[-1] - dat1[0]) / (tim1[-1] - tim1[0])
            x20 = 0.
            x30 = 0.
            sol = minimize(
                lambda x: np.nansum((dat1 - (x[3] * tim1 ** 3 + x[2] * tim1 ** 2 + x[1] * tim1 + x[0])) ** 2),
                np.array([x00, x10, x20, x30]))
            data_low = (sol.x[3] * (np.arange(len(data))) ** 3
                        + sol.x[2] * (np.arange(len(data))) ** 2
                        + sol.x[1] * (np.arange(len(data))) + sol.x[0])

    else:
        if endlen is None:
            endlen = nperseg
        assert endlen < len(data) // 2, '`endlen` is too big!'

        if data_low is None:
            data_rf = rfft(np.array(data_i))
            data_rf[int(len(data_i) / nperseg):] = 0.
            data_low = irfft(data_rf, n=len(data_i))
        else:
            assert len(data_low) == len(data), 'The lengths of `data_low` and `data` must be equal!'

        w1 = tanhstep(np.arange(endlen), center=endlen / 4 * 3, hwhm=endlen / 16, start_from_zero=True)
        w2 = 1 - w1
        w3 = w1[::-1]
        w4 = w2[::-1]
        data_low[:endlen] *= w1
        data_low[-endlen:] *= w3

        # fit the beginning
        if fit_order == 'quadratic':
            msk1 = np.array(mask[:endlen]).astype(bool)
            dat1 = (data_i[:endlen])[msk1]
            tim1 = (time[:endlen])[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1) // 2]
            x10 = (tim1[-1] + tim1[0]) / 2
            x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2),
                           np.array([x00, x10, x20]))
            data_low[:endlen] += (sol.x[2] * (np.arange(endlen) - sol.x[1]) ** 2 + sol.x[0]) * w2
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
            sol = minimize(
                lambda x: np.nansum((dat1 - (x[3] * tim1 ** 3 + x[2] * tim1 ** 2 + x[1] * tim1 + x[0])) ** 2),
                np.array([x00, x10, x20, x30]))
            data_low[:endlen] += (sol.x[3] * (np.arange(endlen)) ** 3
                                  + sol.x[2] * (np.arange(endlen)) ** 2
                                  + sol.x[1] * (np.arange(endlen)) + sol.x[0]) * w2

        # fit the end:
        if fit_order == 'quadratic':
            msk1 = np.array(mask[len(data_i) - endlen:]).astype(bool)
            dat1 = (data[len(data) - endlen:])[msk1]
            tim1 = (time[len(data) - endlen:])[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1) // 2]
            x10 = (tim1[-1] + tim1[0]) / 2
            x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2),
                           np.array([x00, x10, x20]))
            data_low[-endlen:] += (sol.x[2] * (np.arange(endlen) + time[len(data) - endlen] - sol.x[1]) ** 2 + sol.x[
                0]) * w4
        else:
            msk1 = np.array(mask[len(data_i) - endlen:]).astype(bool)
            dat1 = (data[len(data) - endlen:])[msk1]
            tim1 = (time[len(data) - endlen:])[msk1]
            tim10 = tim1[0]
            tim1 = tim1 - tim10
            x00 = (dat1[0] * tim1[-1] - dat1[-1] * tim1[0]) / (tim1[-1] - tim1[0])
            x10 = (dat1[-1] - dat1[0]) / (tim1[-1] - tim1[0])
            x20 = 0.
            x30 = 0.
            sol = minimize(
                lambda x: np.nansum((dat1 - (x[3] * tim1 ** 3 + x[2] * tim1 ** 2 + x[1] * tim1 + x[0])) ** 2),
                np.array([x00, x10, x20, x30]))
            data_low[-endlen:] += (sol.x[3] * (np.arange(endlen) + time[len(data) - endlen] - tim10) ** 3
                                   + sol.x[2] * (np.arange(endlen) + time[len(data) - endlen] - tim10) ** 2
                                   + sol.x[1] * (np.arange(endlen) + time[len(data) - endlen] - tim10) + sol.x[0]) * w4

    return data_low


# noinspection PyTypeChecker
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
    :param data: Data vector to be detrended
    :type data: Union[ndarray, List[float]] # numpy.array or list of float
    :param big_mask:
    :type big_mask:
    :param mask: Masking good data points with `True`, bad data points with `False`.
    :type mask: numpy.ndarray of bool
    :param nperseg: Wavelength below which the frequencies should be included in the trend.
    :type nperseg: int
    :param endlen: Length of the data in the ends to be fitted by quadratic polynomial
    :type endlen: int
    :param nsig_step: Minimal step between the edges of t wo subsequent pieces so that they
        will be detrended separately (in units of high-frequency std of `data`)
    :type nsig_step: float
    :param checklen: Length of gap extension data from the edges used to calculate the
        residuals to determine where are the steps.
    :type checklen: int
    :param residual_threshold_step_in_sigma: In the end, the trend is verified for having steps around holes in
        the residuals. If the residuals before and after the hole differ more than
        by `residual_threshold_step_in_sigma` sigma, the piece will be cut, and
        its ends will be fitted separately.
    :type residual_threshold_step_in_sigma: float
    :param slice_for_residual_step: Length of slice used to calculate the median values around the hole
        which are needed to identify steps after the first trend attempt.
    :type slice_for_residual_step: float
    :param return_trend: Whether to return trend in addition to detrended data
    :type return_trend: bool
    :return: if return_trend:
                 detrended, trend : numpy.arrays
                 Detrended data and trend
             else:
                 detrended : numpy.array
                 `detrended` is detrended data containing zeros in the bad places of masks.
                 `trend` is the low-frequency trend of the data containinhg nans
                 in the bad places of masks
    :rtype:
    """
    threshold_step_size_std = residual_threshold_step_in_sigma  # redefining names
    medlen = slice_for_residual_step
    data = np.array(data)  # copying and going to numpy
    if np.isnan(data).any():
        raise ValueError('data has NaNs!')
    nanmask = isgood(data)
    if not nanmask.all():  # interpolating data if it contains nans
        data_i = interpolate_over_mask(data, mask=nanmask)  # will need for fft
    else:
        data_i = data

    if mask is None:
        mask = nanmask
    else:
        mask = np.array(mask).astype(bool)
        mask *= nanmask

    if big_mask is None:  # big mask shows potential jumps. If there is no big mask, no jumps will be checked for.
        big_mask = clean_small_holes(mask, max_hole=5)  # np.ones(len(data)).astype(bool)
    else:
        big_mask = np.array(big_mask).astype(bool)

    # remove cases when there is only one unmasked point surrounded by masked because these pieces are hard to detrend
    (big_mask[1:-1])[(big_mask & ~np.roll(big_mask, 1) & ~np.roll(big_mask, -1))[1:-1]] = False

    if nperseg is None:
        nperseg = len(data) // 10
    if endlen is None:
        endlen = nperseg

    detrended = np.zeros(len(data))  # will collect here the result

    ind_i = (np.where(mask)[0])[0]  # start where there is the first valid point
    ind_f = (np.where(mask)[0])[-1] + 1  # end with the last valid point
    mask0 = np.array(big_mask[ind_i:ind_f])  # mask of the working region
    endpieces = np.where((~mask0) * np.roll(mask0, 1))[0] + ind_i  # ends of pieces split by holes
    endpieces = np.concatenate((endpieces, [ind_f]))
    begpieces = np.where(mask0 * np.roll(~mask0, 1))[0] + ind_i  # beginnings of pieces split by holes
    begpieces = np.concatenate(([ind_i], begpieces))

    fdat = np.abs(rfft(data_i))
    subset = fdat[-len(data) // 4:]
    if np.any(np.isnan(subset)):
        raise ValueError("Input data is nan!")
    elif np.any(np.isinf(subset)) or np.max(np.abs(subset)) > 1e38:
        raise ValueError("Input data too large or invalid!")

    std_fast_0 = (np.mean((fdat[-len(data) // 4:]) ** 2) / (len(data))) ** 0.5  # standard deviation of fast modes
    std_fast = std_fast_0 + np.max(fdat) / 20000  # I don't remember why exactly this was made, but it worked

    # now we will select the bad pieces which have to be detrended separately:
    beg_big_pieces = [ind_i]
    end_big_pieces = []
    for i in range(1, len(begpieces)):
        i1 = max(endpieces[i - 1] - checklen,
                 begpieces[i - 1])  # if the piece before the hole is too small, take it all
        i2 = min(begpieces[i] + checklen, endpieces[i])  # if the piece after the hole is too small, take it all
        mask1 = big_mask[i1: i2]
        dat1 = (data_i[i1: i2])[mask1]
        tim1 = np.arange(len(mask1))[mask1]

        x00 = np.nanmean(dat1)  # define initial fit parameters
        cent = dat1[len(dat1) // 2]
        x10 = (tim1[-1] + tim1[0]) / 2
        x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
        sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2), np.array([x00, x10, x20]))
        # tr_end = sol.x[2]*(tim1-sol.x[1])**2 + sol.x[0]
        mres = (sol.fun / len(dat1)) ** 0.5  # mean residual

        if mres > nsig_step * std_fast:  # if the fit does not fit the two pieces together, detrend them separately
            end_big_pieces.append(endpieces[i - 1])
            beg_big_pieces.append(begpieces[i])
    end_big_pieces.append(ind_f)

    # make one detrending and check it is ok inside the pieces:
    for i in range(len(beg_big_pieces)):
        piece = np.array(data_i[beg_big_pieces[i]:end_big_pieces[i]])
        mask_p = mask[beg_big_pieces[i]:end_big_pieces[i]]
        big_mask_p = big_mask[beg_big_pieces[i]:end_big_pieces[i]]
        trend_local = trend_fft(piece, nperseg=nperseg, endlen=endlen, mask=mask_p)
        begholes = np.where(big_mask_p * np.roll(~big_mask_p, -1))[0] + 1  # define begholes inside the piece
        endholes = np.where(big_mask_p * np.roll(~big_mask_p, 1))[0]

        if not isgood(trend_local).all():
            raise ValueError('Trend has NaNs!')
        beg_small_pieces = [0]
        end_small_pieces = []
        for j in range(len(begholes)):
            # piece_left = piece[begholes[j] - medlen:begholes[j]]
            # piece_right = piece[endholes[j]:endholes[j] + medlen]
            if ((len(np.where(isgood(piece[begholes[j] - medlen:begholes[j]]))[0]) >= 3)
                    and (len(np.where(isgood(piece[endholes[j]:endholes[j] + medlen]))[0]) >= 3)
                    and ((np.abs(np.nanmedian(piece[begholes[j] - medlen:begholes[j]]
                                              - trend_local[begholes[j] - medlen:begholes[j]])
                                 - np.nanmedian(piece[endholes[j]:endholes[j] + medlen]
                                                - trend_local[endholes[j]:endholes[j] + medlen])
                                 ) / std_fast) > threshold_step_size_std)):
                # inds_step.append((beg_big_pieces[i]+begholes[j], beg_big_pieces[i]+endholes[j]))
                end_small_pieces.append(begholes[j])
                beg_small_pieces.append(endholes[j])
        end_small_pieces.append(len(piece))
        if len(end_small_pieces) == 1:
            detrended[beg_big_pieces[i]: end_big_pieces[i]] = (
                    np.array(data[beg_big_pieces[i]:end_big_pieces[i]]) - trend_local)
        else:
            for k in range(len(beg_small_pieces)):
                small_piece = np.array(piece[beg_small_pieces[k]:end_small_pieces[k]])
                mask_s = np.array(mask_p[beg_small_pieces[k]:end_small_pieces[k]])
                trend_s_0 = np.array(trend_local[beg_small_pieces[k]:end_small_pieces[k]])
                trend_s = trend_fft(small_piece, nperseg=nperseg, endlen=endlen, mask=mask_s, data_low=trend_s_0)
                detrended[beg_big_pieces[i] + beg_small_pieces[k]:beg_big_pieces[i] + end_small_pieces[k]] = (
                        np.array(data[beg_big_pieces[i] + beg_small_pieces[k]:beg_big_pieces[i] + end_small_pieces[
                            k]]) - trend_s)
                # trend[beg_big_pieces[i]:end_big_pieces[i]] = trend_local

    trend = np.array(data_i - detrended)
    if return_trend:
        return detrended, trend
    else:
        return detrended


def detrend_mov_ave(data, length=128, mask=None, method='median', endmethod='mean', ends='symmetric', endlen=None,
                    return_trend=True, keep_overall_level=False, remove_outliers=False, nsig_outliers=Nsig_outliers_0,
                    detr_length_outliers=Detr_length_outliers, remove_outliers_light=False):
    """
    Amateur detrending function. Subtracting the moving mean or median from the data.
    :param data: Data sample from which the moving average has to be subtracted
    :type data: np.array 1d or list
    :param length: The length of sample to calculate moving average. At a given point,
                    half-length to the left and to the right are taken.
    :type length: int
    :param mask:
    :type mask:
    :param method: Kind of moving value to subtract: 'median' or 'mean'
                    The default is 'median', because it is less sensitive to outliers.
    :type method: str
    :param endmethod: Kind of moving value to subtract at the ends of the data (see below): 'median' or 'mean'.
                    The default is 'mean', because it is less sensitive to small
                    amount of data samples available in the ends.
    :type endmethod: str
    :param ends: How to treat the points closer than length/2 to the ends of the data:
                'symmetric' :  endlen is length/4
                               1) if point<endlen, take endlen samples from the right, and up to zero from the left
                               2) if endlen<point<half-length, take the same number of points from left and right
                               3) if point>half-length, take usual half-length from both sides
                               for the second end, the same in opposite direction
                'asymmetric' : endlen is length/16
                               1) if point<half-length, take up to zero from the left, and the length to the
                                    right linearly increases from endlen for point=0 to length/2 at point=length/2
                               2) if point>half-length, take usual half-length from both sides
                               for the second end, the same in opposite direction
    :type ends: str
    :param endlen: Minimal length to take from the end for determining the average in the
        first (last) point. If None, will be assigned depending on the method (see description of ends)
    :type endlen: int or None
    :param return_trend: Whether to return the trend in addition to the detrended data
    :type return_trend: bool
    :param keep_overall_level: If true, the total mean or median will be added to the data (that is preserved)
    :type keep_overall_level: bool
    :param remove_outliers:
    :type remove_outliers:
    :param nsig_outliers:
    :type nsig_outliers:
    :param detr_length_outliers:
    :type detr_length_outliers:
    :param remove_outliers_light:
    :type remove_outliers_light:
    :return: if not return_trend: data with subtracted mean, np.array
             if return_trend: data with subtracted mean, np.array; trend (moving mean), np.array
    :rtype:
    """
    assert ((method == 'mean') or (method == 'median')), 'Incorrect method specified'
    data = np.array(data)

    if mask is None:
        mask = isgood(data)

    mask = np.array(mask)
    assert len(data) == len(mask), 'Data and mask must be of the same length!'

    if remove_outliers:
        data1 = detrend_mov_ave(np.array(data), length=detr_length_outliers, mask=mask, method='median',
                                endmethod='mean',
                                ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False,
                                remove_outliers=False)
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask

    elif remove_outliers_light:
        outlier_mask = np.abs(data - np.nanmean(data[mask])) < nsig_outliers * np.nanstd(data[mask])
        mask *= outlier_mask

    trend = np.ones(len(data)) * np.nan
    if ends == 'asymmetric':
        if endlen is None:
            l4 = int(np.ceil(length / 16))
        elif endlen > int(length / 2):
            l4 = int(length / 2)
        else:
            l4 = endlen
        l2 = int(np.ceil(length / 2))
        if endmethod == 'median':
            for i in np.arange(0, l2):
                if (mask[0:i + int(l4 + (l2 - l4) / l2 * i)]).any():
                    m = np.nanmedian((data[0:i + int(l4 + (l2 - l4) / l2 * i)]
                                      )[mask[0:i + int(l4 + (l2 - l4) / l2 * i)]])
                else:
                    m = np.nanmedian(data[0:i + int(l4 + (l2 - l4) / l2 * i)])
                trend[i] = m
        elif endmethod == 'mean':
            for i in np.arange(0, l2):
                if (mask[0:i + int(l4 + (l2 - l4) / l2 * i)]).any():
                    m = np.nanmean((data[0:i + int(l4 + (l2 - l4) / l2 * i)]
                                    )[mask[0:i + int(l4 + (l2 - l4) / l2 * i)]])
                else:
                    m = np.nanmean(data[0:i + int(l4 + (l2 - l4) / l2 * i)])
                trend[i] = m
        if method == 'median':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                else:
                    m = np.nanmedian(data[i - l2:i + l2])
                trend[i] = m
        elif method == 'mean':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                else:
                    m = np.nanmean(data[i - l2:i + l2])
                trend[i] = m
        if endmethod == 'median':
            for i in np.arange(len(data) - l2, len(data)):
                if (mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]).any():
                    m = np.nanmedian((data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]
                                      )[mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]])
                else:
                    m = np.nanmedian(data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):])
                trend[i] = m
        elif endmethod == 'mean':
            for i in np.arange(len(data) - l2, len(data)):
                if (mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]).any():
                    m = np.nanmean((data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]
                                    )[mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]])
                else:
                    m = np.nanmean(data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):])
                trend[i] = m
    elif ends == 'symmetric':
        if endlen is None:
            l4 = int(np.ceil(length / 4))
        elif endlen > int(length / 2):
            l4 = int(length / 2)
        else:
            l4 = endlen
        l2 = int(np.ceil(length / 2))
        if endmethod == 'median':
            for i in np.arange(0, l4):
                if (mask[0:i + l4]).any():
                    m = np.nanmedian((data[0:i + l4])[mask[0:i + l4]])
                else:
                    m = np.nanmedian(data[0:i + l4])
                trend[i] = m
            for i in np.arange(l4, l2):
                if (mask[0:2 * i]).any():
                    m = np.nanmedian((data[0:2 * i])[mask[0:2 * i]])
                else:
                    m = np.nanmedian(data[0:2 * i])
                trend[i] = m
        elif endmethod == 'mean':
            for i in np.arange(0, l4):
                if (mask[0:i + l4]).any():
                    m = np.nanmean((data[0:i + l4])[mask[0:i + l4]])
                else:
                    m = np.nanmean(data[0:i + l4])
                trend[i] = m
            for i in np.arange(l4, l2):
                if (mask[0:2 * i]).any():
                    m = np.nanmean((data[0:2 * i])[mask[0:2 * i]])
                else:
                    m = np.nanmean(data[0:2 * i])
                trend[i] = m
        if method == 'median':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                else:
                    m = np.nanmedian(data[i - l2:i + l2])
                trend[i] = m
        elif method == 'mean':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                else:
                    m = np.nanmean(data[i - l2:i + l2])
                trend[i] = m
        if endmethod == 'median':
            for i in np.arange(len(data) - l2, len(data) - l4):
                if (mask[2 * i - len(data):]).any():
                    m = np.nanmedian((data[2 * i - len(data):])[mask[2 * i - len(data):]])
                else:
                    m = np.nanmedian(data[2 * i - len(data):])
                trend[i] = m
            for i in np.arange(len(data) - l4, len(data)):
                if (mask[i - l4:]).any():
                    m = np.nanmedian((data[i - l4:])[mask[i - l4:]])
                else:
                    m = np.nanmedian(data[i - l4:])
                trend[i] = m
        elif endmethod == 'mean':
            for i in np.arange(len(data) - l2, len(data) - l4):
                if (mask[2 * i - len(data):]).any():
                    m = np.nanmean((data[2 * i - len(data):])[mask[2 * i - len(data):]])
                else:
                    m = np.nanmean(data[2 * i - len(data):])
                trend[i] = m
            for i in np.arange(len(data) - l4, len(data)):
                if (mask[i - l4:]).any():
                    m = np.nanmean((data[i - l4:])[mask[i - l4:]])
                else:
                    m = np.nanmean(data[i - l4:])
                trend[i] = m
    # elif ends == 'cut':
    #     trend = np.ones(len(data) - 2*l2) * np.nan
    #     l2 = int(np.ceil(length / 2))
    #     if method == 'median':
    #         for i in np.arange(l2, len(data) - l2):
    #             if (mask[i - l2:i + l2]).any():
    #                 m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
    #             else:
    #                 m = np.nanmedian(data[i - l2:i + l2])
    #             trend[i] = m
    #     if method == 'mean':
    #         for i in np.arange(l2, len(data) - l2):
    #             if (mask[i - l2:i + l2]).any():
    #                 m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
    #             else:
    #                 m = np.nanmean(data[i - l2:i + l2])
    #             trend[i] = m
    else:
        raise AssertionError('Incorrect way to treat ends specified!')
    res = data - trend
    if keep_overall_level:
        if method == 'median':
            res = res - np.nanmedian(res) + np.nanmedian(data[mask])
        if method == 'mean':
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
    assert f0 <= fs / 2, 'The specified frequency is out of sampling range'
    # To use scipy.iirnotch we have to convert from out parameters to one accepted by it
    w0 = 2 * np.pi * f0 / fs  # Convert from physical frequency to the digital frequency
    if irl is not None:
        ampl = htol ** (1 / irl)  # amplitude of the pole (close to unity)
        dw = 2 * np.arctan(1 / ampl - 1)  # digital frequency width defined from it
        quality = w0 / dw / 2  # Quality factor, dimensionless parameter that characterizes
        # notch filter -3 dB bandwidth bw relative to its center frequency, quality = w0/bw
        # (for some reason, it works correctly with factor of 1/2 )
    else:  # if the impulse response length is not specified, use df
        if df is not None:
            quality = f0 / df / 2  # iQuality factor. In reality, quality gives w0/dw/2
            # for some reason, it works correctly with factor of 1/2
        else:
            quality = 30  # the default quality factor if no defining parameter is specified
    b, a = iirnotch(f0, quality, fs=fs)  # numerator and denominator polynomials of the IIR filter
    # if we want to notch the frequency not to zero, but to some minimal value,
    # we beed to shift and renormalize the numerator
    if min_val != 0:
        b = b * (1 - min_val) + min_val * a
    elif min_db is not None:
        min_val = 10 ** (min_db / 10)
        b = b * (1 - min_val) + min_val * a
    if out == 'sos':
        sos = tf2sos(b, a)  # convert to second order sections representation
        return sos
    elif out == 'ztk':
        z, p, k = tf2zpk(b, a)  # convert to zeros, poles, gain representation
        return z, p, k
    elif out == 'tf':  # transfer function representation
        return b, a
    else:
        raise AssertionError('Invalid output type specified!')


def mask_outliers_fun(data, mask=None, nsig0=Nsig_outliers_0, nsig1=Nsig_outliers_psd,
                      nsig_positive=Nsig_outliers_0_pos, nsig_negative=Nsig_outliers_0_neg,
                      detr_length=Detr_length_outliers, detr_method='median', detr_endmethod='mean',
                      return_symmetric_mask=False):
    """
    Masks entries which deviate from the general trend more than nsig standard deviations.
    Detrending is made to find the general trend.

    Creates:
    outlier_mask : np.array 1d of bool.
        True where data is good, False where there are outliers.

    :param data:
    :type data: ndarray 1d of float
    :param mask: True is good data, False is bad data.
        If None, will be measured to mask NaN.
    :type mask: ndarray 1d of bool
    :param nsig0: Maximum deviation from the trend in units of std of data (more extreme
        values will be marked as outliers). Is overriden by nsig_positive and nsig_negative.
    :type nsig0: float
    :param nsig1: Same as `nsig0` but for the second check if `return_symmetric_mask` is True
    :type nsig1:
    :param nsig_positive: Maximum positive deviation from the trend in units of std of data (more extreme
        values will be marked as outliers). Overrides nsig. If is None, will be replaced by nsig.
    :type nsig_positive: float
    :param nsig_negative: Maximum positive deviation from the trend in units of std of data (more extreme
        values will be marked as outliers). Dominates over nsig. If is None,
        will be replaced by nsig.
    :type nsig_negative:
    :param detr_length: Detrending length. Default is 128
    :type detr_length: int
    :param detr_method: The type of moving average using to define the trend. The default is 'median',
        because it is less sensitive to outliers. If None, data will not be detrended.
    :type detr_method: None or str : 'mean' or 'median'
    :param detr_endmethod: The type of moving average using to define the trend at the ends of the
        data. The default is 'mean', because it is less sensitive to small
        amount of data samples available in the ends
    :type detr_endmethod: str, 'mean' or 'median'
    :param return_symmetric_mask: Whether to make another check with `nsig1` symmetrically for positive and negative
    :type return_symmetric_mask: bool
    :return:
    :rtype:
    """

    data0 = np.array(data)
    nanmask = isgood(data)
    data0[~nanmask] = np.nanmean(data0)

    if mask is None:
        mask = nanmask

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

    if detr_method is not None:  # For now, we only do mean!
        mov_mean = moving_mean(data=np.array(data), nwind=detr_length, ngap=0, mask=mask1, nsamples_min=32,
                               remove_outliers=False, nsig_outliers=6,
                               return_mask=False, return_total_mean=False,
                               replace_ends=True, endmethod=detr_endmethod)
        assert isgood(mov_mean).all(), 'Moving mean has NaNs!'
        data1 = data - mov_mean
        # data1 = detrend_mov_ave(np.array(data0), length=detr_length, mask=mask1, method=detr_method,
        #                         endmethod=detr_endmethod,
        #                         ends=detr_ends, endlen=detr_endlen, return_trend=False, keep_overall_level=False,
        #                         remove_outliers=False)
    else:
        data1 = data0

    outlier_mask = np.abs(data1 - np.nanmean(data1[mask1])) < nsig0 * np.nanstd(data1[mask1])
    mask1 *= outlier_mask  # first time to remove big outliers that spoil the std

    data_no_mean = data1 - np.nanmean(data1[mask1])
    std_data = np.nanstd(data1[mask1])
    outlier_mask_p = data_no_mean < nsig_positive * std_data
    outlier_mask_n = data_no_mean > - nsig_negative * std_data
    if not return_symmetric_mask:
        return outlier_mask_p * outlier_mask_n
    else:
        outlier_mask = np.abs(data_no_mean) < nsig1 * std_data
        return outlier_mask_p * outlier_mask_n, outlier_mask


def clean_small_holes(mask, max_hole=Max_small_hole):
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
    where_mask = np.where(mask)[0]
    if len(where_mask) > 0:
        ind1 = where_mask[0]
        ind2 = where_mask[-1]
        if max_hole >= ind1 > 0:  # If there is a small hole in the beginning, fill
            mask[: ind1] = True
        if max_hole >= (len(mask) - 1 - ind2) > 0:  # If there is a small hole in the end, fill
            mask[ind2:] = True
        mask1 = np.concatenate((np.ones(ind1), np.array(mask[ind1:ind2 + 1]), np.ones(len(mask) - ind2 - 1))).astype(bool)
        begholes = np.where(mask1 * np.roll(~mask1, -1))[0] + 1
        endholes = np.where(mask1 * np.roll(~mask1, 1))[0]
        lenholes = endholes - begholes
        small_holes = np.where(lenholes <= max_hole)[0]
        for i in range(len(small_holes)):
            mask[begholes[small_holes[i]]:endholes[small_holes[i]]] = True  # mask with only big holes left
    return mask


def clean_small_non_holes(mask, max_non_hole=Max_small_hole):
    """
    Deletes from the mask holes of size <=max_small_hole,
    replacing False with True.

    Parameters
    ----------
    mask : numpy.array 1d of bool
        True is good data, False is bad data
    max_non_hole : int
        Maximum size of hole which is considered to be small

    Returns
    -------
    mask : np.array 1d of bool
        Mask in which small holes are removed, i.e. made True
    """
    mask = np.array(mask).astype(bool)
    mask_1 = clean_small_holes(mask, max_hole=max_non_hole)
    mask_1_inv = ~mask_1
    mask_1_inv = clean_small_holes(mask_1_inv, max_hole=max_non_hole)
    mask_1 = ~mask_1_inv
    return mask & mask_1


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
        Interpolated time. Equally-sampled, no NaNs.
    """
    time2 = copy.copy(time)
    if mask is None:
        mask = isgood(time2)
    dt = np.abs(np.nanmean((time2 - np.roll(time2, 1))[1:]))
    ind1 = (np.where(mask)[0])[0]
    ind2 = (np.where(mask)[0])[-1]
    # if there is a hole in the beginning or end, it will be interpolated later
    time1 = np.array(time2[ind1:ind2 + 1])
    mask1 = np.array(mask[ind1:ind2 + 1])
    begholes = np.where(mask1 * np.roll(~mask1, -1))[0]
    endholes = np.where(mask1 * np.roll(~mask1, 1))[0]
    for j in range(len(begholes)):
        time1[begholes[j] + 1:endholes[j]] = np.interp(
            np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
            np.array([time1[begholes[j]], time1[endholes[j]]]))[1:-1]
    time2[ind1:ind2 + 1] = time1
    if ind1 != 0:
        time2[0:ind1] = (-dt) * np.arange(1, ind1 + 1)[::-1] + time1[0]
    if ind2 != len(time2) - 1:
        time2[ind2 + 1:] = dt * np.arange(1, len(time2) - ind2) + time1[-1]
    return time2


#%% Old vetting (run 12)
def veto_combined(data, template, fn=0.01, maxdev_std=4., maxdev_ampl=None,
                  zero_ends_len_abs=3, zero_ends_len_rel=0.1,
                  return_amplitude=True, return_reason=True, inn_prod_min=0.2,
                  return_scores=False):
    """
    Checks the goodness of fit of data by the template by chi squared and maximum deviation.
    Fits the best amplitude for the template. Performs a chi squared test with this
    template and the maximum deviation test. The conditions for the fit to be good are:
        - The chi squared value is smaller than the point where the chi squared distribution with the number
          of degrees of freedom equal to the lengths of the template (truncated from extra zeros) is fn
        - Maximum relative deviation (in units of std) is smaller than maxdev
    If both tests are satisfied, returns True. Otherwise, False. If return_amplitude,
    the amplitude from the fitting is also returned in absolute units.
    Attention: NO NANS are allowed.

    Returs:
    If return_amplitude:
        If both chi2 and maxdev tests are satisfied, returns (True, amplitude).
        Otherwise (False, amplitude)
    Else:
        If both chi2 and maxdev tests are satisfied, returns True. Otherwise False

    :param data: Data to be compared with the template
    :type data: ndarray 1d of float
    :param template: Template (one-transit, positive-valued, not fft-shifted) to which the data
        should be compared (indices correspond to data indices). The template will be truncated
        to cut extra zeros (one extra zero from each side will be left to show that the template ended
        and reduce the score for peaks that continue beyond the template shape).
    :type template: numpy.array 1d of float
    :param fn: False negative probability for chi squared test with known amplitude.
    :type fn: float
    :param maxdev_std: Relative (in units of std) maximum deviation for the maximum deviation test.
    :type maxdev_std: float
    :param maxdev_ampl: Relative (in units of maximal value of fitted signal) maximum deviation
        for the maximum deviation test
    :type maxdev_ampl: float or None
    :param zero_ends_len_abs: Length of zeros to take at the end of the template from each side in addition
        to the non-zero region of the template. Is needed to make sure that the shape
        of the templite is intdeed fulfilled. Is dominant over zero_end_len_rel
    :type zero_ends_len_abs: int or None
    :param zero_ends_len_rel: If zero_end_len_abs is None, is used to calculate new zero_end_len_abs
        as [(length of nonzero template)*zero_end_len_rel]. This amount of
        Bins is used from each side
    :type zero_ends_len_rel: float
    :param return_amplitude: If True, the fitted amplitude of the transit will be returned together with veto
    :type return_amplitude: bool
    :param return_reason: If True, and if the transit is vetoed, the reason why it was vetoed will be returned
    :type return_reason: bool
    :param inn_prod_min:
    :type inn_prod_min:
    :param return_scores:
    :type return_scores:
    :return:
    :rtype:
    """
    from scipy.stats import chi2
    from scipy.optimize import minimize
    is_good = True
    reason = ''  # if the signal will be vetoed, the reason will be reported
    inds = np.where(~np.isclose(template, 0))[0]  # [[0,-1]]
    inds = np.array(inds, dtype=int)
    assert len(data) - len(inds) >= 4
    if len(data[0:inds[0]]) + len(data[inds[-1] + 1:]) != len(data) - len(inds):
        warnings.warn('Warning: template indices are not together, template is disconnected')

    # subtract linear trend
    inds_outside = np.array([i for i in range(len(data)) if i not in inds])
    data_outside = np.array(data[inds_outside])
    lin_sol = minimize(lambda x: np.sum((data_outside - x[0] - inds_outside * x[1]) ** 2), x0=np.array([0., 0.]))  # find linear trend
    data = data - lin_sol.x[0] - lin_sol.x[1] * np.arange(len(data))
    data_left = data[np.max([0, inds[0] - 20]):inds[0]]
    data_rignt = data[inds[-1]:np.min([inds[-1] + 10, len(data)])]
    if np.abs(np.mean(data_left) - np.mean(data_rignt)) >= 2 * ((np.var(data_left) * len(data_left)
                                                                 + np.var(data_rignt) * len(data_rignt)) / (
                                                                        len(data_left) + len(data_rignt))) ** 0.5:
        reason += 'step '
        is_good = False  # check that it is not a step instead of transit
    data_outside = np.array(data[inds_outside])  # already with trend subtracted
    std_dat = np.std(data_outside)

    if zero_ends_len_abs is None:
        zero_ends_len_abs = int(np.round(len(inds) * zero_ends_len_rel))
    assert zero_ends_len_abs >= 1, 'Incorrect zero_end_len_abs=' + str(zero_ends_len_abs) + '! Should be >=1'
    spare_ind_left = 0
    spare_ind_right = 0
    if inds[0] > 0:  # if possible, include some zeros in the ends where the template is zero to ensure that the shape goes to zero
        # noinspection PyTypeChecker
        spare_ind_left = min(zero_ends_len_abs, inds[0])
        inds = np.concatenate((np.arange(inds[0] - spare_ind_left, inds[0]), inds))
    if inds[-1] < len(template) - 1:
        # noinspection PyTypeChecker
        spare_ind_right = min(zero_ends_len_abs, len(template) - 1 - inds[-1])
        inds = np.concatenate((inds, np.arange(inds[-1] + 1, inds[-1] + 1 + spare_ind_right)))

    temp = np.array(template[inds])
    dat = np.array(data[inds])
    if (spare_ind_left > 0) and (spare_ind_right > 0):
        dat = dat - np.mean(np.concatenate((dat[0:spare_ind_left], dat[len(dat) - spare_ind_right:])))
        # -> although we detrended, we subtract local median
    elif spare_ind_left > 0:
        dat = dat - np.mean(dat[0:spare_ind_left])  # although we detrended, we subtract local median
    elif spare_ind_right > 0:
        dat = dat - np.mean(dat[len(dat) - spare_ind_right:])  # although we detrended, we subtract local median

    inn_prod = dat @ temp / ((dat @ dat) * (temp @ temp)) ** 0.5
    if np.abs(inn_prod) < inn_prod_min:
        reason += 'inner_product '
        is_good = False

    df = len(temp)
    x0 = chi2.ppf(1 - fn, df)
    i_peak = np.argmax(np.abs(dat))
    sol = minimize(lambda x: np.sum((dat - temp * x) ** 2), dat[i_peak] / np.max(temp))
    # sol = minimize(lambda x: np.sum((dat - temp * x) ** 2), np.max(np.abs(dat)) * np.sign(np.max(np.abs(dat))))  # weird initial cond
    ampl = sol.x[0]
    xi2 = sol.fun / std_dat ** 2
    noise = dat - temp * ampl

    maxdev_std_dat = np.max(np.abs(noise) / std_dat)
    maxdev_ampl_dat = np.max(np.abs(noise)) / np.max(np.abs(temp * ampl))

    if xi2 > x0:
        reason += 'chi_squared '
        is_good = False
    if maxdev_std_dat > maxdev_std:
        reason += 'maximal_deviation_std '
        is_good = False
    if (maxdev_ampl is not None) and (maxdev_ampl_dat > maxdev_ampl):
        reason += 'maximal_deviation_amplitude '
        is_good = False

    if return_scores:
        if return_amplitude:
            if return_reason:
                return is_good, [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod,
                                 inn_prod_min], ampl, reason
            else:
                return is_good, [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod,
                                 inn_prod_min], ampl
        else:
            if return_reason:
                return is_good, [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod,
                                 inn_prod_min], reason
            else:
                return is_good, [xi2, x0, maxdev_std_dat, maxdev_std, maxdev_ampl_dat, maxdev_ampl, inn_prod,
                                 inn_prod_min]
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


def get_step_statistic_old(data, mask=None):
    data = np.array(data)
    data = data - np.nanmean(data)

    if mask is None:
        mask = isgood(data)
    else:
        mask &= isgood(data)

    data[~mask] = 0

    # subtract trend
    deg = 3
    coefs = np.polyfit(np.arange(len(data))[mask], data[mask], deg=deg)
    x_dat = np.arange(len(data))
    data = data - np.sum([coefs[i] * x_dat**(deg - i) for i in range(len(coefs))], axis=0)

    mov_mean = moving_mean(data, nwind=(int(len(data) / 8) // 2) * 2, ngap=0, mask=mask, nsamples_min=(int(len(data) / 16) // 2) * 2,
                           remove_outliers=False, nsig_outliers=6, return_mask=False, return_total_mean=False)
    data = data - mov_mean
    data[~mask] = 0

    # slope, intercept = np.polyfit(np.arange(len(data))[mask], data[mask], deg=1)  # Linear fit: degree 1
    # data = data - intercept - slope * np.arange(len(data))
    # data[~mask] = 0

    step_model = np.arange(len(data))
    step_model = step_model / len(step_model)
    step_model = step_model - np.mean(step_model)
    step_convolved = irfft(rfft(data) * np.conj(rfft(step_model)), n=len(data))
    vars_convolved = irfft(rfft(np.ones(len(step_model))) * np.conj(rfft(step_model ** 2)), n=len(data))
    step_convolved = step_convolved / np.sqrt(vars_convolved)
    step_convolved = step_convolved / outlier_free_std(step_convolved[mask])
    return step_convolved


def veto_step_linear_fit(data, indices_center_exclude=None):
    if indices_center_exclude is None:
        indices_all = np.arange(len(data))
        indices_left = indices_all[:len(data) // 2]
        indices_right = indices_all[len(data) // 2:]
    else:
        indices_left = np.arange(0, indices_center_exclude[0])
        indices_right = np.arange(indices_center_exclude[-1] + 1, len(data))
    support_left = indices_left - len(data) // 2
    support_right = indices_right - len(data) // 2
    indices_all = np.concatenate((indices_left, indices_right))
    support_all = np.concatenate((support_left, support_right))
    fit_total = minimize(lambda x: np.sum((data[indices_all] - x[0] - support_all * x[1]) ** 2),
                         x0=np.array([0., 0.]))  # find linear trend from the left
    fit_left = minimize(lambda x: np.sum((data[indices_left] - x[0] - support_left * x[1]) ** 2),
                        x0=np.array([0., 0.]))  # find linear trend from the left
    fit_right = minimize(lambda x: np.sum((data[indices_right] - x[0] - support_right * x[1]) ** 2),
                         x0=np.array([0., 0.]))  # find linear trend from the left
    residuals_left = data[indices_left] - fit_left.x[0] - support_left * fit_left.x[1]
    residuals_right = data[indices_right] - fit_right.x[0] - support_right * fit_right.x[1]
    var_left = unb_var(residuals_left)
    var_right = unb_var(residuals_right)
    residuals_left_total = data[indices_left] - fit_total.x[0] - support_left * fit_total.x[1]
    residuals_right_total = data[indices_right] - fit_total.x[0] - support_right * fit_total.x[1]
    statistic = (- np.sum(residuals_left ** 2) / var_left - np.sum(residuals_right ** 2) / var_right
                 + np.sum(residuals_left_total ** 2) / var_left + np.sum(residuals_right_total ** 2) / var_right)
    # p_value = chi2.sf(statistic, loc=0, scale=1, df=2)
    # The following is the same but faster
    p_value = fast_chi2_01_sf(x=statistic, df=2)
    return p_value


# def veto_step_sliding(data, mask=None, window_exclude=None, min_len_to_process=3, remove_outliers=True,
#                       nsig_outliers=6, nwind_sliding=64, nsamples_min_sliding=32, ngap=0):
#     if mask is None:
#         mask = ~np.isnan(data)
#     else:
#         mask = np.array(mask)
#         mask[np.isnan(data)] = False
#
#     mov_av_left, mov_av_right, mov_var_left, mov_var_right = left_right_moving_mean_and_var(
#         data=data, nwind=nwind_sliding, ngap=ngap, mask=mask, nsamples_min=nsamples_min_sliding,
#         remove_outliers=remove_outliers, nsig_outliers=nsig_outliers, return_mask=False, return_total_mean_var=False)
#
#     if window_exclude is None:
#         indices_all = np.arange(len(data))
#         indices_left = indices_all[:index_split]
#         indices_right = indices_all[index_split:]
#     else:
#         indices_left = np.arange(0, index_split - (window_exclude + 1) // 2)
#         indices_right = np.arange(index_split + window_exclude // 2, len(data))
#     data_left = data[indices_left][mask[indices_left]]
#     data_right = data[indices_right][mask[indices_right]]
#
#     var_full = (mov_var_left / nwind_sliding + mov_var_right / nwind_sliding)
#     statistic = (mov_av_left - mov_av_right) ** 2 / var_full
#     p_value = chi2(loc=0, scale=1, df=1).sf(statistic)
#     return p_value


def veto_step_local(data, mask=None, index_split=None, window_exclude=None, min_len_to_process=3):
    if mask is None:
        mask = isgood(data)
    else:
        mask = np.array(mask)
        mask[~isgood(data)] = False
    if index_split is None:
        index_split = len(data) // 2
    if window_exclude is None:
        indices_all = np.arange(len(data))
        indices_left = indices_all[:index_split]
        indices_right = indices_all[index_split:]
    else:
        indices_left = np.arange(0, index_split - (window_exclude + 1) // 2)
        indices_right = np.arange(index_split + window_exclude // 2, len(data))
    data_left = data[indices_left][mask[indices_left]]
    data_right = data[indices_right][mask[indices_right]]

    if len(data_left) >= min_len_to_process and len(data_right) >= min_len_to_process:
        mean_left = np.mean(data_left)
        mean_right = np.mean(data_right)
        var_left = unb_var(data_left)
        var_right = unb_var(data_right)
        var_full = (var_left / len(data_left) + var_right / len(data_right))
        statistic = (mean_right - mean_left)**2 / var_full
        p_value = chi2(loc=0, scale=1, df=1).sf(statistic)
        return p_value
    else:
        return np.nan


#%% Current vetting

def get_distances_to_holes(big_hole_masks):
    """Finds distances to the edges of the nearby big holes in the mask.
       Before the beginning and after the end of the mask are also considered holes"""
    # big_hole_masks should be non-zero-padded
    if not hasattr(big_hole_masks[0], '__len__'):
        # if it is a list of masks
        big_hole_masks = [big_hole_masks]
        had_len = False
    else:
        had_len = True

    dists_to_edges = []
    for big_mask in big_hole_masks:
        begpieces, endpieces = get_big_hole_edges_indices(big_mask)
        dists = [find_closest_edge_dist(ind=ind, begpieces=begpieces, endpieces=endpieces) for ind in range(len(big_mask))]
        dists_to_edges.append(np.array(dists))

    if not had_len:
        dists_to_edges = dists_to_edges[0]
    return dists_to_edges


def get_big_hole_edges_indices(big_mask):
    """Finds the indices of the beginnings and ends of the pieces separated by big holes"""
    ind_i = (np.where(big_mask)[0])[0]  # start where there is the first valid point
    ind_f = (np.where(big_mask)[0])[-1] + 1  # end with the last valid point
    mask0 = np.array(big_mask[ind_i:ind_f])  # mask of the working region
    endpieces = np.where((~mask0) * np.roll(mask0, 1))[0] + ind_i  # ends of pieces split by holes
    endpieces = np.concatenate((endpieces, [ind_f]))
    begpieces = np.where(mask0 * np.roll(~mask0, 1))[0] + ind_i  # beginnings of pieces split by holes
    begpieces = np.concatenate(([ind_i], begpieces))
    return begpieces, endpieces


def find_closest_edge_dist(ind, begpieces, endpieces):
    """Finds the distance to the closest edge of the non-hole piece."""
    dists = np.concatenate((np.abs(begpieces - ind), np.abs(endpieces - ind)))
    return np.min(dists)


def get_step_statistic(data, mask=None, mask_fit=None, wf_td=None, step_width=768,
                       interpolate=False, interp_smoothing_length=10):
    data = np.array(data)
    data = data - np.nanmean(data)

    if mask is None:
        mask = isgood(data)
    else:
        mask = np.array(mask, dtype=bool)
        mask &= isgood(data)

    if mask_fit is None:
        mask_fit = mask

    data[~mask] = 0
    # subtract linear trend
    slope, intercept = np.polyfit(np.arange(len(data))[mask_fit], data[mask_fit], deg=1)  # Linear fit: degree 1
    data = data - intercept - slope * np.arange(len(data))

    if interpolate:
        data = interpolate_over_mask(flux=data, mask=mask, interp_method='linear',
                                     ends='continue', mitigate_hole_ends=True if ((interp_smoothing_length is not None) or (
                                                                                   interp_smoothing_length == 0)) else False,
                                     endhole_method='median', mit_length=interp_smoothing_length)
    else:
        data[~mask] = 0

    # Making step model
    if step_width is None:
        step_model = np.arange(len(data))
        step_model = step_model / len(step_model)
        step_model = step_model - np.mean(step_model)
    else:
        step_model_piece = np.arange(step_width)
        step_model_piece = step_model_piece / len(step_model_piece)
        step_model_piece = step_model_piece - np.mean(step_model_piece)
        step_model = np.concatenate((step_model_piece[:len(step_model_piece) // 2], np.zeros(len(data) - len(step_model_piece)),
                                     step_model_piece[len(step_model_piece) // 2:]))

    if wf_td is None:
        step_convolved = irfft(rfft(data) * np.conj(rfft(step_model)), n=len(data))
    else:
        step_convolved = irfft(rfft(data) * rfft(wf_td) ** 2 * np.conj(rfft(step_model)), n=len(data))

    step_convolved = step_convolved / outlier_free_std(step_convolved[mask])
    return step_convolved


def veto_parts(data_wh_fd, template_wh_full, indices_padded, mask_padded=None,
               tlen=None, fn=0.01, n_parts=4, min_bins_part=4, nperseg_psd=Nperseg_psd,
               splitting='snr', lobe_hlen=4, return_p_values=False):
    """
    Splits the template into parts and tests between two cases: all parts are of the same amplitude
    or every part has a different amplitude.
    :param data_wh_fd: Whitened data in the Fourier domain for the entire quarter.
    :type data_wh_fd: numpy.ndarray 1d of float
    :param template_wh_full: Whitened template in time domain rolled in such a way that the
        center of the transit is at zero index.
    :type template_wh_full: numpy.ndarray 1d of float
    :param indices_padded: The position where the transit center is located. Should be in the padded units,
        meaning it is counted from the beginning of the padded arrays.
    :type indices_padded: numpy.ndarray 1d of int
    :param mask_padded: Goodness mask corresponding to `data_wh_fd` but in time domain.
    :type mask_padded: numpy.ndarray 1d of bool
    :param tlen: Length of the non-zero part of the template (usually set by the whitening filter, can be
        also affected by the notch filter). If None, will be measured.
    :type tlen: int or None
    :param fn: False negative rate that is needed to ve achieved in the veto. Calculated from chi2 distribution
    :type fn: float
    :param n_parts: Number of parts to which the template should be split (expecting to have the same SNR in them)
    :type n_parts: int
    :param min_bins_part: Minimal number of bins that should be in every part.
        If less, number of parts will be decreased
    :type min_bins_part: int
    :param nperseg_psd: size of the slice that was used for PSD measurement in the pipeline
    :type nperseg_psd: int
    :param splitting: According to what to split the template: equal parts in snr or in just take the main
        lobe of the template, for example if it is too short and can be tested in a usual chi-square veto
        and not in parts veto.
    :type splitting: str: 'snr' or 'lobe'
    :param lobe_hlen: half-length of the lobe if `splitting`=='lobe' or if the number of parts is only one.
    :type lobe_hlen: int
    :param return_p_values: If `True`, will return p-values of the vetoes.
    :type return_p_values: bool
    :return: is_goods, `True` where the trigger passed the veto, and `False` otherwise.
        Same size as the input `indices_padded`
    :rtype: numpy.ndarray of bool. Same size as the input `indices_padded`
    """
    # Indices are assumed to be given in the padded coordinates!
    template_wh_full = np.array(template_wh_full)  # this is template in Time domain
    p_values = None
    if mask_padded is None:
        mask_padded = ~np.ones(len(template_wh_full)).astype(bool)  # data is given in Fourier, do there will be no nans in time
    if splitting == 'snr':
        # PREPARING TEMPLATE
        if tlen is None:
            temp_smthnd = np.abs(template_wh_full - template_wh_full[len(template_wh_full) // 2])
            # -> the value in the middle is supposed to be close to zero, but sometimes no
            temp_smthnd = (temp_smthnd + np.roll(temp_smthnd, 1) + np.roll(temp_smthnd, -1)) / 3
            # -> averaging the rolled template making it smooth
            inds_small = np.where((temp_smthnd / np.max(temp_smthnd) < 0.001))[0]  # choosing where the template is close to zero
            inds_small = inds_small[np.where((inds_small > nperseg_psd) & (inds_small <= len(template_wh_full) - nperseg_psd))[0]]
            # -> if suddenly there is zero in the beginning, ignore
            tlen = inds_small[0] + (len(temp_smthnd) - inds_small[-1])  # taking parts where the template is not close to zero
            # this weird procedure is needed because we are dealing with the whitened template, and its shape can be complex,
            # so at some point it can reach zero, but it does not mean that it is the end of the transit.
            # We are smoothening it, so that if it is zero only in one point,
            # the smoothened one will be nonzero.
            # then, we are setting a threshold and selecting everything non-zero
            # note the rolling! the transit center should be in the zeroth index!
        template_wh_full[tlen // 2: len(template_wh_full) - tlen // 2] = 0.
        template_wh_full = template_wh_full / np.linalg.norm(template_wh_full)
        # DETERMINING NUMBER OF SPLITTING PARTS
        chi2_temp_indices = _split_template_by_snr(template_wh_full=template_wh_full, n_parts=n_parts, min_bins_part=min_bins_part)
        n_parts = len(chi2_temp_indices) - 1
        # FOR TOO SHORT TEMPLATES:
        # if the template is so short that there is only one part, we will just make one vetting on it and not parts.
        if n_parts <= 1:
            if not return_p_values:
                is_good = _veto_lobe_time(data_wh_fd=data_wh_fd, template_wh_full=template_wh_full,
                                          mask_padded=mask_padded, indices_padded=indices_padded, fn=fn, lobe_hlen=lobe_hlen)
            else:
                p_values, is_good = _veto_lobe_time(data_wh_fd=data_wh_fd, template_wh_full=template_wh_full,
                                                    mask_padded=mask_padded, indices_padded=indices_padded, fn=fn,
                                                    lobe_hlen=lobe_hlen, return_p_values=return_p_values)
        # FOR LONG ENOUGH TEMPLATES:
        else:
            if not return_p_values:
                is_good = _split_and_match_veto_parts_time(data_wh_fd=data_wh_fd, indices_padded=indices_padded,
                                                           template_wh_full=template_wh_full, chi2_temp_indices=chi2_temp_indices, fn=fn)
            else:
                p_values, is_good = _split_and_match_veto_parts_time(data_wh_fd=data_wh_fd, indices_padded=indices_padded,
                                                                     template_wh_full=template_wh_full,
                                                                     chi2_temp_indices=chi2_temp_indices, fn=fn,
                                                                     return_p_values=return_p_values)
            if n_parts >= 3:  # at least 3 parts so that there can be less
                if n_parts % 2 == 1:
                    n_parts_additional = 2
                else:
                    n_parts_additional = 3
                chi2_temp_indices_additional = _split_template_by_snr(template_wh_full=template_wh_full, n_parts=n_parts_additional,
                                                                      min_bins_part=min_bins_part)
                if not return_p_values:
                    is_good_additional = _split_and_match_veto_parts_time(data_wh_fd=data_wh_fd, indices_padded=indices_padded,
                                                                          template_wh_full=template_wh_full,
                                                                          chi2_temp_indices=chi2_temp_indices_additional, fn=fn)
                else:
                    p_values_additional, is_good_additional = _split_and_match_veto_parts_time(
                        data_wh_fd=data_wh_fd, indices_padded=indices_padded, template_wh_full=template_wh_full,
                        chi2_temp_indices=chi2_temp_indices_additional, fn=fn, return_p_values=return_p_values)
                    p_values = np.nanmin(np.array([p_values, p_values_additional]), axis=0)
                is_good = is_good & is_good_additional
    elif splitting == 'lobe':
        if not return_p_values:
            is_good = _veto_lobe_time(data_wh_fd=data_wh_fd, template_wh_full=template_wh_full,
                                      mask_padded=mask_padded, indices_padded=indices_padded, fn=fn, lobe_hlen=lobe_hlen)
        else:
            p_values, is_good = _veto_lobe_time(data_wh_fd=data_wh_fd, template_wh_full=template_wh_full,
                                                mask_padded=mask_padded, indices_padded=indices_padded, fn=fn,
                                                lobe_hlen=lobe_hlen, return_p_values=return_p_values)
    else:
        raise ValueError("`splitting` must be 'snr' or 'lobe'!")
    if not return_p_values:
        return is_good
    else:
        return p_values, is_good


def _split_template_by_snr(template_wh_full, n_parts, min_bins_part=4):
    cs = np.cumsum(template_wh_full ** 2)
    # noinspection PyTypeChecker
    chi2_temp_indices = [0] + list(np.searchsorted(cs / cs[-1], (np.arange(n_parts) / n_parts)[1:])) + [len(template_wh_full)]
    # # this was len split, I'm removing it because I don't want to fit zeros.
    # nonzero_inds = np.nonzero(template_wh_full)[0]
    # chi2_temp_indices = list(nonzero_inds[(len(nonzero_inds) * np.arange(n_parts) / n_parts).astype(int)]
    #                          ) + [nonzero_inds[-1] + 1]
    chi2_temp_indices = list(set(chi2_temp_indices))  # remove repetitions
    chi2_temp_indices.sort()
    n_parts = len(chi2_temp_indices) - 1
    # Want every piece to be at least min_bins_part bins:
    while ((chi2_temp_indices - np.roll(chi2_temp_indices, 1))[1:] < min_bins_part).any():
        n_parts -= 1
        cs = np.cumsum(template_wh_full ** 2)
        chi2_temp_indices = [0] + list(np.searchsorted(cs / cs[-1], (np.arange(n_parts) / n_parts)[1:])) + [len(template_wh_full)]
    return chi2_temp_indices


def get_template_by_parts_snr(template_wh_td, n_parts, min_bins_part=4):
    chi2_temp_indices = _split_template_by_snr(template_wh_full=template_wh_td, n_parts=n_parts, min_bins_part=min_bins_part)

    templates_parts = np.zeros((len(chi2_temp_indices) - 1, len(template_wh_td)))

    for i in range(len(chi2_temp_indices) - 1):
        templates_parts[i][chi2_temp_indices[i]: chi2_temp_indices[i + 1]] = template_wh_td[
                                                                             chi2_temp_indices[i]: chi2_temp_indices[i + 1]]
    return templates_parts


def _split_and_match_veto_parts_time(data_wh_fd, indices_padded, template_wh_full, chi2_temp_indices, fn,
                                     return_p_values=False):
    n_parts = len(chi2_temp_indices) - 1
    templates_parts = np.zeros((n_parts, len(template_wh_full)))
    for i in range(n_parts):
        templates_parts[i][chi2_temp_indices[i]: chi2_temp_indices[i + 1]] = template_wh_full[chi2_temp_indices[i]:
                                                                                              chi2_temp_indices[i + 1]]
    norms_templates_parts = np.array([np.linalg.norm(templates_parts[i]) for i in range(len(templates_parts))])
    templates_parts = rfft(templates_parts, axis=1)
    scores = []
    for i in range(n_parts):
        stat = irfft(np.conjugate(templates_parts[i]) * data_wh_fd)
        stat = stat / norms_templates_parts[i]  # outlier_free_std(stat[mask_padded], n_iterations=2, nsig=3)
        # -> this is supposed to ensure the templ normalization, although formally this biases amplitude
        # stat = stat[zero_padding_ends[0]: len(stat) - zero_padding_ends[1]]  # indices are provided in padded coordinates now!
        # this will give a biased amplitude, but since we only look at the amplitudes differences, maybe it is not so bad?
        scores.append(stat)
    stat_full = irfft(np.conjugate(rfft(template_wh_full)) * data_wh_fd)
    # -> norm is one, templ and data were norm
    stat_full = stat_full[indices_padded]
    scores = np.array(scores)[:, indices_padded]

    ampl_scores_parts = np.abs(scores - stat_full * norms_templates_parts.reshape(len(norms_templates_parts), 1))
    # The amplitude difference score in the normalized distribution is this:
    # abs(scores_parts / norms_parts - score_full / norm_full) * norms_parts
    # note norm_full = 1
    p_values_parts = norm.sf(ampl_scores_parts, scale=(1 - 1 / n_parts) ** 0.5, loc=0) * 2
    # TODO abs
    is_good_parts = np.sum((p_values_parts >= fn / n_parts).astype(float), axis=0) >= n_parts // 2

    scores = np.sum(scores ** 2, axis=0) - stat_full ** 2
    # p_values = chi2.sf(scores, loc=0, scale=1, df=n_parts - 1)
    # The following line is the same just faster:
    p_values = fast_chi2_01_sf(x=scores, df=n_parts - 1)
    is_good = np.array(p_values) >= fn
    is_good = is_good & is_good_parts
    # See in the veto_parts_fourier regarding the scale. The following empirical scale can be thought of:
    # measured_scale = (np.var(scores[mask_padded]) / (2 * (n_parts - 1))) ** 0.5
    # But as mentioned in veto_parts_fourier, the norm is already one. I am not completely sure,
    # but I think I checked it, and it was indeed fine.
    if not return_p_values:
        return is_good
    else:
        return p_values, is_good


def _veto_lobe_time(data_wh_fd, template_wh_full, mask_padded, indices_padded, fn, lobe_hlen, return_p_values=False):
    hlen = lobe_hlen
    data_wh_full = irfft(data_wh_fd)
    template_for_fit = np.array(template_wh_full)
    template_for_fit[hlen: len(template_for_fit) - hlen] = 0
    template_for_fit = template_for_fit / np.linalg.norm(template_for_fit)
    template_cut_td = np.concatenate((template_for_fit[-hlen:], template_for_fit[: hlen]))
    template_for_fit = rfft(template_for_fit)
    stat = irfft(np.conjugate(template_for_fit) * data_wh_fd)
    stat = stat / outlier_free_std(stat[mask_padded], n_iterations=2, nsig=3)
    # -> this is supposed to ensure the template normalization, although formally this biases amplitude
    p_values = []
    for ind in indices_padded:
        residuals = (data_wh_full[ind - hlen: ind + hlen] - template_cut_td * stat[ind])
        residuals = residuals[mask_padded[ind - hlen: ind + hlen]]
        score = np.sum(residuals ** 2)
        df = len(residuals) - 1
        if df > 1:
            # p_values.append(chi2.sf(score, loc=0, scale=1, df=df))
            # The following line is the same just faster:
            p_values.append(fast_chi2_01_sf(x=score, df=df))
        else:
            p_values.append(0.)
    is_good = np.array(p_values) >= fn
    if not return_p_values:
        return is_good
    else:
        return p_values, is_good


def veto_parts_fourier(data_wh_fd, template_wh_fd, indices_padded, mask_padded=None, n_parts=3, fn=0.01,
                       wind_len_overlap=64, nperseg_psd=Nperseg_psd, return_p_values=False,
                       high_pass=False):
    """

    The template in the Fourier domain is cut into several parts which undergo windowing in order not to be too long in the time
    domain. The minimal size of a piece is bounded by the window size. If the piece is too big, the number of pieces is decreased.

    :param data_wh_fd: Whitened data in the Fourier domain for the entire quarter.
        Data must be really white, including the constant multiple!
    :type data_wh_fd: numpy.ndarray 1d of float
    :param template_wh_fd: Whitened template in Fourier domain rolled in time in such a way that the
        center of the transit is at zero index. Template should be normalized to have unit norm in the time domain!
        Not to waste time and renormalize it inside the functions!
        Warning: data can be blue because we oversuppress the low frequencies.
    :type template_wh_fd: numpy.ndarray 1d of float
    :param indices_padded: The position where the transit center is located. Should be in the padded units,
        meaning it is counted from the beginning of the padded arrays.
    :type indices_padded: numpy.ndarray 1d of int
    :param mask_padded: Goodness mask corresponding to `data_wh_fd` but in time domain.
    :type mask_padded: numpy.ndarray 1d of bool
    :param n_parts: Number of parts to which the template should be split (expecting to have the same SNR in them)
    :type n_parts: int
    :param fn: False negative rate that is needed to ve achieved in the veto. Calculated from chi2 distribution
    :type fn: float
    :param wind_len_overlap: Must be even! Window length of the overlapping parts into which the template will be split.
           Hann window will be used. All this is to make parts have finite support in the time domain.
    :type wind_len_overlap: int
    :param nperseg_psd: size of the slice that was used for PSD measurement in the pipeline
    :type nperseg_psd: int
    :param return_p_values: If True, the p-values of the veto will be returned together with veto results (is_good).
    :type return_p_values: bool
    :param high_pass:
    :type high_pass:
    :return: is_good, `True` where the trigger passed the veto, and `False` otherwise.
        Same size as the input `indices_padded`
    :rtype: numpy.ndarray of bool. Same size as the input `indices_padded`

    """
    data_wh_fd = np.array(data_wh_fd)
    template_wh_fd_full = np.array(template_wh_fd)
    # when the detrending was performed, the low freqs were removed, so we should not reject a trigger is something is wrong with them:
    template_wh_fd_full[:int(len(template_wh_fd_full) / nperseg_psd * 3 / 2)] = 0.
    template_wh_fd_full = template_wh_fd_full / np.linalg.norm(irfft(template_wh_fd_full))
    # -> must have been normalized initially, but we detrended it

    if high_pass:
        from auxiliary_functions import filter_data_high_pass
        data_wh_fd = filter_data_high_pass(data=None, data_fd=np.array(data_wh_fd), dt=1.,
                                           dt_pass=nperseg_psd * 1.,
                                           in_fourier=True, return_cutoff_index=False, renormalize_to_tail=False)
        template_wh_fd_full = filter_data_high_pass(data=None, data_fd=np.array(template_wh_fd_full), dt=1.,
                                                    dt_pass=nperseg_psd * 1.,
                                                    in_fourier=True, return_cutoff_index=False, renormalize_to_tail=False)

    wind_len = wind_len_overlap  # must be even!
    half_wind_len = wind_len // 2
    window = 0.5 * (1 - np.cos(0.5 * 2 * np.pi * np.arange(wind_len) / wind_len))  # Hann window
    window = 1. - window / np.max(window)

    cs = np.cumsum(template_wh_fd_full ** 2)
    chi2_temp_indices = [0] + list(np.searchsorted(cs / cs[-1], (np.arange(n_parts) / n_parts)[1:])) + [len(template_wh_fd_full)]
    while ((chi2_temp_indices - np.roll(chi2_temp_indices, 1))[1:] < wind_len).any():  # want every piece to be at least to support window
        n_parts -= 1
        cs = np.cumsum(template_wh_fd_full ** 2)
        chi2_temp_indices = [0] + list(np.searchsorted(cs / cs[-1], (np.arange(n_parts) / n_parts)[1:])) + [len(template_wh_fd_full)]
    n_parts = len(chi2_temp_indices) - 1

    if n_parts <= 1:
        is_goods = np.ones(len(indices_padded)).astype(bool)
        if not return_p_values:
            return is_goods
        else:  # p-values are set to be unity if all is good
            return np.ones(len(indices_padded)).astype(bool), is_goods

    ##########################################################################################
    templates_parts = np.zeros((n_parts, len(template_wh_fd_full)), dtype=np.complex64)  # here it is all in the fourier domain
    i = 0
    templates_parts[i][chi2_temp_indices[i]: chi2_temp_indices[i + 1] + half_wind_len
                       ] = template_wh_fd_full[chi2_temp_indices[i]: chi2_temp_indices[i + 1] + half_wind_len]
    templates_parts[i][chi2_temp_indices[i + 1] - half_wind_len: chi2_temp_indices[i + 1] + half_wind_len] *= window

    for i in range(1, n_parts - 1):
        templates_parts[i][chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i + 1] + half_wind_len
                           ] = template_wh_fd_full[chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i + 1] + half_wind_len]
        templates_parts[i][chi2_temp_indices[i + 1] - half_wind_len: chi2_temp_indices[i + 1] + half_wind_len] *= window
        templates_parts[i][chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i] + half_wind_len] *= (1 - window)
    i = n_parts - 1
    templates_parts[i][chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i + 1]
                       ] = template_wh_fd_full[chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i + 1]]
    templates_parts[i][chi2_temp_indices[i] - half_wind_len: chi2_temp_indices[i] + half_wind_len] *= (1 - window)
    ##########################################################################################
    templates_parts_td = irfft(templates_parts, axis=1)
    templ_prod_matrix = np.zeros((n_parts, n_parts))
    for i in range(n_parts):
        for j in range(n_parts):
            prodct = templates_parts_td[i] @ templates_parts_td[j]
            templ_prod_matrix[i, j] = prodct
            if j != i:
                templ_prod_matrix[j, i] = prodct
    templ_prod_matrix_inv = np.linalg.inv(templ_prod_matrix)
    ##########################################################################################
    scores = np.zeros_like(templates_parts_td)
    for i in range(n_parts):
        stat = irfft(np.conjugate(templates_parts[i]) * data_wh_fd)
        # since the noise may be too blue, the distribution may not be with unit variance.
        # -> so we force it to have the same variance as it should be according to the norm
        stat = stat / outlier_free_std(stat[mask_padded], n_iterations=2, nsig=3, renormalize=True) * (templ_prod_matrix[i, i])**0.5
        scores[i] = stat

    stat_full = irfft(np.conjugate(template_wh_fd_full) * data_wh_fd)
    stat_full = stat_full / outlier_free_std(stat_full[mask_padded], n_iterations=2, nsig=3, renormalize=True)
    # -> the template norm was 1, so no need to multiply by it
    # scores = np.array([scores[:, j] @ templ_prod_matrix_inv @ scores[:, j] for j in range(len(scores[0]))])
    # -> this one was too slow, the following one is faster
    scores = np.einsum('ij,jk,ik->i', scores.T, templ_prod_matrix_inv, scores.T)
    # -> this has distr chi2(n_parts), and it has var of trace of unity matrix, which is n_parts
    scores = scores[indices_padded]
    stat_full = stat_full[indices_padded]
    scores = scores - stat_full ** 2
    # they all have the same names to save some memory (if it works like that...)
    # We already renormalized the stuff, so the scale should be 1 here:
    # p_values = chi2.sf(scores, loc=0, scale=1., df=n_parts - 1)
    # -> the following is same but faster:
    p_values = fast_chi2_01_sf(x=scores, df=n_parts - 1)
    is_good = np.array(p_values) >= fn
    if not return_p_values:
        return is_good
    else:
        return p_values, is_good


def veto_threshold_snr_fun(snr):
    """An empirical function that introduces variable snr-dependent threshold for
       veto-parts p-value. Models it as a linear step focused on the snr values of interest.
       Assumes that all the snrs considered are negative, because the empirical investigation was only done for those."""
    min_threshold = 10. ** (-2.)
    lower_snr_bound = 4
    max_threshold = 10. ** (-1.5)
    upper_snr_bound = 5
    if not hasattr(snr, '__len__'):
        abs_snr = np.abs(float(snr))
        if abs_snr < lower_snr_bound:
            return min_threshold
        elif abs_snr < upper_snr_bound:
            return ((max_threshold - min_threshold) / (upper_snr_bound - lower_snr_bound)
                    * (abs_snr - lower_snr_bound) + min_threshold)
        else:
            return max_threshold
    else:
        abs_snr = np.abs(np.array(snr, dtype=float))
        thresholds = np.ones_like(abs_snr)
        thresholds[abs_snr < lower_snr_bound] = min_threshold
        thresholds[(abs_snr >= lower_snr_bound) & (abs_snr < upper_snr_bound)] = (
                (max_threshold - min_threshold) / (upper_snr_bound - lower_snr_bound)
                * (abs_snr[(abs_snr >= lower_snr_bound) & (abs_snr < upper_snr_bound)]
                   - lower_snr_bound) + min_threshold)
        thresholds[abs_snr >= upper_snr_bound] = max_threshold
        return thresholds


#%%

def score_correction(scores_i, return_function=False, two_sided_distr=True):
    scores_i = np.array(scores_i)
    if two_sided_distr:
        argsorted = np.argsort(scores_i)
        argargsorted = np.argsort(argsorted)
        sorted_scores = scores_i[argsorted]
        scores_c = 2 ** 0.5 * erfinv(2 * (np.arange(len(sorted_scores)) + 0.5) / len(sorted_scores) - 1)
        scores_f = scores_c[argargsorted]
    else:
        signs_scores = np.sign(scores_i)
        abs_scores = np.abs(scores_i)
        argsorted = np.argsort(abs_scores)
        argargsorted = np.argsort(argsorted)
        sorted_scores = abs_scores[argsorted]
        scores_c = 2 ** 0.5 * erfinv((np.arange(len(sorted_scores)) + 0.5) / len(sorted_scores))
        scores_f = scores_c[argargsorted] * signs_scores
    if return_function:
        funct = interp1d(sorted_scores, scores_c)
        return scores_f, funct
    else:
        return scores_f


def cdf_max_norm(x, n=2, loc=0, scale=1):
    if not hasattr(loc, '__len__') and not hasattr(scale, '__len__'):
        return norm(loc=loc, scale=scale).cdf(x) ** n
    else:
        loc = loc * np.ones(n)
        scale = scale * np.ones(n)
        prod = 1.
        for i in range(n):
            prod = prod * norm(loc=loc[i], scale=scale[i]).cdf(x)
    return prod


def sf_max_norm(x, n=2, loc=0, scale=1):
    """
    it returns just 1 - cdf
    """
    if not hasattr(loc, '__len__') and not hasattr(scale, '__len__'):
        return 1 - norm(loc=loc, scale=scale).cdf(x) ** n
    else:
        loc = loc * np.ones(n)
        scale = scale * np.ones(n)
        prod = 1.
        for i in range(n):
            prod = prod * norm(loc=loc[i], scale=scale[i]).cdf(x)
    return 1 - prod


def pdf_max_norm(x, n=2, loc=0, scale=1):
    if not hasattr(loc, '__len__') and not hasattr(scale, '__len__'):
        return n * norm(loc=loc, scale=scale).cdf(x) ** (n - 1) * norm(loc=loc, scale=scale).pdf(x)
    else:
        loc = loc * np.ones(n)
        scale = scale * np.ones(n)
        ssum = 0
        for j in range(n):
            prod = norm(loc=loc[j], scale=scale[j]).pdf(x)
            for i in range(n):
                if i != j:
                    prod = prod * norm(loc=loc[i], scale=scale[i]).cdf(x)
            ssum = ssum + prod
    return sum


def unequal_amplitude_veto(stats_raw, stat_vars=None, negative=True):
    stats_raw = np.array(stats_raw)

    if stat_vars is None:
        stat_vars = np.ones(len(stats_raw))
    else:
        stat_vars = np.array(stat_vars)

    mask = (stat_vars != 0) & isgood(stats_raw)
    stats_raw = stats_raw[mask]
    stat_vars = stat_vars[mask]

    if negative:
        stats_raw = - stats_raw

    n_transits = len(stat_vars)
    h = np.ones(n_transits) * stat_vars
    denoms = h * h / stat_vars
    stats_stats = stats_raw * h / stat_vars
    # ampls = stats_stats / denoms
    # # The following can be used if we want to use sets of 2, that is to summarize 2 indices as max
    # # But this is too many dofs, so the test is weaker
    # import itertools
    # combs_inds = list(itertools.combinations(set(range(n_transits)), n_max))
    # combs_inds = [list(el) for el in combs_inds]
    combs_inds = np.arange(n_transits).reshape((n_transits, 1))
    combs_inds_other = [[i for i in range(n_transits) if i not in ci] for ci in combs_inds]
    ampls_combs = np.array([np.nansum(stats_stats[ci]) / np.nansum(denoms[ci]) for ci in combs_inds])
    ampls_other = np.array([np.nansum(stats_stats[io]) / np.nansum(denoms[io]) for io in combs_inds_other])
    vars_combs = np.array([1 / np.nansum(denoms[ci]) for ci in combs_inds])
    vars_other = np.array([1 / np.nansum(denoms[io]) for io in combs_inds_other])
    scales = (vars_combs + vars_other) ** 0.5
    scores = (ampls_combs - ampls_other) / scales
    i_inds_max = np.argmax(scores)
    score = scores[i_inds_max]
    p_val = (1 - cdf_max_norm(score, n=n_transits, loc=0, scale=1))
    return p_val


def fstr(f, nzdig=2, maxdec=5):  # nicely formatted string from numbers
    """
    Takes float and returns string which is more convenient for printing, not having 16 digits or so.
    nzdig : number of non-zero digits to leave
    maxdec : maximal number of decimal digits to have
    """
    f = float(f)
    n = None
    for i in range(maxdec - nzdig + 2):
        if (f * 10 ** float(i)) // 1 > 0:
            n = i
            break

    if n is not None:
        ndec = n + nzdig - 1
        for i in np.arange(n + nzdig - 1, -1, -1):
            if ((f * 10 ** float(i)) // 1) % 10 == 0:
                ndec = i - 1
            else:
                break

        str_to_format = "{:." + str(ndec) + "f}"
        return str_to_format.format(f)
    else:
        str_to_format = "{:." + str(nzdig) + "e}"
        return str_to_format.format(f)


#######################################################################################################################
# Functions for reshuffled time slides #
#######################################################################################################################
def get_pieces_indices(mask, max_hole=Max_hole_pieces_scrampling, max_piece=Max_piece_scrambling, min_piece=Min_piece_scrambling,
                       random_seed=None, random_state=None):
    """
    Produces a way to split an array into pieces of lengths from `min_piece` till `max_piece`. The pieces are primarily taken
    as parts between the holes in the `mask` whose size is more than `max_hole`. If these pieces exceed `max_piece`, they will be
    cut randomly, so that the length of the part is between `min_piece` and `max_piece`.
    :param mask: Mask in which holes will be considered breaks between the pieces
    :type mask: ndarray of bool
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
    :return: inds_pieces, beginnings of the pieces and next indices after the ends of the pieces.
    :rtype: list[list[int,int]]
    """
    mask = np.array(mask).astype(bool)
    if random_state is not None:
        np.random.set_state(random_state)
    elif random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(None)
    ind1 = (np.where(mask)[0])[0]
    ind2 = (np.where(mask)[0])[-1]
    mask1 = np.concatenate((np.ones(ind1), np.array(mask[ind1:ind2 + 1]), np.ones(len(mask) - ind2 - 1))).astype(bool)
    begholes = np.where(mask1 * np.roll(~mask1, -1))[0] + 1  # locates inside the hole
    endholes = np.where(mask1 * np.roll(~mask1, 1))[0]  # locates outside the hole
    lenholes = endholes - begholes
    big_hole_inds = np.where(lenholes >= max_hole)[0]
    if len(big_hole_inds) > 0:
        begholes = begholes[big_hole_inds]
        endholes = endholes[big_hole_inds]

        inds_hole_pieces = [[ind1, begholes[0]]]
        for i in range(len(begholes) - 1):
            inds_hole_pieces.append([endholes[i], begholes[i + 1]])
        inds_hole_pieces.append([endholes[-1], ind2 + 1])
    else:
        inds_hole_pieces = [[ind1, ind2 + 1]]

    inds_pieces = []
    for ind_hp in inds_hole_pieces:
        len_piec = ind_hp[1] - ind_hp[0]
        if len_piec >= max_piece:
            len_cut = int(np.random.uniform(max(len_piec - max_piece, min_piece), min(len_piec - min_piece, max_piece)))
            inds_pieces.append([ind_hp[0], ind_hp[0] + len_cut])
            inds_pieces.append([ind_hp[0] + len_cut, ind_hp[1]])
        else:
            inds_pieces.append(ind_hp)
    return np.array(inds_pieces)


def gen_uniform_piece_lengths(tot_len, n_pieces):
    """
    Splits `tot_len` into `n_pieces` parts so that the distribution of parts length is uniform
    :param tot_len: total length which needs to be split into parts
    :type tot_len: int
    :param n_pieces: number of pieces in which `tot_len` needs to be split
    :type n_pieces: int
    :return: `lens`, the lengths
    :rtype: List[int]
    """
    lens_0 = np.random.uniform(0, tot_len, n_pieces)
    lens_0 = lens_0 / np.sum(lens_0) * tot_len
    lens = np.round(lens_0).astype(int)
    # make sure somehow that the total length is correct
    while np.sum(lens) != tot_len:
        if np.sum(lens) > tot_len:
            lens[np.nanargmax(lens - lens_0)] -= 1
        elif np.sum(lens) < tot_len:
            lens[np.nanargmin(lens - lens_0)] += 1
    return lens


def rearrange_pieces(data, inds_pieces, hole_lens, fill_hole_with=0, inds_target=None,
                     return_inds_target_res=False):
    """
    Cuts `data` into pieces according to `inds_pieces` (which can have any order) and then glues pieces together
    interleaving them with holes of lengths `hole_lens` filled with `fill_hole_with` values.
    Starts filling the new array from the zero index (no hole in the beginning).
    There is hole after the last piece, although its length may be zero.
    In-between the pieces, puts `fill_hole_with` values.
    :param data: the initial data array which will be cut and rearranged.
    :type data: ndarray
    :param inds_pieces: list[list[int, int]]
    :type inds_pieces:
    :param hole_lens:
    :type hole_lens:
    :param fill_hole_with:
    :type fill_hole_with:
    :param inds_target: If not None, the pieces from inds_pieces will put into places of inds_target
    :type inds_target:
    :param return_inds_target_res:
    :type return_inds_target_res: bool
    :return: `new_data`, the rearranged pieces of the `data` interleaved with the holes
    :rtype: ndarray
    """
    data = np.array(data)
    new_data = np.array([fill_hole_with for _ in range(len(data))])
    if inds_target is None:
        inds_target_res = []
        pos = 0  # current position
        for i, ip in enumerate(inds_pieces[:-1]):
            piece = data[ip[0]: ip[1]]
            pos_1 = pos + len(piece)
            new_data[pos: pos_1] = piece
            inds_target_res.append([pos, pos_1])
            pos = pos_1 + hole_lens[i]
            new_data[pos_1: pos] = fill_hole_with
        i = len(inds_pieces) - 1
        ip = inds_pieces[-1]
        piece = data[ip[0]: ip[1]]
        pos_1 = pos + len(piece)
        new_data[pos: pos_1] = piece
        inds_target_res.append([pos, pos_1])
        if i < len(hole_lens):  # if the number of holes was equal to the number of pieces and not one less
            pos = pos_1 + hole_lens[i]
            new_data[pos_1: pos] = fill_hole_with
        if return_inds_target_res:
            return new_data, np.array(inds_target_res)
        else:
            return new_data
    else:
        for i, ip in enumerate(inds_pieces):
            it = inds_target[i]
            new_data[it[0]: it[1]] = data[ip[0]: ip[1]]
        return new_data

# ################################################################################################


def replace_if_deviating(vector, replacement, tester=None, mindev=1, minwid=1, i_min=0):
    sign = np.sign(mindev)
    mindev = np.abs(mindev)
    new_vector = np.array(vector)
    if tester is None:
        tester = replacement - vector
    i = i_min
    while i < len(vector):
        if sign * tester[i] - mindev > 0.:
            i_1 = i
            i_2 = i
            while (i_2 < len(vector)) and (sign * tester[i_2] - mindev > 0):
                i_2 += 1
            if (i_2 - i_1) >= minwid or (i_1 <= 2 and (i_2 - i_1) >= minwid / 2):
                # find the beginning and the end of the replacement region:
                j_1 = i_1
                while (j_1 - 1 >= 0) and (sign * tester[j_1 - 1] > 0):
                    j_1 -= 1
                j_2 = i_2
                while (j_2 < len(vector)) and (sign * tester[j_2] > 0):
                    j_2 += 1  # this is the last slice index, so it can be bigger
                new_vector[j_1:j_2] = replacement[j_1:j_2]
                i = j_2 + 1
            else:
                i = i_2 + 1
        else:
            i += 1
    return new_vector


def replace_beginning(vector, replacement, min_repl_size, return_repl_size=False):
    ind_temp = min_repl_size  # 2 * fmb_int * int(nperseg / npersegs_lim[i])
    sign_0 = np.sign(vector[ind_temp] - replacement[ind_temp])
    ind_temp += 1
    sign = np.sign(vector[ind_temp] - replacement[ind_temp])
    while sign == sign_0 and ind_temp < len(vector) - 1:
        ind_temp += 1
        sign = np.sign(vector[ind_temp] - replacement[ind_temp])
    vector[: ind_temp] = replacement[: ind_temp]
    if not return_repl_size:
        return vector
    else:
        return vector, ind_temp


def pbin_fun(x=None, dx=None, x2=None, x1=None, loc=0, scale=1, distr='norm'):
    """
    attention to factors of 2!!!!!!!
    x: bin_center
    xd: bin width
    alternatively:
    x2: bin upper edge
    x1: bin lower edge
    """
    if x2 is None or x1 is None:
        x2 = x + dx/2
        x1 = x - dx/2
    if distr == 'norm':
        return norm.cdf(x2, loc=0, scale=1) - norm.cdf(x1, loc=loc, scale=scale)
    elif distr == 'norm_abs':
        assert (np.array([x1]) >= 0).all(), 'distribution of absolute values should not have negative bins!'
        assert (np.array([x2]) >= 0).all(), 'distribution of absolute values should not have negative bins!'
        return 2 * (norm.cdf(x2, loc=0, scale=1) - norm.cdf(x1, loc=loc, scale=scale))
    elif distr == 'norm_quadr_bins':
        return (norm.cdf(x2**0.5, loc=0, scale=1) - norm.cdf(x1**0.5, loc=loc, scale=scale)) * 2
    else:
        raise ValueError('Undefined `distr` value!')


def expected_hist(total_counts, bins, loc=0, scale=1, times_per_bins=1, n_sigma=None, distr='norm_abs', non_integer=False):
    """
    times_per_bins: how many times per bin the value can deviate from the allowed interval. the
            deviation will be split into two: above and below same prob
    """
    pbins = pbin_fun(x2=bins[1:], x1=bins[:-1], loc=loc, scale=scale, distr=distr)
    expected = pbins * total_counts
    if not non_integer:
        prob = times_per_bins/(len(bins)-1)  # something which happens one per bin
        upper = binom.isf(prob/2, n=total_counts, p=pbins, loc=0)
        lower = binom.ppf(prob/2, n=total_counts, p=pbins, loc=0)
        return expected, upper, lower
    else:
        # norm_factor = total_counts * (bins[1] - bins[0]) * 2  # factor 2 because both positive and negative values
        stds_predicted = (pbins * (1-pbins) * total_counts)**0.5
        if n_sigma is None:
            prob = times_per_bins/(len(bins)-1)  # something which happens one per bin
            n_sigma = norm.isf(prob/2, loc=loc, scale=scale)  # how far we can go to still be within this prob
        upper = expected + n_sigma * stds_predicted
        lower = expected - n_sigma * stds_predicted
        lower[lower < 0] = 0
        return expected, upper, lower


def cumsum_for_sf_and_conf_ints(counts_bins, n_sigma=1):
    """
    counts_bins should be real counts, no normalization!
    """
    counts_total = np.sum(counts_bins)
    p_be_in_bin = counts_bins / counts_total
    vars_predicted = (p_be_in_bin * (1-p_be_in_bin) * counts_total)
    sf_hist_quadr = (np.cumsum(counts_bins[::-1]))[::-1]
    stds_sf_hist_quadr = ((np.cumsum(vars_predicted[::-1]))[::-1])**0.5
    lower = sf_hist_quadr - n_sigma * stds_sf_hist_quadr
    lower[lower < 0] = 0
    upper = sf_hist_quadr + n_sigma * stds_sf_hist_quadr
    return sf_hist_quadr, lower, upper


def fit_two_gaussian_mix(stats, n_bins=50, snr_0_fit=2.5, beg_min=3):
    """
    Making histograms of the distributions of SNRS squared for every template, fitting the two-gaussian model and
    finding the parameters of the tail gaussian. The first gaussian is supposed to be the standard one because the
    data was brought to units of SNR by normalizing with its variance. However, the tail can be too heavy. Usually,
    the distribution which is a sum of two gaussians fits the data. Therefore, the histogram is fitted for the
    scale anf the amplitude of the tail gaussian. We do not make likelihood optimization of the tail, we just
    fit the CDF of the bins in a stupid way. The likelihood fit with too many points will not work, so it could
    only be used in the very tail. We are not too much afraid that we will overfit large single outliers,
    we do not want to save these outliers anyway. We even cut some of them in the veto.
    :param stats: Statistics in units of SNR for which we make the fit. Should be one-sided
    :type stats: numpy.ndarray 1d of float
    :param n_bins: number of bins to use in the histogram. The recommended is 50
    :type n_bins: int
    :param snr_0_fit: The smallest snr from which to start the fitting of the tail. Recommended 2.5
    :type snr_0_fit: float
    :param beg_min: Consider deviations from the distribution srarting from this SNR
    :type beg_min: float
    """
    stats = np.abs(stats)
    bins_linear = np.linspace(0, np.max(stats) + 0.5, n_bins)
    centr_bins_linear = bins_linear[:-1] + (bins_linear[1] - bins_linear[0]) / 2
    counts_linear, bins_linear = np.histogram(stats, bins=bins_linear, density=False)
    expected, upper, lower = expected_hist(np.sum(counts_linear), bins_linear)

    best_scale = 1.
    best_coef = 0.  # If the mixture is not needed, this will be the answer
    more_than_upper = (counts_linear - upper) > 0  # Checking where the histigram deviates
    total_dev_bins = np.sum(more_than_upper[centr_bins_linear >= beg_min])

    # Whether the histogram has significant deviation in relevant bins:
    if total_dev_bins > 0 and np.max(stats) > beg_min:
        cum_more_than_upper = np.cumsum(more_than_upper)
        # noinspection PyTypeChecker
        ind_dev = np.where((cum_more_than_upper > min(7, cum_more_than_upper[-1] / 3)) & (centr_bins_linear >= beg_min))[0]
        if len(ind_dev) == 0:
            raise (ValueError('Something is very wrong with the distribution'))
        else:
            ind_dev = ind_dev[0] - 1
        snr_start = bins_linear[ind_dev]
        if snr_start == len(bins_linear) - 1:
            snr_start = snr_start - 1

        ###########################################################################
        # QUADRATIC HIST
        stats_squared = np.array(stats) ** 2
        bins_quadr = np.linspace(0, np.max(bins_linear)**2, n_bins)
        counts_quadr_non_norm, bins_quadr = np.histogram(stats_squared, bins=bins_quadr, density=False)
        last_non_zero = np.where(counts_quadr_non_norm > 0)[0][-1]
        counts_quadr_non_norm = counts_quadr_non_norm[:last_non_zero + 1]
        bins_quadr = bins_quadr[:last_non_zero + 2]
        d_bin_quadr = bins_quadr[1] - bins_quadr[0]
        centr_bins_quadr = bins_quadr[:-1] + d_bin_quadr / 2
        norm_factor = 1 / (np.sum(counts_quadr_non_norm))  # * d_bin_quadr)
        sf_hist_quadr, lower_sf_hist_quadr, upper_sf_hist_quadr = cumsum_for_sf_and_conf_ints(counts_quadr_non_norm, n_sigma=1)
        sf_hist_quadr = sf_hist_quadr * norm_factor

        lower_sf_hist_quadr = lower_sf_hist_quadr * norm_factor
        upper_sf_hist_quadr = upper_sf_hist_quadr * norm_factor
        log_sf_hist_quadr = np.log(sf_hist_quadr)
        log_up_sf_hist_quadr = np.log(upper_sf_hist_quadr)
        upper_sf_hist_quadr[upper_sf_hist_quadr == 0] = np.nanmin(upper_sf_hist_quadr[upper_sf_hist_quadr != 0])
        lower_sf_hist_quadr[lower_sf_hist_quadr == 0] = np.nanmin(lower_sf_hist_quadr[lower_sf_hist_quadr != 0])
        log_low_sf_hist_quadr = np.log(lower_sf_hist_quadr)
        log_errors_sf_hist_quadr = (log_up_sf_hist_quadr - log_low_sf_hist_quadr) / 2
        zero_indices = np.where(sf_hist_quadr == 0)[0]
        if len(zero_indices) > 0:
            log_errors_sf_hist_quadr[zero_indices] = log_errors_sf_hist_quadr[zero_indices + 1]

        ###########################################################################
        # OREPATING LOG QUADRATIC FIT (APPROX!)

        indices_fit = np.where(centr_bins_quadr > snr_0_fit ** 2)[0]  # & (sf_hist_quadr>0) &
        if len(indices_fit) > 0:
            if 1 <= len(indices_fit) < 3:
                indices_fit = np.arange(3) - 3 + len(log_sf_hist_quadr)

            bins_fit_sqrt = (centr_bins_quadr[indices_fit]) ** 0.5
            log_sf_hist_quadr_fit = log_sf_hist_quadr[indices_fit]
            log_errors_sf_hist_quadr_fit = log_errors_sf_hist_quadr[indices_fit]

            ###########################################################################
            # INITIAL GUESS
            ind_init_guess = np.where(centr_bins_quadr >= snr_start ** 2)[0]
            if len(ind_init_guess) == 0:
                raise (ValueError('Stat distribution fit initial guess array is empty!'))
            else:
                ind_init_guess = ind_init_guess[0]
            bins_quad_init_guess = centr_bins_quadr[ind_init_guess:]

            log_sf_init_guess = log_sf_hist_quadr[ind_init_guess:]
            log_errors_sf_hist_quadr_init_guess = log_errors_sf_hist_quadr[ind_init_guess:]

            sol = minimize(lambda x: np.sum((log_sf_init_guess - (x[0] + x[1] * bins_quad_init_guess)
                                             ) ** 2 / log_errors_sf_hist_quadr_init_guess ** 2), x0=np.array([0, -1]))
            # if not sol.success:
            #     print('initial guess linear fit failed!!!')
            if sol.x[1] < -1.e-6:
                best_scale = (1 / (-sol.x[1] * 2)) ** 0.5  # initial_guess_scale
            else:
                best_scale = 1.
            if best_scale <= 0:
                raise ValueError('Stat distribution fit best is not positive!')

            ###########################################################################
            # FIT
            normal_part = norm.sf(bins_fit_sqrt, loc=0, scale=1)

            def opt_fun(x, scal):
                return np.sum((log_sf_hist_quadr_fit - np.log(
                    2*(normal_part + np.abs(x[0]) * norm.sf(bins_fit_sqrt, loc=0, scale=scal)) / (1 + np.abs(x[0])))) ** 2
                              / log_errors_sf_hist_quadr_fit ** 2)

            coefs_refinement = [[0.8, 1.7], [0.95, 1.1]]
            for co1, co2 in coefs_refinement:
                scale_grid = np.linspace(max(1.05, best_scale * co1), best_scale * co2, 20)
                ress = []
                funs = []
                successes = []
                for scale in scale_grid:
                    sol = minimize(opt_fun, x0=np.array([0.01]), args=(scale))
                    ress.append(np.abs(sol.x[0]))
                    funs.append(sol.fun)
                    successes.append(sol.success)
                best_sol_index = np.argmin(funs)
                best_scale = scale_grid[best_sol_index]
                best_coef = ress[best_sol_index]
    return best_scale, best_coef


def gauss_sum_pdf(x, scale=1., ampl=0.):
    return (norm.pdf(x, loc=0, scale=1) + ampl * norm.pdf(x, loc=0, scale=scale)) / (1 + ampl)


def log_non_gaussian_correction(stat, ampl, scale):
    ng_pdf = gauss_sum_pdf(stat, scale=scale, ampl=ampl)
    g_pdf = gauss_sum_pdf(stat, scale=1., ampl=0.)
    return np.log(g_pdf) - np.log(ng_pdf)


def select_connected_component(image, center, weights=None, forbid_diagonal=False):
    """

    :param image: image where the connected component is found; will e mad a binary image
    :type image: numpy.ndarray of float or bool
    :param center: coordinates of the point to which the connected component is found
    :type center: tuple or list of two coordinates
    :param weights: If not None, the weights are used to select the neighbors which are not heavier than the cantral point,
        which is the point around which the boundary is expanded.
    :type weights: None or numpy.ndarray 2d of float, same size as `image`
    :param forbid_diagonal: Not allow to select a diagonal neighbor as the new boundary. When the boundary is
        expanded, only the direct steps (non-diagonal) are taken
    :type forbid_diagonal: bool
    :return: mask of points included in the connected component
    :rtype: numpy.ndarray 2d of bool, same size as `image`
    """
    def indices_square(center_coords, d, x_max=None, y_max=None, exclude_angles=False):
        cc = center_coords
        x_left = cc[0] - d
        x_right = cc[0] + d
        span_x = np.arange(x_left, x_right + 1)
        y_left = cc[1] - d
        y_right = cc[1] + d
        span_y = np.arange(y_left, y_right + 1)
        top = [span_x, np.ones(len(span_x), dtype=int) * y_right]
        bottom = [span_x, np.ones(len(span_x), dtype=int) * y_left]
        left = [np.ones(len(span_y), dtype=int) * x_left, span_y]
        right = [np.ones(len(span_y), dtype=int) * x_right, span_y]
        if exclude_angles:
            if len(top[0]) >= 3:
                top = [top[0][1:-1], top[1][1:-1]]
            if len(bottom[0]) >= 3:
                bottom = [bottom[0][1:-1], bottom[1][1:-1]]
            if len(left[0]) >= 3:
                left = [left[0][1:-1], left[1][1:-1]]
            if len(right[0]) >= 3:
                right = [right[0][1:-1], right[1][1:-1]]
        all_coords = np.hstack((top, right, bottom, left))

        all_coords = all_coords[:, all_coords[0] >= 0]
        all_coords = all_coords[:, all_coords[1] >= 0]
        if x_max is not None:
            all_coords = all_coords[:, all_coords[0] <= x_max]
        if y_max is not None:
            all_coords = all_coords[:, all_coords[1] <= y_max]
        # all_coords.T  # - for coordinate pairs list
        # (all_coords[0], all_coords[1])  # - for matrix indexing
        return list(zip(all_coords[0], all_coords[1]))  # gives list of tuples which is super important for array indexing!

    if weights is not None:
        weights = weights / np.nanmax(weights)
    image = image.astype(bool)
    included_mask = np.zeros_like(image).astype(bool)
    if image[center[0], center[1]]:
        x, y = center[0], center[1]
        included_mask[x, y] = True
        coords_boundary = [(x, y)]
        safety_counter = 0
        safety_counter_max = image.size / 8
        while len(coords_boundary) > 0:
            # 0) take all recently included and in image
            # 1) all squares
            # 2) set -> border
            # 3) if there are image true in the border, include them and take as the next border formation
            # 4) the old ones can be deleted because they already served
            new_coords_boundary = []
            for coor_boun in coords_boundary:
                square_coords_point = indices_square(center_coords=coor_boun, d=1, x_max=image.shape[0] - 1,
                                                     y_max=image.shape[1] - 1, exclude_angles=forbid_diagonal)
                # We may want to only select new value which has weight smaller than the considered one,
                # so that everything is descending around the peak:
                if weights is not None:
                    center_weight = weights[coor_boun[0], coor_boun[1]]
                    square_coords_point = [coor for coor in square_coords_point if weights[coor[0], coor[1]] <= center_weight]
                new_coords_boundary += square_coords_point
            if len(new_coords_boundary) == 0:  # It can be that no coord passed the weight
                break
            new_coords_boundary = list(set(new_coords_boundary))
            ncb_reshaped = list(zip(*new_coords_boundary))
            in_image_inds_new_coords_boundary = \
                np.where((image[ncb_reshaped[0], ncb_reshaped[1]]) & (~included_mask[ncb_reshaped[0], ncb_reshaped[1]]))[0]
            if len(in_image_inds_new_coords_boundary) == 0:
                break
            new_coords_boundary = [new_coords_boundary[ind] for ind in in_image_inds_new_coords_boundary]
            ncb_reshaped = list(zip(*new_coords_boundary))
            included_mask[ncb_reshaped[0], ncb_reshaped[1]] = True
            # new_coords_boundary = list(zip(*ncb_reshaped))
            coords_boundary = copy.deepcopy(new_coords_boundary)
            safety_counter += 1
            if safety_counter > safety_counter_max:
                break
    return included_mask


def get_indices_of_influence(vals, ind_center_0, ind_center_1,
                             part_peak=0.5, abs_diff_val=None,
                             abs_thresh=None, rel_thresh=None,
                             forbid_diagonal=False, descending=False):
    peak_val = vals[ind_center_0, ind_center_1]
    if abs_diff_val is not None:
        inds_sum = (vals >= (peak_val - abs_diff_val))
    elif part_peak is not None:
        ambient_val = np.nanmedian(vals)
        inds_sum = (vals > ambient_val + part_peak * (peak_val - ambient_val))
        ambient_val = np.nanmedian(vals[~inds_sum])  # repetitive median to exclude peak
        inds_sum = (vals >= ambient_val + part_peak * (peak_val - ambient_val))
    else:
        if abs_thresh is None:
            abs_thresh = rel_thresh * peak_val
        inds_sum = (vals >= abs_thresh)

    if descending:  # If we do not want to include secondary peaks connected to the primary peak
        weights = vals
    else:
        weights = None
    inds_sum = select_connected_component(inds_sum, center=[ind_center_0, ind_center_1], forbid_diagonal=forbid_diagonal,
                                          weights=weights)
    return inds_sum


def make_periods_array(p_min, p_max, dp=None, duty_cycle=Duty_cycle, t_mission=1300):
    """
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
    per = p_min
    periods = [per]
    if dp is None:
        while per <= p_max:
            dp = duty_cycle * per / t_mission
            per += dp
            periods.append(per)
    else:
        while per <= p_max:
            per += dp
            periods.append(per)
    periods = np.array(periods)
    return periods


def make_periods_edges(p_min, p_max, per_phases_per_chunk, dp=None, duty_cycle=Duty_cycle, dt=0.02043368):
    """
    Creates edges of period chunks for which the periodicity search will be performed separately.
    This is needed because of memory restrictions per core. Returns the edges of the chunks.
    :param per_phases_per_chunk: The total number of period-phase bins which should be in one chunk.
        Is set by memory estimation. Number of phases is different for every period, therefore chunks will have
        different number of periods in them.
    :type per_phases_per_chunk: int
    :param p_min: Minimal period in the search.
    :type p_min: float or None
    :param p_max: Maximal period in the search.
    :type p_max: float or None
    :param dp: Period step. If None, will be calculated from the `duty_cycle`. Usually is always None.
    :type dp: float or None
    :param duty_cycle: Maximal accumulated deviation due to the period error in the end of the mission.
        In units of days.
    :type duty_cycle: float
    :param dt: Time step of the data. Default is the typical Kepler time step.
    :type dt: float
    :return: per_edges, list of pairs of periods: the beginning and the end of every chunk.
    :rtype: list[list[float, float]]
    """
    periods = make_periods_array(p_min=p_min, p_max=p_max, dp=dp, duty_cycle=duty_cycle)
    p_chunks = int(np.sum((periods / dt).astype(int)) / per_phases_per_chunk)
    per_edges = [(p_min + (p_max - p_min) * i / p_chunks) for i in range(p_chunks + 1)]
    per_edges = [[per_edges[i], per_edges[i + 1]] for i in range(len(per_edges) - 1)]
    return per_edges


def make_periods_chunks(p_min, p_max, per_phases_per_chunk, duty_cycle, dt=0.02043368, t_mission=1300):
    """
    Creates the period grid that will be used for periodicity search and splits it into chunks.
        The grid is built according to the duty cycle to cover the periodicities smoothly.
        Splitting into chunks is needed because of memory restrictions per core.
    :param per_phases_per_chunk: The total number of period-phase bins which should be in one chunk.
        Is set by memory estimation. Number of phases is different for every period, therefore chunks will have
        different number of periods in them.
    :type per_phases_per_chunk: int
    :param p_min: Minimal period in the search.
    :type p_min: float or None
    :param p_max: Maximal period in the search.
    :type p_max: float or None
    :param duty_cycle: Maximal accumulated deviation due to the period error in the end of the mission.
        In units of days.
    :type duty_cycle: float
    :param dt: Time step of the data. Default is the typical Kepler time step.
    :type dt: float
    :param t_mission: Total duration of the mission in days.
    :type t_mission: float
    :return: periods_in_chunks, list of chunks of period grids
    :rtype: list of np.ndarray 1d
    """
    if per_phases_per_chunk is not None:
        per_edges = make_periods_edges(p_min=p_min, p_max=p_max, per_phases_per_chunk=per_phases_per_chunk, duty_cycle=duty_cycle, dt=dt)
    else:
        per_edges = [[p_min, p_max]]

    periods = make_periods_array(p_min=p_min, p_max=p_max, duty_cycle=duty_cycle, t_mission=t_mission)
    i_last = 0
    periods_in_chunks = []
    for edge in per_edges:
        end_edge = np.searchsorted(periods, edge[-1])
        periods_in_chunks.append(periods[i_last:end_edge + 1])
        i_last = end_edge + 1
    return periods_in_chunks


def add_safety_overlap(period_list, safety_bins=P_bins_overlap):
    """
    Adds spare period bins to the periods array for the best trigger not to be at the boundary of the array.
    This is needed because the integral score will not be full if the trigger is on the boundary.
    :param period_list: Grid of periods
    :type period_list: numpy.ndarray of float
    :param safety_bins: Number of spare period bins added to the periods array for the best trigger
            not to be at the boundary of the array. This is needed because the integral score will not be full
            if the trigger is on the boundary.
        :type safety_bins: int
    :return: new extended period array
    :rtype: numpy.ndarray of float
    """
    period_list = np.array(period_list)
    dp_low = period_list[1] - period_list[0]
    dp_high = period_list[-1] - period_list[-2]
    array_low = period_list[0] + np.arange(-safety_bins, 0) * dp_low
    array_high = period_list[-1] + np.arange(1, safety_bins + 1) * dp_high
    return np.concatenate((array_low, period_list, array_high))


def find_closest_chunk(per_chunks, period, overlap_bins=0):
    """
    period chunks are defined without overlap in the TargetClass, the overlap is
    added in the periodicity search of the KeplerLightCurve
    :param per_chunks:
    :type per_chunks:
    :param period:
    :type period:
    :param overlap_bins: ZERO. period chunks are defined without overlap in the TargetClass, the overlap is
                         added in the periodicity search of the KeplerLightCurve
    :type overlap_bins: int, default 0
    """
    for i, per_chun in enumerate(per_chunks):
        last_val = per_chun[-1] if i == len(per_chunks) - 1 else per_chunks[i+1][0]
        if (period >= per_chun[0]) and (period <= last_val):
            ind_close = np.argmin(np.abs(np.array(per_chun) - period))
            if ind_close >= overlap_bins and overlap_bins <= len(per_chun) - overlap_bins:
                return per_chun, i


def rough_fold(array, period_bins):
    res = np.zeros(period_bins)
    for i in range(int(np.ceil(len(array) / period_bins))):
        i1 = i * period_bins
        i2 = min(len(array), (i + 1) * period_bins)
        res[: i2 - i1] += array[i1: i2]
    return res


def get_transit_times(first_transit_time, period, max_time):
    return np.array([first_transit_time + i * period
                     for i in range(int(np.ceil((max_time - first_transit_time) / period)))])


def check_transit_overlap(period, ftt, period_ref, ftt_ref, n_transits, timing_error):
    """
    Checks if the transits of the ephemeris set by `ftt` and `period` overlap with the transits
    of the ephemeris set by `ftt_ref` and `period_ref`. The overlap is defined by the transit centers being off
    by less than `timing_error`. The transit durations are not taken into account.
    """
    ephemeris = np.arange(ftt, ftt + period * n_transits, period)[:n_transits]
    ephemeris_ref = np.arange(ftt_ref, ftt + period * n_transits, period_ref)
    if len(ephemeris_ref) > 0:
        min_offsets = np.array([np.min(np.abs(ephemeris_ref - tt)) for tt in ephemeris])
        overlap_mask = min_offsets < timing_error
    else:
        overlap_mask = np.zeros(len(ephemeris)).astype(bool)
    return overlap_mask


def calculate_p_planet_statistical(f_p, f_s, f_sc, pi_p=None, mu_m_mu_p=1):
    kap = mu_m_mu_p
    if pi_p is not None:
        g_p = f_p * pi_p
        g_s = f_s * (1 - pi_p)
        g_sc = f_sc * (1 - pi_p)
    else:
        g_p = f_p
        g_s = f_s
        g_sc = f_sc

    return g_p * kap / (0.5 * (kap * (g_p + g_s) - (1 - kap) * g_p)
                        + 0.5 * ((kap * (g_p + g_s) - (1 - kap) * g_p) ** 2
                                 + 4 * g_p * kap * ((g_p + g_s) * (1 - kap) + g_sc - g_s)) ** 0.5)
