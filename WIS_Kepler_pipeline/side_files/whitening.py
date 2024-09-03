#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 2022

@author: orynaivashtenko
version November 2022


"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation as mad

Nperseg_default = 128
Noverlap_default = 64
Window_psd_default = 'hann'
Subtract_mean_in_psd = True
Use_rfft = True
Smoothen_psds = False


def pad_by_zeros(data, support=None, nperseg=Nperseg_default, return_mask=True):
    """Pads `data` by zeros from right and left to reach the length `support`.

    Padding is needed when some filter is applied in circular convolution, and there is a need to
    avoid crosstalks between the ends. Also, it is useful to have a spare 'nperseg' zeros to
    avoid crosstalk due to the correlation length of the whitening filter.

    Parameters
    ----------
    data : ndarray 1d
        Data vector which needs to be padded by zeros.

    support : int
        Length to which the data should be padded. If `support` is None, then the data is padded
        up to the length of the power of 2 closest to len(`data`)+`nperseg`. It is recommended
        in order to avoid possible crosstalks in circular convolution when applying a whitening filter
        with correlation length `nperseg`.

    nperseg : int
        Length of the slice which will be used to measure PSD. (See `measure_psd` documentation`)

    return_mask : bool
        Whether to return a mask which will mask the padded zeros as bad data. This mask should be
        fed to the PSD measurement function in order not to use zeros in the PSD measurement.

    Returns
    -------
    padded_data : ndarray 1d
        The data padded by zeros
    mask : ndarray 1d
        Masking padded zeros by False and the rest of the points by True
    """
    data = np.array(data)

    if support is None:
        support = 2 ** int(np.ceil(np.log2(len(data) + nperseg)))  # the closest (but bigger) power of 2 to the length of the data

    zerlen1 = (support - len(data)) // 2
    zerlen2 = (support - len(data) + 1) // 2

    padded_data = np.concatenate((np.zeros(zerlen1), data, np.zeros(zerlen2)))
    mask = np.concatenate((np.zeros(zerlen1), np.ones(len(data)), np.zeros(zerlen2))).astype(bool)
    if return_mask:
        return padded_data, mask
    else:
        return padded_data


def unpad(data, padding_mask):
    data = np.array(data)
    padding_mask = np.array(padding_mask)
    if padding_mask.dtype is not np.dtype('bool'):
        padding_mask = padding_mask.astype(bool)
    return data[padding_mask]


def psd2wf(psd, support=None, use_rfft=True):
    """Convert Power Spectrum Density to whitening filter.
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

    use_rfft : bool
        If True, the signal is treated as real, and all ffts are rffts. The length
        of the frequency samples are the nperseg//2+1, where nperseg is the length
        in the rela spece corresponding to this PSD
        Default is True
    Returns
    -------
    wf_td : ndarray 1d
        whitening filter in time domain
    wf_fd : ndarray 1d
        whitening filter in Fourier domain.
    """
    if use_rfft:
        if support is None:
            support = 2 * (len(psd) - 1)
        assert support % 2 == 0, 'support assumed to be even!'
        wpsd = np.array(psd ** (-0.5))
        wpsd[0] = 0.  # if you don't do this, you will have a complete disaster after notch!
        wf_td = np.fft.irfft(wpsd)
        mid_point = len(wf_td) // 2
        result = np.zeros(support)
        result[:mid_point] = wf_td[:mid_point]
        result[-mid_point:] = wf_td[-mid_point:]
        wf_td = np.fft.fftshift(result)
        wf_fd = np.fft.rfft(result)
    else:
        if support is None:
            support = len(psd)
        assert support % 2 == 0, 'support assumed to be even!'
        wpsd = np.array(psd ** (-0.5))
        wpsd[0] = 0.
        wf_td = np.fft.ifft(wpsd)
        mid_point = len(wf_td) // 2
        result = np.zeros(support, dtype=np.complex128)
        result[:mid_point] = wf_td[:mid_point]
        result[-mid_point:] = wf_td[-mid_point:]
        wf_td = np.fft.fftshift(result)
        wf_fd = np.fft.fft(result)
    return wf_td, wf_fd


def measure_psd(noise, mask=None, dt=1, nperseg=Nperseg_default, noverlap=Noverlap_default, window=Window_psd_default,
                support=None, subtract_mean=Subtract_mean_in_psd, use_rfft=Use_rfft, smoothen_psd=Smoothen_psds,
                return_filters=True, return_psd=True, return_freqs=True):
    """Measures Power Spectral Density with Welch's method and produces whitening filter
    The PSD measuring algotithm makes the evenly sampled steps and not adjust the search for
    good pieces according to the placement of the zeros in mask. That is why

    Parameters
    ----------
    noise : ndarray 1d
        The data whose PSD will be measured

    mask : ndarray 1d of bool
        Array of ones where data is good and zeros where data is bad. The bad places will be ignored in PSD measurement.

    dt : float
        Timestep of the time axis corresponding to the data. (Data is supposed to be uniformly sampled!).
        This timestep is only important in generating the frequency array.

    nperseg : int
        Length of the slice used for the averaging of local PSDS in Welch's method. The bigger 'nperseg',
        the better the resolution, but worse statistical precision.

    noverlap : int or float<1
        If int: length of overlap of slices in Welch method.
        If float<1 : relative overlap to nperseg.
        Length of overlap will be int(nperseg*noverlap)

    window : string
        window function by which the signals are multiplied to avoid boundary effects and crosstalk
        Possible options: 'hann', 'boxcar', where 'boxcar' is just normalized bow, or no window
        Default is 'hann'. It is not recommended to use 'boxcar'.

    support : int or None
        Length of the data to which you want to apply the filter. If None, will use the length of the `noise`

    subtract_mean : bool
        If True, the mean of the signal is subtracted. It is recommended to do it because
        otherwise unwanted power may appear in the zero frequency

    use_rfft : bool
        If True, the signal is treated as real, and all ffts are rffts. The length of the frequency samples
        are the nperseg//2+1.

    smoothen_psd : bool
        Whether to apply a fitting to the white tail of the red-noise PSD to reduce the measurement uncertainty there.

    return_filters : bool
        Whether to return whitening filters (both Fourier domain and time domain)

    return_psd : bool
        Whether to return whitening PSD

    return_freqs : bool
        Whether to return frequencies corresponding to the whitening folter or PSD

    Returns
    -------
    (returns on demand conditioned upon parameters)
    wf_fd : ndarray 1d
        Whitening filter in Fourier domain of length corresponding to `support`
    wf_td : ndarray 1d
        Whitening filter in Fourier domain of length corresponding to `support`
    psd : ndarray 1d
        PSD of length corresponding to nperseg
    freqs : ndarray 1d
        Array of frequencies corresponding to length `support` and cadence `dt`
    psdfreqs : ndarray 1d
        Array of frequencies corresponding to nperseg and cadence `dt`
    """
    noise = np.array(noise)

    if mask is None:
        mask = ~np.isnan(noise)
    elif mask.dtype is not np.dtype('bool'):
        mask = mask.astype(bool)

    if noverlap < 1:
        noverlap = int(nperseg * noverlap)
    if noverlap == -1:
        noverlap = nperseg // 2

    windowdict = {'boxcar': np.ones(nperseg) / np.linalg.norm(np.ones(nperseg)),
                  'hann': 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg)) / np.linalg.norm(
                      0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg)))}

    if subtract_mean:
        noise = noise - np.nanmean(noise[mask])
    if use_rfft:
        fftfunc = np.fft.rfft
        fftfreqfunc = np.fft.rfftfreq
    else:
        fftfunc = np.fft.fft
        fftfreqfunc = np.fft.fftfreq

    ind0 = np.argmax(mask)
    psds = [np.abs(fftfunc(windowdict[window] * noise[ind1:ind1 + nperseg])) ** 2 for ind1 in
            range(ind0, len(noise) - nperseg, nperseg - noverlap) if np.sum((~mask)[ind1:ind1 + nperseg]) == 0]
    psd = np.mean(psds, axis=0)

    if smoothen_psd:
        fs = np.arange(len(psd))
        sel_ind = np.arange(len(psd))[-len(psd) // 3 * 2:]

        psd_high = np.array(psd[sel_ind])
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

        residuals = psd - trend
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
        psd = np.array(psd_cut)

    if support is None:
        support = len(noise)
    wf_td, wf_fd = psd2wf(psd, support, use_rfft=use_rfft)

    freqs = fftfreqfunc(support, d=dt)
    psdfreqs = fftfreqfunc(nperseg, d=dt)

    assert return_psd or return_filters, 'Return product not specified!'

    if return_filters:
        if return_psd:
            if return_freqs:
                return wf_fd, wf_td, psd, freqs, psdfreqs
            else:
                return wf_fd, wf_td, psd
        else:
            if return_freqs:
                return wf_fd, wf_td, freqs
            else:
                return wf_fd, wf_td
    else:
        if return_freqs:
            return psd, psdfreqs
        else:
            return psd


def apply_whitening_filter(data, whitening_filter_fd=None, psd=None, use_rfft=Use_rfft):
    """Whitens data with whitening filter in Fourier domain

    If `whitening_filter_fd` is not provided, uses `psd` to get the filter.

    Parameters
    ----------
    data : ndarray 1d
        The vector that will be whitened

    whitening_filter_fd : ndarray 1d
        The whitening filter in Fourier domain which will be used for whitening.
        Should have the length corresponding to the data.

    psd : ndarray 1d
        The Power Spectrum Density that will be used to create a whitening filter if
        `whitening_filter_fd` is not provided.

    use_rfft : bool
        Whether to use rfft or ffr. If rfft is used, the signal is treated as real,
        and then the size of the `whitening_filter_fd` should be 1+len(`data`)//2.

    Returns
    -------
    whitened_data : ndarray 1d
        The data after application of the whitening filter.
    """
    data = np.array(data)
    assert not np.isnan(data).any(), 'There are holes in fluxes, cannot whiten!'

    if whitening_filter_fd is None:
        assert psd is not None, 'Whitening filter or PSD not specified!'
        whitening_filter_fd = psd2wf(psd, len(data), use_rfft=use_rfft)[1]

    if use_rfft:
        fftfunc = np.fft.rfft
        ifftfunc = np.fft.irfft
    else:
        fftfunc = np.fft.fft
        ifftfunc = np.fft.ifft

    whitened_data = ifftfunc(fftfunc(data) * whitening_filter_fd)
    return whitened_data


def whiten(noise, mask=None, other_vectors_to_whiten=None,
           nperseg=Nperseg_default,  detrend_low=True,
           use_rfft=Use_rfft, unpad_results=True, psd=None):
    """Wrapper to whiten the data. See documentation of the used methods

    Parameters
    ----------
    noise : ndarray 1d of float
        The vector whose PSD will ba used for whitening

    mask  : ndarray 1d of bool
        Marking with `False` all the points which should not be used in the PSD estimation.

    other_vectors_to_whiten : list[ndarray 1d] or ndarray 1d
        Other vectors apart from `noise` on which the whitening filter will be applied.
        It can be a template, for example.

    nperseg : int
        The length of the slice that will be used to estimate the PSD of the `noise`.

    detrend_low : bool
        Whether to remove the lowest frequencies which cannot be resolved by the PSD measurement

    use_rfft : bool
        Whether to use rfft or ffr. If rfft is used, the signal is treated as real,
        and then the size of the `whitening_filter_fd` should be 1+len(`data`)//2.

    unpad_results : bool
        Whether to remove zeros from ends of the data which are added in the process in
        order to avoid the crosstalks between the edges during the circular convolution.
        If `False`, then the padding mask will be returned

    Returns
    -------
    whitened data : ndarray 1d
    (on demand)
    """
    if mask is None:
        mask = ~np.isnan(noise)
    noise = np.array(noise)

    dim_other_vectors = 0
    if other_vectors_to_whiten is not None:
        other_vectors_to_whiten = np.array(other_vectors_to_whiten)
        dim_other_vectors = len(other_vectors_to_whiten.shape)
        if dim_other_vectors == 1:
            other_vectors_to_whiten = np.array([other_vectors_to_whiten])
        maxlen = np.max([len(v) for v in other_vectors_to_whiten])
        support = 2 ** int(np.ceil(np.log2(maxlen + nperseg)))
    else:
        support = 2 ** int(np.ceil(np.log2(len(noise) + nperseg)))

    noise, padding_mask = pad_by_zeros(noise, nperseg=nperseg, return_mask=True)
    mask = pad_by_zeros(mask, support=support, return_mask=False).astype(bool)
    if psd is not None:
        wf_td, wf_fd = psd2wf(psd, support=support, use_rfft=use_rfft)
    else:
        wf_fd, wf_td = measure_psd(noise, mask=padding_mask & mask, return_psd=False, return_freqs=False, use_rfft=use_rfft)

    if detrend_low:
        wf_fd[:int(np.ceil(len(noise) / nperseg))] = 0
    whitened_data = apply_whitening_filter(noise, whitening_filter_fd=wf_fd, use_rfft=use_rfft)
    if unpad_results:
        whitened_data = unpad(whitened_data, padding_mask)

    if other_vectors_to_whiten is not None:
        whitened_vectors = []
        other_vectors_padding_masks = []
        for other_vector in other_vectors_to_whiten:
            padded_vector, vector_pad_mask = pad_by_zeros(other_vector, support=support, return_mask=True)
            whitened_vector = apply_whitening_filter(padded_vector, whitening_filter_fd=wf_fd, use_rfft=use_rfft)
            if unpad_results:
                whitened_vector = unpad(whitened_vector, vector_pad_mask)
            else:
                other_vectors_padding_masks.append(vector_pad_mask)
            whitened_vectors.append(whitened_vector)
        if dim_other_vectors == 1:
            whitened_vectors = whitened_vectors[0]
            if not unpad_results:
                other_vectors_padding_masks = other_vectors_padding_masks[0]

        if unpad_results:
            return whitened_data, whitened_vectors
        else:
            return whitened_data, whitened_vectors, padding_mask, other_vectors_padding_masks

    else:
        if unpad_results:
            return whitened_data
        else:
            return whitened_data, padding_mask


def ml_amplitude_estimate(data, template, nperseg=Nperseg_default, transit_indices=None, psd_correction=False,
                          use_rfft=Use_rfft, position_is_known=True, psd=None):
    """

    :param data:
    :type data:
    :param template:
    :type template:
    :param nperseg:
    :type nperseg:
    :param transit_indices:
    :type transit_indices:
    :param psd_correction:
    :type psd_correction:
    :param use_rfft:
    :type use_rfft:
    :param position_is_known:
    :type position_is_known:
    :return:
    :rtype:
    """
    assert len(data) == len(template), 'Length of `data` and `template` do not match!'
    if use_rfft:
        fftfunc = np.fft.rfft
        ifftfunc = np.fft.irfft
    else:
        fftfunc = np.fft.fft
        ifftfunc = np.fft.ifft

    transit_mask = np.ones(len(template))
    if transit_indices is not None:
        if len(transit_indices) == 2:
            transit_indices = np.arange(transit_indices[0], transit_indices[1])
        transit_mask[transit_indices] = 0
        transit_len = len(transit_indices)
    else:
        transit_len = 0
    transit_mask = transit_mask.astype(bool)

    whitened_data, whitened_template = whiten(data, mask=transit_mask, other_vectors_to_whiten=template,
                                              nperseg=nperseg, unpad_results=True, psd=psd)

    if psd_correction:
        matched_filter_stat_raw = np.real(ifftfunc(np.conjugate(fftfunc(whitened_template)) * fftfunc(whitened_data)))
        var_stat_raw = np.nanvar(matched_filter_stat_raw[transit_len + nperseg * 2:
                                                         len(matched_filter_stat_raw) - (transit_len + nperseg * 2)])
        if position_is_known:
            amplitude_est = matched_filter_stat_raw[0] / var_stat_raw
        else:
            matched_filter_stat_max = matched_filter_stat_raw[np.argmax(np.abs(matched_filter_stat_raw))]
            amplitude_est = matched_filter_stat_max / var_stat_raw
    else:
        if position_is_known:
            amplitude_est = (whitened_template @ whitened_data) / (whitened_template @ whitened_template)
        else:
            matched_filter_stat_raw = np.real(ifftfunc(np.conjugate(fftfunc(whitened_template)) * fftfunc(whitened_data)))
            matched_filter_stat_max = matched_filter_stat_raw[np.argmax(np.abs(matched_filter_stat_raw))]
            amplitude_est = matched_filter_stat_max / (whitened_template @ whitened_template)

    return amplitude_est
