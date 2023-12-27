import numpy as np
import copy
import warnings
from scipy.optimize import minimize
from scipy.stats import chi2, norm, binom
from scipy.special import erfc, erfinv
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, tf2sos, tf2zpk
from scipy.signal.windows import tukey

from search_pipeline_parameters import *


def nanargsort(array):
    array = np.array(array)
    notnan_mask = ~np.isnan(array)
    argsort = np.argsort(array[notnan_mask])
    return np.arange(len(array))[notnan_mask][argsort]


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


def cos_vec(h1, h2, wf_fd=None, freqmask=None):
    """
    Calculate cosine angle between two vectors h1, h2, which are given in the time domain
    :param h1: First unwhitened vector in the time domain
    :type h1: ndarray(float)
    :param h2: Second unwhitened vector in the time domain, same length as `h1`
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
    # TODO: warning! freq mask is not participating in the normalization!
    norm_1 = np.sum(np.abs(np.fft.irfft(np.fft.rfft(h1, norm='ortho') * wf_fd * freqmask, norm='ortho', n=len(h1))) ** 2)
    norm_2 = np.sum(np.abs(np.fft.irfft(np.fft.rfft(h2, norm='ortho') * wf_fd * freqmask, norm='ortho', n=len(h2))) ** 2)
    return (np.sum(np.fft.irfft(np.fft.rfft(h1, norm='ortho') * wf_fd * freqmask, norm='ortho', n=len(h1))
                   * np.fft.irfft(np.fft.rfft(h2, norm='ortho') * wf_fd * freqmask, norm='ortho', n=len(h2)))
            / (norm_1 ** 0.5 * norm_2 ** 0.5))


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
    return 1 / (len(arr) - 1) * np.sum(np.abs(arr - np.mean(arr)) ** 2)


def moving_std(data, window_len, mask=None, gap_len=0):
    """NAIVE AND SLOW. USE IT ONLY FOR DEBUGGING OF MOVING_VAR"""
    if mask is None:
        mask = ~np.isnan(data)
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


def moving_var(signal, nwind, ngap=0, mask=None,
               nsamples_min=32,
               remove_outliers=False, nsig_outliers=6,
               detrend_remove_outliers=False, detr_length_outliers=512,
               return_mask=False, return_total_var=False):
    """
    Calculates the moving variance of the signal. At every point, takes the window of length
    nwind symmetrically from the two sides, excludes from the middle the window of length
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
    nwind : int
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
        assert len(signal) == len(mask), 'Data and mask have to be of the same length!'

    if remove_outliers:
        if detrend_remove_outliers:
            data1 = detrend_mov_ave(np.array(signal), length=detr_length_outliers, mask=mask,
                                    method='median', endmethod='mean', ends='symmetric', endlen=None,
                                    return_trend=False, keep_overall_level=False, remove_outliers=True,
                                    nsig_outliers=nsig_outliers)
        else:
            data1 = signal
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask

    tot_var = unb_var(signal[mask])

    signal[~mask] = 0.
    sum1 = np.nancumsum(signal * mask.astype(float))
    sum1 = np.concatenate(([0], sum1[:-1]))
    sum2 = np.nancumsum((signal * mask.astype(float)) ** 2)
    sum2 = np.concatenate(([0], sum2[:-1]))
    counts = np.nancumsum(mask)
    counts = np.concatenate(([0], counts[:-1]))
    mov_var = np.ones(len(signal)) * np.inf

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
    mask_counts = counts >= max(nsamples_min, 2)
    counts = counts.astype(float)  # for the next row
    counts[~mask_counts] = np.nan  # only for division, to avoid dividing by zero is there is some
    mov_var = (sum2 / counts - (sum1 / counts) ** 2) * counts / (counts - 1)
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


def gap_var(signal, mask=None, nwind=Nwind_moving_var_min, ngap=Ngap_moving_var_min,
            remove_outliers=False, detrend_remove_outliers=False, detr_length_outliers=Detr_length_outliers,
            nsig_outliers=Nsig_outliers_stat, return_mask=False, nsamples_min=Nwind_moving_var_min//2,
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
    :param signal: Data from which the moving variance is taken.
    :type signal: numpy.array 1d of float, NO NANS
    :param mask:
    :type mask:
    :param ngap: Length of the gap to exclude around each bin when calculating the
        variance in this bin in order to avoid influence of these
        points on the variance estimation.
    :type ngap: int
    :param remove_outliers:
    :type remove_outliers:
    :param detrend_remove_outliers:
    :type detrend_remove_outliers:
    :param detr_length_outliers:
    :type detr_length_outliers:
    :param nsig_outliers:
    :type nsig_outliers:
    :param return_mask:
    :type return_mask:
    :param nsamples_min:
    :type nsamples_min:
    :param adjustable_slices:
    :type adjustable_slices:
    :param return_total_var:
    :type return_total_var:
    :return: mov_var : Moving variance estimate at each point of the signal
    :rtype: numpy.array 1d of float, len(signal)
    """
    signal = np.array(signal)

    if mask is None:
        mask = ~np.isnan(signal)
    else:
        mask = np.array(mask)
        assert len(signal) == len(mask), 'Data and mask have to be of the same length!'

    if remove_outliers:
        if detrend_remove_outliers:
            data1 = detrend_mov_ave(np.array(signal), length=detr_length_outliers, mask=mask,
                                    method='median', endmethod='mean', ends='symmetric', endlen=None,
                                    return_trend=False, keep_overall_level=False, remove_outliers=True,
                                    nsig_outliers=nsig_outliers)
        else:
            data1 = signal
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask

    total_var = unb_var(signal[mask])

    # return np.ones(len(signal)) * total_var, total_var

    signal[~mask] = 0.
    sum1 = np.nancumsum(signal)
    sum1 = np.concatenate(([0], sum1))
    sum2 = np.nancumsum(signal ** 2)
    sum2 = np.concatenate(([0], sum2))
    summ = np.nancumsum(mask)
    summ = np.concatenate(([0], summ))
    mov_var = []

    hnw = nwind // 2
    hng = ngap // 2

    for i in np.arange(0, hng):
        counts = summ[-1] - summ[i + hng]
        if counts < nsamples_min:
            mov_var.append(total_var)
        else:
            mov_var.append(
                ((sum2[-1] - sum2[i + hng]) / counts - ((sum1[-1] - sum1[i + hng]) / counts) ** 2)
                * counts / (counts - 1))

    for i in range(hng, len(signal) - hng, ):
        counts = summ[-1] - summ[i + hng] + summ[i - hng]
        if counts < nsamples_min:
            mov_var.append(total_var)
        else:
            mov_var.append(((sum2[-1] - sum2[i + hng] + sum2[i - hng]) / counts
                            - (((sum1[-1] - sum1[i + hng] + sum1[i - hng]) / counts) ** 2))
                           * counts / (counts - 1))

    for i in np.arange(len(signal) - hng, len(signal)):
        counts = summ[i - hng]
        if counts < nsamples_min:
            mov_var.append(total_var)
        else:
            mov_var.append(((sum2[i - hng]) / counts
                            - ((sum1[i - hng]) / counts) ** 2)
                           * counts / (counts - 1))
    if not return_total_var:
        if not return_mask:
            return np.abs(np.array(mov_var))  # because small fluctuations around zero can occur
        else:
            return np.abs(np.array(mov_var)), mask
    else:
        if not return_mask:
            return np.abs(np.array(mov_var)), total_var  # because small fluctuations around zero can occur
        else:
            return np.abs(np.array(mov_var)), mask, total_var


def outlier_free_std(data, nsig=3, n_iterations=1, mask=None, detrend=False, detr_len=128):
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
        mask = ~np.isnan(data)
    if detrend:
        data1 = detrend_mov_ave(np.array(data), length=detr_len, mask=mask, method='median', endmethod='mean',
                                ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False)
    else:
        data1 = np.array(data)

    for i in range(n_iterations):
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig * np.nanstd(data1[mask])
        mask = outlier_mask * mask
    return np.nanstd(data[mask])


def pad_by_zeros(data, support=None, minimal_extension=0, in_the_center=False, return_zero_padding_ends=False):
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


def welch_psds(clean_fluxes, psd_masks, nperseg, noverlap, window, rfft=True, return_freqs=False, dt=1., return_npsds=False):
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
    if rfft:
        fftfun = np.fft.rfft
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
    psds = [np.mean(p, axis=0) for p in psds]
    if not return_freqs:
        if not return_npsds:
            return psds
        else:
            return psds, npsds
    else:
        if not return_npsds:
            return psds, [freqfun(nperseg, d=dt) for _ in range(len(psds))]
        else:
            return psds, [freqfun(nperseg, d=dt) for _ in range(len(psds))], npsds


def window_wf_td_fun(wf_fd,  extension_coef=2.5, meas_len=256):
    """
    applies the tukey window to one whitening template in the time domain to make it smooth but preserve the mean zero
    because this will ensure that the lowest frequency gets zero in the Fourier domain whitening filter.
    :param extension_coef:
    :type extension_coef:
    :return:
    :rtype:
        """
    wf_td = np.fft.irfft(wf_fd)
    wind = np.zeros(len(wf_td))
    len_to_save = meas_len  # it is 2 nperseg
    tuk = tukey(int(2 * extension_coef * len_to_save), alpha=1 - 1 / extension_coef, sym=True)
    wind[:len(tuk) // 2] = tuk[-len(tuk) // 2:]
    wind[-len(tuk) // 2:] = tuk[:len(tuk) // 2]
    wf_td = wf_td * wind
    wf_td -= np.mean(wf_td) * wind / np.mean(wind)
    wf_fd = np.abs(np.fft.rfft(wf_td))
    wf_td = np.fft.irfft(wf_fd)  # we need wf_fd for inpainting
    return wf_td, wf_fd


def psd2wf(psd, support=None, rfft=True, tukey_steps_len=0,
           conditional_zero_zeroth_freq=True, always_zero_zeroth_freq=False):
    """
    Generates a whitening filter of the needed length (support) from the PSD.
    Makes sinc-extrapolation, applies tuckey window to the end and to the low frequencies if they drop too quickly
    :param psd: measured PSD in coarse resolution
    :type psd: ndarray
    :param support: the desired length of the extrapolated filter (time domain). If None, take the length corresponding to the psd
    :type support: int
    :param rfft: whether to use rfft (or just fft). If true, the signal is treated as real,
                 and all ffts are rffts. The length of the frequency samples are the nperseg//2+1.
    :type rfft: bool
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
    if rfft:
        ifftfun = np.fft.irfft
        dtype_res = np.float64
        if support is None:
            support = 2 * (len(wf_fd_meas) - 1)
    else:
        if support == 'none':
            support = len(wf_fd_meas)
        ifftfun = np.fft.ifft
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

        wf_td = np.fft.fftshift(result)
        wf_fd = np.abs(np.fft.rfft(result))  # It is complex, which is weird, but what can we do
        if ((conditional_zero_zeroth_freq and (
                np.mean(np.abs(wf_fd[0:len(wf_fd) // 50])) / np.mean(np.abs(wf_fd[5 * len(wf_fd) // 10:7 * len(wf_fd) // 10])) < 0.3))
                or always_zero_zeroth_freq):
            first_measured_bin = int(np.ceil(support / meas_len))
            # wf_fd[:first_measured_bin] = 0.
            lentuk = 2 * first_measured_bin
            wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * tukey(lentuk, alpha=3 * first_measured_bin / lentuk,
                                                              sym=True)[:lentuk // 2]
            wf_td = np.fft.fftshift(ifftfun(wf_fd))
        wf_tds.append(wf_td)
        wf_fds.append(wf_fd)

    if was_vector:
        wf_tds = wf_tds[0]
        wf_fds = wf_fds[0]

    return wf_tds, wf_fds


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
        in order to omit calculation of covariance with Bins from the other side.
    trim_zeros : bool, optional
        Trim after padding the omitted zeros. The default is False.
    Returns
    -------
    np.array two-dimensional
        The covariance matrix (for the time domain).
        If vec is the time domain whitening filter, the result will be
        the INVERSE covariance matrix.
    """
    L = len(vec)
    if pad_by_zeros:
        vec = np.concatenate((vec, np.zeros(L)))
        L = len(vec)
    prod = np.array([np.sum(vec * np.roll(vec, i)) for i in range(L)])
    C = np.array([[prod[k - j] for k in range(L)] for j in range(L)])
    if pad_by_zeros and trim_zeros:
        assert L % 4 == 0, 'The length of the vector has to be even.'
        return C[L // 4:-L // 4, L // 4:-L // 4]
    return C


def inpaint_fluxes_fun(fluxes, masks, wf_tds, dtype=np.float32):
    length = len(fluxes[0])
    # I was afraid that wf_tds are fftshifted to the center, but since here we have the inner product, it is not important
    # fluxes = np.array(fluxes)  # , dtype=dtype)
    inpainted_fluxes = np.zeros((len(fluxes), length), dtype=dtype)
    for q in range(len(fluxes)):
        # filter2cov_matrix:
        prod = np.array([np.sum(wf_tds[q] * np.roll(wf_tds[q], i)) for i in range(length)])  # , dtype=dtype)
        c_inv = np.array([[prod[k - j] for k in range(length)] for j in range(length)])  # , dtype=dtype)

        a_t_c_inv = c_inv[~masks[q]]
        m_inv = np.linalg.inv(a_t_c_inv[:, ~masks[q]])
        f = np.zeros((length, length), dtype=dtype)
        f[~masks[q]] = np.dot(m_inv, a_t_c_inv)
        inpainted_fluxes[q] = fluxes[q] - np.dot(f, fluxes[q])
    return inpainted_fluxes


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
        'zeros' meens to replace by zeros.
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
        mask = ~np.isnan(flux)
    else:
        mask = np.array(mask).astype(bool)

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
        flux[indholes] = nan_average_filt(flux, kernel_size=kernel_size, method=interp_method)[indholes]
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
                for j in range(len(begholes)):
                    i1 = max(begholes[j] - mit_length + 1, 0)
                    i2 = min(endholes[j] + mit_length, len(flux1))
                    flux1[begholes[j] + 1:endholes[j]] = np.interp(
                        np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
                        np.array([np.nanmedian((flux1[i1:begholes[j] + 1])[np.where(mask1[i1:begholes[j] + 1])[0]]),
                                  np.nanmedian((flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            elif endhole_method == 'mean':
                for j in range(len(begholes)):
                    i1 = max(begholes[j] - mit_length + 1, 0)
                    i2 = min(endholes[j] + mit_length, len(flux1))
                    flux1[begholes[j] + 1:endholes[j]] = np.interp(
                        np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
                        np.array([np.nanmean((flux1[i1:begholes[j] + 1])[np.where(mask1[i1:begholes[j] + 1])[0]]),
                                  np.nanmean((flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            else:
                raise AssertionError('Incorrect endhole_method specified!')
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
            raise AssertionError('Unlefined ends parameter specified!')
    return flux


def sinc_interp(f_vec, support, tukey_steps_len=16):
    """
    Works for real-fft! returns real vector!
    """
    if tukey_steps_len < 1:
        tukey_steps_len = support * tukey_steps_len
    t_vec = np.fft.irfft(f_vec)
    meas_len = len(t_vec)
    t_vec = np.fft.ifftshift(np.fft.fftshift(t_vec) * tukey(meas_len, alpha=tukey_steps_len / meas_len, sym=True))
    mid_point = len(t_vec) // 2
    result = np.zeros(support, dtype=np.float64)  # f_vec.dtype)
    result[:mid_point] = t_vec[:mid_point]
    result[-mid_point:] = t_vec[-mid_point:]
    return np.abs(np.fft.rfft(result))


def tanhstep(x, center=0., hwhm=1., start_from_zero=False):
    if type(x) == list or type(x) == np.ndarray:
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
    data_low : numpy.array or list of float
        Smooth low-frequency trend of the `data`

    """
    assert fit_order == 'quadratic' or fit_order == 'cubic', 'Undefined `fit_order` specified!'

    data = np.array(data)
    time = np.arange(len(data))
    nanmask = ~np.isnan(data)
    if not (nanmask).all():
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
            dat1 = (data_i)[msk1]
            tim1 = (time)[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1) // 2]
            x10 = (tim1[-1] + tim1[0]) / 2
            x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2),
                           np.array([x00, x10, x20]))
            data_low = (sol.x[2] * (np.arange(len(data)) - sol.x[1]) ** 2 + sol.x[0])
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
            data_rf = np.fft.rfft(np.array(data_i))
            data_rf[int(len(data_i) / nperseg):] = 0.
            data_low = np.real(np.fft.irfft(data_rf, n=len(data_i)))
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

    Parameters
    ----------
    data : Union[ndarray, List[float]] # numpy.array or list of float
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
    threshold_step_size_std = residual_threshold_step_in_sigma  # redefining names
    medlen = slice_for_residual_step

    data = np.array(data)  # copying and going to numpy

    nanmask = ~np.isnan(data)
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

    fdat = np.abs(np.fft.rfft(data_i))
    std_fast_0 = (np.mean((fdat[-len(data) // 4:]) ** 2) / (len(data))) ** 0.5  # standard deviation of fase modes
    std_fast = std_fast_0 + np.max(fdat) / 20000  # I don't remember why exactly this was made but it worked

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
        # TODO: make here something to avoid repeating detrending if all is fine
        begholes = np.where(big_mask_p * np.roll(~big_mask_p, -1))[0] + 1  # define begholes inside the piece
        endholes = np.where(big_mask_p * np.roll(~big_mask_p, 1))[0]

        beg_small_pieces = [0]
        end_small_pieces = []
        for j in range(len(begholes)):
            if ((len(np.where(~np.isnan(piece[begholes[j] - medlen:begholes[j]]))[0]) >= 3)
                    and (len(np.where(~np.isnan(piece[endholes[j]:endholes[j] + medlen]))[0]) >= 3)
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

    Parameters
    --------------------------
    data : np.array 1d or list. Data sample from which the moving average has to be subtracted
    length : int. The length of sample to calculate moving average. At a given point,
                    half length to the left and to the right are taken.
    method : str. Kind of moving value to subtract: 'median' or 'mean'
                    The default is 'median', because it is less sensitive to outliers.
    endmethod: str. Kind of moving value to subtract at the ends of the data (see below): 'median' or 'mean'.
                    The default is 'mean', because it is less sensitive to small
                    amount of data samples available in the ends.
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
    assert ((method == 'mean') or (method == 'median')), 'Incorrect method specified'
    data = np.array(data)

    if mask is None:
        mask = ~np.isnan(data)

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

    if ends == 'asymmetric':
        trend = []
        res = []
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
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:i + int(l4 + (l2 - l4) / l2 * i)])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod == 'mean':
            for i in np.arange(0, l2):
                if (mask[0:i + int(l4 + (l2 - l4) / l2 * i)]).any():
                    m = np.nanmean((data[0:i + int(l4 + (l2 - l4) / l2 * i)]
                                   )[mask[0:i + int(l4 + (l2 - l4) / l2 * i)]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:i + int(l4 + (l2 - l4) / l2 * i)])
                    trend.append(m)
                    res.append(data[i] - m)
        if method == 'median':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        elif method == 'mean':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if endmethod == 'median':
            for i in np.arange(len(data) - l2, len(data)):
                if (mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]).any():
                    m = np.nanmedian((data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]
                                     )[mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod == 'mean':
            for i in np.arange(len(data) - l2, len(data)):
                if (mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]).any():
                    m = np.nanmean((data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]
                                    )[mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):])
                    trend.append(m)
                    res.append(data[i] - m)
    elif ends == 'symmetric':
        trend = []
        res = []
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
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:i + l4])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(l4, l2):
                if (mask[0:2 * i]).any():
                    m = np.nanmedian((data[0:2 * i])[mask[0:2 * i]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:2 * i])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod == 'mean':
            for i in np.arange(0, l4):
                if (mask[0:i + l4]).any():
                    m = np.nanmean((data[0:i + l4])[mask[0:i + l4]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:i + l4])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(l4, l2):
                if (mask[0:2 * i]).any():
                    m = np.nanmean((data[0:2 * i])[mask[0:2 * i]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:2 * i])
                    trend.append(m)
                    res.append(data[i] - m)
        if method == 'median':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    # trend.append(np.nan)
                    # res.append(np.nan)
                    m = np.nanmedian(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        elif method == 'mean':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if endmethod == 'median':
            for i in np.arange(len(data) - l2, len(data) - l4):
                if (mask[2 * i - len(data):]).any():
                    m = np.nanmedian((data[2 * i - len(data):])[mask[2 * i - len(data):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[2 * i - len(data):])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(len(data) - l4, len(data)):
                if (mask[i - l4:]).any():
                    m = np.nanmedian((data[i - l4:])[mask[i - l4:]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i - l4:])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod == 'mean':
            for i in np.arange(len(data) - l2, len(data) - l4):
                if (mask[2 * i - len(data):]).any():
                    m = np.nanmean((data[2 * i - len(data):])[mask[2 * i - len(data):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[2 * i - len(data):])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(len(data) - l4, len(data)):
                if (mask[i - l4:]).any():
                    m = np.nanmean((data[i - l4:])[mask[i - l4:]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - l4:])
                    trend.append(m)
                    res.append(data[i] - m)
    elif ends == 'cut':
        trend = []
        res = []
        l2 = int(np.ceil(length / 2))
        if method == 'median':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if method == 'mean':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
    else:
        raise AssertionError('Incorrect way to treat ends specified!')
    res = np.array(res)
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
        D = htol ** (1 / irl)  # amplitude of the pole (close to unity)
        dw = 2 * np.arctan(1 / D - 1)  # digital frequency width defined from it
        Q = w0 / dw / 2  # Quality factor, dimensionless parameter that characterizes
        # notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw
        # (for some reason, it works correctly with factor of 1/2 )
    else:  # if the impulse response length is not specified, use df
        if df is not None:
            Q = f0 / df / 2  # iQuality factor. In reality, Q gives w0/dw/2
            # for some reason, it works correctly with factor of 1/2
        else:
            Q = 30  # the default quality factor if no defining parameter is specified
    b, a = iirnotch(f0, Q, fs=fs)  # numerator and denominator polynomials of the IIR filter
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
                      detr_ends='symmetric',
                      detr_endlen=None, return_symmetric_mask=False):
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
        values will be marked as outliers). Is dominated by nsig_positive
        and nsig_negative.
    :type nsig0: float
    :param nsig1: Same as `nsig0` but for the second check
    :type nsig1:
    :param nsig_positive: Maximum positive deviation from the trend in units of std of data (more extreme
        values will be marked as outliers). Dominates over nsig. If is None,
        will be replaced by nsig.
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
    :param detr_ends: See the desctiption in detrend_mov_ave. The default is 'symmetric'.
    :type detr_ends: str, 'symmetric' or 'asymmetric' or 'cut'.
    :param detr_endlen: Minimal length to take feom the end for determining the average in the
        first (last) point. If None, will then be redefined
        depending on the detr_endmethod
    :type detr_endlen: int or None
    :param return_symmetric_mask:
    :type return_symmetric_mask:
    :return:
    :rtype:
    """

    data0 = np.array(data)
    nanmask = ~np.isnan(data)
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

    if detr_method is not None:
        data1 = detrend_mov_ave(np.array(data0), length=detr_length, mask=mask1, method=detr_method,
                                endmethod=detr_endmethod,
                                ends=detr_ends, endlen=detr_endlen, return_trend=False, keep_overall_level=False,
                                remove_outliers=False)
    else:
        data1 = data0

    outlier_mask = np.abs(data1 - np.nanmean(data1[mask1])) < nsig0 * np.nanstd(data1[mask1])
    mask1 *= outlier_mask  # first time to remove big outliers that spoil the std

    outlier_mask_p = (data1 - np.nanmean(data1[mask1])) < nsig_positive * np.nanstd(data1[mask1])
    outlier_mask_n = (data1 - np.nanmean(data1[mask1])) > - nsig_negative * np.nanstd(data1[mask1])
    if not return_symmetric_mask:
        return outlier_mask_p * outlier_mask_n
    else:
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask1])) < nsig1 * np.nanstd(data1[mask1])
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
    ind1 = (np.where(mask)[0])[0]
    ind2 = (np.where(mask)[0])[-1]
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
        Interpolated time. Equally sampled, no NaNs.
    """
    time2 = copy.copy(time)
    if mask is None:
        mask = ~np.isnan(time2)
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


def veto_combined(data, template, fn=0.01, maxdev_std=4., maxdev_ampl=None,
                  zero_ends_len_abs=3, zero_ends_len_rel=0.1,
                  return_amplitude=True, return_reason=True, inn_prod_min=0.2,
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
        spare_ind_left = min(zero_ends_len_abs, inds[0])
        inds = np.concatenate((np.arange(inds[0] - spare_ind_left, inds[0]), inds))
    if inds[-1] < len(template) - 1:
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
    p_value = chi2.sf(statistic, loc=0, scale=1, df=2)
    return p_value


def veto_step(data, indices_center_exclude=None, mask=None, min_len_to_process=3):
    if mask is None:
        mask = np.ones(len(data), dtype=bool)
    if indices_center_exclude is None:
        indices_all = np.arange(len(data))
        indices_left = indices_all[:len(data) // 2]
        indices_right = indices_all[len(data) // 2:]
    else:
        indices_left = np.arange(0, indices_center_exclude[0])
        indices_right = np.arange(indices_center_exclude[-1] + 1, len(data))
    indices_all = np.concatenate((indices_left, indices_right))
    data_left = data[indices_left][mask[indices_left]]
    data_right = data[indices_right][mask[indices_right]]
    if len(data_left) >= min_len_to_process and len(data_right) >= min_len_to_process:
        data_total = data[indices_all][mask[indices_all]]
        mean_total = np.mean(data_total)  #
        residuals_left = data_left - np.mean(data_left)
        residuals_right = data_right - np.mean(data_right)
        var_left = unb_var(residuals_left)
        var_right = unb_var(residuals_right)
        residuals_left_total = data_left - mean_total
        residuals_right_total = data_right - mean_total
        statistic = (- np.sum(residuals_left ** 2) / var_left - np.sum(residuals_right ** 2) / var_right
                     + np.sum(residuals_left_total ** 2) / var_left + np.sum(residuals_right_total ** 2) / var_right)
        p_value = chi2.sf(statistic, loc=0, scale=1, df=1)
        return p_value
    else:
        return None


def veto_parts(data_wh_fd, template_wh_full, indices_padded, mask_padded=None,
               tlen=None, hlen_if_short=5, fn=0.01, n_parts=4, nperseg_psd=Nperseg_psd,
               splitting='snr', lobe_hlen=5):
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
    :return: is_goods
    :rtype: ndarray[bool]
    """
    # for parts snr do the entire template
    # for parts len, do the manually selected template maybe? in the meas_len
    # make partial templates and matched-filter them. maybe indeed easier with the convolution than fft, although not sure

    # maybe make two different sets of splittings. for example, one of them may focus on the central petal of the whitened template
    # and just fit its shape. Yes, take a certain number of central points and make their fit.

    # make then partial veto in the frequency domain, split by snr. do fft of a small frequency range

    # Indices are assumed to be given in the padded coordinates!
    template_wh_full = np.array(template_wh_full)  # this is template in Time domain
    if mask_padded is None:
        mask_padded = ~np.ones(len(template_wh_full)).astype(bool)  # data is given in Fourier, do there will be no nans in time
    if splitting == 'snr':
        # reason = 'inconsistent_parts_time '
        if tlen is None:
            temp_smthnd = np.abs(template_wh_full - template_wh_full[len(template_wh_full) // 2])
            # -> the value in the middle is supposed to be close to zero, but sometines no
            temp_smthnd = (temp_smthnd + np.roll(temp_smthnd, 1) + np.roll(temp_smthnd, -1)) / 3
            # -> averaging the rolled template making it smooth
            inds_small = np.where((temp_smthnd / np.max(temp_smthnd) < 0.001))[0]  # choosing where the template is close to zero
            inds_small = inds_small[np.where((inds_small > nperseg_psd) & (inds_small <= len(template_wh_full) - nperseg_psd))[0]]
            # -> if suddenly there is zero in the beginning, ignore
            tlen = inds_small[0] + (len(temp_smthnd) - inds_small[-1])  # taking parts where the template is not close to zero
            # this weird procedure is needed because we are dealing with the whitened template, and its shape can be complex, so at some point
            # it can reach zero, but it does not mean that it is the end of the transit.
            # We are smoothening it, so that if it is zero only in one point,
            # the smoothened one will be nonzero.
            # then, we are setting a threshold and selecting everything non-zero
            # note the rolling! the transit center should be in the zeroth index!
        template_wh_full[tlen // 2: len(template_wh_full) - tlen // 2] = 0.
        template_wh_full = template_wh_full / np.linalg.norm(template_wh_full)
        cs = np.cumsum(template_wh_full ** 2)
        chi2_temp_indices = [0] + list(np.searchsorted(cs / cs[-1], (np.arange(n_parts) / n_parts)[1:])) + [len(template_wh_full)]
        # # this was len split, I'm removing it because I don't want to fit zeros.
        # nonzero_inds = np.nonzero(template_wh_full)[0]
        # chi2_temp_indices = list(nonzero_inds[(len(nonzero_inds) * np.arange(n_parts) / n_parts).astype(int)]
        #                          ) + [nonzero_inds[-1] + 1]
        chi2_temp_indices = list(set(chi2_temp_indices))  # remove repetitions
        chi2_temp_indices.sort()
        n_parts = len(chi2_temp_indices) - 1
        while ((chi2_temp_indices - np.roll(chi2_temp_indices, 1))[1:] < 4).any():  # want every piece to be at least 4 Bins
            n_parts -= 1
            cs = np.cumsum(template_wh_full ** 2)
            chi2_temp_indices = [0] + list(np.searchsorted(cs / cs[-1], (np.arange(n_parts) / n_parts)[1:])) + [len(template_wh_full)]
        # if the template is so short that there is only one part, we will just make one vetting on it and not parts.
        if len(chi2_temp_indices) <= 2:
            # reason = 'inconsistent_chi2_lobe '
            hlen = hlen_if_short
            data_wh_full = np.fft.irfft(data_wh_fd)
            template_for_fit = np.array(template_wh_full)
            template_for_fit[hlen: len(template_for_fit) - hlen] = 0
            template_for_fit = template_for_fit / np.linalg.norm(template_for_fit)
            template_cut_td = np.concatenate((template_for_fit[-hlen:], template_for_fit[: hlen]))
            template_for_fit = np.fft.rfft(template_for_fit)
            stat = np.real(np.fft.irfft(np.conjugate(template_for_fit) * data_wh_fd))
            stat = stat / outlier_free_std(stat[mask_padded], n_iterations=2, nsig=3)
            # -> this is supposed to ensure the template normalization, although formally this biases amplitude
            p_values = []
            for ind in indices_padded:
                residuals = (data_wh_full[ind - hlen: ind + hlen] - template_cut_td * stat[ind])
                residuals = residuals[mask_padded[ind - hlen: ind + hlen]]
                score = np.sum(residuals ** 2)
                df = len(residuals) - 1
                if df > 1:
                    p_values.append(chi2.sf(score, loc=0, scale=1, df=df))
                else:
                    p_values.append(0.)
            is_good = np.array(p_values) >= fn
        else:
            n_parts = len(chi2_temp_indices) - 1
            templates_parts = np.zeros((n_parts, len(template_wh_full)))
            for i in range(n_parts):
                templates_parts[i][chi2_temp_indices[i]: chi2_temp_indices[i+1]] = template_wh_full[chi2_temp_indices[i]:
                                                                                                    chi2_temp_indices[i+1]]
            norms_templates_parts = [np.linalg.norm(templates_parts[i]) for i in range(len(templates_parts))]
            templates_parts = np.fft.rfft(templates_parts, axis=1)
            scores = []
            for i in range(n_parts):
                stat = np.real(np.fft.irfft(np.conjugate(templates_parts[i]) * data_wh_fd))
                stat = stat / norms_templates_parts[i]  # outlier_free_std(stat[mask_padded], n_iterations=2, nsig=3)
                # -> this is supposed to ensure the templ normalization, although formally this biases amplitude
                # stat = stat[zero_padding_ends[0]: len(stat) - zero_padding_ends[1]]  # indices are provided in padded coordinates now!
                # this will give a biased amplitude, but since we only look at the amplitudes differences, maybe it is not so bad?
                scores.append(stat)
            stat_full = np.real(np.fft.irfft(np.conjugate(np.fft.rfft(template_wh_full)) * data_wh_fd))
            # -> norm is one, templ and data were norm
            stat_full = stat_full[indices_padded]
            scores = np.array(scores)[:, indices_padded]
            scores_parts = np.abs(scores - stat_full)
            p_values_parts = norm.sf(scores_parts, scale=(1 - 1 / n_parts) ** 0.5, loc=0) * 2
            is_good_parts = np.sum((p_values_parts >= fn / n_parts).astype(float), axis=0) >= n_parts // 2

            scores = np.sum(scores ** 2, axis=0) - stat_full ** 2
            p_values = chi2.sf(scores, loc=0, scale=1, df=n_parts - 1)
            is_good = np.array(p_values) >= fn
            is_good = is_good & is_good_parts
    # TODO measured_scale = (np.var(scores[mask_padded]) / (2 * (n_parts - 1))) ** 0.5 -> IF NEEDED
    elif splitting == 'lobe':
        # reason = 'veto_lobe '
        hlen = lobe_hlen
        data_wh_full = np.fft.irfft(data_wh_fd)
        template_for_fit = np.array(template_wh_full)
        template_for_fit[hlen: len(template_for_fit) - hlen] = 0
        template_for_fit = template_for_fit / np.linalg.norm(template_for_fit)
        template_cut_td = np.concatenate((template_for_fit[-hlen:], template_for_fit[: hlen]))
        template_for_fit = np.fft.rfft(template_for_fit)
        stat = np.real(np.fft.irfft(np.conjugate(template_for_fit) * data_wh_fd))
        stat = stat / outlier_free_std(stat[mask_padded], n_iterations=2, nsig=3)
        # -> this is supposed to ensure the template normalization, although formally this biases amplitude
        p_values = []
        for ind in indices_padded:
            residuals = (data_wh_full[ind - hlen: ind + hlen] - template_cut_td * stat[ind])
            residuals = residuals[mask_padded[ind - hlen: ind + hlen]]
            score = np.sum(residuals ** 2)
            df = len(residuals) - 1
            if df > 1:
                p_values.append(chi2.sf(score, loc=0, scale=1, df=df))
            else:
                p_values.append(0.)
        is_good = np.array(p_values) >= fn
    else:
        raise ValueError("`splitting` must be 'snr' or 'lobe'!")
    # reason = reason[:-1] + '_' + str(n_parts) + ' '
    return is_good  # , reason


def veto_parts_fourier(data_wh_fd, template_wh_fd, indices_padded, mask_padded=None, n_parts=3, fn=0.1,
                       wind_len_overlap=64):
    """

    The template in the Fourier domain is cut into several parts which undergo windowing in order not to be too long in the time
    domain. The minimal size of a piece is bounded by the window size. If the piece is too big, the number of pieces is decreased.

    :param data_wh_fd:data must be really white, including the constant multiple!
    :type data_wh_fd:
    :param template_wh_fd: template should be normalized to have unit norm in the time domain! I do not to waste time and renormalize it inside the functions!ß
    warnings: data can be blue because we oversuppress the low frequencies.
    :type template_wh_fd:
    :param indices_padded:
    :type indices_padded:
    :param mask_padded:
    :type mask_padded:
    :param n_parts:
    :type n_parts:
    :param fn:
    :type fn:
    :param wind_len_overlap: must be even!
    :type wind_len_overlap:
    :return:
    :rtype: ndarray[bool]
    """
    data_wh_fd = np.array(data_wh_fd)
    template_wh_fd_full = np.array(template_wh_fd)
    # template_wh_fd_full = template_wh_fd_full / np.linalg.norm(np.fft.irfft(template_wh_fd_full))  # must be normalized initially!
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
        return np.ones(len(indices_padded)).astype(bool)

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
    templates_parts_td = np.fft.irfft(templates_parts, axis=1)
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
        stat = np.real(np.fft.irfft(np.conjugate(templates_parts[i]) * data_wh_fd))
        # no need to apply any norm because it will be divided by the normalization afterwards
        scores[i] = stat
    stat_full = np.real(np.fft.irfft(np.conjugate(template_wh_fd_full) * data_wh_fd))
    # No need to renormalize, it is already in good norm because the template and the noise were normalized initially
    scores = (np.array([scores[:, j] @ templ_prod_matrix_inv @ scores[:, j] for j in range(len(scores[0]))]) - stat_full ** 2)
    # they all have the same names to save some memory (if it works like that...)
    measured_scale = (np.var(scores[mask_padded]) / (2 * (n_parts - 1))) ** 0.5
    # -> since the noise may be too blue, the distribution may not be with unit variance, so chi squared
    # will have a different scale.
    scores = scores[indices_padded]
    p_values = chi2.sf(scores, loc=0, scale=measured_scale, df=n_parts - 1)
    is_good = np.array(p_values) >= fn
    return is_good


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


def unequal_amplitude_veto(values, variances=None, A_all=None, indmax=None, A_1=None, units_of_snr=True):
    values = np.array(values)

    if variances is None:
        variances = np.ones(len(values))
    else:
        variances = np.array(variances)

    variances = variances[~np.isnan(values)]
    values = values[~np.isnan(values)]

    if A_all is None:
        A_all = np.sum(values / variances) / np.sum(1 / variances)
    if indmax is None:
        indmax = np.argmax(values)
    if A_1 is None:
        A_1 = values[indmax]

    expected_mean_H_all = sum(A_all ** 2 / variances) - A_1 * A_all / variances[indmax]
    expected_variance_H_all = np.abs(
        sum(A_all ** 2 / variances) - 2 * A_1 * A_all / variances[indmax] + A_1 ** 2 / variances[indmax])
    statistic = -(A_all * sum(values / variances) - A_1 * values[indmax] / variances[indmax] - expected_mean_H_all)
    if np.isclose(expected_variance_H_all, 0):
        statistic = np.inf
    elif units_of_snr:
        statistic = statistic / expected_variance_H_all ** 0.5
    else:
        statistic = statistic / expected_variance_H_all
    return statistic


def fstr(f, nzdig=2, maxdec=5, flexible_length=True):  # nicely formatted string from numbers
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


#######################################################################################################################
# Functions for reshuffled time slides #
#######################################################################################################################
def get_pieces_indices(mask, max_hole=Max_hole_pieces_scrampling, max_piece=Max_piece_scrambling, min_piece=Min_piece_scrambling):
    """
    Produces a way to split an array into pieces of lengths from `min_piece` till `max_piece`. The pieces are primarily taken
    as parts between the holes in the `mask` whose size is more than `max_hole`. If these pieces exceed `max_piece`, they will be
    cut randomly, so that the length of the part is between `min_piece` and `max_piece`.
    :param mask: mask in which holes will be considered breaks between the pieces
    :type mask: ndarray of bool
    :param max_hole:
    :type max_hole: int
    :param max_piece:
    :type max_piece: int
    :param min_piece:
    :type min_piece: int
    :return: inds_pieces, beginnings of the pieces and next indices after the ends of the pieces.
    :rtype: list[list[int,int]]
    """
    mask = np.array(mask).astype(bool)
    np.random.seed(seed=None)
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


def replace_if_deviating(vector, replacement, tester=None, mindev=1, minwid=1):
    sign = np.sign(mindev)
    mindev = np.abs(mindev)
    new_vector = np.array(vector)
    if tester is None:
        tester = replacement - vector
    i = 0
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


def replace_beginning(vector, replacement, min_repl_size):
    ind_temp = min_repl_size  # 2 * fmb_int * int(nperseg / npersegs_lim[i])
    sign_0 = np.sign(vector[ind_temp] - replacement[ind_temp])
    ind_temp += 1
    sign = np.sign(vector[ind_temp] - replacement[ind_temp])
    while sign == sign_0 and ind_temp < len(vector) - 1:
        ind_temp += 1
        sign = np.sign(vector[ind_temp] - replacement[ind_temp])
    vector[: ind_temp] = replacement[: ind_temp]
    return vector


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
        raise(ValueError('Undefined `distr` value!'))


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
        norm_factor = total_counts * (bins[1] - bins[0]) * 2  # factor 2 because both positive and begative values
        stds_predicted = (pbins * (1-pbins) * total_counts)**0.5
        if n_sigma is None:
            prob = times_per_bins/(len(bins)-1)  # something which happens one per bin
            n_sigma = norm.isf(prob/2, loc=loc, scale=scale)  # how far we can go to still be within this prob
        upper = expected + n_sigma * stds_predicted
        lower = expected - n_sigma * stds_predicted
        lower[lower<0] = 0
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
    stats: should be one-sided
    """
    stats = np.abs(stats)
    bins_linear = np.linspace(0, np.max(stats) + 0.5, n_bins)
    centr_bins_linear = bins_linear[:-1] + (bins_linear[1] - bins_linear[0]) / 2
    counts_linear, bins_linear = np.histogram(stats, bins=bins_linear, density=False)
    expected, upper, lower = expected_hist(np.sum(counts_linear), bins_linear)

    best_scale = 1.
    best_coef = 0.  # if the mixture is not needed, this will be the answer
    more_than_upper = (counts_linear - upper) > 0
    if len(np.where(more_than_upper)[0]) > 0:
        cum_more_than_upper = np.cumsum(more_than_upper)
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
        bins_quadr = np.linspace(0, np.max(stats_squared) + 0.5, n_bins)
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

            sol = minimize(lambda x: np.sum(
                (log_sf_init_guess - (x[0] + x[1] * bins_quad_init_guess)) ** 2 / log_errors_sf_hist_quadr_init_guess ** 2),
                           x0=np.array([0, -1]))
            # if not sol.success:
            #     print('initial guess linear fit failed!!!')
            best_scale = (1 / (-sol.x[1] * 2)) ** 0.5  # initial_guess_scale
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
                scale_grid = np.linspace(max(1.1, best_scale * co1), best_scale * co2, 20)
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

