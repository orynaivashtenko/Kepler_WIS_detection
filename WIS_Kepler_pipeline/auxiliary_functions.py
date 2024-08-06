#!/usr/bin/env python3

import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
import gzip
import shutil
import requests
import wget
from astropy.io import fits

import search_pipeline as spip
import utils
import utils as ut
from search_pipeline_parameters import *
import pipeline_parameters as pp

# TODO: some functions are copied from the old pipeline useful_functions.py. They shadow some methods of the search_pieline,
# TODO: for example pad_by_zeros, measure_psd, psd2wf. I need to reorganize them nicely.


def get_data_fun(target_id, location=Default_data_location):
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
    folder = get_correct_tid_folder(target_id, location, allow_creation=True)

    if len([ff for ff in os.listdir(folder) if '.fits' in ff]) < 1:
        webdir = 'https://archive.stsci.edu/pub/kepler/lightcurves/'
        # webdir = 'https://archive.stsci.edu/missions/kepler/lightcurves/'
        webfolder = webdir + target_id[:4] + '/' + target_id + '/'
        r = requests.get(webfolder)
        lines = [line for line in r.text.splitlines() if 'llc.fits' in line]
        urls = [webfolder + line[line.index('href="kplr' + target_id) + 6: line.index('.fits"') + 5] for line in lines]
        for url in urls:
            wget.download(url, folder)

    times = []
    fluxes = []
    files = [f for f in os.listdir(folder)[:] if 'llc.fits' in f]
    # if len(files)<17:
    #        warnings.warn('Number of files is not equal to the total number quarters. Available '+str(len(files)) + ' files')
    for f in files:
        filc = fits.open(os.path.join(folder, f))
        times.append(filc[1].data['TIME'])  # unit = 'BJD - 2454833'
        fluxes.append(filc[1].data['PDCSAP_FLUX'])
        filc.close()

    firstindices = []
    for tim in times:
        firstindices.append(np.argmax(~np.isnan(tim)))  # argmax is for finding the first instance of not nan
    i0 = np.max(firstindices)  # find the biggest initial index to call all times by it and make sure there is no nans
    # sort them by increasing time because the files in the directory are sorted at random:
    pairs = list(zip(times, fluxes))
    pairs.sort(key=lambda p: p[0][i0])
    times, fluxes = list(map(list, zip(*pairs)))
    return times, fluxes


def find_indices(time, first_time, period, first_index=None, return_times=False):
    if first_time is None:
        first_time = time[first_index]
    max_time = np.nanmax(time)
    expected_times = [first_time + i * period for i in range(int(np.ceil((max_time - first_time) / period)))]
    found_indices = [np.nanargmin(np.abs(time - expected_times[i])) for i in range(len(expected_times))]
    if not return_times:
        return found_indices
    else:
        return found_indices, expected_times


def connect_quarters_fun(time_quarters, data_quarters, zero_padding_ends=None, is_data_cut=False,
                         is_time_cut=False, dt=None, fill_gaps_by_nan=False):
    """
    Takes list of time arrays for each quarter and list of data arrays for aech quarter.
    Cuts zero padding ends. Generates new time array from the smallest up to the biggest
    time of time_quarters. Time quarters are assumed to be equally sampled with the same
    time step dt (will be measured if not given), and quarters are assumed not to be
    shifted with respect to each other by a non-integer number of Bins. If they
    are, they will be rounded to the closest integer Bins of dt.
    Data quarters are added to the positions of the resulting array corresponding
    to the first time bin of their quarter. The gaps are filled by zeros or NaNs

    Parameters
    ----------
    time_quarters : list of numpy.arrays 1d
        Time by quarters. Should be equally sampled with the same
        time step `dt` (will be measured if not given), and quarters are assumed
        not to be shifted with respect to each other by a non-integer number
        of Bins. If they are, they will be rounded to the closest integer Bins
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
        zero_padding_ends = [(0, 0)] * len(time_quarters)
    for i in range(len(time_quarters)):
        if np.isnan(time_quarters[i]).any():
            time_quarters[i] = ut.interpolate_time(np.array(time_quarters[i]))
    if dt is None:
        dts = []
        for j in range(len(time_quarters)):
            dts.append(np.abs(np.nanmean((time_quarters[j] - np.roll(time_quarters[j], 1))[1:])))
        dt = np.nanmean(dts)
    if not is_time_cut:
        t1 = time_quarters[0][zero_padding_ends[0][0]]
        t2 = time_quarters[-1][len(time_quarters[-1]) - 1 - zero_padding_ends[-1][-1]]
    else:
        t1 = time_quarters[0][0]
        t2 = time_quarters[-1][-1]
    time_connected = np.arange(t1, t2 + dt, dt)
    data_connected = np.zeros(len(time_connected))
    if fill_gaps_by_nan:
        data_connected = data_connected * np.nan

    for q in range(len(data_quarters)):
        if not is_data_cut:
            data_q = np.array(
                data_quarters[q][zero_padding_ends[q][0]:len(data_quarters[q]) - zero_padding_ends[q][-1]])
        else:
            data_q = np.array(data_quarters[q])
        if not is_time_cut:
            time_ind = int(np.round((time_quarters[q][zero_padding_ends[q][0]] - t1) / dt))
        else:
            time_ind = int(np.round((time_quarters[q][0] - t1) / dt))
        data_connected[time_ind:time_ind + len(data_q)] = np.array(data_q)
    return time_connected, data_connected


def get_values_from_txt(file_dir=None, filename=None):
    """Returns template bank templates durations in hours

    :param file_dir:
    :type file_dir:
    :param filename:
    :type filename:
    :return:
    :rtype:
    """
    if file_dir is None:
        file_dir = os.path.join(os.path.dirname(__file__), filename)
    with open(file_dir, 'r') as fil1:
        res = (fil1.read().strip()).split()
        res = np.array([float(val) for val in res])
    return res


def get_correct_tid_folder(target_id, location, allow_creation=False):
    """
    Looks around the given folder and finds or creates the correct directory,
    so that the structure is General_all_stars_folder/target_id[:4]/target_id[:4]
    :param target_id: KIC number
    :type target_id: str, will be adjusted to len 9
    :param location:
    :type location:
    :param allow_creation: Whether the function is allowed to create the needed folder if it does not exist
    :type allow_creation: bool
    :return:
    :rtype:
    """
    target_id = str(target_id).rjust(9, '0')

    if allow_creation:
        if os.path.split(location)[-1] == target_id:
            pass
        elif os.path.split(location)[-1] == target_id[:4]:
            if target_id not in os.listdir(location):
                os.mkdir(os.path.join(location, target_id))
            location = os.path.join(location, target_id)
        else:
            if target_id[:4] not in os.listdir(location):
                os.mkdir(os.path.join(location, target_id[:4]))
            location = os.path.join(location, target_id[:4])
            if target_id not in os.listdir(location):
                os.mkdir(os.path.join(location, target_id))
            location = os.path.join(location, target_id)
        if not os.path.exists(location):
            os.mkdir(location)
        return location
    else:
        if os.path.split(location)[-1] == target_id:
            pass
        elif os.path.split(location)[-1] == target_id[:4]:
            if target_id not in os.listdir(location):
                raise FileNotFoundError('No such file or directory: ' + os.path.join(location, target_id))
            else:
                location = os.path.join(location, target_id)
        else:
            if target_id[:4] not in os.listdir(location):
                raise FileNotFoundError('No such file or directory: ' + os.path.join(location, target_id[:4]))
            else:
                location = os.path.join(location, target_id[:4])
                if target_id not in os.listdir(location):
                    raise FileNotFoundError('No such file or directory: ' + os.path.join(location, target_id))
                else:
                    location = os.path.join(location, target_id)
        return location


def get_saving_file_path(output_location, target_id, filename_postfix, file_format):
    """

    :param output_location: The directory in which the output of this tarhet is saved
    :type output_location: str
    :param target_id:
    :type target_id: str
    :param filename_postfix:
    :type filename_postfix: str
    :param file_format:
    :type file_format: str
    :return:
    :rtype:
    """
    filename_postfix = filename_postfix.strip('_')
    if file_format[0] != '.':
        file_format = '.' + file_format
    if filename_postfix[-len(file_format):] != file_format:
        filename_postfix = filename_postfix + file_format
    if target_id is not None:
        file_dir = os.path.join(output_location, str(target_id) + '_' + filename_postfix)
    else:
        file_dir = os.path.join(output_location, str(target_id) + '_' + filename_postfix)
    return file_dir


def check_dict_keyword(target_id, output_folder, keyword='is_good_for_processing', dict_postfix=pp.Dict_target_txt_filename_postfix):
    """
    In the target_dict file, finds the keyword and returns the value of it.
    """
    output_folder = get_correct_tid_folder(target_id=target_id, location=output_folder, allow_creation=False)
    checked_file = os.path.join(output_folder, target_id + '_' + dict_postfix)
    with open(checked_file, 'r') as fil:
        lines = fil.readlines()
    lines = [line.strip().split(':') for line in lines if (len(line.strip()) > 0 and ':' in line)]
    lines = [[line[0].strip(), line[1].strip()] for line in lines]
    res = [line[1] for line in lines if line[0] == keyword]
    if len(res) == 0:
        raise ValueError('Keyword not found')
    else:
        return res[0]


def fold_fluxes(times, fluxes, period, first_time, masks=None, binwid=0.0204336 * 3,
                time_fold=400 * 0.0204336, dt=0.0204336, means=None, template=None, return_std=False):
    if not hasattr(times[0], "__len__"):
        times = [times]
    if not hasattr(fluxes[0], "__len__"):
        fluxes = [fluxes]

    if masks is None:
        masks = [~np.isnan(fl) for fl in fluxes]
    elif not hasattr(masks[0], "__len__"):
        masks = [masks]

    n_timesteps = int(np.round(time_fold / dt))
    # time_fold = n_timesteps * dt
    time_max = np.nanmax(times[-1])
    transit_times = [first_time + period * i for i in range(int((time_max - first_time) / period) + 1)]

    n_bins = int(np.ceil(time_fold / binwid))
    bins_flux = np.zeros(2 * (n_bins // 2) + 1)
    bins_flux_2 = np.zeros(2 * (n_bins // 2) + 1)
    bins_counts = np.zeros(2 * (n_bins // 2) + 1)
    bins_time = np.arange(2 * (n_bins // 2) + 1) * binwid - time_fold / 2 + binwid / 2
    folds_times = []
    folds_fluxes = []

    tt0 = transit_times[0]
    for q in range(len(times)):
        assert len(times[q]) == len(fluxes[q])
        assert len(times[q]) == len(masks[q])
        mask1 = np.array(masks[q])
        time1 = (np.array(times[q]) - tt0 + time_fold / 2) % time_fold
        if means is not None:
            flux1 = (fluxes[q] + means[q]) / means[q]
        else:
            flux1 = np.array(fluxes[q])
        time1 = time1[mask1]
        flux1 = flux1[mask1]
        folds_times.append(time1 - time_fold / 2)
        folds_fluxes.append(flux1)
        for i in range(len(time1)):
            ind = int(time1[i] // binwid)
            bins_flux[ind] += flux1[i]
            bins_flux_2[ind] += (flux1[i])**2
            bins_counts[ind] += 1

    # bins_flux = bins_flux / bins_counts
    inds1 = bins_counts >= 1
    bins_flux[inds1] = bins_flux[inds1] / bins_counts[inds1]
    # estimate of the variance of the measurements:
    bins_flux_2[inds1] = (bins_flux_2[inds1] - bins_flux[inds1] ** 2 * bins_counts[inds1]) / (bins_counts[inds1] - 1)
    bins_flux_2[inds1] = bins_flux_2[inds1] / bins_counts[inds1]  # the variance of the mean is like that
    bins_std = bins_flux_2**0.5

    if template is None:
        if return_std:
            return bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std
        else:
            return bins_time, bins_flux, bins_counts, folds_times, folds_fluxes
    else:
        template_cropped = np.zeros(n_timesteps)
        half_len = min(len(template) // 2, n_timesteps // 2)
        template_cropped[n_timesteps // 2 - half_len: n_timesteps // 2 + half_len] = template[len(template) // 2 - half_len: len(
            template) // 2 + half_len]
        tem_center = np.nanargmax(np.abs(template_cropped))
        time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(template_cropped) - tem_center))) * dt
        # time_template = (np.arange(n_timesteps) - np.argmax(np.abs(template_cropped))) * dt
        # time_template = np.arange(len(template_cropped)) * dt - time_fold / 2
        # template_cropped = template[len(template) // 2 - n_timesteps // 2:len(template) // 2 + n_timesteps // 2]
        # time_template = np.arange(len(template_cropped)) * dt - time_fold / 2
        if return_std:
            return bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped, bins_std
        else:
            return bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped


def mask_ephemeris(times, quarters, masks, big_hole_masks, dt_days,
                   period=None, durations=None, transit_times=None, absolute_phase=None, time_edges_to_mask=None):
    """
    Adds `False` to the mask where there are transits.
    :param times:
    :type times:
    :param quarters:
    :type quarters:
    :param masks:
    :type masks:
    :param big_hole_masks:
    :type big_hole_masks:
    :param dt_days:
    :type dt_days:
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
    if time_edges_to_mask is not None:
        assert hasattr(time_edges_to_mask, '__len__'), 'Variable time_edges_to_mask should have type list[list[float]]!'
        assert hasattr(time_edges_to_mask[0], '__len__'), 'Variable time_edges_to_mask should have type list[list[float]]!'
        for j in range(len(time_edges_to_mask)):
            ind1 = np.nan
            ind2 = np.nan
            for q in quarters:
                if (time_edges_to_mask[j][0] >= np.nanmin(times[q])) and (time_edges_to_mask[j][0] <= np.nanmax(times[q])):
                    ind1 = np.argmin(np.abs(times[q] - time_edges_to_mask[j][0]))
                if (time_edges_to_mask[j][1] >= np.nanmin(times[q])) and (time_edges_to_mask[j][1] <= np.nanmax(times[q])):
                    ind2 = np.argmin(np.abs(times[q] - time_edges_to_mask[j][1]))
                if (not np.isnan(ind1)) or (not np.isnan(ind2)):
                    masks[q][np.nanmax([0, ind1]):np.nanmin([0, len(times[q])])] = False
                    big_hole_masks[q][np.nanmax([0, ind1]):np.nanmin([0, len(times[q])])] = False
    else:
        if not hasattr(transit_times, '__len__'):  # then it has to be float, and we will produce array now
            min_time = np.nanmin(times[0])
            max_time = np.nanmax(times[-1])
            if transit_times is not None:  # it should be float then
                absolute_phase = transit_times % period
            assert (period is not None) and (absolute_phase is not None), 'At least some parameters should be not None!'
            n_min = np.ceil(min_time / period - absolute_phase / period)
            n_max = np.floor(max_time / period - absolute_phase / period)
            transit_times = np.array([absolute_phase + n * period for n in np.arange(n_min, n_max + 1)])
        if hasattr(durations, '__len__'):
            assert len(transit_times) == len(durations), 'Variables `transit_times` and `durations` should either have equal length ' \
                                                         'or `durations` should be float! '
        else:
            durations = np.ones(len(transit_times)) * durations

        for j in range(len(transit_times)):
            half_dur_bins = int(np.ceil(durations[j] / dt_days / 2))
            # -> we will mask this to the left and to the right from the midpoint
            for q in quarters:
                if (transit_times[j] >= np.nanmin(times[q])) and (transit_times[j] <= np.nanmax(times[q])):
                    ind1 = np.nanargmin(np.abs(times[q] - transit_times[j]))
                    masks[q][max(0, ind1 - half_dur_bins): min(len(times[q]), ind1 + half_dur_bins + 1)] = False
                    big_hole_masks[q][max(0, ind1 - half_dur_bins): min(len(times[q]), ind1 + half_dur_bins + 1)] = False
    return masks, big_hole_masks


def get_histogram_extrapolation_coefs_in_log(counts, bins, min_counts_to_include=3, right_from_max_to_left=1.5, counts_are_log=False):
    """Fitting coefficients to extrapolate log of a tail-gaussian histogram

    Fitting a straight line to the log Gaussian of argument squared to make extrapolation and estimate false positive rate for statistics
    scores for which extrapolation of the Monte-Carlo histogram is needed.
    Bins and counts used for fitting are where the counts are not zero and where they are greater than `min_counts_to_include`. This
    is needed in order not to use noisy counts. If `counts` is in units of density, then they are multiplied by the minimal value
    to make equivalent integer counts.
    In the distribution of maxima, the lowest scores are cut, and the log distribution is not straight line, that is why the leftmost
    Bins should not be used for extrapolation. They are rejected by finding the maximum of `counts` and taking the same distance to the
    right that to the left multiplied by right_from_max_to_left. This is done to avoid the curvature.
    Usually, `Bins` are snr squared, and `counts` are either in density units or not. The distribution is supposed to come from
    a maximization of Gaussians. If log(counts) is used, then `counts_are_log` should be set to True.

    :param counts: counts of the histogram in units of density or not
    :type counts: list or ndarray
    :param bins: edges of the histogram Bins
    :type bins: list or ndarray
    :param min_counts_to_include: minimal counts (in units of not density) in the bin for it to participate in the fitting.
    :type min_counts_to_include: int
    :param right_from_max_to_left: location of the lower cutting point of the data taken for fitting. Distance from max of the
    distribution to the right in units of the distance from the beginning till the max of the distribution.
    :type right_from_max_to_left: float
    :param counts_are_log: Whether `counts` are the actual counts or their log.
    :type counts_are_log: bool
    :return: `coef_0`, `coef_1` : fitting coefficients of the distribution log_counts(Bins). `coef_0` is the zero-order coefficient
    (supposed to be positive), `coef_1` is the linear coefficient (supposed to be negative)
    :rtype: tuple(float, float)
    """
    c0 = np.array(counts)
    b0 = np.array(bins)

    b_center = ((b0 + np.roll(b0, -1))[:-1]) / 2  # not the edges of the Bins, but the centers

    min_c0 = np.nanmin(c0[c0 > 0])
    if min_c0 < 1:  # if it was in the density units, put it to counts units to apply min_count
        valid_indices = np.where(c0 >= min_counts_to_include * min_c0)[0]
    else:
        valid_indices = np.where(c0 >= min_counts_to_include)[0]

    c_fit = c0[valid_indices]
    b_fit = b_center[valid_indices]

    first_index_counts = 0
    argmax_c0 = np.nanargmax(c_fit)
    if argmax_c0 > 0:
        first_index_counts = int((1 + right_from_max_to_left) * argmax_c0)
        # take the distance to the right a `right_from_max_to_left` times larger than from the left

    c_fit = c_fit[first_index_counts:]
    b_fit = b_fit[first_index_counts:]

    if not counts_are_log:
        log_c_fit = np.log(c_fit)
    else:
        log_c_fit = c_fit
    sol = minimize(lambda a: np.sum((log_c_fit - (a[0] + b_fit * a[1])) ** 2), np.array([1, -1]))
    return sol.x  # free scalar, linear coefficient


def is_bad_fft(fftdat, amp_max=300, dif_max=10, rfft=True):
    """

    If there are too narrow peaks that cannot be eliminated with measured psd, the star is bad
    ----------------------
    fftdat : np.array
    amp_max : float, in units of sigma_tail
        Limit amplitude of the fft maximum (above it the maximum is bad)
    dif_max : float, unitless
        Maximal difference between the value at point and the surrounding. Bigger is bad.
    rfft : bool
        Whether to use rfft. The signals are real, so we just always use it.
    """
    fftdat = np.array(np.abs(fftdat))
    if rfft:
        tail_sig = np.nanmedian(fftdat[len(fftdat) // 2:])
        flag = False
        if np.mean(np.sort(fftdat[8:])[-4:]) / tail_sig > amp_max:
            flag = True
        else:
            for i in range(8, len(fftdat) - 6):
                if ((fftdat[i] / np.mean(np.concatenate((fftdat[i - 5:i - 2], fftdat[i + 3:i + 6])))) > (
                        35 * (8 - 1) / (i - 1) - 35 * (8 - 1) / (50 - 1) + dif_max)
                        and (fftdat[i] > (3.5 * (8 - 1) / (i - 1) - 3.5 * (8 - 1) / (50 - 1) + 1) * 100 * tail_sig)):
                    flag = True
                    break
        return flag
    else:
        raise AssertionError('Please finish the code for rfft=False !')


def plot_many(list_times, list_values=None, first_index=0, prefix='quarter',
              legend=True, xlabel='', ylabel='', title='', separate=False, first_quarter=0):
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
        if separate == False:
            sep = np.zeros(len(list_times))
        elif separate == True:
            sep = [0]
            for i in range(len(list_times) - 1):
                sep.append(sep[-1] + np.max(list_values[i]) - np.min(list_values[i + 1]))
        else:
            sep = [i * separate for i in range(len(list_times))]
        for i in range(len(list_times)):
            plt.plot(np.real((list_times[i])[first_index:]),
                     np.real(sep[i] + (list_values[i])[first_index:]), label=f'{prefix}:{i + first_quarter}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    else:
        if separate == False:
            sep = np.zeros(len(list_times))
        elif separate == True:
            sep = [0]
            for i in range(len(list_times) - 1):
                sep.append(sep[-1] + np.max(list_times[i]) - np.min(list_times[i + 1]))
        else:
            sep = [i * separate for i in range(len(list_times))]
        for i in range(len(list_times)):
            plt.plot(np.real(sep[i] + (list_times[i])[first_index:]), label=f'{prefix}:{i + first_quarter}')
        plt.ylabel(ylabel)
        plt.title(title)
        if legend:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


def histogram(data, n=25, edgelist=None, edgemin=None, edgemax=None, xscale=None, yscale=None,
              density=False, normalize=True, ylogshift=True, logzeros='zero',
              return_results=False, plot=True, new_figure=True, ax=None, figsize=(9, 6), show=False,
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
    # TODO: fix scale in the log counts
    data = np.array(data[~np.isnan(data)])
    if xscale == 'square' or xscale == 'log':
        data = np.abs(data)

    if edgelist is None:
        if edgemin is None:
            edgemin = np.min(data)
        if edgemax is None:
            edgemax = np.max(data)
        edgelist = [edgemin + (edgemax - edgemin) / n * i for i in range(n + 1)]

    counts, bins = np.histogram(data, bins=edgelist, density=density)
    bins = np.delete(bins, -1) + (bins[1] - bins[0]) / 2

    if xscale == 'square':
        bins = np.abs(np.array(bins)) ** 2
    elif xscale == 'log':
        bins = np.log(np.abs(np.array(bins)))

    if (not density) and normalize:
        counts = counts / np.sum(counts)

    indzeros = None
    if yscale == 'square':
        counts = counts ** 2
    elif yscale == 'log':
        if logzeros == 'zero':
            indzeros = np.where(counts == 0)[0]
            logcounts = np.zeros(len(counts))
            logcounts[counts != 0] = np.log(counts[counts != 0])
            logcounts[counts == 0] = 0.
            counts = logcounts
        elif logzeros == 'nan':
            logcounts = np.zeros(len(counts))
            logcounts[counts != 0] = np.log(counts[counts != 0])
            logcounts[counts == 0] = np.nan
            counts = logcounts
        elif logzeros is None:
            counts = np.log(counts)
        else:
            raise AssertionError('Undefined logzeros parameter specified!')
        if ylogshift:
            sh = np.nanmin(counts[counts != -np.inf]) - (
                        np.nanmax(counts[counts != -np.inf]) - np.nanmin(counts[counts != -np.inf])) / 20
            counts = counts - sh
            if logzeros == 'zero':
                counts[indzeros] = 0

    if plot:
        if new_figure:
            plt.figure(figsize=figsize)
        if ax is None:
            ax = plt.gca()

        if plot_type == 'plot':
            ax.plot(bins, counts, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, alpha=alpha,
                    label=label)
        elif plot_type == 'bar':
            # binwids = (edgelist[-1]-edgelist[0])/(len(edgelist)-1)
            # ax.bar(Bins, counts, width=binwids ,color=color, alpha=alpha, label=label)
            # plt.step(Bins, counts, where='mid', color=color, alpha=alpha, label=label)
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


def get_target_pixel_data_fun(target_id, data_folder, remove_zeroth_quarter=False, remove_quarter_17=False, return_aperture_masks=False):
    target_id = str(target_id).rjust(9, "0")
    data_folder = get_correct_tid_folder(target_id=target_id, location=data_folder, allow_creation=True)

    # noinspection DuplicatedCode
    if len([ff for ff in os.listdir(data_folder) if '.fits' in ff]) < 1:
        download_target_pixel_files(target_id, data_folder)

    tp_fits_filenames = [ff for ff in os.listdir(data_folder) if '.fits' in ff]
    assert len(tp_fits_filenames) >= 1, 'There are no files!'

    times = []
    target_pixels = []
    aperture_masks = []
    for tp_fits_filename in tp_fits_filenames:
        tp_fits_file = fits.open(os.path.join(data_folder, tp_fits_filename))
        time = tp_fits_file[1].data['TIME']
        target_pix = tp_fits_file[1].data['FLUX']
        if return_aperture_masks:
            aperture_masks.append(tp_fits_file[2].data)
        tp_fits_file.close()
        times.append(time)
        target_pixels.append(target_pix)

    # noinspection DuplicatedCode
    reference_times = np.array([(times[q][~np.isnan(times[q])])[0] for q in range(len(times))])
    argsort = np.argsort(reference_times)
    times = [times[ind] for ind in argsort]
    target_pixels = [target_pixels[ind] for ind in argsort]
    if return_aperture_masks:
        aperture_masks = [aperture_masks[ind] for ind in argsort]

    if remove_zeroth_quarter:
        if len(times[0]) < 800:  # Make sure that the zeroth quarter is indeed the short zeroth quarter
            times = times[1:]
            target_pixels = target_pixels[1:]
            if return_aperture_masks:
                aperture_masks = aperture_masks[1:]

    if len(times) < 1:
        raise AssertionError('There is no data to process!')

    if remove_quarter_17:  # This quarter is small, there is a hola, and there might be glitch. we remove it
        if ((1570 >= np.nanmin(times[-1])) & (
                1570 <= np.nanmax(times[-1]))):  # Check that Q17 is here.
            times = times[:-1]
            target_pixels = target_pixels[:-1]
            if return_aperture_masks:
                aperture_masks = aperture_masks[:-1]
    if not return_aperture_masks:
        return times, target_pixels
    else:
        return times, target_pixels, aperture_masks


def get_target_pixel_sky_coords_fun(target_id, data_location, remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True,
                                    dtype=np.float32):
    """

    ra and dec are in decimal degrees
    """
    # import lightkurve as lk
    from lightkurve import targetpixelfile as tpf

    target_id = str(target_id).rjust(9, "0")
    data_folder = get_correct_tid_folder(target_id=target_id, location=data_location)

    # noinspection DuplicatedCode
    if len([ff for ff in os.listdir(data_folder) if '.fits' in ff]) < 1:
        download_target_pixel_files(target_id, data_folder)

    tp_fits_filenames = [ff for ff in os.listdir(data_folder) if '.fits' in ff]
    assert len(tp_fits_filenames) >= 1, 'There are no files!'

    times = []
    fluxes = []  # fluxes for all the pixels
    time_masks = []
    aperture_masks = []
    sky_coords = []
    for tp_fits_filename in tp_fits_filenames:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            tpf1 = tpf.KeplerTargetPixelFile(path=os.path.join(data_folder, tp_fits_filename))
            time_tpf = np.array(tpf1.time.value, dtype=dtype)
            flux_tpf = np.array(tpf1.flux.value, dtype=dtype)
            ra, dec = tpf1.get_coordinates(cadence='all')
            ra = np.array(ra, dtype=dtype)
            dec = np.array(dec, dtype=dtype)
            mask = np.array(tpf1.quality_mask)
            aperture_mask = tpf1.pipeline_mask
            del tpf1

        time = np.zeros(len(mask), dtype=dtype) * np.nan
        flux = np.zeros((len(mask), flux_tpf.shape[1], flux_tpf.shape[2]), dtype=dtype) * np.nan
        coords = np.zeros((2, len(mask), flux_tpf.shape[1], flux_tpf.shape[2]), dtype=dtype) * np.nan

        time[mask] = time_tpf
        flux[mask] = flux_tpf
        coords[0, mask] = ra
        coords[1, mask] = dec

        times.append(time)
        fluxes.append(flux)
        time_masks.append(mask)
        sky_coords.append(coords)
        aperture_masks.append(aperture_mask)

    # noinspection DuplicatedCode
    reference_times = np.array([(times[q][~np.isnan(times[q])])[0] for q in range(len(times))])
    argsort = np.argsort(reference_times)
    times = [times[ind] for ind in argsort]
    fluxes = [fluxes[ind] for ind in argsort]
    aperture_masks = [aperture_masks[ind] for ind in argsort]
    sky_coords = [sky_coords[ind] for ind in argsort]
    time_masks = [time_masks[ind] for ind in argsort]

    if remove_zeroth_quarter:
        if len(times[0]) < 800:  # Make sure that the zeroth quarter is indeed the short zeroth quarter
            times = times[1:]
            fluxes = fluxes[1:]
            aperture_masks = aperture_masks[1:]
            sky_coords = sky_coords[1:]
            time_masks = time_masks[1:]

    if len(times) < 1:
        raise AssertionError('There is no data to process!')

    if remove_quarter_17:  # This quarter is small, there is a hola, and there might be glitch. we remove it
        if ((1570 >= np.nanmin(times[-1])) & (
                1570 <= np.nanmax(times[-1]))):  # Check that Q17 is here.
            times = times[:-1]
            fluxes = fluxes[:-1]
            aperture_masks = aperture_masks[:-1]
            sky_coords = sky_coords[:-1]
            time_masks = time_masks[:-1]

    if cut_edge_nans:
        for q in range(len(times)):
            non_nans = np.where(~np.isnan(times[q]))[0]
            times[q] = times[q][non_nans[0]: non_nans[-1] + 1]
            fluxes[q] = fluxes[q][non_nans[0]: non_nans[-1] + 1]
            sky_coords[q] = sky_coords[q][:, non_nans[0]: non_nans[-1] + 1]
            time_masks[q] = time_masks[q][non_nans[0]: non_nans[-1] + 1]

    # times, fluxes, sky_coords, time_masks, aperture_masks
    return times, fluxes, sky_coords, time_masks, aperture_masks


def get_kepler_quality_masks(target_id, data_location, remove_zeroth_quarter=True, remove_quarter_17=True, cut_edge_nans=True):
    """

    ra and dec are in decimal degrees
    """
    # import lightkurve as lk
    from lightkurve import targetpixelfile as tpf

    target_id = str(target_id).rjust(9, "0")
    data_folder = get_correct_tid_folder(target_id=target_id, location=data_location)

    # noinspection DuplicatedCode
    if len([ff for ff in os.listdir(data_folder) if '.fits' in ff]) < 1:
        download_target_pixel_files(target_id, data_folder)

    tp_fits_filenames = [ff for ff in os.listdir(data_folder) if '.fits' in ff]
    assert len(tp_fits_filenames) >= 1, 'There are no files!'

    times = []
    time_masks = []
    for tp_fits_filename in tp_fits_filenames:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            tpf1 = tpf.KeplerTargetPixelFile(path=os.path.join(data_folder, tp_fits_filename))
            time_tpf = tpf1.time.value

        mask = np.array(tpf1.quality_mask)
        time = np.zeros(len(mask)) * np.nan

        time[mask] = time_tpf

        times.append(time)
        time_masks.append(mask)

    # noinspection DuplicatedCode
    reference_times = np.array([(times[q][~np.isnan(times[q])])[0] for q in range(len(times))])
    argsort = np.argsort(reference_times)
    times = [times[ind] for ind in argsort]
    time_masks = [time_masks[ind] for ind in argsort]

    if remove_zeroth_quarter:
        if len(times[0]) < 800:  # Make sure that the zeroth quarter is indeed the short zeroth quarter
            times = times[1:]
            time_masks = time_masks[1:]

    if len(times) < 1:
        raise AssertionError('There is no data to process!')

    if remove_quarter_17:  # This quarter is small, there is a hola, and there might be glitch. we remove it
        if ((1570 >= np.nanmin(times[-1])) & (
                1570 <= np.nanmax(times[-1]))):  # Check that Q17 is here.
            times = times[:-1]
            time_masks = time_masks[:-1]

    if cut_edge_nans:
        for q in range(len(times)):
            non_nans = np.where(~np.isnan(times[q]))[0]
            times[q] = times[q][non_nans[0]: non_nans[-1] + 1]
            time_masks[q] = time_masks[q][non_nans[0]: non_nans[-1] + 1]

    # times, fluxes, sky_coords, time_masks, aperture_masks
    return times, time_masks


def download_target_pixel_files(target_id, saving_folder):
    """
    # tp_fits_file = fits.open(os.path.join(data_folder, tp_fits_filename))
    # time = tp_fits_file[1].data['TIME']
    # target_pix = tp_fits_file[1].data['FLUX']
    # aperture_mask = tp_fits_file[2].data
    # tp_fits_file.close()

    :param target_id:
    :type target_id:
    :param saving_folder:
    :type saving_folder:
    :return:
    :rtype:
    """
    target_id = str(target_id).rjust(9, "0")
    saving_folder = get_correct_tid_folder(target_id=target_id, location=saving_folder, allow_creation=True)

    webdir = 'https://archive.stsci.edu/pub/kepler/target_pixel_files/'
    # webdir = 'https://archive.stsci.edu/missions/kepler/lightcurves/'
    webfolder = webdir + target_id[:4] + '/' + target_id + '/'
    r = requests.get(webfolder)
    lines = [line for line in r.text.splitlines() if 'lpd-targ.fits.gz' in line]
    urls = [webfolder + line[line.index('href="kplr' + target_id) + 6: line.index('.fits.gz"') + len('.fits.gz')] for line in lines]
    for url in urls:
        file = wget.download(url, saving_folder)
        ungzfile = file[:file.index('.gz')]
        with gzip.open(file, 'rb') as f_in:
            with open(ungzfile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file)


def get_max_pixels_sap(target_pixel_fluxes, number_of_max_pixels_to_remove=0):
    sap_fluxes = []
    for q in range(len(target_pixel_fluxes)):
        mean_stamp = np.nanmean(target_pixel_fluxes[q], axis=0)
        s1, s2 = np.shape(target_pixel_fluxes[q][0])
        msf = np.array(mean_stamp.flatten())
        msf[np.isnan(msf)] = 0

        # masking the needed amount of bright pixels
        ind_star = np.argsort(msf)[len(msf) - number_of_max_pixels_to_remove:]
        mask_pixels = np.ones(len(msf))
        mask_pixels[ind_star] = 0
        mask_pixels = mask_pixels.reshape((s1, s2))

        sap = np.nansum(target_pixel_fluxes[q] * mask_pixels, axis=(1, 2))
        sap[sap == 0] = np.nan  # THIS HAS FAILURE MODES
        sap_fluxes.append(sap)
    return sap_fluxes


def get_sap_from_pixels(target_pixel_fluxes, bit_aperture_masks=None, custom_masks=None):
    """
    Bit 1 - pixel was collected by S/C
    Bit 2 - pixel is in the optimal aperture
    :param target_pixel_fluxes:
    :type target_pixel_fluxes:
    :param bit_aperture_masks:
    :type bit_aperture_masks:
    :param custom_masks:
    :type custom_masks:
    :return:
    :rtype:
    """

    sap_fluxes = []
    for q in range(len(target_pixel_fluxes)):
        if custom_masks is not None:
            summation_mask = custom_masks[q].flatten()
        elif bit_aperture_masks is not None:
            s1, s2 = np.shape(bit_aperture_masks[q])
            summation_mask = bit_aperture_masks[q].flatten()
            # go to binary representation and take the first bit to understand
            # whether the pixel was in the optimal aperture
            summation_mask = [utils.dec_to_bin(e, return_list=True) for e in summation_mask]
            summation_mask = np.array([e[1] if len(e) > 1 else 0 for e in summation_mask]).astype(float)
            summation_mask = summation_mask.reshape((s1, s2))
        else:
            summation_mask = np.ones(np.shape(target_pixel_fluxes[q][0]))

        sap = np.nansum(target_pixel_fluxes[q] * summation_mask, axis=(1, 2))
        sap[sap == 0] = np.nan  # THIS HAS FAILURE MODES
        sap_fluxes.append(sap)
    return sap_fluxes


def get_splitted_saps_from_pixels(target_pixel_fluxes, number_of_pixels_cut=None, part_of_cumsum_2=0.8, part_of_light=None):
    star_part_sap_fluxes = []
    bg_part_sap_fluxes = []
    for q in range(len(target_pixel_fluxes)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_stamp = np.nanmean(target_pixel_fluxes[q], axis=0)
        flatten_mean_stamp = mean_stamp.flatten()
        nanmask = ~np.isnan(flatten_mean_stamp)
        good_flatten_indices = np.argsort(flatten_mean_stamp[nanmask])[::-1]
        good_flatten_mean_stamp = flatten_mean_stamp[nanmask][good_flatten_indices]
        selected_indices = np.arange(len(nanmask))[nanmask][good_flatten_indices]

        if number_of_pixels_cut is not None:
            sep_ind = number_of_pixels_cut
        elif part_of_cumsum_2 is not None:
            cmsm_lght_nrm = np.cumsum(good_flatten_mean_stamp ** 2)
            cmsm_lght_nrm = cmsm_lght_nrm / cmsm_lght_nrm[-1]
            sep_ind = np.argmin(np.abs(cmsm_lght_nrm - part_of_cumsum_2))
        else:
            cmsm_lght_nrm = np.cumsum(good_flatten_mean_stamp)
            cmsm_lght_nrm = cmsm_lght_nrm / cmsm_lght_nrm[-1]
            sep_ind = np.argmin(np.abs(cmsm_lght_nrm - part_of_light))
        star_part_indices = selected_indices[:sep_ind + 1]
        bg_part_indices = selected_indices[sep_ind + 1:]

        sh0 = target_pixel_fluxes[q].shape
        tpf_rsh = target_pixel_fluxes[q].reshape((sh0[0], sh0[1] * sh0[2]))

        star_part_sap_flux = np.nansum(tpf_rsh[:, star_part_indices], axis=-1)
        nanind = np.where(star_part_sap_flux == 0)[0]
        if len(nanind) > 0:
            star_part_sap_flux[nanind] = star_part_sap_flux[nanind] * np.nan
        bg_part_sap_flux = np.nansum(tpf_rsh[:, bg_part_indices], axis=-1)
        nanind = np.where(bg_part_sap_flux == 0)[0]
        if len(nanind) > 0:
            bg_part_sap_flux[nanind] = star_part_sap_flux[nanind] * np.nan

        star_part_sap_fluxes.append(star_part_sap_flux)
        bg_part_sap_fluxes.append(bg_part_sap_flux)
    return star_part_sap_fluxes, bg_part_sap_fluxes


def estimate_ampls_ppms_from_stats(summed_raw_stats, summed_stat_vars, templ_max, flux_trend, trends_std=None,
                                   valid_transit_mask=None):
    """
    :param summed_raw_stats:
    :type summed_raw_stats:
    :param flux_trend:
    :type flux_trend:
    :param summed_stat_vars:
    :type summed_stat_vars:
    :param templ_max: depth of the template that was normalized to one by the L2 norm
    :type templ_max:
    :param trends_std:
    :type trends_std:
    :param valid_transit_mask:
    :type valid_transit_mask:
    :return:
    :rtype:
    """
    if hasattr(templ_max, '__len__'):
        templ_max = templ_max / np.linalg.norm(templ_max)
        templ_max = np.abs(templ_max)
        templ_max = np.nanmax(templ_max)
    else:
        templ_max = np.abs(templ_max)

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)

    # There is an option to specify trend for every transit, but it can also be the same if all were renormalized:
    if not hasattr(flux_trend, '__len__'):
        flux_trend = np.ones(len(summed_raw_stats)) * flux_trend
    if trends_std is None:
        trends_std = np.zeros(len(summed_raw_stats))

    not_missed_transit_mask = (~np.isnan(summed_raw_stats)) & (flux_trend > 0.) & (summed_raw_stats != 0.) & (summed_stat_vars > 0.)
    if valid_transit_mask is not None:
        not_missed_transit_mask = not_missed_transit_mask & valid_transit_mask

    mean_trend = np.nanmean(flux_trend[not_missed_transit_mask])
    mean_trend_std = (np.nanmean(trends_std[not_missed_transit_mask] ** 2) + np.nanvar(flux_trend[not_missed_transit_mask])) ** 0.5

    ampls = np.zeros(len(summed_raw_stats)) * np.nan
    ampls_stds = np.zeros(len(summed_raw_stats)) * np.nan
    normalized_ampls = np.zeros(len(summed_raw_stats)) * np.nan
    ppms = np.zeros(len(summed_raw_stats)) * np.nan
    ppms_stds = np.zeros(len(summed_raw_stats)) * np.nan

    ampls[not_missed_transit_mask] = summed_raw_stats[not_missed_transit_mask] / summed_stat_vars[not_missed_transit_mask]
    ampls_stds[not_missed_transit_mask] = (np.abs(1 / summed_stat_vars[not_missed_transit_mask])**0.5)
    normalized_ampls[not_missed_transit_mask] = (ampls[not_missed_transit_mask] / flux_trend[not_missed_transit_mask] * mean_trend)
    normalized_ampls_stds = ((ampls_stds * mean_trend / flux_trend)**2
                             + (trends_std * ampls * mean_trend / flux_trend ** 2)**2
                             + (mean_trend_std * ampls / flux_trend)**2)**0.5
    ppms[not_missed_transit_mask] = (ampls[not_missed_transit_mask] / flux_trend[not_missed_transit_mask] * templ_max * 1e6)

    ampl = (np.nanmean((ampls[not_missed_transit_mask] / flux_trend[not_missed_transit_mask]))
            * np.nanmean(flux_trend[not_missed_transit_mask]))
    ppm = np.nanmean(ppms[not_missed_transit_mask])
    ppms_stds[not_missed_transit_mask] = np.abs(ppms[not_missed_transit_mask]) * (
            (ampls_stds[not_missed_transit_mask] / ampls[not_missed_transit_mask])**2
            + (trends_std[not_missed_transit_mask] / flux_trend[not_missed_transit_mask])**2)**0.5
    # ppms_stds[not_missed_transit_mask] = (ppms[not_missed_transit_mask])**2 * (
    #     (ampls_stds[not_missed_transit_mask])**2 / (ampls[not_missed_transit_mask])**2
    #     + np.var(flux_trend[not_missed_transit_mask]) / (flux_trend[not_missed_transit_mask])**2)
    # ppms_stds[not_missed_transit_mask] = ((1 / summed_stat_vars[not_missed_transit_mask] / (flux_trend[not_missed_transit_mask]) ** 2
    #                                        + np.var(flux_trend[not_missed_transit_mask])
    #                                        / flux_trend[not_missed_transit_mask] ** 4 * templ_max ** 2 * 1e12) ** 0.5)
    return ampl, ppm, ampls, ampls_stds, normalized_ampls, normalized_ampls_stds, ppms, ppms_stds


def get_centroid_series(target_pixel_fluxes, sky_coords=None, aperture_masks=None, time_masks=None):
    """
    see https://github.com/lightkurve/lightkurve/blob/main/src/lightkurve/targetpixelfile.py, the `estimate_centroids` function to learn
    how they do it. They have the simple centroid, same as here, and also some polynomial-fitting one.

    :param target_pixel_fluxes: fluxes for all the quarters for all the times for all the pixels
    :type target_pixel_fluxes: list of numpy.ndarrays of shape (len_time, pixels_dim_1, pixels_dim_2) of float
    :param sky_coords: right ascensions and declinations for all the quarters for all the times for all the pixels.
        If None, pixel indices (not true detector coordinates!) will be used
    :type sky_coords: None or list of numpy.ndarrays of shape (2, len_time, pixels_dim_1, pixels_dim_2)
    :param aperture_masks: mask of pixels that were used in the optimal aperture to calculate the SAP flux.
    :type aperture_masks: list of numpy.ndarrays of shape (pixels_dim_1, pixels_dim_2)
    :param time_masks: mask of times that are bad, including the Kepler quality mask
    :type time_masks: list of numpy.ndarrays of shape (len_time)
    :return:
    :rtype:
    """
    centroids_x = []  # ra if sky_coords is specified
    centroids_y = []  # dec if sky_coords is specified
    for q in range(len(target_pixel_fluxes)):
        target_pixels_flux = target_pixel_fluxes[q]
        nan_aperture_masks = ~((np.isnan(target_pixels_flux)).all(axis=0))
        if aperture_masks is None:
            aperture_mask = nan_aperture_masks
        else:
            aperture_mask = (aperture_masks[q] * nan_aperture_masks).astype(bool)

        nan_mask = ~((np.isnan(target_pixels_flux)).all(axis=(1, 2)))
        if time_masks is None:
            time_mask = nan_mask
        else:
            time_mask = (time_masks[q] * nan_mask).astype(bool)

        if sky_coords is None:
            y_coor, x_coor = np.indices(np.shape(target_pixels_flux)[1:3])
            x_coor = x_coor * np.ones(np.shape(target_pixels_flux), dtype=float)
            y_coor = y_coor * np.ones(np.shape(target_pixels_flux), dtype=float)
        else:
            y_coor, x_coor = sky_coords[q][1], sky_coords[q][0]

        sap_flux = np.zeros(len(target_pixels_flux)) * np.nan
        centr_x = np.zeros(len(target_pixels_flux)) * np.nan
        centr_y = np.zeros(len(target_pixels_flux)) * np.nan
        sap_flux[time_mask] = np.nansum(target_pixels_flux[time_mask][:, aperture_mask], axis=1)
        centr_x[time_mask] = np.nansum((x_coor[time_mask] * target_pixels_flux[time_mask])[:, aperture_mask], axis=1) / sap_flux[
            time_mask]
        centr_y[time_mask] = np.nansum((y_coor[time_mask] * target_pixels_flux[time_mask])[:, aperture_mask], axis=1) / sap_flux[
            time_mask]
        # centr_x[time_mask] = np.nansum((aperture_mask * x_coor[time_mask] * target_pixel_fluxes[q][time_mask]),
        # axis=(1, 2)) / sap_flux[time_mask]
        # centr_y[time_mask] = np.nansum((aperture_mask * y_coor[time_mask] * target_pixel_fluxes[q][time_mask]),
        # axis=(1, 2)) / sap_flux[time_mask]
        centroids_x.append(centr_x)
        centroids_y.append(centr_y)
    return centroids_x, centroids_y


# def find_snr_at_transit_times(signal_quarters, time_quarters, summed_transit_times, nsig_outliers_0_pos=5, nsig_outliers_0_neg=15,
#                               max_abs=True, time_mismatch_allowed=3):
#     """
#     THIS IS NOT USED BECAUSE WE SWITCHED TO JUST A SHORT PERIODICITY SEARCH IN THE TCE CLASS
#     TODO: look at nsig
#     TODO: which template?
#     TODO: now we have ra, dec, so we can summarize quarters!
#     TODO: maybe periodicity?
#     :param signal_quarters:
#     :type signal_quarters:
#     :param time_quarters:
#     :type time_quarters:
#     :param summed_transit_times:
#     :type summed_transit_times:
#     :param nsig_outliers_0_pos:
#     :type nsig_outliers_0_pos:
#     :param nsig_outliers_0_neg:
#     :type nsig_outliers_0_neg:
#     :param max_abs:
#     :type max_abs:
#     :param time_mismatch_allowed: is specified in bins! (very stupid name then!)
#     :type time_mismatch_allowed: int
#     :return:
#     :rtype:
#     """
#     pass
#     # # making the lightcurve out of this whatever data and doing search:
#     # klc = spip.KeplerLightCurve(target_id=None, get_data=False, times=time_quarters, fluxes=signal_quarters)
#     # klc.condition_data(nsig_outliers_0=nsig_outliers_0_neg, nsig_outliers_0_pos=nsig_outliers_0_pos,
#     #                    nsig_outliers_0_neg=nsig_outliers_0_neg,
#     #                    nsig_outliers_w=nsig_outliers_0_neg, nsig_outliers_w_pos=nsig_outliers_0_pos,
#     #                    nsig_outliers_w_neg=nsig_outliers_0_neg)
#     # klc.condition_templates()
#     # klc.matched_filtering()
#     # klc.cut_zero_padding_ends()
#     #
#     # klc.templates_of_indices = [k for k in range(len(klc.indices_of_templates))
#     #                             for _ in range(klc.indices_of_templates[k][0], klc.indices_of_templates[k][1])]
#
#     # max_stats = []
#     # std_stat_maxs = []
#     # max_templs = []
#     # # max_templs_sh = []
#     # # times_max = []
#     # for i, tt in enumerate(summed_transit_times):
#     #     max_stat_to_std = 0.
#     #     max_stat = None
#     #     std_stat_max = None
#     #     templ_sh_ind = None
#     #     # max_templ = None
#     #     for q in klc.quarters:
#     #         if klc.times[q][~np.isnan(klc.times[q])][0] <= tt <= klc.times[q][~np.isnan(klc.times[q])][-1]:
#     #             ind_tr = np.nanargmin(np.abs(klc.times[q] - tt))
#     #             for k in range(len(klc.statistics_snr)):
#     #                 std_stat = ut.outlier_free_std(klc.statistics_snr[k][q], nsig=3, mask=klc.masks[q], detrend=False, detr_len=128)
#     #                 ti0 = max(0, ind_tr - time_mismatch_allowed)
#     #                 if max_abs:
#     #                     time_index = ti0 + np.nanargmax(np.abs(klc.statistics_snr[k][q][ti0:ind_tr + time_mismatch_allowed]))
#     #                     # -> allow the actual minimum to be a little shifted
#     #                     stat_here = klc.statistics_snr[k][q][time_index]
#     #                     stat_to_std_here = stat_here/std_stat
#     #                     if np.abs(stat_to_std_here) >= np.abs(max_stat_to_std):
#     #                         max_stat = stat_here
#     #                         templ_sh_ind = k
#     #                         std_stat_max = std_stat
#     #                         max_stat_to_std = stat_to_std_here
#     #                         # time_max = klc.times[q][time_index]
#     #                 else:
#     #                     time_index = ti0 + np.nanargmin(klc.statistics_snr[k][q][ti0:ind_tr + time_mismatch_allowed])
#     #                     # -> allow the actual minimum to be a little shifted
#     #                     stat_here = klc.statistics_snr[k][q][time_index]
#     #                     stat_to_std_here = stat_here / std_stat
#     #                     if stat_to_std_here <= max_stat_to_std:
#     #                         max_stat = stat_here
#     #                         templ_sh_ind = k
#     #                         std_stat_max = std_stat
#     #                         max_stat_to_std = stat_to_std_here
#     #                         # time_max = klc.times[q][time_index]
#     #     if templ_sh_ind is not None:  # it can be for example if the expected transit happened between the quarters
#     #         max_templ = klc.templates_of_indices[templ_sh_ind]
#     #         max_stats.append(max_stat)
#     #         std_stat_maxs.append(std_stat_max)
#     #         max_templs.append(max_templ)
#     #         # times_max.append(time_max)
#     #         # max_templs_sh.append(templ_sh_ind)
#     #     else:
#     #         max_stats.append(np.nan)
#     #         std_stat_maxs.append(np.nan)
#     #         max_templs.append(np.nan)
#     # return max_stats, std_stat_maxs, max_templs


def make_bool_aperture_mask(bit_aperture_mask, bit=1):
    """
    Receives pixel quality mask with integers encoding quality flag in bits and
    extracts one `bit` information from every integer making a bool mask from it.
    Bit 0 is whether the pixel was taken,
    Bit 1 is whether it was used in the optimal aperture.
    :param bit_aperture_mask: pixel quality mask with integers encoding quality flag in bits
    :type bit_aperture_mask: ndarray 2d of int
    :param bit: number of bit which needs to be extracted from every integer
    :type bit: int
    :return: bool_mask
    :rtype: ndarray 2d of bool
    """
    s1, s2 = np.shape(bit_aperture_mask)
    bool_mask = [ut.dec_to_bin(e, return_list=True) for e in bit_aperture_mask.flatten()]
    bool_mask = np.array([e[bit] if len(e) > 1 else 0 for e in bool_mask]).astype(bool)
    bool_mask = bool_mask.reshape((s1, s2))
    return bool_mask


# def get_single_pixels_stats(times, target_pixel_fluxes, bit_aperture_masks, summed_transit_times, best_temp,
#                             bins_mismatch_allowed=2):
#     # I do not remember what I intended to do with this. There was some  veto, but I do not remember what I wanted.
#     # Make optimal aperture masks by taking bin 1 value
#     aperture_masks = [make_bool_aperture_mask(msk, bit=1) for msk in bit_aperture_masks]
#
#     target_pixels_flt = []
#     aperture_masks_flt = []
#     for q in range(len(target_pixel_fluxes)):
#         sh0 = target_pixel_fluxes[q].shape
#         target_pixels_flt.append(target_pixel_fluxes[q].reshape((sh0[0], sh0[1] * sh0[2])))
#         aperture_masks_flt.append((aperture_masks[q].reshape((1, sh0[1] * sh0[2])))[0])
#
#     needed_quarters = []
#     times_needed_quarters = []
#     for q in range(len(times)):
#         tms_qrt = []
#         for ii, tt in enumerate(summed_transit_times):
#             if (tt >= times[q][0]) & (tt <= times[q][-1]):
#                 tms_qrt.append(tt)
#         if len(tms_qrt) > 0:
#             needed_quarters.append(q)
#             times_needed_quarters.append(tms_qrt)
#
#     target_pixels_stats = []
#     target_pixels_means = []
#     target_pixels_ppms = []
#     target_pixels_stds_of_ppms = []
#     for iq, q in enumerate(needed_quarters):
#         target_pixels_stats_pxl = []
#         target_pixels_means_pxl = []
#         target_pixels_ppms_pxl = []
#         target_pixels_stds_of_ppms_pxl = []
#         for tpf in [target_pixels_flt[q][:, i] for i in range(len(target_pixels_flt[q][0])) if aperture_masks_flt[q][i]]:
#             lcp = spip.KeplerLightCurve(times=[times[q]], fluxes=[tpf], get_data=False)
#             lcp.condition_data()  # IN OUTLIERS MASK, THE POSITIVE AND THE NEGATIVE NSIG ARE EQUAL BECAUSE WE NEED TO INVERT FAIRLY.
#             lcp.condition_templates(template_index=best_temp)
#             lcp.detect_single_transits(apply_veto=False, gaussianize=False)
#             for tt in times_needed_quarters[iq]:
#                 ind_tr = np.nanargmin(np.abs(times[q] - tt))
#                 # std_stat = ut.outlier_free_std(lcp.statistics_snr[0][0], nsig=3, mask=lcp.masks[0], detrend=False, detr_len=128)
#                 ti0 = max(0, ind_tr - bins_mismatch_allowed)
#                 # time_index = ti0 + np.nanargmax(np.abs(lcp.statistics_snr[0][0][ti0:ind_tr + bins_mismatch_allowed]))
#                 # -> allow the actual minimum to be a little shifted
#                 stat_snr_here = np.nanmin(
#                     [np.nanmin(lcp.statistics_snr[0][0][ti0:ind_tr + bins_mismatch_allowed]) for _ in range(len(lcp.statistics_snr))])
#                 stat_raw_here = np.nanmin(
#                     [np.nanmin(lcp.statistics_raw[0][0][ti0:ind_tr + bins_mismatch_allowed])
#                      for _ in range(len(lcp.statistics_raw))])
#                 ppm_here = (stat_raw_here / lcp.stat_denoms[0][0] / lcp.trends_orig_fluxes[0][ind_tr] * np.max(
#                     np.abs(lcp.tb.templates_shifts[0])) * 1e6)
#                 std_of_ppm = (1 / lcp.stat_denoms[0][0] ** 0.5 * np.std(lcp.trends_orig_fluxes[0])
#                               / lcp.trends_orig_fluxes[0][ind_tr] ** 2 * np.max(np.abs(lcp.tb.templates_shifts[0])) * 1e6)
#                 target_pixels_stats_pxl.append(stat_snr_here)
#                 target_pixels_means_pxl.append(lcp.orig_means[0])
#                 target_pixels_ppms_pxl.append(ppm_here)
#                 target_pixels_stds_of_ppms_pxl.append(std_of_ppm)
#         target_pixels_stats.append(target_pixels_stats_pxl)
#         target_pixels_means.append(target_pixels_means_pxl)
#         target_pixels_ppms.append(target_pixels_ppms_pxl)
#         target_pixels_stds_of_ppms.append(target_pixels_stds_of_ppms_pxl)
#     return target_pixels_stats, target_pixels_means, target_pixels_ppms, target_pixels_stds_of_ppms
