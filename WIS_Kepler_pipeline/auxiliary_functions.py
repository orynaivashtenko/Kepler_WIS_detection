#!/usr/bin/env python3
import copy
import os
os.environ["OMP_NUM_THREADS"] = "1"  #
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  #
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import warnings
from scipy.optimize import minimize
import gzip
import shutil
import requests
import wget
from astropy.io import fits

import utils
import utils as ut
from search_pipeline_parameters import *
import pipeline_parameters as pp
import locations as locs

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
    expected_times = ut.get_transit_times(first_transit_time=first_time, period=period, max_time=max_time)
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


def get_correct_tid_folder(target_id, location, allow_creation=False, with_prefix_folder=True):
    """
    Looks around the given folder and finds or creates the correct directory,
    so that the structure is General_all_stars_folder/target_id[:4]/target_id[:4]
    :param target_id: KIC number
    :type target_id: str, will be adjusted to len 9
    :param location:
    :type location:
    :param allow_creation: Whether the function is allowed to create the needed folder if it does not exist
    :type allow_creation: bool
    :param with_prefix_folder:
    :type with_prefix_folder:
    :return:
    :rtype:
    """
    target_id = str(target_id).rjust(9, '0')
    if with_prefix_folder:
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
    else:
        if os.path.split(location)[-1] == target_id:
            pass
        else:
            if target_id not in os.listdir(location):
                if allow_creation:
                    os.mkdir(os.path.join(location, target_id))
                else:
                    raise FileNotFoundError('No such file or directory: ' + os.path.join(location, target_id))
            location = os.path.join(location, target_id)
        return location


def get_saving_file_path(output_location, target_id, filename_postfix, file_format, insert_after_postfix=None):
    """

    :param output_location: The directory in which the output of this target is saved
    :type output_location: str
    :param target_id:
    :type target_id: str
    :param filename_postfix:
    :type filename_postfix: str
    :param file_format:
    :type file_format: str
    :param insert_after_postfix:
    :type insert_after_postfix: str or None
    :return:
    :rtype:
    """
    filename_postfix = filename_postfix.strip('_')
    if file_format[0] != '.':
        file_format = '.' + file_format
    if filename_postfix[-len(file_format):] != file_format:
        filename_postfix = filename_postfix + file_format

    if insert_after_postfix is not None:
        filename_postfix = filename_postfix[:-len(file_format)] + insert_after_postfix + filename_postfix[-len(file_format):]

    if target_id is not None:
        file_dir = os.path.join(output_location, str(target_id) + '_' + filename_postfix)
    else:
        file_dir = os.path.join(output_location, filename_postfix)
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
                dt=0.0204336, means=None, template=None, return_std=False,
                normalize_by_counts=True):
    if not hasattr(times[0], "__len__"):
        times = [times]
    if not hasattr(fluxes[0], "__len__"):
        fluxes = [fluxes]

    if masks is None:
        masks = [~np.isnan(fl) for fl in fluxes]
    elif not hasattr(masks[0], "__len__"):
        masks = [masks]

    time_fold = period
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

    bins_std = None
    if return_std:
        bins_flux_normalized = np.array(bins_flux)
        # For the entries with enough counts for the unbiased estimate:
        inds1 = (bins_counts >= 2)
        bins_flux_normalized[inds1] = bins_flux[inds1] / bins_counts[inds1]
        # estimate of the variance of the measurements:
        bins_flux_2[inds1] = (bins_flux_2[inds1] - bins_flux_normalized[inds1] ** 2 * bins_counts[inds1]) / (bins_counts[inds1] - 1)
        if normalize_by_counts:
            bins_flux_2[inds1] = bins_flux_2[inds1] / bins_counts[inds1]  # the variance of the mean is like that
        else:
            bins_flux_2[inds1] = bins_flux_2[inds1] * bins_counts[inds1]  # the variance of the sum is like that
        # See notebook from 22/08/24 for the explanation of the formula above

        # For the ones where there is not enough counts for unbiased estimate of variance:
        inds1 = (bins_counts == 1)
        if len(np.where(inds1)[0]) > 0:
            bins_flux_normalized[inds1] = bins_flux[inds1] / bins_counts[inds1]
            # estimate of the variance of the measurements:
            bins_flux_2[inds1] = (bins_flux_2[inds1] - bins_flux_normalized[inds1] ** 2 * bins_counts[inds1]) / bins_counts[inds1]
            if normalize_by_counts:
                bins_flux_2[inds1] = bins_flux_2[inds1] / bins_counts[inds1]  # the variance of the mean is like that
            else:
                bins_flux_2[inds1] = bins_flux_2[inds1] * bins_counts[inds1]  # the variance of the sum is like that

        bins_std = bins_flux_2**0.5
        if normalize_by_counts:
            bins_flux = bins_flux_normalized

    elif normalize_by_counts:
        bins_flux_normalized = np.array(bins_flux)
        inds1 = (bins_counts >= 2)
        bins_flux_normalized[inds1] = bins_flux[inds1] / bins_counts[inds1]
        inds1 = (bins_counts == 1)
        if len(np.where(inds1)[0]) > 0:
            bins_flux_normalized[inds1] = bins_flux[inds1] / bins_counts[inds1]
        bins_flux = bins_flux_normalized

    if template is None:
        if return_std:
            return bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, bins_std
        else:
            return bins_time, bins_flux, bins_counts, folds_times, folds_fluxes
    else:
        # Warning: the template produced here is not multiplied by counts, so it is not the real expectation
        # value of the flux if the flux is not count-normalized!
        template_cropped = np.zeros(n_timesteps)
        half_len = min(len(template) // 2, n_timesteps // 2)
        template_cropped[n_timesteps // 2 - half_len: n_timesteps // 2 + half_len] = template[len(template) // 2 - half_len: len(
            template) // 2 + half_len]
        tem_center = np.nanargmax(np.abs(template_cropped))
        time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(template_cropped) - tem_center))) * dt
        if return_std:
            return bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped, bins_std
        else:
            return bins_time, bins_flux, bins_counts, folds_times, folds_fluxes, time_template, template_cropped


def get_average_whitened_template(whitened_templates_td, temp_number, n_time_templ, dt_days=None, indices_of_templates=None):
    if indices_of_templates is None:
        from template_bank.template_module import Indices_of_templates
        indices_of_templates = Indices_of_templates
    if dt_days is None:
        from template_bank.template_module import dt_lc as dt_days
    n_quarters = len(whitened_templates_td[0])

    iot = indices_of_templates[temp_number]
    if iot[1] - iot[0] > 1:
        temps = [np.nanmean([whitened_templates_td[i][q] for i in range(iot[0], iot[1])], axis=0)
                 for q in range(n_quarters)]
    else:
        temps = whitened_templates_td[indices_of_templates[temp_number][0]]
        temps = [np.array(t) for t in temps]
    for q in range(len(temps)):
        if len(temps[q]) < n_time_templ:
            temps[q] = ut.pad_by_zeros(temps[q], n_time_templ, in_the_center=True)
        elif len(temps[q]) > n_time_templ:
            temps[q] = np.concatenate((temps[q][:n_time_templ // 2], temps[q][- (n_time_templ + 1) // 2:]))
    temp_best_1 = np.array(np.mean([temps[q][:n_time_templ // 2] for q in range(n_quarters)], axis=0))
    temp_best_2 = np.array(np.mean([temps[q][n_time_templ // 2:] for q in range(n_quarters)], axis=0))
    time_template_1 = np.arange(len(temp_best_1)) * dt_days
    time_template_2 = (np.arange(len(temp_best_2)) * dt_days * (-1))[::-1]
    time_template_2 = time_template_2 - (time_template_2[1] - time_template_2[0])
    temp_best = np.concatenate((temp_best_2, temp_best_1))
    time_template = np.concatenate((time_template_2, time_template_1))
    # tem_center = np.nanargmax(np.abs(temp_best))
    # time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * self.klc.dt_days
    return temp_best, time_template


def get_average_stat_template(whitened_templates_td, temp_number, n_time_templ,
                              dt_days=None, indices_of_templates=None):
    temp_best, time_template = get_average_whitened_template(whitened_templates_td=whitened_templates_td, temp_number=temp_number,
                                                             n_time_templ=n_time_templ, dt_days=dt_days,
                                                             indices_of_templates=indices_of_templates)
    temp_best = temp_best / np.linalg.norm(temp_best)
    temp_best = np.roll(temp_best, -np.argmax(temp_best))  # the maximum is supposed to be at zero
    temp_best = np.fft.irfft(np.abs(np.fft.rfft(temp_best)) ** 2)  # for statistic, template is convolution
    tem_center = n_time_templ // 2
    temp_best = np.roll(temp_best, tem_center)  # the maximum is supposed to be at zero before
    time_template = np.concatenate((np.arange(tem_center) - tem_center, np.arange(len(temp_best) - tem_center))) * dt_days
    return temp_best, time_template


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


def get_best_int_score_and_stat_index(int_scores, stats,
                                      delta_stat_close=pp.Delta_stat_close, delta_integral_close=pp.Delta_integral_close):
    int_argmax = np.nanargmax(int_scores)
    int_max = int_scores[int_argmax]
    stat_at_int_max = stats[int_argmax]
    best_arg = int_argmax

    close_inds_check = []
    if int_argmax > 0:
        close_inds_check.append(int_argmax - 1)
    if int_argmax < len(int_scores) - 1:
        close_inds_check.append(int_argmax + 1)

    for ind_new in close_inds_check:
        if (not np.isnan(int_scores[ind_new])) and (not np.isnan(stats[ind_new])):
            delta_integral = int_scores[ind_new] - int_max  # < 0
            delta_stat = stats[ind_new] - stat_at_int_max  # may be > 0
            if delta_stat > - delta_integral * (delta_stat_close / delta_integral_close):
                best_arg = ind_new
                int_max = int_scores[best_arg]
                stat_at_int_max = stats[best_arg]
    return best_arg, int_max, stat_at_int_max


def get_periodicity_results(target_id, output_location, short_periodicity=True,
                            best_periodic_scores_filename_postfix=locs.Best_periodic_scores_filename_postfix,
                            best_short_periodic_scores_filename_postfix=locs.Best_short_periodic_scores_filename_postfix):
    filename_postfix = best_short_periodic_scores_filename_postfix if short_periodicity else best_periodic_scores_filename_postfix
    stats = []
    ints = []
    pers = []
    ftts = []
    temps = []
    gauss_stats = []
    file_path = get_saving_file_path(output_location=output_location, target_id=target_id,
                                     filename_postfix=filename_postfix, file_format='npy')
    best_periodic_scores = np.load(file_path, allow_pickle=True)
    for i_chunk in range(len(best_periodic_scores)):
        stats.append(best_periodic_scores[i_chunk][Best_per_results_index_stats][0])
        ints.append(best_periodic_scores[i_chunk][Best_per_results_index_best_integral][0])
        pers.append(best_periodic_scores[i_chunk][Best_per_results_index_per][0])
        ftts.append(best_periodic_scores[i_chunk][Best_per_results_index_ftt][0])
        temps.append(best_periodic_scores[i_chunk][Best_per_results_index_template][0])
        gauss_stats.append(best_periodic_scores[i_chunk][Best_per_results_index_stats_gauss][0])

    return_array = np.zeros((6, len(ints)))
    return_array[Best_per_results_index_stats] = stats
    return_array[Best_per_results_index_best_integral] = ints
    return_array[Best_per_results_index_per] = pers
    return_array[Best_per_results_index_ftt] = ftts
    return_array[Best_per_results_index_template] = temps
    return_array[Best_per_results_index_stats_gauss] = gauss_stats
    return return_array


def get_connected_stats_from_file(file_path=None, output_location=None, target_id=None,
                                  filename_postfix=Connected_statistic_data_filename_postfix,
                                  return_only_stats=False):
    """
    Load all the connected stats and other fields which will be used in periodicity search or timeslides.
    :param file_path: The exact local file which to load. Overrides `filename_postfix` and `path`
    :type file_path: str or None
    :param filename_postfix: Used in the name convention "target_id + filename_postfix" in
        case `file` is not specified. The location will be `self.output_location`
    :type filename_postfix: str
    :param output_location:
    :type output_location: str or None
    :param target_id:
    :type target_id: str or None
    :param return_only_stats:
    :type return_only_stats: bool
    """
    if file_path is None:
        file_path = get_saving_file_path(output_location=output_location, target_id=target_id,
                                         filename_postfix=filename_postfix, file_format='npy')
    assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"
    connected_data = np.load(file_path, allow_pickle=True)

    num_temp_shifts = (len(connected_data) - 3) // 5  # int(connected_data[0, 0])
    time_full = connected_data[0, 1:]
    big_hole_masks_full = (connected_data[2, 1:]).astype(bool)
    templates_of_indices = []
    line_num_0 = 3

    statistics_full = np.zeros((num_temp_shifts, len(time_full)), dtype=Dtype_r)
    stat_vars_full = np.zeros((num_temp_shifts, len(time_full)), dtype=Dtype_r)
    stat_masks_full = np.zeros((num_temp_shifts, len(time_full)), dtype=bool)
    corrections_full = np.zeros((num_temp_shifts, len(time_full)), dtype=Dtype_r)
    statistics_gaussianized_full = np.zeros((num_temp_shifts, len(time_full)), dtype=Dtype_r)

    for kk in range(num_temp_shifts):
        templates_of_indices.append(connected_data[line_num_0 + kk, 0])
        statistics_full[kk] = connected_data[line_num_0 + kk, 1:]
        stat_masks_full[kk] = (connected_data[line_num_0 + 1 * num_temp_shifts + kk, 1:]).astype(bool)
        stat_vars_full[kk] = connected_data[line_num_0 + 2 * num_temp_shifts + kk, 1:]
        corrections_full[kk] = connected_data[line_num_0 + 3 * num_temp_shifts + kk, 1:]
        statistics_gaussianized_full[kk] = connected_data[line_num_0 + 4 * num_temp_shifts + kk, 1:]

    # making indices_of_templates
    temp_now = templates_of_indices[0]
    indices_of_templates = [[0]]
    for i, temp in enumerate(templates_of_indices):
        if temp != temp_now:
            indices_of_templates[-1].append(i)
            indices_of_templates.append([i])
            temp_now = temp
    indices_of_templates[-1].append(len(templates_of_indices))
    indices_of_templates = np.array(indices_of_templates)
    templates_of_indices = np.array(templates_of_indices)

    if return_only_stats:
        return time_full, statistics_full, stat_masks_full, stat_vars_full, big_hole_masks_full, indices_of_templates
    else:
        return (time_full, statistics_full, stat_masks_full, stat_vars_full, big_hole_masks_full, corrections_full,
                statistics_gaussianized_full, indices_of_templates, templates_of_indices)

# ####################################################################################################
# FUNCTIONS FOR VETO FOLDED
# ####################################################################################################

def mf_high_pass(data, dt, dt_pass, template=None, template_rolled=None, renorm_psd_drift=False):
    """
    ! Warning ! uses the np.fft.rfft(..., norm=None) convention!
    Matched-filtering of `data` and `template` with high-pass filter. That is, the matched filter is informed
    that there is no information below `1/dt_pass` frequency. It creates an effective whitening filter which zeros
    everything below the frequency `1/dt_pass`. This filter is also used in the denominator of the matched-filter,
    so that the score is properly normalized.
    :param data: data array to be matched-filtered
    :type data: numpy.ndarray 1d or 2d
    :param template: template or templates in the time domain. Superseded by `template_rolled`.
        Used if `template_rolled` is `None`.
    :type template: None or numpy.ndarray 1d or 2d
    :param template_rolled: template for all its possible rolls. Is needed when convolution cannot be made because
        the template shape is not shift-invariant. If None, then `template` is used, and convolution is made.
    :type template_rolled: None or numpy.ndarray 2d
    :param dt: time step of the data, should have the same units as `dt_pass`
    :type dt: float
    :param dt_pass: maximal timescale of the high-pass filter. Power at frequencies smaller than `1/dt_pass`
        will be zeroed. Should have the same units as `dt_pass`.
    :type dt_pass: float
    :param renorm_psd_drift:
    :type renorm_psd_drift: bool
    :return: matched-filtering scores in units of snr and in units of amplitude
    :rtype: (numpy.ndarray 1d or 2d, numpy.ndarray 1d or 2d), corresponding to the sizes of `data` and `template`
    """
    data = np.array(data)
    template = np.array(template)
    freqs = np.fft.rfftfreq(data.shape[-1], d=dt)
    inds_pass = freqs > 1 / dt_pass
    wf2_effective = np.zeros(freqs.shape[-1])
    f_data = np.fft.rfft(data, axis=-1)
    wf2_effective[inds_pass] = data.shape[-1] / np.mean((np.abs(f_data) ** 2)[inds_pass])
    if template_rolled is not None:
        blued_data = np.fft.irfft(f_data * wf2_effective, n=len(data), axis=-1)
        numer = np.sum(blued_data * template_rolled, axis=-1)
        blued_template_rolled = np.fft.irfft(np.fft.rfft(template_rolled, axis=-1) * wf2_effective, n=len(data), axis=-1)
        denom = np.sum(blued_template_rolled * template_rolled, axis=-1) ** 0.5
    elif template is not None:
        f_template = np.fft.rfft(template, axis=-1)
        numer = np.fft.irfft(f_data * wf2_effective * np.conjugate(f_template), n=len(data), axis=-1)
        denom = (np.fft.irfft(f_template * wf2_effective * np.conjugate(f_template), n=len(data), axis=-1)[..., 0]) ** 0.5
    else:
        raise ValueError('Either `template` or `template_rolled` should be provided')

    stats_snr = (numer.T / denom).T
    stats_ampl = (numer.T / denom ** 2).T
    if renorm_psd_drift and hasattr(stats_snr, '__len__') and len(stats_snr) > 1:
        stats_snr = stats_snr / ut.outlier_free_std(stats_snr, nsig=3, n_iterations=1, renormalize=True)
    return stats_snr, stats_ampl  # snr, ampl


def mf_high_pass_iterative_rolls(data, dt, dt_pass, template=None, coefficients_template=None, renorm_psd_drift=False):
    """
    ! Warning ! uses the np.fft.rfft(..., norm=None) convention!
    Matched-filtering of `data` and `template` with high-pass filter. That is, the matched filter is informed
    that there is no information below `1/dt_pass` frequency. It creates an effective whitening filter which zeros
    everything below the frequency `1/dt_pass`. This filter is also used in the denominator of the matched-filter,
    so that the score is properly normalized.
    :param data: data array to be matched-filtered
    :type data: numpy.ndarray 1d or 2d
    :param template: template or templates in the time domain.
    :type template: None or numpy.ndarray 1d or 2d
    :param dt: time step of the data, should have the same units as `dt_pass`
    :type dt: float
    :param dt_pass: maximal timescale of the high-pass filter. Power at frequencies smaller than `1/dt_pass`
        will be zeroed. Should have the same units as `dt_pass`.
    :type dt_pass: float
    :param renorm_psd_drift:
    :type renorm_psd_drift: bool
    :return: matched-filtering scores in units of snr and in units of amplitude
    :rtype: (numpy.ndarray 1d or 2d, numpy.ndarray 1d or 2d), corresponding to the sizes of `data` and `template`
    """
    data = np.array(data)
    template = np.array(template)
    freqs = np.fft.rfftfreq(data.shape[-1], d=dt)
    inds_pass = freqs > 1 / dt_pass
    wf2_effective = np.zeros(freqs.shape[-1])
    f_data = np.fft.rfft(data, axis=-1)
    wf2_effective[inds_pass] = data.shape[-1] / np.mean((np.abs(f_data) ** 2)[inds_pass])
    if coefficients_template is None:  # bins_counts ** 0.5
        coefficients_template = np.ones(len(template))
    blued_data = np.fft.irfft(f_data * wf2_effective, n=len(data), axis=-1)

    stats_snr = np.zeros(len(template))
    stats_ampl = np.zeros(len(template))
    for rolling_ind in range(len(template)):
        template_rolled = coefficients_template * np.roll(template, rolling_ind)
        numer = np.sum(blued_data * template_rolled, axis=-1)
        blued_template_rolled = np.fft.irfft(np.fft.rfft(template_rolled, axis=-1) * wf2_effective, n=len(data), axis=-1)
        denom = np.sum(blued_template_rolled * template_rolled, axis=-1) ** 0.5
        stats_snr[rolling_ind] = (numer.T / denom).T
        stats_ampl[rolling_ind] = (numer.T / denom ** 2).T

    if renorm_psd_drift and hasattr(stats_snr, '__len__') and len(stats_snr) > 1:
        stats_snr = stats_snr / ut.outlier_free_std(stats_snr, nsig=3, n_iterations=1, renormalize=True)
    return stats_snr, stats_ampl  # snr, ampl


def renorm_noise_high_pass(data, dt, dt_pass):
    """
    ! Warning ! uses the np.fft.rfft(..., norm=None) convention!
    Renormalizes the noise`data` after high-pass filtering. We want that the noise is white,
    but its lowest frequencies are removes. This means that the high-frequency spectrum of the
    Fourier image of the data should look in the same way as if the data was white. Therefore,
    going to the time domain and measuring the variance is not an option because it will measure
    a smaller variance. This function renormalizes the data in the Fourier space so that the
    power at the high frequencies corresponds to the power of the white noise at the same
    frequencies, as defined by the length of the data in the np.fft.rfft(..., norm=None) convention
    :param data: Noise vector to be renormalized. In time-domain
    :type data: numpy.ndarray 1d
    :param dt: time step of the data, should have the same units as `dt_pass`
    :type dt: float
    :param dt_pass: Maximal timescale of the high-pass filter. Power at frequencies smaller than `1/dt_pass`
        will be zeroed. Should have the same units as `dt_pass`.
    :type dt_pass: float
    :return: Renormalized data vector. In time-domain
    :rtype: numpy.ndarray 1d
    """
    data = np.array(data)
    f_data = np.fft.rfft(data, axis=-1)
    freqs = np.fft.rfftfreq(data.shape[-1], d=dt)
    inds_pass = freqs > 1 / dt_pass
    wf2_effective = np.zeros(freqs.shape[-1])
    wf2_effective[inds_pass] = data.shape[-1] / np.mean((np.abs(f_data)**2)[inds_pass])
    return np.fft.irfft(f_data * wf2_effective**0.5, n=data.shape[-1], axis=-1)


def matched_filter_folded_fluxes_still_too_much_memory(template_number, whitened_templates_td, n_time_templ, binwid,
                                                       indices_of_templates, bins_time, bins_counts, flux_normalized_vars,
                                                       dt_steps_high_pass, dt_days):
    """
    # ------------------------------------------------------------------------------------------------------------
    # CALCULATING THE MATCHED FILTER SCORE ON THE BINNED WHITENED FLUX
    # Matched-filtering of the folded binned flux with the templates corresponding to their expected values.
    # Takes the original `whitened_templates_td` that were used fot the search, in their original time resolution,
    # averages them over sub-bin shifts, and interpolates them (not integrates, which is in general wrong!)
    # to the binned time `bins_time`. The binned template is normalized by the `bins_counts`, as well as the
    # binned flux `bins_flux`, so that the expectations are consistent, and the variances are uniform.
    # Then, the binned flux `bins_flux` is matched-filtered with the binned and the normalized templates
    # The folded template is proportional to the counts, but the matched filtering should be done
    # on the whitened template, and the standard deviation is proportional to the square root of counts.
    # Important: template multiplied by counts is not shift-invariant because counts are different
    # in every bin. Therefore, convolution cannot be used.
    :param template_number:
    :type template_number:
    :param whitened_templates_td:
    :type whitened_templates_td:
    :param n_time_templ:
    :type n_time_templ:
    :param binwid:
    :type binwid:
    :param indices_of_templates:
    :type indices_of_templates:
    :param bins_time:
    :type bins_time:
    :param bins_counts:
    :type bins_counts:
    :param flux_normalized_vars:
    :type flux_normalized_vars:
    :param dt_steps_high_pass:
    :type dt_steps_high_pass:
    :param dt_days:
    :type dt_days:
    :return:
    :rtype:
    """
    folded_template_rolled = get_shift_averaged_rolled_template_normalized_by_bins(
        whitened_templates_td=whitened_templates_td, template_number=template_number, indices_of_templates=indices_of_templates,
        n_time_templ=n_time_templ, dt_days=dt_days, bins_time=bins_time, bins_counts=bins_counts)

    mf_scores, ampl_scores = mf_high_pass(flux_normalized_vars, template=None, template_rolled=folded_template_rolled,
                                          dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                          renorm_psd_drift=True)

    rolling_arg = np.argmax(mf_scores)
    # rolling_val_peak = rolling_arg if rolling_arg <= len(mf_scores) / 2 else len(mf_scores) - rolling_arg
    mf_score_peak = mf_scores[rolling_arg]
    ampl_score_peak = ampl_scores[rolling_arg]

    # max_dist_from_detected_peak_bins = 5  # how far away the pipeline peak can be from the folding peak for the best template
    # if rolling_vals_peaks[np.argmax(mf_scores_peaks)] > max_dist_from_detected_peak_bins:
    #     self.is_bad_fold_residuals = True  # There is some largest peak in the wrong place
    #     self.template_and_stat_of_fold_contaminant = (np.argmax(mf_scores_peaks), np.max(mf_scores_peaks))
    # self.best_temp_folded = np.argmax(mf_scores_peaks * (rolling_vals_peaks <= max_dist_from_detected_peak_bins).astype(float))
    # temp_number = self.best_temp_folded

    # mf_score, ampl_score = np.array(mf_scores[temp_number]), np.array(ampl_scores[temp_number])
    # del mf_scores  # for memory
    # del ampl_scores
    return mf_score_peak, ampl_score_peak, rolling_arg, folded_template_rolled[rolling_arg]


def matched_filter_folded_fluxes(template_number, whitened_templates_td, n_time_templ, binwid,
                                 indices_of_templates, bins_time, bins_counts, flux_normalized_vars,
                                 dt_steps_high_pass, dt_days, renorm_psd_drift=True):
    """
    # ------------------------------------------------------------------------------------------------------------
    # CALCULATING THE MATCHED FILTER SCORE ON THE BINNED WHITENED FLUX
    # Matched-filtering of the folded binned flux with the templates corresponding to their expected values.
    # Takes the original `whitened_templates_td` that were used fot the search, in their original time resolution,
    # averages them over sub-bin shifts, and interpolates them (not integrates, which is in general wrong!)
    # to the binned time `bins_time`. The binned template is normalized by the `bins_counts`, as well as the
    # binned flux `bins_flux`, so that the expectations are consistent, and the variances are uniform.
    # Then, the binned flux `bins_flux` is matched-filtered with the binned and the normalized templates
    # The folded template is proportional to the counts, but the matched filtering should be done
    # on the whitened template, and the standard deviation is proportional to the square root of counts.
    # Important: template multiplied by counts is not shift-invariant because counts are different
    # in every bin. Therefore, convolution cannot be used.
    # ------------------------------------------------------------------------------------------------------------
    The folded template is proportional to the counts, but the matched filtering should be done
    # on the whitened template, amd the standard deviation is proportional to the square root of counts.

    # The following is incorrect because you cannot divide template which will be slided by the bins which will
    # not be slided (in the matched-filtering over all the transit phases)
    # folded_template = np.zeros(len(bins_counts))
    # folded_template[bins_counts > 0] = temp_interp[bins_counts > 0] * bins_counts[bins_counts > 0] ** 0.5
    :param template_number:
    :type template_number:
    :param whitened_templates_td:
    :type whitened_templates_td:
    :param n_time_templ:
    :type n_time_templ:
    :param binwid:
    :type binwid:
    :param indices_of_templates:
    :type indices_of_templates:
    :param bins_time:
    :type bins_time:
    :param bins_counts:
    :type bins_counts:
    :param flux_normalized_vars:
    :type flux_normalized_vars:
    :param dt_steps_high_pass:
    :type dt_steps_high_pass:
    :param dt_days:
    :type dt_days:
    :return:
    :rtype:
    """

    averaged_template, time_template = get_average_whitened_template(whitened_templates_td=whitened_templates_td,
                                                                     temp_number=template_number, n_time_templ=n_time_templ,
                                                                     dt_days=dt_days, indices_of_templates=indices_of_templates)
    # Interpolating the best averaged template
    temp_interp = np.interp(bins_time, time_template, averaged_template)
    temp_interp = - temp_interp / np.linalg.norm(temp_interp)

    mf_scores, ampl_scores = mf_high_pass_iterative_rolls(data=flux_normalized_vars, dt=binwid, dt_pass=dt_steps_high_pass * dt_days,
                                                          template=temp_interp, coefficients_template=bins_counts**0.5,
                                                          renorm_psd_drift=renorm_psd_drift)
    rolling_arg = np.argmax(mf_scores)
    mf_score_peak = mf_scores[rolling_arg]
    ampl_score_peak = ampl_scores[rolling_arg]
    return mf_score_peak, ampl_score_peak, rolling_arg, bins_counts ** 0.5 * np.roll(temp_interp, rolling_arg)


def make_rolled_template_normalized_by_bins(averaged_template, bins_counts):
    """
    THIS TAKES TOO MUCH MEMORY
    :param averaged_template:
    :type averaged_template:
    :param bins_counts:
    :type bins_counts:
    :return:
    :rtype:
    """
    return bins_counts ** 0.5 * np.array([np.roll(averaged_template, ii) for ii in range(len(averaged_template))])


def get_shift_averaged_rolled_template_normalized_by_bins(whitened_templates_td, template_number, indices_of_templates,
                                                          n_time_templ, dt_days, bins_time, bins_counts):
    temp_best, time_template = get_average_whitened_template(whitened_templates_td=whitened_templates_td,
                                                             temp_number=template_number, n_time_templ=n_time_templ,
                                                             dt_days=dt_days, indices_of_templates=indices_of_templates)
    # Interpolating the best averaged template
    temp_interp = np.interp(bins_time, time_template, temp_best)
    temp_interp = - temp_interp / np.linalg.norm(temp_interp)
    folded_template_rolled = make_rolled_template_normalized_by_bins(temp_interp, bins_counts)
    # -> The folded template is proportional to the counts, but the matched filtering should be done
    # on the whitened template, amd the standard deviation is proportional to the square root of counts.

    # The following is incorrect because you cannot divide template which will be slided by the bins which will
    # not be slided (in the matched-filtering over all the transit phases)
    # folded_template = np.zeros(len(bins_counts))
    # folded_template[bins_counts > 0] = temp_interp[bins_counts > 0] * bins_counts[bins_counts > 0] ** 0.5
    return folded_template_rolled


def get_shift_reduced_full_stat_snr(statistics_full, stat_vars_full, temp_index, stat_masks_full=None,
                                    indices_of_templates=None, shift_reduction='minimize', renormalize=False,
                                    return_mask=False):
    """
    :param statistics_full:
    :type statistics_full:
    :param stat_vars_full:
    :type stat_vars_full:
    :param temp_index: Index of the desired template
    :type temp_index: int
    :param indices_of_templates:
    :type indices_of_templates:
    :param shift_reduction: 'select', 'minimize' or 'average'
    :type shift_reduction:
    :param renormalize: Whether to renormalize the final result by the outlier-free variance
    :type renormalize: bool
    :param stat_masks_full:
    :type stat_masks_full:
    :param return_mask:
    :type return_mask:
    :return:
    :rtype:
    """
    statistics_full = copy.deepcopy(statistics_full)
    if indices_of_templates is None:
        from template_bank.template_module import Indices_of_templates

    # Converting to units of SNR:
    statistics_full[stat_vars_full > 0] = statistics_full[stat_vars_full > 0] / (stat_vars_full[stat_vars_full > 0]) ** 0.5

    if stat_masks_full is None:
        stat_masks_full = [np.ones(len(statistics_full[k]), dtype=bool) for k in range(len(statistics_full))]
    stat_masks_full = [stat_masks_full[k] & (stat_vars_full[k] > 0) for k in range(len(statistics_full))]

    if indices_of_templates[temp_index][-1] - indices_of_templates[temp_index][0] > 1:
        if shift_reduction == 'minimize':  # -> minimizing over subbing shifts to find the worst
            statistic_full = np.nanmin(statistics_full[indices_of_templates[temp_index][0]: indices_of_templates[temp_index][-1]], axis=0)
        elif shift_reduction == 'average':  # -> averaging over subbing shifts to find the mean
            statistic_full = np.nanmean(statistics_full[indices_of_templates[temp_index][0]: indices_of_templates[temp_index][-1]], axis=0)
        else:  # -> selecting over subbing shifts to find something
            statistic_full = np.nanmean(statistics_full[indices_of_templates[temp_index][0][0]], axis=0)
        stat_mask_full = np.sum(stat_masks_full[indices_of_templates[temp_index][0]: indices_of_templates[temp_index][-1]],
                                axis=0).astype(bool)
    else:
        statistic_full = statistics_full[indices_of_templates[temp_index][0]]
        stat_mask_full = stat_masks_full[indices_of_templates[temp_index][0]]


    # The stat can be rather bad, we may renormalize by the outlier-free total variance:
    if renormalize:
        statistic_full = statistic_full / ut.outlier_free_std(statistic_full[stat_mask_full], n_iterations=1, nsig=3, renormalize=True)
    if not return_mask:
        return statistic_full
    else:
        return statistic_full, stat_mask_full


def get_near_transit_pieces_from_full_time(transit_times, time_full, data_full, time_wing_days,
                                           mask_full=None, peak_is_min=True, return_mask=False):
    if mask_full is None:
        mask_full = np.ones(len(time_full), dtype=bool)

    times_near_transit_pieces = []
    data_near_transit_pieces = []
    piece_peaks = []
    mask_near_transit_pieces = []
    for trans_tim in transit_times:
        ind_0 = np.nanargmin(np.abs(time_full - (trans_tim - time_wing_days)))
        ind_1 = np.nanargmin(np.abs(time_full - (trans_tim + time_wing_days)))
        data_near_transit_pieces.append(data_full[ind_0:ind_1])
        times_near_transit_pieces.append(time_full[ind_0:ind_1])
        mask_near_transit_pieces.append(mask_full[ind_0:ind_1])
        if peak_is_min:
            piece_peaks.append(np.nanmin(data_full[ind_0:ind_1][mask_full[ind_0:ind_1]]))  # ! Minimization over broad range!
        else:
            piece_peaks.append(np.nanmax(data_full[ind_0:ind_1][mask_full[ind_0:ind_1]]))  # ! Maximization over broad range!
    if not return_mask:
        return times_near_transit_pieces, data_near_transit_pieces, piece_peaks
    else:
        return times_near_transit_pieces, data_near_transit_pieces, mask_near_transit_pieces, piece_peaks


def get_near_transit_pieces_of_whitened_flux(transit_times, times, whitened_fluxes_td, time_wing_days,
                                             masks=None, unpad=False, zero_padding_ends=None, dt_days=None):
    if dt_days is None:
        from template_bank.template_module import dt_lc as dt_days
    times = copy.deepcopy(times)
    whitened_fluxes_td = copy.deepcopy(whitened_fluxes_td)
    if masks is not None:
        masks = copy.deepcopy(masks)
    if unpad:
        for q in range(len(whitened_fluxes_td)):
            if len(times[q]) == len(whitened_fluxes_td[q]):
                times[q] = times[q][zero_padding_ends[q][0]: len(times[q]) - zero_padding_ends[q][1]]
            if masks is not None:
                if len(masks[q]) == len(whitened_fluxes_td[q]):
                    masks[q] = masks[q][zero_padding_ends[q][0]: len(masks[q]) - zero_padding_ends[q][1]]
            whitened_fluxes_td[q] = whitened_fluxes_td[q][zero_padding_ends[q][0]: len(whitened_fluxes_td[q]) - zero_padding_ends[q][1]]
    time_full, whitened_flux_td_full = connect_quarters_fun(times, whitened_fluxes_td,
                                                            zero_padding_ends=None, is_data_cut=True,
                                                            is_time_cut=True, dt=dt_days, fill_gaps_by_nan=True)
    if masks is not None:
        mask_full = (connect_quarters_fun(times, masks, zero_padding_ends=None, is_data_cut=True,
                                          is_time_cut=True, dt=dt_days, fill_gaps_by_nan=False)[1]).astype(bool)
    else:
        mask_full = None
    times_near_transit_pieces, fluxes_near_transit_pieces, masks_near_transit_pieces, piece_peaks = (
        get_near_transit_pieces_from_full_time(
            transit_times=transit_times, time_full=time_full, data_full=whitened_flux_td_full, mask_full=mask_full,
            time_wing_days=time_wing_days, peak_is_min=True, return_mask=True))
    return times_near_transit_pieces, fluxes_near_transit_pieces, masks_near_transit_pieces,  piece_peaks


def get_orig_quarters(times):
    from tce_parameters import get_quarter_start_end_times
    quarter_start_times, quarter_end_times = get_quarter_start_end_times()
    orig_quarters = []
    for tim in times:
        for q_orig in range(len(quarter_start_times)):
            if quarter_start_times[q_orig] <= tim <= quarter_end_times[q_orig]:
                orig_quarters.append(q_orig)
            elif q_orig > 0 and quarter_end_times[q_orig-1] < tim < quarter_start_times[q_orig]:
                orig_quarters.append(np.nan)
    return np.array(orig_quarters)


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


# ##########################################################################################################################
# TARGET PIXEL FUNCTIONS
# ##########################################################################################################################

def get_target_pixel_data_fun(target_id, data_folder, remove_zeroth_quarter=False, remove_quarter_17=False, return_aperture_masks=False,
                              dtype=np.float64):
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
        time = tp_fits_file[1].data['TIME'].astype(dtype)
        target_pix = tp_fits_file[1].data['FLUX'].astype(dtype)
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
                                    dtype=np.float64):
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
            # ra, dec = tpf1.get_coordinates(cadence='all')  # see in the description of get_tpf_coordinates_corrected_origin
            ra, dec = get_tpf_coordinates_corrected_origin(tpf1, cadence='all')
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


def get_tpf_coordinates_corrected_origin(tpf, cadence="all"):
    """
    Copy of
    https://github.com/lightkurve/lightkurve/blob/b74b30795ddb52595e80939713558cdaf9252872/src/lightkurve/targetpixelfile.py#L438
    but with changed origin of the transformation of coordinates, see description in the comments.

    Returns two 3D arrays of RA and Dec values in decimal degrees.

    If cadence number is given, returns 2D arrays for that cadence. If
    cadence is 'all' returns one RA, Dec value for each pixel in every cadence.
    Uses the WCS solution and the POS_CORR data from TPF header.

    :param tpf: instance of lightkurve.targetpixelfile.KeplerTargetPixelFile
    from lightkurve import targetpixelfile
    tpf = targetpixelfile.KeplerTargetPixelFile(target_pixels_file_path)
    :type tpf: lightkurve.targetpixelfile.KeplerTargetPixelFile
    :param cadence: 'all' or int
    :type cadence: Which cadences to return the RA Dec coordinates for.
    :return:
    ra : numpy array, same shape as tpf.flux[cadence]
        Array containing RA values for every pixel, for every cadence.
    dec : numpy array, same shape as tpf.flux[cadence]
        Array containing Dec values for every pixel, for every cadence.
    :rtype:
    """
    w = tpf.wcs
    xx, yy = np.meshgrid(np.arange(tpf.shape[2]), np.arange(tpf.shape[1]))
    pos_corr1_pix = np.copy(tpf.hdu[1].data["POS_CORR1"])
    pos_corr2_pix = np.copy(tpf.hdu[1].data["POS_CORR2"])

    # We zero POS_CORR* when the values are NaN or make no sense (>50px)
    with warnings.catch_warnings():  # Comparing NaNs to numbers is OK here
        warnings.simplefilter("ignore", RuntimeWarning)
        bad = np.any(
            [
                ~np.isfinite(pos_corr1_pix),
                ~np.isfinite(pos_corr2_pix),
                np.abs(pos_corr1_pix - np.nanmedian(pos_corr1_pix)) > 50,
                np.abs(pos_corr2_pix - np.nanmedian(pos_corr2_pix)) > 50,
            ],
            axis=0,
        )
    pos_corr1_pix[bad], pos_corr2_pix[bad] = 0, 0

    # ----------------------------------------------------------------------------------------------------------
    # Add in POSCORRs
    # xx = np.atleast_3d(xx).transpose([2, 0, 1]) + np.atleast_3d(pos_corr1_pix).transpose([1, 2, 0])
    # yy = np.atleast_3d(yy).transpose([2, 0, 1]) + np.atleast_3d(pos_corr2_pix).transpose([1, 2, 0])
    # WARNING: the correction (as in the commented section above) was added in the original lightkurve implementation.
    # However, I noticed that this only makes the tred worse. On the contrary, subtracting the correction improves the trend,
    # therefore I decided to subtract it, as below:
    xx = np.atleast_3d(xx).transpose([2, 0, 1]) - np.atleast_3d(pos_corr1_pix).transpose([1, 2, 0])
    yy = np.atleast_3d(yy).transpose([2, 0, 1]) - np.atleast_3d(pos_corr2_pix).transpose([1, 2, 0])

    # ----------------------------------------------------------------------------------------------------------
    # Pass through WCS
    # WARNING: the following commented line:
    # # ra, dec = w.wcs_pix2world(xx.ravel(), yy.ravel(), 1)
    # was the original implementation of lightkurve package. The last argument,
    # 1, is the origin of the coordinates. It is set to 1, which is the default value for Fortran and FITS files. However, the
    # Numpy and C is 0.
    # The description from
    # https://github.com/astropy/astropy/blob/54578767c44950710562c1333bfe1a5312a083e3/astropy/wcs/docstrings.py#L48
    # origin : int
    #     Specifies the origin of pixel values. The Fortran and FITS
    #     standards use an origin of 1. Numpy and C use array indexing with
    #     origin at 0.
    #
    # I noticed that when using this transformation, the result id not consistent between this function
    # https://github.com/lightkurve/lightkurve/blob/b74b30795ddb52595e80939713558cdaf9252872/src/lightkurve/targetpixelfile.py#L438
    # and this transformation of the Gaia sources to the pixel coordinates
    # https://github.com/lightkurve/lightkurve/blob/main/src/lightkurve/interact.py#L549
    # I needed this for the interact_sky plot of the pixel data and Gaia sources:
    # https://github.com/lightkurve/lightkurve/blob/b74b30795ddb52595e80939713558cdaf9252872/src/lightkurve/targetpixelfile.py#L1378
    # I saw that the transformation is not invertible when using 1 and 0 in direct and inverse transform, so it cannot be
    # consistent, only one of them can possibly be correct.
    ra, dec = w.wcs_pix2world(xx.ravel(), yy.ravel(), 0)
    ra = ra.reshape((pos_corr1_pix.shape[0], tpf.shape[1], tpf.shape[2]))
    dec = dec.reshape((pos_corr2_pix.shape[0], tpf.shape[1], tpf.shape[2]))
    ra, dec = ra[tpf.quality_mask], dec[tpf.quality_mask]
    if cadence != "all":
        return ra[cadence], dec[cadence]
    return ra, dec


def get_tpf_fiducial_frame(tpf):
    """
    :param tpf: instance of lightkurve.targetpixelfile.KeplerTargetPixelFile
    from lightkurve import targetpixelfile
    tpf = targetpixelfile.KeplerTargetPixelFile(target_pixels_file_path)
    :type tpf: lightkurve.targetpixelfile.KeplerTargetPixelFile
    :return: fiducial_frame, index of time array when supposedly the TPF WCS coordinate system is exact
    :rtype: int
    """
    # Try to identify the "fiducial frame", for which the TPF WCS is exact
    zp = (tpf.pos_corr1 == 0) & (tpf.pos_corr2 == 0)
    (zp_loc,) = np.where(zp)

    if len(zp_loc) == 1:
        fiducial_frame = zp_loc[0]
    else:
        fiducial_frame = 0
    return fiducial_frame


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


def get_splitted_saps_from_pixels(target_pixel_fluxes, number_of_pixels_cut=None, part_of_peak=0.3,
                                  part_of_cumsum_2=0.8, part_of_light=None):
    """

    :param target_pixel_fluxes:
    :type target_pixel_fluxes:
    :param number_of_pixels_cut:
    :type number_of_pixels_cut:
    :param part_of_peak: Selects the brightest pixels, which are greater or equal then `part_of_peak` of the peak,
        are a connected component to the peak, and form a descending array around the peak. This helps to avoid
        including contaminant stars to the mask.
    :type part_of_peak: None or float
    :param part_of_cumsum_2:
    :type part_of_cumsum_2:
    :param part_of_light:
    :type part_of_light:
    :return:
    :rtype:
    """
    star_part_sap_fluxes = []
    bg_part_sap_fluxes = []
    for q in range(len(target_pixel_fluxes)):
        with warnings.catch_warnings():  # there can be full-nan slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_stamp = np.nanmean(target_pixel_fluxes[q], axis=0)
        if part_of_peak is not None:

            arg_max = np.nanargmax(mean_stamp)
            peak_coords = arg_max // mean_stamp.shape[-1], arg_max % mean_stamp.shape[-1]
            inclusion_mask = ut.get_indices_of_influence(vals=mean_stamp, ind_center_0=peak_coords[0], ind_center_1=peak_coords[1],
                                                         part_peak=part_of_peak, abs_diff_val=None,
                                                         abs_thresh=None, rel_thresh=None, forbid_diagonal=True, descending=True)
            with warnings.catch_warnings():  # there can be full-nan slices
                warnings.simplefilter("ignore", category=RuntimeWarning)
                star_part_sap_flux = np.nansum(target_pixel_fluxes[q][:, inclusion_mask], axis=-1)
                bg_part_sap_flux = np.nansum(target_pixel_fluxes[q][:, ~inclusion_mask], axis=-1)
        else:
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
            bg_part_sap_flux = np.nansum(tpf_rsh[:, bg_part_indices], axis=-1)

        nanind = np.where(star_part_sap_flux == 0)[0]
        if len(nanind) > 0:
            star_part_sap_flux[nanind] = star_part_sap_flux[nanind] * np.nan

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

    # ampl = (np.nanmean((ampls[not_missed_transit_mask] / flux_trend[not_missed_transit_mask]))
    #         * np.nanmean(flux_trend[not_missed_transit_mask]))
    ampl, ampl_std, ampl_variability = ut.get_mean_and_std_of_mean_two_sigmas_model(
        data=ampls[not_missed_transit_mask] / flux_trend[not_missed_transit_mask],
        stds=ampls_stds[not_missed_transit_mask] / flux_trend[not_missed_transit_mask], mask=None)
    ampl = ampl * np.nanmean(flux_trend[not_missed_transit_mask])
    ampl_std = ampl_std * np.nanmean(flux_trend[not_missed_transit_mask])
    ampl_variability = ampl_variability * np.nanmean(flux_trend[not_missed_transit_mask])

    ppms_stds[not_missed_transit_mask] = np.abs(ppms[not_missed_transit_mask]) * (
            (ampls_stds[not_missed_transit_mask] / ampls[not_missed_transit_mask])**2
            + (trends_std[not_missed_transit_mask] / flux_trend[not_missed_transit_mask])**2)**0.5
    # ppm = np.nanmean(ppms[not_missed_transit_mask])
    ppm, ppm_std, ppm_variability = ut.get_mean_and_std_of_mean_two_sigmas_model(data=ppms[not_missed_transit_mask],
                                                                                 stds=ppms_stds[not_missed_transit_mask], mask=None)
    # ppms_stds[not_missed_transit_mask] = (ppms[not_missed_transit_mask])**2 * (
    #     (ampls_stds[not_missed_transit_mask])**2 / (ampls[not_missed_transit_mask])**2
    #     + np.var(flux_trend[not_missed_transit_mask]) / (flux_trend[not_missed_transit_mask])**2)
    # ppms_stds[not_missed_transit_mask] = ((1 / summed_stat_vars[not_missed_transit_mask] / (flux_trend[not_missed_transit_mask]) ** 2
    #                                        + np.var(flux_trend[not_missed_transit_mask])
    #                                        / flux_trend[not_missed_transit_mask] ** 4 * templ_max ** 2 * 1e12) ** 0.5)
    return (ppm,  ppm_std,  ppm_variability, ppms, ppms_stds,
            ampl, ampl_std, ampl_variability, ampls, ampls_stds,
            normalized_ampls, normalized_ampls_stds)


def get_centroid_series(target_pixel_fluxes, sky_coords=None, aperture_masks=None, time_masks=None, dtype=np.float64):
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
    target_pixel_fluxes = [np.array(e, dtype=dtype) for e in target_pixel_fluxes]
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
            assert ((y_coor[aperture_mask] / np.mean(y_coor[aperture_mask])) != 1.
                    ).any(), 'All the pixels are in the same row! Centroid is constant'
            assert ((x_coor[aperture_mask] / np.mean(x_coor[aperture_mask])) != 1.
                    ).any(), 'All the pixels are in the same column! Centroid is constant'
            x_coor = x_coor * np.ones(np.shape(target_pixels_flux), dtype=dtype)
            y_coor = y_coor * np.ones(np.shape(target_pixels_flux), dtype=dtype)
        else:
            y_coor, x_coor = sky_coords[q][1], sky_coords[q][0]

        sap_flux = np.zeros(len(target_pixels_flux), dtype=dtype) * np.nan
        centr_x = np.zeros(len(target_pixels_flux), dtype=dtype) * np.nan
        centr_y = np.zeros(len(target_pixels_flux), dtype=dtype) * np.nan
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
