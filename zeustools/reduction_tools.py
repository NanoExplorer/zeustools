from datetime import datetime, timezone
import numpy as np
from zeustools import mce_data
from zeustools import leapseconds
from zhklib import database
from glob import glob
import os
from matplotlib import pyplot as plt
import zeustools as zt


def smart_file_finder_(folder, source_name):
    print("START")
    all_files = glob(os.path.join(folder, "*"))
    useful_files = [f for f in all_files if os.path.splitext(f)[1] == '' and os.path.isfile(f)]
    sorted_files = sorted(
        useful_files,
        key=os.path.getmtime
    )
    reduction_chunks = []
    current_bias_steps = []
    current_sky_chops = []
    current_data_files = []
    valid = False
    last_bias = np.array([0])
    last_grating = 0
    bias = None
    grating = None
    for f in sorted_files + ['THE_END']:
        try:
            if f != 'THE_END':
                mcef = mce_data.SmallMCEFile(f)
                bias = zt.get_bias_array(mcef)
                try:
                    grating = int(zt.hk_tools.get_value(f"{f}.hk", 'gratingindex'))
                except FileNotFoundError:
                    grating = last_grating
                #print(grating)
                if grating == 0:
                    grating = last_grating
            if np.any(bias != last_bias) or f == "THE_END" or grating != last_grating:
                #end this chunk
                # if grating != last_grating:
                #     print("grating changed")
                # else:
                #     print("bias changed")
                if valid:
                    #save this chunk
                    chunk = (
                        current_bias_steps,
                        current_sky_chops,
                        current_data_files
                    )
                    reduction_chunks.append(chunk)
                    #print("adding valid chunk")
                current_bias_steps = []
                current_sky_chops = []
                current_data_files = []
                valid = False
                last_bias = bias
                last_grating = grating
                #print("Bias changed, reset")
            
            if 'bias_step' in f and 'magic' not in f:
                current_bias_steps.append(f)
                #print("Bias step!",f)
            elif source_name in f:
                current_data_files.append(f)
                #print("data!",f)
            elif "skychop" in f:
                current_sky_chops.append(f)
                #print("skychop!",f)
            valid = len(current_bias_steps) != 0 and \
                len(current_data_files) != 0 and \
                len(current_sky_chops) != 0

        except IndexError:
            #print(f"skipping invalid file [indexerror] {f}")
            pass
        except mce_data.BadRunfile:
            #print(f"skipping invalid file [badrunfile] {f}")
            pass
        except FileNotFoundError:
            #print(f"skipping invalid file [notfound] {f}")
            pass
        except ValueError:
            print(f"Bad run file {f}")
    return reduction_chunks
            

def smart_to_bo_spec(smart):
    bias_arr = []
    flat_arr = []
    data_arr = []
    folders = []
    for b, s, d in smart:
        bias_tuple = consecutive_files(b)
        flat_tuple = consecutive_files(s)
        data_tuple = consecutive_files(d)
        folders.append(os.path.dirname(b[0]))
        bias_arr.append({"bias_step": [bias_tuple]})
        flat_name = os.path.basename(s[0])[0:-5]
        flat_arr.append({flat_name: [flat_tuple]})
        src_name = os.path.basename(d[0])[0:-5]
        data_arr.append({src_name: [data_tuple]})
    return bias_arr, flat_arr, data_arr, folders
   

def consecutive_files(arr):
    first_n = None
    last_n = None
    for f in arr:
        n = int(f.split("_")[-1])
        if first_n is None:
            first_n = n
            last_n = n
        elif n == last_n+1:
            last_n = n
        else:
            print(f"{f} is not consecutive...")
    return (first_n, last_n)


def smart_file_finder(folder, source_name, run_hk_check=True):
    """ WARNING! Not actually that smart. Please double check all results.
    """
    sp = smart_file_finder_(folder, source_name)
    if run_hk_check:
        for _, _, datafile_list in sp:
            for datafile in datafile_list:
                f = os.path.join(folder, datafile)+'.hk'
                if int(zt.hk_tools.get_value(f, 'gratingindex')) == 0:
                    raise ValueError(f"{f} has invalid grating index")

    return smart_to_bo_spec(sp)


def smarter_file_finder(file_dict):
    biases = []
    flats = []
    datas = []
    folders = []
    for path in file_dict:
        for file_name in file_dict[path]:
            print(path, file_name)
            bias, flat, data, folder = smart_file_finder(path, file_name)
            biases = biases+bias
            flats = flats+flat
            datas = datas+data
            folders = folders+folder
    return (biases, flats, datas, folders)


def nod_loader(filename, number_range):
    """ Filename should include a {num} format string so that we can build
    numbered filenames. Number_range should be a range or iterable of integers
    Returns a list of tuples (mce_data object, time series array, mce_data array, wobbler phase, nod sign)
    """
    i0 = number_range[0]
    data_out = []
    for i in number_range:
        i_str = f"{i:04d}"
        full_fname = filename.format(num=i_str)
        nod_sign = 1 if (i-i0) % 2 == 0 else -1
        mce = mce_data.SmallMCEFile(full_fname)
        ts = np.genfromtxt(full_fname+'.ts', invalid_raise=False)[:, 1]
        mread = mce.Read(row_col=True)
        chop = mread.chop
        data = mread.data
        tslen = ts.shape[0]
        mcelen = data.shape[2]
        endlen = min(tslen, mcelen)
        ts = ts[0:endlen]
        data = data[:, :, 0:endlen]
        data_out.append((mce, ts, data, chop, nod_sign))
    return data_out


def nod_concat(nod_arr):
    """ Given an array of the format generated by nod_loader
    concatenate all the time series to generate one huge time stream
    returns a tuple (ts, mce_data) 
    This is mostly useful just for inspecting the atmospheric variations
    over long time lengths or doing lombscargles.
    """
    ts_list = [nod[1] for nod in nod_arr]
    data_list = [nod[2] for nod in nod_arr]
    ts = np.concatenate(ts_list)
    data = np.concatenate(data_list, axis=2)
    return (ts, data)
    

def hk_matcher(ts):
    """ Given an array of timestamps from the .ts files,
    this function returns an array of timestamps (in GPS time to 
    match the .ts files) and the corresponding array of detector
    temperatures"""
    start = ts[0]
    end = ts[-1]
    # Every programmer hates time zones. OK. 
    # the TS array is in GPS time, which is ~19 seconds off
    # from UTC. 
    
    # utcfromtimestamp generates a datetime with the assumption that the time stamp
    # given is in UTC, but the date time returned is timezone-naive.
    # The datetime returned violates the usual assumptions about naive datetimes. 
    # The reason for this is that naive datetimes are usually assumed to be in 
    # your PC's local time. So the normal fromtimestamp function generates a "standard"
    # naive datetime. If you used replace(tzinfo=local_timezone) on it then everything would be
    # hunky dory, but I don't know how to get the "local" timezone accurately.
    start_gps = datetime.utcfromtimestamp(start-5)
    end_gps = datetime.utcfromtimestamp(end+5)
    print(start_gps)
    tz = timezone.utc
    # The leapseconds module works fine with time-zone naive datetimes. It just 
    # subtracts like 19 seconds (or adds. There's a reason I found a dedicated module).
    # However, the time-zone-naive datetime has to be one of the weird UTC naive datetimes
    # that "violate standard assumptions about datetimes"
    
    # After that, we force the naive timezones into aware timezones in UTC. Because
    # that's what they should be. sigh. Luckily from here on everything is much less 
    # confusing. I *sincerely* *hope* that we **never** ***ever*** observe at midnight UTC
    # on january 1 or july 1.
    start_utc = leapseconds.gps_to_utc(start_gps).replace(tzinfo=tz)
    end_utc = leapseconds.gps_to_utc(end_gps).replace(tzinfo=tz)
    # because EasyThermometry needs aware datetimes.
    
    # The HKPC can be queried in any time zone, and returns time-zone-aware UTC 
    # datetimes.
    thermo = database.EasyThermometry(0, start_date=start_utc, end_date=end_utc)
    print(start_utc)
    print(end_utc)
    #print(thermo.sensors)
    temps_list = thermo.sensors['GRT'][6]
    times_list = []
    for time, temp in temps_list:
        # I modified the leapseconds module so it works on aware datetimes. 
        # the variable "time" here is "aware" and is in UTC
        gps_t = leapseconds.utc_to_gps(time)
        # Luckily, aware datetimes generate correct timestamps.
        times_list.append(gps_t.timestamp())
    array_temp = np.array(thermo.sensors['GRT'][6])
    return np.array(times_list), array_temp[:, 1].astype(float)
    

def flux_calibration(data,
                     flat_flux_density,  # W/m^2/bin
                     ):
    spec_pos, sig, noise = data
    
    scaled_signal = sig * flat_flux_density 
    
    scaled_err = noise * flat_flux_density
    return (spec_pos, scaled_signal, scaled_err)


def wavelength_calibration(data,
                           position_of_line,
                           bin_width  # km/s
                           ):
    spec_pos, _, _ = data
    velocity = (spec_pos - position_of_line) * bin_width
    return (velocity, data[1], data[2])


def cut(data, min_px, max_px):
    out = []
    for i in data:
        out.append(i[min_px:max_px])
    return out


def shift_and_add(data1, data2, px1, px2):
    """
    takes in two spectra. Shifts the second one spectrally by px_offset
    to align the line pixel between the two runs. Weights the spectra appropriately
    TODO: use np.average to clean up this mess.
    """
    arr = data1 + data2
    m = map(np.copy, arr)
    spec, sig, noise, spec2, sig2, noise2 = m
    #print(sig[3],noise[3])
    nan_idxs = np.isnan(sig)
    nan_idx2 = np.isnan(sig2)

    sig[nan_idxs] = 0
    sig2[nan_idx2] = 0
    noise[nan_idxs] = 10e10
    noise2[nan_idx2] = 10e10
    noise[noise < 1e-6] = 10e10  # sometimes superconducting pixels show up unflagged
    noise2[noise2 < 1e-6] = 10e10
    spec -= px1
    spec2 -= px2 
    # That shifts the spectral pixel number so that the line is on position "0" 

    allspecs = np.append(spec, spec2)
    minspec = np.min(allspecs)
    maxspec = np.max(allspecs)
    outspec = np.arange(minspec, maxspec+1, dtype=int)
    outsig = np.zeros_like(outspec, dtype=float)
    outnoise = np.zeros_like(outspec, dtype=float)
    idx = (np.isin(outspec, spec)).nonzero()[0]
    idx2 = np.isin(outspec, spec2).nonzero()[0]
    outsig[idx] += sig/noise**2
    outsig[idx2] += sig2/noise2**2
    outnoise[idx] += 1/noise**2
    outnoise[idx2] += 1/noise2**2
    outnoise = 1/outnoise  # This sets outnoise to 1/wt
    outsig = outsig*outnoise  # divide by weight 
    outnoise = np.sqrt(outnoise)  # now outnoise is actually standard deviation
    outsig[outnoise > 1e9] = np.nan
    return (outspec, outsig, outnoise)


def calulate_bin_overlap(left_1, right_1, left_2, right_2):
    """ Calculate length of overlap of bins

    Args:
        left_1, right_1 : left and right bin edge values for bin 1
        left_2, right_2 : left and right bin edge values for bin 2

    Return:
        fraction of bin 2 covered by bin 1
    Courtesy of Patrick McNamee
    """
    
    return max(
        min(right_1, right_2) - max(left_1, left_2),
        0.
    )/(right_2-left_2)


def regridding_weighted_average(new_grid, orig_wl_data, orig_flux_data, error_data_cut=None):
    """
    Performs a weighted average on unevenly sampled data with respect to a new grid. 
    :param new_grid: two-dimensional numpy array of desired output bin edges
    :param orig_wl_data: list of wavelength arrays for each observation.
    :param orig_flux_data: list of observation array tuples for each observation

    Thanks to Patrick McNamee
    """
    # Build data array. One element per data point.
    array_for_avg = []
    for wl_tuple, data_tuple in zip(orig_wl_data, orig_flux_data):
        px_array, flux_array, error_array = data_tuple
        wl_l_array, wl_r_array = wl_tuple
        for wl_l, wl_r, flux, err in zip(wl_l_array, wl_r_array, flux_array, error_array):
            data_entry = np.array([wl_l, wl_r, flux, err])

            if np.all(np.isfinite(data_entry)) and err > 1e-10:
                if error_data_cut is None or err < error_data_cut:
                    array_for_avg.append(data_entry)

    array_for_avg = np.array(array_for_avg)

    weights = np.zeros((new_grid.shape[0], array_for_avg.shape[0]))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = calulate_bin_overlap(
                array_for_avg[j, 0],
                array_for_avg[j, 1],
                new_grid[i, 0],
                new_grid[i, 1]
            )/array_for_avg[j, -1]**2  # w_i = l_i/\sigma^2
    wt_sum = np.sum(weights, axis=1).reshape(weights.shape[:1] + (1,))
    norm_wts = weights/wt_sum
    regridded_weighted_avg = norm_wts @ array_for_avg[:, 2]
    regridded_err = 1/np.sqrt(wt_sum)
    return (regridded_weighted_avg, regridded_err)


# def shift_add(data,line_px):
# 	"""
# 	takes in a bunch of spectra, then aligns them and does a weighted average
# 	params: list of data spectra
# 	line_px: list of the spectral positions with lines 
#   In-progress improvement to shift_and_add
# 	"""

# 	# create an array to hold the result
# 	specs = np.array([spec - i for (spec,_,_),i in zip(data,line_pix)])
# 	final_spec = np.arange(np.min(specs.flatten()),np.max(specs.flatten()))


def get_drop_indices(spec_pos, px_to_drop):
    line_px = np.array(px_to_drop)[:, None]
    boolarray = np.all(spec_pos != line_px, axis=0)
    return boolarray.nonzero()[0]


def contsub(data, line_px):
    spec_pos, sig, err = data 
    idxs = get_drop_indices(spec_pos, line_px)
    # print(sig)
    # print(idxs)
    continuum, cont_err = np.ma.average(sig[idxs], weights=1/err[idxs]**2, returned = True)
    # print(idxs)
    return (spec_pos, sig-continuum, err, continuum, 1/np.sqrt(cont_err))


def getcsvspec(label, spec):
    stringout = label+', '
    for i in spec:
        stringout += str(i)+", "
    return stringout


def plot_spec(spec, saveas, bounds, do_close = True):
    plt.figure(figsize=(8, 6))
    #spec[1][spec[2]>1e-16] = np.nan
    line = plt.step(spec[0], spec[1], where='mid')
    lncolor = line[0].get_c()
    plt.errorbar(spec[0], spec[1], spec[2], fmt='none', ecolor=lncolor)
    plt.ylabel("Flux, W m$^{-2}$ bin$^{-1}$")
    plt.xlabel("Velocity km s$^{-1}$")
    if bounds is not None and None not in bounds:
        plt.xlim(bounds[0:2])
        plt.ylim(bounds[2:4])
    plt.tight_layout()
    plt.savefig(saveas, dpi=300)
    if do_close:
        plt.close()