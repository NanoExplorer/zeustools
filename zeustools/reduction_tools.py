from datetime import datetime, timezone
import numpy as np
from zeustools import mce_data
from zeustools import leapseconds
from zhklib import database


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
    