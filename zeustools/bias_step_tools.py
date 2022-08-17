import numpy as np
from numpy import ma
from zeustools import dac_converters
from zeustools import plotting

def get_bias_array(mce):
    hdr = mce.runfile.data["HEADER"]
    bias_string = hdr["RB tes bias"]
    return np.fromstring(bias_string, dtype=int, sep=" ")


def naive_data_reduction(chop, cube):
    """Seriously stupid data reduction.
    Works only on the most well-behaved signals
    """
    
    return ma.median(-cube[:, :, chop == 1], axis=2) + ma.median(cube[:, :, chop == 0], axis=2)
    

def bias_step_chop(mce):
    """ Given an MCE SmallDataFile for data taken in bias step mode, compute the correct on/off chop signal matching
    bias low / bias high.

    :param mce: mce data file object
    :returns: array containing square wave chop signal
    """
    hdr = mce.runfile.data["HEADER"]
    data_rate = int(hdr["RB cc data_rate"])
    step_rate = int(hdr["RB cc ramp_step_period"])
    data_points_per_phase = step_rate//data_rate
    npts = mce.Read(row_col=True).data.shape[2]
    i = np.arange(npts)
    chop = (i // data_points_per_phase) % 2
    return chop


def bias_step_resistance(mce):
    """ Given an MCE SmallMCEFile object, compute the reistance of each pixel at the current bias point.
    
    :param mce: mce data file object
    :returns: MCE shaped array contining resistance values

    """
    hdr = mce.runfile.data["HEADER"]
    if mce.data_mode == 1:
        bw = 1
    else:
        bw = 1218

    step_size_dac = int(hdr["RB cc ramp_step_size"])
    # bias = get_bias_array(mce) # may want this later
    data = mce.Read(row_col = True).data
    chop = bias_step_chop(mce)
    delta_current_dac = naive_data_reduction(chop,data)
    dIbias = dac_converters.bias_dac_to_current(step_size_dac)
    dI = dac_converters.fb_dac_to_tes_current(delta_current_dac,
                                              butterworth_constant=bw)
    approx_tes_dV = dIbias * dac_converters.get_shunt_array()
    approx_tes_R = approx_tes_dV / dI 
    return approx_tes_R

def bs_interactive_plotter_factory(mce):
    data = bias_step_resistance(mce)
    cube = mce.Read(row_col=True).data
    chop = bias_step_chop(mce)

    return(plotting.ZeusInteractivePlotter(data,cube,chop=chop))