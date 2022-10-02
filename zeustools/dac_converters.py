import numpy as np
from zeustools import data
import importlib.resources as res

# ----THE FOLLOWING NUMBERS ARE COPIED FROM CARL'S PYTHON SCRIPT---- 
# most units ohms
MCE_BIAS_R = 467
# dewar_bias_R = 49, Old value, not sure where it comes from?
DEWAR_BIAS_R = 132 # Checked from cold ping-through on Oct 2 2022
# These numbers need to be double checked. On the cold ping-thru sheet we have values like 130 ohms
# and Carl's thesis reports 587 ohms for total bias resistance (MCE+Dewar).

CMB_SHUNTS = [0, 3, 4]
ACTPOL_R = 180e-6  # 180 uOhm, 830 nH [ref: Sherry Cho email]; probably same resistance as THz shunt 
CMB_R = 140e-6  # completely ballparked based off of known THz chip resistance giving TESs 4 mOhm normal R

DEWAR_FB_R = 5350  # Assuming 4 kOhm in the MCE, checked during cold ping-through on Oct 2 2022
# resistances of SQ1FB lines was average 1.35 kOhm

# DEWAR_FB_R = 5280  # = one MileOhm (~~joke~~)
# Seriously though, in Carl's script this value was 5280 ohm
# but on the cold ping through sheet, it is 1.28 kOhm or 1280 ohm.
# We think there are 4 kOhm in the MCE itself

# unitless
BUTTERWORTH_CONSTANT = 1218  # When running in data mode 2 and the low pass 
# filter is in the loop, all signals are multiplied by this factor

# relative to the TES inductance
REL_FB_INDUCTANCE = 9  # This means that for a change of 1 uA in the 
# TES, the squid will have to change 9 uA to keep up

# Volts. Note that these are bipolar voltages, eg the bias card can emit +- 5v
MAX_BIAS_VOLTAGE = 5
MAX_FB_VOLTAGE = 0.958  # seems a bit weird... still don't know where this number is from. Seems correct though

BIAS_DAC_BITS = 16
FB_DAC_BITS = 14


def real_units(bias, fb, col=0, whole_array=False,
                # noqa: E127
                mce_bias_R = MCE_BIAS_R,
                dewar_bias_R = DEWAR_BIAS_R,
                cmb_shunts = CMB_SHUNTS,
                actpol_R = ACTPOL_R,  
                cmb_R = CMB_R,
                dewar_fb_R = DEWAR_FB_R, 
                butterworth_constant = BUTTERWORTH_CONSTANT,  
                rel_fb_inductance = REL_FB_INDUCTANCE, 
                max_bias_voltage = MAX_BIAS_VOLTAGE,
                max_fb_voltage = MAX_FB_VOLTAGE,  
                bias_dac_bits = BIAS_DAC_BITS,
                fb_dac_bits = FB_DAC_BITS
               ):
    """ Given an array of biases and corresponding array of feedbacks (all in DAC units)
    calculate the actual current and voltage going through the TES. Note that to ensure consistency
    the Feedback DAC numbers should be shifted so that their zero values make sense.

    :param bias: Bias array in DAC units
    :param fb: feedback array in DAC units
    :param col: Optional. the MCE column that you are calculating for.
        This lets us select the correct resistor values

    :param whole_array: Optional. If True, assumes that the value of the 
        "fb" param is a whole mce data array, 
        so we can handle resistors automatically.
    :param cmb_shunts: List of columns assumed to have the "CMB6" style shunt chip
        though this is probably wrong, it makes things consistent. All other columns are
        assumed to have "actpol" style shunt chips.
    :param dewar_fb_R: Although this is listed as the dewar feedback resistance, it is really the MCE + dewar in one.
        At present we use MCE_R=4000, Dewar_R=1280. Unit:ohms.

    There are a lot of other parameters, and hopefully they're explanitory enough. They are
    mostly intrinsic properties of the system, but until we are absolutely certain of their
    values we need to be able to tweak them a little. Some quirks carried over from Carl's script:


    :return: (TES voltage array, TES current array) in Volts and Amps respectively.
    
    Todo: Different chips may have different parameters. We currently handle this by assuming
    the array has a uniform normal resistance of 4 mOhm.

    """
    if not whole_array:
        if col in cmb_shunts:
            shunt_R = cmb_R  # = 180 uOhm
            # We might be using "THz" interface chips on these columns
        else:
            shunt_R = actpol_R  # Anecdotal evidence suggests 
            # we use "ActPol" interface chips for most columns,
            # and mike niemack's thesis says they are 700 uOhm. however that's wrong. Mike niemacks thesis
            # was on ACT not actpol.

    else:
        shunt_R = get_shunt_array(actpol_R, cmb_R, cmb_shunts)
    
    bias_current = bias_dac_to_current(bias, bias_dac_bits, max_bias_voltage, dewar_bias_R, mce_bias_R)
    tes_current = fb_dac_to_tes_current(fb, butterworth_constant, fb_dac_bits, max_fb_voltage, dewar_fb_R, rel_fb_inductance)
    
    shunt_current = bias_current - tes_current
    
    if whole_array:
        tes_voltage = shunt_current * shunt_R[None, :, None]  # geez
        # This is what you have to do if you want to multiply
        # by an array with axis=1...
        # np.multiply claims to have an axis argument,
        # but I couldn't get it to work
    else:
        tes_voltage = shunt_current * shunt_R
    
    return(tes_voltage, tes_current)


def get_shunt_array(actpol_R = ACTPOL_R,
                    cmb_R = CMB_R,
                    cmb_shunts = CMB_SHUNTS):
    shunt_R = np.repeat(actpol_R, 24)
    for i in cmb_shunts:
        shunt_R[i] = cmb_R
    return shunt_R


def bias_dac_to_current(bias, 
                        bias_dac_bits = BIAS_DAC_BITS, 
                        max_bias_voltage = MAX_BIAS_VOLTAGE, 
                        dewar_bias_R = DEWAR_BIAS_R, 
                        mce_bias_R = MCE_BIAS_R):
    """ Given bias dac values, return the bias current. This works in absolute because the 
    bias DAC is absolute. However it does also work in relative, i.e. for converting bias step
    size. """
    bias_raw_voltage = bias / 2**bias_dac_bits * max_bias_voltage * 2
    # last factor of 2 is because voltage is bipolar
    bias_current = bias_raw_voltage/(dewar_bias_R+mce_bias_R)
    return bias_current


def fb_dac_to_tes_current(fb, 
                          butterworth_constant = BUTTERWORTH_CONSTANT, 
                          fb_dac_bits = FB_DAC_BITS, 
                          max_fb_voltage = MAX_FB_VOLTAGE, 
                          dewar_fb_R = DEWAR_FB_R, 
                          rel_fb_inductance = REL_FB_INDUCTANCE):
    """ Given feedback DAC values, return current values. This works both in absolute terms (i.e., if 
    you are passing in shifted feedback values) and in relative terms because we're only mulitplying."""
    fb_real_dac = fb / butterworth_constant
    fb_raw_voltage = fb_real_dac / 2**fb_dac_bits * max_fb_voltage * 2 
    # again, last factor of 2 is because voltage is bipolar
    fb_current = fb_raw_voltage / dewar_fb_R
    tes_current = fb_current / rel_fb_inductance
    with res.open_text(data, "column_sign.dat") as file:
        table = np.loadtxt(file)
    tes_current = tes_current * table[:, 1]
    return tes_current

def correct_signs(cube):
    with res.open_text(data,"column_sign.dat") as signfile:
        signtable = np.loadtxt(signfile)
    return cube * signtable[:,1][None,:,None]

def mcefile_get_butterworth_constant(mce):
    if mce.data_mode == 1:
        bw = 1
    else:
        bw = 1218
    return bw