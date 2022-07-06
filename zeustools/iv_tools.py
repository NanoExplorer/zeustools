import os
from zeustools import mce_data
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
from zeustools import plotting as zt_plotting
import zeustools as zt
try:
    am = zt.ArrayMapper()  # YES I JUST MADE A GLOBAL VARIABLE. DO I REGRET IT? NO. WILL IT HURT? ABSOLUTELY.
except:
    print("FAILED TO CREATE ARRAY MAP")
# Whenever you import plotting.py you'd better have a 'config' directory in your working directory or else.
# HAHA THAT INLCUDES BUILDING THE DOCS 
# TODO: fix this. I think the arraymapper should have a default config somehow
# Still needs fixing, but at least it's less annoying!


def super_remover(data):
    """ Attempt to remove unlocked data from IV curves by finding the first jump of larger than 1e7 

    :param data: masked array containing IV curve datacube.
    """
    # this advanced function can do whole mce data files
    # though it assumes that "data" is a masked array already

    # first find all the huge jumps
    diffs = data[:, :, 1:]-data[:, :, :-1]
    good = np.abs(diffs) < 1e7
    # now find the largest bias with a huge jump (remember data is collected at large bias first)
    last_false_index = np.argmin(good, axis=2)
    # Now we have the index of the first huge jump for every pixel.

    # bleh, let's just for loop it
    # set all entries after the determined index to "masked"
    # if this becomes a speed bottleneck then let's use that magic 
    # jit thing, numba
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i, j, last_false_index[i, j]:] = np.ma.masked
    
    return data


def find_transition(bias, data):
    """ Attempt to find the superconducting transition by fitting the normal branch and looking for deviations.
    """
    # First, do a really rough fit to the first 10 data points, which should be guaranteed to be normal
    result = stats.linregress(bias[0:10], data[0:10])
    # Find the residuals
    subdata = data - (bias * result.slope + result.intercept)

    # Find the maximum residual in the expected direction for the transition, relative to the normal slope.
    # (deviation will always be in opposite direction of slope! If we see things getting steeper, that's bad)
    if result.slope < 0:
        transition_guess_idx = np.argmin(subdata)
    else:
        transition_guess_idx = np.argmax(subdata)
    trans_guess = bias[transition_guess_idx]
    return trans_guess


def find_transition_index(bias, data):
    guess = find_transition(bias, data)
    orig_guess = guess
    # I don't really like this, but it's the best I've got right now.
    # Technically it's actually extremely robust, it just seems hacky
    # The index of the transition is obviously not part of the normal branch,
    # So we move to increased bias to ensure we make it to the normal branch
    if guess < 1:
        # assume real units!
        guess += 4e-8
    else:
        # assume dac units!
        guess += 1000
    # print(f"guess={guess:.2e}")
    # Find the closest index to the guess bias value
    idx = np.argmin(np.abs(bias-guess))
    t_idx = np.argmin(np.abs(bias-orig_guess))
    return idx, t_idx


def linear_normal_fit(bias, data):
    end_idx, trans_idx = find_transition_index(bias, data)

    # print(end_idx)
    if end_idx == 0:
        end_idx = 10
    regression = stats.linregress(bias[0:end_idx], data[0:end_idx])
    return regression, trans_idx


def fixed_slope_interceptor(bias, data, slope):
    end_idx,_ = find_transition_index(bias, data)
    valid_bias = bias[0:end_idx]
    valid_data = data[0:end_idx]
    
    de_sloped = valid_data - valid_bias*slope
#     plot(valid_bias,valid_data)
#     plot(valid_bias,valid_bias*slope)
    return(np.average(de_sloped))


class IVHelper:
    """ Load in an IV curve or several IV curves and perform useful operations with them"""
    def __init__(self):
        pass
        self.filenames = []
        self.temperatures = []
        self.temperatures_int = []
        self.mce_data = []  
        self.data = []  # list of data files
        self.is_real_units = False
        self.bias = []  # List of bias arrays
        self.cache = {}

    def __add__(self,other):
        result = IVHelper()
        assert self.is_real_units == other.is_real_units
        result.filenames = self.filenames + other.filenames
        result.temperatures = self.temperatures + other.temperatures
        result.temperatures_int = self.temperatures_int + other.temperatures_int
        result.mce_data = self.mce_data + other.mce_data
        result.data = self.data + other.data
        result.is_real_units = self.is_real_units
        result.bias = self.bias + other.bias
        # Could figure out how to add cache too :P
        return result

    def load_file(self, file, temp=None):
        self.filenames.append(file)
        if temp is None:
            if "mK" in file:
                i = file.find("mK")
            elif "mk" in file:
                i = file.find("mk")
            self.temperatures.append(file[i-3:i+2])
            self.temperatures_int.append(int(file[i-3:i]))
        else:
            assert type(temp) is int
            self.temperatures.append(temp)
            self.temperatures_int.append(temp)

        mcefile = mce_data.SmallMCEFile(file)
        self.mce_data.append(mcefile)
        mce_data_read = mcefile.Read(row_col=True).data
        mce_data_masked = np.ma.array(mce_data_read)
        # let's go ahead and clean up the data now
        # we can always unmask it later if we need to

        mce_data_cleaned = super_remover(mce_data_masked)
        # This removes a lot of junk, but doesn't always get it all

        self.data.append(mce_data_cleaned)
        bias = np.loadtxt(file+".bias", dtype=int, skiprows=1)
        tile_arg = list(mce_data_cleaned.shape)
        tile_arg[2] = 1  # this is probably going to be [33,24,1] which is
        # how we want to tile the bias array. Now the bias array will 
        # match the data array in shape, which doesn't change anything
        # until we switch to real units, where the bias voltage can be different
        # at the same time for different pixels. 
        bias = np.tile(bias, tile_arg)
        self.bias.append(bias)

    def load_directory(self, directory):
        """
        Given a directory containing IV curve data, this function will load the IV 
        data and make it ready for processing.
        For best results, organize the folder so it only contains IV
        files and the associated run, out, and bias files.
        Also, each filename should contain the bath temperature it was taken at
        in the format 110mK
        """
        # Clear cache, just in case this object is being reused to load a different set of IVs

        self.cache = {}
        # enumerate files in directory
        (_, _, all_files) = next(os.walk(directory))
        # filter to only MCE data files
        data_file_names = [x for x in all_files if x[-4:] != ".run" and x[-4:] != ".out" and x[-5:] != ".bias"]
        
        for file in data_file_names:
            self.load_file(directory + file)

    def switch_to_real_units(self):
        """ changes all internal data files from DAC units to real units
        As of now, this step is irreversible, so if you want to go back
        to DAC units you'll have to reload the directory. But really, it should be easy
        since I keep the mce data array.
        """

        # Do nothing if we're already in real units
        if self.is_real_units:
            return
        else:
            # Clear cache, b/c if any IVs have been calculated, they're in wrong units now
            self.cache = {}
            self.is_real_units = True
            for i in range(len(self.bias)):
                self.bias[i], self.data[i] = real_units(self.bias[i], 
                                                        self.data[i], 
                                                        whole_array=True)

    def get_temperature_colorbar_norm(self):
        norm = matplotlib.colors.Normalize(vmin=min(self.temperatures_int),
                                           vmax=max(self.temperatures_int))
        return norm

    def get_corrected_ivs(self, col, row, clean_again=True):
        """Returns all iv curves for col,row,
        currected to have normal y-intercept 0 and normal slope fixed """
        if (col, row) in self.cache:
            # print("CACHE HIT")
            return self.cache[(col, row)]
        else:
            # print("CALCULATING")
            all_slopes = []
            good_data = []
            for i in range(len(self.data)):
                # print(f"PROCESSING DATA {i}")
                one_px = self.data[i][row, col]

                # make sure there is at least some data here
                if not one_px.mask.all():

                    params, trans_idx = linear_normal_fit(self.bias[i][row, col], 
                                                          one_px)
                    if clean_again:
                        self.data[i][row, col, trans_idx:] = np.ma.masked
                        # Be aware, although this will mask a ton of squid unlocks
                        # it will also mask a few real superconducting branches
                        # however, I don't think that's too big a deal.
                        # also remember the bad data will be the last data points
                        # even though they're to the left in the plot
                    # note! if we're in real units, slope will be in mhos (ohms^-1).
                    # print(params.slope)
                    abslope = np.abs(params.slope)
                    # print(f"DATA {i} SLOPE IS {abslope:.3e}")
                    if not self.is_real_units and abslope > 4500 and abslope < 7000:
                        all_slopes.append(params.slope)
                        good_data.append(i)
                    elif self.is_real_units and abslope > 100 and abslope < 700:
                        all_slopes.append(params.slope)
                        good_data.append(i)
                    else:
                        # print(f"Rejecting IV curve with 'normal slope' {params.slope:.2e}")
                        pass
            avg_slope = np.ma.average(all_slopes)
            # print(f"average slope! {avg_slope:.2e}")
            new_data = []
            new_bias = []
            new_temp = []

            for i in good_data:
                one_px = self.data[i][row, col]
                new_intercept = fixed_slope_interceptor(self.bias[i][row, col], one_px, avg_slope)
                # print(new_intercept)
                if avg_slope < 0:
                    new_data.append(-(one_px - new_intercept))
                else:
                    new_data.append(one_px - new_intercept)
                new_bias.append(self.bias[i][row, col])
                new_temp.append(self.temperatures_int[i])
            self.cache[(col, row)] = (new_bias, 
                                      new_data, 
                                      new_temp, 
                                      avg_slope)
        return(new_bias, new_data, new_temp, avg_slope)


def real_units(bias, fb, col=0, whole_array=False,
                # ----THE FOLLOWING NUMBERS ARE COPIED FROM CARL'S PYTHON SCRIPT----  # noqa: E127
                # all units Ohms    
                mce_bias_R = 467,
                dewar_bias_R = 49,
                # These numbers need to be double checked. On the cold ping-thru sheet we have values like 130 ohms
                # and Carl's thesis reports 587 ohms for total bias resistance.

                cmb_shunts = [0, 3, 4],
                actpol_R = 180e-6,  # 180 uOhm, 830 nH [ref: Sherry Cho email]; probably same resistance as THz shunt 
                cmb_R = 140e-6,  # completely ballparked based off of known THz chip resistance giving TESs 4 mOhm normal R
                dewar_fb_R = 5280,  # = one MileOhm (~~joke~~)
                # Seriously though, in Carl's script this value was 5280 ohm
                # but on the cold ping through sheet, it is 1.28 kOhm or 1280 ohm.
                # We think there are 4 kOhm in the MCE itself

                # unitless
                butterworth_constant = 1218,  # When running in data mode 2 and the low pass 
                # filter is in the loop, all signals are multiplied by this factor

                # relative to the TES inductance
                rel_fb_inductance = 9,  # This means that for a change of 1 uA in the 
                # TES, the squid will have to change 9 uA to keep up

                # Volts. Note that these are bipolar voltages, eg the bias card can emit +- 5v
                max_bias_voltage = 5,
                max_fb_voltage = 0.958,  # seems a bit weird... still don't know where this number is from

                bias_dac_bits = 16,
                fb_dac_bits = 14
               ):
    """ Given an array of biases and corresponding array of feedbacks (all in DAC units)
    calculate the actual current and voltage going through the TES.
    
    :param bias: Bias array in DAC units
    :param fb: feedback array in DAC units
    :param col: Optional. the MCE column that you are calculating for.
        This lets us select the correct resistor values

    :param whole_array: Optional. If True, assumes that the value of the 
        "fb" param is a whole mce data array, 
        so we can handle resistors automatically.

    There are a lot of other parameters, and hopefully they're explanitory enough. They are
    mostly intrinsic properties of the system, but until we are absolutely certain of their
    values we need to be able to tweak them a little.

    :return: (TES voltage array, TES current array) in Volts and Amps respectively.
    
    Todo: Different chips may have different parameters. We currently handle this by assuming
    the array has a uniform normal resistance of 4 mOhm.

    """
    if col in cmb_shunts:
        shunt_R = cmb_R  # = 180 uOhm
        # We might be using "THz" interface chips on these columns
    else:
        shunt_R = actpol_R  # Anecdotal evidence suggests 
        # we use "ActPol" interface chips for most columns,
        # and mike niemack's thesis says they are 700 uOhm.

    if whole_array:
        shunt_R = np.repeat(actpol_R, fb.shape[1])
        for i in cmb_shunts:
            shunt_R[i] = cmb_R
    
    bias_raw_voltage = bias / 2**bias_dac_bits * max_bias_voltage * 2
    # last factor of 2 is because voltage is bipolar
    bias_current = bias_raw_voltage/(dewar_bias_R+mce_bias_R)
    fb_real_dac = fb / butterworth_constant
    fb_raw_voltage = fb_real_dac / 2**fb_dac_bits * max_fb_voltage * 2 
    # again, last factor of 2 is because voltage is bipolar
    fb_current = fb_raw_voltage / dewar_fb_R
    tes_current = fb_current / rel_fb_inductance
    
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


class InteractiveIVPlotter(zt_plotting.ZeusInteractivePlotter):
    def __init__(self, directory,
                 power_temp=130, file=False, file_temp_override=None):
        # If Plot_power = True, the 2-d array plot/colorbar
        # will show the saturation powers at the temperature = power_temp
        # Otherwise (plot_power=False) it will show normal resistance
        if type(directory) is IVHelper:
            self.ivhelper = directory
        elif not file:
            self.ivhelper = IVHelper()
            self.ivhelper.load_directory(directory)
        else:
            self.ivhelper = IVHelper()
            self.ivhelper.load_file(directory,temp=file_temp_override)
        self.ivhelper.switch_to_real_units()
        self.last_colorbar = None
        # maybe get slopes for every px and use them as bitmaps?
        # ideally use sat power?
        self.power_temp = power_temp
        self.build_data()

        super().__init__(self.rn_data,
                         None,
                         ts=1,  # ugly hack
                         # prevents parent class
                         # from attempting to auto
                         # generate ts 
                         flat=1  # basically same
                         # tricks the onclick method
                         # into letting us display the "flat"
                         # which in our case means plotting 
                         # IV curve instead of PR curve
                         # on right click
                         )
    
    def __sub__(self, other):
        # VERY SPECIFIC. BE CAREFUL
        result = InteractiveIVPlotter(self.ivhelper, power_temp=self.power_temp)
        if self.power_temp != other.power_temp:
            print("warning, subtracting powers from different temperatures")
        result.ivhelper = self.ivhelper + other.ivhelper
        result.power_data = self.power_data - other.power_data
        # result.power_data[result.power_data < 0] = np.ma.masked
        return result

    def build_data(self):
        shape = self.ivhelper.data[0].shape
        slopes = np.ones((shape[0], shape[1]))
        powers = np.ma.ones((shape[0], shape[1]), fill_value=np.nan)
        for i in range(shape[0]):
            for j in range(shape[1]):
                bias, data, temp, slope = self.ivhelper.get_corrected_ivs(j, i)
                slopes[i, j] = slope
                try:
                    t_idx = temp.index(self.power_temp)
                    power = bias[t_idx] * data[t_idx]
                    powers[i, j] = np.ma.min(power)
                except ValueError:
                    # print(f"power_temp={self.power_temp} was not found for px col,row={j},{i}")
                    powers[i, j] = np.ma.masked

        data = 1/slopes
        data = np.ma.array(np.abs(data))
        data[data > 0.007] = np.ma.masked

        powers[powers < 0] = np.ma.masked

        self.power_data = powers
        self.rn_data = data        

    def interactive_plot_power(self, array='all'):
        array = zt.array_name(array)
        data = np.ma.copy(self.power_data)
        if array == "a":
            data[:, 12:] = np.ma.masked
        elif array == "b":
            data[:, :12] = np.ma.masked
        self.data = data
        self.interactive_plot()
        self.cb.set_label("Power (W)")

    def interactive_plot_rn(self):
        self.data = self.rn_data
        self.interactive_plot()
        self.cb.set_label("TES Normal Resistance (ohm)")

    def update_colorbar(self, sm):
        if self.last_colorbar:
            self.last_colorbar.update_normal(sm)
        else:
            self.last_colorbar = plt.colorbar(sm, ax=self.ax2, label="Temperature [mK]")

    # override
    def bottom_plot(self):
        row, col = am.phys_to_mce(*self.click_loc)
        bias, data, temp, slope = self.ivhelper.get_corrected_ivs(col, row)
        cmap = plt.cm.plasma
        norm = self.ivhelper.get_temperature_colorbar_norm()
        for i in range(len(bias)):
            # print(data,bias)
            tes_voltage, tes_current = (bias[i], data[i])
            resistance = tes_voltage / tes_current
            power = tes_voltage * tes_current
            self.ax2.plot(power*1e12, resistance*1e3, ".", c=cmap(norm(temp[i])))

        self.ax2.set_xlabel("power (pW)")
        self.ax2.set_ylabel("resistance (mOhm)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        self.update_colorbar(sm)
        self.ax2.set_ylim(0, 6)
        self.ax2.set_xlim(0, 20)

    # override
    def bottom_flat(self):
        self.ax2.clear()  # haha, we probably just made a plot here. NEVERMIND! gone.
        row, col = am.phys_to_mce(*self.click_loc)
        cmap = plt.cm.plasma
        norm = self.ivhelper.get_temperature_colorbar_norm()
        bias, data, temp, slope = self.ivhelper.get_corrected_ivs(col, row)
        for i in range(len(data)):
            self.ax2.plot(bias[i], data[i], c=cmap(norm(temp[i])))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        self.update_colorbar(sm)
        self.ax2.set_xlim(0, 3e-7)
        self.ax2.set_ylim(0, 7e-5)
        self.ax2.set_xlabel("bias voltage (V)")
        self.ax2.set_ylabel("feedback current (A)")

    def detectors_hist(self,title,bins=30,arrays=[350,450],plot_rn=False):
        plt.figure()
        ax=plt.gca()
        if plot_rn:
            dat_pw = self.rn_data.filled()*1000
            ax.set_xlabel("resistance (m$\\Omega$)")
        else:
            dat_pw = self.power_data.filled()*1e12
            ax.set_xlabel("power (pW)")

        if 350 in arrays:
            n,bins,p=plt.hist(dat_pw[:,:5].flatten(),bins=bins,label="350 $\mu$m",color="C2")
        if 450 in arrays:
            n,bins,p=plt.hist(dat_pw[:,5:12].flatten(),bins=bins,alpha=0.8,label="450 $\mu$m",color="C1")
        if 200 in arrays:
            n,bins,p=plt.hist(dat_pw[:,12:19].flatten(),bins=bins,alpha=0.8,label="200 $\mu$m",color="C0")
        if 600 in arrays:
            n,bins,p=plt.hist(dat_pw[:,19:21].flatten(),bins=bins,alpha=0.8,label="600 $\mu$m",color="C3")

        plt.title(title)

        ax.set_ylabel("# detectors")
        ax.legend()
        return bins

if __name__ == "__main__":
    iv_plotter = InteractiveIVPlotter("data/")


