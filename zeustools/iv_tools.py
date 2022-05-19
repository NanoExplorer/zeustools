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
#Whenever you import plotting.py you'd better have a 'config' directory in your working directory or else.
#HAHA THAT INLCUDES BUILDING THE DOCS 
# TODO: fix this. I think the arraymapper should have a default config somehow
# Still needs fixing, but at least it kinda works in most cases now!


def super_remover(data):
    # this advanced function can do whole mce data files
    # though it assumes that "data" is a masked array already

    # first find all the huge jumps
    diffs = data[:,:,1:]-data[:,:,:-1]
    good = np.abs(diffs) < 1e7
    # now find the largest bias with a huge jump (remember data is collected at large bias first)
    last_false_index = np.argmin(good,axis=2)
    #Now we have the index of the first huge jump for every pixel.

    #bleh, let's just for loop it
    # set all entries after the determined index to "masked"
    # if this becomes a speed bottleneck then let's use that magic 
    # jit thing, numba
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i, j, last_false_index[i, j]:] = np.ma.masked
    
    return data


def find_transition(bias, data):
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
    guess = find_transition(bias,data)

    # I don't really like this, but it's the best I've got right now.
    # Technically it's actually extremely robust, it just seems hacky
    # The index of the transition is obviously not part of the normal branch,
    # So we move to increased bias to ensure we make it to the normal branch
    if guess < 1:
        #assume real units!
        guess += 4e-8
    else:
        #assume dac units!
        guess += 1000
    #print(f"guess={guess:.2e}")
    # Find the closest index to the guess bias value
    idx = np.argmin(np.abs(bias-guess))
    return idx

def linear_normal_fit(bias, data):
    end_idx = find_transition_index(bias, data)
    # print(end_idx)
    if end_idx == 0:
        end_idx = 10
    return(stats.linregress(bias[0:end_idx], data[0:end_idx]))


def fixed_slope_interceptor(bias, data, slope):
    end_idx = find_transition_index(bias, data)
    valid_bias = bias[0:end_idx]
    valid_data = data[0:end_idx]
    
    de_sloped = valid_data - valid_bias*slope
#     plot(valid_bias,valid_data)
#     plot(valid_bias,valid_bias*slope)
    return(np.average(de_sloped))


class IVHelper:
    def __init__(self):
        pass
        self.filenames = []
        self.temperatures = []
        self.temperatures_int = []
        self.mce_data = []  
        self.data = []
        self.is_real_units = False
        self.bias = []
        self.cache = {}

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
            self.filenames.append(file)
            i = file.find("mK")
            self.temperatures.append(file[i-3:i+2])
            self.temperatures_int.append(int(file[i-3:i]))

            full_path = directory+file
            mcefile = mce_data.SmallMCEFile(full_path)
            self.mce_data.append(mcefile)
            mce_data_read = mcefile.Read(row_col=True).data
            mce_data_masked = np.ma.array(mce_data_read)
            # let's go ahead and clean up the data now
            # we can always unmask it later if we need to

            mce_data_cleaned = super_remover(mce_data_masked)
            self.data.append(mce_data_cleaned)
            bias = np.loadtxt(full_path+".bias", dtype=int, skiprows=1)
            tile_arg = list(mce_data_cleaned.shape)
            tile_arg[2] = 1 # this is probably going to be [33,24,1] which is
            # how we want to tile the bias array. Now the bias array will 
            # match the data array in shape, which doesn't change anything
            # until we switch to real units, where the bias voltage can be different
            # at the same time for different pixels. 
            bias = np.tile(bias,tile_arg)
            self.bias.append(bias)

    def switch_to_real_units(self):
        """ changes all internal data files from DAC units to real units
        As of now, this step is irreversible, so if you want to go back
        to DAC units you'll have to reload the directory. But really, it should be easy
        since I keep the mce data array.
        """
        # Clear cache, b/c if any IVs have been calculated, they're in wrong units now
        self.cache = {}
        # Do nothing if we're already in real units
        if self.is_real_units:
            return
        else:
            self.is_real_units = True
            for i in range(len(self.bias)):
                self.bias[i],self.data[i] = real_units(self.bias[i],self.data[i])

    def get_temperature_colorbar_norm(self):
        norm = matplotlib.colors.Normalize(vmin=min(self.temperatures_int),
                                           vmax=max(self.temperatures_int))
        return norm

    def get_corrected_ivs(self,col,row):
        """Returns all iv curves for col,row,
        currected to have normal y-intercept 0 and normal slope fixed """
        if (col,row) in self.cache:
            #print("CACHE HIT")
            return self.cache[(col,row)]
        else:
            #print("CALCULATING")
            all_slopes = []
            good_data = []
            for i in range(len(self.data)):
                #print(f"PROCESSING DATA {i}")
                one_px = self.data[i][row,col]

                #make sure there is at least some data here
                if not one_px.mask.all():

                    params = linear_normal_fit(self.bias[i][row,col],one_px)
                    # note! if we're in real units, slope will be in mhos (ohms^-1).
                    #print(params.slope)
                    abslope = np.abs(params.slope)
                    #print(f"DATA {i} SLOPE IS {abslope:.3e}")
                    if not self.is_real_units and abslope > 4500 and abslope < 7000 :
                        all_slopes.append(params.slope)
                        good_data.append(i)
                    elif self.is_real_units and abslope >100 and abslope < 800:
                        all_slopes.append(params.slope)
                        good_data.append(i)
                    else:
                        print(f"Rejecting IV curve with 'normal slope' {params.slope:.2e}")
            avg_slope = np.ma.average(all_slopes)
            #print(f"average slope! {avg_slope:.2e}")
            new_data = []
            new_bias = []
            new_temp = []

            for i in good_data:
                one_px = self.data[i][row, col]
                new_intercept = fixed_slope_interceptor(self.bias[i][row,col], one_px, avg_slope)
                # print(new_intercept)
                if avg_slope < 0:
                    new_data.append(-(one_px - new_intercept))
                else:
                    new_data.append(one_px - new_intercept)
                new_bias.append(self.bias[i][row,col])
                new_temp.append(self.temperatures_int[i])
            self.cache[(col,row)] = (new_bias,new_data,new_temp,avg_slope)
        return(new_bias, new_data, new_temp, avg_slope)

def real_units(bias,fb):
    """ Given an array of biases and corresponding array of feedbacks (all in DAC units)
    calculate the actual current and voltage going through the TES.
    Returns: (TES voltage array, TES current array) in Volts and Amps respectively.
    Todo: Different chips may have different parameters. Need to figure out which 
    parameters vary and allow for that.
    """
    #----THE FOLLOWING NUMBERS ARE COPIED FROM CARL'S PYTHON SCRIPT----
    #all units Ohms    
    mce_bias_R = 467
    dewar_bias_R = 49
    shunt_R = 180e-6 # = 180 uOhm
    dewar_fb_R = 5280 # = one MileOhm (~~joke~~)

    #unitless
    butterworth_constant = 1218 # When running in data mode 2 and the low pass 
    #filter is in the loop, all signals are multiplied by this factor

    #relative to the TES inductance
    rel_fb_inductance = 9 # This means that for a change of 1 uA in the 
    # TES, the squid will have to change 9 uA to keep up

    #Volts. Note that these are bipolar voltages, eg the bias card can emit +- 5v
    max_bias_voltage = 5
    max_fb_voltage = 0.958 #seems a bit weird... still don't know where this number is from

    bias_dac_bits = 16
    fb_dac_bits = 14
    
    bias_raw_voltage = bias / 2**bias_dac_bits * max_bias_voltage * 2
    # last factor of 2 is because voltage is bipolar
    bias_current = bias_raw_voltage/(dewar_bias_R+mce_bias_R)
    fb_real_dac = fb / butterworth_constant
    fb_raw_voltage = fb_real_dac / 2**fb_dac_bits * max_fb_voltage * 2 
    # again, last factor of 2 is because voltage is bipolar
    fb_current = fb_raw_voltage / dewar_fb_R
    tes_current = fb_current / rel_fb_inductance
    
    shunt_current = bias_current - tes_current
    
    tes_voltage = shunt_current * shunt_R
    
    return(tes_voltage,tes_current)

class InteractiveIVPlotter(zt_plotting.ZeusInteractivePlotter):
    def __init__(self, directory):
        self.ivhelper = IVHelper()
        self.ivhelper.load_directory(directory)
        self.ivhelper.switch_to_real_units()
        self.last_colorbar = None
        # maybe get slopes for every px and use them as bitmaps?
        # ideally use sat power?
        shape = self.ivhelper.data[0].shape
        slopes = np.ones((shape[0],shape[1]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                _,_,_,slope = self.ivhelper.get_corrected_ivs(j,i)
                slopes[i,j] = slope

        data = 1/slopes
        data = np.ma.array(np.abs(data))
        data[data>0.007] = np.ma.masked
        super().__init__(data,
                         None,
                         ts=1  # ugly hack
                         # prevents parent class
                         # from attempting to auto
                         # generate ts 
                         )
    
    # override
    def bottom_plot(self):
        row, col = am.phys_to_mce(*self.click_loc)
        bias, data, temp, slope = self.ivhelper.get_corrected_ivs(col, row)
        cmap = plt.cm.plasma
        norm = self.ivhelper.get_temperature_colorbar_norm()
        for i in range(len(bias)):
            #print(data,bias)
            tes_voltage, tes_current= (bias[i],data[i])
            resistance = tes_voltage / tes_current
            power = tes_voltage * tes_current
            self.ax2.plot(power*1e12,resistance*1e3,".",c=cmap(norm(temp[i])))

        self.ax2.set_xlabel("power (pW)")
        self.ax2.set_ylabel("resistance (mOhm)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if self.last_colorbar:
            self.last_colorbar.update_normal(sm)
        else:
            self.last_colorbar = plt.colorbar(sm, ax=self.ax2, label="Temperature [mK]")
        self.ax2.set_ylim(0,6)
        self.ax2.set_xlim(0,20)

    # override
    def bottom_flat(self):
        flat_to_plot = self.flat[am.phys_to_mce(*self.click_loc)]
        flat_to_plot = flat_to_plot - min(flat_to_plot)

