import os
from zeustools import mce_data
import numpy as np
import matplotlib
from scipy import stats

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
    result = stats.linregress(bias[0:10], data[0:10])
    subdata = data - (bias * result.slope + result.intercept)
    transition_guess_idx = np.argmax(subdata)
    trans_guess = bias[transition_guess_idx]
    return trans_guess


def linear_normal_fit(bias, data):
    trans_guess = find_transition(bias, data)
    trans_guess += 1000
    end_idx = np.argmin(np.abs(bias-trans_guess))
    if end_idx == 0:
        end_idx = 10
    return(stats.linregress(bias[0:end_idx], data[0:end_idx]))


def fixed_slope_interceptor(bias, data, slope):
    trans_guess = find_transition(bias, data)
    trans_guess += 1000
    end_idx = np.argmin(np.abs(bias-trans_guess))
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
        self.bias = []

    def load_directory(self, directory):
        """
        Given a directory containing IV curve data, this function will load the IV 
        data and make it ready for processing.
        For best results, organize the folder so it only contains IV
        files and the associated run, out, and bias files.
        Also, each file should containthe bath temperature it was taken at
        in the format 110mK
        """
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
            self.bias.append(bias)

    def get_temperature_colorbar_norm(self):
        norm = matplotlib.colors.Normalize(vmin=min(self.temperatures_int),
                                           vmax=max(self.temperatures_int))
        return norm

    def get_corrected_ivs(self,col,row):
        """Returns an all iv curves for col,row,
        currected to have normal y-intercept 0 and normal slope fixed """
        
        all_slopes = []
        good_data = []
        for i in range(len(self.data)):
            one_px = self.data[i][row,col]
            #make sure there is at least some data here
            if not one_px.mask.all():
                params = linear_normal_fit(self.bias[i],one_px)
                #print(params.slope)
                abslope = np.abs(params.slope)
                if abslope > 4500 and abslope < 7000:
                    all_slopes.append(params.slope)
                    good_data.append(i)
        avg_slope = np.average(all_slopes)

        new_data = []
        new_bias = []
        new_temp = []

        for i in good_data:
            one_px = self.data[i][row,col]
            new_intercept = fixed_slope_interceptor(self.bias[i],one_px,avg_slope)
            if avg_slope<0:
                new_data.append(-(one_px - new_intercept))
            else:
                new_data.append(one_px - new_intercept)
            new_bias.append(self.bias[i])
            new_temp.append(self.temperatures_int[i])
        return(new_bias,new_data,new_temp)
