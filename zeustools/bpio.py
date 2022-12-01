import numpy as np
import pandas
from numpy import ma

def extract_one_spatial_position(a, spat):
    """ DEPRECATED, since Bo's script now uses different formatting
    Takes the numpy z file from Bo's script and return a 1-d spectrum
    of a single spatial position.

    :param a: this is the npz load object, loaded from Bo's reduction script
    :param spat: This is the spatial position you would like to extract.
    :return: a 4xn array, where return[0:4] are spatial position, signal, noise, and
        weight.
    """
    array_map=a["arr_0"]
    signal =  a["arr_1"]
    noise =   a["arr_2"]
    weight =  a["arr_3"]
    idxs = (array_map[:,3]==spat).nonzero()[0]
    sorting_indexes = array_map[idxs, 0].argsort()
    sorted_idxs = idxs[sorting_indexes]
    return np.array([array_map[sorted_idxs, 0],
                    signal[sorted_idxs],
                    noise[sorted_idxs],
                    weight[sorted_idxs]])


def load_data_and_extract(fname, spat, err_file=None):
    """ Automatically loads an csv file from Bo's reduction script
    and returns a 1-d spectrum of a single spatial position.

    :param fname: this is the filename of the .npz file from Bo's script
    :param spat: This is the spatial position you would like to extract.
    :return: tuple of: 1d array of spectral position, 1d array of spectrum, 
        and 1d array of noise. Both sig and noise arrays are masked
        arrays, where the mask has been set to True for nans
    """
    signal = ma.array(pandas.read_csv(fname).iloc[spat,4:],dtype=float)
    err_file = err_file or fname.replace("flux","err")
    noise = ma.array(pandas.read_csv(err_file).iloc[spat,4:],dtype=float)
    nan_idx = np.logical_or(np.isnan(signal),np.isnan(noise))
    signal[nan_idx] = ma.masked
    noise[nan_idx] = ma.masked
    spectral_position = np.arange(len(signal))
    return spectral_position,signal,noise


def extract_from_beamfile(fname, beam, spat, arrnums):
    """  DEPRECATED, since Bo's script now uses different formatting
    Sometimes you want to read out the reduced data for individual 
    beams from Bo's reduction script. This is the function that handles that.
    Given a filename, this will grab one spatial position from that beam.

    :param fname: the file name of the beamfile that you want to load
    :param beam: the beam number that you want to extract
    :param spat: the spatial position you want
    :param arrnums: determines whether you get the data before or after the dead pixel
        subtraction, clean routine, independent component analysis, etc. Ask Bo what number you should input.
    :return: a 4xn array, where return[0:4] are spatial position, signal, noise, and
        weight.  


    """
    flatname = fname.replace("beam_spec","flat")
    flatname = flatname.replace("_subdead","")
    a = np.load(fname)
    flat = np.load(flatname)
    flatspec = flat['arr_1']
    array_map=a["arr_0"]
    signal =  a[f"arr_{arrnums*3+1}"][beam]/flatspec
    noise =   a[f"arr_{arrnums*3+2}"][beam]/flatspec
    weight =  a[f"arr_{arrnums*3+3}"][beam]
    idxs = (array_map[:,3]==spat).nonzero()[0]
    sorting_indexes = array_map[idxs, 0].argsort()
    sorted_idxs = idxs[sorting_indexes]
    return np.array([array_map[sorted_idxs, 0],
                    signal[sorted_idxs],
                    noise[sorted_idxs],
                    weight[sorted_idxs]])
