from numba import njit
import numpy as np
from . import numba_polyfit as poly
import numba
import numba.typed
import zeustools as zt
"""
This module contains several routines written with Numba to process data
as quickly as possible. This allows us to process data on the fly
and send it to APEX for calibration purposes.
"""


@njit
def chunk_data(chop, cube):
    """
    Collect each data set associated with a single chop/wobble phase
    into a more manageable format.

    :param chop: numpy array of chop phase for each data point
    :param cube: MCE time series data file [rows,cols,timepts]
    :return: list of chunks, where each chunk is a numpy array with shape
        [rows,cols,numpts]
        list of phases, representing the phase of the chunk.
        
    e.g.
        if you have data like [[[1,2,3,4]]] (i.e. one pixel on the mce) 
        with chop like [1,1,0,0] you will get back
        list([[[1,2]]],[[[3,4]]]),list(1,0)
    """
    
    lastchop = chop[0]
    start_idx = 0
    if len(chop) != cube.shape[2]:
        raise RuntimeError("invalid shapes")
        
    chunks = numba.typed.List()
    phases = numba.typed.List()
    
    for i in range(len(chop)):
        if lastchop != chop[i] or i == len(chop)-1: 
            # when switching phases or ending data file
            # bundle data together:
            chunks.append(cube[:, :, start_idx:i])
            phases.append(lastchop)
            # set starting point for new chunk:
            start_idx = i 
            lastchop = chop[i]
            
    return (chunks, phases)


@njit
def chunk_data_1d(chop, ts):
    """
    Collect each data set associated with a single chop/wobble phase
    into a more manageable format.
    
    :param chop: numpy array of chop phase for each data point
    :param ts: 1-d time series
    :return: lists of floats instead of the arrays returned by :func:`chunk_data`
    """
    
    lastchop = chop[0]
    start_idx = 0
    if len(chop) != len(ts):
        raise RuntimeError("invalid shapes")
        
    chunks = numba.typed.List()
    phases = numba.typed.List()
    
    for i in range(len(chop)):
        if lastchop != chop[i] or i == len(chop)-1: 
            # when switching phases or ending data file
            # bundle data together:
            chunks.append(ts[start_idx:i])
            phases.append(lastchop)
            # set starting point for new chunk:
            start_idx = i 
            lastchop = chop[i]
            
    return (chunks, phases)


@njit
def offset_chunk_data(chop, cube):
    """
    Instead of chunking entire phases, chunk half-phases.
    otherwise, this is the same as :func:`chunk_data`
    """
    lastchop = chop[0]
    start_idx = 0
    if len(chop) != cube.shape[2]:
        raise RuntimeError("invalid shapes")
        
    chunks = numba.typed.List()
    phases = numba.typed.List()
    last_chop_size = 0
    for i in range(len(chop)):

        if lastchop != chop[i] or i == len(chop)-1: 
            # when switching phases or ending data file
            # bundle data together:
            half_chop_size = (i-start_idx)//2
            half_chop_idx = i - half_chop_size
            if half_chop_size < last_chop_size-1:
                break
            chunks.append(cube[:, :, start_idx:half_chop_idx])
            chunks.append(cube[:, :, half_chop_idx:i])
            phases.append(lastchop)
            phases.append(lastchop)
            # set starting point for new chunk:
            start_idx = i 

            lastchop = chop[i]
            last_chop_size = half_chop_size
            
    return (chunks, phases)


@njit
def offset_subtract_chunks(chunks, phases, lophase=0, weights=None):
    """ similar to subtract_chunks, but instead of doing on-off, we do
    average(half_of_on - half_of_off, -half_of_off + half_of_next_on) to
    remove snakes.

    :param chunks: list of 'chunks', first element returned by :func:`offset_chunk_data`.
    :param phases: list of phases, second element of offset_chunk_data return
    :param lophase: which chop phase to treat as "off" chop, i.e. the sign. usually 1, I think.
    :param weights: weights to associate with chunks of data, can be obtained with :func:`calculate_chunk_weights`
    """
    weight_values = weights if weights is not None else np.ones_like(chunks)
    chunk_shape = chunks.shape
    time_series = np.zeros((chunk_shape[0], chunk_shape[1], chunk_shape[2]//2-1))
    weights_out = np.ones_like(time_series)
    for i, phase in enumerate(phases):
        if i+1 < chunk_shape[2]:
            if phase != phases[i+1]:
                if phase == lophase:
                    sign = -1
                else:
                    sign = 1
                w1 = weight_values[:, :, i]
                w2 = weight_values[:, :, i+1]
                first_chunk = chunks[:, :, i] 
                second_chunk = chunks[:, :, i+1] 
                w = 1/(1/w1 + 1/w2)
                # normalize W because we are not averaging first and
                # second, instead we are just adding them.
                time_series[:, :, i//2] = (first_chunk-second_chunk)*sign
                weights_out[:, :, i//2] = w
    return time_series, weights_out


def offset_data_reduction(chop, cube, lophase = 1):
    # lophase = 1 appears to be correct for calibration data like Uranus
    cube = zt.dac_converters.correct_signs(cube)
    chunks, phases = offset_chunk_data(chop, cube) 
    #print(chunks[-1].shape,chunks[-2].shape, chunks[-3].shape,chunks[-4].shape,chunks[-5].shape)
    reduced_chunks = reduce_chunks(chunks)
    #print(reduced_chunks[1, 1])
    time_series, _ = offset_subtract_chunks(reduced_chunks, phases, lophase=lophase)
    #print(time_series[1,1])
    s = time_series.shape
    if s[2] % 2 != 0:
        time_series = time_series[:, :, :s[2]-1]
    #print(s)
    newshape = (s[0], s[1], s[2]//2, 2)
    #print(newshape)
    
    for_avg = time_series.reshape(newshape)
    #print(for_avg[1, 1])
    final_data = np.mean(for_avg, axis=3)
    return final_data


@njit
def numba_mean(args):
    return np.mean(args)


@njit
def numba_median(args):
    return np.median(args)


@njit
def numba_std(args):
    return np.std(args)


@njit
def reduce_chunks(chunks, fn=numba_median):
    #     np_chunks = np.array(chunks)
    #     np.median(np_chunks,axis=2)
    # in an ideal world this would be equivalent to above
    # unfortunately we often have a phase with one less data point than it should
    chunk_shape = chunks[0].shape
    new_chunks = np.zeros((chunk_shape[0], chunk_shape[1], len(chunks)))
    # chunk_shape[2] is the parameter that can vary
    for i in range(len(chunks)):
        chunk = chunks[i]
        for j in range(chunk_shape[0]):
            for k in range(chunk_shape[1]):
                new_chunks[j, k, i] = fn(chunk[j, k])
    return (new_chunks)


@njit
def reduce_chunks_1d(chunks, fn=numba_mean):
    #     np_chunks = np.array(chunks)
    #     np.median(np_chunks,axis=2)
    # in an ideal world this would be equivalent to above
    # unfortunately we often have a phase with one less data point than it should
    new_chunks = np.zeros((len(chunks)))
    for i in range(len(chunks)):
        chunk = chunks[i]
        new_chunks[i] = fn(chunk)
    return (new_chunks)


@njit
def subtract_chunks(reduced_chunks, phases, lowphase=0):
    chunk_shape = reduced_chunks.shape
    time_series = np.zeros((chunk_shape[0], chunk_shape[1], chunk_shape[2]//2))
    if phases[0] == lowphase:
        sign = -1
    else:
        sign = 1
        
    for i in range(0, chunk_shape[2], 2):
        lochunk = reduced_chunks[:, :, i+1]
        hichunk = reduced_chunks[:, :, i]
        time_series[:, :, i//2] = (hichunk - lochunk)*sign
    return time_series
    
        
@njit
def calculate_chunk_hash_std(chunks):
    return reduce_chunks(chunks, fn=numba_std)
    

@njit
def calculate_chunk_weights(chunks):
    std = calculate_chunk_hash_std(chunks)
    chunk_n = np.ones(len(chunks))
    for i, c in enumerate(chunks):
        chunk_n[i] = float(c.shape[2])
    return chunk_n/std**2 / 4
    # equivalent to 1/(sigma^2) where sigma = std/sqrt(n) the 4 is for 400 hz sample/100 hz thermal
    # the 4 is here, because we have less independent samples than chunk_n
    

def simple_data_reduction(chop, cube):
    cube = zt.dac_converters.correct_signs(cube)
    chunks, phases = chunk_data(chop, cube)
    reduced_chunks = reduce_chunks(chunks)
    time_series = subtract_chunks(reduced_chunks, phases)
    return time_series
    

@njit
def subtract_all_model_snake(cube, snake):
    shape = cube.shape
    subtracted_cube = np.zeros_like(cube)
    slopes = np.zeros((shape[0], shape[1]))
    intercepts = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            coeffs = poly.fit_poly(snake, cube[i, j], 1)
            m = coeffs[0]
            b = coeffs[1]
            subtracted_cube[i, j] = cube[i, j]-(snake*m+b)
            slopes[i, j] = m
            intercepts[i, j] = b
    return subtracted_cube, slopes, intercepts


def weighted_data_reduction(chop, cube, lophase = 1, exclude_last_phases = 2):
    """ Given chop data and data cube, reduce data and return a 
    flux value, weight, and standard deviation for each chop pair
    in lists
    """
    # lophase = 1 appears to be correct for calibration data like Uranus
    # Exclude_last_phases corrects for the fact that *sometimes* it seems the last phase or so of data has drastically reduced signal 
    # compared to the rest of the stream.
    cube = zt.dac_converters.correct_signs(cube)
    chunks, phases = offset_chunk_data(chop, cube)
    #print(chunks[-1].shape,chunks[-2].shape, chunks[-3].shape,chunks[-4].shape,chunks[-5].shape)
    c_wts = calculate_chunk_weights(chunks)
    reduced_chunks = reduce_chunks(chunks)
    #print(reduced_chunks[1, 1])
    time_series, sub_wts = offset_subtract_chunks(
        reduced_chunks, 
        phases, 
        lophase=lophase,
        weights=c_wts
    )
    #print(time_series[1,1])
    s = time_series.shape
    time_series = time_series[:, :, :s[2]-exclude_last_phases]
    sub_wts = sub_wts[:, :, :s[2]-exclude_last_phases]
    s = time_series.shape
    if s[2] % 2 != 0:  # Ensure we have an even number of half-chop phases for the next step
        time_series = time_series[:, :, :s[2]-1]
        sub_wts = sub_wts[:, :, :s[2]-1]
    #print(s)
    newshape = (s[0], s[1], s[2]//2, 2)
    #print(newshape)
    sub_std_wts = 1/np.std(time_series, axis=2)**2
    for_avg = time_series.reshape(newshape)
    w_for_avg = sub_wts.reshape(newshape)
    #print(for_avg[1, 1])
    final_data, final_wts = np.average(
        for_avg, 
        axis=3, 
        weights=w_for_avg, 
        returned=True
    )
    return final_data, final_wts, sub_std_wts*2  # multiply by two because we just averaged the half-chop phases


class ReductionStepResult:
    def __init__(self, data, err_stdev, err_hash, err_spec):
        self.data = data
        self.err_stdev = err_stdev
        self.err_hash = err_hash
        self.err_spec = err_spec


class Reducer:
    """ This class is designed to provide access to
    reduction routines and the various intermediate data
    products if required 

    Intermediate data products that we want:
        spectrum for each beam pair 
        spectrum for each beam?
        time-based standard deviation
        spectrum-based standard deviation
        hash/sqrt(n) error estimation

    desired api:
        reducer.load_raw_data( list of raw data files )
        reducer.make_beams( median/mean/etc, baseline subtraction option )
        reducer.make_beam_pairs( mean/median, baseline subtraction option )
        reducer.add_beam_pairs(mean/median, baseline subtraction option )
        reducer.shift_and_add_obs( ? )

    each step should populate the parameters 
    """
    def __init__(self):
        self.orig_data = None
        self.beams = None
        self.beampairs = None
        self.observations = None
        self.final_result = None

    def load_data_from_dict_lists(
        self,
        data_list,
        flat_list,
        folder_list,
        spatial_pos,
        spectral_pos  # or grating index and line wavelength?
    ):
        """ Set up reduction from the lists of dictionaries
        that I used to run my "helper object" for Bo's pipeline
        """
        pass

    def reduce_nods(self):
        pass

    def reduce(self):
        """ This function reduces all of the data as specified
        by setting the reduction params """
        pass

