import numpy as np


def extract_one_spatial_position(a, spat):
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


def load_data_and_extract(fname, spat):
    a = np.load(fname)
    return extract_one_spatial_position(a, spat)


def extract_from_beamfile(fname, beam, spat, arrnums):
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
