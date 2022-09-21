# This script attempts to bias the array using bias step measurements
# For best results the array should not be entirely superconducting

# Note this script will be run as python 3
#import numpy as np
import subprocess
from glob import glob
from zeustools import mce_data
import zeustools as zt

MAS_PATH="/data/cryo/current_data/"

def run_bias_step(fname,frames):
    subprocess.call([
        "bias_step",
        "--filename",
        fname,
        "--frames",
        frames,

    ])


def set_new_bias(arr):
    subprocess.call([
        "mas_param",
        "set",
        "tes_bias_idle"
    ] + arr)
    subprocess.call([
        "mce_cmd",
        "-qx",
        "rb",
        "tes",
        "bias"
    ] + arr)


# def get_bias(arr):
#     m = MCE()
#     m.read("tes", "bias")


def get_next_filename(fname):
    #if "{num}" in fname:
    #find file names and the next sequential number:
    path = MAS_PATH
    files = glob(path + fname.format(num="????"))
    if len(files) == 0:
        no = 0
    else:
        lastfile = sorted(files)[-1]
        no = lastfile.replace(path + fname.format(num=""), "")
        no = int(no)+1
    return fname.format(num="{:04d}".format(no))


def main(tune_px):
    am = zt.ArrayMapper()
    f = get_next_filename("bias_step_magic_{num}")
    run_bias_step(f,400)
    md = mce_data.SmallMCEFile(MAS_PATH+f)
    didi = zt.bias_step_di_di(md)
    # didi > 1 = superconducting
    # didi >= 0 = normal-ish
    # didi ~ -0.05 = good
    # superconducting_px = didi > 1
    # normal_px = np.logical_and(didi > 0, didi < 1)
    # good_px = didi < -0.02 
    bias = zt.get_bias_array(md)
    finished_cols = []
    for i, p in enumerate(tune_px):
        slope = didi[am.phys_to_mce(p)]
        print(f"di/di column {i}: {slope:.4f}")
        if slope > 1:
            #superconducting
            bias[i] += 300
            print("                     increase by 300")
        elif slope > 0.05:
            #normal
            bias[i] -= 200
            print("                     decrease by 200")
        elif slope > -0.02:
            #getting closer...
            delta = int(20 * (slope+0.05)*100)
            bias[i] -= delta
            print(f"                     decrease by {delta}")
        else:
            print("                     no change")
            finished_cols.append(i)
    accept = input("accept? [y/n]")
    if accept == 'y':
        set_new_bias(bias)
        for i in sorted(finished_cols)[::-1]:
            del tune_px[i]
        if len(tune_px!=0):
            main(tune_px)
    else:
        exit()

if __name__ == "__main__":
    main([
        (3, 0, 400),
        (8, 0, 400),
        (11, 0, 400),
        (15, 0, 400),
        (18, 0, 400),
        (24, 0, 400),
        (27, 0, 400),
        (29, 0, 400),
        (32, 0, 400),
        (36, 3, 400)
    ])
