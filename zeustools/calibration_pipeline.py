import numpy as np
from astropy import units
from astropy import constants as const
import configparser
import matplotlib
from matplotlib import pyplot as plt 
import sys
from zeustools.bpio import load_data_and_extract
from zeustools.calibration import flat_to_wm2
from zeustools import atm_util
from zeustools import plotting


font = {'size': 18}
matplotlib.rc('figure', figsize=(8, 6))
matplotlib.rc('font', **font)


def flux_calibration(data,
                     flat_flux_density,  # W/m^2/bin
                     ):
    spec_pos, sig, noise = data
    
    scaled_signal = sig * flat_flux_density 
    
    scaled_err = noise * flat_flux_density
    return (spec_pos, scaled_signal, scaled_err)


def wavelength_calibration(data,
                           position_of_line,
                           bin_width  # km/s
                           ):
    spec_pos, _, _ = data
    velocity = (spec_pos - position_of_line) * bin_width
    return (velocity, data[1], data[2])


def cut(data, min_px, max_px):
    out = []
    for i in data:
        out.append(i[min_px:max_px])
    return out


def shift_and_add(data1, data2, px1, px2):
    """
    takes in two spectra. Shifts the second one spectrally by px_offset
    to align the line pixel between the two runs. Weights the spectra appropriately
    TODO: use np.average to clean up this mess.
    """
    arr = data1 + data2
    m = map(np.copy, arr)
    spec, sig, noise, spec2, sig2, noise2 = m
    nan_idxs = np.isnan(sig)
    nan_idx2 = np.isnan(sig2)

    sig[nan_idxs] = 0
    sig2[nan_idx2] = 0
    noise[nan_idxs] = 10e10
    noise2[nan_idx2] = 10e10
    noise[noise < 1e-6] = 10e10  # sometimes superconducting pixels show up unflagged
    noise2[noise2 < 1e-6] = 10e10
    spec -= px1
    spec2 -= px2 
    # That shifts the spectral pixel number so that the line is on position "0" 

    allspecs = np.append(spec, spec2)
    minspec = np.min(allspecs)
    maxspec = np.max(allspecs)
    outspec = np.arange(minspec, maxspec+1, dtype=int)
    outsig = np.zeros_like(outspec, dtype=float)
    outnoise = np.zeros_like(outspec, dtype=float)
    idx = (np.isin(outspec, spec)).nonzero()[0]
    idx2 = np.isin(outspec, spec2).nonzero()[0]
    outsig[idx] += sig/noise**2
    outsig[idx2] += sig2/noise2**2
    outnoise[idx] += 1/noise**2
    outnoise[idx2] += 1/noise2**2
    outnoise = 1/outnoise  # This sets outnoise to 1/wt
    outsig = outsig*outnoise  # divide by weight 
    outnoise = np.sqrt(outnoise)  # now outnoise is actually standard deviation
    outsig[outnoise > 1e9] = np.nan
    return (outspec, outsig, outnoise)

# def shift_add(data,line_px):
# 	"""
# 	takes in a bunch of spectra, then aligns them and does a weighted average
# 	params: list of data spectra
# 	line_px: list of the spectral positions with lines 
#   In-progress improvement to shift_and_add
# 	"""

# 	# create an array to hold the result
# 	specs = np.array([spec - i for (spec,_,_),i in zip(data,line_pix)])
# 	final_spec = np.arange(np.min(specs.flatten()),np.max(specs.flatten()))


def get_drop_indices(spec_pos, px_to_drop):
    line_px = np.array(px_to_drop)[:, None]
    boolarray = np.all(spec_pos != line_px, axis=0)
    return boolarray.nonzero()[0]


def contsub(data, line_px):
    spec_pos, sig, err = data 
    idxs = get_drop_indices(spec_pos, line_px)
    # print(sig)
    # print(idxs)
    continuum = np.average(sig[idxs], weights=1/err[idxs]**2)
    # print(idxs)
    return (spec_pos, sig-continuum, err)


def getcsvspec(label, spec):
    stringout = label+', '
    for i in spec:
        stringout += str(i)+", "
    return stringout


def plot_spec(spec, saveas, bounds, do_close = True):
    plt.figure(figsize=(8, 6))
    #spec[1][spec[2]>1e-16] = np.nan
    line = plt.step(spec[0], spec[1], where='mid')
    lncolor = line[0].get_c()
    plt.errorbar(spec[0], spec[1], spec[2], fmt='none', ecolor=lncolor)
    plt.ylabel("Flux, W m$^{-2}$ bin$^{-1}$")
    plt.xlabel("Velocity km s$^{-1}$")
    if bounds is not None and None not in bounds:
        plt.xlim(bounds[0:2])
        plt.ylim(bounds[2:4])
    plt.tight_layout()
    plt.savefig(saveas, dpi=300)
    if do_close:
        plt.close()


def run_pipeline():
    # Load in configuration values, and parse them into correct data types
    # TODO:are there better config file systems, and/or better ways of
    # organizing this?
    config = configparser.ConfigParser()
    if len(sys.argv) > 1:
        settings = sys.argv[1]
    else:
        settings = 'settings.ini'
    config.read(settings)
    globalvals = config["GLOBAL"]
    outputfile = globalvals["output_filename"]
    spatial_position = globalvals.getint("spat_pos")
    lambda_line = globalvals.getfloat("lambda_line_px")
    px_delta_lambda = globalvals.getfloat("px_delta_lambda")
    skytemp = globalvals.getfloat("sky_temp")
    cabintemp = globalvals.getfloat("cabin_temp")
    beamsize = globalvals.getfloat("beam_size_steradian")
    docalib = globalvals.getboolean("do_calib")
    unflatten = globalvals.getboolean("unflatten")
    docontsub = globalvals.getboolean("do_contsub")
    line_positions = globalvals["where_is_line_flux"]
    line_positions = list(map(int, line_positions.split(',')))
    ptcoup = globalvals.getfloat("pt_src_coupling")
    teleff = globalvals.getfloat("telescope_efficiency")
    do_indiv_plots = globalvals.getboolean("individual_reduction_plots")
    plot_min_x = globalvals.getfloat("plot_min_x")
    plot_max_x = globalvals.getfloat("plot_max_x")
    plot_min_y = globalvals.getfloat("plot_min_y")
    plot_max_y = globalvals.getfloat("plot_max_y")

    plt_bounds = (plot_min_x, plot_max_x, plot_min_y, plot_max_y)
    print(plt_bounds)
    atm_trans_filename = globalvals["atm_trans_table"]
    transmission_calculator = atm_util.TransmissionHelper(atm_trans_filename)
    transmission_calculator.observing_freq = (const.c/(lambda_line*units.micron)).to("GHz").value

    # Use astropy's nice units converter to get bin width in km/s
    px_kms = (px_delta_lambda/lambda_line*const.c).to("km/s").value

    # Figure out what the boolean options should be. 
    if unflatten:
        docalib = False

    # Convert the list of [REDUCTION*] sections in the settings.ini file to something more useful
    reductions_settings = [config[name] for name in config.sections() if name != "GLOBAL"]
    reductions = []
    pwvs = []
    # Loop over those sections to load in and calibrate each data scan chunk
    for rsett in reductions_settings:
        # the configparser doesn't throw an exception if a thing is 
        # not defined it just returns none.
        # I dont know how I feel 
        # about that

        # Get the per-data-chunk values
        pwv = rsett.getfloat("pwv")
        pwvs.append(pwv)
        alt = rsett.getfloat("alt")
        pwv_corr = pwv / atm_util.airmass_factor(alt)
        sky_transp = float(transmission_calculator.interp_internal_freq(pwv_corr))
        
        print(f"zpwv={pwv:.2f}, alt = {alt:.1f}, actual_pwv={pwv_corr:.2f}, transp = {sky_transp:.2f}")
        spec_pos = rsett.getint("spec_pos_of_line")
        minpx = rsett.getint("min_spec_px")
        maxpx = rsett.getint("max_spec_px")

        # load this data chunk, and chop off pixels we don't like

        data = load_data_and_extract(rsett["path"], spatial_position)
        data = cut(data, minpx, maxpx)

        if docalib:
            # Figure out how many w/m^2/bin the skychop was worth
            flat_flux_value = flat_to_wm2(
                sky_transp,
                lambda_line*units.micron,
                px_delta_lambda*units.micron,
                sky_temp=skytemp*units.K,
                cabin_temp=cabintemp*units.K,
                beam_size=beamsize*units.steradian).value
            # add telescope and sky stuff to calibration value
            flat_flux_value = flat_flux_value/sky_transp/ptcoup/teleff
            # Apply that scaling to the data
            data = flux_calibration(data, flat_flux_value)
        if do_indiv_plots:
            plot_spec((data[0]-spec_pos, data[1], data[2]), 
                      f"{outputfile}_{rsett.name}.png", 
                      plt_bounds)
        reductions.append((data, spec_pos))

    # If we only have one reduction, shifting-and-adding can't happen
    if len(reductions) > 1:
        for i in range(len(reductions)-1):
            data, spec_pos = reductions[i]
            data2, specpos2 = reductions[i+1]
            reductions[i+1] = (shift_and_add(data, data2, spec_pos, specpos2), 0)
            # this is kind of ugly because it's 1 am and gordon wants this yesterday
            # TODO: cleaner code. play around with "reduce" function and stuff
    else:
        # Move the spectrum so that the line is on pixel
        # due to the fact I'm using tuples I have to rebuild the tuple...
        spec, data, err, wt = reductions[-1][0]
        line_px = reductions[-1][1]

        reductions[-1] = ((spec-line_px, data, err, wt), 0)

    # Now that we have a co-added spectrum, we should subtract the continuum
    addedspec = reductions[-1][0]

    if docontsub:
        addedspec = contsub(addedspec, line_positions)
        
        # I told the user to make line_positions relative to the spectral pixel of the line.
        # When we shifted and added, we shifted the line to be at "spectral position" 0
        # So if the line started on pixel 7 (and 8) the user should've given us
        # "the line is at pixel 7" and "flux can be found on pixels 0 and 1."
        # Now that the line is at 0, the correct places to find flux is 0 and 1.
        # This almost makes sense to 1 AM me. TODO: clean up the explanation and make sure it's true
    velspec = wavelength_calibration(addedspec, 0, px_kms)

    # plot_spec(velspec, f"{outputfile}.png", plt_bounds)
    # That plot is pretty redundant now that we have the atm plotter
    if docalib:
        y_ax = 1e-18
    else:
        y_ax = 1

    plotting.spectrum_atm_plotter(velspec[0],
                                  velspec[1],
                                  velspec[2],
                                  outputfile,
                                  transmission_calculator,
                                  min(pwvs),
                                  y_scaling = y_ax,
                                  bounds = plt_bounds)
    plt.savefig(f"{outputfile}_atmosphere.png")
    # Finally, output the csv for gordon.
    with open(outputfile+".csv", 'w') as csvf:
        if docalib:
            labely = "signal W m^-2 bin^-1"
        else:
            labely = "raw data (fraction of flat)"
        labelx = "velocity relative to galaxy redshift km/s"
        csvf.write(getcsvspec(labely, velspec[1])+'\n')
        csvf.write(getcsvspec(labelx, velspec[0])+'\n')
        csvf.write(getcsvspec("error in signal same units as signal", velspec[2])+'\n')


if __name__ == '__main__':
    run_pipeline()
