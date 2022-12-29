import numpy as np
from astropy import units
from astropy import constants as const
import configparser
import matplotlib
from matplotlib import pyplot as plt 
import sys
from zeustools.bpio import load_data_and_extract
from zeustools.calibration import flat_to_wm2
from zeustools import transmission
from zeustools import plotting
from zeus2_toolbox import pipeline as z2pipl
from zeustools import data as ztdata
import importlib.resources as res
import os

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
    #print(sig[3],noise[3])
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


def calulate_bin_overlap(left_1, right_1, left_2, right_2):
    """ Calculate length of overlap of bins

    Args:
        left_1, right_1 : left and right bin edge values for bin 1
        left_2, right_2 : left and right bin edge values for bin 2

    Return:
        fraction of bin 2 covered by bin 1
    Courtesy of Patrick McNamee
    """
    
    return max(
        min(right_1, right_2) - max(left_1, left_2),
        0.
    )/(right_2-left_2)


def regridding_weighted_average(new_grid, orig_wl_data, orig_flux_data):
    """
    Performs a weighted average on unevenly sampled data with respect to a new grid. 
    :param new_grid: two-dimensional numpy array of desired output bin edges
    :param orig_wl_data: list of wavelength arrays for each observation.
    :param orig_flux_data: list of observation array tuples for each observation

    Thanks to Patrick McNamee
    """
    # Build data array. One element per data point.
    array_for_avg = []
    for wl_tuple, data_tuple in zip(orig_wl_data, orig_flux_data):
        px_array, flux_array, error_array = data_tuple
        wl_l_array, wl_r_array = wl_tuple
        for wl_l, wl_r, flux, err in zip(wl_l_array, wl_r_array, flux_array, error_array):
            data_entry = np.array([wl_l, wl_r, flux, err])
            if np.all(np.isfinite(data_entry)) and err > 1e-10:
                array_for_avg.append(data_entry)

    array_for_avg = np.array(array_for_avg)

    weights = np.zeros((new_grid.shape[0], array_for_avg.shape[0]))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] = calulate_bin_overlap(
                array_for_avg[j, 0],
                array_for_avg[j, 1],
                new_grid[i, 0],
                new_grid[i, 1]
            )/array_for_avg[j, -1]**2  # w_i = l_i/\sigma^2
    wt_sum = np.sum(weights, axis=1).reshape(weights.shape[:1] + (1,))
    norm_wts = weights/wt_sum
    regridded_weighted_avg = norm_wts @ array_for_avg[:, 2]
    regridded_err = 1/np.sqrt(wt_sum)
    return (regridded_weighted_avg, regridded_err)


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
    continuum, cont_err = np.ma.average(sig[idxs], weights=1/err[idxs]**2, returned = True)
    # print(idxs)
    return (spec_pos, sig-continuum, err, continuum, 1/np.sqrt(cont_err))


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
    transmission_calculator = transmission.AtmosphereTransmission(atm_trans_filename)
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


class ReductionHelper:
    def __init__(self):
        # =========================== reduction configuration ===========================

        self.obs_log_dir = "../../APEX_2022/obslogs"  # path to the folder containing the APEX html observation log
        # files, leave None if you don't need/have obs logs
        self.band = 400  # choose the band you would like to use for the array map, the
        # accepted values are 200, 350, 400 and 450, leave None if you
        # want to use the whole array map

        self.data_dir = "../../APEX_2022/20220916"  # path to the folder containing the data
        self.write_dir = "LPSJ0226"  # path to the folder to save the reduction result like figures
        # or tables, leave None if you want to use the current folder

        self.parallel = True  # flag whether to run the reduction in parallel mode
        self.table_save = True  # flag whether to save the reduction result as csv table
        self.plot = False  # flag whether to plot the reduction result
        self.plot_ts = True  # flag whether to plot the time series of each beam use in the
        # reduction
        self.reg_interest = None  # the region of interest of the array to plot in the format
        # of dictionary, e.g.
        # REG_INTEREST={'spat_spec':[1, 11]} if you only want to
        #  see the result of the pixel at [1, 11]
        # REG_INTEREST={'spat_spec_list':[[1, 11], [1, 12]]} if you
        #  want to see the result of a list of pixels
        # REG_INTEREST={'spat':1} if you want to check all the
        #  pixels at the spatial position 1
        # REG_INTEREST={'spat_ran':[0, 2]} if you want to check
        #  spatial position 0 through 2
        # REG_INTEREST={'spat_ran':[0, 2], 'spec_ran':[6, 10]} will
        #  show the result for all the pixels that are both in
        #  spatial position range 0 to 2, and spectral index range
        #  6 to 10
        # leave None to plot the time series of all the pixels in
        # the array, which can take a lot of time and slow down the
        # reduction; please refer to the API document for
        # ArrayMap.take_where() method for the accepted keywords
        self.plot_flux = True  # flag whether to plot the flux of each beam
        self.plot_show = False  # flag whether to show the figures, can slow down the reduction
        self.plot_save = True  # flag whether to save the figures as png files
        self.analyze = True  # flag whether to perform pixel performance analyze based on rms
        # and power spectrum
        self.do_desnake = False  # flag whether to perform desnaking
        self.ref_pix = [2,10]  # [spat_pos, spec_idx] of the reference pixel used to select
        # other good pixels to build the snake model, e.g. [1, 11] means
        # the pixel at spatial position 1 and spectral index 11 will be
        # used as the reference, only matters if DO_DESNAKE=True
        self.do_smooth = False  # flag whether to use a gaussian kernel to smooth the time
        # series to remove the long term structure, an alternative
        # de-trending process to desnaking
        self.do_ica = True  # flag whether to use ICA decomposition to remove the correlated
        # noise
        self.spat_excl = (0, 2)  # list of the range of the spatial positions to be excluded
        # from being used to build correlated noise model by ICA,
        # should include at least +/- one spatial position to the
        # target, e.g. if the source is placed at spat_pos=1,
        # SPAT_EXCL should be [0, 2], or even [0, 3] or broader range
        # if it appears extended
        # ========================= run the reduction pipeline =========================
        self.inject_signal = None
        with res.open_text(ztdata, "arrayA_map.dat") as array_file:
            map_arr = np.loadtxt(array_file, usecols=range(0, 4), dtype=int)
        #array_map = z2pipl.ArrayMap.read("/data2/share/zeus-2/ref/array_map_excel_alternative_20211101.csv")
        self.array_map = z2pipl.ArrayMap(arr_in=map_arr)
        self.array_map.set_band(self.band)
        self.obs_log = z2pipl.ObsLog.read_folder(self.obs_log_dir)
        self.sign = z2pipl.ObsArray.read_table("sign.csv").to_obs()
        self.g = z2pipl.ObsArray.read_table("thermal/g.csv").to_obs()
        self.n = z2pipl.ObsArray.read_table("thermal/n.csv").to_obs()
        self.Tc = z2pipl.ObsArray.read_table("thermal/Tc.csv").to_obs()
        self.Rn = z2pipl.ObsArray.read_table("thermal/Rn.csv").to_obs()
        self.g_bath = self.g * 1E3 * .13**(self.n.data_-1) / (self.Tc*1E-3)**(self.n.data_-1)
        self.zoc = transmission.ZeusOpticsChain(config="2019")

    def reduce(self, flat_header, data_header, bs_header):
        # Process skychops / flat files
        flat_result = z2pipl.reduce_skychop(
            flat_header=flat_header, 
            data_dir=self.data_dir, 
            write_dir=self.write_dir, 
            write_suffix="", 
            array_map=self.array_map, 
            obs_log=self.obs_log, 
            pix_flag_list=None,
            parallel=self.parallel, 
            return_ts=False, 
            return_pix_flag_list=True, 
            table_save=True,
            plot=self.plot, 
            plot_ts=self.plot_ts, 
            reg_interest=self.reg_interest, 
            plot_flux=self.plot_flux,
            plot_show=self.plot_show, 
            plot_save=self.plot_save, analyze=self.analyze
        )
        flat_flux, flat_err, flat_pix_flag_list = flat_result[:2] + flat_result[-1:]
        sign_use = self.sign if self.array_map is None else self.sign.to_obs_array(self.array_map)

        # Process bias step data
        bs_result = z2pipl.reduce_bias_step(
            data_header=bs_header,
            data_dir=self.data_dir,
            write_dir=self.write_dir,
            write_suffix="",
            array_map=self.array_map,
            obs_log=self.obs_log,
            pix_flag_list=None,
            sign=sign_use,
            parallel=self.parallel,
            do_smooth=True,
            do_clean=False,
            return_ts=False,
            return_pix_flag_list=True,
            table_save=True,
            plot=self.plot,
            plot_ts=self.plot_ts,
            reg_interest=self.reg_interest,
            plot_flux=self.plot_flux,
            plot_show=self.plot_show,
            plot_save=self.plot_save,
            analyze=self.analyze
        )
        bs_flux, bs_err, bs_pix_flag_list = bs_result[:2] + bs_result[-1:]

        # Process science data
        flat_flux, flat_err, pix_flag_list = 1, 0, []
        zobs_result = z2pipl.reduce_zobs(
            data_header=data_header,
            data_dir=self.data_dir,
            write_dir=self.write_dir,
            array_map=self.array_map,
            obs_log=self.obs_log,
            pix_flag_list=pix_flag_list,
            flat_flux=flat_flux,
            flat_err=flat_err,
            parallel=self.parallel,
            stack=self.do_ica,
            do_desnake=self.do_desnake,
            ref_pix=self.ref_pix,
            do_smooth=self.do_smooth,
            return_ts=True,
            do_ica=self.do_ica,
            spat_excl=self.spat_excl,
            return_pix_flag_list=True,
            table_save=self.table_save,
            plot=self.plot,
            plot_ts=self.plot_ts,
            reg_interest=self.reg_interest,
            plot_flux=self.plot_flux,
            plot_show=self.plot_show,
            plot_save=self.plot_save,
            analyze=self.analyze,
            use_hk=True,
            grat_idx=None,
            pwv=None,
            elev=None,
            inject_sig=self.inject_signal
        )

        #perform calculations...
        zobs_flux, zobs_err, zobs_pix_flag_list = zobs_result[:2] + zobs_result[-1:]
        grat_idx = z2pipl.configure_helper(obs=zobs_flux, keyword="gratingindex", supersede=True)
        pwv = z2pipl.configure_helper(obs=zobs_flux, keyword="mm PWV", supersede=True)
        elev = z2pipl.configure_helper(obs=zobs_flux, keyword="Elevation", supersede=True)
        suffix = ("" if not self.do_desnake else "_desnake") + ("" if not self.do_smooth else "_smooth") + \
        ("" if not self.do_ica else "_ica")
        # spectrum flattened by skychop
        flat_flux, flat_err, flat_pix_flag_list = flat_result[:2] + flat_result[-1:]
        norm_flux = zobs_flux/flat_flux
        norm_err = ((zobs_err/flat_flux)**2 + (zobs_flux/flat_flux * flat_err/flat_flux)**2).sqrt()
        pix_flag_list = zobs_pix_flag_list
        atm_trans = z2pipl.get_transmission_obs_array(self.array_map, pwv=pwv, elev=elev)
        # atm_trans_raw = z2pipl.get_transmission_raw_obs_array(self.array_map, pwv=pwv, elev=elev, grat_idx=grat_idx)
        bs_flux = bs_flux.proc_along_time("nanmean")
        bs_err = bs_err.proc_along_time("nanmean")
        zobs_flat_pix_flag_list = z2pipl.auto_flag_pix_by_flux( 
            norm_flux, norm_err, pix_flag_list=pix_flag_list)
        # spectrum with absolute calibration

        shunt_r = z2pipl.ObsArray(np.choose(np.any(self.array_map.mce_col_[:, None] == (0, 3, 4), axis=1), 
                                            (z2pipl.ACTPOL_R, z2pipl.CMB_R))[:, None], array_map=self.array_map)
        s_r = 0.5 - .5 * (bs_flux / 1.125 / shunt_r * self.Rn.to_obs_array(self.array_map) * .9)
        s_r.fill_by_mask(s_r.data_ < 0.1)
        s_r.to_table().write(os.path.join(self.write_dir, z2pipl.build_header(data_header)+"_bs_responsivity.csv"), overwrite=True)
        s_r_err = bs_err * .5 / 1.125 / shunt_r * self.Rn.to_obs_array(self.array_map) * .9

        bias = z2pipl.get_bias_obs(zobs_flux)
        zobs_i = z2pipl.fb_to_i_tes(zobs_flux)
        zobs_v = z2pipl.fb_to_v_tes(bias, zobs_flux, self.array_map.mce_col_)
        zobs_i_err = z2pipl.fb_to_i_tes(zobs_err)
        atm_trans = z2pipl.get_transmission_obs_array(self.array_map, pwv=pwv, elev=elev)
        atm_trans_raw = z2pipl.get_transmission_raw_obs_array(self.array_map, pwv=pwv, elev=elev, grat_idx=grat_idx)
        filt_trans = z2pipl.ObsArray(self.zoc.get_transmission_microns(self.array_map.array_wl_)[:, None], 
                                     array_map=self.array_map, obs_id="optical chain")
        filt_trans.to_table().write(os.path.join(self.write_dir, z2pipl.build_header(data_header)+"_inst_trans.csv"), overwrite=True)
        d_freq = z2pipl.ObsArray(abs(z2pipl.wl_to_freq(self.array_map.array_wl_) * 
                                 self.array_map.array_d_wl_ / self.array_map.array_wl_)[:, None], 
                                 array_map=self.array_map, obs_id="d freq")
        a_eff = np.pi * (12 / 2) ** 2  # [m2] effectiive area
        tele_eff = z2pipl.ObsArray(np.choose(self.array_map.array_wl_ > 400, (.18, .3))[:, None], 
                                   array_map=self.array_map, obs_id="telescope_efficiency") 
        tele_eff.to_table().write(os.path.join(self.write_dir, z2pipl.build_header(data_header)+"_tele_eff.csv"), overwrite=True)

        norm_flux = zobs_i * zobs_v / -sign_use / s_r / filt_trans / tele_eff / atm_trans / d_freq / 1E9 / a_eff / 1E-26 
        norm_err = zobs_v * ((zobs_i_err / s_r)**2 + (zobs_i * (s_r_err / s_r))**2).sqrt() / filt_trans / tele_eff / atm_trans / d_freq / 1E9 / a_eff / 1E-26
        pix_flag_list = bs_pix_flag_list + zobs_pix_flag_list

        norm_flux.to_table().write(os.path.join(self.write_dir, z2pipl.build_header(data_header)+suffix+"_spec_corr.csv"), overwrite=True)
        norm_err.to_table().write(os.path.join(self.write_dir, z2pipl.build_header(data_header)+suffix+"_spec_err_corr.csv"), overwrite=True)

        # zobs_corr_pix_flag_list = z2pipl.auto_flag_pix_by_flux(
        #     norm_flux, norm_err, pix_flag_list=pix_flag_list)
        return norm_flux, norm_err


if __name__ == '__main__':
    run_pipeline()
