from matplotlib import pyplot as plt
from IPython.display import display
from IPython.display import Markdown as md
import numpy as np
import zeustools as zt
import astropy.units as u
import astropy.constants as c
from zeustools.reduction_tools import regridding_weighted_average
import matplotlib
import num2tex
from zeustools import calibration

atm = zt.transmission.AtmosphereTransmission()
gc = zt.grating_cal.GratingCalibrator()

num2tex.option_configure_help_text = False
num2tex.configure(exp_format='cdot', option_configure_help_text=False)
num2tex = num2tex.num2tex


def plot_spec(spec, errbar_limit=1, **kwargs):
    #plt.figure(figsize=(8, 6))
    #spec[1][spec[2]>1e-16] = np.nan

    useful_data = spec[1]
    useful_data[spec[2] > errbar_limit] = np.nan
    useful_data[spec[2] < 1e-6] = np.nan
    useful_data[np.isnan(spec[2])] = np.nan
    useful_data[useful_data == 0] = np.nan
    line = plt.step(spec[0], useful_data, where='mid', **kwargs)
    lncolor = line[0].get_c()
    plt.errorbar(spec[0], 
                 useful_data, 
                 spec[2], 
                 fmt='none', 
                 ecolor=lncolor, 
                 alpha=line[0].get_alpha())
    plt.ylabel("Flux, Jy")
    plt.tight_layout()


def wl_contsub(data, line_wl_range):
    wl, sig, err = data 
    idxs = np.logical_or(wl < line_wl_range[0], wl > line_wl_range[1])
    # print(sig)
    # print(idxs)
    continuum, cont_err = np.ma.average(sig[idxs], 
                                        weights=1/err[idxs]**2, 
                                        returned = True)
    # print(idxs)
    return (wl, sig-continuum, err, continuum, 1/np.sqrt(cont_err))


def wl_baseline_std(data, line_wl_range):
    wl, sig, err = data[0:3]
    idxs = np.logical_or(wl < line_wl_range[0], wl > line_wl_range[1])
    # print(sig)
    # print(idxs)
    std = np.ma.std(sig[idxs])
    # print(idxs)
    return std


def wl_linear_contsub(data, line_wl_range):
    wl, sig, err = data 
    idxs = np.logical_or(wl < line_wl_range[0], wl > line_wl_range[1])
    #print(idxs)
    res = np.polyfit(wl[idxs], sig[idxs], 1, w=1/err[idxs]**2)
    new_sig = sig - res[0]*wl - res[1]
    #print(res)
    #plt.plot(wl,res[0]*wl + res[1],label="Continuum fit")
    return (wl, new_sig, err, res)


def wl_zeroth_contsub(data, line_wl_range):
    wl, sig, err = data 
    idxs = np.logical_or(wl < line_wl_range[0], wl > line_wl_range[1])
    res = np.polyfit(wl[idxs], sig[idxs], 0, w=1/err[idxs]**2)
    new_sig = sig - res[0]
    #print(res)
    return (wl, new_sig, err)


def drop_nans(data):
    x, y, e = data[0:3]
    nans, f = zt.nan_helper(y)
    x1, y1, e1 = (x[~nans], y[~nans], e[~nans])
    nans, f = zt.nan_helper(x1)
    return (x1[~nans], y1[~nans], e1[~nans])


def spec_to_wm2(data, w_km_s):
    wl, flux, err = data[0:3]
    centers = wl
    widths = w_km_s
    flux_wm2 = calibration.jy_to_wm2(flux*u.jansky,
                                     centers*u.micron,
                                     widths*u.km/u.s).value
    err_wm2 = calibration.jy_to_wm2(err*u.jansky,
                                    centers*u.micron,
                                    widths*u.km/u.s).value
    return (wl, flux_wm2, err_wm2)


def line_estimator(data, line_wavs, bins):
    wl, flux, err = data[0:3]
    widths_km_s = (bins[:, 1]-bins[:, 0])/wl*c.c.to("km/s").value
    
    line_px_mask = np.logical_not(np.logical_or(wl < line_wavs[0],
                                                wl > line_wavs[1]))
    line_px_idx = line_px_mask.nonzero()[0]
    
    wm2 = spec_to_wm2(data, widths_km_s)
    
    line_flux, line_err = line_calc(wm2, line_px_idx)
    obs_wav = data[0][line_px_idx[0]]
    line_jykms = calibration.wm2_to_jy_km_s(line_flux*u.W/u.m**2,
                                            obs_wav*u.micron)
    err_jykms = calibration.wm2_to_jy_km_s(line_err*u.W/u.m**2,
                                           obs_wav*u.micron)
    return line_flux, line_err, line_jykms, err_jykms


def line_calc(data, line_px):
    sq_err = 0
    flux = 0
    # print(line_px)
    for i in line_px:
        sq_err += data[2][i]**2
        flux += data[1][i]
        # print(data[1][i])
        
    err = np.sqrt(sq_err)
   
    return (flux, err)


def displaymd(text):
    display(md(text))


def data_handler(
    flux_files,
    err_files,
    spatial_pos,
    grating_idx, 
    line_wl_range, 
    name, 
    plot_indiv=False,
    line_name="[OIII] 88 $\\mu$m",
    errbar_limit=1,
    error_data_cut=None,
    plot_details=True,
    save_final_fig=True
):
    """Run an automatic process pipeline style reduction on data that has been
    reduced with bo's pipeline.
    :param flux_files: list of files containing flux data, usually ending in `_spec_corr`
    :param err_files: list of files containing error data, ending in `spec_err_corr`
    :param spatial_pos: list of spatial positions observed in each file
    :param grating_idx: grating index for each file
    :param line_wl_range: 2-tuple or array containing minimum wavelength of line and maximum
    :param name: Name to be displayed in titles and plots
    :plot_indiv: boolean, whether to make an additional plot containing all of the data file plotted individually
    :line_name: Line name, is displayed in a plot legend.
    :errbar_limit: cutoff, largest error bar to be displayed in plots. Does not affect calculations except for the errbar analysis calculation
    :error_data_cut: cutoff, largest error bar for data to be included in calculations!
    """
    if line_wl_range[0] < 400:
        band = 350
        spectral_array = np.arange(20)
    else:
        band = 450
        spectral_array = np.arange(20, 40)
        
    displaymd(f"## Automatic Processing for {name}")
    # Load data
    displaymd("### Raw Data")
    flux_data = [zt.bpio.load_data_and_extract(f, s, err_file=e) for f, s, e in zip(flux_files, spatial_pos, err_files)]
    # calculate original grids
    orig_wl_edges = []
    for spat, idx in zip(spatial_pos, grating_idx):
        orig_wl_edges.append((gc.phys_px_to_wavelength(spectral_array-0.5, spat, band, idx),
                              gc.phys_px_to_wavelength(spectral_array+0.5, spat, band, idx)))
    grid_arr = np.array(orig_wl_edges)
    
    # create new grid
    new_grid = np.arange(np.min(grid_arr), np.max(grid_arr), 0.6)
    bins_new = np.stack((new_grid[:-1], new_grid[1:]), axis=1)
    if band == 450:
        for i, (px, flux, error) in enumerate(flux_data):
            flux_data[i] = (px[20:], flux[20:], error[20:])
    else:
        for i, (px, flux, error) in enumerate(flux_data):
            flux_data[i] = (px[:20], flux[:20], error[:20])
    # AVERAGE ALL DATA TO NEW GRID
    if plot_indiv:
        displaymd("### Spectra of individual observations")
        plt.figure()
        i = 0
        for spat, idx, flux in zip(spatial_pos, grating_idx, flux_data):
            orig_wl_centers = gc.phys_px_to_wavelength(spectral_array, spat, band, idx)
            plot_spec((orig_wl_centers, flux[1]+i, flux[2]), errbar_limit=errbar_limit)
            plt.title(f"{name} Individual Obs. Spectra")
            plt.xlabel("Wavelength [$\\mu$m]")
            plt.axhline(y=0, color='k', linewidth=0.5)
            plt.tight_layout()
            i += 1
        plt.show()
        
    regridded_flux, regridded_err = regridding_weighted_average(bins_new, 
                                                                orig_wl_edges, 
                                                                flux_data, 
                                                                error_data_cut=error_data_cut)
    new_grid_centers = np.average(bins_new, axis=1)
    
    # Raw data plot
    plt.figure()
    plot_spec((new_grid_centers,
               regridded_flux,
               regridded_err[:, 0]),
              errbar_limit=errbar_limit,
              label="Raw spectrum")

    #plt.axvline(obs_wav)
    plt.title(f"{name} Spectrum")
    plt.xlabel("Wavelength [$\\mu$m]")
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    displaymd("### Continuum Subtraction")
    new_data = drop_nans((new_grid_centers, regridded_flux, regridded_err[:, 0]))
    _, _, bins = drop_nans((new_grid_centers, regridded_flux, bins_new))
    subbed_data = wl_contsub(new_data, line_wl_range)
    linear_sub_data = wl_linear_contsub(new_data, line_wl_range)
    
    # compute line location
    line_px_mask = np.logical_not(np.logical_or(new_data[0] < line_wl_range[0],
                                                new_data[0] > line_wl_range[1]))
    line_px_idx = line_px_mask.nonzero()[0]
    step_idx = np.repeat(line_px_idx, 2)
    
    # print continuum flux
    continuum, cont_err = subbed_data[3:5]  # continuum flux Jy and error
    displaymd(f"Continuum flux: ${continuum:.3f} \\pm {cont_err:.3f}$ Jy")
    
    # Make the fancy/advanced figure!!!
    plt.figure(figsize=(8, 8))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3.5, 1])
    ax1 = plt.subplot(gs[0])

    if plot_details:
        plot_spec((new_grid_centers,
                   regridded_flux,
                   regridded_err[:, 0]),
                  errbar_limit=errbar_limit,
                  label="Raw spectrum",
                  alpha=0.5,
                  color="C1")
    plot_spec(linear_sub_data,
              errbar_limit=errbar_limit,
              label="Continuum-subtracted",
              color="C0")
    if plot_details:
        plt.plot(new_data[0], 
                 new_data[0]*linear_sub_data[3][0]+linear_sub_data[3][1],
                 alpha=0.5, label="Continuum model",
                 color="C2")
    plt.axhline(y=0, color='k', linewidth=1)

    plt.title(f"Processed Spectrum, {name}")
    x = bins[line_px_idx].flatten()
    y = linear_sub_data[1][step_idx]
    plt.fill_between(x, y, color="C0", label=line_name, alpha=0.3)
    plt.grid(which='major', linewidth=0.4)
    plt.legend()
    ax1.tick_params(axis="x", labelbottom=False)

    ax2 = plt.subplot(gs[1], sharex=ax1)
    plt.grid(which='major', linewidth=0.4)
    plt.xlabel("Wavelength [$\\mu$m]")
    plt.ylabel("Atmospheric\nTransmission")
    new_grid_centers[np.isnan(regridded_flux)] = np.nan
    x = np.linspace(np.nanmin(new_grid_centers), np.nanmax(new_grid_centers), 500)
    t = atm.interp_um(x, 0.5)
    ax2.plot(x, t)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(f"{name}.png",dpi=300)
    plt.show()
    # End fancy/advanced figure

    displaymd("### Line flux estimation")
    line_flux, line_err, line_jykms, err_jykms = line_estimator(linear_sub_data, line_wl_range, bins)
    displaymd(f"3-sigma limit: ${num2tex(line_err*3,precision=3)}$ W m$^{{-2}}$")
    displaymd(f"Computed line flux: ${num2tex(line_flux,precision=3)} \\pm {num2tex(line_err,precision=3)}$ W m$^{{-2}}$")
    displaymd(f"Computed line flux {line_jykms:.0f} $\\pm$ {err_jykms:.0f}")
    displaymd(f"SNR = {line_jykms/err_jykms:.0f}")

    average_err_bar = np.ma.mean(linear_sub_data[2])
    baseline_computed_err = wl_baseline_std(linear_sub_data, line_wl_range)

    displaymd("### Errorbar information")
    displaymd(f"Average error bar from final spectrum = {average_err_bar:.2f}")
    displaymd(f"Standard deviation of final spectrum baseline = {baseline_computed_err:.2f}")
    return line_flux, line_err
