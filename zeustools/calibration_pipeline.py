import numpy as np
from zeustools import transmission
from zeus2_toolbox import pipeline as z2pipl
from zeustools import data as ztdata
import zeustools as zt
import importlib.resources as res
import os
import pandas as pd


def baseline_subtractor(
    flux_pd,    
    err_pd,
    flat_pd,
    ignore_px = []
):
    """ MODIFIES flux_pd so that the baseline is subtracted out of all the 350 $\\mu$m data 
    Note:ONLY 350 microns! 
    Note: This function is JANKY
    """ 
    for i in range(len(flux_pd)):  # Go through all the rows
        flux_arr = np.array(flux_pd.iloc[i, 4:], dtype=float)  #I'm not a huge fan of pandas indexing.
        err_arr = np.array(err_pd.iloc[i, 4:], dtype=float)  # I'm always using iloc for everything
        spatial_pos = flux_pd.spatial_position[i]  # But yeah, we extract the ith row and make it into 
        # a numpy array because I can deal with those.

        flat_arr = np.array(flat_pd[flat_pd.spatial_position == spatial_pos].iloc[:, 4:], dtype=float)[0]
        # Do the same for the flat
        #print(flat_arr)
        for array in (0, 1):  # Process 350 and 450 array separately
            x = np.arange(20*array, 20*array+20)  # Spatial positions for whichever array we're on 
            y = flux_arr[x]/flat_arr[x] 
            e = np.abs(err_arr[x]/flat_arr[x])
            nans, f = zt.nan_helper(y)
            nans = np.logical_or(nans, e < 1e-8)

            for i_px in ignore_px:
                nans[i_px] = True  

            if np.all(nans):
                res = 0
            else:
                # Linear fit, excluding nans
                try:
                    #res = np.polyfit(x[~nans], y[~nans], 0, w=1/e[~nans]**2)
                    res = np.average(y[~nans], weights=1/e[~nans]**2)
                except np.linalg.LinAlgError:
                    print("Linear algebra error at x,y,e:", x, y, e)
                    return
            for x1, j in enumerate(x):
                # Write the data back to the pandas frames. 
                flux_pd.at[i, f"spectral_index={j}"] = (y[x1]-res)*1000
                err_pd.at[i, f"spectral_index={j}"] = (e[x1])*1000
    return flux_pd, err_pd


def make_subtracted_beampair_file(
    flux_file: str, 
    err_file: str,
    flat_file: str,
    replace="beam_pairs",
    replace_with="subtd_beam_pairs",
    replace_with_2="subtd_spec",
    replace_with_std="subtd_spec_std",
    line_px = []
):
    flux_pd = pd.read_csv(flux_file)  # Read flux, error, and flat tables
    err_pd = pd.read_csv(err_file)
    flat_pd = pd.read_csv(flat_file)
    flux_pd, err_pd = baseline_subtractor(flux_pd, err_pd, flat_pd, ignore_px=line_px)  # Subtract
    flux_pd.to_csv(flux_file.replace(replace, replace_with), index=False)
    # Write all the individual subtracted beampairs
    
    result_pd = flux_pd.iloc[:0, :].copy()
    res_err_pd = flux_pd.iloc[:0, :].copy()
    std_err_pd = flux_pd.iloc[:0, :].copy()
    for i in range(9):  #for every spatial position average all the beam pairs
        spatial_df = flux_pd[flux_pd.spatial_position == i]  # extract row from df
        spatial_flux = np.ma.array(spatial_df.iloc[:, 4:])  # put it into a numpy array
        spatial_err = np.ma.array(err_pd[flux_pd.spatial_position == i].iloc[:, 4:])
        spatial_err[spatial_err < 0.08] = np.ma.masked

        spatial_flux[np.isnan(spatial_flux)] = np.ma.masked  # filter nans
        spatial_err[np.isnan(spatial_err)] = np.ma.masked
        #print(np.ma.min(spatial_err), np.ma.median(spatial_err))
        result, weight = np.ma.average(spatial_flux, axis=0, weights=1/spatial_err**2, returned=True)
        std = np.ma.std(spatial_flux,axis=0)/np.sqrt(spatial_flux.count(axis=0))
        # Average!
        result = result.filled(np.nan)
        weight = weight.filled(np.nan)
        std = std.filled(np.nan)
        result_pd.loc[i] = list(spatial_df.iloc[0, :4])+list(result)
        res_err_pd.loc[i] = list(spatial_df.iloc[0, :4])+list(1/np.sqrt(weight))
        std_err_pd.loc[i] = list(spatial_df.iloc[0, :4])+list(std)
        # Write new pandas tables. This is apparently a bad way to do that, but shrug
        
    result_pd.to_csv(flux_file.replace(replace, replace_with_2), index=False)
    res_err_pd.to_csv(err_file.replace(replace, replace_with_2), index=False)
    std_err_pd.to_csv(err_file.replace(replace, replace_with_std), index=False)
    return res_err_pd
        

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
        self.ref_pix = [2, 10]  # [spat_pos, spec_idx] of the reference pixel used to select
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

    def make_file_name(self, specification: dict, kind = "") -> str:
        """ Given a 'header' style specification dictionary,
        return the beginning of a filename that corresponds to it.
        ie, {"skychop":[(0,20)]}, 'flux' -> "skychop_0000-0020_flux.csv"
        """
        filename = ''
        lownums = []
        highnums = []
        for key in specification:
            for tup in specification[key]:
                lownums.append(tup[0])
                highnums.append(tup[1])
            filename = key

        num_tuple = (min(lownums), max(highnums))
        if kind != '':
            kind = "_" + kind + '.csv'
        file_spec = f"{filename}_{num_tuple[0]:04}-{num_tuple[1]:04}{kind}"
        return os.path.join(self.write_dir, file_spec)

    def make_file_name_suffix(self, specification: dict, kind: str) -> str:
        return f"{self.make_file_name(specification)}{self.get_file_suffix()}_{kind}.csv"

    def get_file_suffix(self):
        return ("" if not self.do_desnake else "_desnake") + ("" if not self.do_smooth else "_smooth") + \
        ("" if not self.do_ica else "_ica")

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
        suffix = self.get_file_suffix()

        norm_flux.to_table().write(os.path.join(self.write_dir, z2pipl.build_header(data_header)+suffix+"_spec_corr.csv"), overwrite=True)
        norm_err.to_table().write(os.path.join(self.write_dir, z2pipl.build_header(data_header)+suffix+"_spec_err_corr.csv"), overwrite=True)

        # zobs_corr_pix_flag_list = z2pipl.auto_flag_pix_by_flux(
        #     norm_flux, norm_err, pix_flag_list=pix_flag_list)
        return norm_flux, norm_err

# if __name__ == '__main__':
#     run_pipeline()