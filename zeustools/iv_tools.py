import os
from zeustools import mce_data
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats
from scipy import optimize
from zeustools import plotting as zt_plotting
import zeustools as zt
import zeustools.dac_converters as dacc

am = zt.ArrayMapper()  


def super_remover(data):
    """ Attempt to remove unlocked data from IV curves by finding the first jump of larger than 1e7 

    :param data: masked array containing IV curve datacube.
    """
    # this advanced function can do whole mce data files
    # though it assumes that "data" is a masked array already

    # first find all the huge jumps
    diffs = data[:, :, 1:]-data[:, :, :-1]
    good = np.abs(diffs) < 1e7
    # now find the largest bias with a huge jump (remember data is collected at large bias first)
    last_false_index = np.argmin(good, axis=2)
    # Now we have the index of the first huge jump for every pixel.

    # bleh, let's just for loop it
    # set all entries after the determined index to "masked"
    # if this becomes a speed bottleneck then let's use that magic 
    # jit thing, numba
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i, j, last_false_index[i, j]:] = np.ma.masked
    
    return data


def find_transition(bias, data):
    """ Attempt to find the superconducting transition by fitting the normal branch and looking for deviations.
    """
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
    guess = find_transition(bias, data)
    orig_guess = guess
    # I don't really like this, but it's the best I've got right now.
    # Technically it's actually extremely robust, it just seems hacky
    # The index of the transition is obviously not part of the normal branch,
    # So we move to increased bias to ensure we make it to the normal branch
    if guess < 1:
        # assume real units!
        guess += 4e-8
    else:
        # assume dac units!
        guess += 1000
    # print(f"guess={guess:.2e}")
    # Find the closest index to the guess bias value
    idx = np.argmin(np.abs(bias-guess))
    t_idx = np.argmin(np.abs(bias-orig_guess))+1
    return idx, t_idx


def linear_normal_fit(bias, data):
    end_idx, trans_idx = find_transition_index(bias, data)

    # print(end_idx)
    if end_idx == 0:
        end_idx = 10
    regression = stats.linregress(bias[0:end_idx], data[0:end_idx])
    return regression, trans_idx


def fixed_slope_interceptor(bias, data, slope):
    end_idx, _ = find_transition_index(bias, data)
    valid_bias = bias[0:end_idx]
    valid_data = data[0:end_idx]
    
    de_sloped = valid_data - valid_bias*slope
#     plot(valid_bias,valid_data)
#     plot(valid_bias,valid_bias*slope)
    return (np.average(de_sloped))


class IVHelper:
    """ Load in an IV curve or several IV curves and perform useful operations with them"""
    def __init__(self):
        pass
        self.filenames = []
        self.temperatures = []
        self.temperatures_int = []
        self.mce_data = []  
        self.data = []  # list of data files
        self.is_real_units = False
        self.bias = []  # List of bias arrays
        self.cache = {}

    def __add__(self, other):
        result = IVHelper()
        assert self.is_real_units == other.is_real_units
        result.filenames = self.filenames + other.filenames
        result.temperatures = self.temperatures + other.temperatures
        result.temperatures_int = self.temperatures_int + other.temperatures_int
        result.mce_data = self.mce_data + other.mce_data
        result.data = self.data + other.data
        result.is_real_units = self.is_real_units
        result.bias = self.bias + other.bias
        # Could figure out how to add cache too :P
        return result

    def load_file(self, file, temp=None):
        self.filenames.append(file)
        if temp is None:
            if "mK" in file:
                i = file.find("mK")
            elif "mk" in file:
                i = file.find("mk")
            else:
                raise ValueError("Could not determine temperature from file name, no temperature override provided")
            self.temperatures.append(file[i-3:i+2])
            self.temperatures_int.append(int(file[i-3:i]))
        else:
            assert type(temp) is int
            self.temperatures.append(temp)
            self.temperatures_int.append(temp)

        mcefile = mce_data.SmallMCEFile(file)
        self.mce_data.append(mcefile)
        mce_data_read = mcefile.Read(row_col=True).data
        mce_data_masked = np.ma.array(mce_data_read)
        # let's go ahead and clean up the data now
        # we can always unmask it later if we need to

        mce_data_cleaned = super_remover(mce_data_masked)
        # This removes a lot of junk, but doesn't always get it all

        self.data.append(-dacc.correct_signs(mce_data_cleaned))
        # Minus sign was determined empirically!!!!
        bias = np.loadtxt(file+".bias", dtype=int, skiprows=1)
        tile_arg = list(mce_data_cleaned.shape)
        tile_arg[2] = 1  # this is probably going to be [33,24,1] which is
        # how we want to tile the bias array. Now the bias array will 
        # match the data array in shape, which doesn't change anything
        # until we switch to real units, where the bias voltage can be different
        # at the same time for different pixels. 
        bias = np.tile(bias, tile_arg)
        self.bias.append(bias)

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
            self.load_file(directory + file)

    def switch_to_real_units(self):
        """ changes all internal data files from DAC units to real units
        As of now, this step is irreversible, so if you want to go back
        to DAC units you'll have to reload the directory. But really, it should be easy
        since I keep the mce data array.
        """

        # Do nothing if we're already in real units
        if self.is_real_units:
            return
        else:
            # Clear cache, b/c if any IVs have been calculated, they're in wrong units now
            self.cache = {}
            self.is_real_units = True
            # for i in range(len(self.bias)):
            #     self.bias[i], self.data[i] = real_units(self.bias[i], 
            #                                             self.data[i], 
            #                                             whole_array=True)
            # cant do this until intercepts are zeroed

    def get_temperature_colorbar_norm(self):
        norm = matplotlib.colors.Normalize(vmin=min(self.temperatures_int),
                                           vmax=max(self.temperatures_int))
        return norm

    def get_corrected_ivs(self, col, row, clean_again=True):
        """Returns all iv curves for col,row,
        currected to have normal y-intercept=0 
        Note that if the real_units member variable is true 
        the last value of the returned tuple is an actual resistance in ohms
        instead of a slope in mhos like it used to be."""
        try:
            # print("CACHE HIT")
            return self.cache[(col, row)]
        except KeyError:
            pass
        # print("CALCULATING")
        all_slopes = []
        good_data = []
        for i in range(len(self.data)):
            # print(f"PROCESSING DATA {i}")
            one_px = self.data[i][row, col]

            # make sure there is at least some data here
            if one_px.mask.all():
                continue

            params, trans_idx = linear_normal_fit(self.bias[i][row, col], 
                                                  one_px)
            if clean_again:
                self.data[i][row, col, trans_idx:] = np.ma.masked
                # Be aware, although this will mask a ton of squid unlocks
                # it will also mask a few real superconducting branches
                # however, I don't think that's too big a deal.
                # also remember the bad data will be the last data points
                # even though they're to the left in the plot
            # print(params.slope)
            abslope = np.abs(params.slope)
            # print(f"DATA {i} SLOPE IS {abslope:.3e}")
            if 4500 < abslope < 7000:
                all_slopes.append(params.slope)
                good_data.append(i)
            # elif self.is_real_units and abslope > 100 and abslope < 700:
            #     all_slopes.append(params.slope)
            #     good_data.append(i)
            # real units dont work that way anymore
            # else:
            #     # print(f"Rejecting IV curve with 'normal slope' {params.slope:.2e}")
            #     pass

        avg_slope = np.ma.average(all_slopes)
        # print(f"average slope! {avg_slope:.2e}")
        new_data = []
        new_bias = []
        new_temp = []

        for i in good_data:
            one_px = self.data[i][row, col]
            new_intercept = fixed_slope_interceptor(self.bias[i][row, col], one_px, avg_slope)
            # print(new_intercept)
            # if avg_slope < 0:
            #     new_data.append(-(one_px - new_intercept))
            # else:
            real_fb = one_px - new_intercept
            if self.is_real_units:
                voltage, current = dacc.real_units(self.bias[i][row, col], real_fb, col=col)
                new_data.append(current)
                new_bias.append(voltage)
            else:
                new_data.append(real_fb)
                new_bias.append(self.bias[i][row, col])
            new_temp.append(self.temperatures_int[i])
        if self.is_real_units:
            avg_slope = dacc.dac_normal_slope_to_ohms(avg_slope, col)
        self.cache[(col, row)] = (new_bias, 
                                  new_data, 
                                  new_temp, 
                                  avg_slope)
        return (new_bias, new_data, new_temp, avg_slope)


class InteractiveIVPlotter(zt_plotting.ZeusInteractivePlotter):
    def __init__(self, 
                 directory,
                 power_temp=130, 
                 file=False, 
                 file_temp_override=None,
                 real_units=True):
        # If Plot_power = True, the 2-d array plot/colorbar
        # will show the saturation powers at the temperature = power_temp
        # Otherwise (plot_power=False) it will show normal resistance
        if type(directory) is IVHelper:
            self.ivhelper = directory
        elif not file:
            self.ivhelper = IVHelper()
            self.ivhelper.load_directory(directory)
        else:
            self.ivhelper = IVHelper()
            self.ivhelper.load_file(directory, temp=file_temp_override)
        if real_units:
            self.ivhelper.switch_to_real_units()

        self.last_colorbar = None
        # maybe get slopes for every px and use them as bitmaps?
        # ideally use sat power?
        self.power_temp = power_temp
        self.build_data()
        self.plot_clean = True

        super().__init__(self.rn_data,
                         None,
                         # ts=1,  # ugly hack
                         # prevents parent class
                         # from attempting to auto
                         # generate ts / now broken/fixed
                         flat=1  # basically same
                         # tricks the onclick method
                         # into letting us display the "flat"
                         # which in our case means plotting 
                         # IV curve instead of PR curve
                         # on right click
                         )
    
    def __sub__(self, other):
        # VERY SPECIFIC. BE CAREFUL
        result = InteractiveIVPlotter(self.ivhelper, power_temp=self.power_temp)
        if self.power_temp != other.power_temp:
            print("warning, subtracting powers from different temperatures")
        result.ivhelper = self.ivhelper + other.ivhelper
        result.power_data = self.power_data - other.power_data
        # result.power_data[result.power_data < 0] = np.ma.masked
        return result

    def build_data(self):
        shape = self.ivhelper.data[0].shape
        slopes = np.ones((shape[0], shape[1]))
        all_temps = self.ivhelper.temperatures_int
        powers = np.ma.masked_all((shape[0], shape[1], len(all_temps)))
        powers.fill_value = np.nan

        for i in range(shape[0]):
            for j in range(shape[1]):
                bias, data, temps, slope = self.ivhelper.get_corrected_ivs(j, i)
                slopes[i, j] = slope
                for k, temp in enumerate(temps):
                    try:
                        t_idx = all_temps.index(temp)
                        power = bias[k] * data[k]
                        power[power < 0] = np.ma.masked
                        powers[i, j, t_idx] = np.ma.min(power)
                    except ValueError:
                        # print(f"power_temp={self.power_temp} was not found for px col,row={j},{i}")
                        pass

        # data = 1/slopes
        data = np.ma.array(slopes)
        data[data > 0.007] = np.ma.masked

        powers[powers < 0] = np.ma.masked
        self.powers = powers
        self.power_data = powers[:, :, all_temps.index(self.power_temp)]
        self.rn_data = data        

    def get_min_bias_and_resistance(self):
        shape = self.ivhelper.data[0].shape
        all_temps = self.ivhelper.temperatures_int
        min_r = np.ma.masked_all((shape[0], shape[1], len(all_temps)))
        min_r.fill_value = np.nan
        min_b = np.ma.masked_all((shape[0], shape[1], len(all_temps)))
        min_b.fill_value = np.nan

        for i in range(shape[0]):
            for j in range(shape[1]):
                bias, data, temps, slope = self.ivhelper.get_corrected_ivs(j, i)
                for k, temp in enumerate(temps):
                    try:
                        t_idx = all_temps.index(temp)
                        power = bias[k] * data[k]
                        power[power < 0] = np.ma.masked
                        idx = np.argmin(power)
                        min_b[i, j, t_idx] = bias[k][idx]
                        min_r[i, j, t_idx] = bias[k][idx]/data[k][idx]
                    except ValueError:
                        pass
        return (min_b, min_r)
    
    def interactive_plot_power(self, array='all'):
        array = zt.array_name_2(array)[0]
        data = np.ma.copy(self.power_data)
        if array == "a":
            data[:, 12:] = np.ma.masked
        elif array == "b":
            data[:, :12] = np.ma.masked
        self.data = data
        self.interactive_plot()
        self.cb.set_label("Power (W)")

    def interactive_plot_rn(self):
        self.data = self.rn_data
        self.interactive_plot()
        self.cb.set_label("TES Normal Resistance (ohm)")

    def update_colorbar(self, sm, ax):
        if self.last_colorbar and ax is self.ax2:
            self.last_colorbar.update_normal(sm)
        else:
            self.last_colorbar = plt.colorbar(sm, ax=ax, label="Temperature [mK]")

    # override
    def bottom_plot(self):
        row, col = am.phys_to_mce(*self.click_loc)
        self.power_resistance_plot(row, col, self.ax2)

    def power_resistance_plot(self, row, col, ax, cbar_inset = None):
        bias, data, temp, slope = self.ivhelper.get_corrected_ivs(col, row)
        cmap = plt.cm.plasma
        norm = self.ivhelper.get_temperature_colorbar_norm()
        for i in range(len(bias)):
            # print(data,bias)
            tes_voltage, tes_current = (bias[i], data[i])
            resistance = tes_voltage / tes_current
            power = tes_voltage * tes_current
            if not self.plot_clean:
                power = power.data
                resistance = resistance.data
            if self.ivhelper.is_real_units:
                ax.plot(power*1e12, resistance*1e3, ".", c=cmap(norm(temp[i])))
            else:
                ax.plot(power, resistance, ".", c=cmap(norm(temp[i])))
        if self.ivhelper.is_real_units:
            ax.set_xlabel("power (pW)")
            ax.set_ylabel("resistance (mOhm)")
            ax.set_ylim(0, 6)
            ax.set_xlim(0, 20)
        else:
            ax.set_xlabel("squid fb * det bias")
            ax.set_ylabel("det bias / squid fb")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if cbar_inset is not None:
            print("ping")
            plt.gcf().colorbar(
                sm, 
                cax=cbar_inset, 
                label="Temperature [mK]", 
                orientation="horizontal")
        else:
            self.update_colorbar(sm, ax)

    # override
    def bottom_flat(self):
        self.ax2.clear()  # haha, we probably just made a plot here. NEVERMIND! gone.
        row, col = am.phys_to_mce(*self.click_loc)
        cmap = plt.cm.plasma
        norm = self.ivhelper.get_temperature_colorbar_norm()
        bias, data, temp, slope = self.ivhelper.get_corrected_ivs(col, row)
        for i in range(len(data)):
            if self.plot_clean:
                self.ax2.plot(bias[i], data[i], c=cmap(norm(temp[i])))
            else:
                self.ax2.plot(bias[i].data, data[i].data, c=cmap(norm(temp[i])))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        self.update_colorbar(sm, ax=self.ax2)
        if self.ivhelper.is_real_units:
            self.ax2.set_xlim(0, 3e-7)
            self.ax2.set_ylim(0, 7e-5)
            self.ax2.set_xlabel("bias voltage (V)")
            self.ax2.set_ylabel("feedback current (A)")
        else:
            self.ax2.set_xlabel("bias")
            self.ax2.set_ylabel("sq feedback")           

    def detectors_hist(self, 
                       title,
                       bins=30,
                       arrays=[350, 450],
                       plot_rn=False,
                       data_override=None,
                       xlabel="none",
                       ax = None):
        """
        Plots a histogram of detectors based on a parameter. Default paramater is 
        saturation power, but can be overridden by passing `plot_rn=True` or by passing
        custom data to `data_override`. 
        :param title: Title of plot. If `None`, plot will remain untitled.
        :param bins: Bins parameter to pass to matplotlib's hist function.
            As such, this can be either an integer (number of bins) or an array of bin edges.
            If it is an integer, the bins used for each array will be different!
            In the past, if this was an integer, the first array plotted would set the bins
            and each subsequent array would use the same bin edges. This could lead to missing
            pixels if the first array plotted had a smaller range than subsequent ones.
        :param arrays: A list specifying the detector arrays to be included on the plot.
            Should be a list containing any combination of `200`, `350`, `450`, or `600`.
        :param plot_rn: A Boolean indicating that the normal resistance should be plotted
            instead of saturation power. Has no effect if `data_override` is also passed.
        :param data_override: An mce-style array containing the parameter you'd like to plot
            for each pixel. This overrides the plot_rn parameter and expects you to set the 
            `xlabel` parameter too
        :param xlabel: The x-axis label to use. Only used if `data_override` is not `None`.
        :param ax: Which axis, if any, to use for plotting the histogram. If `None`, we create
            a new figure automatically.

        :return: the bins used by whichever plot was made last. Warning: use with caution
            as this does not necessarily represent the bins used for every detector array!

        """
        if ax is None:
            plt.figure()
            ax = plt.gca()
        if data_override is not None:
            dat_pw = data_override.filled()
            ax.set_xlabel(xlabel)
        elif plot_rn:
            dat_pw = self.rn_data.filled()*1000
            ax.set_xlabel("resistance (m$\\Omega$)")
        else:
            dat_pw = self.power_data.filled()*1e12
            ax.set_xlabel("power (pW)")

        if 200 in arrays:
            n, bins_new, p = ax.hist(
                dat_pw[:, 12:19].flatten(),
                bins=bins,
                alpha=0.8,
                label="200 $\\mu$m",
                color="C0"
            )
        if 350 in arrays:
            n, bins_new, p = ax.hist(
                dat_pw[:, :5].flatten(),
                bins=bins,
                label="350 $\\mu$m",
                color="C2"
            )
        if 450 in arrays:
            n, bins_new, p = ax.hist(
                dat_pw[:, 5:12].flatten(),
                bins=bins,
                alpha=0.8,
                label="450 $\\mu$m",
                color="C1"
            )
        if 600 in arrays:
            n, bins_new, p = ax.hist(
                dat_pw[:, 19:21].flatten(),
                bins=bins,
                alpha=0.8,
                label="600 $\\mu$m",
                color="C3"
            )
        if title is not None:
            ax.set_title(title)
        if plt.rcParams['text.usetex']:
            ax.set_ylabel("\\# detectors")
        else:
            ax.set_ylabel("# detectors")

        ax.legend()
        return bins_new


class InteractiveThermalPlotter(InteractiveIVPlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.ma.masked_all((self.powers.shape[0], self.powers.shape[1]))
        self.Tc = np.ma.masked_all((self.powers.shape[0], self.powers.shape[1]))
        self.n = np.ma.masked_all((self.powers.shape[0], self.powers.shape[1]))
        self.K.fill_value = np.nan
        self.Tc.fill_value = np.nan
        self.n.fill_value = np.nan
        T_bath = np.array(self.ivhelper.temperatures_int)
        for i in range(self.powers.shape[0]):
            for j in range(self.powers.shape[1]):
                try:
                    P_sat = self.powers[i, j]
                    good_data = np.logical_not(P_sat.mask)
                    popt, pcov = optimize.curve_fit(
                        psat_fitter,
                        T_bath[good_data],  
                        P_sat[good_data],
                        p0=[3.1, 4.3e-18, 172.5], 
                        maxfev=8000
                    )
                    self.K[i, j] = popt[1]
                    self.Tc[i, j] = popt[2]
                    self.n[i, j] = popt[0]
                except RuntimeError:
                    pass
                except TypeError:
                    pass
                except ValueError:
                    pass
        test_1 = self.n < 1.5
        test_2 = self.Tc < 100
        tests = np.logical_or(test_1, test_2)
        self.K[tests] = np.ma.masked
        self.Tc[tests] = np.ma.masked
        self.n[tests] = np.ma.masked

    def interactive_plot_k(self, array='all'):
        self._interactive_plot_array(self.K*1e12, array)
        self.cb.set_label("K [pW/mK$^n$]")

    def interactive_plot_tc(self, array="all"):
        self._interactive_plot_array(self.Tc, array)
        self.cb.set_label("Tc [mK]")

    def interactive_plot_n(self, array="all"):
        self._interactive_plot_array(self.n, array)
        self.cb.set_label("n")

    def interactive_plot_g(self, array="all"):
        self._interactive_plot_array(self.n*self.K*(self.Tc)**(self.n-1)*1e12, array)
        self.cb.set_label("G [pW/mK]")

    def _interactive_plot_array(self, data, array):
        data = np.ma.copy(data)
        array = zt.array_name_2(array)[0]
        if array == "a":
            data[:, 12:] = np.ma.masked
        elif array == "b":
            data[:, :12] = np.ma.masked
        self.data = data
        self.interactive_plot()

    def bottom_plot(self):
        row, col = am.phys_to_mce(*self.click_loc)
        self.power_bath_plot(row, col, self.ax2)

    def power_bath_plot(self, row, col, ax):
        T_bath = self.ivhelper.temperatures_int
        power = self.powers[row, col]
        n = self.n.data[row, col]
        k = self.K.data[row, col]
        t = self.Tc.data[row, col]
        ax.plot(T_bath, power*1e12, '.', label="Data")
        ax.plot(T_bath.sorted(),
                psat_fitter(T_bath, n, k, t)*1e12,
                label=f"K={k*1e12:.2e} [pW/mK]\nT$_c$={t:.0f} [mK]\nn={n:.2f}")
        ax.legend()
        ax.set_xlabel("T$_{bath}$ [mK]")
        ax.set_ylabel("P$_{sat}$ [pW]")


def psat_fitter(Tbath, n, K, T_c):
    return K*(T_c**n-Tbath**n)


class InteractiveThermalGPlotter(InteractiveIVPlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.G = np.ma.masked_all((self.powers.shape[0], self.powers.shape[1]))
        self.Tc = np.ma.masked_all((self.powers.shape[0], self.powers.shape[1]))
        self.n = np.ma.masked_all((self.powers.shape[0], self.powers.shape[1]))
        self.G.fill_value = np.nan
        self.Tc.fill_value = np.nan
        self.n.fill_value = np.nan
        T_bath = np.array(self.ivhelper.temperatures_int)
        for i in range(self.powers.shape[0]):
            for j in range(self.powers.shape[1]):
                try:
                    P_sat = self.powers[i, j]
                    good_data = np.logical_not(P_sat.mask)
                    popt, pcov = optimize.curve_fit(psat_g_fitter,
                        T_bath[good_data],  # noqa: E128
                        P_sat[good_data],
                        p0=[3.1, 0.3e-12, 172.5],
                        bounds=([3, 5e-14, 150], [4, 2e-12, 220]),
                        sigma=np.full_like(P_sat[good_data], 1e-12),
                        maxfev=8000)
                    #print(popt)
                    self.G[i, j] = popt[1]
                    self.Tc[i, j] = popt[2]
                    self.n[i, j] = popt[0]
                except RuntimeError:
                    pass
                except TypeError:
                    pass
                except ValueError:
                    pass
        test_1 = self.n < 1.5
        test_2 = self.Tc < 100
        tests = np.logical_or(test_1, test_2)
        self.G[tests] = np.ma.masked
        self.Tc[tests] = np.ma.masked
        self.n[tests] = np.ma.masked

    def interactive_plot_tc(self, array="all"):
        self._interactive_plot_array(self.Tc, array)
        self.cb.set_label("Tc [mK]")

    def interactive_plot_n(self, array="all"):
        self._interactive_plot_array(self.n, array)
        self.cb.set_label("n")

    def interactive_plot_g(self, array="all"):
        self._interactive_plot_array(self.G*1e12, array)
        self.cb.set_label("G [pW/mK]")

    def _interactive_plot_array(self, data, array):
        data = np.ma.copy(data)
        array = zt.array_name_2(array)[0]
        if array == "a":
            data[:, 12:] = np.ma.masked
        elif array == "b":
            data[:, :12] = np.ma.masked
        self.data = data
        self.interactive_plot()

    def bottom_plot(self):
        row, col = am.phys_to_mce(*self.click_loc)
        self.power_bath_plot(row, col, self.ax2)

    def power_bath_plot(self, row, col, ax):
        T_bath = self.ivhelper.temperatures_int
        power = self.powers[row, col]
        n = self.n.data[row, col]
        g = self.G.data[row, col]
        t = self.Tc.data[row, col]
        ax.plot(T_bath, power*1e12, '.', label="Data")
        ax.plot(sorted(T_bath), 
                psat_g_fitter(sorted(T_bath), n, g, t) * 1e12,
                label=f"G={g*1e12:.2e} [pW/mK]\nT$_c$={t:.0f} [mK]\nn={n:.2f}")
        ax.legend()
        ax.set_xlabel("T$_{bath}$ [mK]")
        ax.set_ylabel("P$_{sat}$ [pW]")
        if col < 12:
            ax.set_ylim(0, 5)
        else:
            ax.set_ylim(0, 15)

    def thermal_hist_G(self, title, arrays=[350, 450], bins=30, ax=None):
        return self.detectors_hist(
            title,
            bins=bins,
            arrays=arrays,
            data_override=self.G*1e12,
            xlabel="G [pW/mK]",
            ax=ax
        )

    def thermal_hist_Tc(self, title, arrays=[350, 450], bins=30, ax=None):
        return self.detectors_hist(
            title,
            bins=bins,
            arrays=arrays,
            data_override=self.Tc,
            xlabel="Tc [mK]",
            ax=ax
        )


def psat_g_fitter(Tbath, n, g, T_c):
    return g * (T_c**n - Tbath**n)/n / T_c**(n-1)


if __name__ == "__main__":
    iv_plotter = InteractiveIVPlotter("data/")
