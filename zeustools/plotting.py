import numpy as np
from matplotlib import pyplot as plt
import zeustools as zt
from numpy import ma
from matplotlib.colors import ListedColormap
from enum import Enum
from astropy import constants as const
from astropy import units
import matplotlib
import traceback

am = zt.ArrayMapper()  

BAD_DATA_CMAP = ListedColormap(["tab:orange", "deeppink"])


def plot_array(somedata, ax=None, s=60, bad_px=False):
    """ Plots mce data nicely but in a less advanced way than Bo.

    :param somedata: this should be an array of size (33,24), i.e., mce data format.
        Each element is the 'flux' value of that pixel. This routine will colorize based on flux.

    :param ax: a pyplot axis object in case you want to use yours instead of generating a new one.

    :param s: size value for the squares representing pixels. Tweak this to your liking.

    :return: A tuple containing 3 mpl objects:

        0. scatter plot object
        1. color bar object
        2. axis that the plots were added to

    :rtype: 3-Tuple
    """
    somedata.fill_value = np.nan
    mce_grid = am.grid_map()
    if ax is None:
        # we need an axis to plot on, so if one wasn't supplied, make it.
        g = plt.figure(dpi=300)
        g.gca().set_aspect(1)
        ax = g.gca()
    else:
        # we always want our pixels to be square.
        ax.set_aspect(1)
        
    # This makes a scatter plot. A square shaped marker is placed at every spectral and spatial
    # position, colored according to the flux value of that pixel.
    stuff = ax.scatter(mce_grid[:, :, 1], -mce_grid[:, :, 0], c=somedata.filled(), s=s, marker='s')
    cb = plt.colorbar(stuff, orientation='horizontal', ax=ax, aspect=50, fraction=0.05, shrink=0.9, pad=0.07)
    if bad_px:
        baddata = somedata.copy()
        baddata.mask = np.logical_not(somedata.mask)
        ax.scatter(mce_grid[:, :, 1], -mce_grid[:, :, 0], c=baddata.filled(), s=s, marker='s', cmap=BAD_DATA_CMAP)

    # The sizing of the colorbar is really fineagely. Might need to work on that TODO: sometime.
    return stuff, cb, ax


class FakeEvent:
    """ Used internally in case you want to manually trigger one of the event handlers"""
    def __init__(self, key, xdata, ydata):
        self.key = key
        self.xdata = xdata
        self.ydata = ydata
        self.button = key


def get_physical_location(xdata, ydata):
    """ Used internally. The center of each pixel is at integer values, so the pixel extends
    from value - 0.5 to value+0.5. Here we correct for that and also tell you which array 
    your pixel is on. Usable with the plot_array and ZeusInteractivePlotter.
    """

    xclk = (xdata+0.5)//1  # clk stands for click
    yclk = (ydata+0.5)//1

    if yclk <= -10:
        # this is the 400 micron array
        spatial = -yclk-10
        spectral = xclk
        array = 400
    elif xclk < 12:
        # this is the 600 micron array
        spatial = -yclk
        spectral = xclk
        array = 600
    else:
        spatial = -yclk
        spectral = xclk-20
        array = 200
    return spectral, spatial, array


class ZeusInteractivePlotter():
    """ This is an awesome object that gives you some of the features of the zalpha script!

    Tested in ipython notebooks, you first need to initialize such that you are plotting interactively.
    This is accomplished by running something like ::

        %matplotlib notebook

    Then, you can pass in a data array in mce/flux format (like you would get out of the super naive
    data reduction ``np.median(-cube[:,:,chop==1],axis=2)+np.median(cube[:,:,chop==0],axis=2)`` or use 
    in ``plot_array()`` above) and a data cube (raw mce format), call ``interactive_plot()`` and TA DAH!
    You can now click on any pixel on the array and immediately view its time series. You have a few 
    additional options too:

        1. Click on a pixel to view its time series
        2. Right click to view a pixel's time series and flat time series (if the flat cube is provided)
        3. Middle click (or if that doesn't work press 'a' while hovering) to add a time series without clearing the previous one
        4. Press 'f' while hovering to add a time series+flat series without clearing
        5. Press 'c' while hovering to mask that pixel, removing it from the colorbar calculation and array plot.

    This is a work in progress, and is not very robust. If you suspect something is wrong, try accessing
    ``error`` member variable. If it exists it contains the most recent error that occurred. You can also::

        import traceback
        traceback.print_tb(z2ip.error.__traceback__)
        z2ip.error

    that blurb will print the error type and also its traceback. Usually exceptions are thrown
    into Ginnungagap by Trickster God Loki on behalf of the matplotlib event system, so they don't 
    appear immediately.
    """
    def __init__(self,data,cube,ts=None,flat=None,chop=None):
        if type(data) is np.ndarray:
            data = np.ma.array(data)
        if type(cube) is np.ndarray:
            cube = np.ma.array(cube)
        self.data = data
        self.data.fill_value = np.nan
        if ts is not None and len(ts)<cube.shape[2]:
            cube = cube[:,:,:len(ts)]
        self.cube = cube
        if self.cube is not None:
            self.cube.fill_value= np.nan
            # Happens when using the iv plotters
        if ts is not None and len(ts) > cube.shape[2]:
            ts = ts[:cube.shape[2]]
        self.ts = ts
        self.flat = flat
        self.heightratio = [2, 1]
        self.figsize = (10, 10)
        self.markersize = 127
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.debug = False  # if true, prints status messages directly onto the plot.
        self.badpx = True  # whether to show bad pixels.
        self.badpx_corr_thresh = 0.5
        self.chop = chop
        self.click_loc = (10,0,400)
        self.linewidth=1
        self.bottom_plot_fmt = '-'

    def interactive_plot(self):

        # Initialize subplots
        if self.fig is None:
            self.fig, (self.ax, self.ax2) = plt.subplots(2, 1,
                                                         gridspec_kw={'height_ratios': self.heightratio},
                                                         figsize=self.figsize)
        # Perform initial draw of top plot (array map)
        self.redraw_top_plot()

        # "Intelligently" optimize the layout
        if not plt.rcParams['figure.constrained_layout.use']:
            self.fig.tight_layout()

        # Set up the text messaging system
        if self.debug:
            self.text = self.ax.text(0, 5, "Information appears here", va="bottom", ha="left")
        # Crude messaging system. Like exceptions, print statements are piped to /dev/null by Linus Torvalds
        # for some reason when inside a callback for matplotlib.

        # Make a fake time series if needed
        if self.ts is None and self.cube is not None:
            # with iv plotters cube is none and this errors
            self.ts = np.arange(self.cube.shape[2])

        # Set up keybinds
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)

    def redraw_top_plot(self):
        # Draw the array in the top panel, and store some objects
        # in member variables
        self.scatter, self.cb, _ = plot_array(self.data,
                                              ax=self.ax,
                                              s=self.markersize,
                                              bad_px=self.badpx) 

    def static_plot(self, px_for_btm_plot=None, btm_plot_type=None):
        # Use this in case you want to generate a plot
        # showing all the same data but without interactivity
        
        # btm plot type can be a ClickType enum

        self.fig, (self.ax, self.ax2) = plt.subplots(2, 1,
                                                     gridspec_kw={'height_ratios': self.heightratio},
                                                     figsize=self.figsize)
        self.redraw_top_plot()

        self.fig.tight_layout()

        if self.ts is None:
            self.ts = np.arange(self.cube.shape[2])

        if px_for_btm_plot is not None:
            spectral, spatial, array = px_for_btm_plot
            data_to_plot = self.cube[am.phys_to_mce(*px_for_btm_plot)]
            data_to_plot = data_to_plot - ma.min(data_to_plot)
            self.ax2.plot(self.ts, data_to_plot, label=f"data({spectral},{spatial})")

            if btm_plot_type == ClickType.TS_FLAT_ONLY and self.flat is not None:
                flat_to_plot = self.flat[am.phys_to_mce(*px_for_btm_plot)]
                flat_to_plot = flat_to_plot - ma.min(flat_to_plot)

                if len(flat_to_plot) > len(self.ts):
                    ts_to_plot = self.ts
                    flat_to_plot = flat_to_plot[0:len(self.ts)]
                elif len(flat_to_plot) < len(self.ts):
                    ts_to_plot = self.ts[0:len(flat_to_plot)]
                else:
                    ts_to_plot = self.ts

                self.ax2.plot(ts_to_plot, flat_to_plot, label=f"flat({spectral},{spatial}")
            self.ax2.legend()

    def run_signal_correlator(self):
        #Runs the function big_signal_bad_px_finder(chop,cube,corrthresh=0.5) on the data in the object
        #Requires self.chop to be set to the chop signal
        mask_append = zt.big_signal_bad_px_finder(self.chop,self.cube,corrthresh=self.badpx_corr_thresh)
        orig_mask = self.data.mask 
        new_mask = np.logical_or(mask_append,orig_mask)
        #print(new_mask)
        self.data.mask = new_mask

    def onclick(self, event):
        # tx = 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata)
        # `event.xdata // 1` and `event.ydata // 1` will give you the pixel location that was clicked on. 
        # text.set_text(tx)
        
        # modifies time series plot (ax2)
        # button 1 clears plot and plots time series.
        # button 3 is sadly both the middle mouse and right click, it clears, plots time series and flat series
        # button 2 can be supplied by fakeevent, and adds a time series without clearing
        # button 4 supplied by fakeevent; adds time series and flat without clearing
        button = event.button
        if event.inaxes is not self.ax:
            return

        try:
            spectral, spatial, array = get_physical_location(event.xdata, 
                                                             event.ydata)
            if button == 1 or button == 3:
                self.ax2.clear()
                if self.debug: 
                    self.text.set_text(f"clicked{event.button}")
                # These are the only times an actual click occured
            self.click_loc = (spectral, spatial, array)
            self.bottom_plot()
            
            # Plot the flat too if it was given.
            if self.flat is not None and (button == 3 or button == 4):
                self.bottom_flat()

            self.ax2.legend()

            #self.fig.canvas.draw_idle()
        except Exception as e:
            self.error = e
            # raise

    def bottom_plot(self):
        data_to_plot = self.cube[am.phys_to_mce(*self.click_loc)]
        data_to_plot = data_to_plot - np.ma.min(data_to_plot)
        self.ax2.plot(self.ts, 
                      data_to_plot, 
                      self.bottom_plot_fmt,
                      label=f"data({self.click_loc[0]},{self.click_loc[1]})",
                      linewidth=self.linewidth,
                      )

    def bottom_flat(self):
        flat_to_plot = self.flat[am.phys_to_mce(*self.click_loc)]
        flat_to_plot = flat_to_plot - ma.min(flat_to_plot)

        if len(flat_to_plot) > len(self.ts):
            ts_to_plot = self.ts
            flat_to_plot = flat_to_plot[0:len(self.ts)]
        elif len(flat_to_plot) < len(self.ts):
            ts_to_plot = self.ts[0:len(flat_to_plot)]
        else:
            ts_to_plot = self.ts

        self.ax2.plot(ts_to_plot, flat_to_plot, label=f"flat({self.click_loc[0]},{self.click_loc[1]})")

        #fig.tight_layout()
    def onkey(self, event):
        try:
            if event.key == 'c':
                spectral, spatial, array = get_physical_location(event.xdata, event.ydata)
                if self.data[am.phys_to_mce(spectral, spatial, array)] is ma.masked:
                    self.data[am.phys_to_mce(spectral, spatial, array)] = self.data.data[am.phys_to_mce(spectral, spatial, array)]
                else:
                    self.data[am.phys_to_mce(spectral, spatial, array)] = ma.masked
                    #this if/else is so ugly, but I don't know how to do it better
                    # Could just do .mask = !.mask? test it out later
                self.cb.remove() #Looks like clearing the axis might prevent the colorbar from removing correctly
                self.ax.clear()
                self.redraw_top_plot()
                if self.debug: 
                    self.text = self.ax.text(0, 5, f"min:{np.nanmin(self.data.filled())},max:{np.nanmax(self.data.filled())}", va="bottom", ha="left")
                #Here we need to remake the text object because it was destroyed 
                #when we cleared ax.
            elif event.key == 'a':
                self.onclick(FakeEvent(2, event.xdata, event.ydata))
            elif event.key == 'f':
                self.onclick(FakeEvent(4, event.xdata, event.ydata))
            if self.debug: 
                self.text.set_text(f"Pressed {event.key}")
        except Exception as e:
            self.error = e
            #raise

    def check_for_errors(self):
        """ Untested. Prints out the most recent exception thrown in the interactive
        plot routine, since those get lost in the void otherwise. """
        try:
            err = self.error
            print(err)
            traceback.print_tb(err.__traceback__)
        except AttributeError:
            print("No errors")
            # except this attribute errror
            # that we used
            # to signify
            # that there
            # were no
            # errors
            return


class MultiInteractivePlotter(ZeusInteractivePlotter):
    def __init__(self, multi_data, multi_cube, ts=None, flat=None, chop=None):
        """ All the 'multi' arguments should be lists. This lets you
        not have to worry if one or more of your data files had different
        lengths or whatever"""
        self.multi_data = multi_data
        self.multi_cube = multi_cube
        self.multi_ts = ts
        self.multi_flat = flat 
        self.multi_chop = chop
        self.ZeusInteractivePlotter.__init__(self, multi_data[0], self.multi_cube[0])

    def set_index(self, i):
        if i < 0:
            i = len(self.multi_data)-1
        if i >= len(self.multi_data):
            i = 0

        self.multi_index = i

        self.data = self.multi_data[i]
        self.data.fill_value = np.nan
        self.cube = self.multi_cube[i]
        self.ts = self.multi_ts[i]
        self.flat = self.multi_flat[i]

    def onkey(self, event):
        try:
            if event.key == 'left':
                self.set_index(self.multi_index - 1)
            elif event.key == 'right':
                self.set_index(self.multi_index+1)
            else:
                ZeusInteractivePlotter.onkey(self, event)
        except Exception as e:
            self.error = e


class ClickType(Enum):
    TS_ONLY = 1
    TS_FLAT_ONLY = 3
    TS_ADD = 2
    TS_FLAT_ADD = 4


def spectrum_atm_plotter(velocity, 
                         spectrum, 
                         errbars, 
                         title, 
                         atm_helper, 
                         pwv, 
                         y_scaling=1e-18,
                         bounds=None):
    """
    Makes a 2-panel plot with atmospheric transmission on the bottom and the observed spectrum on top
    """
    fig = plt.figure(figsize=(10, 8))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3.5, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    axs = (ax1, ax2)
    number_appeared_on_plot = y_scaling
    line = axs[0].step(velocity, spectrum/number_appeared_on_plot, where='mid')
    axs[0].set_title(title)
    axs[0].set_xlim(min(velocity)-200, max(velocity)+200)
    lncolor = line[0].get_c()
    axs[0].set_ylabel(f"Flux Density, 10$^{{{np.log10(y_scaling):.0f}}}$ W m$^{{-2}}$ bin$^{{-1}}$\n(preliminary calibration)")
    axs[1].set_xlabel("Velocity offset, km s$^{-1}$")
    axs[0].errorbar(velocity, spectrum/number_appeared_on_plot, errbars/number_appeared_on_plot, fmt='none', ecolor=lncolor)
    axs[1].set_ylabel("Atmospheric\nTransmission")
    atm_velocity = np.linspace(min(velocity)-200, max(velocity)+200, 100)
    atm_ghz_deltav = atm_helper.observing_freq*units.GHz*(1-atm_velocity*units.km/units.s/const.c)
    atm_ghz = atm_ghz_deltav.to("GHz").value
    atm_trans = atm_helper.interp(atm_ghz, pwv)
    axs[1].plot(atm_velocity, atm_trans)
    plt.setp(ax1.get_xticklabels(), visible=False)
    axs[0].plot([min(velocity)-200, max(velocity)+200], [0, 0], 'k', linewidth=0.5)
    print(bounds)
    if bounds is not None and None not in bounds:
        axs[0].set_xlim(bounds[0:2])
        axs[0].set_ylim(bounds[2:4])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
