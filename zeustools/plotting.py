import numpy as np
from matplotlib import pyplot as plt
import zeustools as zt
from numpy import ma

am = zt.ArrayMapper()  # YES I JUST MADE A GLOBAL VARIABLE. DO I REGRET IT? NO. WILL IT HURT? ABSOLUTELY.
#When you import plotting.py you'd better have a 'config' directory in your working directory or else.
# TODO: fix this


def plot_array(somedata,ax=None,s=60):
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
    mce_grid = am.grid_map()
    if ax is None:
        #we need an axis to plot on, so if one wasn't supplied, make it.
        g=plt.figure(dpi=300)
        g.gca().set_aspect(1)
        ax=g.gca()
    else:
        #we always want our pixels to be square.
        ax.set_aspect(1)
        
    #This makes a scatter plot. A square shaped marker is placed at every spectral and spatial
    #position, colored according to the flux value of that pixel.
    stuff= ax.scatter(mce_grid[:,:,1],-mce_grid[:,:,0],c=somedata,s=s,marker='s')
    cb = plt.colorbar(stuff,orientation='horizontal',ax=ax,aspect=50,fraction=0.05,shrink=0.9,pad=0.07)
    #The sizing of the colorbar is really fineagely. Might need to work on that TODO: sometime.
    return stuff,cb,ax


class FakeEvent:
    """ Used internally in case you want to manually trigger one of the event handlers"""
    def __init__(self,key,xdata,ydata):
        self.key=key
        self.xdata=xdata
        self.ydata=ydata
        self.button=key
        

def get_physical_location(xdata,ydata):
    """ Used internally. The center of each pixel is at integer values, so the pixel extends
    from value - 0.5 to value+0.5. Here we correct for that and also tell you which array 
    your pixel is on. Usable with the plot_array and ZeusInteractivePlotter."""

    xclk = (xdata+0.5)//1
    yclk = (ydata+0.5)//1

    if yclk <= -10:
        #this is the 400 micron array
        spatial= -yclk-10
        spectral = xclk
        array = 400
    elif xclk < 12:
        #this is the 600 micron array
        spatial = -yclk
        spectral = xclk
        array = 600
    else:
        spatial = -yclk
        spectral = xclk-20
        array = 200
    return spectral,spatial,array


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
        3. Middle click (or if that doesn't work press 'a' while hovering) to add a time series without clearing
        the previous one
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
    def __init__(self,data,cube,ts=None,flat=None):
        self.data=data
        self.data.fill_value=np.nan
        self.cube=cube
        self.ts=ts
        self.flat=flat
        self.heightratio=[2,1]
        self.figsize = (10,10)
        self.markersize = 127
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.debug = False  # if true, prints status messages directly onto the plot.
        
    def interactive_plot(self):
        self.fig,(self.ax,self.ax2) = plt.subplots(2,1,
                                                   gridspec_kw={'height_ratios':self.heightratio},
                                                   figsize=self.figsize)
        self.scatter,self.cb,_=plot_array(self.data.filled(),ax=self.ax,s=self.markersize) 
        # _ above would be === self.ax already
        
        self.fig.tight_layout()

        if self.debug: self.text=self.ax.text(0,5, "Information appears here", va="bottom", ha="left")
        #Crude messaging system. Like exceptions, print statements are piped to /dev/null by Linus Torvalds
        # for some reason when inside a callback for matplotlib.

        if self.ts is None:
            self.ts = np.arange(self.cube.shape[2])
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)

    def onclick(self,event):
        # tx = 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata)
        # `event.xdata // 1` and `event.ydata // 1` will give you the pixel location that was clicked on. 
        # text.set_text(tx)
        
        #modifies time series plot (ax2)
        #button 1 clears plot and plots time series.
        #button 3 is sadly both the middle mouse and right click, it clears, plots time series and flat series
        # button 2 can be supplied by fakeevent, and adds a time series without clearing
        # button 4 supplied by fakeevent; adds time series and flat without clearing
        button=event.button

        try:
            if button==1 or button==3:
                self.ax2.clear()
                if self.debug: self.text.set_text(f"clicked{event.button}")
                    # These are the only times an actual click occured

            spectral,spatial,array = get_physical_location(event.xdata,event.ydata)

            data_to_plot = self.cube[am.phys_to_mce(spectral,spatial,array)]
            data_to_plot = data_to_plot - min(data_to_plot)
            self.ax2.plot(self.ts,data_to_plot,label=f"data({spectral},{spatial})")

            #Plot the flat too if it was given.
            if self.flat is not None and (button==3 or button==4):
                flat_to_plot = self.flat[am.phys_to_mce(spectral,spatial,array)]
                flat_to_plot = flat_to_plot - min(flat_to_plot)

                if len(flat_to_plot) > len(self.ts):
                    ts_to_plot = self.ts
                    flat_to_plot = flat_to_plot[0:len(self.ts)]
                elif len(flat_to_plot) < len(self.ts):
                    ts_to_plot = self.ts[0:len(flat_to_plot)]

                self.ax2.plot(ts_to_plot,flat_to_plot,label=f"flat({spectral},{spatial}")
            self.ax2.legend()
        except Exception as e:
            self.error = e
            #raise

        #fig.tight_layout()
    def onkey(self,event):
        try:
            if event.key=='c':
                spectral,spatial,array = get_physical_location(event.xdata,event.ydata)
                if self.data[am.phys_to_mce(spectral,spatial,array)] is ma.masked:
                    self.data[am.phys_to_mce(spectral,spatial,array)] = self.data.data[am.phys_to_mce(spectral,spatial,array)]
                else:
                    self.data[am.phys_to_mce(spectral,spatial,array)] = ma.masked
                    #this if/else is so ugly, but I don't know how to do it better
                self.ax.clear()
                self.cb.remove()
                self.scatter,self.cb,_=plot_array(self.data.filled(),ax=self.ax,s=127)
                if self.debug: self.text = self.ax.text(0,5, f"min:{np.nanmin(self.data.filled())},max:{np.nanmax(self.data.filled())}", va="bottom", ha="left")
            elif event.key == 'a':
                self.onclick(FakeEvent(2,event.xdata,event.ydata))
            elif event.key=='f':
                self.onclick(FakeEvent(4,event.xdata,event.ydata))
            if self.debug: self.text.set_text(f"Pressed {event.key}")
        except Exception as e:
            self.error = e
            #raise


# Here, have some dead code:

# def plotArray(flatfilename,datafilename):
    
#     chop,ts,cube=zt.load_data_raw(datafilename)
#     flatchop,flatts,flatcube = zt.load_data_raw(flatfilename)
#     nicets = ts-ts[0]
#     ndcube = zt.nd_reject_outliers(cube)
#     d=naive_data_reduction(chop,cube)/naive_data_reduction(flatchop,flatcube)
#     spatial_offset_400 = np.zeros_like(am.arrays['a'])
#     spatial_offset_400[:,0] = 10

#     spectral_offset_200 = np.zeros_like(am.arrays['b'])
#     spectral_offset_200[:,1] = 20

#     amap = am.arrays['a'] + spatial_offset_400
#     bmap = am.arrays['b'] + spectral_offset_200
#     cmap = am.arrays['c']
#     full_map = np.concatenate((amap,bmap,cmap))
#     mce_grid = np.zeros((33,24,2),dtype=int)
#     mce_grid[full_map[:,2],full_map[:,3]] = full_map[:,0:2]
#     nonexistant_pixels = np.logical_and(mce_grid[:,:,1]==0, mce_grid[:,:,0]==0)
#     # ^ lists spectral/spactial coordinate pairs that do not contain pixels
    
#     cleaned_data=zt.nd_reject_outliers(d.ravel(),axis=0,MAD_chop=200).reshape(33,24)
#     # ^ attempts naive filtering of the signal
    
#     cleaned_data.mask = np.logical_or(cleaned_data.mask,nonexistant_pixels)
#     maxpts = ma.array(np.max(ndcube,axis=2),mask=cleaned_data.mask)
#     minpts = ma.array(np.min(ndcube,axis=2),mask=cleaned_data.mask)
#     stdpts = ma.array(np.std(ndcube,axis=2),mask=cleaned_data.mask)
#     g=figure(dpi=300)
#     g.gca().set_aspect(1)
#     scatter(mce_grid[:,:,1],-mce_grid[:,:,0],c=np.log(cleaned_data),s=50,marker='s')
#     colorbar(orientation='horizontal')
# old code? May have been incorporated into am.grid_map() NOPE HAHA
