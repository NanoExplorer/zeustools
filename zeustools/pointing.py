import numpy as np
from matplotlib import pyplot as plt
import zeustools as zt
import scipy.optimize as opt

import traceback


def raster_calc(arr, stepsize, initial_guess=(1000, 0, 0, 8, 0), do_plot=True, bounds=None):
    """Calculate the pointing offset to apply to center the source.
    This does a nicer job plotting the raster than the other functions.

    Parameters:
    arr: a numpy array containing the values of each position
    stepsize: arcseconds offset between raster positions
    initial_guess: the initial guess for the 2d gaussian fit.
        It's a tuple of (amplitude, x position, y position, sigma, baseline)
    do_plot: whether or not to make a plot of the raster and gaussian fit. (default: True)
    bounds: same as initial guess, except 2 tuples for max and min bounds for each parameters

    Returns:
    best-fit paramteters

    Side-Effects:
    prints fit data and the pcorr function you should run at APEX
    makes a plot if do_plot is True
    """
    # Determine number of beams on a side
    num_steps=arr.shape[0]
        
    # Determine distance from left beam to right beam in arcsec
    boxsize = (num_steps-1)*stepsize

    # This is the distance from the center to the left, right, etc. edge
    boxbound = boxsize/2

    # Create default bounding box (I felt like this was too complicated to put directly into the call signature)
    if bounds is None:
        bounds= ((-np.inf,-boxbound*1.5,-boxbound*1.5,stepsize,-np.inf),
                 (np.inf,boxbound*1.5,boxbound*1.5,boxsize*1.5,np.inf))

    # Need a 1d array for fitting...
    data = arr.ravel()

    # generate a grid of coordinates in arcset offset to feed to the fitting routine
    x = np.linspace(-boxbound,boxbound,num_steps)
    y = np.linspace(-boxbound,boxbound,num_steps)
    x, y = np.meshgrid(x, y)

    try:
        #Run the fitting routine!
        popt, pcov = opt.curve_fit(zt.twoD_Gaussian, 
                                   (x, y), 
                                   data, 
                                   p0=initial_guess,
                                   bounds=bounds)
        #Use the result to print the offset command needed for input into APECS
        print(f"apply pcorr {popt[1]:.1f}, {popt[2]:.1f}")

        #Everyone wants to see the FWHM not the gaussian sigma, so do that conversion
        fwhm = 2*np.sqrt(2*np.log(2))*popt[3]
        print(f"source amplitude: {popt[0]:.0f}, source fwhm: {fwhm:.2f}")
        print(f"baseline: {popt[4]:.0f}")
    except RuntimeError:
        #Whoops, the fit routine failed. 
        #ones we want to worry about!!
        print("Could not find source. Consider passing in your own initial_guess")
        print("initial_guess=(amplitude, az offset, alt offset, size, baseline)")
        popt=(1,0,0,1,0)
        traceback.print_exc()

    if do_plot:
        #Create a new grid for the plotting routine
        #This goes to the EDGES of each square instead of the centers.
        box = num_steps*stepsize/2
        fx = np.linspace(-box, box, 50)
        fy = np.linspace(-box, box, 50)
        fx, fy = np.meshgrid(fx, fy)

        data_fitted = zt.twoD_Gaussian((fx, fy), *popt)
        fig, ax = plt.subplots(1, 1)
        #pxcorr = stepsize/2
        ax.imshow(arr, origin='bottom',
                  extent=(fx.min(), fx.max(), fy.min(), fy.max()))
        ax.contour(fx, fy, data_fitted.reshape(50, 50), 4, colors='w')
        plt.show()

    return popt


def raster_load(filestem,firstnum,sidesize,pixels, existing_data=[]):
    """Load in and plot a pointing raster scan.

    Parameters:
    filestem: All of the filename except the number. 
        e.g.: /data/cryo/current_data/saturn_191202_
    firstnum: The first number that was part of the raster
    sidesize: The number of raster pointings on a side
        e.g.: For a 3x3 raster would be 3.
    pixels: array of (mce_row,mce_col) for pixels to sum to make the raster plot
        tip: use ArrayMapper.phys_to_mce(spec,spat) to get this number easily
    do_plot: Set this to False if you want to just load in the data and not make a plot
    existing_data: Used by the Caching Raster to speed up the loading process

    Returns: 
    np array of amplitudes of signal at each pointing position
    
    Side-Effects:
    makes a plot if do_plot = True
    modifies the existing_data array
    """
    altaz = np.zeros((sidesize,sidesize))
    alt=0
    az=0
    going=+1
    pixels=np.array(pixels)
    for filenumber in range(firstnum,firstnum+sidesize**2):
        # make the file name formatted properly
        filenum=f"{filenumber:04d}"
        totalsig=0
        # if there are existing data, use them instead:
        # TODO: could do snake subtraction and other post-processing
        if filenumber-firstnum<len(existing_data):
            chop_on,ts_on,chop_off,ts_off=existing_data[filenumber-firstnum]
            for row,col in pixels:
                sig=np.median(chop_on[row,col])-np.median(chop_off[row,col])
                totalsig+=sig
        else:
            try:
                chop_on,ts_on,chop_off,ts_off=zt.processChop(f"{filestem}{filenum}")
                existing_data.append((chop_on,ts_on,chop_off,ts_off))
                for row,col in pixels:
                    sig=np.median(chop_on[row,col])-np.median(chop_off[row,col])
                    totalsig+=sig
            except FileNotFoundError:
                pass

        #Each row of the raster goes in a different direction.
        #making a zigzag pattern:
        # ----->
        # ^
        # <-----
        #      ^
        # ----->
        altaz[alt,az]=totalsig
        if az==sidesize-1 and going==1:
            alt+=1
            going = -going
        elif az==0 and going==-1:
            alt+=1
            going = -going
        else:
            az+= going
    return(altaz)


class CachingRaster:
    """Reduces latency on the raster_plot function by only loading in each mce datafile once"""

    def __init__(self):
        self.known_rasters=[]
        self.cached_data=[]
    
    def go(self,filestem,start_filename,num_beams,pixels):
        """Load in pointing raster scan. NOW WITH CACHING! FOR ALL YOUR LOW-LATENCY PLOTTING NEEDS
        This is a wrapper around raster_load, so has the same paramters except existing_data

        Parameters:
        filestem: The folder that contains date folders. 
            e.g.: /data/cryo/current_data/
        start_filename: The first number that was part of the raster
        num_beams: The number of raster pointings on a side
            e.g.: For a 3x3 raster would be 3.
        pixels: array of (mce_row,mce_col) for pixels to sum to make the raster plot
            tip: use ArrayMapper.phys_to_mce(spec,spat) to get this number easily
        do_plot: Set this to False if you want to just load in the data and not make a plot
        existing_data: Used by the Caching Raster to speed up the loading process

        Returns: 
        np array of amplitudes of signal at each pointing position
        """

        #make a unique name for the scan
        #TODO: why am I not using a dictionary?
        raster_name=(filestem,start_filename,num_beams)
        if raster_name in self.known_rasters:
            i = self.known_rasters.index(raster_name)
            dat=raster_load(*raster_name,
                            pixels,
                            existing_data=self.cached_data[i])
        else:
            self.known_rasters.append(raster_name)
            new_cached_data=[]
            dat=raster_load(*raster_name,
                            pixels,
                            existing_data=new_cached_data)
            self.cached_data.append(new_cached_data)          
        return dat 


def pfRaster(filestem,firstnum,sidesize):
    """ Uses the .pf files generated by the acquisition software to make a raster
    
    Parameters:
    filestem: The folder that contains date folders. 
        e.g.: /data/cryo/current_data/
    firstnum: The first number that was part of the raster
    sidesize: The number of raster pointings on a side
        e.g.: For a 3x3 raster would be 3.

    Returns: 
    np array of amplitudes of signal at each pointing position
    """
    altaz = np.zeros((sidesize,sidesize))
    alt=0
    az=0
    going=+1
    for filenumber in range(firstnum,firstnum+sidesize**2):
        filenum=f"{filenumber:04d}"
        good_data=zt.readPf(f"{filestem}{filenum}.pf")
        #print(f"{np.mean(good_data):.1f}")
        altaz[alt,az]=np.mean(good_data)
        if az==sidesize-1 and going==1:
            alt+=1
            going = -going
        elif az==0 and going==-1:
            alt+=1
            going = -going
        else:
            az+= going
    return(altaz)


def make_spectral_pointing_plots(date,fstem,fnums,specs,spats,array='a'):
    am=zt.ArrayMapper()
    for i in fnums:
        fname=zt.makeFileName(date, fstem, i)
        values3=zt.processChopBetter(fname)

        for spat_pos in spats:
            pixels=[(x, spat_pos, array) for x in specs]
            for p in pixels:
                mce_loc2=am.phys_to_mce(*p)
                pxwant=values3[mce_loc2]
                plt.plot(pxwant[:-1:2]-pxwant[1::2], label=f"spec {p[0]}")
            plt.legend()
            plt.title(f"spatial position {spat_pos} of file {fname}")
            plt.show()
            #plt.close()

# stuff


def make_added_pointing_plots(date,fstem,fnums,specs,spats):
    am=zt.ArrayMapper()
    for i in fnums:
        fname = zt.makeFileName(date,fstem,i)
        values3 = zt.processChopBetter(fname)

        for spat_pos in spats:
            pixels = [(x,spat_pos,'a') for x in specs]
            addedsig = np.zeros(values3.shape[2]//2)
            for p in pixels:
                mce_loc2=am.phys_to_mce(*p)
                pxwant=values3[mce_loc2]
                addedsig+=pxwant[:-1:2]-pxwant[1::2]
            plt.plot(addedsig,label=f"spec {p[0]}")
            plt.legend()
            plt.title(f"spatial position {spat_pos} of file {fname}")
            plt.show()
            #plt.close()