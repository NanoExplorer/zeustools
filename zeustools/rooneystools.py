import argparse
import numpy as np
from numpy import ma
from scipy import optimize
from zeustools.codystools import readChopFile
from zeustools import mce_data
from matplotlib import pyplot as plt
from zeustools import data
import importlib.resources as res


def gaussian(x, sigma, mu, a):
    """I never understood why numpy or scipy don't have their own gaussian function.
    """
    return a*np.exp(-1/2*((x-mu)/(sigma))**2)
    # e^-a(x+b)^2 -> e^(1/2((x-mu)/sigma)**2) -> e^( 1/(2*sigma^2) (x-mu)^2) their a = my 1/(2sigma^2)


def gaussian_integral(sigma, mu, a):
    """Same parameter order as my "gaussian" function, but no x input. Returns integral from -infinity to infinity of gaussian(x,sigma,mu,a) dx
    """
    return a*np.sqrt(np.pi*2)*sigma


# From https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
def twoD_Gaussian(pos, amplitude, xo, yo, sigma_x, offset):
    """Good for doing pointing. Doesn't have too many parameters,
    for example we set sigma_x = sigma_y and we don't have a theta.
    """
    sigma_y = sigma_x
    theta = 0
    x, y = pos
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) +
                                  c*((y-yo)**2)))
    return g.ravel()


class chi_sq_solver:
    """A chi squared analysis tool I built. I don't know 
    whether it's still useful. Kept around just in case.
    It might also assume your data is Poisson...
    Or I've just forgotten how chi squared works?
    """
    def __init__(self,
                 bins,
                 ys,
                 function,
                 guesses,
                 bounds=None
                 ):
        # Initialize object members
        self.bins = bins
        self.ys = ys
        self.centerbins = self.centers(self.bins)
        self.function = function
        # Calculate minimum chi squared fit
        self.result = optimize.minimize(self.chi_sq, guesses, bounds=bounds)
        
    def chi_sq(self, args):
        # Return the chi_squared statistic for the binned data and the arguments for the function
        sum = 0
        for i in range(len(self.ys)-1):
            try:
                E = self.function(self.centerbins[i], *args)
                if self.ys[i] != 0:
                    sum += (self.ys[i]-E)**2/self.ys[i]
            except FloatingPointError:
                print("Oh no! There was a floating point error.")
                # print(self.centerbins[i],*args)
                # exit()
            # print(self.ys[i],E,sum)
        # print(args,sum)
        return sum

    def centers(self, bins):
        # Calculate the centers of the bins, or the average x value of each bin.
        # Would be mathematically cool if we could average the x values of all the data points,
        # but that probably shouldn't affect anything substantially.
        centers = []
        for i in range(len(bins)-1):
            centers.append((bins[i+1]-bins[i])/2+bins[i])
            # Take the average value of two bins and add it to the lower bin value
        return centers


def nd_mad(nparray, axis, extrainfo=False):
    """This is the backend for :func:`nd_mads_from_median`. 
    Given a numpy ndarray, we calculate the median absolute deviation

    NOTE! in scipy version 1.3 they added a median absolute deviation method!
    you might consider using that instead in some cases.

    However this function gives you the option of returning the distance
    from the median for every point on the array.
    """
    hasnan = np.isnan(nparray)
    if np.any(hasnan):
        nparray = ma.array(nparray, mask=hasnan)
    med = ma.median(nparray, axis=axis)
    expand_slicing = tuple((np.newaxis if x==axis else slice(None) for x in range(len(nparray.shape))))  # noqa: E225
    dist_from_med = ma.abs(nparray-med[expand_slicing])
    mdev = ma.median(dist_from_med, axis=axis)
    if extrainfo:
        return mdev, dist_from_med
    else:
        return mdev


def nd_mads_from_median(nparray, axis, zero_padding_factor=0.0001):
    """Returns the number of mads from the median for each data point, with median and mad computed over the specified axis

    """
    # slice(None) is equivalent to ':'
    expand_slicing = tuple((np.newaxis if x==axis else slice(None) for x in range(len(nparray.shape))))  # noqa: E225
    mdev, dist_from_med = nd_mad(nparray, axis, extrainfo=True)
    # print(type(mdev))
    if type(mdev) is np.ndarray or type(mdev) is ma.core.MaskedArray:
        mdev[mdev==0] = zero_padding_factor  # noqa: E225
        mdev = mdev[expand_slicing]
    else:
        if mdev == 0:
            mdev = zero_padding_factor

    s = dist_from_med/mdev
    return s


def nd_reject_outliers(nparray, MAD_chop=5, axis=2):
    """Returns a masked array masking outliers on a certain axis.

    The input array is examined using the median absolute deviation method
    and outliers greater than the MAD_chop number are masked. 
    """
    s = nd_mads_from_median(nparray, axis)
    return ma.masked_array(nparray, mask=s > MAD_chop)


def parseCmdArgs(argumentList, helpList, typeList):
    """Parses command-line arguments.

    arguments:
    3 lists of argument data. First is the list of arguments. Each
    argument is a list that will be passed to the add_argument function of argparse
    Each element in the second list is the help string for that argument,
    The last list contains the type.

    Usage::

        common.parseCmdArgs([['settings'],
                             ['-v','--verbose']],
                            ['Path to settings file','flag to print extra info'],
                            [str,'bool'])
    """
    parser = argparse.ArgumentParser()
    for argument, theHelp, theType in zip(argumentList, helpList, typeList):
        if theType == 'bool':
            parser.add_argument(*argument, help=theHelp, action='store_true')
        else:
            parser.add_argument(*argument, help=theHelp, type=theType)
    return parser.parse_args()


def nd_nan_outliers(nparray, MAD_chop=5, axis=2):
    """Returns an array with nans instead of outliers on a certain axis

    The input array is examined using the median absolute deviation method
    and outliers greater than the MAD_chop number are changed to np.nan. 
    """
    s = nd_mads_from_median(nparray, axis)
    nparray[s > MAD_chop] = np.nan
    return nparray


def histGaussian(something, nbins=50):
    n, bins, patches = plt.hist(something, nbins)
    guessavg = np.mean(something)
    guessstd = np.std(something)
    guessamp = len(something)/nbins
    robot = chi_sq_solver(bins, n, gaussian, np.array([guessstd, guessavg, guessamp]),
                          bounds=[(1, None), (None, None), (0, None)])
    domain = np.arange(min(bins), max(bins), 100)
    model = gaussian(domain, *(robot.result.x))
    plt.plot(domain, model)
    return robot.result


def readPf(filename):
    """ Loads in a .pf file. Those contain the exact data that was
    sent to the APEX servers for pointing and/or focus calibration.
    We send lots of 1s to APEX because they do on-off/off and we 
    think we know better than them so we do all kinds of acrobatics.

    This script undoes the acrobatics so you see exactly what APEX
    would see.

    :param filename: The filename you want to load (INCLUDE the .pf extension)

    :return: 1-D array of data points (usually a time series)
    :rtype: numpy.array
    """
    data = np.loadtxt(filename,dtype=str)
    phase= data[:,3].astype(int)
    ampl = data[:,4].astype(float)
    good_data = ampl[phase==2] - 1
    return(good_data)


def load_data_raw(filename):
    """ Loads an MCE data file, chop file and TS file, returning
    the raw data for each of them.

    Note that if this turns out to be a "stare" dataset,
    there won't be chop or TS files. So we make some surrogate
    chop and ts data sets. The chop will be all 0s and the ts
    will be a monotonicall increasing array.

    :param filename: the file name that you want to load.

    :return: A tuple containing 3 numpy arrays:

        0. Chop phase (whether we're looking at the source or the sky)
        1. Time stamp
        2. Time series data cube as usual from mce_data (use :class:`.ArrayMapper` to index it)

    :rtype: 3-Tuple of numpy.array
    """
    mcefile = mce_data.SmallMCEFile(filename)
    mcedata = mcefile.Read(row_col=True).data
    chop = readChopFile(filename)[1]
    try:
        ts = np.loadtxt(f"{filename}.ts")

        tstimes = ts[0:len(chop),1]  # This is necessary because sometimes the 
        # ts file has an extra data point for some reason (off by 1 error in 
        # arduino code or timing error?)

    except OSError:
        print("No TS found. Assuming un-chopped data...")
        chop = np.zeros_like(mcedata[0,0])
        tstimes = np.arange(len(mcedata[0,0]))

    return (chop,tstimes,mcedata)


def load_data_handle_outliers(filename,mode="labchop"):
    data = load_data_raw(filename)
    if mode == "sky":
        return (data[0],data[1],nd_reject_outliers(data[2]))
    elif mode == "labchop":
        chop,ts,datacube = data
        datacube = ma.array(datacube)  # assigning ma.masked to a normal array does nothing...
        datacube[:,:,chop==1] = nd_reject_outliers(datacube[:,:,chop==1])
        datacube[:,:,chop==0] = nd_reject_outliers(datacube[:,:,chop==0])
        return (chop,ts,datacube)


def get_ts_time_offset(filename):
    """ Turns out that it's not straightforward what's going on between
    the gps card (and therefore the info in the .ts file) and the time
    on the pc (ie, what you get when you call datetime.now() or CTIME).

    I thought it could be like the TAI/GPS/leap second stuff, but that isn't 
    really it either. (unless the MCE PC clock is set to TAI somehow.)
    
    Anyway, this method returns gps_card_time - pc_time by looking
    at the run file and the ts file. The run file contains the pc_time
    that mce_run was launched, and the ts file contains the gps_card_time
    of the first data point.

    Note, I do not do this here, but you could also check date modified
    for the data file and the run file---the runfile is written before data
    collection is commenced, and obviously the data file has to be modified
    after the last data point is collected. or the last entry in the ts file

    returns t_GPS - t_PC = t_offset
    so 
    t_GPS = t_offset + t_PC (i.e. add return value to PC time to convert so 
    it matches GPS time)
    """

    mcefile = mce_data.SmallMCEFile(filename)
    pc_time = int(mcefile.runfile.data['FRAMEACQ']['CTIME'].strip())
    gps_times = np.loadtxt(f"{filename}.ts")
    return gps_times[0][1]-pc_time


def processChop(filename):
    """ Loads in MCE data file, chop file, and TS file.

    :return:  a tuple containing:

        0. data points where chop is on
        1. ts for chop on
        2. data for chop off
        3. ts for chop off

    """
    chopchop, tstimes, chopdata = load_data_raw(filename)
    chop_on = chopdata[:, :, chopchop == 1]
    chop_off = chopdata[:, :, chopchop == 0]
    ts_on = tstimes[chopchop == 1]
    ts_off = tstimes[chopchop == 0]
    return(chop_on, ts_on, chop_off, ts_off)


def array_name_2(name):
    """"A rose by any other name would smell as sweet"

    Converts various names for the 3 different physical arrays into the 
    internal scheme used by this class (which is the same as the letter used by 
    ``arrayX_map.dat``, much to Thomas's dismay. At least this is an easy way to convert
    to that scheme.)

    That scheme is follows:

    * array "A" is the 400 um array (350/450)
    * array "B" is the 200 um array
    * array "C" is the 600 um array. 
    
    simply pass any name (e.g. ``"400"``, ``400``, ``350``, ``"200"``) and 
    this will return the correct letter to use in ``arrayX_map.dat`` 

    :param name: Human readable name for the array. Should be a string or number.
    :return: one of 'a', 'b', or 'c'. a1 means 350 side, a2 means 450 side, a means generically
        the 450 array, b is the 200 array and c is the 600 array.
    """
    name = str(name).upper()
    if name == '400' or name == 'A':
        name = 'a'
    elif name == '350':
        name = 'a1'
    elif name == '450':
        name = 'a2'
    elif name == '200' or name == 'B':
        name = 'b'
    elif name == '600' or name == 'C':
        name = 'c'
    elif name == "ALL":
        name = "*"  # I have no idea whether this is a good idea
    else:
        raise ValueError("invalid array name")
    return name


def array_name(name):
    """"A rose by any other name would smell as sweet"

    Converts various names for the 3 different physical arrays into the 
    internal scheme used by this class (which is the same as the letter used by 
    ``arrayX_map.dat``, much to Thomas's dismay. At least this is an easy way to convert
    to that scheme.)

    That scheme is follows:

    * array "A" is the 400 um array (350/450)
    * array "B" is the 200 um array
    * array "C" is the 600 um array. 
    
    simply pass any name (e.g. ``"400"``, ``400``, ``350``, ``"200"``) and 
    this will return the correct letter to use in ``arrayX_map.dat`` 

    :param name: Human readable name for the array. Should be a string or number.
    :return: one of 'a', 'b', or 'c'.
    :raises: ValueError if you pass 'all' in because this is supposed to 
             just deal with file names. 
    """
    arr = array_name_2(name)
    if arr=='*':
        raise ValueError("array 'all' not allowed: use array_name_2 function instead")
    return arr[0]


class ArrayMapper:
    """ This class makes it easy to address mce data.

    MCE data is stored in "Row,Col" format, where the rows and columns correspond to how
    the pixels are wired to the electronics. This format is useful when debugging electronics
    issues but not so much for viewing data. So, to index raw MCE data you usually call ::

        mcefile = mce_data.SmallMCEFile(filename)
        mcedata = mcefile.Read(row_col=True).data

    Then, ``mcedata`` will be a 3-dimensional datacube. You index this cube with ``mcedata[row, col, time]``

    The physical layout of the detector has a spatial direction and a spectral direction. To 
    index a pixel using spatial position and spectral position, you either need to reformat
    the ``mcedata`` object or convert the spatial position and spectral position to a row and column.
    This does the latter.

    The phys_to_mce function in this class will return an index you can use immediately 
    to index an mce_data datacube in order to address a pixel by its physical location.

    :param path: (optional) path to a folder containing the three ``arrayX_map.dat`` files.
        The default array map is the one most recently updated in 2021 with input from Bo, and 
        is now included in the package by default!   
    """
    def __init__(self,path=None):
        if path:
            self.arrays = {'a':np.loadtxt(f"{path}/arrayA_map.dat",usecols=range(0,4),dtype=int),
                           'b':np.loadtxt(f"{path}/arrayB_map.dat",usecols=range(0,4),dtype=int),
                           'c':np.loadtxt(f"{path}/arrayC_map.dat",usecols=range(0,4),dtype=int)}
        else:
            self.arrays={}
            for array in ['a','b','c']:
                with res.open_text(data,f"array{array.upper()}_map.dat") as array_file:
                    self.arrays[array] = np.loadtxt(array_file,usecols=range(0,4),dtype=int)
        #these arrays have the following 4columns:
        # spatial, spectral, mcerow, mcecol
    
    def phys_to_mce(self,spec,spat,array):
        """Given the physical position of a pixel (spectral position, spatial position, array name)
        returns the mce_row and mce_col of that pixel. That can be used to directly address
        the pixel from the mce datacube.

        :param spec: The spectral position of the pixel to index.
        :param spat: The spatial position of the pixel to index.
        :param array: The array the pixel is on. Should be a number or string, like "400" for the 400 micron array.

        :return: Tuple of (mce_row, mce_col) of the pixel.

        :Example: 
            To load a data file and get the time series for spectral pixel 7 on spatial position 1
            of the 400 micron array::

                chop, ts, datacube = load_data_raw(...)
                am = ArrayMapper()
                idx = am.phys_to_mce(7,1,400)
                time_series = datacube[idx]

            It is recommended that you initialize ``am = ArrayMapper()`` once and 
            reuse the ``am`` object over the lifetime of your code.

        TODO: better way to index multiple pixels. 
        """
        
        array_to_use = self.arrays[array_name(array)]
        is_correct_spatialpos = array_to_use[:,0] == spat
        is_correct_spectpos = array_to_use[:,1] == spec
        is_correct_px = np.logical_and(is_correct_spectpos, is_correct_spatialpos)
        px = np.where(is_correct_px)[0][0]
        return(array_to_use[px, 2], array_to_use[px, 3])

    def grid_map(self):
        """Return a grid of spectral positions and spatial positions for each mce_row and mce_column
        
        Note that for ease of use the 400 array has 10 added to all its spatial positions, 
        and the 200 array has 20 added to its spectral positions. If the entries are masked, there
        is nothing wired to that MCE location (usually indicates broken wires that have been routed
        around).

        :return: Numpy array of shape (33,24,2). The first two axes match the MCE; axis 0 selects an MCE 
            row, while axis 1 selects an MCE column. This way if you take 
            ``np.grid_map()[mce_row,mce_col]`` you get back ``[spatial_position,spectral_position]``

        """
        #remember spat spec row col
        spatial_offset_400 = np.zeros_like(self.arrays['a'])
        spatial_offset_400[:,0] = 10

        spectral_offset_200 = np.zeros_like(self.arrays['b'])
        spectral_offset_200[:,1] = 20

        amap = self.arrays['a'] + spatial_offset_400
        bmap = self.arrays['b'] + spectral_offset_200
        cmap = self.arrays['c']
        full_map = np.concatenate((amap,bmap,cmap))
        mce_grid = ma.zeros((33,24,2),dtype=int) 
        mce_grid[:] = ma.masked
        # print(mce_grid)
        mce_grid[full_map[:,2],full_map[:,3]] = full_map[:,0:2]

        return mce_grid


def processChopBetter(fname):
    """Loads in MCE data file and chop file.

    returns an array with the medians of each chop cycle for all pixels. 

    TODO: needs a *better* name   
    """
    index,chop = readChopFile(fname)
    chopstarts = 1+np.where(chop[:-1]!=chop[1:])[0]
    chopstarts = np.append([0],chopstarts)
    chopbounds = np.append(chopstarts,len(chop))
    datafile = mce_data.SmallMCEFile(fname)
    data = datafile.Read(row_col=True).data
    values = []
    for i,irange in enumerate(zip(chopbounds[:-1], chopbounds[1:])):
        value=np.median(data[:, :, irange[0]:irange[1]], axis=2)
        values.append(value)
    values=np.array(values)
    #note that right now index 0 is chop number,
    #index 1 is mce_row
    #index 2 is mce_col
    values=np.moveaxis(values, 0, 2)
    #now index 0,1 is row,col
    #index 2 is chop number
    return values


def makeFileName(date,name,number):
    """ Simple routine to convert a date, filename, and file index into a real
    data file.

    :Example: Say you have a data file in /data/cryo/20191130/saturn_191130_0023.
        Calling::

            makeFileName('20191130','saturn',23)

        will return::

            '20191130/saturn_191130_0023`

        which you can easily append to the parent folder name.

    :param date: String date in ``yyyymmdd`` format
    :param name: source name 
    :param number: integer of file number
    :return: folder and file path relative to the data folder (e.g. /data/cryo)

    TODO: it would be nice if this didn't depend so much on data types and such

    """
    return f"{date}/{name}_{date[2:]}_{number:04d}"


def loadts(filename):
    """ Really simple wrapper function to load a ``.ts`` file.

    Does literally the following::

        times=np.loadtxt(f"{filename}.ts")[:,1]

    :param filename: the location of the data file to read the time stamps for. Don't include the ``.ts`` extension
    :return: 1-D numpy array of timestamps 

    """

    times=np.loadtxt(f"{filename}.ts")[:,1]
    return times


#https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array#6520696
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])

    WARNING: this used to do nans, now it does infs too. 

    old version: return np.isnan(y), lambda z: z.nonzero()[0]
    """

    return ~np.isfinite(y), lambda z: z.nonzero()[0]


def nan_interp(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])


def cube_nan_interpolator(cube):
    #despite its name, this function will also interp over masked values
    
    rows, cols, times = cube.shape
    # flatcube = cube.reshape(rows*cols,times)
    cube.fill_value = np.nan
    filledcube = cube.filled()
    for i in range(rows):
        for j in range(cols):
            nans, x = nan_helper(filledcube[i,j])
            filledcube[i, j, nans] = np.interp(x(nans), x(~nans), filledcube[i,j,~nans])
            
    return filledcube
    

def big_signal_bad_px_finder(chop, cube, corrthresh=0.5):
    # uses correlation coefficients to attempt to detect good pixels.
    # Correlates the cube against the chop signal, so only expect this to work
    # on sources with big signals.
    # Note that this returns a FLAG array. So true means bad pixel.
    rows, cols, times = cube.shape
    
    interpedcube = cube_nan_interpolator(cube)
    
    flatcube = interpedcube.reshape(rows*cols, times)

    corrcoeff = np.corrcoef(chop, flatcube)[0][1:].reshape(rows, cols)
    # we did some fancy index magic there. The corrcoef is a matrix of size rows*cols x rows*cols
    # and it's symmetrical. We only want the correlation with the chop signal (hence the [0]).
    # And we don't need to know that chop-chop correlation === 1 (hence the [1:].
    
    return np.abs(corrcoeff) < corrthresh


