import argparse
import numpy as np
from numpy import ma
from scipy import optimize
from zeustools.codystools import readChopFile
from zeustools import mce_data
from matplotlib import pyplot as plt


def gaussian(x,sigma,mu,a):
    return a*np.exp(-1/2*((x-mu)/(sigma))**2)
# From https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m


def twoD_Gaussian(pos, amplitude, xo, yo, sigma_x, offset):
    sigma_y = sigma_x
    theta=0
    x,y=pos
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) +
                                  c*((y-yo)**2)))
    return g.ravel()


class chi_sq_solver:
    def __init__(self,
                 bins,
                 ys,
                 function,
                 guesses,
                 bounds=None
                 ):
        #Initialize object members
        self.bins = bins
        self.ys = ys
        self.centerbins = self.centers(self.bins)
        self.function = function
        #Calculate minimum chi squared fit
        self.result = optimize.minimize(self.chi_sq,guesses,bounds=bounds)
        
    def chi_sq(self, args):
        #Return the chi_squared statistic for the binned data and the arguments for the function
        sum = 0
        for i in range(len(self.ys)-1):
            try:
                E = self.function(self.centerbins[i], *args)
                if self.ys[i] != 0:
                    sum += (self.ys[i]-E)**2/self.ys[i]
            except FloatingPointError:
                print("Oh no! There was a floating point error.")
                #print(self.centerbins[i],*args)
                #exit()
            #print(self.ys[i],E,sum)
        #print(args,sum)
        return sum

    def centers(self,bins):
        #Calculate the centers of the bins, or the average x value of each bin.
        #Would be mathematically cool if we could average the x values of all the data points,
        #but that probably shouldn't affect anything substantially.
        centers = []
        for i in range(len(bins)-1):
            centers.append((bins[i+1]-bins[i])/2+bins[i])
            #Take the average value of two bins and add it to the lower bin value
        return centers


def nd_mad(nparray,axis,extrainfo=False):
    med = np.nanmedian(nparray,axis=axis)
    expand_slicing= tuple((np.newaxis if x==axis else slice(None) for x in range(len(nparray.shape))))
    dist_from_med = np.abs(nparray-med[expand_slicing])
    mdev=np.median(dist_from_med,axis=axis)
    if extrainfo:
        return mdev,dist_from_med
    else:
        return mdev


def nd_mads_from_median(nparray,axis,zero_padding_factor=0.0001):
    """Returns the number of mads from the median for each data point, with median and mad computed over the specified axis

    """
    #slice(None) is equivalent to ':'
    expand_slicing= tuple((np.newaxis if x==axis else slice(None) for x in range(len(nparray.shape))))
    mdev,dist_from_med = nd_mad(nparray,axis,extrainfo=True)
    if type(mdev) is np.ndarray:
        mdev[mdev==0] = zero_padding_factor
        mdev = mdev[expand_slicing]
    else:
        if mdev==0:
            mdev=zero_padding_factor

    s = dist_from_med/mdev
    return s


def nd_reject_outliers(nparray,MAD_chop=5,axis=2):
    """Returns a masked array masking outliers on a certain axis.

    The input array is examined using the median absolute deviation method
    and outliers greater than the MAD_chop number are masked. 
    """
    s=nd_mads_from_median(nparray,axis)
    return ma.masked_array(nparray,mask=s > MAD_chop)


def parseCmdArgs(argumentList,helpList,typeList):
    """Parses command-line arguments.

    arguments:
    3 lists of argument data. First is the list of arguments. Each
    argument is a list that will be passed to the add_argument function of argparse
    Each element in the second list is the help string for that argument,
    The last list contains the type.

    Usage::
    
        common.parseCmdArgs([['settings'],
                             ['-o','--override']],
                            ['Settings json file','array indices in the format a:b to extract from infile list'],
                            [str,str])

    """
    parser = argparse.ArgumentParser()
    for argument,theHelp,theType in zip(argumentList,helpList,typeList):
        if theType == 'bool':
            parser.add_argument(*argument,help=theHelp,action='store_true')
        else:
            parser.add_argument(*argument,help=theHelp,type=theType)
    return parser.parse_args()


def nd_nan_outliers(nparray,MAD_chop=5,axis=2):
    """Returns an array with nans instead of outliers on a certain axis

    The input array is examined using the median absolute deviation method
    and outliers greater than the MAD_chop number are changed to np.nan. 
    """
    s=nd_mads_from_median(nparray,axis)
    nparray[s>MAD_chop]=np.nan
    return nparray


def histGaussian(something,nbins=50):
    n,bins,patches = plt.hist(something,nbins)
    guessavg = np.mean(something)
    guessstd = np.std(something)
    guessamp = len(something)/nbins
    robot = chi_sq_solver(bins, n, gaussian, np.array([guessstd, guessavg, guessamp]),
                          bounds=[(1, None), (None, None), (0, None)])
    domain = np.arange(min(bins),max(bins),100)
    model = gaussian(domain,*(robot.result.x))
    plt.plot(domain,model)
    return robot.result


def readPf(filename):
    data = np.loadtxt(filename,dtype=str)
    phase= data[:,3].astype(int)
    ampl = data[:,4].astype(float)
    good_data = ampl[phase==2] - 1
    return(good_data)


def processChop(filename):
    """ Loads in MCE data file, chop file, and TS file.

    Returns a tuple containing:
        0: data points where chop is on
        1: ts for chop on
        2: data for chop off
        3: ts for chop off

    """
    chopfile = mce_data.SmallMCEFile(filename)
    chopdata = chopfile.Read(row_col=True).data
    chopchop = readChopFile(filename)[1]
    ts = np.loadtxt(f"{filename}.ts")
    tstimes = ts[0:len(chopchop),1]
    # tsindex = ts[0:len(chopchop),0].astype(int)
    chop_on = chopdata[:, :, chopchop==1]
    chop_off= chopdata[:, :, chopchop==0]
    ts_on = tstimes[chopchop==1]
    ts_off= tstimes[chopchop==0]
    return(chop_on,ts_on,chop_off,ts_off)


class ArrayMapper:
    def __init__(self):
        self.arrays = {'a':np.loadtxt("config/arrayA_map.dat",usecols=range(0,4),dtype=int),
                       'b':np.loadtxt("config/arrayB_map.dat",usecols=range(0,4),dtype=int),
                       'c':np.loadtxt("config/arrayC_map.dat",usecols=range(0,4),dtype=int)}
        #these arrays have columns:
        # spatial, spectral, mcerow, mcecol
    
    def array_name(self,name):
        """Converts various names for the arrays into the internal scheme used by this class"""
        name=str(name)
        if name=='400' or name=='350' or name=='450' or name=='A':
            name = 'a'
        elif name=='200' or name=='B':
            name = 'b'
        elif name=='600' or name=='C':
            name='c'
        if not (name=='a' or name=='b' or name=='c'):
            raise ValueError("invalid array name")
        return name
    
    def phys_to_mce(self,spec,spat,array):
        """Given a physical position of a pixel (spectral position, spatial position, array name)
        returns the mce_row,mce_col of that pixel"""
        
        array_to_use = self.arrays[self.array_name(array)]
        is_correct_spatialpos = array_to_use[:,0] == spat
        is_correct_spectpos = array_to_use[:,1] == spec
        is_correct_px = np.logical_and(is_correct_spectpos, is_correct_spatialpos)
        px = np.where(is_correct_px)[0][0]
        return(array_to_use[px, 2], array_to_use[px, 3])


def processChopBetter(fname):
    """Loads in MCE data file and chop file.

    returns an array with the medians of each chop cycle for all pixels. 

    TODO: needs a better name   
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
    return f"{date}/{name}_{date[2:]}_{number:04d}"


def loadts(filename):
    times=np.loadtxt(f"{filename}.ts")[:,1]
    return times
