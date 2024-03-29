import numpy as np
from numba import njit, objmode


def readChopFile(fname):
    """Reads in the .chop file so you know when the chopper was on or off"""
    filename = str(fname + '.chop')
    fileID = open(filename,'r')
    dataArray = np.genfromtxt(fileID, delimiter=' ')
    fileID.close()
    
    cidx = dataArray[:, 0]
    chop = dataArray[:, 1]

    return cidx, chop


@njit(cache=True)
def gaussianSimple(vec, a, v0, sigma):
    expFrac = -1.0*((vec-v0)**2.0)/(2.0*sigma**2.0)
    return a*np.exp(expFrac)


@njit(cache=True)
def createModelSnakeEntireArray(tseries, goodSnakeCorr=0.85, minSnakeNumber=10, bestPixel=(6,14),fastFreq=0.25,quiet=False):
    # If I put this in here njit can cache it otherwise it cannot.
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
        """

        return np.isnan(y), lambda z: z.nonzero()[0]
    """Creates a model snake given the time series of the entire array

    First correlates all pixels against each other to find a trend.
    Px correlated better than `goodSnakeCorr` with the `bestPixel` are 
    used to generate the model snake.

    Model snake is generated by throwing out high-frequency fourier components.
    This is equivalent to convolving the time series with a gaussian.
    """
    #first, get the time series of the best pixel (which will be used to generate the template)
    bestTS = tseries[bestPixel]
    #print(bestTS)
    #Cross-corelate the best pixel time series with all other time series:
    corMat = np.zeros((4,np.shape(tseries)[0]*np.shape(tseries)[1]))
    count = 0
    good_count=0
    for i in range(np.shape(tseries)[0]):
        for j in range(np.shape(tseries)[1]):
            non_nan_indexes = np.isfinite(bestTS) & np.isfinite(tseries[i,j,:])
            a = np.corrcoef(bestTS[non_nan_indexes], tseries[i,j][non_nan_indexes])
            #print(a)
            corMat[0,count] = i
            corMat[1,count] = j
            if np.isfinite(a[0,1]):
                corMat[2,count] = np.abs(a[0,1])
                corMat[3,count] = np.sign(a[0,1])
                if np.abs(a[0,1])>goodSnakeCorr:
                    #print(i,j)
                    good_count += 1
            count+=1
    print(str(good_count) + " correlated px")

    #Find out which pixels are best correlated with the best time series     
    corMatSorted = corMat[:,corMat[2,:].argsort()]
    goodIdxs = np.where((corMatSorted[2,:] > goodSnakeCorr) & (corMatSorted[2,:] < 1.0))[0]
    #If none are well correlated, combine the minimum allowable number in order to get the snake template
    if len(goodIdxs) == 0:
        goodIdxs = np.arange(-1-minSnakeNumber,-1)
    #Make the model snake by combining the best-correclated time series
    thisModelSnake = np.zeros_like(bestTS)
    #normFactor = 0.0
    #plt.figure()
    for ll in goodIdxs:
        chunkRMS = np.nanstd(tseries[int(corMatSorted[0,ll]),int(corMatSorted[1,ll]),:][0:1000])
        # if not quiet:
        #     print("CHUNKRMS =" + str(chunkRMS) + "ll="+ str(ll))
        medianZeroCurTS = tseries[int(corMatSorted[0,ll]),int(corMatSorted[1,ll]),:] - np.nanmedian(tseries[int(corMatSorted[0,ll]),int(corMatSorted[1,ll]),:])
        #print chunkRMS
        ampNormTS = corMatSorted[3,ll]*medianZeroCurTS/(3.0*np.nanstd(medianZeroCurTS))
        #plt.plot(ampNormTS)

        if chunkRMS > 10:
            nans,x=nan_helper(ampNormTS)
            ampNormTS[nans]=np.interp(x(nans),x(~nans),ampNormTS[~nans])
            thisModelSnake += ampNormTS/(chunkRMS**2.0)
        #normFactor += chunkRMS**2.0
    #print(thisModelSnake)
    thisModelSnake = thisModelSnake - np.nanmedian(thisModelSnake)
    thisModelSnake = thisModelSnake/(3.0*np.nanstd(thisModelSnake))
    #plt.plot(thisModelSnake)
    #Create an array of time values, so that we can use the FFT in order to remove any signal from the model snakes    
    #Take fourier series and eliminate fast components
    #Mirror the time series and append it to the end to reduce edge effects.
    #This gives better results than not doing it, but there may still be a better way
    origlen=len(thisModelSnake)
    with objmode(fourier='complex128[:]'):
        fourier = np.fft.fft(np.append(thisModelSnake,thisModelSnake[::-1]))
    #fourier = np.fft.fft(thisModelSnake)
    #print(fourier)
    #exit()
    thisn = thisModelSnake.size*2
    timestep = 1/398.72
    with objmode(freq='float64[:]'):
        freq = np.fft.fftfreq(thisn, d=timestep)

    gaussianWindow = gaussianSimple(freq, 1.0, 0.0, fastFreq)
    fourier = fourier*gaussianWindow

    with objmode(ifftModelSnake='complex128[:]'):
        ifftModelSnake = np.fft.ifft(fourier)
    modelSnake = ifftModelSnake.real[0:origlen]
    
    #plt.figure()
    #plt.plot(modelSnake)
        
    return modelSnake
