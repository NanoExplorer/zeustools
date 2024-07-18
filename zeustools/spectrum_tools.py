import numpy as np
rng = np.random.default_rng()


def fftnoise(f):
    # https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
    f = np.array(f,dtype='complex')
    Np = (len(f)-1)//2
    phases = rng.normal(scale=np.sqrt(np.pi/2),size=Np) # variance = pi/2 since standard deviation is sqrt(pi/2)
    phases - np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real*np.sqrt(len(f))

def pwr(i):
    return (i*i.conj()).real

def average_using_log_bins(data_x,data_y,min_freq,max_freq,npts):
    bins_plus_centers = np.logspace(
        np.log10(min_freq),
        np.log10(max_freq),
        npts*2+1)
    bins = bins_plus_centers[::2]
    centers = bins_plus_centers[1::2]
    test,test_edge = np.histogram(data_x,bins=bins,weights=data_y)
    count,_ = np.histogram(data_x,bins=bins)
    avg = test/count
    return centers,avg 