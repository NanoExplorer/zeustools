import numpy as np



def fftnoise(f):
    # https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
    f = np.array(f,dtype='complex')
    Np = (len(f)-1)//2
    phases = rng.normal(scale=0.25,size=Np) * 2 * np.pi
    phases - np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real*np.sqrt(len(f))

def pwr(i):
    return (i*i.conj()).real