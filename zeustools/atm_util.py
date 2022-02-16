import numpy as np
from astropy import units as units
from astropy import constants as consts
import scipy.interpolate as interp
import pandas

class TransmissionHelper:
    def __init__(self,file):
        table=pandas.read_csv(file)
        self.freqs = np.array(table.iloc[:,0],dtype=float)
        self.transmissions = np.array(table.iloc[:,1:],dtype=float)
        self.pwvs = np.array(table.columns[1:],dtype=float)
        self.observing_freq = 0

    def interp(self,freq,pwv):
        """
        params: 
            freq: frequency in GHz
            pwv: pwv in mm
        """
        print(freq,pwv)
        # print(self.freqs)
        # print(self.pwvs)
        return interp.interpn((self.freqs,self.pwvs),self.transmissions,(freq,pwv))

    def interp_internal_freq(self,pwv):
        return self.interp(self.observing_freq,pwv)

def airmass_factor(elev):
    elev = elev*np.pi/180
    return(np.sin(elev))
