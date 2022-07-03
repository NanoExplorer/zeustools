import numpy as np
from astropy import units as units
from astropy import constants as consts
import scipy.interpolate as interp
import pandas
from zeustools import data
import importlib.resources as res


class AtmosphereTransmission:
    """ 
    This class provides a method to calculate the sky
    transmission from APEX at arbitrary PWV and wavelength.
    It accomplishes this by using a downloaded copy of the APEX weather
    data (included with the package).

    """
    def __init__(self):
        with res.open_text(data,"pwv_database.csv") as file:
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

FILTER_NAMES = {
    "ir":"IRFilter_C15_Front Snout",
    "k2329":"LPF_50cm-1_K2329_Slit-400Array", # Apparently this is in the Lyot stop and the front.
    # I'm not a huge fan of that fact. 
    "k2338":"BPF_350um_K2338",
    "w1586":"BPF_450um_W1586 CILCO",
    "b688":"LPF_58cm-1_B688_Slit-200Array",
    "k2330":"BPF_645um_K2330",
    "b676":"BPF_200um_B676 Thumper",
    "k2338":"BPF_350um_K2338"
    "k2586":"S3313R9" # This is the newest 350 micron bandpass filter. 
    # It was designed to mimic the zeus-1 350 micron filter.
    "w1018":"T0555R10" # This is the zeus-1 350 micron filter.
}

class FilterTransmission:
    def __init__(self, filterType):
        if filterType == "zitex":
            self.load_csv("zitex.csv")
        elif filterType == "k2586" or filterType == "w1018":
            self.load_excel("Replacement 350um_BP_Sept 2019.xls",filterType)
        else:
            self.load_excel("Z2_CurrentFilters.xlsx",filterType)

    def load_excel(self,fname,sheetname,x_col="Wave#(cm-1)",y_col="Transmission",x_unit="cm-1"):
        with res.open_binary(data,fname) as xlfile:
            xl = pd.ExcelFile(xlfile)
            df = pd.read_excel(xl,FILTER_NAMES[sheetname],skiprows=1)
            if x_unit == "cm-1":
                self.wl = 10000/df[x_col]
            else:
                raise ValueError("x unit not supported")
            self.transmission = df[y_col]

    def load_csv(self,name,x_col="wavelength",y_col="transmission",x_unit="um"):
        with res.open_text(data,name) as csvfile:
            csv = pd.read_csv(csvfile)
            self.wl = csv[x_col]
            self.transmission = csv[y_col]
            if x_unit != "um":
                raise ValueError("x unit not supported")

    def interp(self,freq):
        pass

class GratingTransmission(FilterTransmission):
    def __init__(self,gratingType):
        with res.open_text(data,)


class ZeusOpticsChain:
    def __init__(self,config="2021"):
        if config="2021":
            self.filters = {
                "common":["entrance","zitex","scatter","ir","k2329"],
                "350":["k2329","k2586"],
                "450":["k2329","w1586"],
                "200":["b688","b676"],
                "600":["b688","k2330"]
            }
            self.grating="shiny"
        elif config="2019":
            self.filters = {
                "common":["entrance","zitex","scatter","ir","k2329"],
                "350":["k2329","k2586"],
                "450":["k2329","w1586"],
                "200":["b688","b676"],
                "600":["b688","k2330"]
            }
            self.grating="dull"
        elif config="lab_late_2019":
            self.filters = {
                "common":["entrance","zitex","scatter","ir","k2329"],
                "350":["k2329","w1018"],
                "450":["k2329","w1586"],
                "200":["b688","b676"],
                "600":["b688","k2330"]
            }
            self.grating="dull"
        elif config="lab_2019":
            self.filters = {
                "common":["entrance","zitex","scatter","ir","k2329"],
                "350":["k2329","k2338"],
                "450":["k2329","w1586"],
                "200":["b688","b676"],
                "600":["b688","k2330"]
            }
            self.grating="dull"
        else:
            raise ValueError("unknown configuration name.")
        self.filter_objs = {}
        for key,value in self.filters:
            for filter_name in value:
                if filter_name in self.filter_objs:
                    continue
                self.filter_objs["filter_name"] = FilterTransmission(filter_name)



    def get_transmission_microns(self,wl):
        if 300<wl<400:




def airmass_factor(elev):
    elev = elev*np.pi/180
    return(np.sin(elev))

