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
    """
    This class provides a good way to interpolate the transmission of the ZEUS-2 filters.
    For most of the filters, we have spreadsheets of their measured transmission, but for the
    zitex filter I digitized the plot of G110 zitex transmission from Benford 1999.

    Create a FilterTransmission object providing one of the filter names/keys from the FILTER_NAMES 
    dictionary, and then call the interp function with a wavelength in microns to get back its transmission at 
    that wavelength.

    """
    def __init__(self, filterType):
        """Initializes a FilterTransmission object with the transmission properties of a certain
        type of filter.

        :param filterType: The name of the filter to load. e.g., `k2338`, `zitex`, or `ir`.

        :return: FilterTransmission object corresponding to the named filter.
        """
        bounds_error=True
        if filterType == "zitex":
            self._load_csv("zitex.csv") # Digitized from Benford Gaidis and Kooi 1999
            bounds_error=False
        elif filterType == "window":
            # HDPE digitized from Birch and Dromey 1981
            self._load_excel("hdpe.xlsx","wavenumber-nepers-hdpe",x_col="wavenumber",y_col="transmission")
        elif filterType == "scatter":
            self.interpolator = lambda s,x: 0.95
        elif filterType == "k2586" or filterType == "w1018":
            self._load_excel("Replacement 350um_BP_Sept 2019.xls",filterType)
        else:
            self._load_excel("Z2_CurrentFilters.xlsx",filterType)
        self.interpolator = interp.interp1d(self.wl,self.transmission,bounds_error=bounds_error,fill_value=0.95)

    def _load_excel(self,fname,sheetname,x_col="Wave#(cm-1)",y_col="Transmission",x_unit="cm-1"):
        with res.open_binary(data,fname) as xlfile:
            xl = pd.ExcelFile(xlfile)
            df = pd.read_excel(xl,FILTER_NAMES[sheetname],skiprows=1)
            if x_unit == "cm-1":
                self.wl = 10000/df[x_col]
            else:
                raise ValueError("x unit not supported")
            self.transmission = df[y_col]

    def _load_csv(self,name,x_col="wavelength",y_col="transmission",x_unit="um"):
        with res.open_text(data,name) as csvfile:
            csv = pd.read_csv(csvfile)
            self.wl = csv[x_col]
            self.transmission = csv[y_col]
            if x_unit != "um":
                raise ValueError("x unit not supported")

    def interp(self,wl):
        """Run the interpolator at the given wavelength (in microns)

        :param wl: The desired wavelength or wavelengths to extract the transmission for.

        :return: numpy array of transmissions corresponding to the given wavelengths.
        """
        return self.interpolator(wl)

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

    def compute_transmission(self,wl,filters):
        total_transmission=np.ones_like(wl)
        for f in filters:
            total_transmission*= self.filter_objs[f].interp(wl)

    def get_transmission_microns(self,wl):
        if 300<wl<400:
            return self.compute_transmission(wl, self.filters["common"]+self.filters["350"])
        if 400<wl<500:
            return self.compute_transmission(wl, self.filters["common"]+self.filters["450"])
        if 500<wl<700:
            return self.compute_transmission(wl, self.filters["common"]+self.filters["600"])
        if 100<wl<300:
            return self.compute_transmission(wl, self.filters["common"]+self.filters["200"])




def airmass_factor(elev):
    elev = elev*np.pi/180
    return(np.sin(elev))

