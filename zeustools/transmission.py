import numpy as np
from astropy import units as units
from astropy import constants as consts
import scipy.interpolate as interp
import pandas as pd
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
        with res.open_text(data, "pwv_database.csv") as file:
            table = pd.read_csv(file)
        self.freqs = np.array(table.iloc[:, 0], dtype=float)
        self.transmissions = np.array(table.iloc[:, 1:], dtype=float)
        self.pwvs = np.array(table.columns[1:], dtype=float)
        self.observing_freq = 0

    def interp(self, freq, pwv):
        """
        interpolate the sky transmission at a given PWV and frequency.
        
        :param freq: frequency in GHz
        :param pwv: pwv in mm
        """
        # print(freq,pwv)
        # print(self.freqs)
        # print(self.pwvs)
        return interp.interpn((self.freqs, self.pwvs), self.transmissions, (freq, pwv))

    def interp_um(self, wav, pwv):
        wav = wav * units.micron
        freq = (consts.c/wav).to("GHz").value
        return self.interp(freq, pwv)

    def interp_internal_freq(self, pwv):
        """ 
        interpolate the sky transmission at a given PWV, using the frequency in the member variable `observing_freq`
        
        :param pwv: pwv in mm
        """
        return self.interp(self.observing_freq, pwv)


FILTER_NAMES = {
    "ir": "IRFilter_C15_Front Snout",
    "k2329": "LPF_50cm-1_K2329_Slit-400Array",  # Apparently this is in the Lyot stop and the front.
    # I'm not a huge fan of that fact. 
    "k2338": "BPF_350um_K2338",
    "w1586": "BPF_450um_W1586 CILCO",
    "b688": "LPF_58cm-1_B688_Slit-200Array",
    "k2330": "BPF_645um_K2330",
    "b676": "BPF_200um_B676 Thumper",
    "k2338": "BPF_350um_K2338",
    "k2586": "S3313R9",  # This is the newest 350 micron bandpass filter. 
    # It was designed to mimic the zeus-1 350 micron filter.
    "w1018": "T0555R10",  # This is the zeus-1 350 micron filter.
    "window": "wavenumber-nepers-hdpe"
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
        self.filterType = filterType
        bounds_error=True
        if filterType == "zitex":
            self._load_csv("zitex.csv") # Digitized from Benford Gaidis and Kooi 1999
            bounds_error=False
        elif filterType == "window":
            # HDPE digitized from Birch and Dromey 1981
            self._load_excel("hdpe.xlsx",filterType,x_col="wavenumber",y_col="transmission")
        elif filterType == "scatter":
            self.interpolator = lambda x: 0.95
            return
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

    def _load_csv(self,name,x_col="wavelength",y_col="transmission",x_unit="um",skip_rows=0):
        if type(x_col) is not list:
            x_col = [x_col]
        if type(y_col) is not list:
            y_col = [y_col]
        self.wl = np.array([])
        self.transmission = np.array([])
        with res.open_text(data,name) as csvfile:
            csv = pd.read_csv(csvfile,skiprows=skip_rows)
            for x_col_n,y_col_n in zip(x_col,y_col):
                self.wl=np.hstack([self.wl,csv[x_col_n]])
                self.transmission = np.hstack([self.transmission,csv[y_col_n]])
                if x_unit != "um":
                    raise ValueError("x unit not supported")

    def interp(self,wl):
        """Run the interpolator at the given wavelength (in microns)

        :param wl: The desired wavelength or wavelengths to extract the transmission for.

        :return: numpy array of transmissions corresponding to the given wavelengths.
        """
        return self.interpolator(wl)

# class GratingTransmission(FilterTransmission):
#     def __init__(self,gratingType):
#         with res.open_text(data,)

class GratingTransmission(FilterTransmission):
    """ The grating transmission files are a bit different from the filter data files.
    This class takes care of that difference, and gives you back an object that is for all intents and purposes
    identical to a FilterTransmission object
    """
    def __init__(self,gratingType,fileName="grating_eff.csv",bounds=True):
        orders = ["3","4","5","9"]
        gratingCode = gratingType[0].lower()
        x_cols = [f"{gratingCode}x{order}" for order in orders]
        y_cols = [f"{gratingCode}y{order}" for order in orders]
        self._load_csv(fileName,x_col=x_cols,y_col=y_cols,skip_rows=1)
        self.transmission/=100
        self.interpolator = interp.interp1d(self.wl,self.transmission,bounds_error=bounds)



class ZeusOpticsChain:
    """ This is the main way for you to handle transmission calculations.
    It takes into account all the filters in the system as well as the grating efficiency
    and the tuning ranges of the grating. You can also specify the time period you are interested in
    so that we can load the correct combination of filters and grating etc.

    :param config: This string lets you define the time period that you want. Currently available 
        time periods are ``2021`` to load the configuration for APEX 2021, which includes the old "shiny" grating.
        ``2019`` loads the configuration at APEX in 2019, where we introduced the new k2586 350 micron bandpass filter.
        ``lab_2019`` loads the configuration for the lab tests in 2019 and 2018, where the 350 micron bandpass
        filter was the "k2338" variety, and ``lab_late_2019`` loads the config when the "w1018" bandpass filter was
        installed on the 350 micron array.
    """
    def __init__(self,config="2021"):
        if config=="2021":
            self.filters = {
                "common":["window","zitex","scatter","ir","k2329"],
                "350":["k2329","k2586"],
                "450":["k2329","w1586"],
                "200":["b688","b676"],
                "600":["b688","k2330"]
            }
            self.grating="shiny"
        elif config=="2019":
            self.filters = {
                "common":["window","zitex","scatter","ir","k2329"],
                "350":["k2329","k2586"],
                "450":["k2329","w1586"],
                "200":["b688","b676"],
                "600":["b688","k2330"]
            }
            self.grating="dull"
        elif config=="lab_late_2019":
            self.filters = {
                "common":["window","zitex","scatter","ir","k2329"],
                "350":["k2329","w1018"],
                "450":["k2329","w1586"],
                "200":["b688","b676"],
                "600":["b688","k2330"]
            }
            self.grating="dull"
        elif config=="lab_2019":
            self.filters = {
                "common":["window","zitex","scatter","ir","k2329"],
                "350":["k2329","k2338"],
                "450":["k2329","w1586"],
                "200":["b688","b676"],
                "600":["b688","k2330"]
            }
            self.grating="dull"
        else:
            raise ValueError("unknown configuration name.")
        self.filter_objs = {}
        for key in self.filters.keys():
            value = self.filters[key]
            for filter_name in value:
                if filter_name in self.filter_objs:
                    continue
                self.filter_objs[filter_name] = FilterTransmission(filter_name)
        self.grating_obj= GratingTransmission(self.grating)
        self.tuning_ranges = GratingTransmission(self.grating,fileName="z2_tuning_ranges.csv",bounds=False)

    def compute_transmission(self,wl,filters):
        """This is mostly an internal-use function.
        Given a combination of filters, multiply them all at the given wavelengths."""
        if len(wl)==0:
            return
        total_transmission=np.ones_like(wl,dtype=float)
        for f in filters:
            this_filter_trans = self.filter_objs[f].interp(wl)
            total_transmission*= this_filter_trans
            # print(f,this_filter_trans,total_transmission)
        return total_transmission


    def get_transmission_microns(self,wl,show_tuning_range=False):
        """ Use this method for all your transmission computation needs! Once you have
        initialized the object, supply this method with a wavelength or array of wavelengths
        in order to compute the throughput at those wavelengths. 

        :param wl: wavelength or numpy array of wavelengths of interest
        :param show_tuning_range: Optional. If this is True, we will return "0" transmission for wavelengths
            that cannot be observed

        :return: numpy array of throughput fractions.
        """
        if type(wl) is not np.ndarray:
            wl = np.array(wl)
        out = np.zeros_like(wl,dtype=float)
        values_350 = np.logical_and(wl>300,wl<400)
        values_450 = np.logical_and(wl>=400,wl<500)
        values_650 = np.logical_and(wl>=500,wl<700)
        values_200 = np.logical_and(wl>100,wl<=300)
        for i,f in zip([values_200,values_350,values_450,values_650],["200","350","450","600"]):
            out[i] = self.compute_transmission(wl[i], self.filters["common"]+self.filters[f])
        out = out * self.grating_obj.interp(wl)
        if show_tuning_range:
            out = out * self.tuning_ranges.interp(wl)
        return out

    def get_details_table(self,wl):
        table = []
        for filter_name in self.filter_objs:
            transmission = []
            for w in wl:
                
                try:
                    transmission.append(self.filter_objs[filter_name].interp(w))
                except ValueError:
                    transmission.append(np.nan)

            table.append([filter_name]+transmission)
        return table
