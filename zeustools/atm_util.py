import numpy as np
from astropy import units as units
from astropy import constants as consts
import scipy.interpolate as interp
import pandas

class TransmissionHelper:
    def __init__(self,file):
        table=pandas.read_csv(file)
        self.freqs = table.iloc[:,0]
        self.transmissions = table.iloc[:,1:]
        self.pwvs = np.array(table.columns[1:],dtype=float)

    def interp(self,freq,pwv):
        return interp.interpn((self.freqs,self.pwvs),self.transmissions,(freq,pwv))

def airmass_factor(elev):
    elev = elev*pi/180
    return(np.sin(elev))

def run_pipeline():
    #Load in configuration values, and parse them into correct data types
    #TODO:are there better config file systems, and/or better ways of
    #organizing this?
    config = configparser.ConfigParser()
    if len(sys.argv)>1:
        settings = sys.argv[1]
    else:
        settings = 'settings.ini'
    config.read(settings)
    globalvals = config["GLOBAL"]
    outputfile = globalvals["output_filename"]
    spatial_position = globalvals.getint("spat_pos")
    lambda_line = globalvals.getfloat("lambda_line_px")
    px_delta_lambda = globalvals.getfloat("px_delta_lambda")
    skytemp = globalvals.getfloat("sky_temp")
    cabintemp = globalvals.getfloat("cabin_temp")
    beamsize = globalvals.getfloat("beam_size_steradian")
    docalib = globalvals.getboolean("do_calib")
    unflatten = globalvals.getboolean("unflatten")
    docontsub = globalvals.getboolean("do_contsub")
    line_positions = globalvals["where_is_line_flux"]
    line_positions = list(map(int,line_positions.split(',')))
    ptcoup = globalvals.getfloat("pt_src_coupling")
    teleff = globalvals.getfloat("telescope_efficiency")

    #Use astropy's nice units converter to get bin width in km/s
    px_kms = (px_delta_lambda/lambda_line*const.c).to("km/s").value

    #Figure out what the boolean options should be. 
    if unflatten:
        docalib = False

    #Convert the list of [REDUCTION*] sections in the settings.ini file to something more useful
    reductions_settings=[config[name] for name in config.sections() if name != "GLOBAL"]
    reductions = []

    #Loop over those sections to load in and calibrate each data scan chunk
    for rsett in reductions_settings:
        #the configparser doesn't throw an exception if a thing is 
        #not defined it just returns none.
        #I dont know how I feel 
        #about that

        #Get the per-data-chunk values
        sky_transp = rsett.getfloat("atm_transmission")
        spec_pos = rsett.getint("spec_pos_of_line")
        minpx = rsett.getint("min_spec_px")
        maxpx = rsett.getint("max_spec_px")

        #load this data chunk, and chop off pixels we don't like

        data = load_data_and_extract(rsett["path"], spatial_position)
        data = cut(data,minpx,maxpx)

        if docalib:
            #Figure out how many w/m^2/bin the skychop was worth
            flat_flux_value = flat_to_wm2(
                sky_transp,
                lambda_line*units.micron,
                px_delta_lambda*units.micron,
                sky_temp=skytemp*units.K,
                cabin_temp=cabintemp*units.K,
                beam_size=beamsize*units.steradian).value
            #add telescope and sky stuff to calibration value
            flat_flux_value = flat_flux_value/sky_transp/ptcoup/teleff
            #Apply that scaling to the data
            data = flux_calibration(data,flat_flux_value)

        plot_spec((data[0]-spec_pos,data[1],data[2],data[3]),f"{outputfile}_{rsett.name}.png")
        reductions.append((data,spec_pos))

    #If we only have one reduction, shifting-and-adding can't happen
    if len(reductions) > 1:
        for i in range(len(reductions)-1):
            data,spec_pos = reductions[i]
            data2,specpos2 = reductions[i+1]
            reductions[i+1]=(shift_and_add(data,data2,spec_pos,specpos2),0)
            #this is kind of ugly because it's 1 am and gordon wants this yesterday
            #TODO: cleaner code. play around with "reduce" function and stuff
    else:
        # Move the spectrum so that the line is on pixel
        # due to the fact I'm using tuples I have to rebuild the tuple...
        spec, data, err, wt = reductions[-1][0]
        line_px = reductions[-1][1]

        reductions[-1] = ((spec-line_px, data, err, wt),0)

    #Now that we have a co-added spectrum, we should subtract the continuum
    addedspec=reductions[-1][0]

    if docontsub:
        addedspec = contsub(addedspec,line_positions)
        
        #I told the user to make line_positions relative to the spectral pixel of the line.
        #When we shifted and added, we shifted the line to be at "spectral position" 0
        #So if the line started on pixel 7 (and 8) the user should've given us
        #"the line is at pixel 7" and "flux can be found on pixels 0 and 1."
        #Now that the line is at 0, the correct places to find flux is 0 and 1.
        #This almost makes sense to 1 AM me. TODO: clean up the explanation and make sure it's true
    if docalib:
        velspec = wavelength_calibration(addedspec, 0, px_kms)
        #print(velspec)
    else:
        velspec = addedspec

    plot_spec(velspec,f"{outputfile}.png")
    #Finally, output the csv for gordon.
    with open(outputfile+".csv",'w') as csvf:
        if docalib:
            labely = "signal W m^-2 bin^-1"
            labelx = "velocity relative to galaxy redshift km/s"
        else:
            labely = "raw data (fraction of flat probably)"
            labelx = "shifted pixel number (0 is the line pixel)"
        csvf.write(getcsvspec(labely, velspec[1])+'\n')
        csvf.write(getcsvspec(labelx, velspec[0])+'\n')
        csvf.write(getcsvspec("error in signal same units as signal",velspec[2])+'\n')


if __name__ == '__main__':
    run_pipeline()
