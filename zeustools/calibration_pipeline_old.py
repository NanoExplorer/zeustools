import numpy as np
from astropy import units
from astropy import constants as const
import configparser
import matplotlib
from matplotlib import pyplot as plt 
import sys
from zeustools.bpio import load_data_and_extract, extract_from_beamfile
from zeustools.calibration import flat_to_wm2


font = {'size': 18}
matplotlib.rc('figure', figsize=(8, 6))
matplotlib.rc('font', **font)


def flux_calibration(data,
                     flat_flux_density,  # W/m^2/bin
                     ):
    spec_pos,sig,noise,wt = data
    
    scaled_signal = sig * flat_flux_density 
    
    scaled_err = noise * flat_flux_density
    return (spec_pos, scaled_signal, scaled_err,wt)


def wavelength_calibration(data,
                           position_of_line,
                           bin_width  # km/s
                           ):
    spec_pos, _, _, _ = data
    velocity = (spec_pos - position_of_line) * bin_width
    return (velocity, data[1], data[2], data[3])


def cut(data, min_px, max_px):
    out = []
    for i in data:
        out.append(i[min_px:max_px])
    return out


def shift_and_add(data1, data2, px1, px2):
    """
    takes in two spectra. Shifts the second one spectrally by px_offset
    to align the line pixel between the two runs. Weights the spectra appropriately
    TODO: use np.average to clean up this mess.
    """
    spec,sig,noise,wt = data1
    spec2,sig2,noise2,wt2 = data2

    nan_idxs = np.isnan(sig)
    nan_idx2 = np.isnan(sig2)

    sig[nan_idxs] = 0
    sig2[nan_idx2] = 0
    noise[nan_idxs] = 10e10
    noise2[nan_idx2] = 10e10
    spec -= px1
    spec2 -= px2 
    #That shifts the spectral pixel number so that the line is on position "0" 

    allspecs = np.append(spec, spec2)
    minspec = np.min(allspecs)
    maxspec = np.max(allspecs)
    outspec = np.arange(minspec, maxspec+1, dtype=int)
    outsig = np.zeros_like(outspec, dtype=float)
    outnoise= np.zeros_like(outspec, dtype=float)
    idx = (np.isin(outspec, spec)).nonzero()[0]
    idx2 = np.isin(outspec, spec2).nonzero()[0]
    outsig[idx] += sig/noise**2
    outsig[idx2] += sig2/noise2**2
    outnoise[idx] += 1/noise**2
    outnoise[idx2] += 1/noise2**2
    outnoise = 1/outnoise
    outsig = outsig*outnoise
    outnoise = np.sqrt(outnoise)
    outsig[nan_idxs] = np.nan
    return(outspec, outsig, outnoise, None)


def get_drop_indices(spec_pos,px_to_drop):
    line_px =np.array(px_to_drop)[:,None]
    boolarray = np.all(spec_pos != line_px,axis=0)
    return boolarray.nonzero()[0]


def contsub(data,line_px):
    spec_pos, sig, err, wt = data 
    idxs = get_drop_indices(spec_pos, line_px)
    #print(sig)
    #print(idxs)
    continuum = np.average(sig[idxs], weights=1/err[idxs]**2)
    #print(idxs)
    return (spec_pos,sig-continuum, err, wt)


def getcsvspec(label,spec):
    stringout = label+', '
    for i in spec:
        stringout += str(i)+", "
    return stringout


def plot_spec(spec,saveas):
    line = plt.step(spec[0], spec[1], where='mid')
    lncolor = line[0].get_c()
    plt.errorbar(spec[0], spec[1], spec[2], fmt='none', ecolor=lncolor)
    plt.savefig(saveas, dpi=300)
    plt.close()


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
        usebeamspec = rsett.getboolean("use_beamspec")
        if usebeamspec:
            beamnum = rsett.getint('beam_number')
            bs_arrnum=rsett.getint('beamspec_array_number')

        #Get the per-data-chunk values
        sky_transp = rsett.getfloat("atm_transmission")
        spec_pos = rsett.getint("spec_pos_of_line")
        minpx = rsett.getint("min_spec_px")
        maxpx = rsett.getint("max_spec_px")

        #load this data chunk, and chop off pixels we don't like
        if usebeamspec:
            data = extract_from_beamfile(rsett['path'],beamnum*2,spatial_position,bs_arrnum)
            data2 = extract_from_beamfile(rsett['path'],beamnum*2+1,spatial_position,bs_arrnum)
            addedsig = -data[1]+data2[1]
            addednoise = np.sqrt(data[2]**2+data2[2]**2)
            addedwt = data[3]+data2[3]
            data = np.array([data[0],addedsig,addednoise,addedwt])
        else:
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
        if unflatten:
            #load in the flat file
            flatpath = rsett['path'].replace("final_spec","flat")
            if usebeamspec:
                raise RuntimeError("beamspec not yet implemented with unflattening")
            flat = load_data_and_extract(flatpath,spatial_position)
            flatcut = cut(flat,minpx,maxpx)
            #multiply the flattened data by the flat to get back the original data
            data = data * flatcut
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
