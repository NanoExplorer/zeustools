from astropy import units
from astropy import constants as const
import numpy as np


def flat_to_wm2(sky_transparency,
                obs_wavelength,
                pixel_deltalambda,
                sky_temp=270*units.K,
                cabin_temp=288*units.K,
                beam_size=((6*units.arcsec/2)**2*np.pi/np.log(2)).to(units.steradian)
                ):
    """Get the power in W/m^2/bin that the detectors see when
    doing a skychop. This can be multiplied by the flat-divided spectra later.
    
    :param sky_transparency: The line-of-sight transparency of the sky. 
        This is usually calculated by get_real_pwv,
        then querying the APEX sky model.
    :param obs_wavelength: Astropy Quantity containing the lab-frame wavelength of the line
    :param pixel_deltalambda: Astropy Quantity containing the difference in wavelength between light 
        seen by the line pixel and a pixel next to it.
        TODO: how linear is it really?
    :param sky_temp: astropy quantity guess of sky temperature - usually around 0 Celsius
    :param cabin_temp: astropy quantity temperature in cabin
    :param beam_size: astropy quantity beam size
    """
    frequency = (const.c/obs_wavelength).to(units.Hz)
    bt = units.brightness_temperature(frequency)
    opacity = 1-sky_transparency
    temp_delta = cabin_temp-(sky_temp*opacity)
    brightness_temp = (temp_delta).to("Jy/steradian",equivalencies=bt)
    spectral_stuff = units.spectral_density(obs_wavelength) 
    calib_flux_density = (brightness_temp*beam_size).to("W / (m^2 Hz)",equivalencies=spectral_stuff)
    bin_width = (const.c/obs_wavelength - const.c/(obs_wavelength+pixel_deltalambda)).to("Hz")
    return calib_flux_density*bin_width


def get_real_pwv(pwv,altitude):
    """Given the zenith PWV (reported by APEX) and altitude of source,
    returns the real amount of water between the telescope and space.
    
    Basically returns pwv/cos(zenith_angle)

    :param pwv: zenith PWV reported by APEX
    :param altitude: Altitude of source
    """
    zenith_angle = 90-altitude
    airmass = 1/np.cos(zenith_angle*np.pi/180)
    return pwv*airmass
