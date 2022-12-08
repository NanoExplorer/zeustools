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


def bb_temp_watts(obs_wavelength,
                  pixel_deltalambda,
                  source_temp=270*units.K
                  ):
    """Get the power in W that the detectors see when
    looking at a source. This is useful for dark IV tests and stuff

    :param obs_wavelength: Astropy Quantity containing the wavelength observed
    :param pixel_deltalambda: Astropy Quantity containing the difference in wavelength between light 
        seen by the line pixel and a pixel next to it.
        TODO: how linear is it really?
    :param source_temp: astropy quantity source temperature - 77 K for LN2, ~285 K for roomtemp
    """
    frequency = (const.c/obs_wavelength).to(units.Hz)
    bt = units.brightness_temperature(frequency)
    brightness_temp = source_temp.to("Jy/steradian", equivalencies=bt)
    spectral_stuff = units.spectral_density(obs_wavelength) 
    zeus_beam_radius = np.arcsin(1/2.7/2)
    zeus_beam_sr = zeus_beam_radius**2 * np.pi * units.steradian
    zeus_400_px_size = 1.26e-3
    zeus_400_px_area = zeus_400_px_size**2 * units.m**2
    etendue = zeus_beam_sr * zeus_400_px_area
    calib_flux_density = (brightness_temp*etendue).to("W / (m^2 Hz)", equivalencies=spectral_stuff)
    bin_width = (const.c/obs_wavelength - const.c/(obs_wavelength+pixel_deltalambda)).to("Hz")
    return calib_flux_density*bin_width


def get_real_pwv(pwv, altitude):
    """Given the zenith PWV (reported by APEX) and altitude of source,
    returns the real amount of water between the telescope and space.
    
    Basically returns pwv/cos(zenith_angle)

    :param pwv: zenith PWV reported by APEX
    :param altitude: Altitude of source
    """
    zenith_angle = 90-altitude
    airmass = 1/np.cos(zenith_angle*np.pi/180)
    return pwv*airmass


def jy_to_wm2(flux, wavelength, width):
    conv_fact = 1e-26*units.watt/units.m**2/units.Hz/units.jansky
    flux = conv_fact * flux * width / wavelength

    return flux.to("W/(m^2)")


def wm2_to_jy(flux, wavelength, width):
    conv_fact = 1e-26*units.watt/units.m**2/units.Hz/units.jansky
    flux = flux * wavelength / width / conv_fact

    return flux.to("Jy")
