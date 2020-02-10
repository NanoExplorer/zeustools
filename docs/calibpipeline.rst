The ZEUS-2 Calibration Pipeline
===============================

When you install the pip module, you automatically get the calibration_pipeline script available for use. You can either run it directly with

``$ calibration_pipeline``

which will automatically look for the configuration file ``settings.ini``, load it, and reduce the data described by it, or with

``$ calibration_pipeline custom_ini_file.ini``

so that you can specify the exact configuration file to use.

The Process
-----------

First we calibrate each reduction from Bo's pipeline, multiplying by the flux of the flat file and dividing by atmospheric transmission. This value is also divided by the telescope efficiencies. 
Then we shift each reduction to match the line pixels, and finally we do a weighted average of the spectra.

Configuration
-------------

The calibration pipeline requires a settings file to tell it what to do. This is in the form of a `.INI` file, which is a plain text file. Each line in this file can be one of three things:

1. A section declaration, like ``[REDUCTION1]``. This starts a new section, and any variables declared after the heading belong to that section (until the next section heading)

2. A variable declaration, like ``pwv = 0.3``. Everything before the equals sign is the variable name, and everything after is the variable content. In this program it should be pretty clear which variables should be numbers and which should be text.

3. A comment, like ``# The variable "pwv" specifies the precipitable water vapor during the observation``

The ``[GLOBAL]`` section
~~~~~~~~~~~~~~~~~~~~~~~~

This section defines variables that don't change while observing one source. Below are the required variables that need to be defined, example values, and descriptions of each.

lambda_line_px = 205.564
    Wavelength in microns of observed spectral line

px_delta_lambda = 0.17
    Spectral width of a pixel in microns. I usually determine it from the calculator spreadsheet by parking two adjacent pixels at the same wavelength, but there should be a better way.

spat_pos = 7  
    Spatial position to extract from Bo's reduction file. Usually 7 for 200 micron observations, 2 for 450 micron observations, and 1 for 350 micron observations.

pt_src_coupling = 0.33
    During calibration, the flux value is divided by pt_src_coupling,

telescope_efficiency = 1
    The flux value is also divided by telescope efficiency.

cabin_temp = 288
    Temperature in K of the cabin

sky_temp = 270
    Temperature in K of the sky. This and the cabin temp value are used to calibrate Jy/sr of the flat field.

beam_size_steradian = 9.588e-10
    Used to convert from Kelvins (brightness temperature, converts to Jy/sr) to just Janskies.
    This beam size is what I used for the 200 um array. It represents our 6" beam size.
    I calculated it using 2*pi*ln(2)*(6"/2)^2 but I'm not completely confident that's right.
    I think the ln(2) has something to do with Airy

do_contsub = false
    Whether or not to apply continuum subtraction. This will (weighted) average all the 
    non-line-pixels and subtract a constant from all pixels. 
    Let me know if you would enjoy seeing an option for linear continuum subtraction.

where_is_line_flux = 0,1
    defines pixels relative to the spec_pos_of_line (below) where you are finding line flux
    used to decide which pixels to ignore when adding up the continuum.

do_calib = true
   if do calib is false, then we return a spectrum in flat fraction. 
   If this is false we ignore most values like efficiency, temperatures, beam size, lambda, and atm trans.
   Turning this off works best with exactly 1 reduction, because averaging unscaled data is confusing.

unflatten = false
   if unflatten is true, then we return a spectrum in data numbers. 
   If this is true we ignore the value of do_calib, and 
   the program will ignore all the things it ignores if do_calib is false.
   Best used with exactly 1 reduction, because averaging raw data numbers is meaningless

output_filename = ngc4945mosaic_50_7
   We produce a .csv file containing the spectrum for Gordon to mess around with.
   We also produce a plot and a copy of this .ini file for posterity.
   Now we also produce .png files with rough spectra.

The reduction sections
~~~~~~~~~~~~~~~~~~~~~~

Now we ask you to define each observing chunk. You'll need the name of the file that
Bo's pipeline created (usually ends in ``_final_spec.npz``), the atmospheric transmission at the time,
and which spectral position the line was placed on. 
For each observing chunk you can put whatever you want in the brackets besides ``[GLOBAL]``.
and you can have as many sections like this as you need. 
for example, you can add a ``[REDUCTION3]`` section or a section called ``[NGC4945dec02]``.

path = path_to_final_spec.npz
    The location of the beam that you want to process.

atm_transmission = 0.097
    TODO: take in pwv and altitude and query the APEX atmosphere calculator
    MEGA-TODO: put all that info into the .hk file or something

spec_pos_of_line = 6
    Which spectral position is the line in for this reduction?

min_spec_px=0
    define the range of pixels you want to include in the plot. 
    Useful to avoid atmospheric features.

max_spec_px=15
    same as above

The following sections are only needed if you want to read in individual beams, and the exact details are not yet worked out:

use_beamspec = True
    Turn this to True if we are using the beam spec file reated by bo's program.
    If this is true, you need the following two values also:

beamspec_array_number = 1
    If use_beamspec is True,
    ask bo which arr_# you need to be using. This refers to the set of 3 arrs 
    arr_0 is always the array map,
    arr_1-3 might be only after dead pixel subtraction (array_number=0)
    arr_4-6 might be desnaking and dead pixel, etc. (array_number =1)

beam_number = 25
    If use_beamspec is True, this tells us which beam to extract.