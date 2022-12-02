# This grating calibration duplicates
# the functionality of the 2016
# grating calculator spreadsheet
# but has extra features.
# warning: lots of hardcoded parameters ahead.
# flake8: noqa

import numpy as np
import zeustools as zt

class GratingCalibrator():
    """ This class contains all the information needed to compute
    grating indices and wavelengths. This is mostly Carl's code, and it
    is a duplicate of the functionality of his Excel spreadsheet.
    """
    def __init__(self):
        #---------------------------------------------------------------------------#
        #Initial Parameters - Don't edit
        #---------------------------------------------------------------------------#
        #ZEUSII values
        self.alpha_min= 53.33                #minimum grating (this if the physical limit)
        #alpha_max = 74.4                #maximum grating (this if the physical limit)
        self.alpha_max = 73.9                #maximum grating (this if the physical limit)
        self.Rd = 200.0                      #Stepper Motter steps per revolution of the motor 
                                        #drive shaft moves the grating 1.8 degrees
        self.Rg = 256.0/360.0                #The gear ratio for the grating, 1 revolution (360 degrees of the drive shaft)
                                        #moves the grating 256 degrees 
        self.alpha_adj = 0.0                 #can be used if the grating experienced a shift in 
                                        #the grating angle due to removing/reinstalling the grating. 
        self.index_max = 2830                #corresponds to when the grating is at alpha_min,     
        self.index_min = 0                   #corresponds to when the grating is at alpha_max
                                        #can be updated to reflect preferences in the index
        self.py_shift = 0.0                  #the difference from the fit and on-sky pixel position of a line
                                        #e.g. if the CO6-5 appears on 15 but was predicted to be on 11
                                        #change to -4


        #Using ZEEMAX and the CO gas cell measurements we determine that GI 2354 corresponds to grating angle 56.7
        #as such we determine alpha_min and alpha_max
        # ref_alpha = 56.7
        # ref_gi = 2354

        self.ref_alpha = 53.9796828

        self.ref_gi = 2834

        self.alpha_max_index = self.ref_alpha-(self.index_max-self.ref_gi)/(self.Rd*self.Rg) #angle of the grating when the index is at its maximun
        self.alpha_min_index = self.ref_alpha+self.ref_gi/(self.Rd*self.Rg) #angle of the grating when the index is at its minimun
        #alpha_min_index = 73.2515625
        #--------------------------------------------------------------------------#
        # Grating Calibration Coefficients -  fit by CDF 2015.08*20
        # Based on the CO65 and CO76 gas cell measurements from June 2015 and 450 micron filter transmission test
        # very long wavelength and very short wavelengths should be used cautiously until more calibration data is added
        # Also ZEEMAX outputs were added, with the CO76 line setting the grating angle reference and then the 13CO8-7 line
        # observed at APEX
        #--------------------------------------------------------------------------#
        # coeff = np.array([-6.7267513308e+00,-1.3926784420e-02,1.3353606943e+01,-3.8961499591e+03,5.4591088823e+03,-1.9728510661e+03,
        #                   1.4224075236e-02,-4.4158261382e-01,2.3470622734e+02])
        self.coeffs_400um=[
        0.9800910119,
        -0.0017009051035,
        -0.87654448327,
        36.248043521,
        459.42373214,
        -80.04474108,
        -0.0017003774252,
        -1.5498032937,
        102.04705483]

        self.coeffs_200um = [0.64752068721,
        -0.0001073140822,
        -0.61523295551,
        199.05515763,
        304.94503285,
        -86.845008895,
        0.0077590412166,
        -3.0230170211,
        389.1049963]

        self.coeffs_600um = [0.87857172439,
        -0.00025250113336,
        -0.99083543371,
        29.684331753,
        333.03418821,
        50.056757672,
        0.004374603024,
        -3.8314543603,
        25.683846046]

        self.am = zt.ArrayMapper()

    def phys_to_coeffs(self,spec,array,return_order=False):
        """ Given a location on the array, look up the correct
        set of coefficients for the best-fit grating model. Optionally
        return the grating order that falls onto the location you supplied.

        :param int spec: Spectral Position on the array
        :param int array: Array name (i.e. 400, 200, micron arrays etc)
        """
        arr = zt.array_name(array)
        if arr == 'a':
            # 400 um array
            coeff = self.coeffs_400um
            if spec > 20:
                order = 4
            else:
                order = 5

        elif arr == 'b':
            coeff = self.coeffs_200um
            order = 9

        elif arr == 'c':
            coeff = self.coeffs_600um
            order = 3

        if return_order:
            return coeff,order
        else:
            return coeff

    def phys_px_to_wavelength(self,spec,spat,array,index):
        """ Given a location on the array and a grating index, compute
        the wavelength of light that should be falling on the detector.
        
        Parameters can be numpy arrays or numbers, except for `array`.
        
        :param int spec: Spectral position on array.
        :param int spat: Spatial position on the array.
        :param array: Which array to use
        :param int index: Grating index 
        :return: wavelength of light falling on specified detector(s)
        """
        a = self.alpha(index)
        
        coeff,order = self.phys_to_coeffs(spec,array,return_order=True)

        return self.cal_wavelength(order, a, spat, spec, coeff)

    def phys_wavelength_to_index(self,spec,spat,array,wavelength):
        """ Given a location on the array and a wavelength of light, compute
        the grating index that should be used to obtain the specified wavelength.
        
        Parameters can be numpy arrays or numbers, except for `array`.
        
        :param int spec: Spectral position on array.
        :param int spat: Spatial position on the array.
        :param array: Which array to use
        :param wavelength: Wavelength in microns  
        :return: Grating index 
        """
        coeff = self.phys_to_coeffs(spec,array)
        order = self.wavelength_to_order(wavelength)

        return self.cal_index(order, wavelength, spat, spec, coeff)

    def spat_wavelength_index_to_spec(self,spat,array,wavelength,index):
        """ Given a spatial position on the array, a grating index, and a 
        wavelength, compute the spectral position that should see
        the specified wavelength.
        
        Parameters can be numpy arrays or numbers, except for `array`.
        
        :param int spat: Spatial position on the array.
        :param array: Which array to use
        :param wavelength: Wavelength in microns
        :param int index: Grating index 
        :return: Spectral position seeing the specified wavelength.
        """
        order = self.wavelength_to_order(wavelength)
        coeff = self.order_to_coeffs(order)

        return self.cal_px(order,spat,wavelength,self.alpha(index),coeff)

    #------------------------------------------------------------------------------#
    #       Basica Functions that relate grating index (Ig) and angle (alpha)      #
    #------------------------------------------------------------------------------#

    #function which defines the grating angle alpha as a function of the grating or 
    #   stepper motor index Ig
    def alpha(self,Ig):
        alpha = self.alpha_min_index-Ig/(self.Rd*self.Rg)
        return alpha
        
    def index(self,alpha):
        index = -(self.Rg*self.Rd)*(alpha-self.alpha_min_index)
        return index

    #function which defines the order of the grating given the wavelength being observed
    #   additional logic will need to be added here to deal with the shortwavelength array

    def wavelength_to_order(self,wavelength):
        if wavelength < 250:
            return 9
        elif wavelength > 500:
            return 3
        elif wavelength < 395:
            return 5
        elif wavelength > 395:
            return 4

    def order_to_coeffs(self,n):
        n = int(n)
        if n == 9:
            return self.coeffs_200um
        elif n == 4 or n == 5:
            return self.coeffs_400um
        elif n==3:
            return self.coeffs_600um
        else:
            raise ValueError("invalid grating order")

    def wavelength_to_coeffs(self,wavelength):
        order = self.wavelength_to_order(wavelength)
        return self.order_to_coeffs(order)
            



    #will calculate a wavelength at a given pixel (py,px), when given an order n,
    #grating angle alpha, and calibration coefficient array
    def cal_wavelength(self,n,alpha,py,px,coeff):
        n = float(n)
        py = float(py)+float(self.py_shift)
        px = float(px)
        alpha =float(alpha)
        sina = np.sin(np.pi/180*alpha)
        px_quadratic = (coeff[6]*px**2+coeff[7]*px+coeff[8])
        wavelength =  5/n*(coeff[0]*(py+px_quadratic)*sina
                        +coeff[1]*(py+px_quadratic)**2
                        +coeff[2]*(py+px_quadratic)
                        +coeff[3]
                        +coeff[4]*sina
                        +coeff[5]*sina**2) 
        return round(wavelength,5)    

    #will calculate a grating index to place specific wavelength in grating order n
    # on a pixel (py,px), given a calibration coefficient array    
    def cal_index(self,n,wavelength,py,px,coeff):
        py = float(py)+float(self.py_shift)
        px = float(px)
        n = float(n)
        wavelength = float(wavelength)
        px_quadratic = (coeff[6]*px**2+coeff[7]*px+coeff[8])
        
        a= coeff[5]
        b= coeff[0]*(py+px_quadratic)+coeff[4]
        c= (coeff[1]*(py+px_quadratic)**2
            +coeff[2]*(py+px_quadratic)
            +coeff[3])-wavelength*n/5
        # sina = x where ax^2 + bx + c = 0
        
        alpha = np.arcsin((-b+np.sqrt(b**2-4*a*c))/(2*a))*180/np.pi
        # print( a,b,c,alpha)
        cal_index = round(self.index(alpha))
        return round(cal_index,1)

    #will calculate the SPATIAL position of a pixel (py) given an order, SPECTRAL position (px), 
    #wavelength and grating angle,  coefficient array      
    def cal_py(self,n,px,wavelength,alpha,coeff):
        n = float(n)
        wavelength = float(wavelength)
        alpha =float(alpha)
        sina = np.sin(np.pi/180*alpha)
        px_quadratic = (coeff[6]*px**2+coeff[7]*px+coeff[8])
        a = coeff[1]
        b = coeff[0]*sina + coeff[2]
        c = -(wavelength*n/5-(coeff[3]+coeff[4]*sina+coeff[5]*sina**2))
        # print(a,b,c)
        py = (((-b-np.sqrt(b**2-4*a*c))/(2*a)))-px_quadratic
        return round(py,1)-float(self.py_shift)

    #will calculate the SPECTRAL position of a pixel (px) given an order, SPATIAL position (py), 
    #wavelength and grating angle,  coefficient array  
    def cal_px(self,n,py,wavelength,alpha,coeff):
        n = float(n)
        py = float(py)+float(self.py_shift)
        wavelength = float(wavelength)
        alpha =float(alpha)
        sina = np.sin(np.pi/180*alpha)
        a = coeff[1]
        b = coeff[0]*sina + coeff[2]
        c = -(wavelength*n/5-(coeff[3]+coeff[4]*sina+coeff[5]*sina**2))
        ax = coeff[6]
        bx = coeff[7]
        cx = py + coeff[8] - (((-b-np.sqrt(b**2-4*a*c))/(2*a)))
        # print(a,b,c,ax,bx,cx)
        px = (-bx-np.sqrt(bx**2-4*ax*cx))/(2*ax)
        return round(px,1)
    


co87    =  325.225163 #micron
co76    =  371.650429 #micron
co65    =  433.556227 #micron
co65_13 =  453.49765  #micron
co87_13 =  340.181208 #micron
ci      =  370.414364 #micron

oiii52  =  51.8145    #micron
oiii88  =  88.356     #micron
nii122  = 121.898     #micron
nii205  = 205.1782    #micron
cii158  = 157.7409    #micron
oi63    =  63.1837    #micron
oi145   = 145.526     #micron

def wavelength_z(source_redshift,line):
    return (1+source_redshift)*line

def wavelength_v(source_velocity,line):
    return (1+source_velocity/299792)*line