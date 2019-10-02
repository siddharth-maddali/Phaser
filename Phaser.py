"""
    Basic Python module for phasing Bragg CDI data

	    Siddharth Maddali
	    Argonne National Laboratory
	    2017-2018
                                                                               
                                                                               
	                                                                               
	                    *****/(/****,***//#%#((*.,**,                              
	                   .*,***,,,,,,,,,,,/(/%%%#%&(#%#(,                        
	           ....    /&&&&&&(*/((((//*,,/#/%%%*/((*/%%%%%%%/                     
	         ,,,******/%&&&&@&&&@@@@@@@@&@&&&&&&&&&&&&%%%%%%%%#(/,//***            
	 .,....,.,,,****,**&&&&&@&&(//#&&@@@@@@@@@@@@@@@@@&&&&&&&&%%,,*,*(((/*..     
	        /(/*//////(&&&&&&&&&&&&&@@@&&&&%##(((/////((((##%@@&&&&&&&##%%%%%#%( 
	        **/*////((%&&&&&&@&&&&&&@@@&&&&&&&&&&&&&&&&&&&&&&@@&&&&&@@&&&&&&%%%@.
	         .,///****&&%##%&&&&&&&&&@@@@@@@@@@@@@@@@&&&&&&&&&&&&&&&&@@@@@@@@@@@@&*
	                  &(/((&&&&&&&&&&&&&&&&&&&&&&&&&&%&&%&%%%%%&&%&&&%&&&&@@@@@&/
	                    .*(%&&&%%%&%&%%%%&%&%%/                             .,,,.  
	                             #%%%%%&&&&%#(,                                    
	                               *((#######((                                    
	                                /(########(.                                   
	                                .(#########/                                   
	                                 /(########(.                                  
	                                 ,(########(*                                  
	                                  ((########(                                  
	                                  ,(########(,                                 
	                                   ((#######((                                 
	                                   *((#%##((.                                
	                                    ((#######(/                                
	                                    /((#######(                                
	                                    .((#######((.                              
	                                     *(########(.                              
	                                      /(((/*,                                  
	                                                                               
"""

import numpy as np
from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
#from numpy.fft import fftshift, fftn, ifftn
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

import PostProcessing as post
import GPUModule as accelerator

# plugin modules
import RecipeParser


class Phaser( 
        Core.Mixin,             # core CPU algorithms and routines
        RecipeParser.Mixin      # for handling recipe strings
    ):

    def __init__( self,
            modulus,
            support,
            beta=0.9, 
            binning=1,          # for high-energy CDI. Set to 1 for regular phase retrieval.
            parallel=False,
            gpu=False,
            outlog='',          # matters only for GPU
            random_start=True 
            ):
        self._modulus           = fftshift( modulus )
        self._support           = support
        self._beta              = beta

        self._modulus_sum       = modulus.sum()
        self._support_comp      = 1. - support
        if random_start:
            self._cImage            = np.exp( 2.j * np.pi * np.random.rand( 
                                    binning*self._modulus.shape[0], 
                                    binning*self._modulus.shape[1], 
                                            self._modulus.shape[2]
                                    ) ) * self._support
        else:
            self._cImage            = 1. * support
        
        self._cImage_fft_mod = np.absolute( fftn( self._cImage ) )

        self._error             = []
        self._UpdateError()
        self.generateAlgoDict()

        if gpu==True:
            self.gpusolver = accelerator.Solver( self.generateGPUPackage(), outlog=outlog )

        return

