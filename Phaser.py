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
import GPUModule as accelerator

try:
    from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
except: 
    from numpy.fft import fftshift, fftn, ifftn

# plugin modules
import Core, RecipeParser
import ER, HIO, SF
import GaussPCC, Morphology

class Phaser( 
        Morphology.Mixin,       # Routines to manipulate object support
        GaussPCC.Mixin,         # New ER accounting for partial coherence
        Core.Mixin,             # core CPU algorithms and routines
        RecipeParser.Mixin,     # for handling recipe strings
        ER.Mixin,               # error reduction
        HIO.Mixin,              # hybrid input/output
        SF.Mixin                # solvent flipping
    ):

    def __init__( self,
            modulus,
            beta=0.9, 
            binning=1,      # for high-energy CDI. Set to 1 for regular phase retrieval.
            gpu=False,
            pcc=False,
            random_start=True 
            ):
        self._modulus           = fftshift( modulus )
        self._arraySize         = tuple( this*shp for this, shp in zip( [ binning, binning, 1 ], self._modulus.shape ) )
        self._initializeSupport()
        self._support_comp      = 1. - self._support
        self._beta              = beta

        if random_start:
            self._cImage            = np.exp( 2.j * np.pi * np.random.random_sample( self._arraySize ) ) * self._support
        else:
            self._cImage            = 1. * self._support

       
        self._cachedImage       = np.zeros( self._cImage.shape ).astype( complex )
        self._cImage_fft_mod = np.absolute( fftn( self._cImage ) )

        self._error             = []
        self.BinaryErosion = self.BinaryErosionCPU
        self._UpdateError()
        self.generateAlgoDict()

        if gpu==True:
            gpack = self.generateGPUPackage( pcc=pcc )
            self.gpusolver = accelerator.Solver( gpack )

        #if pcc==True:
        #    self._pccSolver = PCSolver( np.absolute( self._modulus )**2, gpack )
        #    self._kernel_f = self._pccSolver.getBlurKernel()
        #    self._ModProject = self._ModProjectPC

        return

