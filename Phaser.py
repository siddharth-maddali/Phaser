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

from logzero import logger

# plugin modules
import Core, RecipeParser
import ER, HIO, SF
#import GAFFT

class Phaser( 
        Core.Mixin,             # core CPU algorithms and routines
        RecipeParser.Mixin,     # for handling recipe strings
        ER.Mixin,               # error reduction
        HIO.Mixin,              # hybrid input/output
        SF.Mixin                # solvent flipping
    ):

    def __init__( self,
            modulus,
            support,
            beta=0.9, 
            binning=1,              # for high-energy CDI. Set to 1 for regular phase retrieval.
            gpu=False,
            version='legacy',       # ...or 'diffgeom'. If 'diffgeom', uses diffraction geometry to obtain real-space object on orthogonal grid.
            Brecip=None,            # should be provided if 'diffgeom'. Obtained from the ExperimentalGeometry module.
            random_start=True 
            ):
        
        self._myfftn = fftn         # usual 3D FFT propagators in the case of 
        self._myifftn = ifftn       # geometry-agnostic phase retrieval

        if version not in [ 'legacy', 'diffgeom' ]:
            logger.warning( 'Unrecognized version. Reverting to legacy behavior. ' )

        self._modulus           = fftshift( modulus )
        self._support           = fftshift( support )
        self._beta              = beta
        
#        if version=='diffgeom' and Brecip==None:
#            logzero.error( 'Need to provide matrix Brecip in the detector frame.' )
#            return
#        else:
#            self._Bq = Brecip
#            self.setupDiffractionGeometry() # overwrites handles for some Core routines.

#        self._modulus_sum       = modulus.sum()
        self._support_comp      = 1. - support
        if random_start:
            self._cImage            = np.exp( 2.j * np.pi * np.random.rand( 
                                    binning*self._modulus.shape[0], 
                                    binning*self._modulus.shape[1], 
                                            self._modulus.shape[2]
                                    ) ) * self._support
        else:
            self._cImage            = 1. * support
       
        self._cachedImage       = np.zeros( self._cImage.shape ).astype( complex )
        self._cImage_fft_mod = np.absolute( self._myfftn( self._cImage ) )

        self._error             = []
        self._UpdateError()
        self.generateAlgoDict()
#        if version=='diffgeom': # use diffraction geometry
#            self._setupDiffractionGeometry( Brecip )


        if gpu==True:
            self.gpusolver = accelerator.Solver( self.generateGPUPackage() )

        return

