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


class Phaser:

    def __init__( self,
            modulus,
            support,
            beta=0.9, 
            binning=1,      # for high-energy CDI. Set to 1 for regular phase retrieval.
            random_start=True, 
            gpu=False,
            outlog=''       # matters only for GPU
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

        if gpu==True:
            self.gpusolver = accelerator.Solver( 
                self.generateGPUPackage(),
                outlog=outlog
            )

# Writer function to manually update support
    def UpdateSupport( self, support ):
        self._support = support
        self._support_comp = 1. - self._support
        return

# Writer function to manualy reset image
    def ImageRestart( self, cImg, reset_error=True ):
        self._cImage = cImg
        if reset_error:
            self._error = []
        return

## Reader function for the retrieved image
#    def finalImage( self ):
#        return self._cImage
#
## Reader function for estimated support
#    def finalSupport( self ):
#        return self._support

# Reader function for the final computed modulus
    def Modulus( self ):
#        return np.absolute( fftshift( fftn( fftshift( self._cImage ) ) ) )
        return np.absolute( fftshift( fftn( self._cImage ) ) )

# Reader function for the error metric
    def Error( self ):
        return self._error

# Now, defining the phase retrieval algorithms based on the object metadata.

# Updating the error metric
    def _UpdateError( self ):
        self._error += [ 
            ( 
                ( self._cImage_fft_mod - self._modulus )**2 * self._modulus 
            ).sum() / self._modulus_sum 
        ]
        return

# Error reduction algorithm
    def ErrorReduction( self, num_iterations ):
        for i in tqdm( list( range( num_iterations ) ), desc=' ER' ):
            self._ModProject()
            self._cImage *= self._support
            self._cImage_fft_mod = np.absolute( fftn( self._cImage ) )
            self._UpdateError()
        return

# Hybrid input/output algorithm
    def HybridIO( self, num_iterations ):
        for i in tqdm( list( range( num_iterations ) ), desc='HIO' ):
            origImage = self._cImage.copy() 
            self._ModProject()
            self._cImage = ( self._support * self._cImage ) +\
                self._support_comp * ( origImage - self._beta * self._cImage )
            self._UpdateError()
        return

# Solvent flipping algorithm
    def SolventFlipping( self, num_iterations ):
        for i in tqdm( list( range( num_iterations ) ), desc=' SF' ):
            self._ModHatProject()
            self._ModProject()
            self._UpdateError()
        return

# Basic shrinkwrap with gaussian blurring
    def ShrinkWrap( self, sigma, thresh ):
        result = gaussian_filter( 
            np.absolute( self._cImage ), 
            sigma, mode='constant', cval=0.
        )
        self._support = ( result > thresh*result.max() ).astype( float )
        self._support_comp = 1. - self._support
        return

# The projection operator into the modulus space of the FFT.
# This is a highly nonlinear operator.
    def _ModProject( self ):
        self._cImage = ifftn( 
            self._modulus * np.exp( 1j * np.angle( fftn( self._cImage ) ) ) 
        )
        return

# The reflection operator in the plane of the (linear)
# support operator. This operator is also linear.
    def _SupReflect( self ):
        self._cImage = 2.*( self._support * self._cImage ) - self._cImage
        return

# The projection operator into the 'mirror image' of the
# ModProject operator in the plane of the support projection
# operator. The involvement of the ModProject operator
# makes this also a highly nonlinear operator.
    def _ModHatProject( self ):
        self._SupReflect()
        self._ModProject()
        self._SupReflect()
        return

# The alignment oeprator that centers the object after phase retrieval.
    def Retrieve( self ):
        self.finalImage = self._cImage
        self.finalSupport = self._support
        self.finalImage, self.finalSupport = post.centerObject( 
            self.finalImage, self.finalSupport
        )
        return

# Generates a package for the GPU module to read and generate tensors.
    def generateGPUPackage( self ):
        mydict = { 
            'modulus':self._modulus, 
            'support':self._support, 
            'beta':self._beta, 
            'cImage':self._cImage
        }
        return mydict

