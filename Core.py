##############################################################
#
#	Core:
#           Contains all the core routines and algorithm 
#           implementations for CPU.
#
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    Oct 2019
#           6xlq96aeq@relay.firefox.com
#
##############################################################

import collections

import numpy as np
import functools as ftools

try: 
    from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
except:
    from numpy.fft import fftshift, fftn, ifftn

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label


import PostProcessing as post

class Mixin:

# Writer function to manually update support
    def UpdateSupport( self, support ):
        self._support = support
        self._support_comp = 1. - self._support
        return

# Writer function to manually reset image
    def ImageRestart( self, cImg, fSup, reset_error=True ):
        self._cImage = fftshift( cImg )
        self._support = fftshift( fSup )
        if reset_error:
            self._error = []
        return

# Reader function for the final computed modulus
    def Modulus( self ):
        return np.absolute( fftshift( fftn( self._cImage ) ) )

# Reader function for the error metric
    def Error( self ):
        return self._error

# Updating the error metric
    def _UpdateError( self ):
#        self._error += [ 
#            ( 
#                ( self._cImage_fft_mod - self._modulus )**2 * self._modulus 
#            ).sum() / self._modulus_sum 
#        ]
        self._error += [ 
            ( 
                ( self._cImage_fft_mod - self._modulus )**2
            ).sum()
        ]
        return

# The projection operator into the modulus space of the FFT.
# This is a highly nonlinear operator.
    def _ModProject( self ):
        self._cImage = ifftn( 
            self._modulus * np.exp( 1j * np.angle( fftn( self._cImage ) ) ) 
        )
        return

# Projection operator into the support space.
# This is a linear operator.
    def _SupProject( self ):
        self._cImage *= self._support
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

# Update the inferred signal modulus
    def _UpdateMod( self ):
        self._cImage_fft_mod = np.absolute( fftn( self._cImage ) )
        return

# cache current real-space solution (used in HIO)
    def _CacheImage( self ):
        self._cachedImage = self._cImage.copy() 
        return

# update step used in CPU HIO
    def _UpdateHIOStep( self ):
        self._cImage = ( self._support * self._cImage ) +\
            self._support_comp * ( self._cachedImage - self._beta * self._cImage )
        return

# CPU-specific shrinkwrap implementation
    def Shrinkwrap( self, sigma, thresh ):
        result = gaussian_filter( 
            np.absolute( self._cImage ), 
            sigma, mode='constant', cval=0.
        )
        self._support = ( result > thresh*result.max() ).astype( float )
        self._support_comp = 1. - self._support
        return

# The alignment operator that centers the object after phase retrieval.
    def Retrieve( self ):
        self.finalImage = self._cImage
        self.finalSupport = self._support
        self.finalImage, self.finalSupport = post.centerObject( 
            self.finalImage, self.finalSupport
        )
        return

# Generates a package for the GPU module to read and generate tensors.
    def generateGPUPackage( self, pcc=False, pcc_params=None ):
        mydict = { 
            'array_shape':self._support.shape,
            'modulus':self._modulus, 
            'support':self._support, 
            'beta':self._beta, 
            'cImage':self._cImage,
            'pcc':pcc, 
            'pcc_params':pcc_params
        }
        return mydict


    def _initializeSupport( self, sigma=0.575 ):
        temp = np.log10( np.absolute( fftshift( fftn( self._modulus ) ) ) )
        mask = ( temp > sigma*temp.max() ).astype( float )
        labeled, features = label( mask )
        support_label = list( dict( sorted( collections.Counter( labeled.ravel() ).items(), key=lambda item:-item[1] ) ).keys() )[1]
        self._support = np.zeros( self._arraySize )
        self._support[ np.where( labeled==support_label ) ] = 1.
        self._support = fftshift( self._support )
#        self.BinaryErosion( 1 )
        return


