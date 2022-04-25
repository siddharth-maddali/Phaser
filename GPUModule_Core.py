###########################################################
#
#    GPUModule_Core: 
#        Core definitions and operations for Phaser GPU
#
#    Siddharth Maddali
#    Argonne National Laboratory 
#    January 2020
#    6xlq96aeq@relay.firefox.com
#
###########################################################

import tensorflow as tf
import numpy as np
import functools as ftools
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import PostProcessing as post

try:
    from pyfftw.interfaces.numpy_fft import fftshift
except: 
    from numpy.fft import fftshift

class Mixin: 

    def ImportCore( self, varDict ):
        self._modulus = tf.constant( varDict[ 'modulus' ], dtype=tf.complex64 )
        
        if varDict[ 'free_vox_mask' ] is None:
            self.mask = np.ones_like(self._modulus.numpy())
            self.unmask = self.mask
            
        else:
            self.mask = varDict[ 'free_vox_mask' ]
            self.unmask = self.mask == 0
            
        self.mask = tf.Variable(self.mask,dtype=tf.float32)
        self.unmask = tf.Variable(self.unmask,dtype='bool')
        self._support = tf.Variable( varDict[ 'support' ], dtype=tf.complex64 )
        self._support_comp = tf.Variable( 1. - varDict[ 'support' ], dtype=tf.complex64 )
        self._beta = tf.constant( varDict[ 'beta' ], dtype=tf.complex64 )
        self._cImage = tf.Variable( varDict[ 'cImage' ], dtype=tf.complex64 )
        self._cachedImage = tf.Variable( np.zeros( varDict[ 'cImage' ].shape ), dtype=tf.complex64  )

        self._modulus_sum = tf.reduce_sum( self._modulus )
        self._cImage_fft_mod = tf.Variable( tf.abs( tf.signal.fft3d( self._cImage ) ) )
        self.BinaryErosion = self.__GPUErosion__
        self._error = []
        msk = np.where(self._modulus.numpy() > 3,1,0)
        self.nonzero_mask = tf.Variable(msk,dtype=tf.float32)
#         self.fourier_sup = (msk*self.mask.numpy()).sum()
        
        self._UpdateError()


        x, y, z = np.meshgrid( *[ np.arange( -n//2., n//2. ) for n in varDict[ 'support' ].shape ] )
        self._rsquared = tf.constant( 
            ftools.reduce( lambda a, b: a+b, [ fftshift( this )**2 for this in [ x, y, z ] ] ), 
            dtype=tf.complex64
        )
              # used for GPU shrinkwrap

        return

    def _UpdateError( self ):
        self._error.append(tf.reduce_sum((self._cImage_fft_mod - tf.abs( self._modulus ))**2).numpy())
#         self._error.append( tf.reduce_mean((self._cImage_fft_mod - tf.abs( self._modulus ))**2 ,self.nonzero_mask*self.mask) /tf.boolean_mask(tf.abs(self._modulus)**2,self.nonzero_mask*self.mask)).numpy())


        return

    def _UpdateMod( self ):
        self._cImage_fft_mod.assign( tf.abs( tf.signal.fft3d( self._cImage ) ) )

        return

    def UpdateSupport( self, support ):
        self._support.assign( tf.cast( support, dtype=tf.complex64 ) )
        self._support_comp.assign( 1. - self._support )
        return

    def _CacheImage( self ):
        self._cachedImage.assign( self._cImage )
        return

    def _UpdateHIOStep( self ):
        self._cImage.assign(tf.signal.ifft3d( self._modulus * tf.exp( 1.j * tf.cast( 
                    tf.math.angle( tf.signal.fft3d( self._cImage ) ), 
                    dtype=tf.complex64 ) 
            ) ))
#         self._cImage.assign( tf.where(self.unmask,
#             ( self._support * self._cImage ) +\
#             self._support_comp * ( self._cachedImage - self._beta * self._cImage ),self._cImage))
        return
        

# GPU-specific shrinkwrap routine
    def Shrinkwrap( self, sigma, thresh ):
        kernel = 1. / ( sigma * np.sqrt( 2. * np.pi ) ) * tf.exp( -0.5 * self._rsquared / ( sigma**2 ) )
        kernel_ft = tf.signal.fft3d( kernel )
        ampl_ft = tf.signal.fft3d( tf.cast( tf.abs( self._cImage ), dtype=tf.complex64 ) )
        #blurred = tf.signal.fftshift( tf.abs( tf.signal.ifft3d( kernel_ft * ampl_ft ) ) )
        blurred = tf.abs( tf.signal.ifft3d( kernel_ft * ampl_ft ) )
        new_support = tf.where( blurred > thresh * tf.reduce_max( blurred ), 1., 0. )
        self.UpdateSupport( new_support )
        return

    def _ModProject( self ):
        self._cImage.assign(tf.signal.ifft3d( self._modulus * tf.exp( 1.j * tf.cast( 
                    tf.math.angle( tf.signal.fft3d( self._cImage ) ), 
                    dtype=tf.complex64 
                ) 
            ) ))
#         self._cImage.assign( tf.where(self.unmask,
#             tf.signal.ifft3d( self._modulus * tf.exp( 1.j * tf.cast( 
#                     tf.math.angle( tf.signal.fft3d( self._cImage ) ), 
#                     dtype=tf.complex64 
#                 ) 
#             ) ),self._cImage))
        return


    def _SupProject( self ):
        self._cImage.assign( self._cImage * self._support )
        return

    def _SupReflect( self ):
        self._cImage.assign( 
           2. * ( self._support * self._cImage ) - self._cImage 
        )
        return

    def Retrieve( self ):
        

        self.finalImage, self.finalSupport = post.centerObject( 
            self._cImage.numpy(), np.absolute( self._support.numpy() )
        )
        self._cImage = tf.Variable(self.finalImage,dtype=tf.complex64)
        self._support = tf.Variable(self.finalSupport,dtype=tf.complex64)
        
        return


