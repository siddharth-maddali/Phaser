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

import PostProcessing as post

class Mixin: 

    def ImportCore( self, varDict ):
        self._modulus = tf.constant( varDict[ 'modulus' ], dtype=tf.complex64 )
        self._support = tf.Variable( varDict[ 'support' ], dtype=tf.complex64 )
        self._support_comp = tf.Variable( 1. - varDict[ 'support' ], dtype=tf.complex64 )
        self._beta = tf.constant( varDict[ 'beta' ], dtype=tf.complex64 )
        self._cImage = tf.Variable( varDict[ 'cImage' ], dtype=tf.complex64 )
        self._cachedImage = tf.Variable( np.zeros( varDict[ 'cImage' ].shape ), dtype=tf.complex64  )

        self._modulus_sum = tf.reduce_sum( self._modulus )
        self._cImage_fft_mod = tf.Variable( tf.abs( tf.signal.fft3d( self._cImage ) ) )

        self._error = []
        self._UpdateError()

        x, y, z = np.meshgrid( *[ np.linspace( -n//2., n//2. ) for n in varDict[ 'support' ].shape ] )
        self._rsquared = tf.constant( 
            ftools.reduce( lambda a, b: a+b, [ this**2 for this in [ x, y, z ] ] ), 
            dtype=tf.complex64
        )
              # used for GPU shrinkwrap
        return

    def _UpdateError( self ):
        self._error.append( 
            tf.reduce_sum(
                ( self._cImage_fft_mod - tf.abs( self._modulus ) )**2
            ).numpy()
        )
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
        self._cImage.assign( 
            ( self._support * self._cImage ) +\
            self._support_comp * ( self._cachedImage - self._beta * self._cImage )
        )
        return

# GPU-specific shrinkwrap routine
    def Shrinkwrap( self, sigma, thresh ):
        kernel = 1. / ( sigma * np.sqrt( 2. * np.pi ) ) * tf.exp( -0.5 * self._rsquared / ( sigma**2 ) )
        kernel_ft = tf.signal.fft3d( kernel )
        ampl_ft = tf.signal.fft3d( tf.cast( tf.abs( self._cImage ), dtype=tf.complex64 ) )
        blurred = tf.signal.fftshift( tf.abs( tf.signal.ifft3d( kernel_ft * ampl_ft ) ) )
        new_support = tf.where( blurred > thresh * tf.reduce_max( blurred ), 1., 0. )
        self.UpdateSupport( new_support )
        return

    def _ModProject( self ):
        self._cImage.assign( 
            tf.signal.ifft3d( self._modulus * tf.exp( 1.j * tf.cast( 
                    tf.math.angle( tf.signal.fft3d( self._cImage ) ), 
                    dtype=tf.complex64 
                ) 
            ) )
        )
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
        return
