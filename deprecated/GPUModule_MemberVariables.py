##############################################################
#
#	GPUModule_MemberVariables:
#	    Contains a mixin for defining Tensorflow variables
#	    within the GPUModule.Solver class.
#	
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except: 
    import tensorflow as tf

class Mixin:

    def defineMemberVariables( self, varDict ):

        # Array coordinates
        x, y, z = np.meshgrid( 
            list( range( varDict[ 'cImage' ].shape[0] ) ),
            list( range( varDict[ 'cImage' ].shape[1] ) ),
            list( range( varDict[ 'cImage' ].shape[2] ) )
        )
        y = y.max() - y

        x = x - x.mean()
        y = y - y.mean()
        z = z - z.mean()

        # Tensorflow variables specified here.
        with tf.device( '/gpu:0' ):
            self._modulus = tf.constant( varDict[ 'modulus' ], dtype=tf.complex64, name='mod_measured' )
            self._support = tf.Variable( varDict[ 'support' ], dtype=tf.complex64, name='sup' )
            self._support_comp = tf.Variable( 1. - varDict[ 'support' ], dtype=tf.complex64, name='Support_comp' )
            self._cImage = tf.Variable( varDict[ 'cImage' ], dtype=tf.complex64, name='Image' )
            self._buffImage = tf.Variable( varDict[ 'cImage' ], dtype=tf.complex64, name='buffImage' )
            self._beta = tf.constant( varDict[ 'beta' ], dtype=tf.complex64, name='beta' )
            self._probSize = self._cImage.shape
            self._thresh = tf.placeholder( dtype=tf.float32, name='thresh' )
            self._sigma = tf.placeholder( dtype=tf.float32, name='sigma' )
            self._neg = tf.constant( -1., dtype=tf.float32 )
            self._two = tf.constant(  2., dtype=tf.float32 )

            # shrinkwrap-specific variables
            self._x = tf.constant( x, dtype=tf.float32, name='x' )
            self._y = tf.constant( y, dtype=tf.float32, name='y' )
            self._z = tf.constant( z, dtype=tf.float32, name='z' )
            self._kernelFFT = tf.Variable( varDict[ 'support' ], dtype=tf.complex64, name='kernelFFT' )
            self._blurred = tf.Variable( varDict[ 'support' ], dtype=tf.float32, name='blurred' )
            self._dist = tf.Variable( tf.zeros( self._x.shape, dtype=tf.float32 ), name='dist' )
            self._intermedFFT = tf.Variable( tf.zeros( self._cImage.shape, dtype=tf.complex64 ), name='intermedFFT' )

            # This is executed only if high-energy phasing is required.
            if 'bin_left' in varDict.keys():
                bL = varDict[ 'bin_left' ]
                sh = bL.shape
                self._binL = tf.constant( 
                    bL.reshape( sh[0], sh[1], 1 ).repeat( varDict[ 'modulus' ].shape[-1], axis=2 ), 
                    dtype=tf.complex64, 
                    name='binL'
                )
                self._binR = tf.constant( 
                    bL.T.reshape( sh[1], sh[0], 1 ).repeat( varDict[ 'modulus' ].shape[-1], axis=2 ), 
                    dtype=tf.complex64, 
                    name='binR'
                )
                self._scale = tf.constant( varDict[ 'scale' ], dtype=tf.complex64, name='scale' )
                self._binned = tf.Variable( tf.zeros( self._modulus.shape, dtype=tf.complex64 ), name='binned' )
                self._expanded = tf.Variable( tf.zeros( self._support.shape, dtype=tf.complex64 ), name='expanded' )
                self._scaled = tf.Variable( tf.zeros( self._modulus.shape, dtype=tf.complex64 ), name='scaled' )

        return
