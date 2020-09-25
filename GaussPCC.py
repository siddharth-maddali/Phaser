##############################################################
#
#    GaussPCC: 
#        Simple partial coherence correction module 
#        which assumes a Gaussian partial coherence 
#        function in 3D.
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        Oct 2019
#        smaddali@alumni.cmu.edu
#
##############################################################

import numpy as np
import tensorflow as tf

try: 
    from pyfftw.interfaces.numpy_fft import fftshift
except:
    from numpy.fft import 

class Mixin: 

    def setUpPCC( self, gpack ):
        x, y, z = np.meshgrid( *[ np.arange( -n//2., n//2. ) in gpack[ 'support' ].shape ] )
        pts = np.concatenate( tuple( this.reshape( 1, -1 ) for this in [ x, y, z ] ), axis=0 )
        if 'initial_guess' not in gpack.keys():
            l1p, l2p, l3p, psip, thetap, phip = 2., 2., 2., 0., 0., 0.
        else:
            l1p, l2p, l3p, psip, thetap, phip = tuple( vardict[ 'initial_guess' ] )

        # tf constants
        self._intensity = tf.constant( gpack[ 'modulus' ]**2, dtype=tf.
        self._q = tf.constant( pts, dtype=tf.float32 )
        self._v1, self._v2, self._v3 = tuple( 
            tf.constant( np.roll( np.array( [ 1., 0., 0. ] ).reshape( -1, 1 ), shift=n, axis=0 ), dtype=tf.float32 ) 
            for n in [ 0, 1, 2 ] 
        )

        self._nskew0 = tf.constant( np.array( [ [ 0., 0., 0. ], [ 0., 0., -1. ], [ 0., 1., 0. ] ] ), dtype=tf.float32 )
        self._nskew1 = tf.constant( np.array( [ [ 0., 0., 1. ], [ 0., 0., 0. ], [ -1., 0., 0. ] ] ), dtype=tf.float32 )
        self._nskew2 = tf.constant( np.array( [ [ 0., -1., 0. ], [ 1., 0., 0. ], [ 0., 0., 0. ] ] ), dtype=tf.float32 )
        self._I = tf.eye( 3 )

        # tf variables
        self._l1, self._l2, self._l3, self._psi, self._theta, self._phi = tuple( 
            tf.Variable( this, dtype=tf.float32 ) for this in [ l1p, l2p, l3p, psip, thetap, phip ] 
        )
        self._mD = tf.diag( [ self._l1, self._l2, self._l3 ] )
        self._n0 = tf.sin( self._theta ) * tf.cos( self._phi )
        self._n1 = tf.sin( self._theta ) * tf.sin( self._phi )
        self._n2 = tf.cos( self._theta )
        self._n elf._n0*self._v0 + self._n1*self._v1 + self._n2*self._v2
        self._nskew = self._n0*self._nskew0 + self._n1*self._nskew1 + self._n2*self._nskew2
        self._R = tf.cos( self._psi )*self._I +\
            tf.sin( self._psi )*self._nskew +\
            ( 1. - tf.cos( self._psi ) )*tf.matmul( self._n, tf.transpose( self._n ) )
        self._C = tf.matmul( self._R, tf.matmul( tf.matmul( self._mD, self._mD ), tf.transpose( self._R ) ) )

        self._blurKernel = tf.reshape( 
            tf.exp( -0.5 * tf.reduce_sum( self._q * tf.matmul( self._C, self._q ), axis=0 ) ), 
            shape=self._coherentEstimate.shape
        ) * ( self._l1 * self._l2 * self._l3 ) / ( 2. * np.pi )
        self._pkfft = tf.Variable( np.zeros( self._probSize ), dtype=tf.complex64, name='pkfft' )


        return



