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
#        6xlq96aeq@relay.firefox.com
#
##############################################################

import numpy as np
import tensorflow as tf

try: 
    from pyfftw.interfaces.numpy_fft import fftshift, fftn, ifftn
except:
    from numpy.fft import fftshift, fftn, ifftn


class Mixin: # inherited by Phaser module

    def _ModProjectPC( self ):
        self._patt = tf.signal.fft3d( tf.cast( tf.abs( tf.signal.fft3d( self._cImage ) )**2, dtype=tf.complex64 ) )
        pcoh_est = tf.sqrt( tf.cast( tf.signal.ifft3d( self._patt * self._kernel_f ), dtype=tf.float32 ) )
        self._cImage.assign( self._cImage * self._modulus / pcoh_est )
        return


class Solver( tf.Module ): 

    def __init__( self, measured_intensity, gpack ):
        self._signal = tf.constant( measured_intensity, dtype=tf.float32 )
        pts, parm_list = self._setupDomain( gpack=gpack )
        self._setupConstants( pts )
        self._setupVariables( parm_list )
        self._setupAuxiliary()
        self._updateBlurKernel()
        #self._setupOptimizer()
        return

    def setCoherentEstimate( self, intensity ):
        self._cohEst_f = tf.constant( fftn( intensity ), dtype=tf.float32 )
        return

    def getBlurKernel( self ):
        self._updateBlurKernel()
        return tf.constant( self._blurKernel_f.numpy(), dtype=tf.complex64 )

#    def _ModProject( self ): # this definition replaces existing definition of _ModProject
#        self._patt = tf.signal.fft3d( tf.cast( tf.abs( tf.signal.fft3d( self._cImage ) )**2, dtype=tf.complex64 ) )
#        self._modulus_estimated = tf.cast( 
#            tf.signal.ifft3d( self._blurKernel_f * self._patt ), 
#            dtype=tf.float32
#        )
#        self._cImage.assign( 
#            self._cImage * tf.sqrt( self._modulus / self._modulus_estimated )
#        )
#        return

    def _setupDomain( self, gpack ):
        x, y, z = tuple( fftshift( this ) for this in np.meshgrid( *[ np.arange( -n//2., n//2. ) for n in gpack[ 'support' ].shape ] ) )
        pts = np.concatenate( tuple( this.reshape( 1, -1 ) for this in [ x, y, z ] ), axis=0 )
        if 'initial_guess' not in gpack.keys():
            #l1p, l2p, l3p, psip, thetap, phip = 2., 2., 2., 0., 0., 0.
            parm_list = 2., 2., 2., 0., 0., 0.
        else:
            parm_list = tuple( vardict[ 'initial_guess' ] )
        return pts, parm_list

    def _setupConstants( self, pts ):
        self._q = tf.constant( pts, dtype=tf.float32 )
        self._v0, self._v1, self._v2 = tuple( 
            tf.constant( np.roll( np.array( [ 1., 0., 0. ] ).reshape( -1, 1 ), shift=n, axis=0 ), dtype=tf.float32 ) 
            for n in [ 0, 1, 2 ] 
        )
        self._nskew0 = tf.constant( np.array( [ [ 0., 0., 0. ], [ 0., 0., -1. ], [ 0., 1., 0. ] ] ), dtype=tf.float32 )
        self._nskew1 = tf.constant( np.array( [ [ 0., 0., 1. ], [ 0., 0., 0. ], [ -1., 0., 0. ] ] ), dtype=tf.float32 )
        self._nskew2 = tf.constant( np.array( [ [ 0., -1., 0. ], [ 1., 0., 0. ], [ 0., 0., 0. ] ] ), dtype=tf.float32 )
        self._I = tf.eye( 3 )
        return

    def _setupAuxiliary( self ):
        self._mD = tf.linalg.diag( [ self._l1, self._l2, self._l3 ] )
        self._n0 = tf.sin( self._theta ) * tf.cos( self._phi )
        self._n1 = tf.sin( self._theta ) * tf.sin( self._phi )
        self._n2 = tf.cos( self._theta )
        self._n  = self._n0*self._v0 + self._n1*self._v1 + self._n2*self._v2
        self._nskew = self._n0*self._nskew0 + self._n1*self._nskew1 + self._n2*self._nskew2
        self._R = tf.cos( self._psi )*self._I + tf.sin( self._psi )*self._nskew + ( 1. - tf.cos( self._psi ) )*tf.matmul( self._n, tf.transpose( self._n ) )
        self._C = tf.matmul( self._R, tf.matmul( tf.matmul( self._mD, self._mD ), tf.transpose( self._R ) ) )
        return


    def _setupVariables( self, parm_list ):
        self._l1, self._l2, self._l3, self._psi, self._theta, self._phi = tuple( 
            tf.Variable( this, dtype=tf.float32 ) for this in parm_list
        )
        return

    @tf.function
    def _updateBlurKernel( self ):
        self._blurKernel = tf.reshape( 
            tf.exp( -0.5 * tf.reduce_sum( self._q * tf.matmul( self._C, self._q ), axis=0 ) ), 
            shape=self._support.shape
        ) * ( self._l1 * self._l2 * self._l3 ) / ( 2. * np.pi )
        self._blurKernel_f = tf.signal.fft3d( tf.cast( self._blurKernel, dtype=tf.complex64 ) )
        return
    
    @tf.function
    def Predict( self ):
        self._updateBlurKernel()
        self._modulus_estimated = tf.sqrt( tf.cast( tf.signal.ifft3d( self._cohEst_f * self._blurKernel_f ), dtype=tf.float32 ) )
        return  

    @tf.function
    def Objective( self ):
        return tf.reduce_sum( 
            ( 
                self._modulus - self._modulus_estimated
            )**2
        )

