##############################################################
#
#	GPUModule_GaussianPCF:
#	    Contains a mixin for partial coherence methods 
#	    within the GPUModule.Solver class. Models the 
#           partial coherence function as a 3D multivariate 
#           Gaussian.
#	
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

import numpy as np
import tensorflow as tf
from numpy.fft import fftshift
#from tqdm import tnrange


class Mixin:

    def initializeGaussianPCF( self, vardict, array_shape, gpu, learning_rate=1.e-1 ):

        x, y, z = np.meshgrid( 
            list( range( array_shape[0] ) ),
            list( range( array_shape[1] ) ),
            list( range( array_shape[2] ) )
        )
        y = y.max() - y

        x = ( fftshift( x - x.mean() ) ).reshape( 1, -1 )
        y = ( fftshift( y - y.mean() ) ).reshape( 1, -1 )
        z = ( fftshift( z - z.mean() ) ).reshape( 1, -1 )
        pts = np.concatenate( ( x, y, z ), axis=0 )
    
        if 'initial_guess' not in vardict.keys():
            l1p, l2p, l3p, psip, thetap, phip = 2., 2., 2., 0., 0., 0.
        else:
            l1p, l2p, l3p, psip, thetap, phip = tuple( vardict[ 'initial_guess' ] )

        with tf.device( '/gpu:%d'%gpu ):
            with tf.name_scope( 'GaussianPCF' ):
                self._roll = [ n // 2 for n in self._probSize ] # defined in GPUModule_Base

                with tf.name_scope( 'Constants' ):
                    self._coherentEstimate = tf.Variable( tf.zeros( self._cImage.shape, dtype=tf.float32 ), name='coherentEstimate' )
                    self._intensity = tf.constant( vardict[ 'modulus' ]**2, dtype=tf.float32, name='Measurement' )
                    self._q = tf.constant( pts, dtype=tf.float32, name='domainPoints' )
                    self._v0 = tf.constant( np.array( [ 1., 0., 0. ] ).reshape( -1, 1 ), dtype=tf.float32 )
                    self._v1 = tf.constant( np.array( [ 0., 1., 0. ] ).reshape( -1, 1 ), dtype=tf.float32 )
                    self._v2 = tf.constant( np.array( [ 0., 0., 1. ] ).reshape( -1, 1 ), dtype=tf.float32 )
                    self._nskew0 = tf.constant( np.array( [ [ 0., 0., 0. ], [ 0., 0., -1. ], [ 0., 1., 0. ] ] ), dtype=tf.float32 )
                    self._nskew1 = tf.constant( np.array( [ [ 0., 0., 1. ], [ 0., 0., 0. ], [ -1., 0., 0. ] ] ), dtype=tf.float32 )
                    self._nskew2 = tf.constant( np.array( [ [ 0., -1., 0. ], [ 1., 0., 0. ], [ 0., 0., 0. ] ] ), dtype=tf.float32 )
                    self._one = tf.constant( 1., dtype=tf.float32 )
                    self._neg = tf.constant( -0.5, dtype=tf.float32 )
                    self._twopi = tf.constant( ( 2 * np.pi )**( 3./2. ), dtype=tf.float32 )
                    self._I = tf.eye( 3 )

                with tf.name_scope( 'Parameters' ):
                    self._l1 = tf.Variable( l1p, dtype=tf.float32, name='Lambda1' )                 #
                    self._l2 = tf.Variable( l2p, dtype=tf.float32, name='Lambda2' )                 # Sqrt of eigenvalues of covariance matrix
                    self._l3 = tf.Variable( l3p, dtype=tf.float32, name='Lambda3' )                 #
                    self._psi = tf.Variable( psip, dtype=tf.float32, name='Psi' )                   # Rotation angle of eigenbasis
                    self._theta = tf.Variable( thetap, dtype=tf.float32, name='Theta' )             # Polar angle of rotation axis
                    self._phi = tf.Variable( phip, dtype=tf.float32, name='Phi' )                   # Azimuth angle of rotation axis

                with tf.name_scope( 'Auxiliary' ):
                    self._FreqSupportMask = tf.placeholder( dtype=tf.float32, shape=self._intensity.shape )
                    self._mD = tf.diag( [ self._l1, self._l2, self._l3 ] )
                    self._n0 = tf.sin( self._theta ) * tf.cos( self._phi )
                    self._n1 = tf.sin( self._theta ) * tf.sin( self._phi )
                    self._n2 = tf.cos( self._theta )
                    self._n = self._n0*self._v0 + self._n1*self._v1 + self._n2*self._v2
                    self._nskew = self._n0*self._nskew0 + self._n1*self._nskew1 + self._n2*self._nskew2
                    self._R = tf.cos( self._psi )*self._I +\
                        tf.sin( self._psi )*self._nskew +\
                        ( self._one - tf.cos( self._psi ) )*tf.matmul( self._n, tf.transpose( self._n ) )
                    self._C = tf.matmul( self._R, tf.matmul( tf.matmul( self._mD, self._mD ), tf.transpose( self._R ) ) )

                with tf.name_scope( 'Blurring' ):
                    self._pkfft = tf.Variable( np.zeros( self._probSize ), dtype=tf.complex64, name='pkfft' )

                    self._blurKernel = tf.reshape( 
                        tf.exp( self._neg * tf.reduce_sum( self._q * tf.matmul( self._C, self._q ), axis=0 ) ), 
                        shape=self._coherentEstimate.shape
                    ) * ( self._l1 * self._l2 * self._l3 ) / self._twopi

                    self._tf_intens_f = tf.fft3d( tf.cast( self._coherentEstimate, dtype=tf.complex64 ) )
                    self._tf_blur_f = tf.fft3d( tf.cast( self._blurKernel, dtype=tf.complex64 ) )
                    self._tf_prod_f = self._tf_intens_f * self._tf_blur_f
                    self._imgBlurred = tf.abs( tf.ifft3d( self._tf_prod_f ) )

                with tf.name_scope( 'Optimizer' ):
                    self._var_list = [ self._l1, self._l2, self._l3, self._psi, self._theta, self._phi ]
                    
                    self._poissonNLL = tf.reduce_mean( 
                        self._FreqSupportMask * ( self._imgBlurred - self._intensity * tf.log( self._imgBlurred ) )
                    )
                    self._poissonOptimizer = tf.train.AdagradOptimizer( learning_rate=vardict[ 'pcc_learning_rate' ], name='poissonOptimize' )
                    self._trainPoisson = self._poissonOptimizer.minimize( self._poissonNLL, var_list=self._var_list )
                    self._currentGradients = [ 
                        n[0] for n in self._poissonOptimizer.compute_gradients( self._poissonNLL, var_list=self._var_list )
                    ]
                   
#                    self._gaussNLL = tf.reduce_mean( 
#                        self._FreqSupportMask * ( tf.sqrt( self._imgBlurred ) - tf.sqrt( self._intensity ) )**2
#                    )
#                    self._gaussOptimizer = tf.train.AdagradOptimizer( learning_rate=vardict[ 'pcc_learning_rate' ], name='gaussOptimize' )
#                    self._trainGauss = self._gaussOptimizer.minimize( self._gaussNLL, var_list=self._var_list )
#                    self._currentGradients = [ 
#                        n[0] for n in self._gaussOptimizer.compute_gradients( self._gaussNLL, var_list=self._var_list )
#                    ]



                with tf.name_scope( 'Preparation' ):
                    self._getCoherentEstimate  = tf.assign( 
                        self._coherentEstimate, 
                        tf.cast( self._intermedInt, dtype=tf.float32 ), 
                        name='getCoherentEstimate' 
                    )
                    self._getPCFKernelFFT = tf.assign( 
                        self._pkfft, 
                        tf.fft3d( tf.cast( self._blurKernel, dtype=tf.complex64 ) ), 
                        name='getPCFKernelFFT' 
                    )

                with tf.name_scope( 'Convolution' ):
                    self._convolveWithCoherentEstimate = tf.assign( 
                        self._intermedInt, 
                        tf.cast( tf.abs( tf.ifft3d( self._pkfft * tf.fft3d( self._intermedInt ) ) ), dtype=tf.complex64 ), 
                        name='convolveWithCoherentEstimate'
                    )

        self._progress = []

###########################################################################################
#   Estimates the Gassian partial coherence function while keeping the current estimate 
#   of the coherent scattering signal fixed.
###########################################################################################

    def GaussPCC( self, min_iterations=200, max_iterations=10000, iterations_per_checkpoint=50, tol=1.e-5, mask_fraction=1. ):
#        print( self.__sess__.run( self._var_list ) )
        # first, generating mask
        shp = self._cImage.get_shape().as_list()
        mask = np.zeros( shp )
        mask[ 
            np.floor( shp[0]//2*( 1.-abs(mask_fraction) ) ).astype( int ):np.ceil( shp[0]//2*( 1.+abs(mask_fraction) ) ).astype( int ), 
            np.floor( shp[1]//2*( 1.-abs(mask_fraction) ) ).astype( int ):np.ceil( shp[1]//2*( 1.+abs(mask_fraction) ) ).astype( int ), 
            np.floor( shp[2]//2*( 1.-abs(mask_fraction) ) ).astype( int ):np.ceil( shp[2]//2*( 1.+abs(mask_fraction) ) ).astype( int )
        ]= 1.
        if mask_fraction < 0.:
            mask = 1. - mask
        feed_dict = { self._FreqSupportMask:mask }
        self.__sess__.run( self._getIntermediateFFT )
        self.__sess__.run( self._getIntermediateInt )
        self.__sess__.run( self._getCoherentEstimate )
        this_grad = np.linalg.norm( np.array( self.__sess__.run( self._currentGradients, feed_dict=feed_dict ) ) )
        normalizr = self.__sess__.run( tf.reduce_sum( self._blurKernel ) )
        checkpoint = self.__sess__.run( self._var_list )
        obj = self.__sess__.run( self._poissonNLL, feed_dict=feed_dict )
#        obj = self.__sess__.run( self._gaussNLL, feed_dict=feed_dict )
        checkpoint.extend( [ obj, this_grad, normalizr ] )
        self._progress.append( checkpoint )
        n_iter = 1
        while n_iter < min_iterations or ( this_grad > tol and n_iter < max_iterations ):
            self.__sess__.run( self._trainPoisson, feed_dict=feed_dict )
#            self.__sess__.run( self._trainGauss, feed_dict=feed_dict )
            this_grad = np.linalg.norm( np.array( self.__sess__.run( self._currentGradients, feed_dict=feed_dict ) ) )
            if n_iter % iterations_per_checkpoint == 0:
                normalizr = self.__sess__.run( tf.reduce_sum( self._blurKernel ) )
                checkpoint = self.__sess__.run( self._var_list )
                obj = self.__sess__.run( self._poissonNLL, feed_dict=feed_dict )
#                obj = self.__sess__.run( self._gaussNLL, feed_dict=feed_dict )
                checkpoint.extend( [ obj, this_grad, normalizr ] )
                self._progress.append( checkpoint )
            n_iter += 1

        normalizr = self.__sess__.run( tf.reduce_sum( self._blurKernel ) )
        checkpoint = self.__sess__.run( self._var_list )
        obj = self.__sess__.run( self._poissonNLL, feed_dict=feed_dict )
#        obj = self.__sess__.run( self._gaussNLL, feed_dict=feed_dict )
        checkpoint.extend( [ obj, this_grad, normalizr ] )
        self._progress.append( checkpoint )


#        if n_iter >= max_iterations-1:
#            print( 'GaussPCC warning: max number of iteration reached: %d'%max_iterations )
#        else:
#            print( 'GaussPCC completed in %d iterations.'%n_iter )

        return

###########################################################################################
#   Performs <num_iterations> iterations of error reduction, while correcting for partial 
#   coherence of the incident wave field.
###########################################################################################
    def PCER( self, num_iterations, show_progress=False ):
        self.__sess__.run( self._getPCFKernelFFT )
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc=' ER' )
        else:
            allIterations = list( range( num_iterations ) )
        for n in allIterations:
            self.BlurSignal()
            self.__sess__.run( self._modproject )
            self.__sess__.run( self._supproject )
        return

###########################################################################################
#   Convolves the estimated coherent scattering with the estimated blur kernel.
###########################################################################################
    def BlurSignal( self ):
        self.__sess__.run( self._getIntermediateFFT )
        self.__sess__.run( self._getIntermediateInt )
        self.__sess__.run( self._convolveWithCoherentEstimate )
        return









            


