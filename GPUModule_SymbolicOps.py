##############################################################
#
#	GPUModule_SymbolicOps:
#	    Contains a mixin for defining Tensorflow ops
#	    within the GPUModule.Solver class.
#	
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

import numpy as np
import tensorflow as tf

class Mixin:

    def defineSymbolicOps( self, varDict ):
        
        with tf.device( '/gpu:0' ):
            with tf.name_scope( 'FFT' ):
                self._getIntermediateFFT = tf.assign( 
                    self._intermedFFT, 
                    tf.fft3d( self._cImage ), 
                    name='intermedFFT'
                )

            if 'bin_left' in varDict.keys():
                with tf.name_scope( 'highEnergy' ):
                    self._getIntermediateFFT = tf.assign( 
                        self._intermedFFT, 
                        tf.fft3d( self._cImage ), 
                        name='intermedFFT'
                    )
                    self._binThis = tf.assign( 
                        self._binned, 
                        tf.transpose( 
                            tf.matmul( 
                                tf.matmul( 
                                    tf.transpose( self._binL, [ 2, 0, 1 ] ), 
                                    tf.transpose( 
                                        tf.cast( tf.square( tf.abs( self._intermedFFT ) ), dtype=tf.complex64 ), 
                                        [ 2, 0, 1 ]
                                    )
                                ), 
                                tf.transpose( self._binR, [ 2, 0, 1 ] )
                            ), [ 1, 2, 0 ] 
                        ), 
                        name='Binning'
                    )
                    self._scaleThis = tf.assign( 
                        self._scaled,
                        tf.divide( self._modulus, tf.sqrt( self._binned ) ), 
                        name='Scaling'
                    )
                    self._expandThis = tf.assign( 
                        self._expanded, 
                        tf.transpose( 
                            tf.matmul( 
                                tf.matmul( 
                                    tf.transpose( self._binR, [ 2, 0, 1 ] ), 
                                    tf.transpose( self._scaled, [ 2, 0, 1 ] )
                                ), 
                                tf.transpose( self._binL, [ 2, 0, 1 ] )
                            ), [ 1, 2, 0 ] 
                        ), 
                        name='Expansion'
                    )
                    self._HEImgUpdate = tf.assign( 
                        self._cImage, 
                        tf.multiply( 
                            self._support, 
                            tf.ifft3d( self._scale * tf.multiply( self._expanded, self._intermedFFT ) ) 
                        ), 
                        name='HEImgUpdate'
                    )
                    self._HEImgCorrect = tf.assign( 
                        self._cImage, 
                        self._cImage + tf.multiply( self._support_comp, self._buffImage - self._beta*self._cImage ), 
                        name='HEImgCorrect' 
                    )
    
            else:   
                with tf.name_scope( 'ER' ):
                    self._modproject = tf.assign( 
                        self._cImage, 
                        tf.ifft3d( 
                                self._modulus *\
                                tf.exp( tf.complex( 
                                    tf.zeros( self._cImage.shape ), 
                                    tf.angle( self._intermedFFT ) 
                                ) 
                            )
                        ), 
                        name='modProject' 
                    )

                with tf.name_scope( 'HIO' ):
                    self._disrupt = tf.assign( 
                        self._cImage, 
                        ( self._support * self._cImage ) +\
                            self._support_comp * ( self._buffImage - self._beta*self._cImage ), 
                        name='disrupt'
                    )
            
            with tf.name_scope( 'bufferImage' ):
                self._dumpimage = tf.assign( self._buffImage, self._cImage, name='dumpImage' )

            with tf.name_scope( 'Support' ):
                self._supproject = tf.assign( 
                    self._cImage, 
                    self._cImage * self._support, 
                    name='supProject' 
                )

        #   Shrinkwrap-related assignments
            with tf.name_scope( 'Shrinkwrap' ):
                self._getNewDist = tf.assign( 
                    self._dist, 
                    tf.exp( 
                        self._neg * ( 
                            self._x*self._x + self._y*self._y + self._z*self._z 
                        ) / ( self._two * self._sigma * self._sigma )
                    ), 
                    name='getNewDist'
                )
                self._getKernel = tf.assign( 
                    self._kernelFFT, 
                    tf.fft3d( 
                        tf.cast( 
                            self._dist, 
                            tf.complex64
                        )
                    ), 
                    name='kernelFFT'
                )
                self._getBlurred = tf.assign( 
                    self._blurred, 
                    tf.abs( 
                        tf.ifft3d( 
                            tf.multiply( 
                                tf.fft3d( tf.cast( tf.abs( self._cImage ), tf.complex64 ) ), 
                                self._kernelFFT
                            )
                        )
                    ), 
                    name='getBlurred' 
                )
                self._rollX = tf.assign( 
                    self._blurred, 
                    tf.concat( 
                        ( 
                            self._blurred[ (self._probSize[0]//2):, :, : ], 
                            self._blurred[ :(self._probSize[0]//2), :, : ] 
                        ), axis=0 ), 
                    name='rollX'
                )
                self._rollY = tf.assign( 
                    self._blurred, 
                    tf.concat( 
                        ( 
                            self._blurred[ :, (self._probSize[1]//2):, : ], 
                            self._blurred[ :, :(self._probSize[1]//2), : ] 
                        ), axis=1 ), 
                    name='rollY'
                )
                self._rollZ = tf.assign( 
                    self._blurred, 
                    tf.concat( 
                        ( 
                            self._blurred[ :, :, (self._probSize[2]//2): ], 
                            self._blurred[ :, :, :(self._probSize[2]//2) ] 
                        ), axis=2 ), 
                    name='rollZ'
                )
                self._updateSupport = tf.assign( 
                    self._support, 
                    tf.cast( self._blurred > self._thresh * tf.reduce_max( self._blurred ), tf.complex64 ), 
                    name='updateSup' 
                )
                self._updateSupComp = tf.assign( 
                    self._support_comp, 
                    tf.cast( self._blurred <= self._thresh * tf.reduce_max( self._blurred ), tf.complex64 ), 
                    name='updateSupComp' 
                )

        return

