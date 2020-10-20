###########################################################
#
#    GPU modding for Phaser. Always run in a 
#    virtualenv set up for Tensorflow 2.x
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        2018
#
###########################################################

import numpy as np
import tensorflow as tf

import time

import PostProcessing as post

# Class 'Solver' inherits methods from the mixins defined in the following modules.
import GPUModule_Core, RecipeParser
import ER, HIO, SF
import GaussPCC

from GaussPCC import PCSolver

class Solver( 
        GPUModule_Core.Mixin, 
        RecipeParser.Mixin, 
        ER.Mixin, 
        HIO.Mixin, 
        SF.Mixin, 
        GaussPCC.Mixin
    ):
    
    def __init__( self, gpack ):   
        # see Phaser.py for definition of gpack
        self.ImportCore( gpack )
        self.generateAlgoDict()
        if gpack[ 'pcc' ]==True: 
            #self._cImage.assign( self._cImage * tf.cast( tf.reduce_sum( tf.abs( self._modulus )**2 ) / tf.reduce_sum( tf.abs( self._cImage )**2 ), dtype=tf.complex64 ) )
            self._cImage.assign( 
                self._cImage * tf.sqrt( 
                    tf.reduce_sum( tf.cast( tf.abs( self._modulus ), dtype=tf.complex64 ) ) / tf.reduce_sum( tf.cast( tf.abs( tf.signal.fft3d( self._cImage ) )**2, dtype=tf.complex64 ) )
                )
            )
            self._pccSolver = PCSolver( np.absolute( self._modulus )**2, gpack )
            self._kernel_f = self._pccSolver.getBlurKernel()
            self._ModProject = self._ModProjectPC
            self._algodict[ 'PCC' ] = self.PCC
        return



   



