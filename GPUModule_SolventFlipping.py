##############################################################
#
#	GPUModule_SolventFlipping:
#	    Contains a mixin for solvent flipping methods
#           within the GPUModule.Solver class.
#
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    Jan 2019
#
##############################################################

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Mixin:

    def initializeSF( self ):
        with tf.name_scope( 'SF' ):
            self._twoC = tf.constant( 2., dtype=tf.complex64 )
            self._ones = tf.constant( 
                np.ones( self._probSize ), 
                dtype=tf.complex64, 
                name='Ones' 
            )
            self._supreflect = tf.assign( 
                self._cImage, 
                ( self._twoC*self._support - self._ones ) * self._cImage, 
                name='supReflect'
            )
            return
        



###########################################################################################
#   Performs <num_iterations> iterations of solvent flipping
###########################################################################################
    def SF( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc=' SF' )
        else:
            allIterations = list( range( num_iterations ) )
        for n in allIterations:
            self.__sess__.run( self._supreflect )
            self.__sess__.run( self._modproject )
            self.__sess__.run( self._supreflect )
            self.__sess__.run( self._modproject )
            self.updateErrorGPU()
        return


