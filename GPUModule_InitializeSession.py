##############################################################
#
#	GPUModule_InitializeSession:
#	    Contains a mixin for initializing a Tensorflow 
#	    session within the GPUModule.Solver class.
#	
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

import numpy as np
import tensorflow as tf

class Mixin:

    def initializeSession( self ):

        config = tf.ConfigProto( allow_soft_placement=True, log_device_placement=True )
        config.gpu_options.allow_growth=True
        self.__sess__ = tf.Session( config=config )
        if len( self.log_directory ) > 0:
            writer = tf.summary.FileWriter( self.log_directory )
            writer.add_graph( self.__sess__.graph )
        self.__sess__.run( tf.global_variables_initializer() )
        return

