##############################################################
#
#	GPUModule_ShrinkWrap:
#	    Contains a mixin for the shrinkwrap algorithm 
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

###########################################################################################
#   Shrinkwrap (Gaussian blurring followed by thresholding)
###########################################################################################
    def Shrinkwrap( self, sigma, thresh ):
        self.__sess__.run( self._getNewDist, feed_dict={ self._sigma:sigma } )
        self.__sess__.run( self._getKernel )
        self.__sess__.run( self._getBlurred )
        self.__sess__.run( self._rollX )
        self.__sess__.run( self._rollY )
        self.__sess__.run( self._rollZ )
        self.__sess__.run( self._updateSupport, feed_dict={ self._thresh:thresh } )
        self.__sess__.run( self._updateSupComp, feed_dict={ self._thresh:thresh } )
        self.__sess__.run( self._supproject )
        return
