##############################################################
#
#	GPUModule_ObjectiveFunction:
#	    Contains a mixin for computing the objective 
#	    function designed to be robust to Poission 
#	    noise, see Godard et al (Opt. Express 2012)
#
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except: 
    import tensorflow as tf

class Mixin:

##############################################################
#   Initializes objective function routine.
##############################################################
    def initializeObjectiveFunction( self ):
        self.errorMetric = []
        with tf.name_scope( 'Error' ):
            self._error = tf.Variable( 0., dtype=tf.float32, name='objFunc' )
            self._getErrorNow = tf.assign( 
                self._error, 
                tf.reduce_sum( 
                    tf.square( 
                        tf.abs( self._modulus ) - tf.abs( self._intermedFFT )
                    )
                )
            )
        return

##############################################################
#   Updates error and adds new element to 
##############################################################
    def updateErrorGPU( self ):
        self.__sess__.run( self._getErrorNow )
        self.errorMetric.append( self._error.eval( session=self.__sess__ ) )
                        # this has a lot of overhead.
        return




