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

class Mixin:

##############################################################
#   Initializes objective function routine.
#   block_size is how much the error array grows in GPU 
#   memory each time it fills up.
##############################################################
    def initializeObjectiveFunction( block_size=500 ):
        self.__block_size = block_size
        self.__iterations = 0
        self._error = tf.Variable( 
            np.zeros( self.__block_size ), 
            dtype=tf.float32, 
            name='objFunc' 
        )
        return

    def updateError():
        if self.__iterations % self.__block_size == 0:
            self.__sess__.run( 
                tf.assign( 
                    self._error, 
                    np.concatenate( 
                        ( 
                            self._error.eval( session=self.__sess__ ), 
                            np.zeros( self.__block_size )
                        )
                    ), 
                    validate_shape=False
                )
            )
        self.__sess__.run(
            tf.assign( 
                self._error[ self.__iterations ], 
                tf.reduce_sum( tf.square( ( self._modulus - tf.abs( self._intermedFFT ) ) ) )

            )
        )
        self.__iterations = self.__iterations + 1
        return




