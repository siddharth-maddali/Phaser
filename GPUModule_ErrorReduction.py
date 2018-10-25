##############################################################
#
#	GPUModule_ErrorReduction:
#	    Contains a mixin for error reduction methods 
#	    within the GPUModule.Solver class.
#	
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    April 2018
#
##############################################################

class Mixin:

###########################################################################################
#   Performs <num_iterations> iterations of error reduction
###########################################################################################
    def ER( self, num_iterations ):
        for n in list( range( num_iterations ) ):
            self.__sess__.run( self._getIntermediateFFT )
            self.__sess__.run( self._modproject )
            self.__sess__.run( self._supproject )
            self.updateError()
        return

###########################################################################################
#   Performs <num_iterations> iterations of high-energy error reduction
###########################################################################################
    def HEER( self, num_iterations ):
        for n in list( range( num_iterations ) ):
            self.__sess__.run( self._getIntermediateFFT )
            self.__sess__.run( self._binThis )
            self.__sess__.run( self._scaleThis )
            self.__sess__.run( self._expandThis )
            self.__sess__.run( self._HEImgUpdate )
            self.__sess__.run( self._supproject )
        return
