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

from tqdm import tqdm

class Mixin:

###########################################################################################
#   Performs <num_iterations> iterations of error reduction
###########################################################################################
    def ER( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc=' ER' )
        else:
            allIterations = list( range( num_iterations ) )
        for n in allIterations:
            self.__sess__.run( self._getIntermediateFFT )
            self.__sess__.run( self._modproject )
            self.__sess__.run( self._supproject )
            self.updateErrorGPU()
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
