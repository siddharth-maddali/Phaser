#    GPU modding for FastPhaseRetriever. Always run in a 
#    virtualenv set up for tensorflow
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        2018

import numpy as np
import tensorflow as tf
import time

# Class 'Solver' inherits methods from the mixins defined in the following modules.
import GPUModule_MemberVariables, GPUModule_SymbolicOps, GPUModule_InitializeSession
import GPUModule_ErrorReduction, GPUModule_HybridInputOutput, GPUModule_ShrinkWrap
import GPUModule_ObjectiveFunction

class Solver( 
        GPUModule_MemberVariables.Mixin, 
        GPUModule_SymbolicOps.Mixin, 
        GPUModule_InitializeSession.Mixin, 
        GPUModule_ErrorReduction.Mixin, 
        GPUModule_HybridInputOutput.Mixin, 
        GPUModule_ShrinkWrap.Mixin
    ):
    
    def __init__( self, varDict, outlog='' ):   
        # see Phaser for definition of varDict
        self.log_directory = outlog     # directory for computational graph dump
        self.defineMemberVariables( varDict )
        self.defineSymbolicOps( varDict )
        self.initializeObjectiveFunction()
        self.initializeSession()

    def Compute( self ):
        self.finalImage = self._cImage.eval( session=self.__sess__ )
        self.finalSupport = np.absolute( self._support.eval( session=self.__sess__ ) )
        self.errorTrend = self._error.eval( session=self.__sess__ )[:self.__iterations]
        self.__sess__.close()
        return

   



