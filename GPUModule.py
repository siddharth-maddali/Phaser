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

class Solver( 
        GPUModule_Core.Mixin, 
        RecipeParser.Mixin, 
        ER.Mixin, 
        HIO.Mixin, 
        SF.Mixin, 
        GaussPCC.Mixin
    ):
    
    def __init__( self, varDict ):   
        # see Phaser.py for definition of varDict
        self.ImportCore( varDict )
        self.generateAlgoDict()
#        print( varDict.keys() )
#        print( varDict[ 'pcc' ] )
        if varDict[ 'pcc' ]==True: 
            self.setUpPCC( varDict )
        return



   



