###########################################################
#
#    HIO.py: 
#        Plugin for hybrid input/output
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        January 2020
#        smaddali@alumni.cmu.edu
#
###########################################################

from tqdm import tqdm

class Mixin:

    def HIO( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc='HIO' )
        else:
            allIterations = list( range( num_iterations ) )
        for i in allIterations:
            origImage = self._cImage.copy() 
            self._ModProject()
            self._cImage = ( self._support * self._cImage ) +\
                self._support_comp * ( origImage - self._beta * self._cImage )
            self._UpdateError()
        return
