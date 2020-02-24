###########################################################
#
#    ER.py: 
#        Plugin for error reduction
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        January 2020
#        smaddali@alumni.cmu.edu
#
###########################################################

from tqdm import tqdm

class Mixin: 

    def ER( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc=' ER' )
        else:
            allIterations = list( range( num_iterations ) )
        for i in allIterations:
            self._ModProject()
            self._SupProject()
            self._UpdateMod()
            self._UpdateError()
        return
