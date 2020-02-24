###########################################################
#
#    SF.py: 
#        Plugin for solvent flipping
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        January 2020
#        smaddali@alumni.cmu.edu
#
###########################################################

from tqdm import tqdm

class Mixin:

    def SF( self, num_iterations, show_progress=False ):
        if show_progress:
            allIterations = tqdm( list( range( num_iterations ) ), desc=' SF' )
        else:
            allIterations = list( range( num_iterations ) )
        for i in allIterations:
            self._SupReflect()
            self._ModProject()
            self._UpdateError()
        self._SupProject() # project one last time into the support space
        return
