##############################################################
#
#	Morphology:
#           Contains morphological functions, typically 
#           to manipulate object support during phase 
#           retrieval.
#
#	    Siddharth Maddali
#	    Argonne National Laboratory
#	    Nov 2020
#           6xlq96aeq@relay.firefox.com
#
##############################################################

import numpy as np
import tensorflow as tf

from scipy.ndimage.morphology import binary_erosion
try: 
    from pyfftw.interfaces.numpy_fft import fftshift
except:
    from numpy.fft import fftshift

BEStruct = np.ones( ( 3, 3, 3 ) ) # structuring element for 3D binary erosion

class Mixin:

# CPU method for binary erosion
    def BinaryErosionCPU( self, num_erosions=1 ):
        temp = np.absolute( fftshift( self._support ) ).astype( bool )
        eroded = binary_erosion( temp, structure=BEStruct, iterations=num_erosions )
        self._support = fftshift( eroded.astype( complex ) )
        return
    
# GPU method for binary erosion, wraps CPU method in Tensorflow
    def BinaryErosionGPU( self, num_erosions=1 ):
        self._support = tf.py_function( 
            func=self.BinaryErosionCPU, 
            inp=[ num_erosions ], 
            Tout=tf.complex64 
        )
        return
