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



class Mixin:


# CPU method for binary erosion
    def __CPUErosion__( self, num_erosions=1 ):
        temp = np.absolute( fftshift( self._support ) ).astype( bool )
        eroded = binary_erosion( temp, structure=self.BEStruct, iterations=num_erosions )
        self._support = fftshift( eroded.astype( complex ) )
        return
    
# GPU method for binary erosion, wraps CPU method in Tensorflow
    def __GPUErosion__( self, num_erosions=1, kernel_size=[ 1, 3, 3, 3, 1 ] ):
#        self._support = tf.py_function( 
#            func=self.BinaryErosionCPU, 
#            inp=[ num_erosions ], 
#            Tout=tf.complex64 
#        )
        sup_rankraised = tf.cast( 
            tf.expand_dims( 
                tf.expand_dims( self._support, axis=-1 ), 
                axis=0
            ), 
            dtype=tf.float32 
        )
        for n in range( num_erosions ):
            sup_rankraised = -tf.nn.max_pool3d( 
                -sup_rankraised, 
                ksize=kernel_size, 
                strides=1, 
                padding='SAME'
            )

        self._support = tf.cast( 
            tf.squeeze( sup_rankraised ), 
            dtype=tf.complex64
        )
        return
