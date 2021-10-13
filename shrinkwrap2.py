import tensorflow as tf
import numpy as np
import functools as ftools
def Shrinkwrap( amp,sigma, thresh ):
    amp = tf.constant(amp,dtype=tf.float32)
    x, y, z = np.meshgrid( 
        *[ np.linspace( -n//2, n//2-1, n ) for n in amp.shape ] 
    )

    rsquared = tf.constant( 
        ftools.reduce( lambda a, b: a+b, [ this**2 for this in [ x, y, z ] ] ), 
        dtype=tf.complex64
    )
    kernel = 1. / ( sigma * np.sqrt( 2. * np.pi ) ) * tf.exp( -0.5 * rsquared / ( sigma**2 ) )
    kernel_ft = tf.signal.fft3d( kernel )
    ampl_ft = tf.signal.fft3d( tf.cast( amp, dtype=tf.complex64 ) )
    blurred = tf.signal.fftshift( tf.abs( tf.signal.ifft3d( kernel_ft * ampl_ft ) ) )
    new_support = tf.where( blurred > thresh * tf.reduce_max( blurred ), 1., 0. )
    sup = new_support

    return tf.cast(sup,dtype=tf.float32)