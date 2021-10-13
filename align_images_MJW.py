import numpy as np
from tensorflow import signal as s
import tensorflow as tf
from scipy.ndimage import center_of_mass as com
import functools as ftools
def center(arr):
    center = com(np.absolute(arr))
    for n in [ 0,1,2]:
        arr = np.roll(arr,arr.shape[n]//2 - int(np.round(center[n],0)),axis=n)
    center = com(np.absolute(arr))
    return arr
    
def cross_correlation(a, b):
    A = s.ifftshift(s.fft3d(s.fftshift(conj_reflect(a))))
    B = s.ifftshift(s.fft3d(s.fftshift(b)))
    CC = A * B
    return s.ifftshift(s.ifft3d(s.fftshift(CC)))


def conj_reflect(arr):
    F = s.ifftshift(s.fft3d(s.fftshift(arr)))
    return s.ifftshift(s.ifft3d(s.fftshift(tf.math.conj(F))))


   
    
def check_get_conj_reflect(arr1, arr2,verbose=False):
    arr1 = tf.constant(arr1,dtype=tf.complex64)
    arr2 = tf.constant(arr2,dtype=tf.complex64)
    conj_arr2 = tf.constant(conj_reflect(arr2),dtype=tf.complex64)
    support1 = tf.constant(shrink_wrap(tf.abs(arr1), 1., .1),dtype=tf.complex64)
    support2 = tf.constant(shrink_wrap(tf.abs(arr2), 1., .1),dtype=tf.complex64)
    support3 = tf.constant(shrink_wrap(tf.abs(conj_arr2), 1., .1),dtype=tf.complex64)
    cc1 = cross_correlation(support1,support3 )
    cc2 = cross_correlation(support1, support2)

    if np.amax(cc1) > np.amax(cc2):
        if verbose:
            print('flip')
        return center(conj_arr2.numpy())
         
    else:
        return center(arr2.numpy())
    

def shrink_wrap( amp,sigma, thresh ):
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

    return sup

def check_get_conj_reflect_opposite(arr1, arr2,verbose=False):
    arr1 = tf.constant(arr1,dtype=tf.complex64)
    arr2 = tf.constant(arr2,dtype=tf.complex64)
    conj_arr2 = tf.constant(conj_reflect(arr2),dtype=tf.complex64)
    support1 = tf.constant(shrink_wrap(tf.abs(arr1), 1., .1),dtype=tf.complex64)
    support2 = tf.constant(shrink_wrap(tf.abs(arr2), 1., .1),dtype=tf.complex64)
    support3 = tf.constant(shrink_wrap(tf.abs(conj_arr2), 1., .1),dtype=tf.complex64)
    cc1 = cross_correlation(support1,support3 )
    cc2 = cross_correlation(support1, support2)

    if np.amax(cc1) < np.amax(cc2):
        if verbose:
            print('flip')
        return center(conj_arr2.numpy())
         
    else:
        return center(arr2.numpy())