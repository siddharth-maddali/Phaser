import numpy as np
from tensorflow import signal as s
import tensorflow as tf
from shrinkwrap import Shrinkwrap as shrink_wrap
from scipy.ndimage import center_of_mass as com
import matplotlib.pyplot as plt
from miscellaneous import translate

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


def check_get_conj_reflect_us(ar1, ar2,verbose=False):
    
    arr1 = tf.constant(ar1,dtype=tf.complex64)
    arr2 = tf.constant(ar2,dtype=tf.complex64)
    
    conj_arr2 = tf.constant(conj_reflect(arr2[0]),dtype=tf.complex64)
    support1 = tf.constant(shrink_wrap(tf.abs(arr1[0]), 1., .1),dtype=tf.complex64)
    support2 = tf.constant(shrink_wrap(tf.abs(arr2[0]), 1., .1),dtype=tf.complex64)
    support3 = tf.constant(shrink_wrap(tf.abs(conj_arr2), 1., .1),dtype=tf.complex64)
    cc1 = cross_correlation(support1,support3 ).numpy()
    cc2 = cross_correlation(support1,support2 ).numpy()

    if np.amax(cc1) > np.amax(cc2):
        cc = cc1
        inds = np.stack(np.where(cc==cc.max())).T[0]
        
        if verbose:
            print('flip')
        val = np.array([conj_reflect(arr2[i]).numpy() for i in range(3)])
        amp = np.absolute(val)[0]
        sup = np.where(amp>0.01*amp.max(),1,0)
        u = np.angle(val)*sup
        val = amp*np.exp(1j*u)

        shp = np.array(amp.shape)//2
        
        
        val = translate(val,inds,shp)

        return val#abbs*np.exp(1j*ang)
         
    else:
        cc = cc2
        inds = np.stack(np.where(cc==cc.max())).T[0]
        
        shp = np.array(ar2.shape[1:])//2
        ar2 = translate(ar2,inds,shp)
        return ar2#np.array([center(ar2[i]) for i in range(3)])
    
def check_get_conj_reflect_us_opp(ar1, ar2,verbose=False):
    
    arr1 = tf.constant(ar1,dtype=tf.complex64)
    arr2 = tf.constant(ar2,dtype=tf.complex64)
    
    conj_arr2 = tf.constant(conj_reflect(arr2[0]),dtype=tf.complex64)
    support1 = tf.constant(shrink_wrap(tf.abs(arr1[0]), 1., .1),dtype=tf.complex64)
    support2 = tf.constant(shrink_wrap(tf.abs(arr2[0]), 1., .1),dtype=tf.complex64)
    support3 = tf.constant(shrink_wrap(tf.abs(conj_arr2), 1., .1),dtype=tf.complex64)
    cc1 = cross_correlation(support1,support3 ).numpy()
    cc2 = cross_correlation(support1, support2).numpy()

    if np.amax(cc1) < np.amax(cc2):
        if verbose:
            print('flip')
        val = np.array([conj_reflect(arr2[i]).numpy() for i in range(3)])
        ang = np.angle(val)*support3.numpy()
        abbs = np.absolute(val)*support3.numpy()
        return abbs*np.exp(1j*ang)
         
    else:
        return ar2#np.array([center(ar2[i]) for i in range(3)])

    
    
def check_get_conj_reflect(arr1, arr2,verbose=False):
    
    arr1 = tf.constant(arr1,dtype=tf.complex64)
    arr2 = tf.constant(arr2,dtype=tf.complex64)
    conj_arr2 = tf.constant(conj_reflect(arr2),dtype=tf.complex64)
    support1 = tf.constant(shrink_wrap(tf.abs(arr1), 1., .1),dtype=tf.complex64)
    support2 = tf.constant(shrink_wrap(tf.abs(arr2), 1., .1),dtype=tf.complex64)
    support3 = tf.constant(shrink_wrap(tf.abs(conj_arr2), 1., .1),dtype=tf.complex64)
    cc1 = cross_correlation(support1,support3 ).numpy()
    cc2 = cross_correlation(support1, support2).numpy()

    if np.amax(cc1) > np.amax(cc2):
        cc = cc1
        inds = np.stack(np.where(cc==cc.max())).T[0]
        if verbose:
            print('flip')
            
        conj_arr2 = conj_arr2.numpy()
        conj_arr2 = np.where(np.absolute(conj_arr2)>0.01*np.absolute(conj_arr2).max(),conj_arr2,0)
        shp = np.array(conj_arr2.shape)//2
        conj_arr2 = translate(conj_arr2,inds,shp)
        return conj_arr2
         
    else:
        cc = cc2
        inds = np.stack(np.where(cc==cc.max())).T[0]
        arr2 = arr2.numpy()
        shp = np.array(arr2.shape)//2
        arr2 = translate(arr2,inds,shp)
        return arr2