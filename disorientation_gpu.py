# import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

import numpy as np
#convert euler angles to quaternions
def rmat_2_quat(rmat):
#     rmat = np.array(rmat).T
    r = R.from_matrix(rmat) #rmat is a matrix transforming vector from crystal frame to sample frame
     
    quat = r.as_quat()
    
    for num, val in enumerate(quat):
        if val[3] < 0: #q1,q2,q3,q0 format
            quat[num] = -1.0*quat[num]
    quat = np.roll(quat,1,axis=1)
    
    # rmat = r1.as_matrix()
    # rmat_inv = r.as_matrix()

    return quat

#compute quaternion product
# @tf.function
def QuatProd(p, q):
    p0 = tf.reshape(p[:,0],(p[:,0].shape[0],1))
    q0 = tf.reshape(q[:,0],(q[:,0].shape[0],1))
    
    l = tf.reduce_sum(tf.multiply(p[:,1:],q[:,1:]),1)
    
    prod1 = tf.math.multiply(p0,q0)[:,0] - l
    
    prod2 = tf.math.multiply(p0,q[:,1:]) + tf.math.multiply(q0,p[:,1:]) + tf.linalg.cross(p[:,1:],q[:,1:])
    m = tf.transpose(tf.stack([prod1,prod2[:,0],prod2[:,1],prod2[:,2]]))
    
    return m

#invert quaternion
# @tf.function
def invQuat(p):
    
    q = tf.transpose(tf.stack([-p[:,0],p[:,1],p[:,2],p[:,3]]))
    

    return q


#calculate the disorientation between two sets of quaternions ps and qs
# @tf.function
def calc_disorient(y_true,y_pred):
#     y_true = tf.Variable(y_true)
#     y_pred = tf.Variable(y_pred)

    
    #sort quaternion for cubic symmetry trick
    p = tf.sort(tf.abs(QuatProd(invQuat(y_true),y_pred)))
    
    #calculate last component of two other options
    p1 = (p[:,2]+p[:,3])/2**(1/2)
    p2 = (p[:,0]+p[:,1]+p[:,2]+p[:,3])/2
    vals = tf.transpose(tf.stack([p[:,-1],p1,p2]))
    #pick largest value and find angle
    
    max_val = tf.math.reduce_max(vals,axis=1)
    mis = (2*tf.acos(max_val))
    
    return np.degrees(replacenan(mis).numpy())

#replace NaNs with zeros
def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)