# import numpy as np

from scipy.spatial.transform import Rotation as R
import numpy as np
#convert euler angles to quaternions
def rmat_2_quat(rmat):
#     rmat = np.array(rmat).T
    r = R.from_matrix(rmat)
    
    r1 = r.inv() #to match the massif convention, where the Bunge Euler/Rotation is transformation from sample to crystal frame. 
    quat = r1.as_quat()
    
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
    p0 = np.reshape(p[:,0],(p[:,0].shape[0],1))
    q0 = np.reshape(q[:,0],(q[:,0].shape[0],1))
    
    l = np.sum(p[:,1:]*q[:,1:],axis=1)
    
    prod1 = (p0*q0)[:,0] - l
    
    prod2 = p0*q[:,1:] + q0*p[:,1:] + np.cross(p[:,1:],q[:,1:])
    m = np.stack([prod1,prod2[:,0],prod2[:,1],prod2[:,2]]).T
    
    return m

#invert quaternion
# @tf.function
def invQuat(p):
    
    q = np.stack([-p[:,0],p[:,1],p[:,2],p[:,3]]).T
    

    return q


#calculate the disorientation between two sets of quaternions ps and qs
# @tf.function
def calc_disorient(y_true,y_pred):
#     y_true = tf.Variable(y_true)
#     y_pred = tf.Variable(y_pred)

    
    #sort quaternion for cubic symmetry trick
    p = np.sort(np.absolute(QuatProd(invQuat(y_true),y_pred)))
    
    #calculate last component of two other options
    p1 = (p[:,2]+p[:,3])/2**(1/2)
    p2 = (p[:,0]+p[:,1]+p[:,2]+p[:,3])/2
    vals = np.stack([p[:,-1],p1,p2]).T
    #pick largest value and find angle
    
    max_val = vals.max(axis=1)
    mis = 2*np.arccos(max_val)
    
    return np.degrees(replacenan(mis))

#replace NaNs with zeros
def replacenan(t):
    return np.where(np.isnan(t), np.zeros_like(t), t)

def rot_mat_from_uvwhkl(uvw,hkl):
    b = uvw/np.linalg.norm(uvw)
    n = hkl/np.linalg.norm(hkl)
    
    t = np.cross(n,b)
    t = t/np.linalg.norm(t)
    
    rot_mat = np.stack([b,t,n])
    return rot_mat