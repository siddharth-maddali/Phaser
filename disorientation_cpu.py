# import numpy as np

from scipy.spatial.transform import Rotation as R
import numpy as np
import json
import itertools
#convert euler angles to quaternions
def rmat_2_quat(rmat):

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

def find_common_ref(grain_dict_path,grainID1,grainID2):
    
    with open(grain_dict_path,'r') as file:
        data = json.load(file)
    
    specori_1 = np.array(data['grain_%s'%grainID1]['Spec_Orientation'])
    specori_2 = np.array(data['grain_%s'%grainID2]['Spec_Orientation'])
    print('Spec Orient grain 1',specori_1)
    print('Spec Orient grain 2',specori_2)
    uvwhkl_1 = np.array([specori_1[0], np.cross(specori_1[0],specori_1[1])]) 
    uvwhkl_2 = np.array([specori_2[0], np.cross(specori_2[0],specori_2[1])]) 

    o_1 = rot_mat_from_uvwhkl(uvwhkl_1[0],uvwhkl_1[1])
    o_2 = rot_mat_from_uvwhkl(uvwhkl_2[0],uvwhkl_2[1])


    q1 = rmat_2_quat([o_1])
    q2 = rmat_2_quat([o_2])
    mis = calc_disorient(q1,q2)
    print('misorientation between two crystals:',mis[0])

    gs = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]])
    gs = gs/np.linalg.norm(gs,axis=1)[:,np.newaxis]


    samp_vec_1 = [o_1@g for g in gs]
    samp_vec_2 = [o_2@g for g in gs]


    combos = list(itertools.product(samp_vec_1,samp_vec_2))
    # print(samp_vec_1)
    # print('\n',samp_vec_2)
    angle = lambda a,b: 90-np.abs(90-np.nan_to_num(np.arccos(a@b),0)*180/np.pi)

    angles = [angle(a[0],a[1]) for a in combos]
    # print(angles)
    ind = angles.index(min(angles))


    print('two vectors:',combos[ind],angles[ind])


    ang_1 = [angle(combos[ind][0],s) for s in samp_vec_1]
    ang_2 = [angle(combos[ind][1],s) for s in samp_vec_2]

    g1 = gs[ang_1.index(min(ang_1))]*np.sqrt(3)
    g2 = gs[ang_2.index(min(ang_2))]*np.sqrt(3)
    
    if np.arccos(combos[ind][0]@combos[ind][1])>np.pi/2:
        g2 *= -1
    print('angle closest vectors:',min(angles),'degrees')
    print('grain %s:'%grainID1,g1)
    print('grain %s'%grainID2,g2)
    return g1,g2