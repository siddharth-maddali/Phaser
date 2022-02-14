from skimage import measure
from skimage import measure
from skimage.draw import ellipsoid
from scipy.ndimage.measurements import center_of_mass as com
import time
from scipy.optimize import minimize
from align_images import conj_reflect,check_get_conj_reflect
# from itertools import combinataions
import align_images as ai
from miscellaneous import plot_u,translate,center_u
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

def align_sup(amp_1,amp_2,u_11,u_2,plot=True):
    
    
    # compute complex conjugate twin for grain 2. Grain 1 boundaries must be compared to both conjugates
    amp_2s = [amp_2,np.absolute(conj_reflect(np.complex64(amp_2)).numpy())]
    sup_2_conj = np.where(amp_2s[1]>0.01*amp_2s[1].max(),1,0)
    u_2s = [u_2,np.stack([np.angle(conj_reflect(amp_2*np.exp(1j*u)).numpy())*sup_2_conj for u in u_2])]
    u_2s = [u_2s[i]*amp_2s[i] for i in range(2)]

    winner_n = []
    winner_cc = []
    conj = False
    
    #loop through the two possible conjugates for grain 2, checking with grain 1
    for i,amp2 in enumerate(amp_2s):
        u_22 = u_2s[i]
        s = amp_1.shape[0]//2
        
        #compute nth discrete difference
        l1 = np.diff(amp_1,axis=1)
        l2 = np.diff(amp2,axis=1)
        #sum over all axes
        l1 = l1.sum(axis=(0,2))
        l2 = l2.sum(axis=(0,2))
        #find maximum and minimum indices to identify potential boundaries
        new1 = [np.where(l1==l1.max())[0][0]+1,np.where(l1==l1.min())[0][0]]
        new2 = [np.where(l2==l2.max())[0][0]+1,np.where(l2==l2.min())[0][0]]
        ns = [[new1[0],new2[1]],[new1[1],new2[0]]]
        ccs = []
        pairs = []
        
        #loop through boundaries in 'ns'
        for nn in ns:
            n1,n2 = nn
            
            #shrinkwrap amp_1 and amp2 to find support
            s1 = np.where(amp_1>0.01*amp_1.max(),1,0)
            s1[:,n1+1:,:] = 0
            s1[:,:n1,:] = 0
            s1[s1!=0] = 1
            

            s2 = np.where(amp2>0.01*amp2.max(),1,0)
            s2[:,n2+1:,:] = 0
            s2[:,:n2,:] = 0
            s2[s2!=0]=1
            

            #multiply us by support
            u_1 = u_11*s1
            u_2 = u_22*s2

            #center displacement
            for i in range(3):
                u_1[i][s1==1] -= u_1[i][s1==1].mean()
                u_2[i][s2==1] -= u_2[i][s2==1].mean()
            
            #create complex boundary objects and compute cross correlation
            B_1 = np.complex64(s1*np.exp(1j*u_1))
            B_2 = np.complex64(s2*np.exp(1j*u_2))
            cc = ai.cross_correlation(B_1,B_2).numpy().sum(axis=0)

            inds = np.stack(np.where(cc==cc.max())).T[0]
            
            pairs.append([[s,n1+1,s],[inds[0],n2,inds[2]]])
            ccs.append(cc.max())
            
        ind = ccs.index(max(ccs))
        winner_cc.append(max(ccs))
        winner_n.append(pairs[ind])
    
    ind = winner_cc.index(max(winner_cc))
    new1,new2 = winner_n[ind]
    if ind ==1:
        conj = True
        amp_2 = amp_2s[1]
        print('conj')
    
    amp_1 = translate(amp_1,new1,[s,s,s])
    amp_2 = translate(amp_2,new2,[s,s,s])
    flip = False

    if np.sum(amp_1[:,:s,:])<np.sum(amp_1[:,s:,:]):
        flip = True
    
    if plot:
        plt.imshow((amp_1+amp_2)[:,:,s])
        plt.show()

    return new1,new2,flip,conj


import tensorflow as tf
import align_images as ai
def align_grains(obj1,obj2,coupled=True):
    u_1 = obj1['u']
    amp_1 = obj1['amp']
    amp_1_sum = amp_1.sum()
    amp_1 = amp_1/amp_1.sum()
    sup_1 = obj1['sup']
    
    u_2 = obj2['u']
    amp_2 = obj2['amp']
    amp_2_sum = amp_2.sum()
    amp_2 = amp_2/amp_2.sum()
    sup_2 = obj2['sup']
    
    shp = amp_2.shape[0]//2
    center=[shp,shp]
    new_1,new_2,flip,conj = align_sup(amp_1,amp_2,u_1,u_2,plot=False)


    if conj:
        print('True')
        img2 = amp_2*np.exp(u_2*1j)
        img2 = np.stack([conj_reflect(i) for i in img2])

        amp_2 = np.where(np.absolute(img2)>0.01*amp_2.max(),np.absolute(img2),0)[0]
        sup_2 = np.where(np.absolute(img2)>0.01*amp_2.max(),1,0)[0]
        u_2 = np.angle(img2)*np.where(amp_2[np.newaxis,:,:,:]>0.01*amp_2.max(),1,0)


    
    if flip:
        u_1,u_2 = u_2,u_1
        amp_1,amp_2 = amp_2,amp_1
        new_1,new_2 = new_2,new_1
        amp_1_sum,amp_2_sum = amp_2_sum,amp_1_sum
        sup_1,sup_2 = sup_2,sup_1
        
    sup_1 = translate(sup_1,new_1,[shp,shp,shp])  
    u_1 = translate(u_1,new_1,[shp,shp,shp])
    amp_1 = translate(amp_1,new_1,[shp,shp,shp])
    


    
    sup_2 = translate(sup_2,new_2,[shp,shp,shp])
    u_2 = translate(u_2,new_2,[shp,shp,shp])
    amp_2 = translate(amp_2,new_2,[shp,shp,shp])
#     plt.imshow((amp_1+amp_2)[:,:,64])
#     plt.show()
#     mask = sup_1 == 1

#     stack = np.stack([amp_1/amp_1.sum(),amp_2/amp_2.sum()])
#     maxx = np.max(stack,axis=0)
#     amp_1 = np.where(maxx == amp_1/amp_1.sum(),amp_1,0)
#     amp_2 = np.where(maxx == amp_2/amp_2.sum(),amp_2,0)

    u_1[:,:,shp:,:]=0
    amp_1[:,shp:,:] = 0
    sup_1[:,shp:,:] = 0
    u_2[:,:,:shp,:] = 0
    amp_2[:,:shp,:] = 0
    sup_2[:,:shp,:] = 0

    sup_1 = np.where(amp_1>0.01*amp_1.max(),1,0)
    sup_2 = np.where(amp_2>0.01*amp_2.max(),1,0)
#     u_1 *= sup_1
#     u_2 *= sup_2
    #################################################
    uu = u_1+u_2
#     plot_u(uu)
#     plt.plot(uu[1,64,:,64])
#     plt.ylabel('$u_y$')
#     plt.xlabel('y')
#     plt.title('discontinuous')
#     plt.show()
    u_3 = ramp_us(u_1,u_2,sup_1,sup_2,coupled)
    
    u_3 = center_u(u_3,(sup_1+sup_2)==1)
#     plt.plot(u3[1,64,:,64])
#     plt.title('continuous')
#     plt.ylabel('$u_y$')
#     plt.xlabel('y')
#     plt.show()
    #################################################
    s1 = copy(sup_1)
    s2 = copy(sup_2)
    if flip:
        s1 = copy(sup_2)
        s2 = copy(sup_1)
    return u_3,amp_1*amp_1_sum+amp_2*amp_2_sum,s1,s2

from copy import copy
def ramp_us(u_1,u_2,sup_1,sup_2,coupled=True):
    shp = sup_1.shape[1]//2
    
    mask = np.zeros(sup_1.shape,dtype=bool)
    mask[shp-10:shp+10,:,shp-10:shp+10] = 1
    sup1 = copy(sup_1)
#     sup1[:,:shp-1,:]=0
#     sup1[~mask]=0
    sup2 = copy(sup_2)
#     sup2[~mask]=0
#     sup2[:,shp+1:,:] = 0
    
    
    left = np.array([0.5*(u[:,shp-1,:]-u[:,shp-2,:])+u[:,shp-1,:] for u in u_1])
    right = np.array([0.5*(u[:,shp,:]-u[:,shp+1,:])+u[:,shp,:] for u in u_2])
#     right = u_2[:,:,shp,:]
    
    offset = [np.mean(left[i][sup1[:,shp-1,:]==1])-np.mean(right[i][sup2[:,shp,:]==1]) for i in range(3)]
    
    offset = np.stack([o*sup_2 for o in offset])
    
    u_2 = np.array([u_2[i]+offset[i] for i in range(3)])*sup_2
    u_3 = u_1+u_2
#     right = np.array([0.5*(u[:,shp,:]-u[:,shp+1,:])+u[:,shp,:] for u in u_2])
    if coupled:
        left = np.array([0.5*(u[:,shp-1,:]-u[:,shp-2,:])+u[:,shp-1,:] for u in u_3])
        right = np.array([0.5*(u[:,shp,:]-u[:,shp+1,:])+u[:,shp,:] for u in u_3])
        u_3[:,:,shp,:] = u_3[:,:,shp,:] + (left-right)/2
        u_3[:,:,shp-1,:] = u_3[:,:,shp-1,:] - (left-right)/2
    return u_3









