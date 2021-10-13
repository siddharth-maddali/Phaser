import numpy as np


def center_u(u,sup):
    for i in range(3):
        u[i,:,:,:] -= np.sum(u[i,:,:,:])/np.sum(sup)
    return u