
import run_prep as prep
import format_data as dt
import run_rec as rec
import run_disp as dsp
import create_experiment as ce
import cohere.src_py.beamlines.viz as v
import beamlines.aps_34idc.detectors as det
import beamlines.aps_34idc.disp as disp
import beamlines.aps_34idc.diffractometers as diff
import tifffile as tfile
import Geometry_Correction as GC
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def rotate(work_dir_name,dir_name,specfile,scans,data,step_size,final_dims,plot=False):
    
    
    
    shape = np.array(data.shape)
    half_dim = shape//2
    
    exp_dir = ce.create_exp(dir_name,scans,work_dir_name,specfile=specfile)
    conf_dict = dsp.get_conf_dict(exp_dir)
    params = disp.DispalyParams(conf_dict)
    det_name = params.detector
    diff_name = params.diffractometer
    params.set_instruments(det.create_detector(det_name), diff.create_diffractometer(diff_name))
    g1,g2,B_recip = disp.set_geometry(shape,params)

    print('side length', np.abs(np.linalg.det(B_recip))**(1/3))
    
    half_dim = np.array(data.shape)//2
    if plot:
        fig, ax = plt.subplots(figsize=(15,7),ncols=3)
        ax[0].matshow(data[:,:,shape[2]//2],norm=LogNorm())
        ax[1].matshow(data[:,shape[1]//2,:],norm=LogNorm())
        ax[2].matshow(data[shape[0]//2,:,:].T,norm=LogNorm())
        plt.show()
    


    data = GC.interpolate(data,B_recip,step_size)
    maxHere = np.where(data==data.max())
    for n in [ 0, 1, 2 ]: 
        data = np.roll( data, data.shape[n]//2 - maxHere[n], axis=n )
    
    
    mid1 = np.array(data.shape)//2
    mid2 = np.array(final_dims)//2
   
    data = data[mid1[0]-mid2[0]:mid1[0]+mid2[0],mid1[1]-mid2[1]:mid1[1]+mid2[1],mid1[2]-mid2[2]:mid1[2]+mid2[2]]
    
    data = np.moveaxis(data,0,1)
    
    if plot:
        half_dim = np.array(data.shape)//2
        fig, ax = plt.subplots(figsize=(15,7),ncols=3)
        ax[0].matshow(data[:,:,half_dim[2]],norm=LogNorm())
        ax[1].matshow(data[:,half_dim[1],:],norm=LogNorm())
        ax[2].matshow(data[half_dim[0],:,:].T,norm=LogNorm())
        plt.show()
    
    return data,g2,B_recip

import h5py
def save_as_hdf(peaks,gs,outfile):
    
    data_file = h5py.File(outfile,'w')
    for pk,g in zip(peaks,gs):
        data_file.create_group('%s'%g)
        data_file['%s'%g].create_dataset('reflection',data=g,dtype='i')
        data_file['%s'%g].create_dataset('data',data=pk,dtype='f')
    data_file.close()
    return