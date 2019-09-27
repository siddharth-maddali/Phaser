
# Phase retrieval with Python: `Phaser`

<img align="right" src="1db4_star_trek_phaser_remote_replica.jpg" width=400>

Created by: Siddharth Maddali

This presentation, along with the Python modules, is available at:<br/>
https://github.com/siddharth-maddali/Phaser

   # Introduction
   - Basic Python tutorial of module `Phaser` for BCDI phase retrieval.

   - For data from Beamline 34-ID-C of the Advanced Photon Source

   - Much simpler to use and modify than older Matlab code currently in use.

   - Current dependencies:
       - `numpy` (linear algebra)
       - `scipy` (advanced algorithms, reading Matlab files)
       - `tqdm` (for progress bar displays)
       - `tensorflow`, `tensorflow-gpu`
       - Can be installed in the usual way in Python: `pip install <module>`.

# Recommended Python setup
   - Preferably GNU/Linux or Mac (I don't know much about Windows)

   - Python running in a virtual environment (`virtualenv`) 
       - Recommended setup for Tensorflow 
       - Install instructions [here](https://www.tensorflow.org/install/install_linux#InstallingVirtualenv)

   - Anaconda: very fast, Intel Math Kernel Library (MKL) for backend.
       - Sometimes does not play well with Tensorflow
       - Install instructions [here](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04)

   - Once Python is installed, the iPython shell can be started from the Bash shell with:
       ```
       $ ipython --pylab
       ```
   - All code runs in iPython shell.
       - Line-by-line
       - Running a script:
       ```python
       %run -i <filename>.py # don't forget the %
       ```

# CPU tutorial

## Basic imports


```python
import numpy as np

# import custom modules
import Phaser as ph
import ExperimentalGeometry as exp
import TilePlot as tp

# plotting modules
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Module to read Matlab .mat files
import scipy.io as sio
```


```python
# this has no effect in the iPython shell, to be used in the Jupyter notebook only.
%matplotlib notebook
```

## Loading data set into Python


```python
dataset = sio.loadmat( 'data.mat' )
print( dataset.keys() )
    # NOTE: if you opened this file in Matlab,
    # you'd see only the 'data' variable in the workspace.
    
data = dataset[ 'data' ] # the 3D data is now a numpy array.
print( 'Array size = ', data.shape )
```

    dict_keys(['data', '__globals__', '__header__', '__version__'])
    Array size =  (128, 128, 70)


## Pre-processing the dataset

Assumes that the following are already done:
   1. Stray scattering removed
   1. White-field correction done
   1. Background/hot pixels taken care of


```python
# If necessary, trim the dataset to even dimensions 
# by removing the last image in the stack. Typically 
# this is not necessary for in-plane dimensions since 
# detectors are usually even-pixeled.

# data = data[:,:,:-1]
print( 'Array size = ', data.shape )

maxHere = [ n[0] for n in np.where( data==data.max() ) ]
print( 'Bragg peak initially at: ', maxHere )

# Now centering Bragg peak in the array. If this is not done, you will see 
# a phase ramp in the final reconstruction.
for n in [ 0, 1, 2 ]: 
    data = np.roll( data, data.shape[n]//2 - maxHere[n], axis=n )
    
maxHereNow = [ n[0] for n in np.where( data==data.max() ) ]
print( 'Bragg peak now at: ', maxHereNow )
```

    Array size =  (128, 128, 70)
    Bragg peak initially at:  [69, 71, 38]
    Bragg peak now at:  [64, 64, 35]


## Creating initial support for phase retrieval
   - This gets updated with a shrinkwrap algorithm
   - Initial support should never be bigger than $1/3$ of array size.


```python
shp = data.shape
supInit = np.zeros( shp )
supInit[ #   // means integer division in Python3, as opposed to /, the usual floating point division
    ( shp[0]//2 - shp[0]//6 ):( shp[0]//2 + shp[0]//6 ), 
    ( shp[1]//2 - shp[1]//6 ):( shp[1]//2 + shp[1]//6 ), 
    ( shp[2]//2 - shp[2]//6 ):( shp[2]//2 + shp[2]//6 )
] = 1.
```

## Create a phase retrieval solver object for CPU


```python
PR = ph.Phaser( 
    modulus=np.sqrt( data ), 
    support=supInit.copy() 
        # TODO: remove this; automatically initialize support inside
)
```

## Shrinkwrap
Implemented in the method `PR.Shrinkwrap( sigma, thresh )`. 
   - Object modulus is convolved with a Gaussian (std deviation `sigma`)
   - Thresholded to fraction `thresh` of the maximum value.

## Example recipe for phase retrieval
   - 150 iterations of error reduction (ER), with support-updating every 30 iterations
   - 300 iterations of hybrid input-output (HIO)
   - 100 iterations of solvent-flipping (SF) with support-update every 25 iterations
   - 300 iterations of hybrid input-output (HIO)
   - 450 iterations of ER again, with support-updating every 90 iterations


```python
sigma = np.linspace( 5., 3., 5 )    #
for sig in sigma:                   #  150 iters. of error reduction
    PR.ER( 30, show_progress=True ) #  with shrinkwrap every 30 iters.
    PR.Shrinkwrap( sig, 0.1 )       #
    
PR.HIO( 300, show_progress=True )   #  300 iterations of hybrid I/O

sigma = np.linspace( 3., 2., 4 )
for sig in sigma:                   #
    PR.SF( 25, show_progress=True ) #  100 iterations of solvent flipping, 
    PR.Shrinkwrap( sig, 0.1 )       #  shrinkwrap every 25 iterations.
    
PR.HIO( 300, show_progress=True )   #  300 iterations of hybrid I/O

sigma = np.linspace( 2., 1., 5 )
for sig in sigma:                   #
    PR.ER( 90, show_progress=True ) #  450 iterations of error reduction, 
    PR.Shrinkwrap( sig, 0.1 )       #  shrinkwrap every 90 iterations.
```

     ER: 100%|██████████| 30/30 [00:06<00:00,  4.31it/s]
     ER: 100%|██████████| 30/30 [00:06<00:00,  4.54it/s]
     ER: 100%|██████████| 30/30 [00:06<00:00,  4.55it/s]
     ER: 100%|██████████| 30/30 [00:06<00:00,  4.56it/s]
     ER: 100%|██████████| 30/30 [00:06<00:00,  4.54it/s]
    HIO: 100%|██████████| 300/300 [01:03<00:00,  4.74it/s]
     SF: 100%|██████████| 25/25 [00:09<00:00,  2.68it/s]
     SF: 100%|██████████| 25/25 [00:09<00:00,  2.69it/s]
     SF: 100%|██████████| 25/25 [00:08<00:00,  2.80it/s]
     SF: 100%|██████████| 25/25 [00:09<00:00,  2.68it/s]
    HIO: 100%|██████████| 300/300 [01:03<00:00,  4.76it/s]
     ER: 100%|██████████| 90/90 [00:20<00:00,  4.49it/s]
     ER: 100%|██████████| 90/90 [00:20<00:00,  4.49it/s]
     ER: 100%|██████████| 90/90 [00:19<00:00,  4.52it/s]
     ER: 100%|██████████| 90/90 [00:20<00:00,  4.48it/s]
     ER: 100%|██████████| 90/90 [00:19<00:00,  4.50it/s]


## Extracting image and support from black box


```python
PR.Retrieve() # centers the compact object in array
img = PR.finalImage
sup = PR.finalSupport
```

## Visualizing the result

### Scatterer amplitude

```python
fig, im, ax = tp.TilePlot( 
    ( 
        np.absolute( img[64,29:99,:] ), 
        np.absolute( img[29:99,64,:] ), 
        np.absolute( img[29:99,29:99,30] )                                         
    ), dd
    ( 1, 3 ), 
    ( 9, 3 )
)

# fig.suptitle( 'Amplitude' )
ax[0].set_xlabel( 'YZ' )
ax[1].set_xlabel( 'ZX' )
ax[2].set_xlabel( 'XY' )

fig.savefig( 'images/scattererAmp.jpg' )
```

<img src="images/scattererAmp.jpg">

### Scatterer support

```python
fig, im, ax = tp.TilePlot( 
    ( 
        sup[64,29:99,:], sup[29:99,64,:], sup[29:99,29:99,30]
                                                 
    ), 
    ( 1, 3 ), 
    ( 9, 3 )
)

# fig.suptitle( 'Support' )
ax[0].set_xlabel( 'YZ' )
ax[1].set_xlabel( 'ZX' )
ax[2].set_xlabel( 'XY' )

fig.savefig( 'images/scattererSup.jpg')
```

<img src="images/scattererSup.jpg">

### Scatterer phase

```python
fig, im, ax = tp.TilePlot( 
    ( 
        np.angle( img[64,29:99,:] ), 
        np.angle( img[29:99,64,:] ), 
        np.angle( img[29:99,29:99,30] )
                                                 
    ), 
    ( 1, 3 ), 
    ( 9, 3 )
)

# fig.suptitle( 'Phase' )
ax[0].set_xlabel( 'YZ' )
ax[1].set_xlabel( 'ZX' )
ax[2].set_xlabel( 'XY' )
fig.savefig( 'images/scattererPhs.jpg')
```

<img src="images/scattererPhs.jpg">

### Reconstruction error

```python
plt.figure()
plt.semilogy( PR.Error() )
plt.grid()
plt.xlabel( 'Iteration (n)' )
plt.ylabel( 'Error' )

plt.savefig( 'images/reconError.jpg' )
```

<img src="images/reconError.jpg">

### Transforming from array to real-world coordinates
   - Matlab has better isosurface plotting than Python (for now)
   - Dump transformed object to `.mat` file, then view in Matlab.

`exp.ScatteringGeometry` is a black box that computes the scattering geometry given the experimental parameters in use at Beamline 34-ID-C of the Advanced Photon Source, in the following manner:

```python
sg = exp.ScatteringGeometry( 
    arm=0.65,                     # sample-detector distance, meters
    dtheta=0.01,                  # rocking curve step, degrees
    recipSpaceSteps=data.shape,   # pixel span of data set
    gamma=9.6035,                 # degrees
    delta=33.18675                # degrees
)

Breal, Brecip = sg.getSamplingBases() # get sampling basis into array like this.
```

...then you can dump all the computations into a `.mat` file, and use Matlab's isosurface plotting capabilities (the plotting script `plotParticle.m` is available in this repo).

**Note**: The `ExperimentalGeometry` module is specific to the experimental setup at 34-ID-C end station of the Advanced Photon Source (used for Bragg coherent diffractive imaging). 
For the appropriate module corresponding to other BCDI experiments at the APS (say, 1-ID-E), please [open an issue](https://github.com/siddharth-maddali/Phaser/issues) and I'll see if I can arrange for a module.

```python
sio.savemat( 
    'phasingResult-2.mat', 
    { 
        'img':img, 
        'sup':sup, 
        'data':data, 
        'Breal':Breal, 
        'Brecip':Brecip
    }
)
```

# Recipe strings

`Phaser` can also run an entire phase retrieval recipe by parsing "recipe strings", which are simply Python strings that encode a phase retrieval recipe. 

   - Recipe string for 30 iterations of ER is simply:<br/>`'ER:30'`.  
   
   - Recipe string for 100 ER followed by 25 HIO, followed by 40 SF:<br/>`'ER:100+HIO:25+SF:40'`
   
   - Shrinkwrap recipe string format:<br/>`'SR:<sigma>:<thresh>'`

## Generating the recipe string for the CPU recipe above


```python
recipestr1 = '+'.join( [ 'ER:30+SR:%.2f:0.1'%sig for sig in np.linspace( 5., 3., 5 ) ] )
recipestr2 = '+'.join( [ 'SF:25+SR:%.2f:0.1'%sig for sig in np.linspace( 3., 2., 4 ) ] )
recipestr3 = '+'.join( [ 'ER:90+SR:%.2f:0.1'%sig for sig in np.linspace( 2., 1., 5 ) ] )
recipestr = '+HIO:300+'.join( [ recipestr1, recipestr2, recipestr3 ] )
print( recipestr )
```

    ER:30+SR:5.00:0.1+ER:30+SR:4.50:0.1+ER:30+SR:4.00:0.1+ER:30+SR:3.50:0.1+ER:30+SR:3.00:0.1+HIO:300+SF:25+SR:3.00:0.1+SF:25+SR:2.67:0.1+SF:25+SR:2.33:0.1+SF:25+SR:2.00:0.1+HIO:300+ER:90+SR:2.00:0.1+ER:90+SR:1.75:0.1+ER:90+SR:1.50:0.1+ER:90+SR:1.25:0.1+ER:90+SR:1.00:0.1


## Running the recipe using the string

Start with a new `Phaser` object...


```python
PR_alt = ph.Phaser( 
    modulus=np.sqrt( data ), 
    support=supInit.copy() 
)
PR_alt.runRecipe( recipestr )
```

... to get the same result!

   - Recipe strings are used in parallelized phase retrieval.
   - `tqdm`-enabled progress bars are disabled for this mode of operation.
   - Currently recipe strings are implemented for CPU only.

# GPU tutorial


```python
PR2 = ph.Phaser( 
    modulus=np.sqrt( data ), 
    support=supInit.copy(), 
    gpu=True
).gpusolver
```

## Recipe for GPU phase retrieval (identical to earlier CPU recipe)
   - 150 iterations of error reduction (ER), with support-updating every 30 iterations
   - 300 iterations of hybrid input-output (HIO)
   - 100 iterations of solvent flipping (SF) with support-update every 25 iterations
   - 300 iterations of hybrid input-output (HIO)
   - 450 iterations of ER again, with support-updating every 90 iterations


```python
sigma = np.linspace( 5., 3., 5 )        
for sig in tqdm( sigma, desc=' ER' ):   
    PR2.ER( 30 )                        #  150 iters. of error reduction
    PR2.Shrinkwrap( sig, 0.1 )          #  with shrinkwrap every 30 iters.
              
    
PR2.HIO( 300, show_progress=True )      #  300 iterations of hybrid I/O

sigma = np.linspace( 3., 2., 4 )
for sig in tqdm( sigma, desc=' SF' ):                       
    PR2.SF( 25 )                        #  100 iterations of solvent flipping, 
    PR2.Shrinkwrap( sig, 0.1 )          #  shrinkwrap every 25 iterations.
    
PR2.HIO( 300, show_progress=True )      #  300 iterations of hybrid I/O

sigma = np.linspace( 2., 1., 5 )
for sig in tqdm( sigma, desc=' ER' ):                   
    PR2.ER( 90 )                        #  450 iterations of error reduction, 
    PR2.Shrinkwrap( sig, 0.1 )          #  shrinkwrap every 90 iterations.
```

     ER: 100%|██████████| 5/5 [00:06<00:00,  1.36s/it]
    HIO: 100%|██████████| 300/300 [00:09<00:00, 30.40it/s]
     SF: 100%|██████████| 4/4 [00:04<00:00,  1.09s/it]
    HIO: 100%|██████████| 300/300 [00:09<00:00, 30.78it/s]
     ER: 100%|██████████| 5/5 [00:11<00:00,  2.37s/it]


## Extracting image and support from black box `PR2`


```python
PR2.Retrieve()
img2 = PR2.finalImage
sup2 = PR2.finalSupport

# note that manual centering of the object in the array is
# not necessary, this is already done in the Compute() 
# routine within the GPU module.
```

## Visualizing the result

### Scatterer amplitude

```python
fig, im, ax = tp.TilePlot( 
    ( 
        np.absolute( img2[64,29:99,:] ), 
        np.absolute( img2[29:99,64,:] ), 
        np.absolute( img2[29:99,29:99,30] )
                                                 
    ), 
    ( 1, 3 ), 
    ( 9, 3 )
)

# fig.suptitle( 'Amplitude' )
ax[0].set_xlabel( 'YZ' )
ax[1].set_xlabel( 'ZX' )
ax[2].set_xlabel( 'XY' )
fig.savefig( 'images/scattererAmp_gpu.jpg')
```

<img src="images/scattererAmp_gpu.jpg">

### Scatterer Support

```python
fig, im, ax = tp.TilePlot( 
    ( 
        sup2[64,29:99,:], sup2[29:99,64,:], sup2[29:99,29:99,30]
                                                 
    ), 
    ( 1, 3 ), 
    ( 9, 3 )
)

# fig.suptitle( 'Support' )
ax[0].set_xlabel( 'YZ' )
ax[1].set_xlabel( 'ZX' )
ax[2].set_xlabel( 'XY' )
fig.savefig( 'images/scattererSup_gpu.jpg')
```

<img src="images/scattererSup_gpu.jpg">

### Scatterer phase

```python
fig, im, ax = tp.TilePlot( 
    ( 
        np.angle( img[64,29:99,:] ), 
        np.angle( img[29:99,64,:] ), 
        np.angle( img[29:99,29:99,30] )
                                                 
    ), 
    ( 1, 3 ), 
    ( 9, 3 )
)

# fig.suptitle( 'Phase' )
ax[0].set_xlabel( 'YZ' )
ax[1].set_xlabel( 'ZX' )
ax[2].set_xlabel( 'XY' )
fig.savefig( 'images/scattererPhs_gpu.jpg')
```

<img src="images/scattererPhs_gpu.jpg">

# Upcoming features
   - Parallelization for guided algorithms using `mpi4py`
   - A simple partial coherence correction module
