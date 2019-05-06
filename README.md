
# Phase retrieval with Python: `Phaser`

<img align="right" src="1db4_star_trek_phaser_remote_replica.jpg" width=400>

## Created by: Siddharth Maddali

### This presentation, along with the Python modules, is available at:<br/>
https://github.com/siddharth-maddali/Phaser

   ## Introduction
   - Basic Python tutorial of module `Phaser` for BCDI phase retrieval.

   - For data from Beamline 34-ID

   - Much simpler to use and modify than legacy Matlab code.

   - Current dependencies:
       - `numpy` (linear algebra)
       - `scipy` (advanced algorithms, reading Matlab files)
       - `tqdm` (for progress bar displays)
       - `tensorflow`, `tensorflow-gpu`
       - Can be installed in the usual way in Python: `pip install <module>`.

## Recommended Python setup
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

    dict_keys(['__version__', '__header__', '__globals__', 'data'])
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
Implemented in the method `PR.ShrinkWrap( sigma, thresh )`. 
   1. Object modulus is convolved with a Gaussian (std deviation `sigma`)
   1. Thresholded to fraction `thresh` of the maximum value.

## Example recipe for phase retrieval
   - 150 iterations of error reduction (ER), with support-updating every 30 iterations
   - 300 iterations of hybrid input-output (HIO)
   - 100 iterations of solvent-flipping (SF) with support-update every 25 iterations
   - 300 iterations of hybrid input-output (HIO)
   - 450 iterations of ER again, with support-updating every 90 iterations


```python
sigma = np.linspace( 5., 3., 5 )    #
for sig in sigma:                   #  150 iters. of error reduction
    PR.ErrorReduction( 30 )         #  with shrinkwrap every 30 iters.
    PR.ShrinkWrap( sig, 0.1 )       #
    
PR.HybridIO( 300 )

sigma = np.linspace( 3., 2., 4 )
for sig in sigma:                   #
    PR.SolventFlipping( 25 )        #  100 iterations of solvent flipping, 
    PR.ShrinkWrap( sig, 0.1 )       #  shrinkwrap every 25 iterations.
    
PR.HybridIO( 300 )                  #  300 iterations of hybrid I/O

sigma = np.linspace( 2., 1., 5 )
for sig in sigma:                   #
    PR.ErrorReduction( 90 )         #  450 iterations of error reduction, 
    PR.ShrinkWrap( sig, 0.1 )       #  shrinkwrap every 90 iterations.
```

     ER: 100%|██████████| 30/30 [00:07<00:00,  4.02it/s]
     ER: 100%|██████████| 30/30 [00:06<00:00,  4.30it/s]
     ER: 100%|██████████| 30/30 [00:07<00:00,  4.28it/s]
     ER: 100%|██████████| 30/30 [00:07<00:00,  4.24it/s]
     ER: 100%|██████████| 30/30 [00:07<00:00,  4.25it/s]
    HIO: 100%|██████████| 300/300 [01:15<00:00,  3.99it/s]
     SF: 100%|██████████| 25/25 [00:11<00:00,  2.18it/s]
     SF: 100%|██████████| 25/25 [00:11<00:00,  2.18it/s]
     SF: 100%|██████████| 25/25 [00:11<00:00,  2.23it/s]
     SF: 100%|██████████| 25/25 [00:12<00:00,  2.02it/s]
    HIO: 100%|██████████| 300/300 [01:16<00:00,  3.94it/s]
     ER: 100%|██████████| 90/90 [00:20<00:00,  4.30it/s]
     ER: 100%|██████████| 90/90 [00:20<00:00,  4.38it/s]
     ER: 100%|██████████| 90/90 [00:22<00:00,  4.05it/s]
     ER: 100%|██████████| 90/90 [00:21<00:00,  4.14it/s]
     ER: 100%|██████████| 90/90 [00:22<00:00,  4.07it/s]


## Extracting image and support from black box


```python
PR.Retrieve() # centers the compact object in array
img = PR.finalImage
sup = PR.finalSupport
```


## Scatterer amplitude
<img src="images/scattererAmp.jpg">


## Scatterer support
<img src="images/scattererSup.jpg">


## Scatterer phase
<img src="images/scattererPhs.jpg">


## Reconstruction error

<img src="images/reconError.jpg">

## Transforming to lab coordinates
   - Matlab has better isosurface plotting than Python (for now)
   - Dump transformed object to `.mat` file, then view in Matlab.

`exp.ScatteringGeometry` is a black box that computes the scattering geometry given the usual input parameters, in this manner:


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

...then you can dump all the computations into a `.mat` file, and use Matlab's isosurface plotting capabilities.

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

## Creating a phaser object for GPU


```python
PR2 = ph.Phaser( 
    modulus=np.sqrt( data ), 
    support=supInit.copy(), 
    gpu=True
).gpusolver
```

## Similar recipe for GPU phase retrieval
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

     ER: 100%|██████████| 5/5 [00:06<00:00,  1.22s/it]
    HIO: 100%|██████████| 300/300 [00:13<00:00, 22.12it/s]
     SF: 100%|██████████| 4/4 [00:05<00:00,  1.38s/it]
    HIO: 100%|██████████| 300/300 [00:12<00:00, 23.48it/s]
     ER: 100%|██████████| 5/5 [00:16<00:00,  3.32s/it]


## Extracting image and support from black box `PR2`


```python
PR2.Retrieve()
img2 = PR2.finalImage
sup2 = PR2.finalSupport

# note that manual centering of the object in the array is
# not necessary, this is already done in the Compute() 
# routine within the GPU module.
```


<h2>Scatterer amplitude</h2>

<img src="images/scattererAmp_gpu.jpg">


## Scatterer support

<img src="images/scattererSup_gpu.jpg">


## Scatterer phase

<img src="images/scattererPhs_gpu.jpg">

# Coming up...
   - Major refactoring to make the code more efficient
   - A simple partial coherence correction module
