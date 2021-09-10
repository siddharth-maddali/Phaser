<img align="left" src="./logo.png" width=300>

# `Phaser`: BCDI Phase retrieval in Python



## Created by: Siddharth Maddali
### Argonne National Laboratory

<a href="https://doi.org/10.5281/zenodo.4305131" style="float: left;"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4305131.svg" alt="DOI"></a>


## MOST RECENT UPDATES
   * Genetic algorithms now properly parallelize on the GPU with MPI. 
   Look at new script `Guided_GPU.py` for GPU usage of genetic algorithms. 
   
## Full changelog

   1. `Core.ImageRestart` now automatically `fftshift`s input arrays.
   1. Now suppressing Tensorflow messages in the command line (comment `os.environ` statement in `Phaser.py` to undo this). 
   1. `GPUModule_Core.ImageRestart` now shadows functionality of `Core.ImageRestart`. 
   1. Slight tweak to `Guided.py` logging.
   1. New file `Guided_GPU.py` runs genetic algorithms using GPU + `mpi4py`. 
   1. ASCII-art of new logo in `Phaser.py`
   1. `PostProcessing.centerObject` now removes phase ramps. 

   # Introduction
   - Basic Python tutorial of module `Phaser` for BCDI phase retrieval.

   - Contains diffraction geometry modules for the 34-ID-C setup at the Advanced Photon Source.
       - Can be easily adapted to other geometries, **please open an issue as a feature request if you need this done for your beamline**.

   - Modular, much simpler to use and modify than existing Matlab legacy code.

   - Current dependencies (as  determined by [`pipreqs`](https://github.com/bndr/pipreqs))
   ```
mmatplotlib==3.1.3
scikit_image==0.16.2
tensorflow==2.2.0
mpi4py==3.0.3
tqdm==4.42.1
numpy==1.18.1
scipy==1.4.1
pyfftw==0.12.0
pyvista==0.32.0
pyvistaqt==0.5.0
skimage==0.0
vtk==9.0.3
```
 
- These modules are based on my current Python environment. 
- All modules can be installed in the usual way: `pip install <module>`.
- The `tensorflow 1.x`-compatible library is available on branch `tensorflow-1.x` of this repo.
       

# Quick start

A full tutorial on using Phaser to reconstruct your BCDI data is available [here](https://nbviewer.jupyter.org/github/siddharth-maddali/Phaser/blob/master/basic_tutorial.ipynb).
