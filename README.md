<img align="left" src="./logo.png" width=300>

# `Phaser`: BCDI Phase retrieval in Python



## Created by: Siddharth Maddali
### Argonne National Laboratory

<a href="https://doi.org/10.5281/zenodo.4305131" style="float: left;"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4305131.svg" alt="DOI"></a>


   # Introduction
   - Basic Python tutorial of module `Phaser` for BCDI phase retrieval.

   - Contains diffraction geometry modules for the 34-ID-C setup at the Advanced Photon Source.
       - Can be easily adapted to other geometries, **please open an issue as a feature request if you need this done for your beamline**.

   - Modular, much simpler to use and modify than existing Matlab legacy code.

   - Current dependencies (as  determined by [`pipreqs`](https://github.com/bndr/pipreqs))
   ```
tqdm==4.50.2
vtk==9.0.1
matplotlib==3.3.2
pyvistaqt==0.2.0
scipy==1.5.3
scikit_image==0.17.2
tensorflow_gpu==2.3.1
pyvista==0.27.2
numpy==1.19.2
pyFFTW==0.12.0
mpi4py==3.0.3
skimage==0.0
tensorflow==2.4.1
   ```
 
- These modules are based on my current Python environment. 
- All modules can be installed in the usual way: `pip install <module>`.
- The `tensorflow 1.x`-compatible library is available on branch `tensorflow-1.x` of this repo.
       

# Quick start

A full tutorial on using Phaser to reconstruct your BCDI data is available [here](https://nbviewer.jupyter.org/github/siddharth-maddali/Phaser/blob/master/tutorial.ipynb).
