#########################################################
#
#    PlotLikeMatlab: 
#        3D plotting fucntionality for people used to Matlab
#        tired of the lack of such functionality in Python. 
#        Familiar 3D function names, along with image dump 
#        capabilities.
#
#        Siddharth Maddali
#        June 2020
#
#########################################################

import pyvista as pv
import pyvistaqt as pvqt
import numpy as  np
import vtk

from skimage import measure # for marching cubes
from scipy import interpolate
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter

def isosurface( 
    data,                       # complex-valued array 
    isoval=0.5,                 # isovalue of surface plot (this value chosen from erf inflection)
    Br=np.eye( 3 ),             # real-space basis
    offset=[0., 0., 0. ],       # offset of object centroid
    cartesian_frame=False,      # hack to ensure that the default matrix frame is aligned with the Cartesian frame if necessary.
    smooth=True,                # Gaussian kernel smoothing
    plot_handle=None,           # if not provided, plot in new figure
    mc_spacing=( 1., 1., 1. ),  # spacing for marching cubes algorithm
    colorbar=None,
    axes=True,
    cmap='viridis'
):
    """
    isosurface:
    Designed to work similar to Matlab's isosurface function.
    To add a color scale, the 'colorbar' function argument (default None) should look something like:  
    
    colorbar = { 
        'title':'My title', 
        'vertical':True, 
        'interactive':True,
        'label_font_size':10,
        'title_font_size':25, 
        'font_family':'times', 
        'color':[ 0., 0., 0. ] # black lettering, for default white background. In general, grayscale.
    }
    """
    if smooth:
        amp_s, phs_s = tuple( [ gaussian_filter( field( data ), sigma=1. ) for field in [ np.absolute, np.angle ] ] )
    else: 
        amp_s, phs_s = tuple( [ field( data ) for field in [ np.absolute, np.angle ] ] )
    amp_s /= amp_s.max()
    verts, faces, norms, vals = measure.marching_cubes_lewiner( amp_s, level=isoval, spacing=mc_spacing )
    i, j, k = tuple( [ np.arange( n ) for n in data.shape ] )
    phase = interpolate.interpn( ( i, j, k ), phs_s, verts )
    verts -= verts.mean( axis=0, keepdims=True ).repeat( verts.shape[0], axis=0 )
    if cartesian_frame: 
        R = Rotation.from_rotvec( -np.pi/2. * np.array( [ 0., 0., 1. ] ) ).as_matrix()
    else: 
        R = np.eye( 3 )
    verts = ( R @ Br @ verts.T ).T
    verts += np.array( offset ).reshape( 1, -1 ).repeat( verts.shape[0], axis=0 )
    faces = np.concatenate( ( 3*np.ones( ( faces.shape[0], 1 ) ), faces ), axis=1 ).astype( int )
    plotdata = pv.PolyData( verts, np.hstack( faces ) )
    if plot_handle==None: 
        plot_handle=pvqt.BackgroundPlotter()

    plot_handle.set_background( color='white' )
    plot_handle.add_mesh( plotdata, scalars=phase, specular=0.5, cmap=cmap )
    if colorbar != None:
        plot_handle.add_scalar_bar( **colorbar )

    if axes: 
        plot_handle.add_axes( 
            interactive=True, 
            line_width=10, 
            color=[ 0., 0., 0. ], 
            xlabel='', 
            ylabel='', 
            zlabel=''
        )
    return plot_handle
   
def savefig3d( plt3d, file_prefix ):
    expo = vtk.vtkGL2PSExporter()
    expo.SetFilePrefix( file_prefix )
    expo.SetFileFormatToPDF()
    expo.SetRenderWindow( plt3d.ren_win )
    expo.Write()
    return
    
