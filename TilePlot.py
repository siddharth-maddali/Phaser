import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys
from colorsys import hsv_to_rgb
from matplotlib.colors import ListedColormap


def TilePlot( input_arrays, tile_layout, 
        figure_size=( 15, 15 ), 
        color_scales=True,
        log_norm=False
        ):
    if len( input_arrays ) != tile_layout[0]*tile_layout[1]:
        print >> sys.stderr, "Image count inconsistent with layout. "
        return
    image_handles = []
    axis_handles = []
    fig = plt.figure( figsize=figure_size )
    for i in range( tile_layout[0] ):
        for j in range( tile_layout[1] ):
            thisImg = input_arrays[ i*tile_layout[1] + j ]
            ax = plt.subplot2grid( ( tile_layout[0], tile_layout[1] ), ( i, j ), colspan=1 )
            if log_norm==False:
                im = ax.imshow( thisImg, interpolation='none' )
            else:
                im = ax.imshow( thisImg, interpolation='none', norm=LogNorm() )
            image_handles.append( im )
            axis_handles.append( ax )
#             if ( i*tile_layout[1] + j + 1) % tile_layout[1] == 0:
#                 norm = mpl.colors.Normalize( vmin=-np.pi/2., vmax=np.pi/2. )
#                 im.set_norm( norm )
            if color_scales: fig.colorbar( im, ax=ax, fraction=0.046, pad=0.04 ).ax.tick_params( labelsize=20 )

    fig.tight_layout()
    return fig, image_handles, axis_handles

def colorizeImage( img, s=1. ):
    r = np.abs( img )
    arg = np.angle( img )

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = ( r - r.min() ) / ( r.max() - r.min() )
    c = np.vectorize( hsv_to_rgb )( h, l, s ) # --> tuple
    c = np.array( c )  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes( 0, 2)
    temp = c.reshape( c.shape[0]*c.shape[1], -1 )
    cm = ListedColormap( temp[ np.where( np.any( temp > 0., axis=1 ) )[0], : ] )
    return c, cm

def boxBounds( imgShape, shift=1. ):
    return np.array( [ 
        [ shift, shift ], 
        [ shift, imgShape[1]-shift ], 
        [ imgShape[0]-shift, imgShape[1]-shift ], 
        [ imgShape[0]-shift, shift ], 
        [ shift, shift ]
        ]
    ) - 0.5
