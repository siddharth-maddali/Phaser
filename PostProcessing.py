##############################################################
#   This module contains post-processing routines for after the 
#   GPU iterations.
#	
#   Siddharth Maddali
#   Argonne National Laboratory
#   2018
##############################################################

import numpy as np

def centerObject( img, sup ):
    
    imgC = img.copy()
    supC = sup.copy()
    span = np.where( supC > 0.5 )
    for n in list( range( len( span ) ) ):
        if 1+span[n].max()-span[n].min()==supC.shape[n]: # i.e. obj split across periodic bnd
            imgC = np.roll( imgC, imgC.shape[n]//2, axis=n )
            supC = np.roll( supC, supC.shape[n]//2, axis=n )
#    span = np.where( supC > 0.5 )

    y, x, z = np.meshgrid( 
        np.arange( supC.shape[0] ), 
        np.arange( supC.shape[1] ), 
        np.arange( supC.shape[2] ) 
    )
    sm = supC.sum()
    xc = ( x*supC ).sum() // sm
    yc = ( y*supC ).sum() // sm
    zc = ( z*supC ).sum() // sm
    c = [ int( xc ), int( yc ), int( zc ) ]

    for n in list( range( 3 ) ):
        imgC = np.roll( imgC, imgC.shape[n]//2 - c[n], axis=n )
        supC = np.roll( supC, supC.shape[n]//2 - c[n], axis=n )


    return imgC, supC


