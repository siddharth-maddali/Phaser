#####################################################################
#
#    Guided_PCC.py: 
#        Script demonstrating parallelized implementation of 
#        Phaser+PCC using genetic algorithms. Implemented with 
#        tensorflow-gpu and mpi4py.
#
#    NOTE: 
#        The partial coherence function of the winning worker is
#        shared between all workers for the next iteration.
#
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        September 2021
#        6xlq96aeq@relay.firefox.com
#
#####################################################################

import numpy as np
import scipy.io as sio
import time
import sys

from datetime import datetime
from mpi4py import MPI
from argparse import Namespace

import FigureOfMerit as fom
import Phaser as ph
import warnings

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # worker index
size = comm.Get_size()  # worker pool size

warnings.filterwarnings( 'ignore', category=FutureWarning )
    # this doesn't seem to work

if rank==0:
    print( '\nParallelizing on %d workers. '%size )
    sys.stdout.flush()

############# USER EDIT #########################

# number of generations to breed forward
numGenerations = 30

# phase retrieval recipe used by all parallel workers
wave_1 = '+'.join( [ 'ER:10+SR:%.2f:0.1+ER:5+PCC:10'%sig for sig in np.linspace( 3., 1., 20 ) ] ) # support should have converged pretty well by now
wave_2 = '+'.join( [ 'ER:10+PCC:200+ER:10' ] * 5 )
wave_3 = '+'.join( [ 'ER:200+SR:1.0:0.1' ] * 5 )
wave_4 = 'ER:200'
recipe = '+'.join( [ wave_1, wave_2, wave_3, wave_4 ] )

# load data set
signal = Namespace( **sio.loadmat( 'singleScrewDislocation.mat' ) ).signal

# choose comparison metric for solutions
figureOfMerit = fom.Chi

# output .mat file
outfile = 'guidedResult.mat'

#################################################

# generate initial support
shp = signal.shape
supInit = np.zeros( shp )
supInit[ 
    (shp[0]//2-shp[0]//6):(shp[0]//2+shp[0]//6), 
    (shp[1]//2-shp[1]//6):(shp[1]//2+shp[1]//6), 
    (shp[2]//2-shp[2]//6):(shp[2]//2+shp[2]//6)
] = 1. # i.e. a box 1/3 the size of the array
time.sleep( 1 )

# initialize worker pool
workID = 'Worker-%d'%rank
worker = ph.Phaser( 
    gpu=True,
    modulus=np.sqrt( signal ), 
    support=supInit.copy()
).gpusolver
print( '%s: Online. '%workID )
sys.stdout.flush()

# start parallel phasing
time.sleep( 1 )
for generation in list( range( numGenerations ) ):
    if rank==0:
        print( '___________ Generation %d ____________'%generation )
        sys.stdout.flush()

    tstart = datetime.now()
    worker.runRecipe( recipe )
    worker.Retrieve()
    tstop = datetime.now()

    img = worker.finalImage
    sup = worker.finalSupport
    fm = figureOfMerit( worker.Modulus(), np.sqrt( signal )  )

    print( '%s: Phased in '%workID, tstop-tstart, ', cost = %.2f'%fm  )
    sys.stdout.flush()	

    all_fms = [ rank, fm ]
    all_fms = comm.gather( all_fms, root=0 )
    if rank==0:
        results = np.array( all_fms )
        here = np.where( results[:,1]==results[:,1].min() )
        winning_rank = here[0][0]
    else:
        winning_rank = None
	
    winning_rank = comm.bcast( winning_rank, root=0 )
    #print( '%s: Winning rank = %d'%( workID, winning_rank ) )
    if rank==0 and generation < numGenerations-1:
        print( 'Breeding solution %d into the others...'%winning_rank )
        sys.stdout.flush() 

    if rank==winning_rank:
        winning_img = img.copy()
        new_sup = sup.copy()
    else:
        winning_img = np.empty( img.shape, dtype=np.complex64 )
        new_sup = np.empty( sup.shape, dtype=np.float32 )

    #print( '%s: '%workID, winning_img.dtype, new_sup.dtype )

    comm.Bcast( winning_img, root=winning_rank )
    comm.Bcast( new_sup, root=winning_rank )
    #print( '%s: Debug point...'%workID )

    new_img = np.sqrt( winning_img * img )
    new_sup = ( new_sup + sup > 0.5 ).astype( float ) # the union of two supports
    
    worker.ImageRestart( new_img, new_sup )


if rank==winning_rank:
    print( 'Final solution: worker %d. '%rank )
    sio.savemat( 
        outfile, 
        { 
            'img':new_img, 
            'sup':new_sup
        }
    )
    print( 'Dumped final solution to %s. '%outfile )
    print( 'Done. ' )

