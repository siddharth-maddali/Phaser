#####################################################################
#
#    Guided.py: 
#        Script demonstrating parallelized implementation of 
#        Phaser using genetic algorithms. Implemented with 
#        mpi4py.
#
#        Siddharth Maddali
#        Argonne National Laboratory
#        February 2020
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

# number of generations to breed forward
numGenerations = 30

############# USER EDIT #########################

# define phase retrieval recipe string here
er1 = 'ER:5+'+'+ER:20+'.join( [ 'SR:%.1f:0.1'%sig    for sig in np.linspace( 3., 2., 7 ) ] )+'+ER:5'
sf  = 'ER:5+'+'+ER:20+'.join( [ 'SR:%.1f:0.1'%sig    for sig in np.linspace( 2., 1., 3 ) ] )+'+ER:5'
er2 = 'ER:5+'+'+ER:20+'.join( [ 'SR:1.:0.1'          for sig in np.linspace( 1., 1., 3 ) ] )+'+ER:5'
recipe = er1 + '+HIO:100+' + sf + '+HIO:100+' + er2

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

