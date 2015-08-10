import numpy as np
NGRID=25

xmax=0.016
xmin=-xmax
ymax=xmax
ymin=-ymax
zmax=0.1
zmin=0.

N_wires=6
r_wires = 0.008
wire_current=1.e6/N_wires
N=100 #wire segments
display_every_n_point=1
low_cutoff_distance=0.0000001

N_iterations=100000
N_particles=10

electron_charge = -1.60217657e-19
electron_mass = 9.10938291e-31

deuteron_mass = 3.343583719e-27

#3 keV
#thermal velocity?

qmratio=electron_charge/deuteron_mass
velocity_scaling=1e6
dt=0.01/velocity_scaling
MU=4e-7*np.pi
