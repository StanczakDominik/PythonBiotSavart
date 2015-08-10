from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
from biot_params import *
import sys
import shutil
import os.path, os

show_quiver=True

number_of_arguments = len(sys.argv)
if number_of_arguments==1:
    folder_name="RK4"
else:
    folder_name=str(sys.argv[1]) +"/RK4"
    if not os.path.isdir(folder_name):
        sys.exit("Directory does not exist")

for i in range(N_wires):
    angle = 2*i*np.pi/N_wires
    x_wire_pos=r_wires*np.cos(angle)
    y_wire_pos=r_wires*np.sin(angle)
    z_wire=np.linspace(zmin,zmax,N)
    x_wire=np.ones_like(z_wire)*x_wire_pos
    y_wire=np.ones_like(z_wire)*y_wire_pos
    mlab.plot3d(x_wire,y_wire,z_wire, np.zeros_like(z_wire), tube_radius=None)

if show_quiver:
    grid_positions=np.loadtxt(folder_name+"grid_positions.dat")
    grid_B=np.loadtxt(folder_name+"grid_B.dat")
    x_display=grid_positions[::display_every_n_point,0]
    y_display=grid_positions[::display_every_n_point,1]
    z_display=grid_positions[::display_every_n_point,2]
    bx_display=grid_B[::display_every_n_point,0]
    by_display=grid_B[::display_every_n_point,1]
    bz_display=grid_B[::display_every_n_point,2]
    mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display, opacity = 0.1)


for particle_i in range(N_particles):
    try:
        print("Loading particle " + str(particle_i))
        positions=np.loadtxt(folder_name+str(particle_i)+"positions.dat")
        x_positions=positions[:,0]
        y_positions=positions[:,1]
        z_positions=positions[:,2]
        time=np.arange(len(z_positions))
        plot=mlab.plot3d(x_positions, y_positions, z_positions, time, colormap='Spectral', tube_radius=None)
        colorbar=mlab.colorbar(plot)
    except IOError:
        print("Failed to load particle " + str(particle_i))
        break
#mlab.points3d(x_display,y_display,z_display, B_magnitude_squared[::display_every_n_point])
print("Loading finished")
mlab.show()
