from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
from biot_params import *

for i in range(N_wires):
	print("wire " +str(i))
	angle = 2*i*np.pi/N_wires
	x_wire_pos=r_wires*np.cos(angle)
	y_wire_pos=r_wires*np.sin(angle)
	z_wire=np.linspace(zmin,zmax,N)
	x_wire=np.ones_like(z_wire)*x_wire_pos
	y_wire=np.ones_like(z_wire)*y_wire_pos
	mlab.plot3d(x_wire,y_wire,z_wire, np.zeros_like(z_wire), tube_radius=None)

grid_positions=np.loadtxt("grid_positions.dat")
grid_B=np.loadtxt("grid_B.dat")
print(grid_B)
x_display=grid_positions[::display_every_n_point,0]
y_display=grid_positions[::display_every_n_point,1]
z_display=grid_positions[::display_every_n_point,2]
bx_display=grid_B[::display_every_n_point,0]
by_display=grid_B[::display_every_n_point,1]
bz_display=grid_B[::display_every_n_point,2]
#mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display)

for particle_i in range(N_particles):
	try:
		x_positions=np.loadtxt(str(particle_i)+"x_positions.dat")
		y_positions=np.loadtxt(str(particle_i)+"y_positions.dat")
		z_positions=np.loadtxt(str(particle_i)+"z_positions.dat")
		mlab.plot3d(x_positions, y_positions, z_positions, tube_radius=None)
	except IOError:
		break
#mlab.points3d(x_display,y_display,z_display, B_magnitude_squared[::display_every_n_point])
print("Loading finished")
mlab.show()
