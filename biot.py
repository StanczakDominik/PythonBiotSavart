from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
NGRID=25
x=y=z=np.linspace(-5,5,NGRID)

grid_positions=np.zeros((NGRID**3,3))
for ix, vx in enumerate(x):
	for iy, vy in enumerate(y):
		for iz, vz in enumerate(z):
			row = NGRID**2*ix+NGRID*iy+iz
			grid_positions[row, 0] = vx
			grid_positions[row, 1] = vy
			grid_positions[row, 2] = vz

theta=np.linspace(0,2*np.pi, 1000)
x_wire = 2.5*np.cos(theta*4)
y_wire = 2.5*np.sin(theta*4)
z_wire = np.zeros_like(x_wire)
# z_wire = np.linspace(-4.9,4.9,1000)

# N=100
# z_wire=np.linspace(-5,5,N)
# x_wire=y_wire=np.zeros_like(z_wire)


wire_current = 1
wire = np.vstack((x_wire, y_wire, z_wire)).T
wire_gradient = np.gradient(wire)[0]
wire_length = np.sqrt(np.sum(wire_gradient**2, axis=1))


grid_B=np.zeros_like(grid_positions)
for index, wire_segment in enumerate(wire):
	wire_segment_length = wire_gradient[index,:]*wire_length[index]
	rprime=(grid_positions-wire_segment)
	distances = np.sum(rprime**2, axis=1)**(3./2.)
	denominator = np.vstack((distances, distances, distances)).T
	differential=np.cross(wire_segment_length, rprime)/denominator

	low_cutoff_indices=distances<0.0001
	indices_cut_off=np.sum(low_cutoff_indices)
	if(indices_cut_off>0):
		#print(np.sum(low_cutoff_indices))
		differential[low_cutoff_indices, :] = 0
	grid_B += differential
grid_B*=wire_current*1e7
grid_B[np.isinf(grid_B)] = np.nan

# plt.plot(mins, maxes, "ko")
# plt.show()

display_every_n_point=1
x_display=grid_positions[::display_every_n_point,0]
y_display=grid_positions[::display_every_n_point,1]
z_display=grid_positions[::display_every_n_point,2]
bx_display=grid_B[::display_every_n_point,0]
by_display=grid_B[::display_every_n_point,1]
bz_display=grid_B[::display_every_n_point,2]
B_magnitude_squared=np.sqrt(np.sum(grid_B**2, axis=1))


electron_charge = 1.60217657e-19
electron_mass = 9.10938291e-31
qmratio=electron_charge/electron_mass
dt=0.001
def calculate_field(r):
	field=np.zeros((1,3))
	for index, wire_segment in enumerate(wire):
		wire_segment_length = wire_gradient[index,:]*wire_length[index]
		rprime=(r-wire_segment)
		distances = np.sum(rprime**2)**(3/2)
		denominator = np.vstack((distances, distances, distances)).T
		differential=np.cross(wire_segment_length, rprime)/denominator
		field += differential
		return field[0]
def boris_step(r, v, dt):
	field = calculate_field(r)
	t = qmratio*field*dt/2.
	vprime = v + np.cross(v,t)
	s = 2*t/(1.+np.sum(t*t))
	v = v + np.cross(vprime,s)
	r+=v*dt
	return r,v

x_positions=[]
y_positions=[]
z_positions=[]
energies=[]
r = np.array([2.,1.,0.])
v = np.array([20.,0.,5.])
dummy, v = boris_step(r,v,-dt/2.)
print("Moving particle")
for i in range(100000):
	r,v = boris_step(r,v,dt)
	x_iter, y_iter, z_iter = r
	#print(x_iter, y_iter, z_iter)
	x_positions.append(x_iter)
	y_positions.append(y_iter)
	z_positions.append(z_iter)
	energies.append(np.sum(v**2))
mlab.plot3d(x_wire,y_wire,z_wire)
mlab.plot3d(x_positions, y_positions, z_positions)
mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display)
#mlab.points3d(x_display,y_display,z_display, B_magnitude_squared[::display_every_n_point])
mlab.show()
plt.plot(energies)
plt.show()
