from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
from biot_params import *
import os.path
import scipy.spatial
import sys
import shutil

np.random.seed(2)

number_of_arguments = len(sys.argv)
if number_of_arguments==1:
    folder_name=""
else:
    folder_name=str(sys.argv[1]) +"/"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
	shutil.copy2('biot.py',folder_name)
	shutil.copy2('biot_plotter.py',folder_name)
	shutil.copy2('biot_params.py',folder_name)

if(os.path.isfile(folder_name+"grid_positions.dat")):
	grid_positions=np.loadtxt(folder_name+"grid_positions.dat")
	print("Loaded grid positions")
else:
	x,dx=np.linspace(xmin,xmax,NGRID,retstep=True)
	y,dy=np.linspace(ymin,ymax,NGRID,retstep=True)
	z,dz=np.linspace(zmin,zmax,NZGRID,retstep=True)

	grid_positions=np.zeros((NGRID**2*NZGRID,3))
	for ix, vx in enumerate(x):
		for iy, vy in enumerate(y):
			for iz, vz in enumerate(z):
				row = NZGRID*NGRID*ix+NZGRID*iy+iz
				grid_positions[row, 0] = vx
				grid_positions[row, 1] = vy
				grid_positions[row, 2] = vz

	np.savetxt(folder_name+"grid_positions.dat", grid_positions)
	print("Saved grid positions")
if(os.path.isfile(folder_name+"grid_B.dat")):
	grid_B=np.loadtxt(folder_name+"grid_B.dat")
	print("Loaded grid fields")
else:
	B0 = MU*wire_current/2.*np.pi
	grid_B = np.zeros_like(grid_positions)
	distances = np.sqrt(np.sum(grid_positions[:,:2]**2, axis=1))
	indices_inside = distances < r_wires
	indices_outside=np.logical_not(indices_inside)
	orientation=(grid_positions/np.dstack((distances, distances, distances)))[0]
	print distances.shape, grid_B.shape, orientation.shape
	print grid_B[indices_inside,0]
	print distances[indices_inside]
	print orientation[indices_inside,0]
	grid_B[indices_inside,0] = B0 * distances[indices_inside]/r_wires*orientation[indices_inside,1]
	grid_B[indices_inside,1] = B0 * distances[indices_inside]/r_wires*orientation[indices_inside,0]*(-1)
	grid_B[indices_outside,0] = B0 * r_wires / distances[indices_outside]*orientation[indices_outside,1]
	grid_B[indices_outside,1] = B0 * r_wires / distances[indices_outside]*orientation[indices_outside,0]*(-1)
	grid_B[:,2] = 0.
	print np.isinf(grid_B)
	grid_B[np.isinf(grid_B)] = 0
	grid_B[np.isnan(grid_B)] = 0
	# mlab.plot3d(x_wire,y_wire,z_wire, tube_radius=None)
	np.savetxt(folder_name+"grid_B.dat", grid_B)
	print("Saved grid fields")

# grid_B[np.isinf(grid_B)] = 0

x_display=grid_positions[::display_every_n_point,0]
y_display=grid_positions[::display_every_n_point,1]
z_display=grid_positions[::display_every_n_point,2]
bx_display=grid_B[::display_every_n_point,0]
by_display=grid_B[::display_every_n_point,1]
bz_display=grid_B[::display_every_n_point,2]
B_magnitude_squared=np.sqrt(np.sum(grid_B**2, axis=1))
# mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display)

print(np.max(B_magnitude_squared))
dt_cyclotron = np.abs(0.1*2*np.pi/np.max(B_magnitude_squared)/qmratio)
print("dt = " + str(dt))
print("dt cyclotron = " + str(dt_cyclotron))
dt = dt_cyclotron

mytree = scipy.spatial.cKDTree(grid_positions)

def calculate_field(r):
	distances, indices = mytree.query(r, k=25)
	weights =1./(distances)
	sum_weights=np.sum(weights)
	local_B=grid_B[indices]

	interpolated_BX = np.sum(local_B[:,0]*weights)/sum_weights
	interpolated_BY = np.sum(local_B[:,1]*weights)/sum_weights
	interpolated_BZ = np.sum(local_B[:,2]*weights)/sum_weights
	array = np.array([interpolated_BX,interpolated_BY,interpolated_BZ])
	return array

def boris_step(r, v, dt):
	field = calculate_field(r)
	t = qmratio*field*dt/2.
	vprime = v + np.cross(v,t)
	s = 2*t/(1.+np.sum(t*t))
	v = v + np.cross(vprime,s)
	r+=v*dt
	r=r+v*dt
	return r,v

for particle_i in range(N_particles):
	x_positions=np.zeros(N_iterations)
	y_positions=np.zeros(N_iterations)
	z_positions=np.zeros(N_iterations)
	v_velocities=np.zeros(N_iterations)
	energies=np.zeros(N_iterations)
	r=np.random.rand(3)
	r[:2]=r[:2]*(xmax-xmin)+xmin
	r[2] = r[2]*(zmax-zmin)+zmin
	r/=2.
	v=(np.random.rand(3)*(xmax-xmin)+xmin)*velocity_scaling
	print("Moving particle " + str(particle_i), r, v)
	dummy, v = boris_step(r,v,-dt/2.)
	for i in range(N_iterations):
		r,v = boris_step(r,v,dt)
		#print(i, r,v)
		x_iter, y_iter, z_iter = r
		vz = v[2]
		if x_iter > xmax or x_iter < xmin or y_iter > ymax or y_iter < ymin or z_iter > zmax or z_iter < zmin:
			print("Ran out of the area at i=" + str(i))
			x_positions[i-1:]=x_iter
			y_positions[i-1:]=y_iter
			z_positions[i-1:]=z_iter
			v_velocities[i-1:]=vz
			energies[i-1:]=np.sum(v**2)
			break
		else:
			x_positions[i]=x_iter
			y_positions[i]=y_iter
			z_positions[i]=z_iter
			v_velocities[i]=vz
			energies[i]=np.sum(v**2)

	np.savetxt(folder_name+str(particle_i)+"x_positions.dat", x_positions)
	np.savetxt(folder_name+str(particle_i)+"y_positions.dat", y_positions)
	np.savetxt(folder_name+str(particle_i)+"z_positions.dat", z_positions)
	np.savetxt(folder_name+str(particle_i)+"v_velocities.dat", z_positions)
	np.savetxt(folder_name+str(particle_i)+"energies.dat", energies)

	# plt.plot(energies)
	# plt.title("Energia. Wzgledna wariacja = " +str((max(energies)-min(energies))/((max(energies)+min(energies))/2)))
	# plt.ylim(min(energies), max(energies))
	# plt.savefig(folder_name + str(particle_i)+"energies.png")
	# plt.clf()
	
	# mlab.plot3d(x_positions, y_positions, z_positions, tube_radius=None)

	# plt.plot(v_velocities, "bo-")
	# plt.show()

#####mlab.points3d(x_display,y_display,z_display, B_magnitude_squared[::display_every_n_point])
# mlab.show()

print("Finished.")
