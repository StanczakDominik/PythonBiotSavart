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

np.random.seed(1)

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
    grid_B=np.zeros_like(grid_positions)
    for i in range(N_wires):
        # print("wire " +str(i))
        angle = 2*i*np.pi/N_wires
        x_wire_pos=r_wires*np.cos(angle)
        y_wire_pos=r_wires*np.sin(angle)
        z_wire=np.linspace(zmin,zmax,N)
        x_wire=np.ones_like(z_wire)*x_wire_pos
        y_wire=np.ones_like(z_wire)*y_wire_pos

        wire = np.vstack((x_wire, y_wire, z_wire)).T
        wire_gradient = np.gradient(wire)[0]
        wire_length = np.sqrt(np.sum(wire_gradient**2, axis=1))
        wire_gradient[:,0]*=dx
        wire_gradient[:,1]*=dy
        wire_gradient[:,2]*=dz
        for index, wire_segment in enumerate(wire):
            wire_segment_length = wire_gradient[index,:]*wire_length[index]
            rprime=(grid_positions-wire_segment)
            distances = np.sum(rprime**2, axis=1)**(3./2.)
            denominator = np.vstack((distances, distances, distances)).T
            differential=np.cross(wire_segment_length, rprime)/denominator*wire_current*1e7
            low_cutoff_indices=distances<low_cutoff_distance
            indices_cut_off=np.sum(low_cutoff_indices)
            if(indices_cut_off>0):
                differential[low_cutoff_indices, :] = 0
            grid_B += differential*MU/(4*np.pi)
        grid_B[np.isinf(grid_B)] = np.nan
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

def field_interpolation(r):
    distances, indices = mytree.query(r, k=25)
    weights =1./(distances)
    sum_weights=np.sum(weights)
    local_B=grid_B[indices]

    interpolated_BX = np.sum(local_B[:,0]*weights)/sum_weights
    interpolated_BY = np.sum(local_B[:,1]*weights)/sum_weights
    interpolated_BZ = np.sum(local_B[:,2]*weights)/sum_weights
    array = np.array([interpolated_BX,interpolated_BY,interpolated_BZ])
    return array

def boris_step(r, v, dt, calculate_field):
    field = calculate_field(r)
    t = qmratio*field*dt/2.
    vprime = v + np.cross(v,t)
    s = 2*t/(1.+np.sum(t*t))
    v = v + np.cross(vprime,s)
    r+=v*dt
    r=r+v*dt
    return r,v

for particle_i in range(N_particles):
    positions=np.zeros((N_iterations,3))
    velocities=np.zeros((N_iterations,3))
    r=np.random.rand(3)
    r[:2]=r[:2]*(xmax-xmin)+xmin
    r[2] = r[2]*(zmax-zmin)+zmin
    r/=2.
    v=(np.random.rand(3)*(xmax-xmin)+xmin)*velocity_scaling
    print("Moving particle " + str(particle_i), r, v)
    dummy, v = boris_step(r,v,-dt/2., field_interpolation)
    for i in range(N_iterations):
        r,v = boris_step(r,v,dt, field_interpolation)
        #print(i, r,v)
        x_iter, y_iter, z_iter = r
        if x_iter > xmax or x_iter < xmin or y_iter > ymax or y_iter < ymin or z_iter > zmax or z_iter < zmin:
            print("Ran out of the area at i=" + str(i))
            positions=positions[:i,:]
            velocities=velocities[i:,:]
            break
        else:
            positions[i,:]=r
            velocities[i,:]=v
    np.savetxt(folder_name+str(particle_i)+"positions.dat", positions)
    np.savetxt(folder_name+str(particle_i)+"velocities.dat", positions)

print("Finished.")
