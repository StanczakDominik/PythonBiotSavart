from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
import os.path
import scipy.spatial
import sys
import shutil

NGRID=75
NZGRID=NGRID*5

xmax=0.016
xmin=-xmax
ymax=xmax
ymin=-ymax
zmax=0.1/2.
zmin=-zmax

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

qmratio=-electron_charge/deuteron_mass
velocity_scaling=1e6
dt=0.01/velocity_scaling
MU=4e-7*np.pi

######Folder name management#################
number_of_arguments = len(sys.argv)
if number_of_arguments==1:
    folder_name=""
else:
    folder_name=str(sys.argv[1]) +"/"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    shutil.copy2('biot.py',folder_name)
    shutil.copy2('plot.py',folder_name)

#########Grid functions####################

def nonuniform_grid():
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
    return grid_positions, dx, dy, dz

def uniform_grid():             #doesn't quite work yet
    dx = dy = (ymax - ymin)/NGRID
    dz = (zmax-zmin)/NZGRID
    step_size=min([dx, dz])

    x = y = np.arange(xmin, xmax, step_size)
    z = np.arange(zmin, zmax, step_size)
    grid_positions=np.zeros((NGRID**2*NZGRID,3))
    for ix, vx in enumerate(x):
        for iy, vy in enumerate(y):
            for iz, vz in enumerate(z):
                row = NZGRID*NGRID*ix+NZGRID*iy+iz
                grid_positions[row, 0] = vx
                grid_positions[row, 1] = vy
                grid_positions[row, 2] = vz
    return grid_positions, dx, dy, dz

def load_grid(grid_calculation_function, mode_name=""):
    if(os.path.isfile(folder_name+mode_name+"grid_positions.dat")):
        grid_positions=np.loadtxt(folder_name+mode_name+"grid_positions.dat")
        dx, dy, dz = np.loadtxt(folder_name + mode_name + "step_sizes.dat")
        print("Loaded grid positions")
    else:
        grid_positions, dx, dy, dz = grid_calculation_function()
        np.savetxt(folder_name + mode_name + "grid_positions.dat", grid_positions)
        step_sizes = np.array((dx, dy, dz))
        np.savetxt(folder_name + mode_name + "step_sizes.dat", step_sizes)
        print("Saved grid positions")
    return grid_positions, dx, dy, dz

#########Magnetic field functions#########
def exact_single_wire_field_grid(N_wires = 1, r_wires=0,mode_name=""):
    print("Calculating field via exact single wire linear ramp formula")
    B0 = MU*wire_current/5.*np.pi
    grid_B = np.zeros_like(grid_positions)
    distances = np.sqrt(np.sum(grid_positions[:,:2]**2, axis=1))
    indices_inside = distances < r_wires
    indices_outside=np.logical_not(indices_inside)
    orientation=(grid_positions/np.dstack((distances, distances, distances)))[0]
    grid_B[indices_inside,0] = B0 * distances[indices_inside]/r_wires*orientation[indices_inside,1]
    grid_B[indices_inside,1] = B0 * distances[indices_inside]/r_wires*orientation[indices_inside,0]*(-1)
    grid_B[indices_outside,0] = B0 * r_wires / distances[indices_outside]*orientation[indices_outside,1]
    grid_B[indices_outside,1] = B0 * r_wires / distances[indices_outside]*orientation[indices_outside,0]*(-1)
    grid_B[:,2] = 0.
    grid_B[np.isinf(grid_B)] = 0
    grid_B[np.isnan(grid_B)] = 0
    return grid_B

def biot_savart_field(N_wires=6, r_wires=0.08, mode_name=""):
    print("Calculating field via Biot Savart")
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
    return grid_B

def load_field(field_generation_function, field_mode_name="", grid_mode_name=""):
    if(os.path.isfile(folder_name+grid_mode_name+field_mode_name+"grid_B.dat")):
        grid_B=np.loadtxt(folder_name+grid_mode_name+field_mode_name+"grid_B.dat")
        print("Loaded grid fields")
    else:
        grid_B=field_generation_function(N_wires=N_wires, r_wires=r_wires, mode_name=field_mode_name)
        np.savetxt(folder_name+grid_mode_name+field_mode_name+"grid_B.dat", grid_B)
        print("Saved grid fields")
    return grid_B

###########Solving fields at particle positions

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

def exact_single_wire_field(r):
    B=np.zeros(3)
    B0 = MU*wire_current/5.*np.pi
    distances = np.sqrt(np.sum(r[:2]**2))
    orientation=r/distances

    if distances<r_wires:
        B[0] = B0 * distances/r_wires*orientation[1]
        B[1] = B0 * distances/r_wires*orientation[0]*(-1)
    else:
        B[0] = B0 * r_wires / distances*orientation[1]
        B[1] = B0 * r_wires / distances*orientation[0]*(-1)
    B[np.isinf(B)] = 0
    B[np.isnan(B)] = 0
    return B

############Particle pushing algorithms
def boris_step(r, v, dt, calculate_field):
    field = calculate_field(r)
    t = qmratio*field*dt/2.
    vprime = v + np.cross(v,t)
    s = 2*t/(1.+np.sum(t*t))
    v = v + np.cross(vprime,s)
    r+=v*dt
    r=r+v*dt
    return r,v

def RK4_step(r,v,dt, calculate_field):
	field1 = calculate_field(r)
	k1v = qmratio*np.cross(v,field1)
	k1r = v

	r2 = r + k1r*dt/2.
	v2 = v + k1v*dt/2.
	field2 = calculate_field(r2)
	k2v = qmratio*np.cross(v2,field2)
	k2r = v2

	r3 = r + k2r*dt/2.
	v3 = v + k2v*dt/2.
	field3 = calculate_field(r3)
	k3v = qmratio*np.cross(v3, field3)
	k3r = v3

	r4 = r + k3r*dt
	v4 = v + k3v*dt
	field4 = calculate_field(r4)
	k4v = qmratio*np.cross(v4, field4)
	k4r = v4

	r += dt/6.*(k1r+2*(k2r+k3r)+k4r)
	v += dt/6.*(k1v+2*(k2v+k3v)+k4v)

	return r,v

def particle_loop(pusher_function, field_calculation_function, mode_name, N_particles, N_iterations, save_every_n_iterations=10, save_velocities=False, seed=1):
    np.random.seed(seed)
    N_iterations=int(N_iterations)
    N_particles=int(N_particles)
    print("Beginning push...")
    for particle_i in range(N_particles):
        positions=np.zeros((N_iterations,3))
        if save_velocities:velocities=np.zeros((N_iterations,3))
        r=np.random.rand(3)
        r[:2]=r[:2]*(xmax-xmin)+xmin
        r[2] = r[2]*(zmax-zmin)+zmin
        r/=2.
        v=np.zeros(3)
        v[:2]=(np.random.rand(2)*(xmax-xmin)+xmin)*velocity_scaling
        v[2]=(np.random.rand()*(zmax-zmin)+zmin)*velocity_scaling
        print("Moving particle " + str(particle_i), r, v)
        if (pusher_function==boris_step):
            dummy, v = pusher_function(r,v,-dt/2., field_calculation_function)
        for i in range(N_iterations):
            r,v = pusher_function(r,v,dt, field_calculation_function)
            x_iter, y_iter, z_iter = r
            if x_iter > xmax or x_iter < xmin or y_iter > ymax or y_iter < ymin or z_iter > zmax or z_iter < zmin:
                print("Ran out of the area at i=" + str(i))
                positions=positions[:i,:]
                if save_velocities: velocities=velocities[i:,:]
                break
            else:
                positions[i,:]=r
                if save_velocities: velocities[i,:]=v
        #TODO: make it only save every n interations in the first place to lower running memory reqs
        np.savetxt(folder_name+mode_name+str(particle_i)+"positions.dat", positions[::save_every_n_iterations])
        if save_velocities: np.savetxt(folder_name+mode_name+str(particle_i)+"velocities.dat", velocities[::save_every_n_iterations])
    print("Push finished.")
    return positions

##########################Diagnostics

def calculate_variances(exact_trajectory, trial_trajectory):
    lengths = (len(exact_trajectory), len(trial_trajectory))
    min_len=min(lengths)
    return np.sum((exact_trajectory[:min_len]-trial_trajectory[:min_len])**2, axis=1)

def compare_trajectories(exact_trajectory, trial_trajectory):
    variances = calculate_variances(exact_trajectory, trial_trajectory)
    sum_of_variances = np.sum(variances)
    plt.plot(variances)
    plt.title("Total variance = " + str(sum_of_variances))
    plt.ylabel("square difference")
    plt.xlabel("iterations")
    plt.savefig(folder_name + "Trajectory_comparison.png")
    plt.clf()

#####################Visualization


def display_wires(N_wires=1, r_wires=0):
    print("Loading wires")
    for i in range(N_wires):
        angle = 2*i*np.pi/N_wires
        x_wire_pos=r_wires*np.cos(angle)
        y_wire_pos=r_wires*np.sin(angle)
        z_wire=np.linspace(zmin,zmax,N)
        x_wire=np.ones_like(z_wire)*x_wire_pos
        y_wire=np.ones_like(z_wire)*y_wire_pos
        mlab.plot3d(x_wire,y_wire,z_wire, np.zeros_like(z_wire), tube_radius=None)

def display_quiver(grid_mode_name="", field_mode_name="", display_every_n_point=1):
    print("Loading quiver")
    grid_positions=np.loadtxt(folder_name+grid_mode_name+"grid_positions.dat")
    grid_B=np.loadtxt(folder_name+grid_mode_name+field_mode_name+"grid_B.dat")
    x_display=grid_positions[::display_every_n_point,0]
    y_display=grid_positions[::display_every_n_point,1]
    z_display=grid_positions[::display_every_n_point,2]
    bx_display=grid_B[::display_every_n_point,0]
    by_display=grid_B[::display_every_n_point,1]
    bz_display=grid_B[::display_every_n_point,2]
    mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display, opacity = 0.01)

def display_particles(mode_name="", colormap="Spectral", all_colorbars=False):
    particle_i=0
    while True:
        particle_file_name=folder_name+mode_name+str(particle_i)+"positions.dat"
        if(os.path.isfile(particle_file_name)):
            positions=np.loadtxt(particle_file_name)
            x_positions=positions[:,0]
            y_positions=positions[:,1]
            z_positions=positions[:,2]
            time=np.arange(len(z_positions))
            plot=mlab.plot3d(x_positions, y_positions, z_positions, time, colormap=colormap, tube_radius=None)
            print("Loaded particle " + str(particle_i) + " for display")
        else:
            print("Failed to load particle " + str(particle_i))
            break
        particle_i+=1
        if all_colorbars: mlab.colorbar(plot)
    if not all_colorbars: colorbar=mlab.colorbar(plot)
    print("Loading particle display finished")

if __name__ =="__main__":
    grid_positions, dx, dy, dz=load_grid(mode_name="nonuniform", grid_calculation_function=nonuniform_grid)
    grid_B=load_field(field_generation_function=biot_savart_field, grid_mode_name="nonuniform", field_mode_name="biot")
    mytree = scipy.spatial.cKDTree(grid_positions)

    #############Set time############################33
    B_magnitude = np.sqrt(np.sum(grid_B**2, axis=1))
    print("Maximum field magnitude = " + str(np.max(B_magnitude)))
    dt_cyclotron = np.abs(0.1*2*np.pi/np.max(B_magnitude)/qmratio)
    print("dt = " + str(dt))
    print("dt cyclotron = " + str(dt_cyclotron))
    dt = dt_cyclotron

    compare_trajectories(
        particle_loop(pusher_function=boris_step, field_calculation_function = field_interpolation,
        mode_name = "boris", N_particles = 1, N_iterations=1e5),
    particle_loop(pusher_function=RK4_step, field_calculation_function = field_interpolation,
        mode_name = "RK", N_particles = 1, N_iterations=1e5)
        )

    print("Finished calculation.")

    from plot import *
