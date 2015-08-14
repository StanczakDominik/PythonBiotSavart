from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
import os.path
import scipy.spatial
import sys
import shutil

#Grid parameters
NGRID=50
NZGRID=NGRID

#Region parameters
xmax=0.016
xmin=-xmax
ymax=xmax
ymin=-ymax
zmax=0.1/2.
zmin=-zmax

#Wire parameters for Biot-Savart
N_wires=1
r_wires = 0.008
wire_current=1.e6/N_wires
N=int(NGRID*12.5) #wire segments
low_cutoff_distance=0.0000000000001

display_every_n_point=1 #display every n point vectors of magnetic field

#Simulation parameters
N_iterations=100000
N_particles=10
N_interpolation=8
velocity_scaling=1e6 #for random selection of initial velocity
dt=0.01/velocity_scaling

#Physical constants. All units in SI
electron_charge = -1.60217657e-19
electron_mass = 9.10938291e-31
deuteron_mass = 3.343583719e-27
qmratio=-electron_charge/deuteron_mass
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

def uniform_grid():
    dx = dy = (ymax - ymin)/NGRID
    dz = (zmax-zmin)/NZGRID
    step_size=min([dx, dz])

    x = y = np.arange(xmin, xmax, step_size)
    z = np.arange(zmin, zmax, step_size)
    NGRID_local = len(x)
    NZGRID_local = len(z)
    grid_positions=np.zeros((NGRID_local**2*NZGRID_local,3))
    for ix, vx in enumerate(x):
        for iy, vy in enumerate(y):
            for iz, vz in enumerate(z):
                row = NZGRID_local*NGRID_local*ix+NZGRID_local*iy+iz
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
def exact_ramp_field_grid(N_wires = 1, r_wires=0,mode_name="", N=N):
    print("Calculating field via exact linear ramp formula")
    B0 = MU*wire_current/5./np.pi
    grid_B = np.zeros_like(grid_positions)
    distances = np.sqrt(np.sum(grid_positions[:,:2]**2, axis=1))
    indices_inside = distances < r_wires
    indices_outside=np.logical_not(indices_inside)
    orientation=(grid_positions/np.dstack((distances, distances, distances)))[0]
    grid_B[indices_inside,0] = B0 * distances[indices_inside]/r_wires*orientation[indices_inside,1]
    grid_B[indices_inside,1] = -B0 * distances[indices_inside]/r_wires*orientation[indices_inside,0]
    grid_B[indices_outside,0] = B0 * r_wires / distances[indices_outside]*orientation[indices_outside,1]
    grid_B[indices_outside,1] = -B0 * r_wires / distances[indices_outside]*orientation[indices_outside,0]
    grid_B[:,2] = 0.
    low_cutoff_indices=distances<low_cutoff_distance
    indices_cut_off=np.sum(low_cutoff_indices)
    if(indices_cut_off>0):
        grid_B[low_cutoff_indices, :] = 0
    grid_B[np.isinf(grid_B)] = 0
    grid_B[np.isnan(grid_B)] = 0
    return grid_B

def exact_single_wire_field_grid(N_wires = 1, r_wires=0,mode_name="", N=N):
    print("Calculating field via exact single wire ramp formula")
    B0 = MU*wire_current/2./np.pi
    grid_B = np.zeros_like(grid_positions)
    distances = np.sqrt(np.sum(grid_positions[:,:2]**2, axis=1))
    orientation=(grid_positions/np.dstack((distances, distances, distances)))[0]
    grid_B[:,0] = -B0 / distances*orientation[:,1]
    grid_B[:,1] = B0 / distances*orientation[:,0]
    grid_B[:,2] = 0.
    low_cutoff_indices=distances<low_cutoff_distance
    indices_cut_off=np.sum(low_cutoff_indices)
    if(indices_cut_off>0):
        grid_B[low_cutoff_indices, :] = 0
    grid_B[np.isinf(grid_B)] = 0
    grid_B[np.isnan(grid_B)] = 0
    return grid_B


def biot_savart_field(N_wires=6, r_wires=0.08, wire_current=1e6, mode_name="", N=N):
    print("Calculating field via Biot Savart")
    grid_B=np.zeros_like(grid_positions)
    for i in range(N_wires):
        angle = 2*i*np.pi/N_wires
        x_wire_pos=r_wires*np.cos(angle)
        y_wire_pos=r_wires*np.sin(angle)
        z_wire=np.linspace(zmin,zmax,N)
        x_wire=np.ones_like(z_wire)*x_wire_pos
        y_wire=np.ones_like(z_wire)*y_wire_pos

        wire = np.vstack((x_wire, y_wire, z_wire)).T
        wire_gradient = np.gradient(wire)[0]
        wire_length = np.sqrt(np.sum(wire_gradient**2, axis=1))
        wire_gradient *= np.vstack((wire_length, wire_length, wire_length)).T
        for index, wire_segment in enumerate(wire):

            wire_segment_length = wire_gradient[index,:]
            rprime=(grid_positions-wire_segment)
            distances = np.sum(rprime**2, axis=1)**(3./2.)
            denominator = np.vstack((distances, distances, distances)).T
            differential=np.cross(wire_segment_length, rprime)/denominator*wire_current
            low_cutoff_indices=distances<low_cutoff_distance
            indices_cut_off=np.sum(low_cutoff_indices)
            if(indices_cut_off>0):
                differential[low_cutoff_indices, :] = 0
            grid_B += differential*MU/np.pi/(4)
        grid_B[np.isinf(grid_B)] = np.nan
    grid_B*=N*10 # a correction factor to get the proper result - no idea why!
    return grid_B

def load_field(field_generation_function, field_mode_name="", grid_mode_name="", N_wires=N_wires, r_wires=r_wires, N=N):
    if(os.path.isfile(folder_name+grid_mode_name+field_mode_name+"grid_B.dat")):
        grid_B=np.loadtxt(folder_name+grid_mode_name+field_mode_name+"grid_B.dat")
        print("Loaded grid fields")
    else:
        grid_B=field_generation_function(N_wires=N_wires, r_wires=r_wires, mode_name=field_mode_name)
        np.savetxt(folder_name+grid_mode_name+field_mode_name+"grid_B.dat", grid_B)
        print("Saved grid fields")
    return grid_B

###########Solving fields at particle positions

def field_interpolation(r, N_interpolation=N_interpolation):
    distances, indices = mytree.query(r, k=N_interpolation)
    weights =1./(distances)**8
    sum_weights=np.sum(weights)
    local_B=grid_B[indices]

    interpolated_BX = np.sum(local_B[:,0]*weights)/sum_weights
    interpolated_BY = np.sum(local_B[:,1]*weights)/sum_weights
    interpolated_BZ = np.sum(local_B[:,2]*weights)/sum_weights
    array = np.array([interpolated_BX,interpolated_BY,interpolated_BZ])
    return array

def exact_ramp_field(r, N_interpolation = N_interpolation):
    B=np.zeros(3)
    B0 = MU*wire_current/5./np.pi
    distances = np.sqrt(np.sum(r[:2]**2))
    orientation=r/distances

    if distances<r_wires:
        B[0] = B0 * distances/r_wires*orientation[1]
        B[1] = -B0 * distances/r_wires*orientation[0]
    else:
        B[0] = B0 * r_wires / distances*orientation[1]
        B[1] = -B0 * r_wires / distances*orientation[0]
    B[np.isinf(B)] = 0
    B[np.isnan(B)] = 0
    return B

def exact_single_wire_field(r, N_interpolation = N_interpolation):
    B=np.zeros(3)
    B0 = MU*wire_current/2./np.pi
    distances = np.sqrt(np.sum(r[:2]**2))
    orientation=r/distances

    B[0] = B0 / distances*orientation[1]
    B[1] = -B0 / distances*orientation[0]
    B[np.isinf(B)] = 0
    B[np.isnan(B)] = 0
    return B

############Particle pushing algorithms
def boris_step(r, v, dt, calculate_field, N_interpolation=N_interpolation):
    field = calculate_field(r, N_interpolation = N_interpolation)
    t = qmratio*field*dt/2.
    vprime = v + np.cross(v,t)
    s = 2*t/(1.+np.sum(t*t))
    v = v + np.cross(vprime,s)
    r+=v*dt
    r=r+v*dt
    return r,v

def RK4_step(r,v,dt, calculate_field, N_interpolation=N_interpolation):
	field1 = calculate_field(r, N_interpolation = N_interpolation)
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

def particle_loop(pusher_function, field_calculation_function, mode_name, N_particles,
        N_iterations, save_every_n_iterations=10, save_velocities=False, seed=1,
        N_interpolation=N_interpolation):
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
        print("Moving particle " + str(particle_i))
        print(r)
        print(v)
        if (pusher_function==boris_step):
            dummy, v = pusher_function(r,v,-dt/2., field_calculation_function)
        for i in range(N_iterations):
            r,v = pusher_function(r,v,dt, field_calculation_function, N_interpolation=N_interpolation)
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

compared_trajectories_number=0
def compare_trajectories(exact_trajectory, trial_trajectory):
    global compared_trajectories_number
    variances = calculate_variances(exact_trajectory, trial_trajectory)
    sum_of_variances = np.sum(variances)
    plt.plot(variances)
    plt.title("Total variance = " + str(sum_of_variances))
    plt.ylabel("square difference")
    plt.xlabel("iterations")
    plt.savefig(folder_name + "Trajectory_comparison" + str(compared_trajectories_number)+".png")
    compared_trajectories_number+=1
    plt.clf()
    return sum_of_variances

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
    quiver=mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display, opacity = 0.01)
    mlab.vectorbar(quiver, orientation='vertical')

def display_difference_quiver(grid1, grid2, display_every_n_point=1):
    x_display=grid_positions[::display_every_n_point,0]
    y_display=grid_positions[::display_every_n_point,1]
    z_display=grid_positions[::display_every_n_point,2]
    grid1fig = mlab.figure()
    grid1plot=mlab.quiver3d(x_display, y_display, z_display, grid1[:,0], grid1[:,1], grid1[:,2], opacity = 0.2, figure=grid1fig, colormap="Blues")
    grid2fig = mlab.figure()
    grid2plot=mlab.quiver3d(x_display, y_display, z_display, grid2[:,0], grid2[:,1], grid2[:,2], opacity = 0.2, figure=grid2fig, colormap="Reds")
    scale=np.max(grid2)/np.max(grid1)
    print("======SCALE " + str(scale) + "==========")
    # grid1*=scale
    difference = grid1-grid2
    bx_display=difference[::display_every_n_point,0]
    by_display=difference[::display_every_n_point,1]
    bz_display=difference[::display_every_n_point,2]
    difffig = mlab.figure()
    diffplot=mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display, opacity = 0.2, figure=difffig)
    mlab.quiver3d(x_display, y_display, z_display, grid2[:,0], grid2[:,1], grid2[:,2], opacity = 0.2, figure=grid2fig)
    mlab.colorbar(diffplot)
    mlab.colorbar(grid1plot)
    mlab.colorbar(grid2plot)
    return scale

def display_particles(mode_name="", colormap="Spectral", all_colorbars=False):
    print("Displaying particles from mode " + mode_name)
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
    grid_positions, dx, dy, dz=load_grid(grid_calculation_function=uniform_grid)
    print(grid_positions)
    grid_B=load_field(field_generation_function=exact_ramp_field_grid)

    #####Comparing fields calculated via exact result and biot savart for single wire case
    # grid_exact = exact_single_wire_field_grid(N=N)
    # grid_exact = load_field(field_generation_function=exact_single_wire_field_grid)
    # scales=[]
    # N_list=[5, 10, 15, 20, 25, 30, 40, 50, 60, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 750, 800, 900, 1000]
    # for N in N_list:
    #     print N
    #     scales.append(display_difference_quiver(grid_exact, biot_savart_field(N=N, N_wires = 1, r_wires=0.)))
    # display_difference_quiver(grid_B, grid_exact)
    # print("Biot savart - niebieskie, exact - czerwone")
    # plt.plot(N_list, scales)
    # plt.show()

    #Generate a tree for indexing
    mytree = scipy.spatial.cKDTree(grid_positions)

    #############Set time############################33
    B_magnitude = np.sqrt(np.sum(grid_B**2, axis=1))
    print("Maximum field magnitude = " + str(np.max(B_magnitude)))
    # exact_B_magnitude = np.sqrt(np.sum(grid_exact   **2, axis=1))
    # print("Maximum exact field magnitude = " + str(np.max(exact_B_magnitude)))
    dt_cyclotron = np.abs(0.1*2*np.pi/np.max(B_magnitude)/qmratio)
    print("dt = " + str(dt))
    print("dt cyclotron = " + str(dt_cyclotron))
    dt = dt_cyclotron
    dt=1e-11
    seed=4
    iters=1e7
    N_particles=10


    exact_path = particle_loop(pusher_function=boris_step, field_calculation_function = exact_ramp_field,
        mode_name = "boris_exact", N_particles = N_particles, N_iterations=iters,seed=seed)
    # N_interpolation_list=range(2,50)
    # variances=[]
    # for N_interpolation in N_interpolation_list:
    test_path=particle_loop(pusher_function=boris_step, field_calculation_function = field_interpolation,
            mode_name = "boris_interpolation", N_particles = N_particles, N_iterations=iters,seed=seed,
            N_interpolation=N_interpolation)
        # variance = compare_trajectories(exact_path, test_path)
        # variances.append(variance)
    print("Finished calculation.")
    # plt.plot(N_interpolation_list, variances)
    # plt.show()

    # display_wires(N_wires=1, r_wires=0)
    display_quiver()
    display_particles(mode_name="boris_interpolation", colormap="Blues")
    display_particles(mode_name="boris_exact", colormap="Reds")

    # print("Finished display")
    mlab.show()
