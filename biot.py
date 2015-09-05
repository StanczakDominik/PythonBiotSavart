from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
import os.path
import scipy.spatial
import sys
import shutil
import h5py
import pdb

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
    shutil.copy2('plot.py',folder_name)

def append_to_file(file, array):
    # print("Array begins with")
    # print(array[:3,:])
    # print("Array ends with")
    # print(array[-3:,:])
    # length = len(array)
    with open(file, 'ab') as file:
        np.savetxt(file, array)
    # print("Successfully appended an array of length %d" % length)

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
def exact_ramp_field_grid(grid_positions, N_wires = 1, r_wires=0,mode_name="", N=N):
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

def exact_single_wire_field_grid(grid_positions, N_wires = 1, r_wires=0,mode_name="", N=N):
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


def biot_savart_field_grid(grid_positions, N_wires=6, r_wires=0.08, wire_current=1e6, mode_name="", N=N):
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

def load_field(field_generation_function, grid_positions, field_mode_name="", grid_mode_name="", N_wires=N_wires, r_wires=r_wires, N=N):
    if(os.path.isfile(folder_name+grid_mode_name+field_mode_name+"grid_B.dat")):
        grid_B=np.loadtxt(folder_name+grid_mode_name+field_mode_name+"grid_B.dat")
        print("Loaded grid fields")
    else:
        grid_B=field_generation_function(N_wires=N_wires, r_wires=r_wires, mode_name=field_mode_name, grid_positions=grid_positions)
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
    B=np.zeros_like(r)
    B0 = MU*wire_current/5./np.pi
    distances = np.sqrt(np.sum(r[:,:2]**2,axis=1))[:,np.newaxis]
    orientation=r/distances
    index = distances<r_wires
    not_index=np.logical_not(index)
    B[[index],0]=B0*distances/r_wires*orientation[[index],1]
    B[[index],1]=-B0*distances/r_wires*orientation[[index],0]
    B[[not_index],0]=B0*r_wires/distances*orientation[[not_index],1]
    B[[not_index],1]=-B0*r_wires/distances*orientation[[not_index],0]
    B[np.isinf(B)] = 0
    B[np.isnan(B)] = 0
    return B

def exact_single_wire_field(r, N_interpolation = N_interpolation):
    B=np.zeros_like(r)
    B0 = MU*wire_current/2./np.pi
    distances = np.sqrt(np.sum(r[:,:2]**2, axis=1))
    orientation=r/distances

    B[:,0] = B0 / distances*orientation[:,1]
    B[:,1] = -B0 / distances*orientation[:,0]
    B[np.isinf(B)] = 0
    B[np.isnan(B)] = 0
    return B

def test_z_field(r, N_interpolation = N_interpolation):
    B = np.zeros_like(r)
    B[:,2]=1.
    return B

############Particle pushing algorithms
def boris_step(r, v, dt, calculate_field, N_interpolation=N_interpolation):
    field = calculate_field(r, N_interpolation = N_interpolation)
    t = qmratio*field*dt/2.
    vprime = v + np.cross(v,t, axis=1)
    s = 2*t/(1.+np.sum(t*t))
    dv = np.cross(vprime,s, axis=1)
    v += dv
    dr=v*dt
    r+=dr
    return r,v

def RK4_step(r,v,dt, calculate_field, N_interpolation=N_interpolation):
    field1 = calculate_field(r, N_interpolation = N_interpolation)
    k1v = qmratio*np.cross(v,field1, axis=1)
    k1r = v

    r2 = r + k1r*dt/2.
    v2 = v + k1v*dt/2.
    field2 = calculate_field(r2)
    k2v = qmratio*np.cross(v2,field2, axis=1)
    k2r = v2

    r3 = r + k2r*dt/2.
    v3 = v + k2v*dt/2.
    field3 = calculate_field(r3)
    k3v = qmratio*np.cross(v3, field3, axis=1)
    k3r = v3

    r4 = r + k3r*dt
    v4 = v + k3v*dt
    field4 = calculate_field(r4)
    k4v = qmratio*np.cross(v4, field4, axis=1)
    k4r = v4

    r += dt/6.*(k1r+2*(k2r+k3r)+k4r)
    v += dt/6.*(k1v+2*(k2v+k3v)+k4v)

    return r,v

def particle_loop(pusher_function, field_calculation_function, mode_name, N_particles,
        N_iterations, save_every_n_iterations=10, save_velocities=False, seed=1,
        N_interpolation=N_interpolation, continue_run=False, dt=dt, preset_r=None, preset_v=None):
    print("""

    Running simulation of mode %s with %d particles.
    Pusher algorithm is %s.
    Field is calculated using %s.
    %d iterations with timestep %e, %d particles.
    Saves data every %d iterations. Random seed is %d.""" %(mode_name, N_particles,
     pusher_function.__name__,
     field_calculation_function.__name__,
     N_iterations, dt, N_particles,
     save_every_n_iterations, seed))
    if continue_run: print("    This is a continued run.")
    if save_velocities: print("    Velocities are saved.")
    print("\n")
    np.random.seed(seed)
    N_iterations=int(N_iterations)
    N_particles=int(N_particles)
    Dump_every_N_iterations=N_iterations/100
    total_data_length = int(N_iterations/save_every_n_iterations)
    with h5py.File(folder_name+mode_name+".hdf5", 'w') as loop_file:
        loop_file.attrs['pusher_function'] = str(pusher_function)
        loop_file.attrs['field_calculation_function'] = str(field_calculation_function)
        loop_file.attrs['N_particles'] = N_particles
        loop_file.attrs['N_iterations'] = N_iterations
        loop_file.attrs['save_every_n_iterations'] = save_every_n_iterations
        loop_file.attrs['seed'] = seed
        loop_file.attrs['N_interpolation'] = N_interpolation
        loop_file.attrs['continue_run'] = continue_run
        loop_file.attrs['dt'] = dt

        if preset_r is not None:
            loop_file.attrs['preset_r'] = preset_r
        else:
            loop_file.attrs['preset_r'] = "No preset r"
        if preset_v is not None:
            loop_file.attrs['preset_v'] = preset_v
        else:
            loop_file.attrs['preset_v'] = "No preset v"


        positions_dataset = loop_file.create_dataset("positions", (N_particles,3,total_data_length), dtype='float')
        velocities_dataset = loop_file.create_dataset("velocities", (N_particles,3,total_data_length), dtype='float')

        if preset_r is not None:
            #if there are preset initial conditions (N_particles should be 1 for this case)
            r=preset_r
        else:
            #generate initial conditions at random
            r=np.random.random((N_particles,3))
            r[:,:2]=r[:,:2]*(xmax-xmin)+xmin
            r[:,2] = r[:,2]*(zmax-zmin)+zmin
            r/=2.
        if preset_v is not None:
            v=preset_r
        else:
            v=np.random.random((N_particles,3))
            v[:,:2]=(v[:,:2]*(xmax-xmin)+xmin)*velocity_scaling
            v[:,2]=(v[:,2]*(zmax-zmin)+zmin)*velocity_scaling

        positions_dataset.attrs['starting_position']=r
        velocities_dataset.attrs['starting_velocity']=v

        print(r)
        print(v)
        if (pusher_function==boris_step):
            dummy, v = pusher_function(r,v,-dt/2., field_calculation_function)
        ended_on_region_exit=False
        for i in xrange(N_iterations):
            #Enter loop
            if not i%Dump_every_N_iterations:
                print("Iteration %d out of %d"%(i,N_iterations))
            counter_to_save_data=i%save_every_n_iterations
            #Push position and velocity
            running_particles=np.isfinite(r[:,0])
            r[running_particles],v[running_particles] = pusher_function(r[running_particles],v[running_particles],dt, field_calculation_function, N_interpolation=N_interpolation)
            # r[running_particles],v[running_particles] = pusher_function(r[running_particles],v[running_particles],dt, field_calculation_function, N_interpolation=N_interpolation)
            #Check for particle leaving region
            x_iter, y_iter, z_iter = r[:,0], r[:,1], r[:,2]
            if not counter_to_save_data:
                data_save_index=i//save_every_n_iterations
                positions_dataset[:,:,data_save_index]=r
                velocities_dataset[:,:,data_save_index]=v
            if not np.any(running_particles):
                print("All of the particles have run out!")
                positions_dataset[:,:,data_save_index+1:]=np.nan
                velocities_dataset[:,:,data_save_index+1:]=np.nan
                break
                # use isnan for removed particles
            ran_out_x=np.logical_or(x_iter<xmin,x_iter>xmax)
            ran_out_y=np.logical_or(y_iter<ymin,y_iter>ymax)
            ran_out_z=np.logical_or(z_iter<zmin,z_iter>zmax)
            ran_out=np.logical_or(ran_out_x, np.logical_or(ran_out_y,ran_out_z))
            # pdb.set_trace()
            r[ran_out,:]=np.nan
            v[ran_out,:]=np.nan
            # if x_iter > xmax or x_iter < xmin or y_iter > ymax or y_iter < ymin or z_iter > zmax or z_iter < zmin:
            #     print("Ran out of the area at i=" + str(i))
            #     ended_on_region_exit = True #prevent program from saving position after leaving the loop
            #     break #quit the for loop

    print("Push finished.")

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

def test_pusher_algorithms():
    #Simulation parameters
    N_iterations=int(1e8)
    Dump_every_N_iterations=int(1e6)
    N_particles=1
    N_interpolation=8
    velocity_scaling=1e6 #for random selection of initial velocity
    dt=0.01/velocity_scaling
    seed=1
    dt=1e-10

    RK4_path = particle_loop(pusher_function=RK4_step, field_calculation_function = test_z_field,
        mode_name = "RK4", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
        save_velocities=True, continue_run=False, dt=dt, preset_r=np.array([0.008, 0., 0.]), preset_v=np.array([0,1000.,0]))
    boris_path = particle_loop(pusher_function=boris_step, field_calculation_function = test_z_field,
        mode_name = "boris", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
        save_velocities=True, continue_run=False, dt=dt, preset_r=np.array([0.008, 0., 0.]), preset_v=np.array([0,1000.,0]))

    print("Finished calculation.")
    RK4_x = RK4_path[:,0]
    RK4_y = RK4_path[:,1]
    boris_x = boris_path[:,0]
    boris_y = boris_path[:,1]

    plt.plot(RK4_x, RK4_y, "ro-", label="RK4")
    plt.plot(boris_x, boris_y, "bo-", label="Boris")
    plt.grid()
    plt.legend()
    plt.show()
    print("Finished display")
    plot_energies("boris", "RK4")
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
    quiver=mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display, opacity=0.01)
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

def plot_xy_positions(*args):
    for mode_name, style in args:
        with h5py.File(folder_name+mode_name+".hdf5", "r") as f:
            particle_positions=f['positions']
            for particle in xrange(f.attrs['N_particles']):
                x=particle_positions[particle,0,:]
                y=particle_positions[particle,1,:]
                plt.plot(x,y, style, label=mode_name)
    plt.grid()
    plt.legend()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()

def plot_energies(mode_name1, mode_name2):
    print("Printing energies")
    particle_i = 0
    while True:
        particle_file_name1=folder_name+mode_name1+str(particle_i)+"velocities.dat"
        particle_file_name2=folder_name+mode_name2+str(particle_i)+"velocities.dat"
        if(os.path.isfile(particle_file_name1) and os.path.isfile(particle_file_name2)):
            energies1 = np.sum(np.loadtxt(particle_file_name1)**2, axis=1)
            energies2 = np.sum(np.loadtxt(particle_file_name2)**2, axis=1)
            plt.plot(energies1, label=("Particle " + str(particle_i) + " " + mode_name1))
            plt.plot(energies2, label=("Particle " + str(particle_i) + " " + mode_name2))
            particle_i+=1
            plt.legend()
            plt.grid()
            plt.xlabel("Iterations")
            plt.ylabel("Energy")
            plt.savefig(folder_name+"Energies" + mode_name1 + mode_name2 + ".png")
            plt.show()
            plt.clf()
        else:
            print("Failed to load particle " + str(particle_i))
            break

def display_particles(mode_name="", colormap="Spectral", all_colorbars=False):
    print("Displaying particles from mode " + mode_name)
    particle_i=0
    while True:
        particle_file_name=folder_name+mode_name+str(particle_i)+"positions.dat"
        if(os.path.isfile(particle_file_name)):
            positions=np.loadtxt(particle_file_name)
            print(positions[-3:,:])
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
    return positions

def load_particle_trajectory(mode_name=""):
    print("Loagin particle from mode " + mode_name)
    particle_file_name=folder_name+mode_name+"0positions.dat"
    if(os.path.isfile(particle_file_name)):
        positions=np.loadtxt(particle_file_name)
        return positions

# if __name__ =="__main__":
#     pass
