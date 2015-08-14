from biot import *

# display_wires(N_wires=6, r_wires=r_wires)
display_quiver()
display_particles(mode_name="boris_interpolation", colormap="Blues")
display_particles(mode_name="boris_exact", colormap="Reds")
print("Finished display")
mlab.show()
