from biot import *

read_parameters("boris")
for i in xrange(10):
    display_particles(modes=(("boris", "Blues"),), particles=(i,))
    mlab.show()
