from biot import *

colormaps={0.1:"Blues", 1.:"Greens", 10.:"Reds"}
# for n in colormaps.keys():
#     display_particles(mode_name="RK4_exact" + str(n), colormap=colormaps[n])

n=1.
smalldtpath=display_particles(mode_name="RK4_exact" + str(n), colormap=colormaps[n])
smalldttime=np.arange(0,1e-7-1e-13/n,1e-13/n)
n=0.1
longdtpath=display_particles(mode_name="RK4_exact" + str(n), colormap=colormaps[n])
longdttime=np.arange(0,1e-7-1e-13/n,1e-13/n)

n=10.
thirddtpath=display_particles(mode_name="RK4_exact" + str(n), colormap=colormaps[n])
thirddttime=np.arange(0,1e-7-1e-13/n,1e-13/n)

smalldtx=smalldtpath[:,0]
smalldty=smalldtpath[:,1]
smalldtz=smalldtpath[:,2]
longdtx=longdtpath[:,0]
longdty=longdtpath[:,1]
longdtz=longdtpath[:,2]
thirddtx=thirddtpath[:,0]
thirddty=thirddtpath[:,1]
thirddtz=thirddtpath[:,2]


plt.plot(smalldttime, smalldtx, 'g-')
plt.plot(longdttime, longdtx, 'b-')
print(len(thirddttime), len(thirddtx))
plt.plot(thirddttime, thirddtx, 'r-')
plt.show()

plt.plot(smalldttime, smalldty, 'g-')
plt.plot(longdttime, longdty, 'b-')
plt.plot(thirddttime, thirddtx, 'r-')
plt.show()

plt.plot(smalldttime, smalldtz, 'g-')
plt.plot(longdttime, longdtz, 'b-')
plt.plot(thirddttime, thirddtx, 'r-')
plt.show()
#n=10 path is really short

mlab.show()
