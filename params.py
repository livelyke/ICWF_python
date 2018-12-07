from numpy import arange, array, pi, array

new_psi0 = False
hard_bc_nuc = False
R_cut = .05

Amplitude = 0*.35
nu = .1288268
optical_cycle = 1./nu
form = 'sin2'
tf_laser = .000001*15*optical_cycle
tf_tot = 250
dt = .01
tf_laser_index = int(tf_laser/dt)
order = 4
plot_step = 50
psi_save_interval =50# plot_step
#atomic units used, i.e. length is all in bohr radii 
#electronic grid spacing
e_spacing = .4
#nuclear grid spacing
n_spacing = .08
#the size of the simulation 'box' around the electrons
#can be somwehat small as one electron will be centered around one nuclei
box_l = 18.

#equillibrium H2 bond length, experimental value
R0 =.8

#r1x_a = array(r1x)
#r2x_a = array(r2x)
#Rx_a = array(Rx)
#define parameters determining initial wavefunction
#put the electrons between the two nuclei
r10 = -R0/2 
r20 = R0/2
re_sig2 = 1

#develop grids along electronic and nuclear coordinates
#offset the electron grids by 1/2 the e_spacing to avoid singularities
#code in many places is dependent on r1x, r2x being the same size!
r1x = [x for x in arange(-box_l/2.,box_l/2.,e_spacing)]
r2x = [x for x in arange(-box_l/2., box_l/2., e_spacing)]
Rx = [x for x in arange(.3,3.5,n_spacing)]

r1x_array = array(r1x)
r2x_array = array(r2x)
Rx_array = array(Rx)

r1x_V_array = array([x for x in arange(-box_l/2., box_l/2.,.1*e_spacing)])
r2x_V_array = array([x for x in arange(-box_l/2., box_l/2.,.1*e_spacing)])
Rx_V_array = array([x for x in arange(.3, 3.5,.1*n_spacing)])

#nuclear dispersion should be small, .02 is largeish
Rsig2 = .01
M1 = 1836.
M2 = 1836.
mu = M1*M2/(M1+M2)

Cr2 =.1 # Coulomb Radius squared
threshold=.1
num_trajs = 700
