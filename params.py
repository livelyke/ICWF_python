from numpy import arange

new_psi0 = True
hard_bc_nuc = False
R_cut = .05

#atomic units used, i.e. length is all in bohr radii 
#electronic grid spacing
e_spacing = .1
#nuclear grid spacing
n_spacing = .01
#the size of the simulation 'box'
box_l = 6

#equillibrium H2 bond length
R0 =.39146

#develop grids along electronic and nuclear coordinates
#offset the electron grids by 1/2 the e_spacing to avoid singularities
r1x = [x for x in arange(-box_l/2,box_l/2,e_spacing)]
r2x = [x for x in arange(-box_l/2 - e_spacing/2, box_l/2- e_spacing/2, e_spacing)]
Rx = [x for x in arange(.1*R0,2*R0,n_spacing)]
#define parameters determining initial wavefunction
#put the electrons between the two nuclei
r10 = -R0/2 - e_spacing
r20 = R0/2 + e_spacing
re_sig2 = .5

#nuclear dispersion should be small, .02 is largeish
Rsig2 = .01
M1 = 1836
M2 = 1836
mu = M1*M2/(M1+M2)

Cr2 = .1 # Coulomb Radius squared

