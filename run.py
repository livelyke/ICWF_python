import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from params import *
from functions import *
import cwf

if new_psi0 == True:
    with Timer() as t:
        psi0 = calc_psi0(psi0)
    print('Time to calculate wf was ', t.interval)
#else:
    #psi0 = load_psi0()
    #psi=np.load('./gs_wavefunctions/SD-.00001tol-granular-r-box_l-12/psi-gs.npy')
    #psi0 = np.load('./gs_wavefunctions/SD-.000001tol-granular-r-box_l-16/psi-gs.npy')

tau = .0001
tol = .00001
plotDensity = False
psi0 = np.load('./gs_wavefunctions/sym-SD-.00001/psi-gs.npy')
#psi = np.load('./gs_wavefunctions/SD-.0001-R-box-3.5/psi-gs.npy')
#psi = cwf.e2_separable_correlated_wf()
#cwf.propagate_hermitian_cwf(psi,tf_tot,tf_laser,False,False)
#cwf.ICWF_propagate(psi0,tf_tot,tf_laser,'1e')
#cwf.propagate_2e_hermitian_cwf(psi[0],100,1,True, False, False)
#mesh = cwf.initialize_2e_mesh(threshold,psi0,True)
#cwf.conditional_H_2e(psi0[:,:,mesh[-1]].transpose(), psi0[mesh[0],mesh[1],:],mesh[0],mesh[1],mesh[2])
#psi = np.load('./gs_wavefunctions/test-box/psi-gs.npy')
#BOs = np.load('./gs_wavefunctions/test-box/BO_states/BO_states.npy')
#BOs = np.load('./gs_wavefunctions/SD-.00001tol-granular-r-box_l-12/BO_states/BO_states.npy')
with Timer() as t:
    T_dep_RK4_routine(psi0,tf_tot,tf_laser,dt, Amplitude,form)
print(str(int(tf_tot/dt)) + ' steps took ', t.interval)
"""
#psi_t_file = './psi-t-form-ramp-A-0.2-tf-laser-300-tf-tot-600-nu-0.1406/'
#psi_t = np.load(psi_t_file + '/psi-164.50999999999536.npy')
#with Timer() as t:
#    Time_propagate(psi_t,dt,tf_tot,tf_laser,order,plotDensity, BOs, 164.51, psi_t_file)
#print(str(int(tf_tot/dt)) + ' steps took ', t.interval)

#New gs wavefunction generation routines.
with Timer() as t:
    BOe, BOs = sd_BO(tol,1)
print('convergance across ', len(Rx), ' different nucelar positions took ', t.interval)
#cg_BO(tol)
BOe = np.array(BOe)
BOs = np.array(BOs)
np.save('./gs_wavefunctions/sym-SD-.00001/BO_energies', BOe)
np.save('./gs_wavefunctions/sym-SD-.00001/BO_states', BOs)

E_tot, psi_tot = SD(.00001, False, None)
np.save('./gs_wavefunctions/sym-SD-.00001/E_tot', E_tot)
np.save('./gs_wavefunctions/sym-SD-.00001/psi-gs', psi_tot)
"""
