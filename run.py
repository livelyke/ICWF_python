import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from params import *
from functions import *

if new_psi0 == True:
    with Timer() as t:
        psi0 = calc_psi0(psi0)
    print('Time to calculate wf was ', t.interval)
else:
    psi0 = load_psi0()

tau = .005
tol = .001
plotDensity = True
gs_relax(psi0,tau,tol,plotDensity)

tf = 10
dt = .001
order = 4
psi_gs = np.load('./gs_wavefunctions/tol-.001-e-.1-n-.02-box-6-equal-mass-soft-nuclei/gs-psi-tol0.001.npy')
#Time_propagate(psi_gs,dt,tf,order,plotDensity)
#plot_rho_R(psi_gs)
