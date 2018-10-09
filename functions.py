import time
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy.sparse import csr_matrix
from params import *

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self,*args):
        self.stop = time.time()
        self.interval = self.stop - self.start

def find_index(L, val):
    return L.index(min(L, key=lambda x:abs(x-val)))

if hard_bc_nuc == True:
    Ri_cut = find_index(Rx,R_cut)
else:
    Ri_cut = 0

def poop_gauss(r1x,r1x0,r1sigma2, r2x,r2x0,r2sigma2,R,R1x0,R1sigma2,R2x0,R2sigma2):
    return np.exp(-(r1x-r1x0)**2/r1sigma2)*np.exp(-(r2x-r2x0)**2/r2sigma2)\
            *np.exp(-(mu*R/M1-R1x0)**2/R1sigma2)*np.exp(-(mu*R/M2+R2x0)**2/R2sigma2)

def poop_calc_psi0(psi0):
    print('Calculating the initial Wavefunction without numba')
    r1i = 0
    r2i = 0
    Ri = 0
    for r1 in r1x:
        r2i=0
        for r2 in r2x:
            Ri=0
            for R in Rx:
                psi0[r1i][r2i][Ri] = poop_gauss(r1,r10,1,r2,r20,1,R,-R0/2,Rsig2,R0/2,Rsig2)
                Ri += 1
            r2i += 1
        r1i += 1
    return psi0


#for initial wavefunction as direct product gaussians
#nuclear wavefunctions are also gaussians, rewritten in reduced coordinates
#By construction with R = R1-R2 for an initial separation of R0, along the R axis, R2 is at 0
#   while R1 is at R0
@jit(nogil=True)
def gauss(r1x,r1x0,r1sigma2, r2x,r2x0,r2sigma2,R,R1x0,R1sigma2,R2sigma2):
    return np.exp(-(r1x-r1x0)**2/r1sigma2)*np.exp(-(r2x-r2x0)**2/r2sigma2)\
            *np.exp(-(mu*R/M1-R1x0)**2/R1sigma2)*np.exp(-(-mu*R/M2)**2/R2sigma2) + 0j


psi0 = np.zeros((len(r1x),len(r2x),len(Rx))) + 0j

#print(len(psi0[0][:][0]))
@jit(nogil=True)
def calc_psi0(psi0):
    print('Calculating the initial Wavefunction with numba')
    #position array index values
    r1i = 0
    r2i = 0
    Ri = 0
    #wavefunction norm calculated on the fly
    psi2 = 0
    for r1 in r1x:
        r2i=0
        for r2 in r2x:
            Ri=0
            for R in Rx[Ri_cut:]: #from the nuclear cut off onwards
                psi0[r1i,r2i,Ri] = gauss(r1,r10,re_sig2,r2,r20,re_sig2,R,R0,Rsig2,Rsig2)
                #add to the discretized integral of psi**2
                psi2 += psi0[r1i,r2i,Ri]**2
                Ri += 1
            r2i += 1
        r1i += 1
    #complete the integral of psi**2 and take the square root
    psi2 = np.sqrt(psi2*e_spacing**2*n_spacing)
    #normalize the wavefunction
    psi0 = psi0/psi2
    np.save('./psi0',psi0)
    return psi0

def load_psi0():
    return np.load('./psi0.npy')

def T_test_psi(psi):
    testpsi = np.zeros(psi.shape)
    for x1i in range(0,len(r1x)):
        for x2i in range(0,len(r2x)):
            for Ri in range(Ri_cut,len(Rx)):
                testpsi[x1i,x2i,Ri] += np.exp(r1x[x1i] + r2x[x2i] + Rx[Ri])
    return testpsi

@jit(nogil=True)
def integrate_over_electrons(psi):
    print('Integrating over electronic coordinates')
    out = np.zeros(len(Rx))
    for Ri in range(Ri_cut,len(Rx)):
        val = 0
        for r1i in range(0,len(r1x)):
            for r2i in range(0,len(r2x)):
                val += abs(psi[r1i,r2i,Ri])**2
        out[Ri] = val*e_spacing**2

    return out

@jit(nogil=True)
def one_electron_density(psi):
    print('Calculating one electron Density')
    out = np.zeros(len(r1x))
    for r1i in range(0,len(r1x)):
        val = 0
        for r2i in range(0,len(r2x)):
            for Ri in range(Ri_cut,len(Rx)):
                val += abs(psi[r1i,r2i,Ri])**2
        out[r1i] = val
    for r2i in range(0,len(r2x)):
        val = 0
        for r1i in range(0,len(r1x)):
            for Ri in range(Ri_cut,len(Rx)):
                val += abs(psi[r1i,r2i,Ri])**2
        out[r2i] += val
    #N=2 times 1/2 average, times integration differential
    return out*e_spacing*n_spacing

@jit(nogil=True)
def normalize(psi):
    val = 0
    for r1i in range(0,len(r1x)):
        for r2i in range(0,len(r2x)):
            for Ri in range(Ri_cut,len(Rx)):
                val += abs(psi[r1i,r2i,Ri])**2
    norm = np.sqrt(val*n_spacing*e_spacing**2)
    print('norm is ', norm)
    return psi/(norm)


def plot_rho_R(psi):
    with Timer() as t:
        rho_R = integrate_over_electrons(psi)
    print('Time to integrate over electrons ', t.interval)
    plt.plot(Rx, rho_R,'.')
    plt.show()

def plot_rho_r(psi):
    with Timer() as t:
        rho_r = one_electron_density(psi)
    print('Time to generate one electron densities', t.interval)
    plt.plot(r1x, rho_r,'.')
    plt.show()


#let a be the soft core square radii
#there is probably some particularly clever way to get all these in minimal loops.
#Can do V on Psi here, or separately as V*psi get timing
@jit(nogil=True)
def calc_V(psi):
    V = np.zeros(psi.shape) +0j
    for r1i in range(0,len(r1x)):
        for r2i in range(0,len(r2x)):
            #electron electron interaction
            V[r1i,r2i,:] += np.sqrt( (r1x[r1i] - r2x[r2i])**2 + Cr2)**(-1)*psi[r1i,r2i,:]

        #grab the r1 electron nuclear interactions while looping over r1
        for Ri in range(Ri_cut,len(Rx)):
            V[r1i,:,Ri] += (-np.sqrt( (r1x[r1i] - mu*Rx[Ri]/M1)**2 + Cr2)**(-1) -\
                             np.sqrt( (r1x[r1i] + mu*Rx[Ri]/M2)**2 + Cr2)**(-1))*psi[r1i,:,Ri]
    
    for Ri in range(Ri_cut,len(Rx)):
        #Nuclear Interaction
        V[:,:,Ri] += Rx[Ri]**(-1)*psi[:,:,Ri]
        #V[:,:,Ri] += np.sqrt(Rx[Ri]**2+Cr2)**(-1)*psi[:,:,Ri]
        #grab the r2 electron nuclear interactions while looping over R
        for r2i in range(0,len(r2x)):
            V[:,r2i,Ri] += (-np.sqrt( (r2x[r2i] - mu*Rx[Ri]/M1)**2 + Cr2)**(-1) -\
                             np.sqrt( (r2x[r2i] + mu*Rx[Ri]/M2)**2 + Cr2)**(-1))*psi[:,r2i,Ri]

    return V

        

#Because of reduction in dimensionality inherent in finite difference, to avoid everything having to go from [1:-1] 
#The derivative value at the end points is set to 0
#can speed these up via sparse matrix formatting
Lap_e1 = np.zeros((len(r1x), len(r1x)))
Lap_e2 = np.zeros((len(r2x),len(r2x)))
Lap_n = np.zeros((len(Rx),len(Rx)))
for i in range(0,len(r1x)):
    for j in range(0,len(r1x)):
        if i==j:
            Lap_e1[i,j] = -2
        if j==i+1 or j==i-1:
            Lap_e1[i,j] = 1
for i in range(0,len(r2x)):
    for j in range(0,len(r2x)):
        if i==j:
            Lap_e2[i,j] = -2
        if j==i+1 or j==i-1:
            Lap_e2[i,j] = 1
for i in range(0,len(Rx)):
    for j in range(0,len(Rx)):
        if i==j:
            Lap_n[i,j] = -2
        if j==i+1 or j==i-1:
            Lap_n[i,j] = 1

#Complete calculation of the lapalacian and divide by twice the mass, multiplying my -1
T_e1 = csr_matrix((-1/(2*e_spacing**2))*Lap_e1 + 0j)
T_e2 = csr_matrix((-1/(2*e_spacing**2))*Lap_e2 + 0j)
T_n = csr_matrix((-1/(2*mu*n_spacing**2))*Lap_n+ 0j)
@jit(nogil=True)
def calc_T(psi):
    psi_out = np.zeros(psi.shape) + 0j
    
    for x1i in range(0,len(r1x)):
        for x2i in range(0,len(r2x)):
            psi_out[x1i,x2i,:] += T_n.dot(psi[x1i,x2i,:])

    for Ri in range(Ri_cut,len(Rx)):
        for x2i in range(0,len(r2x)):
            psi_out[:,x2i,Ri] += T_e1.dot(psi[:,x2i,Ri])
        for x1i in range(0,len(r1x)):
            psi_out[x1i,:,Ri] += T_e2.dot(psi[x1i,:,Ri])
    return psi_out


@jit(nogil=True)
def H(psi):
    return calc_V(psi)+calc_T(psi)

@jit(nogil=True)
def calc_E(psi):
    Hpsi = H(psi)
    val = 0j
    for x1i in range(0,len(r1x)):
        for x2i in range(0,len(r2x)):
            for Ri in range(Ri_cut,len(Rx)):
                val += np.conj(psi[x1i,x2i,Ri])*Hpsi[x1i,x2i,Ri]
    return val*n_spacing*e_spacing**2


@jit(nogil=True)
def imag_prop(psi_in, tau, tol, plotDensity):
    psi = np.zeros(psi0.shape) + 0j

    psi = psi_in - tau*H(psi_in) + tau**2/2 * H(H(psi0)) #- tau**3/6 * H(H(H(psi0))) 

    psi = normalize(psi)

    diff = abs(psi - psi_in)
    mdiff = diff.max()
    if(mdiff > tol):
        print('max difference = ', mdiff)
        if plotDensity==True:
            plt.pause(0.0001)
            fig_rho_e.set_ydata(one_electron_density(psi))
            fig_rho_n.set_ydata(integrate_over_electrons(psi))

            ax_e.relim()
            ax_e.autoscale()
            ax_n.relim()
            ax_n.autoscale()

            iter_gs.append(iter_gs[-1]+1)
            E_gs.append(calc_E(psi))

            fig_E.set_xdata(iter_gs)
            fig_E.set_ydata(E_gs)
            ax_E.relim()
            ax_E.autoscale()

            fig1.canvas.draw()
            fig1.canvas.flush_events
        else:
            plt.pause(0.0001)
            iter_gs.append(iter_gs[-1]+1)
            E_gs.append(calc_E(psi))

            fig_E.set_xdata(iter_gs)
            fig_E.set_ydata(E_gs)
            ax_E.relim()
            ax_E.autoscale()

            fig1.canvas.draw()
            fig1.canvas.flush_events

        imag_prop(psi,tau,tol,plotDensity)
    else:
        np.save('./gs-psi-tol' + str(tol), psi)

#plotting wrapper for recursive function imag_prop
def gs_relax(psi0, tau, tol, plotDensity): 
    global E_gs, iter_gs, fig_E, fig1, ax_E
    plt.ion()
    if plotDensity==True:
        global ax_e, ax_n, fig_rho_e, fig_rho_n
        fig1, (ax_e,ax_n, ax_E) = plt.subplots(3,1)
        plt.pause(0.0001)
        #fig_rho_e, = ax_e.plot(r1x, psi0[:,find_index(r2x,R0/4),find_index(Rx,R0)],'.')
        fig_rho_e, =ax_e.plot(r1x, one_electron_density(psi0),'.')
        #fig_rho_n, = ax_n.plot(Rx, psi0[find_index(r2x,-R0/4),find_index(r2x,R0/4),:],'.')
        fig_rho_n, = ax_n.plot(Rx, integrate_over_electrons(psi0),'.')
    else:
        fig1, ax_E = plt.subplots(1,1)

    E_gs = [calc_E(psi0)]
    iter_gs = [0]
    fig_E, = ax_E.plot(iter_gs,E_gs, '.')
    plt.pause(0.0001)
    fig1.canvas.draw()
    fig1.canvas.flush_events

    imag_prop(psi0, tau, tol, plotDensity)

@jit(nogil=True)
def time_propagator_kernel(psi, dt, tf, order, plotDensity):
    iter_time.append(iter_time[-1]+dt)

    if(iter_time[-1] != tf):
        if order==4:
            psi = psi - 1j*H(psi)*dt - dt**2/2 * H(H(psi)) + 1j*dt**3/6 * H(H(H(psi))) \
                    + dt**4/24 * H(H(H(H(psi))))
        if order==3:
            psi = psi - 1j*H(psi)*dt - dt**2/2 * H(H(psi)) + 1j*dt**3/6 * H(H(H(psi))) 
 
        if order==2:
            psi = psi - 1j*H(psi)*dt - dt**2/2 * H(H(psi))

        if plotDensity==True:
            plt.pause(0.0001)
            fig_rho_e.set_ydata(one_electron_density(psi))
            fig_rho_n.set_ydata(integrate_over_electrons(psi))
 
            ax_e.relim()
            ax_e.autoscale()
            ax_n.relim()
            ax_n.autoscale()
 
            E_t.append(calc_E(psi))
 
            fig_E.set_xdata(iter_time)
            fig_E.set_ydata(E_t)
            ax_E.relim()
            ax_E.autoscale()
 
            fig1.canvas.draw()
            fig1.canvas.flush_events
        else:
            plt.pause(0.0001)
            E_t.append(calc_E(psi))
 
            fig_E.set_xdata(iter_time)
            fig_E.set_ydata(E_t)
            ax_E.relim()
            ax_E.autoscale()
 
            fig1.canvas.draw()
            fig1.canvas.flush_events
 
        time_propagator_kernel(psi,dt,tf,order, plotDensity)
    else:
        np.save('./final-wf-at-' + str(tf), psi)

def Time_propagate(psi,dt,tf, order,plotDensity): 
    global E_t, iter_time, fig_E, fig1, ax_E
    plt.ion()
    if plotDensity==True:
        global ax_e, ax_n, fig_rho_e, fig_rho_n
        fig1, (ax_e,ax_n, ax_E) = plt.subplots(3,1)
        plt.pause(0.0001)
        #fig_rho_e, = ax_e.plot(r1x, psi0[:,find_index(r2x,R0/4),find_index(Rx,R0)],'.')
        fig_rho_e, =ax_e.plot(r1x, one_electron_density(psi),'.')
        #fig_rho_n, = ax_n.plot(Rx, psi0[find_index(r2x,-R0/4),find_index(r2x,R0/4),:],'.')
        fig_rho_n, = ax_n.plot(Rx, integrate_over_electrons(psi),'.')
    else:
        fig1, ax_E = plt.subplots(1,1)

    E_t = [calc_E(psi)]
    iter_time = [0]
    fig_E, = ax_E.plot(iter_time,E_t, '.')
    plt.pause(0.0001)
    fig1.canvas.draw()
    fig1.canvas.flush_events

    time_propagator_kernel(psi,dt,tf,order,plotDensity)
