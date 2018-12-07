import time
import numpy as np
from matplotlib import pyplot as plt
from numba import jit, prange
from scipy.sparse import csr_matrix
from importlib import reload
import params
reload(params)
from params import *
from subprocess import call
from os import listdir

#{{{ Timer, initialize, integrate over electrons, plot
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self,*args):
        self.stop = time.time()
        self.interval = self.stop - self.start

@jit(nogil=True)
def find_index(L, val):
    #if (type(val)==int or type(val) ==float or type(val) ==np.float64):
    #    return L.index(min(L, key=lambda x:abs(x-val)))
    #elif (type(L)==list and ((type(val)==np.ndarray) or (type(val)==list))):
    #    a = [0 for x in range(len(val))]
    #    for i in range(len(val)):
    #        a[i] = int((L.index(min(L, key=lambda x:abs(x-val[i])))))
    #    return a
    #elif (type(L)==np.ndarray and ((type(val)==np.ndarray) or (type(val)==list))):
    a = np.zeros(val.shape,dtype=int)
    for i in range(val.shape[0]):
        a[i] = int(round(np.abs(L - val[i]).argmin()))
    return a

if hard_bc_nuc == True:
    Ri_cut = find_index(Rx,R_cut)
else:
    Ri_cut = 0

def gap_photon(gap):
    gap_eV = gap*27.2114
    nu_s = gap_eV/(4.136*10**(-15))
    nu_au = nu_s*(2.419*10**(-17))
    return nu_au

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
                psi0[r1i,r2i,Ri] = gauss(r1,Rx[Ri]/2,re_sig2,r2,-Rx[Ri]/2,re_sig2,R,R0,Rsig2,Rsig2)\
                                   +gauss(r2,Rx[Ri]/2,re_sig2,r1,-Rx[Ri]/2,re_sig2,R,R0,Rsig2,Rsig2)
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
    return e_spacing**2*sum(sum(np.abs(psi)**2))

@jit(nogil = True)
def one_electron_density(psi):
    print('Calculating one electron Density')
    psi2 = np.abs(psi)**2
    out = np.zeros(len(r1x))
    for r1i in range(0,len(r1x)):
        out[r1i] = sum(np.sum(psi2[r1i,:,:],0))
    for r2i in range(0,len(r2x)):
        out[r2i] += sum(np.sum(psi2[:,r2i,:],0))
    return out*e_spacing*n_spacing


@jit(nogil=True)
def normalize(psi):
    psi2 = np.abs(psi)**2 
    norm = np.sqrt(sum(sum(sum(psi2)))*n_spacing*e_spacing**2)
    print('norm is ', norm)
    return psi/(norm)

def plot_rho_R(psi):
    with Timer() as t:
        rho_R = integrate_over_electrons(psi)
    print('Time to integrate over electrons ', t.interval)
    plt.plot(Rx, rho_R,'.')
    plt.show()

#relies on r1x and r2x being the same size
def plot_rho_r(psi):
    with Timer() as t:
        rho_r = one_electron_density(psi)
    print('Time to generate one electron densities', t.interval)
    plt.plot(r1x, rho_r,'.')
    plt.show()

#}}}

#{{{ calc_V, calc_T, H, calc_E
@jit(nogil=True)
def construct_V(V):
    for r1i in range(0,len(r1x)):
        for r2i in range(0,len(r2x)):
           #electron electron interaction
            V[r1i,r2i,:] += np.sqrt( (r1x[r1i] - r2x[r2i])**2 + Cr2)**(-1)

       #grab the r1 electron nuclear interactions while looping over r1
        for Ri in range(Ri_cut,len(Rx)):
            V[r1i,:,Ri] += (-np.sqrt( (r1x[r1i] - mu*Rx[Ri]/M1)**2 + Cr2)**(-1) -\
                             np.sqrt( (r1x[r1i] + mu*Rx[Ri]/M2)**2 + Cr2)**(-1))
    
    for Ri in range(Ri_cut,len(Rx)):
        #Nuclear Interaction
        V[:,:,Ri] += Rx[Ri]**(-1)
        #grab the r2 electron nuclear interactions while looping over R
        for r2i in range(0,len(r2x)):
            V[:,r2i,Ri] += (-np.sqrt( (r2x[r2i] - mu*Rx[Ri]/M1)**2 + Cr2)**(-1) -\
                             np.sqrt( (r2x[r2i] + mu*Rx[Ri]/M2)**2 + Cr2)**(-1))

    return V

V_kernel = construct_V(np.zeros( (len(r1x), len(r2x), len(Rx)) )) + 0j

@jit(nopython=True,parallel=True,nogil=True)
def calc_V(psi,V):
    return V*psi


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

mu_e = (M1+M2)/(M1+M2+1)
#Complete calculation of the lapalacian and divide by twice the mass, multiplying my -1
T_e1 = csr_matrix((-1/(2*mu_e*e_spacing**2))*Lap_e1 + 0j)
T_e2 = csr_matrix((-1/(2*mu_e*e_spacing**2))*Lap_e2 + 0j)
T_n = csr_matrix((-1/(2*mu*n_spacing**2))*Lap_n+ 0j)

T_buff = np.zeros( (len(r1x), len(r2x), len(Rx)) ) + 0j
@jit(nogil=True)
def calc_T(psi, buff):
    buff[:,:,:] = 0j 
    for x1i in range(0,len(r1x)):
        for x2i in range(0,len(r2x)):
            buff[x1i,x2i,:] += T_n.dot(psi[x1i,x2i,:])

    for Ri in range(Ri_cut,len(Rx)):
        for x2i in range(0,len(r2x)):
            buff[:,x2i,Ri] += T_e1.dot(psi[:,x2i,Ri])
            buff[x2i,:,Ri] += T_e2.dot(psi[x2i,:,Ri])
    return buff

#This is faster, and returns identical results to the above on its own, but differen when inside H
Te_tot = np.kron(T_e1,T_e2)
@jit(nogil=True)
def calc_T2(psi, buff):
    buff[:,:,:] = 0j
    for Ri in range(Ri_cut,len(Rx)):
        buff[:,:,Ri] += Te_tot.dot(psi[:,:,Ri])
    for r1i in range(len(r1x)):
        for r2i in range(len(r2x)):
            buff[r1i,r2i,:] += T_n.dot(psi[r1i,r2i,:])
    return buff


@jit(nogil=True)
def H(psi):
    return calc_V(psi, V_kernel)+calc_T(psi, T_buff)

@jit(parallel=True, nogil=True)
def D(psi):
    Dpsi = np.zeros(psi.shape)+0j
    for r2i in prange(psi.shape[1]):
        for Ri in prange(psi.shape[2]):
            Dpsi[:,r2i,Ri] -= psi[:,r2i,Ri]*r1x[:]
    for r1i in prange(psi.shape[0]):
        for Ri in prange(psi.shape[2]):
            Dpsi[r1i,:,Ri] -= psi[r1i,:,Ri]*r2x[:]
    if(M1 != M2):
        lamb = (M2-M1)/(M2+M1)
        for r1i in prange(psi.shape[0]):
            for r2i in prange(psi.shape[1]):
                Dpsi[r1i,r2i,:] += lamb*Rx[:]*psi[r1i,r2i,:]
    return Dpsi
@jit(nogil=True)
def shape_E(tf_laser,dt, A, form):
    time_steps = int(tf_laser/dt) + 1
    Eform = np.zeros(time_steps)
    #frequency in s is 6.177*10**15 Hz, 
    #calculated from difference between BO gs and excited around peak of nuclear wavepacket
    #with conversion factor of 2.419*10**(-17) s / au -> .1494 Hz,au
    if (form=='sin2'):
        for i in range(time_steps):
            Eform[i] = np.sin(2*np.pi*nu*i*dt)*np.sin(np.pi*i*dt/(tf_laser))**2
    elif (form=='ramp'):
        for i in range(time_steps):
            if(i*dt<10*optical_cycle):
                Eform[i] = np.sin(2*np.pi*nu*i*dt)*(i*dt/(10.*optical_cycle))
            elif(i*dt>10*optical_cycle and i*dt < tf_laser-10*optical_cycle):
                Eform[i] = np.sin(2*np.pi*nu*i*dt)
            else:
                Eform[i] = np.sin(2*np.pi*nu*i*dt)*(tf_laser/(10*optical_cycle)-(i*dt/(10*optical_cycle)))
    return A*Eform

@jit(nogil=True)
def Ht(psi,Et):
    return calc_V(psi, V_kernel)+calc_T(psi, T_buff) + Et*D(psi)

@jit(nogil=True)
def calc_E(psi):
    Hpsi = H(psi)
    return sum(sum(sum(np.conj(psi)*Hpsi)))*n_spacing*e_spacing**2

@jit(nogil=True)
def calc_Et(psi,Et):
    Hpsi = Ht(psi,Et)
    return sum(sum(sum(np.conj(psi)*Hpsi)))*n_spacing*e_spacing**2
#}}}

@jit(nogil=True)
def inner_product_full(bra,ket):
    return sum(sum(sum(np.conj(bra)*ket)))*n_spacing*e_spacing**2

#This requires the psi and kets to be normalized
@jit(nogil=True)
def filter_projection(psi, kets): 
    #where psi is the wavefunction to be filtered and ket is the ket to filter out
    projection = np.zeros(psi.shape) +0j
    for ket_i in kets:
        projection += inner_product_full(ket_i,psi)*ket_i
        #projection += inner_product_full(psi,ket_i)*psi
    return psi - projection


#{{{ imaginary propagation
@jit(nogil=True)
def imag_prop_kernel(psi_in,tau):
    return psi_in - tau*H(psi_in)
 
#plotting wrapper for recursive function imag_prop
def gs_relax(psi0, tau, tol, plotDensity): 
    psi = np.zeros(psi0.shape) + 0j
    plt.ion()
    if plotDensity==True:
        fig1, (ax_e,ax_n, ax_E) = plt.subplots(3,1)
        plt.pause(0.0001)
        fig_rho_e, =ax_e.plot(r1x, one_electron_density(psi0),'.')
        fig_rho_n, = ax_n.plot(Rx, integrate_over_electrons(psi0),'.')
    else:
        fig1, ax_E = plt.subplots(1,1)

    E_gs = [calc_E(psi0)]
    iter_gs_plot = [0]
    iter_gs = 0
    fig_E, = ax_E.plot(iter_gs,E_gs, '.')
    plt.pause(0.0001)
    fig1.canvas.draw()
    fig1.canvas.flush_events

    psi = imag_prop_kernel(psi0,tau)
    psi = normalize(psi)

    iter_gs += 1
    E_gs.append(calc_E(psi))
    while(abs((-E_gs[-2] + E_gs[-1])) > tol):
        psi = imag_prop_kernel(psi,tau)
        psi = normalize(psi)
        iter_gs += 1
        print('E diff is ', abs((-E_gs[-2] + E_gs[-1])))
        if(iter_gs%10 == 0): 
            #needed two E_gs vals to initiate while loop, but due to jumping of energy between calls
            #have to reset the second E_gs val to have plotting resolution at low E
            if(iter_gs==10):
                E_gs[-1] = calc_E(psi)
            else:
                E_gs.append(calc_E(psi))
            iter_gs_plot.append(iter_gs)
            plt.pause(0.0001)
            if plotDensity==True:
                fig_rho_e.set_ydata(one_electron_density(psi))
                fig_rho_n.set_ydata(integrate_over_electrons(psi))
                
                ax_e.relim()
                ax_e.autoscale()
                ax_n.relim()
                ax_n.autoscale()   
                 
            fig_E.set_xdata(iter_gs_plot)
            fig_E.set_ydata(E_gs)
            ax_E.relim()
            ax_E.autoscale()
            
            fig1.canvas.draw()
            fig1.canvas.flush_events
         
    np.save('./gs-psi-tol' + str(tol), psi)


#Excited state relaxation based around projection filtering
#first order Taylor expansion of exp(-H1 * tau)
#where H1 = (1-P0)H(1-P0)
@jit(nogil=True)
def es_relax_kernel(psi_in, psi_gs, tau):
    return psi_in - tau*filter_projection(H(filter_projection(psi_in, psi_gs)),psi_gs)

def es_relax(psi0,psi_gs,tau,tol,plotDensity):
    plt.ion()
    if plotDensity==True:
        fig1, (ax_e,ax_n, ax_E) = plt.subplots(3,1)
        plt.pause(0.0001)
        fig_rho_e, =ax_e.plot(r1x, one_electron_density(psi0),'.')
        fig_rho_n, = ax_n.plot(Rx, integrate_over_electrons(psi0),'.')
    else:
        fig1, ax_E = plt.subplots(1,1)

    E_ex1 = [calc_E(psi0)]
    iter_ex1_plot = [0]
    iter_ex1 = 0
    fig_E, = ax_E.plot(iter_ex1,E_ex1, '.')
    plt.pause(0.0001)
    fig1.canvas.draw()
    fig1.canvas.flush_events

    psi = np.zeros(psi0.shape) + 0j
    psi = es_relax_kernel(psi0,psi_gs,tau)
    psi = normalize(psi)

    iter_ex1 += 1
    E_ex1.append(calc_E(psi))
    while(abs((-E_ex1[-2] + E_ex1[-1])) > tol):
        psi = es_relax_kernel(psi,psi_gs,tau)
        psi = normalize(psi)
        iter_ex1 += 1
        print('E_diff is ', abs(-E_ex1[-2] + E_ex1[-1]))
        if(iter_ex1%10 == 0): 
            #needed two E_ex1 vals to initiate while loop, but due to jumping of energy between calls
            #have to reset the second E_ex1 val to have plotting resolution at low E
            if(iter_ex1==10):
                E_ex1[-1] = calc_E(psi)
            else:
                E_ex1.append(calc_E(psi))
            iter_ex1_plot.append(iter_ex1)
            if plotDensity==True:
                plt.pause(0.0001)
                fig_rho_e.set_ydata(one_electron_density(psi))
                fig_rho_n.set_ydata(integrate_over_electrons(psi))
                
                ax_e.relim()
                ax_e.autoscale()
                ax_n.relim()
                ax_n.autoscale()   
                
                
                fig_E.set_xdata(iter_ex1_plot)
                fig_E.set_ydata(E_ex1)
                ax_E.relim()
                ax_E.autoscale()
                
                fig1.canvas.draw()
                fig1.canvas.flush_events
            else:
                E_ex1.append(calc_E(psi))
                iter_ex1_plot.append(iter_ex1)
                plt.pause(0.0001)
                fig_E.set_xdata(iter_ex1_plot)
                fig_E.set_ydata(E_ex1)
                ax_E.relim()
                ax_E.autoscale()
                
                fig1.canvas.draw()
                fig1.canvas.flush_events
         
    np.save('./ex1-psi-tol' + str(tol), psi)

#}}}
#{{{ steepest descent full

@jit(nogil=True)
def check_convergence(psi):
    Hpsi = H(psi)
    E = sum(sum(sum(np.conj(psi)*Hpsi)))*n_spacing*e_spacing**2
    #difference vector
    d_vec = Hpsi - E*psi
    #distance from correct solution
    d = np.sqrt(sum(sum(sum(np.conj(d_vec)*d_vec)))*n_spacing*e_spacing**2)
    return d, E

#now the real challage here is that psi is of dimension (len(r1x),len(r2x),len(Rx))
#while BOgs is of dimension (len(Rx),len(r1x),len(r2x))
#calculate <\phi | \Psi> for each position of R, then construct the subtracted wavefunction as
# <\phi | \Psi> |\phi>
@jit(nogil=True)
def filter_BO_gs(psi_in, BOgs):
    subtracted_psi = np.zeros(psi_in.shape) + 0j
    phi_on_psi = 0j
    for Ri in range(len(Rx)):
        phi_on_psi = inner_product_el(BOgs[Ri],psi_in[:,:,Ri])
        for r1i in range(len(r1x)):
            for r2i in range(len(r2x)):
                subtracted_psi[r1i,r2i,Ri] = phi_on_psi*BOgs[Ri,r1i,r2i]

    return psi_in - subtracted_psi

@jit(nogil=True)
def calc_excited_psi_full():
    psi0 = np.zeros((len(r1x),len(r2x),len(Rx)))
    

@jit(nogil=True)
def SD(tol, filter_BOgs, BOgs):
    eigenvalues = []
    if (filter_BOgs==False):
        psi_m = calc_psi0(psi0)
        #psi_m = normalize(np.ones(psi0.shape))
    if (filter_BOgs == True):
        #Make a god awful fist guess
        psi_m = normalize(np.ones(psi0.shape))
    conj = np.zeros(psi_m.shape) +0j 

    #calculate the 0th iteration eigenvalue guess
    lam = calc_E(psi_m)
    r = lam*psi_m - H(psi_m)
    if(filter_BOgs == True):
        r = filter_BO_gs(r,BOgs)
    
    alpha = inner_product_full(r,r)/calc_E(r)

    psi_m = psi_m + alpha*r

    #check for convergence
    dist, lam = check_convergence(psi_m)

    #enter algorithm
    while (dist > tol):
        r = lam*psi_m - H(psi_m)
        if(filter_BOgs == True):
            r = filter_BO_gs(r,BOgs)
        alpha = inner_product_full(r,r)/calc_E(r)

        psi_m = psi_m + alpha*r

        #chec convergence, update lambda 
        dist, lam = check_convergence(psi_m)
        print('dist=',dist)
    eigenvalues.append(lam)
    return eigenvalues, psi_m
    

#}}}
#{{{ Time Propagation
@jit(nogil=True)
def RK4_routine(psi,tf,dt):
    t=0.000000
    i=0
    while(t<tf):
        K1 = -1j*H(psi)
        K2 = -1j*H(psi + dt*K1/2)
        K3 = -1j*H(psi+dt*K2/2)
        K4 = -1j*H(psi+dt*K3)
        psi += (dt/6)*(K1 + 2*K2 + 2*K3 + K4)
        t+=dt
        i+=1
        if(i%psi_save_interval==0):
            np.save('./dump/1e-icwf-comp/psi-'+"{:.6}".format(t),psi)
            print("{:.6}".format(t))
            print('Norm = ',np.sqrt(np.sum(np.sum(np.sum(np.abs(psi)**2)))*e_spacing**2*n_spacing))

@jit(nogil=True)
def particle_velocities(psi, e1_mesh, e2_mesh,R_mesh):
    #number of trajectoeries changes
    nt = e1_mesh.shape[0]
    e1_vel = np.zeros(nt)
    e2_vel = np.zeros(nt)
    R_vel = np.zeros(nt)
    for i in range(nt):
        e1_vel[i] = (np.gradient(psi[:,e2_mesh[i],R_mesh[i]],e_spacing)[e1_mesh[i]]/psi[e1_mesh[i],e2_mesh[i],R_mesh[i]]).imag
        e2_vel[i] = (np.gradient(psi[e1_mesh[i],:,R_mesh[i]],e_spacing)[e2_mesh[i]]/psi[e1_mesh[i],e2_mesh[i],R_mesh[i]]).imag
        R_vel[i] = (np.gradient(psi[e1_mesh[i],e2_mesh[i],:],n_spacing)[R_mesh[i]]/psi[e1_mesh[i],e2_mesh[i],R_mesh[i]]).imag
    #print('e1_vel[0] = ', e1_vel[0])
    #plt.plot((np.gradient(psi[:,e2_mesh[0],R_mesh[0]],e_spacing)/psi[e1_mesh[0],e2_mesh[0],R_mesh[0]]).imag)
    #plt.show()
    return e1_vel, e2_vel, R_vel/mu

@jit(nogil=True)
def T_dep_RK4_routine(psi,tf,tf_laser,dt, A, form):
    t=0.000000
    t_index = 0
    E_form = shape_E(tf_laser,.5*dt, A, form)
    dir_name = './sym-psi-t-form-'+form+'-A-'+str(Amplitude)+'-tf-laser-'\
                +str(tf_laser)+'-tf-tot-'+str(tf_tot) + '-nu-'+str(nu)
    err = call(['mkdir', dir_name])
    call(['cp','params.py',dir_name+'/README'])

    e1_mesh = np.load('./dump/1e/init_e1_mesh.npy')
    e2_mesh = np.load('./dump/1e/init_e2_mesh.npy')
    R_mesh = np.load('./dump/1e/init_R_mesh.npy')
    pos_e1 = r1x_array[e1_mesh]
    pos_e2 = r2x_array[e2_mesh]
    pos_R = Rx_array[R_mesh]

    psi_buff = np.zeros(psi.shape, dtype=np.complex128)
    e1_mesh_buff = np.zeros(num_trajs)
    e2_mesh_buff = np.zeros(num_trajs)
    R_mesh_buff = np.zeros(num_trajs)
    #if(err != 0):
    #    return 'Check psi Directory'
    while(t<tf_laser-dt):
        E_t = E_form[2*t_index]
        E_t_half = E_form[2*t_index+1]
        E_t_adv = E_form[2*t_index+2]
        
        K1 = -1j*Ht(psi, E_t)
        e1_K1, e2_K1, R_K1 = particle_velocities(psi,e1_mesh,e2_mesh,R_mesh)
        psi_buff = psi+dt*K1/2
        e1_mesh_buff = find_index(r1x_array,pos_e1+dt*e1_K1/2)
        e2_mesh_buff = find_index(r2x_array,pos_e2+dt*e2_K1/2)
        R_mesh_buff = find_index(Rx_array,pos_R+dt*R_K1/2)
        K2 = -1j*Ht(psi_buff, E_t_half)
        e1_K2, e2_K2, R_K2 = particle_velocities(psi_buff,e1_mesh_buff,e2_mesh_buff,R_mesh_buff)
        psi_buff = psi+dt*K2/2
        e1_mesh_buff = find_index(r1x_array,pos_e1+dt*e1_K2/2)
        e2_mesh_buff = find_index(r2x_array,pos_e2+dt*e2_K2/2)
        R_mesh_buff = find_index(Rx_array,pos_R+dt*R_K2/2)

        K3 = -1j*Ht(psi_buff, E_t_half)
        e1_K3, e2_K3, R_K3 = particle_velocities(psi_buff,e1_mesh_buff,e2_mesh_buff,R_mesh_buff)
        psi_buff = psi+dt*K3
        e1_mesh_buff = find_index(r1x_array,pos_e1+dt*e1_K3)
        e2_mesh_buff = find_index(r2x_array,pos_e2+dt*e2_K3)
        R_mesh_buff = find_index(Rx_array,pos_R+dt*R_K3)

        K4 = -1j*Ht(psi_buff, E_t_adv)
        e1_K4, e2_K4, R_K4 = particle_velocities(psi_buff,e1_mesh_buff,e2_mesh_buff,R_mesh_buff)

        psi += (dt/6)*(K1 + 2*K2 + 2*K3 + K4)
        pos_e1 += (dt/6)*(e1_K1 + 2*e1_K2 + 2*e1_K3 + e1_K4)
        pos_e2 += (dt/6)*(e2_K1 + 2*e2_K2 + 2*e2_K3 + e2_K4)
        pos_R += (dt/6)*(R_K1 + 2*R_K2 + 2*R_K3 + R_K4)

        print('time=',t)
        print('pos_e1[0] = ', pos_e1[0])
        print('vel_e1[0] = ', (e1_K1[0] + 2*e1_K2[0] + 2*e1_K3[0] + e1_K4[0]))
        print('pos_e2[0] = ', pos_e2[0])
        print('vel_e2[0] = ', (e2_K1[0] + 2*e2_K2[0] + 2*e2_K3[0] + e2_K4[0]))
        print('pos_R[0] = ', pos_R[0])
        print('vel_R[0] = ', (R_K1[0] + 2*R_K2[0] + 2*R_K3[0] + R_K4[0]))
        print('\n')

        t+=dt
        t_index +=1
        if(t_index%psi_save_interval==0):
            np.save(dir_name+'/psi-'+"{:.6}".format(t),psi)
            print('Norm = ',np.sqrt(np.sum(np.sum(np.sum(np.abs(psi)**2)))*e_spacing**2*n_spacing))
            print("{:.1f}".format(100*t/tf_tot)+'% done')
    if(t>=tf_laser-1.5*dt):
        while(t<tf):
            K1 = -1j*H(psi)
            e1_K1, e2_K1, R_K1 = particle_velocities(psi,e1_mesh,e2_mesh,R_mesh)
            psi_buff = psi+dt*K1/2
            e1_mesh_buff = find_index(r1x_array,pos_e1+dt*e1_K1/2)
            e2_mesh_buff = find_index(r2x_array,pos_e2+dt*e2_K1/2)
            R_mesh_buff = find_index(Rx_array,pos_R+dt*R_K1/2)

            K2 = -1j*H(psi_buff)
            e1_K2, e2_K2, R_K2 = particle_velocities(psi_buff,e1_mesh_buff,e2_mesh_buff,R_mesh_buff)
            psi_buff = psi+dt*K2/2
            e1_mesh_buff = find_index(r1x_array,pos_e1+dt*e1_K2/2)
            e2_mesh_buff = find_index(r2x_array,pos_e2+dt*e2_K2/2)
            R_mesh_buff = find_index(Rx_array,pos_R+dt*R_K2/2)

            K3 = -1j*H(psi_buff)
            e1_K3, e2_K3, R_K3 = particle_velocities(psi_buff,e1_mesh_buff,e2_mesh_buff,R_mesh_buff)
            psi_buff = psi+dt*K3
            e1_mesh_buff = find_index(r1x_array,pos_e1+dt*e1_K3)
            e2_mesh_buff = find_index(r2x_array,pos_e2+dt*e2_K3)
            R_mesh_buff = find_index(Rx_array,pos_R+dt*R_K3)

            K4 = -1j*H(psi_buff)
            e1_K4, e2_K4, R_K4 = particle_velocities(psi_buff,e1_mesh_buff,e2_mesh_buff,R_mesh_buff)

            psi += (dt/6)*(K1 + 2*K2 + 2*K3 + K4)
            pos_e1 += (dt/6)*(e1_K1 + 2*e1_K2 + 2*e1_K3 + e1_K4)
            pos_e2 += (dt/6)*(e2_K1 + 2*e2_K2 + 2*e2_K3 + e2_K4)
            pos_R += (dt/6)*(R_K1 + 2*R_K2 + 2*R_K3 + R_K4)
        
            print('time=',t)
            print('pos_e1[0] = ', pos_e1[0])
            print('vel_e1[0] = ', (e1_K1[0] + 2*e1_K2[0] + 2*e1_K3[0] + e1_K4[0]))
            print('pos_e2[0] = ', pos_e2[0])
            print('vel_e2[0] = ', (e2_K1[0] + 2*e2_K2[0] + 2*e2_K3[0] + e2_K4[0]))
            print('pos_R[0] = ', pos_R[0])
            print('vel_R[0] = ', (R_K1[0] + 2*R_K2[0] + 2*R_K3[0] + R_K4[0]))
            print('\n')

            t+=dt
            t_index +=1
            if(t_index%psi_save_interval==0):
                np.save(dir_name+'/psi-'+"{:.6}".format(t),psi)
                print("{:.1f}".format(100*t/tf_tot)+'% done')
                print('Norm = ',np.sqrt(np.sum(np.sum(np.sum(np.abs(psi)**2)))*e_spacing**2*n_spacing))
        
#note that t here is the time index not the absolute time
@jit(nogil=True)
def time_propagator_kernel(psi, t, dt, E_form_t, order, t_initial_index):
    #check if the time iterator is within the laser iteration
    if(t<tf_laser_index-t_initial_index):
        if order==4:
            psi = psi - 1j*Ht(psi, E_form_t)*dt - dt**2/2 * Ht(Ht(psi, E_form_t), E_form_t) \
                    + 1j*dt**3/6 * Ht(Ht(Ht(psi, E_form_t), E_form_t), E_form_t) \
                    + dt**4/24 * Ht(Ht(Ht(Ht(psi, E_form_t), E_form_t), E_form_t), E_form_t)
        if order==3:
            psi = psi - 1j*Ht(psi, E_form_t)*dt - dt**2/2 * Ht(Ht(psi, E_form_t), E_form_t) \
                    + 1j*dt**3/6 * Ht(Ht(Ht(psi, E_form_t), E_form_t), E_form_t) \
  
        if order==2:
            psi = psi - 1j*Ht(psi, E_form_t)*dt - dt**2/2 * Ht(Ht(psi, E_form_t), E_form_t) 
    else:
        if order==4:
            psi = psi - 1j*H(psi)*dt - dt**2/2 * H(H(psi)) \
                    + 1j*dt**3/6 * H(H(H(psi))) \
                    + dt**4/24 * H(H(H(H(psi))))
        if order==3:
            psi = psi - 1j*H(psi)*dt - dt**2/2 * H(H(psi)) \
                    + 1j*dt**3/6 * H(H(H(psi))) \
  
        if order==2:
            psi = psi - 1j*H(psi)*dt - dt**2/2 * H(H(psi)) 

    return psi

#need to define global variable psi_t_file where the restart psi_t comes from
# if t_initial is 0 this doesn't matter
# if it is a restart run The old directory name will be changed at the end of the function to reflect the new simulation time
#   NOTE!!!!, on restart, the total simulation time will continue to tf_tot, measured in absolute time
#   So, if you are restarting at 250, and tf_tot is 300, the simulation will only propagate forward by 50
#   This is also true for the laser. 
#   Make sure that for restarting runs with rampdown, that you give tf_laser as the restart time + 10*optical_cycle
#   for smooth ramping
def Time_propagate(psi,dt,tf_tot, tf_laser, order,plotDensity, BOs, t_initial, psi_t_file):
    #if this is not a restart run
    if(t_initial==0):
        #create the name for the folder where the wavefunctions will be stored
        dir_name = './psi-t-form-'+form+'-A-'+str(Amplitude)+'-tf-laser-'+str(tf_laser)+'-tf-tot-'+str(tf_tot) + '-nu-'+str(nu)
        dir_contents = listdir()
        #if one exists under the same laser form, Amplitude, laser pulse duration and total duration
        # ask if it should be deleted
        if dir_name in dir_contents:
            delete = input('Do you want to delete the currently existing folder? (y/n) ')
            if (delete=='y'):
                call(['rm', '-rf', dir_name])
        #if it does not exist, make it
        else:
            call(['mkdir', dir_name])
    #if this is a restart run
    else:
        #for now save everything in the old folder
        dir_name = psi_t_file

        #The wavefunction A, nu, and dt must be the same to pick the simulation back up
        # not to mention the axes and everything else
        throw_away = input('Double Check that the Amplitude, frequency and dt are the same as in the restart!\n Wavefunctions are going to be deleted next!!!!!!!!! (hit Enter to continue) ')
        del throw_away

        print('deleting existing wavefunctions past restart time')
        #get existing directory contents
        dir_contents = listdir(psi_t_file)
        #go through the directory contents
        for wavefunction in dir_contents:
            #if the object in the directory is a wavefunction
            if (wavefunction[0:4] == 'psi-'):
                #check the time step of the wavefunction
                #timestep goes from the dash, index 4, to (inclusive) the decimal point before npy, index -5
                if (float(wavefunction[4:-4])>t_initial):
                    call(['rm', dir_name+'/'+wavefunction])
        print('wavefunctions deleted, may god have mercy on your soul')

    #save the parameters under which the simulation was run
    #all READMEs should agree
    call(['cp', 'params.py', dir_name+'/README'+str(t_initial)])
    call(['touch', dir_name+'/ACTIVE'])
    if(t_initial==0):
        E_form = shape_E(tf_laser,dt,Amplitude, form)
    else:
        E_form = new_shape_E(t_initial, tf_laser, dt, Amplitude)
    plt.ion()
    if plotDensity==True:
        fig1, (ax_e,ax_n, ax_E, ax_norm, ax_occu0, ax_occu1, ax_wf) = plt.subplots(7,1)
        plt.pause(0.0001)
        #fig_rho_e, = ax_e.plot(r1x, psi0[:,find_index(r2x,R0/4),find_index(Rx,R0)],'.')
        fig_rho_e, = ax_e.plot(r1x, one_electron_density(psi),'.')
        #fig_rho_n, = ax_n.plot(Rx, psi0[find_index(r2x,-R0/4),find_index(r2x,R0/4),:],'.')
        fig_rho_n, = ax_n.plot(Rx, integrate_over_electrons(psi),'.')
    else:
        fig1, (ax_E, ax_norm, ax_occu0, ax_occu1, ax_wf) = plt.subplots(5,1)

    E_t = [calc_Et(psi, 0)]
    norm = [np.sqrt(sum(sum(sum(np.conj(psi)*psi)))*e_spacing**2*n_spacing)]
    occupations = BO_occu(psi,BOs)
    gs_occu = [occupations[0]]
    ex_occu = [occupations[1]]
    wf = [E_form[0]]
    fig1.suptitle('Amplitude '+str(Amplitude) + ' Frequency '+str(nu))
    iter_time_plot = [t_initial]
    iter_time = t_initial
    t_initial_index = int(t_initial/dt)
    iter_time_step = 0
    fig_E, = ax_E.plot(iter_time,E_t, '.')
    fig_norm, = ax_norm.plot(iter_time,norm, '.')
    fig_occu0, = ax_occu0.plot(iter_time,gs_occu, '.')
    fig_occu1, = ax_occu1.plot(iter_time,ex_occu, '.')
    fig_wf, = ax_wf.plot(iter_time,wf, '.')
    plt.pause(0.0001)
    fig1.canvas.draw()
    fig1.canvas.flush_events

    while (iter_time < tf_tot):
        iter_time += dt
        iter_time_step += 1
        E_form_t = E_form[iter_time_step]
        psi = time_propagator_kernel(psi, iter_time_step, dt, E_form_t, order, t_initial_index)
        if (int(iter_time/dt)%plot_step==0):
            iter_time_plot.append(iter_time) 
            if(iter_time < tf_laser):
                E_t.append(calc_Et(psi,iter_time_step,E_form))
                wf.append(E_form[iter_time_step])
            else:
                E_t.append(calc_E(psi))
                wf.append(0)
            norm.append(np.sqrt(sum(sum(sum(np.conj(psi)*psi)))*e_spacing**2*n_spacing))
            occupations = BO_occu(psi,BOs)
            gs_occu.append(occupations[0])
            ex_occu.append(occupations[1])
        

            plt.pause(0.0001)
            if (plotDensity == True):
                fig_rho_e.set_ydata(one_electron_density(psi))
                fig_rho_n.set_ydata(integrate_over_electrons(psi))
         
                ax_e.relim()
                ax_e.autoscale()
                ax_n.relim()
                ax_n.autoscale()    
            
            fig_E.set_xdata(iter_time_plot)
            fig_E.set_ydata(E_t)
            ax_E.relim()
            ax_E.autoscale()
            fig_norm.set_xdata(iter_time_plot)
            fig_norm.set_ydata(norm)
            ax_norm.relim()
            ax_norm.autoscale()
            fig_occu0.set_xdata(iter_time_plot)
            fig_occu1.set_xdata(iter_time_plot)
            fig_occu0.set_ydata(gs_occu)
            fig_occu1.set_ydata(ex_occu)
            ax_occu0.relim()
            ax_occu0.autoscale()
            ax_occu1.relim()
            ax_occu1.autoscale()
            fig_wf.set_xdata(iter_time_plot)
            fig_wf.set_ydata(wf)
            ax_wf.relim()
            ax_wf.autoscale()
         
            fig1.canvas.draw()
            fig1.canvas.flush_events
            if (int(iter_time/dt)%psi_save_interval==0):
                np.save(dir_name+'/psi-'+"{:.2f}".format(iter_time), psi)
            np.save(dir_name+'/ex(t)',np.array(ex_occu))
            np.save(dir_name+'/gs(t)',np.array(gs_occu))

    if(t_initial==0):            
        np.save(dir_name+'/E(t)',np.array(E_t))
        np.save(dir_name+'/final-wf-tf-' + str(tf_tot), psi)
        np.save(dir_name+'/gs(t)',np.array(gs_occu))
        np.save(dir_name+'/ex(t)',np.array(ex_occu))
        np.save(dir_name+'/norm(t)',np.array(norm))
        np.save(dir_name+'/time',np.array(iter_time_plot))
    else:
        np.save(dir_name+'/E(t)-from-'+str(t_initial),np.array(E_t))
        np.save(dir_name+'/final-wf-tf-' + str(tf_tot)+'-from-'+str(t_initial), psi)
        np.save(dir_name+'/gs(t)-from-'+str(t_initial),np.array(gs_occu))
        np.save(dir_name+'/ex(t)-from-'+str(t_initial),np.array(ex_occu))
        np.save(dir_name+'/norm(t)-from-'+str(t_initial),np.array(norm))
        np.save(dir_name+'/time-from-'+str(t_initial),np.array(iter_time_plot))
        call(['mv', dir_name, './psi-t-form-'+form+'-A-'+\
            str(Amplitude)+'-tf-laser-'+str(int(tf_laser+t_initial))+'-tf-tot-'+str(int(tf_tot+t_initial))])
    call(['rm', dir_name+'/ACTIVE'])
        



#time_start must be absolute time, not time index!!!!
#only supported for ramp lasers
@jit(nogil=True)
def new_shape_E(time_start, new_tf_laser, dt, A):
    time_steps = int((new_tf_laser - time_start)/dt) + 2 
    Eform = np.zeros(time_steps)

    for i in range(time_steps):
        absolute_time = time_start + i*dt
        if(absolute_time<10*optical_cycle):
            Eform[i] = np.sin(2*np.pi*nu*absolute_time)*(absolute_time/(10.*optical_cycle))
        #if the absolute time is above 10 optical cylce ramp up,
        # as well as less than the duration we want the new laser pulse to be (i*dt=new_tf_laser)
        # then we're in the full sine wave
        elif(absolute_time>10*optical_cycle and absolute_time < new_tf_laser-10*optical_cycle):
            Eform[i] = np.sin(2*np.pi*nu*absolute_time)
        else:
            Eform[i] = np.sin(2*np.pi*nu*absolute_time)*(new_tf_laser/(10*optical_cycle)-(absolute_time/(10*optical_cycle)))
    return A*Eform
     

#}}}
#{{{ Electronic BO stuff
#{{{ H_el set up
#mu_e is 1 in BO approx
T_e1_BO = csr_matrix((-1/(2*e_spacing**2))*Lap_e1 + 0j)
T_e2_BO = csr_matrix((-1/(2*e_spacing**2))*Lap_e2 + 0j)
#relies on r1x and r2x being the same size
T_el_buff = np.zeros(psi0[:,:,0].shape) +0j
@jit(nogil=True)
def calc_Tel(psi_el, buff):
    buff[:,:] =  0j
    if (len(r1x) == len(r2x)):
        for xi in range(len(r1x)):
            buff[:,xi] += T_e1_BO.dot(psi_el[:,xi])
            buff[xi,:] += T_e2_BO.dot(psi_el[xi,:])
    else:
        for xi in range(len(r2x)):
            buff[:,xi] += T_e1_BO.dot(psi_el[:,xi])
        for xi in range(len(r1x)):
            buff[xi,:] += T_e2_BO.dot(psi_el[xi,:])

    return buff


#need to deal with each subspecies interaction separately
#create kernels for each of these separately, store them in kernel array
#store as [Vee, Ve1n, Ve2n], add 1/R separately
#These kernels correctly reconstruct the potential as checked against Mathematica
@jit(nogil=True)
def construct_Vel(V_kernels):
    for r1i in range(len(r1x)):
        for r2i in range(len(r2x)):
            #electron electron interaction
            #SHOULD THERE BE A 1/2 HERE? I dont thing so since sum over i != j
            V_kernels[0][r1i,r2i] += np.sqrt( (r1x[r1i] - r2x[r2i])**2 + Cr2)**(-1)

        #grab the r1 electron nuclear interactions while looping over r1
        for Ri in range(Ri_cut,len(Rx)):
            V_kernels[1][r1i,Ri] += (-np.sqrt( (r1x[r1i] - (mu/M1)*Rx[Ri])**2 + Cr2)**(-1) -\
                             np.sqrt( (r1x[r1i] + (mu/M2)*Rx[Ri])**2 + Cr2)**(-1))
    
    for Ri in range(Ri_cut,len(Rx)):
        #grab the r2 electron nuclear interactions while looping over R
        for r2i in range(0,len(r2x)):
            V_kernels[2][r2i,Ri] += (-np.sqrt( (r2x[r2i] - (mu/M1)*Rx[Ri])**2 + Cr2)**(-1) -\
                             np.sqrt( (r2x[r2i] + (mu/M2)*Rx[Ri])**2 + Cr2)**(-1))

    return V_kernels

Vel_kernels = [np.zeros((len(r1x),len(r2x))) + 0j, np.zeros((len(r1x),len(Rx))) + 0j,\
             np.zeros((len(r2x),len(Rx))) + 0j]

Vel_kernels = construct_Vel(Vel_kernels)


#agrees with slower more direct method of matvec'ing
@jit(nogil=True)
def calc_Vel(psi_el, Vel_kernels, Ri):
    #Since r1/r2 are independent subsystems, one has to apply the potential kernels separately
    if(len(r1x)!=len(r2x)):
        Ve1n_psi = np.zeros(psi_el.shape) + 0j
        for i in range(len(r2x)):
            Ve1n_psi[:,i] += Vel_kernels[1][:,Ri]*psi_el[:,i]
        Ve2n_psi = np.zeros(psi_el.shape) + 0j
        for i in range(len(r1x)):
            Ve2n_psi[i,:] += Vel_kernels[2][:,Ri]*psi_el[i,:]
        return  Vel_kernels[0][:,:]*psi_el[:,:] + Ve1n_psi +  1./Rx[Ri]*psi_el[:,:] + Ve2n_psi
    
    else:
        Ven_psi = np.zeros(psi_el.shape) + 0j
        for i in range(len(r2x)):
            Ven_psi[:,i] += Vel_kernels[1][:,Ri]*psi_el[:,i]
            Ven_psi[i,:] += Vel_kernels[2][:,Ri]*psi_el[i,:]
        return  Vel_kernels[0][:,:]*psi_el[:,:] + Ven_psi  + 1./Rx[Ri]*psi_el[:,:]

@jit(nogil=True)
def cVel(Ri):
    Vpsi = 1.0*np.zeros((len(r1x),len(r2x))) + 0.0j
    for r1i in range(len(r1x)):
        for r2i in range(len(r2x)):
            #Vpsi[r1i,r2i] += psi_el[r1i,r2i]/np.sqrt( (r1x[r1i] - r2x[r2i])**2 + Cr2)
            #Vpsi[r1i,r2i] -= psi_el[r1i,r2i]/np.sqrt( (r1x[r1i] - (mu/M1)*Rx[Ri])**2 + Cr2) 
            #Vpsi[r1i,r2i] -= psi_el[r1i,r2i]/np.sqrt( (r1x[r1i] + (mu/M2)*Rx[Ri])**2 + Cr2) 
            #Vpsi[r1i,r2i] -= psi_el[r1i,r2i]/np.sqrt( (r2x[r2i] - (mu/M1)*Rx[Ri])**2 + Cr2) 
            #Vpsi[r1i,r2i] -= psi_el[r1i,r2i]/np.sqrt( (r2x[r2i] + (mu/M2)*Rx[Ri])**2 + Cr2)
            #Vpsi[r1i,r2i] += psi_el[r1i,r2i]/Rx[Ri]
            Vpsi[r1i,r2i] += 1./np.sqrt( (r1x[r1i] - r2x[r2i])**2 + Cr2)
            Vpsi[r1i,r2i] -= 1./np.sqrt( (r1x[r1i] - (mu/M1)*Rx[Ri])**2 + Cr2) 
            Vpsi[r1i,r2i] -= 1./np.sqrt( (r1x[r1i] + (mu/M2)*Rx[Ri])**2 + Cr2) 
            Vpsi[r1i,r2i] -= 1./np.sqrt( (r2x[r2i] - (mu/M1)*Rx[Ri])**2 + Cr2) 
            Vpsi[r1i,r2i] -= 1./np.sqrt( (r2x[r2i] + (mu/M2)*Rx[Ri])**2 + Cr2)
            Vpsi[r1i,r2i] += 1./Rx[Ri]

    return Vpsi  
#}}}
#{{{ Normalization, H_el, calc_E_el, plotting, constructing
#define nomalization function over the electronic subsystem
@jit(nogil=True)
def N_el(ket):
    return ket/np.sqrt( sum(sum(np.conj(ket)*ket))*e_spacing**2)


@jit(nogil=True)
def H_el(psi_el,Ri):
    return calc_Vel(psi_el,Vel_kernels,Ri) + calc_Tel(psi_el,T_el_buff)

@jit(nogil=True)
def calc_E_el(psi_el,Ri):
    return sum(sum(np.conj(psi_el)*H_el(psi_el,Ri)))*e_spacing**2

#normalize and return first order imaginary time expansion
@jit(nogil=True)
def imag_prop_el(psi_el, Ri, tau):
    return N_el(psi_el - tau*H_el(psi_el,Ri))

@jit(nogil=True)
def inner_product_el(bra,ket):
    return sum(sum(np.conj(bra)*ket))*e_spacing**2

@jit(nogil=True)
def filter_projection_el(psi, kets): 
    #where psi is the wavefunction to be filtered and ket is the ket to filter out
    projection = np.zeros(psi.shape) +0j
    for ket_i in kets:
        projection += inner_product_el(ket_i,psi)*ket_i
    return psi - projection

@jit(nogil=True)
def filtered_imag_prop_el(psi_el,Ri,tau, filters):
    return N_el(psi_el - tau*filter_projection_el(H_el(filter_projection_el(psi_el,filters),Ri),filters))

#for each BO calculation start the initial electronic wavefunction as two gaussians centered around
#   the nuclei at that given nuclear separation value
#   note that this assumes the M1=M2 
@jit(nogil=True)
def construct_electronic_wf(psi_el0, Ri):
    for r1i in range(len(r1x)):
        for r2i in range(len(r2x)):
            psi_el0[r1i,r2i] = np.exp(-(r1x[r1i] - Rx[Ri]/2)**2/re_sig2)\
                                *np.exp(-(r2x[r2i] + Rx[Ri]/2)**2/re_sig2) \
                                + np.exp(-(r2x[r2i] - Rx[Ri]/2)**2/re_sig2)\
                                 *np.exp(-(r1x[r1i] + Rx[Ri]/2)**2/re_sig2) +0j
    return N_el(psi_el0)

#for each BO calculation start the initial electronic wavefunction as two gaussians centered around
#   the nuclei at that given nuclear separation value 
#this is properly the first excited state, could do fancy footwork with cosines to get higher ones
@jit(nogil=True)
def construct_excited_electronic_wf(psi_el0, Ri):
    for r1i in range(len(r1x)):
        for r2i in range(len(r2x)):
            psi_el0[r1i,r2i] =(np.exp(-(r1x[r1i] - Rx[Ri]/2)**2/re_sig2)\
                 *np.exp(-(r2x[r2i] + Rx[Ri]/2)**2/re_sig2)
                  - np.exp(-(r2x[r2i] - Rx[Ri]/2)**2/re_sig2)\
                 *np.exp(-(r1x[r1i] + Rx[Ri]/2)**2/re_sig2) ) +0j
    return N_el(psi_el0)

@jit(nogil=True)
def BO_electronic_density(psi_el):
    out = np.zeros(len(r1x))
    psi_el2 = abs(psi_el)**2
    out[:] = np.sum(psi_el2,0)
    out[:] += np.sum(psi_el2,1)
    return out*e_spacing            

@jit(nogil=True)
def check_convergence_el(psi, Ri):
    Hpsi = H_el(psi,Ri)
    E = (sum(sum(np.conj(psi)*Hpsi)))*e_spacing**2
    #difference vector
    d_vec = Hpsi - E*psi
    #distance from correct solution
    d = np.sqrt(sum(sum(np.conj(d_vec)*d_vec))*e_spacing**2)
    return d, E

#}}}
#{{{ SD and cgs
#Start with cgs for gs, later expand algorithm for any number of excited states
# to extend this to search for excited states, add an option for number of states,
# then when initializing the algorithm build a list for each up to the number of excited states
# then for loop over the number of states, embedding the while condition inside,
# making sure to filter each time!!
@jit(nogil=True)
def sd_el(Ri, tol, num_excited):
    eigenstates = []
    eigenvalues = []
    for i in range(num_excited+1):
        #make an initial guess
        #may need a random component to access different directions of function space
        if(i==0):
            psi_m = construct_electronic_wf(np.zeros((len(r1x),len(r2x))),Ri) + 0j
        if(i>0):
            #could write in better excited states for greater than first
            psi_m = N_el(filter_projection_el(\
                construct_excited_electronic_wf(np.zeros((len(r1x),len(r2x))),Ri) + 0j, \
                eigenstates))
        conj = np.zeros(psi_m.shape) +0j 

        #calculate the 0th iteration eigenvalue guess
        lam = calc_E_el(psi_m,Ri)
        r = lam*psi_m - H_el(psi_m,Ri)
        if(i>0):
            r = filter_projection_el(r,eigenstates)
        
        alpha = inner_product_el(r,r)/calc_E_el(r,Ri)

        psi_m = psi_m + alpha*r

        #check for convergence
        dist, lam = check_convergence_el(psi_m,Ri)

        #enter algorithm
        while (dist > tol):
            r = lam*psi_m - H_el(psi_m,Ri)
            if(i>0):
                r = filter_projection_el(r,eigenstates)
            alpha = inner_product_el(r,r)/calc_E_el(r,Ri)

            psi_m = psi_m + alpha*r

            #chec convergence, update lambda 
            dist, lam = check_convergence_el(psi_m, Ri)
            print('dist=',dist)
        print(i, ' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!state done!')
        eigenstates.append(psi_m)
        eigenvalues.append(lam)
    return eigenvalues, eigenstates

@jit(nogil=True)
def cg_el(Ri,tol):
    #make an initial guess
    #may need a random component to access different directions of function space
    psi_m = construct_electronic_wf(np.zeros((len(r1x),len(r2x))),Ri) + 0j
    lam = calc_E_el(psi_m1,Ri)
    d = lam*psi_m - H_el(psi_m,Ri)
    r = d
    ri_inner = inner_product_el(r,r)
    beta = 0
    dist, lam = check_convergence_el(psi_m,Ri)
    while (dist > tol):
        Hd = H_el(d,Ri)
        alpha = inner_product_el(r,r)/inner_product_el(d,Hd)
        psi_m = psi_m + alpha*d
        ri_inner = inner_product_el(r,r)
        r = r - alpha*Hd
        beta = inner_product_el(r,r)/ri_inner
        d = r + beta*d
        dist,lam = check_convergence_el(psi_m,Ri)
        print('dist = ', dist)
    return lam        

def sd_BO(tol, num_excited):
    BO_energies = []
    BO_states = []
    for i in range(num_excited+1):
        BO_energies.append([])
        BO_states.append([])
    for Ri in range(len(Rx)):
        energies, states = sd_el(Ri,tol,num_excited)
        for i in range(num_excited+1):
            BO_energies[i].append(energies[i])
            BO_states[i].append(states[i])
    return BO_energies, BO_states

def cg_BO(tol):
    for Ri in range(len(Rx)):
        BO_0.append(cg_el(Ri,tol))
#}}}
#{{{ BO imaginary time prop
#could save electronic eigenstates at each Ri as well rather easily
#somehow this is faster without jit
#@jit(nogil=True)
def calculate_BO(tau, tol):
    psi_el_buffer = np.zeros( (len(r1x), len(r2x)) )
    #ones = np.ones(psi_el_buffer.shape) + 0j #/ (sum(sum(np.ones(psi_el_buffer.shape)))*e_spacing**2)
    for Ri in range(len(Rx)):
        #imaginary time relax the electronic system to a tolerance, record the energy value.
        #then filter the electronic gs out to get the first excited electronic BO state, and its energy value.
        psi_el = construct_electronic_wf(psi_el_buffer, Ri) + 0j
        #psi_el = construct_excited_electronic_wf(psi_el_buffer, Ri) + 0j
        #psi_el = ones 
        psi_el0 = psi_el
        dis, E0 = check_convergence_el(psi_el0,Ri)

        while (dis > tol):
            psi_el0 = imag_prop_el(psi_el0, Ri, tau)
            dis, E0 = check_convergence(psi_el0, Ri)
            print('E0 dis = ',dis)
        BO_0.append(E0)
        plt.plot(r1x,BO_electronic_density(psi_el0))
        plt.show()
        """
        #calculate the first excited state
        psi_el1 = construct_excited_electronic_wf(psi_el_buffer, Ri) + 0j
        #psi_el1 = psi_el
        dis, E1 = check_convergence(psi_el1,Ri)

        while (dis > tol):
            psi_el1 = filtered_imag_prop_el(psi_el1, Ri, tau, [psi_el0])
            dis, E1 = check_convergence(psi_el1, Ri)
            print('E1 dis = ',dis)
        BO_1.append(E1)

        #calculate the second excited state
        psi_el2 = construct_excited_electronic_wf(psi_el_buffer, Ri) + 0j
        dis, E2 = check_convergence(psi_el2,Ri)

        while (dis > tol):
            psi_el2 = imag_prop_el(psi_el2, Ri, tau)
            dis, E2 = check_convergence(psi_el2, Ri)
            print('E2 dis = ',dis)
        BO_2.append(E2)

        """
        """
        #calculate third excited state
        E1 = calc_E_el(psi_el, Ri)
        psi_el3 = filtered_imag_prop_el(psi_el, Ri, tau, [psi_el0, psi_el1, psi_el2])
        E2 = calc_E_el(psi_el3, Ri) 

        #converge the electronic eigenvalue problem to the ground state
        counter = 0
        while (abs(E2 - E1) > tol):
            print(abs(E2-E1))
            counter += 1
            psi_el3 = filtered_imag_prop_el(psi_el1, Ri, tau, [psi_el0, psi_el1,psi_el2])
            if(counter%2==0):
                E2 = calc_E_el(psi_el3, Ri)
            else:
                E1 = calc_E_el(psi_el3, Ri)
        #plt.plot(r1x,BO_electronic_density(psi_el0))
        #plt.plot(r1x,BO_electronic_density(psi_el1))
        #plt.show()
        BO_3.append(min([E1,E2]))
        """
        print('BO relax is ', float(Ri)/len(Rx)*100, '% of the way done')

#}}}
#{{{ BO occupations

@jit(nogil=True)
def BO_occu(psi_full, psi_els):
    #note that psi els are organized as such
    #   psi_els[gs or excited, Nuclear index, r1x, r2x]
    egs_on_psi = np.zeros(len(Rx)) + 0j
    eex_on_psi = np.zeros(len(Rx)) + 0j
    for Ri in range(len(Rx)):
        egs_on_psi[Ri] = sum(sum(np.conj(psi_els[0,Ri])*psi_full[:,:,Ri]))*e_spacing**2
        eex_on_psi[Ri] = sum(sum(np.conj(psi_els[1,Ri])*psi_full[:,:,Ri]))*e_spacing**2
    gs = np.abs(sum(np.conj(egs_on_psi)*egs_on_psi)*n_spacing)**2
    ex = np.abs(sum(np.conj(eex_on_psi)*eex_on_psi)*n_spacing)**2
    return gs, ex
    
    

#}}}
@jit(nogil=True)
def D_el(psi_el,Ri):
    Dpsi = np.zeros(psi_el.shape)+0j
    for r2i in range(len(r2x)):
        Dpsi[:,r2i] -= psi_el[:,r2i]*r1x[:]
    for r1i in range(len(r1x)):
        Dpsi[r1i,:] -= psi_el[r1i,:]*r2x[:]

    return Dpsi


#}}}
