from importlib import reload
import functions as f
reload(f)
import params as p
reload(p)
import numpy as np
from matplotlib import pyplot as plt
from numba import jit, prange, njit#, vectorize, float32, complex64, int32, complex128, float64, int64
from math import floor
#import os
from subprocess import call
from test import print_norm
from decimal import Decimal

num_trajs = p.num_trajs
@jit( nogil=True)
def e2_separable_correlated_wf():
    out = np.zeros((len(p.r1x), len(p.r2x), len(p.Rx)))+0j
    for r1i in range(len(p.r1x)):
        for r2i in range(len(p.r2x)):
            for Ri in range(len(p.Rx)):
                #out[r1i, r2i, Ri] = (np.exp(-(p.r1x[r1i]-1)**2)*np.exp(-(p.r2x[r2i]+1)**2) \
                #    + np.exp(-(p.r1x[r2i] - 1)**2)*np.exp(-(p.r2x[r1i]+1)**2))*np.exp(-p.Rx[Ri]**2)
                out[r1i, r2i, Ri] = (np.exp(-(p.r1x[r1i]-1)**2)*np.exp(-(p.r2x[r2i]+1)**2))\
                                        *np.exp(-p.Rx[Ri]**2)
    return out/np.sqrt(np.sum(np.sum(np.sum(np.abs(out)**2)))*p.e_spacing**2*p.n_spacing)

#{{{mesh initializers
#may be better to have the pdf defined globally rather than calculated locally and forgotten
@jit( nogil=True)
def initialize_mesh(threshold, psi0, Unique, num_trajs):
    print('Initalizing Mesh')
    pdf = np.conj(psi0)*psi0#*e_spacing**2*n_spacing
    r1_mesh = []
    r2_mesh = []
    R_mesh = []
    #for making sure the slice is unique
    triplets = []
    #start with nuclear conditional slices
    for i in range(num_trajs):
        valR = 0
        valr1 = 0
        valr2 = 0
        val_array = [valr1, valr2, valR]
        #check all three indices at once, assures value sampling
        #but there's no way to tell how long this will take
        #doing the indices separately would also work.
        iterations = 0
        while (min(val_array)<threshold):#(len(r1_mesh)<num_trajs):
            iterations += 1
            if(iterations%100000==0):
                print('Saturated Unique positions at 1 million guesses.')
                Unique=False
                #return ValueError("Threshold too high")
            r1_alpha = floor(len(p.r1x)*np.random.rand())
            r2_alpha = floor(len(p.r2x)*np.random.rand())
            R_alpha = floor(len(p.Rx)*np.random.rand())
            if(Unique==True):
                #make sure that the slice found is unique
                if([r1_alpha, r2_alpha, R_alpha] in triplets[:i]):
                    val_array[0] = i+threshold
                    continue
            #check if the integral over the conditional slice's pdf is within the tolerance  
            val_array[0] = np.sum(pdf[:,r2_alpha,R_alpha])*p.e_spacing
            val_array[1] = np.sum(pdf[r1_alpha,:,R_alpha])*p.e_spacing
            val_array[2] = np.sum(pdf[r1_alpha,r2_alpha,:])*p.n_spacing
        r1_mesh.append(r1_alpha)
        r2_mesh.append(r2_alpha)
        R_mesh.append(R_alpha)
        triplets.append([r1_alpha, r2_alpha, R_alpha])
    print('mesh initialized')
    return np.array(r1_mesh), np.array(r2_mesh), np.array(R_mesh)

@jit( nogil=True)
def initialize_2e_mesh(threshold,psi0, Unique):
    pdf = np.abs(psi0)**2#*e_spacing**2*n_spacing
    r1_mesh = []
    r2_mesh = []
    R_mesh = []
    #for making sure the slice is unique
    triplets = []
    #start with nuclear conditional slices
    for i in range(num_trajs):
        valR = 0
        val_e = 0
        val_array = [val_e, valR]
        #check all three indices at once, assures value sampling
        #but there's no way to tell how long this will take
        #doing the indices separately would also work.
        iterations = 0
        while (min(val_array)<threshold):#(len(r1_mesh)<num_trajs):
            iterations += 1
            if(iterations%100000==0):
                print('Saturated Unique positions at 1 million guesses.')
                Unique=False
            r1_alpha = floor(len(p.r1x)*np.random.rand())
            r2_alpha = floor(len(p.r2x)*np.random.rand())
            R_alpha = floor(len(p.Rx)*np.random.rand())
            if(Unique==True):
                #make sure that the slice found is unique
                if([r1_alpha, r2_alpha, R_alpha] in triplets[:i]):
                    val_array[0] = i+threshold
                    continue
            #check if the integral over the conditional slice's pdf is within the tolerance  
            val_array[0] = np.sum(np.sum(pdf[:,:,R_alpha]))*p.e_spacing**2
            val_array[1] = np.sum(pdf[r1_alpha,r2_alpha,:])*p.n_spacing
        r1_mesh.append(r1_alpha)
        r2_mesh.append(r2_alpha)
        R_mesh.append(R_alpha)
        triplets.append([r1_alpha, r2_alpha, R_alpha])
        print('initialize mesh is ' + str(100*float(i)/num_trajs) +'% done')
    return r1_mesh, r2_mesh, R_mesh
    
@jit( nogil=True)
def check_uniqueness(r1,r2,R):
    triplets = []
    for i in range(len(r1)):
        sli = [r1[i],r2[i],R[i]]
        triplets.append(sli)
        if (sli in triplets[:i] ):
            original = triplets.index(sli)
            return 'repeated value ' + str(i) + ' original value ' + str(original)
    return 'COOL!'
#}}}
#{{{ velocity calculators
#pass this function the conditional wavefunctions psi_full[:,e2_mesh,R_mesh]
#calculate across the entire mesh at once
@jit( nogil=True)
def calc_e1_vel(psi_e):
    #the shape should be (num_traj, dim_e1)
    if (psi_e.shape[1] == num_trajs):
        psi_e = psi_e.transpose()
    vel = np.zeros((num_trajs, len(p.r1x))) + 0j
    grad_psi = np.zeros(len(p.r1x)) + 0j
    for traj in range(num_trajs):
        grad_psi = np.gradient(psi_e[traj,:],p.e_spacing)
        #the below does an element by element division
        vel[traj,:] = grad_psi/psi_e[traj,:]
    return vel.imag


#the same as the above but more efficient for case when len(r1x) == len(r2x)
#Calculates the velocity field everywhere for each trajectory, then return this field
#evaluated at the current trajectory position
@jit( nogil=True)
def calc_e_vel(psi_e1, psi_e2, e1_mesh, e2_mesh):
    #the shape should be (num_traj, dim_e1)
    #This automatic correction will not help if by chance num_traj == dim_e1
    if (psi_e1.shape[1] == num_trajs):
        psi_e1 = psi_e1.transpose()
    vel = np.zeros((2, num_trajs)) + 0j
    for traj in range(num_trajs):
        grad_psi_1 = np.gradient(psi_e1[traj,:],p.e_spacing)
        grad_psi_2 = np.gradient(psi_e2[traj,:],p.e_spacing)
        #the below does an element by element division
        vel[0,traj] = (grad_psi_1/psi_e1[traj,:])[e1_mesh[traj]]
        vel[1,traj] = (grad_psi_2/psi_e2[traj,:])[e2_mesh[traj]]
    return vel[0].imag, vel[1].imag

#psi_e2 will come in as (num_trajs, dim_e1, dim_e2)
@jit( parallel=True,nogil=True)
def calc_2e_vel(psi_2e, e1_mesh, e2_mesh):
    #the shape should be (num_traj, dim_e1)
    vel = np.zeros((2, num_trajs))
    for traj in prange(num_trajs):
        grad_psi_1 = np.gradient(psi_2e[traj,:,e2_mesh[traj]],p.e_spacing)
        grad_psi_2 = np.gradient(psi_2e[traj,e1_mesh[traj],:],p.e_spacing)
        #the below does an element by element division
        vel[0,traj] = (grad_psi_1/psi_2e[traj,:,e2_mesh[traj]])[e1_mesh[traj]].imag
        vel[1,traj] = (grad_psi_2/psi_2e[traj,e1_mesh[traj],:])[e2_mesh[traj]].imag
    return vel[0], vel[1]

#Note that the mass term must be different from mu 
@jit( nogil=True)
def calc_n_vel(psi_n, R_mesh):
    #the shape should be (num_traj, dim_e1)
    if (psi_n.shape[1] == num_trajs):
        psi_n = psi_n.transpose()
    vel = np.zeros(num_trajs) + 0j
    #vel_field = np.zeros((num_trajs,len(p.Rx))) + 0j
    grad_psi = np.zeros(len(p.Rx)) + 0j
    for traj in range(num_trajs):
        grad_psi = np.gradient(psi_n[traj,:],p.n_spacing)
        #the below does an element by element division
        vel[traj] = (grad_psi/psi_n[traj,:])[R_mesh[traj]]
        #vel_field[traj,:] = (grad_psi/psi_n[traj,:])
        #vel[traj] = vel_field[traj][R_mesh[traj]]
    return vel.imag/p.mu#, vel_field.imag/p.mu

@jit( parallel=True,nogil=True)
def ICWF_vel(C,psi_el,psi_R, e1_mesh, e2_mesh, R_mesh):
    el1_num = np.zeros(psi_el.shape[0],dtype=np.complex128)
    el2_num = np.zeros(psi_el.shape[0],dtype=np.complex128)
    R_num = np.zeros(psi_R.shape[0],dtype=np.complex128)
    denom = 0j
    for i in prange(psi_el.shape[0]):
        el1_num[i] = C[i]*psi_R[i,R_mesh[i]]*np.gradient(psi_el[i,:,e2_mesh[i]])[e1_mesh[i]]
        el2_num[i] = C[i]*psi_R[i,R_mesh[i]]*np.gradient(psi_el[i,e1_mesh[i],:])[e2_mesh[i]]
        R_num[i] = C[i]*psi_el[i,e1_mesh[i],e2_mesh[i]]*np.gradient(psi_R[i,:])[R_mesh[i]]
        denom += C[i]*psi_el[i,e1_mesh[i],e2_mesh[i]]*psi_R[i,R_mesh[i]]
    return (el1_num/denom).imag, (el2_num/denom).imag, (R_num/denom).imag

@jit(nogil=True)
def ICWF_1e_vel(C,psi_el1,psi_el2,psi_R, e1_mesh, e2_mesh, R_mesh, num_trajs):
    el1_num = np.zeros(num_trajs,dtype=np.complex128)
    el2_num = np.zeros(num_trajs,dtype=np.complex128)
    #tester = np.zeros(r1_dim,dtype=np.complex128)
    R_num = np.zeros(num_trajs,dtype=np.complex128)
    denom = np.zeros(num_trajs,dtype=np.complex128)
    grad_el1 = np.zeros((num_trajs,r1_dim),dtype=np.complex128)
    grad_el2 = np.zeros((num_trajs,r2_dim),dtype=np.complex128)
    grad_R = np.zeros((num_trajs,R_dim),dtype=np.complex128)
    for i in range(num_trajs):
        grad_el1[i] = np.gradient(psi_el1[i,:], p.e_spacing)
        grad_el2[i] = np.gradient(psi_el2[i,:], p.e_spacing)
        grad_R[i] = np.gradient(psi_R[i,:], p.n_spacing)
    for i in range(num_trajs):
        el1_num[i] = np.dot(C,psi_el2[:,e2_mesh[i]]*psi_R[:,R_mesh[i]]*grad_el1[:,e1_mesh[i]])
        el2_num[i] = np.dot(C,psi_el1[:,e1_mesh[i]]*psi_R[:,R_mesh[i]]*grad_el2[:,e2_mesh[i]])
        R_num[i] = np.dot(C,psi_el1[:,e1_mesh[i]]*psi_el2[:,e2_mesh[i]]*grad_R[:,R_mesh[i]])
        denom[i] = np.dot(C,psi_el1[:,e1_mesh[i]]*psi_el2[:,e2_mesh[i]]*psi_R[:,R_mesh[i]])
    #for r in range(r1_dim):
    #    tester[r] = np.dot(C,psi_el2[:,e2_mesh[0]]*psi_R[:,R_mesh[0]]*grad_el1[:,r])/denom[0]
    #print('el1_vel[0] = ', (el1_num[0]/denom[0]).imag)
    #plt.plot(tester.imag)
    #plt.plot(np.abs(tester2))
    #plt.title('ICWF')
    #plt.show()
    return (el1_num/(denom)).imag, (el2_num/(denom)).imag, (R_num/(p.mu*denom)).imag

def return_vels(C,psi_el1,psi_el2,psi_R, e1_mesh, e2_mesh, R_mesh, num_trajs):
    el1_num = np.zeros((num_trajs,r1_dim),dtype=np.complex128)
    el2_num = np.zeros((num_trajs,r2_dim),dtype=np.complex128)
    R_num = np.zeros((num_trajs,R_dim),dtype=np.complex128)
    denom = np.zeros(num_trajs,dtype=np.complex128)
    grad_el1 = np.zeros((num_trajs,r1_dim),dtype=np.complex128)
    grad_el2 = np.zeros((num_trajs,r2_dim),dtype=np.complex128)
    grad_R = np.zeros((num_trajs,R_dim),dtype=np.complex128)
    for i in range(num_trajs):
        grad_el1[i] = np.gradient(psi_el1[i,:], p.e_spacing)
        grad_el2[i] = np.gradient(psi_el2[i,:], p.e_spacing)
        grad_R[i] = np.gradient(psi_R[i,:], p.n_spacing)
    for i in range(num_trajs):
        for j in range(len(C)):
            el1_num[i] += C[j]*psi_el2[j,e2_mesh[i]]*psi_R[j,R_mesh[i]]*grad_el1[j]
            el2_num[i] += C[j]*psi_el1[j,e1_mesh[i]]*psi_R[j,R_mesh[i]]*grad_el2[j]
            R_num[i] += C[j]*psi_el2[j,e2_mesh[i]]*psi_el1[j,e1_mesh[i]]*grad_R[j]
        denom[i] = np.dot(C,psi_el1[:,e1_mesh[i]]*psi_el2[:,e2_mesh[i]]*psi_R[:,R_mesh[i]])
        el1_num[i]/=denom[i]
        el2_num[i]/=denom[i]
        R_num[i]/=denom[i]
    return (el1_num).imag, (el2_num).imag, (R_num/p.mu).imag
#}}}
#{{{ Density functions
@jit( nogil=True)
def cwf_nuclear_density(psi_n):
    rho_R = np.zeros(p.Rx_array.shape)
    for i in range(num_trajs):
        rho_R += np.abs(psi_n[i])**2
    return rho_R/(np.sum(rho_R)*p.n_spacing)

@jit( nogil=True)
def cwf_nuclear_density_BHF(psi_n):
    rho_R = np.zeros(len(p.Rx))
    for i in range(num_trajs):
        rho_R += np.abs(psi_n[i,i])**2
    return rho_R/(np.sum(rho_R)*p.n_spacing)

@jit( nogil=True)
def cwf_electronic_density(psi_e1,psi_e2):
    rho_e = np.zeros(len(p.r1x))
    for i in range(num_trajs):
        rho_e += np.abs(psi_e1[i])**2 + np.abs(psi_e2[i])**2
    return 2*rho_e/(sum(rho_e)*p.e_spacing)

@jit( nogil=True)
def cwf_2e_density(two_e_wfs):
    rho_e = np.zeros(len(p.r1x))
    for i in range(num_trajs):
        rho_e += .5*np.sum(np.abs(two_e_wfs[i,:,:])**2,1)*p.e_spacing \
                + .5*np.sum(np.abs(two_e_wfs[i,:,:])**2,0)*p.e_spacing
    return 2*rho_e/(np.sum(rho_e)*p.e_spacing)
#}}}
#{{{ T and V
#takes in a given two electron conditional wavefunction, i.e. with a given classical nuclear pos.
#like everything else in the two electron conditional wf approach, len(r1x) needs = len(r2x)
r1x_a = p.r1x_array
r2x_a = p.r2x_array
Rx_a = p.Rx_array
r1_dim = r1x_a.shape[0]
r2_dim = r2x_a.shape[0]
R_dim = Rx_a.shape[0]
Cr2 = p.Cr2
mu = p.mu
M1 = p.M1
M2 = p.M2
@jit( nogil=True)
def two_e_T(two_e_wf):
    out = np.zeros(two_e_wf.shape)+0j
    for r1i in range(len(p.r1x)):
        out[r1i,:] += f.T_e1.dot(two_e_wf[r1i,:])
        out[:,r1i] += f.T_e1.dot(two_e_wf[:,r1i])
    return out

@jit( nogil=True)
def construct_2e_V():
    V2e = np.zeros((len(p.r1x),len(p.r2x),len(p.Rx)))
    for Ri in range(len(p.Rx)):
        for r2i in range(len(p.r2x)):
            V2e[:,r2i,Ri] = 1.0/(np.sqrt( (p.r1x - p.r2x[r2i])**2 + Cr2)) + 1.0/p.Rx[Ri]\
                        - 1.0/(np.sqrt( (p.r1x - mu*p.Rx[Ri]/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (p.r1x + mu*p.Rx[Ri]/M2)**2 + Cr2)\
                        - 1.0/(np.sqrt( (p.r2x[r2i] - mu*p.Rx[Ri]/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (p.r2x[r2i] + mu*p.Rx[Ri]/M2)**2 + Cr2)
    return V2e
V2e = construct_2e_V()

#Define cwf potential kernels such that for the electron 1 potential one references
#   (mesh_e2, mesh_R)
# and for the electron 2 potential one refernces
#   (mesh_e1, mesh_R)
# and for the nuclear potential one refrences
#   (mesh_e1,mesh_e2)
#New addition is the static potential of the other, trajectory fixed species
@jit( nogil=True)
def construct_cwf_V1():
    V1 = np.zeros((len(p.r1x),len(p.r2x),len(p.Rx)))
    for r2i in range(len(p.r2x)):
        for Ri in range(len(p.Rx)):
            V1[:,r2i,Ri] =  1.0/(np.sqrt( (p.r1x - p.r2x[r2i])**2 + Cr2)) \
                        - 1.0/(np.sqrt( (p.r1x - mu*p.Rx[Ri]/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (p.r1x + mu*p.Rx[Ri]/M2)**2 + Cr2)
                        #+1.0/p.Rx[Ri]
                        #- 1.0/(np.sqrt( (p.r2x[r2i] - mu*p.Rx[Ri]/M1)**2 +Cr2))\
                        #- 1.0/np.sqrt( (p.r2x[r2i] + mu*p.Rx[Ri]/M2)**2 + Cr2)
    return V1


@jit( nogil=True)
def construct_cwf_V2():
    V2 = np.zeros((len(p.r1x),len(p.r2x),len(p.Rx)))
    for r1i in range(len(p.r1x)):
        for Ri in range(len(p.Rx)):
            V2[r1i,:,Ri] = 1.0/(np.sqrt( (p.r2x - p.r1x[r1i])**2 + Cr2))\
                        - 1.0/(np.sqrt( (p.r2x - mu*p.Rx[Ri]/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (p.r2x + mu*p.Rx[Ri]/M2)**2 + Cr2)
                        #1.0/p.Rx[Ri] 
                        #- 1.0/(np.sqrt( (p.r1x[r1i] - mu*p.Rx[Ri]/M1)**2 +Cr2))\
                        #- 1.0/np.sqrt( (p.r1x[r1i] + mu*p.Rx[Ri]/M2)**2 + Cr2)             
    return V2

@jit( nogil=True)
def construct_cwf_VR():
    VR = np.zeros((len(p.r1x),len(p.r2x),len(p.Rx)))
    for r1i in range(len(p.r1x)):
        for r2i in range(len(p.r2x)):
            VR[r1i,r2i,:] = 1.0/p.Rx_array- 1.0/(np.sqrt( (p.r2x[r2i] - mu*p.Rx_array/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (p.r2x[r2i] + mu*p.Rx_array/M2)**2 + Cr2)\
                        - 1.0/(np.sqrt( (p.r1x[r1i] - mu*p.Rx_array/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (p.r1x[r1i] + mu*p.Rx_array/M2)**2 + Cr2)
                        #+ 1.0/np.sqrt( (p.r1x[r1i] - p.r2x[r2i])**2 + Cr2)
    return VR

@jit( nogil=True)
def construct_VeR():
    Ve1R = np.zeros((len(p.r1x),len(p.Rx)))
    Ve2R = np.zeros((len(p.r2x),len(p.Rx)))
    for r1i in range(len(p.r1x)):
        Ve1R[r1i,:] = - 1.0/np.sqrt( (p.r1x[r1i] + mu*p.Rx_array/M2)**2 + Cr2)\
              - 1.0/(np.sqrt( (p.r1x[r1i] - mu*p.Rx_array/M1)**2 +Cr2))
        Ve2R[r1i,:] = - 1.0/np.sqrt( (p.r2x[r1i] + mu*p.Rx_array/M2)**2 + Cr2)\
              - 1.0/(np.sqrt( (p.r2x[r1i] - mu*p.Rx_array/M1)**2 +Cr2))
    return Ve1R, Ve2R
@jit( nogil=True)
def construct_Vee():
    Vee = np.zeros((len(p.r1x),len(p.r2x)))
    for r1i in range(len(p.r1x)):
        Vee[r1i,:] = 1.0/np.sqrt( (p.r1x[r1i] - p.r2x_array)**2 + Cr2)
    return Vee

#V1 = construct_cwf_V1()
#V2 = construct_cwf_V2()
#VR = construct_cwf_VR()
#Vfull = f.construct_V(np.zeros( (len(p.r1x), len(p.r2x), len(p.Rx) )))
#V_flat = np.ravel(Vfull)
Ve1R,Ve2R = construct_VeR()
Vee = construct_Vee()
Rinv_flat = np.array([1/p.Rx_array for x in range(num_trajs)])

@jit( nogil=True)
def construct_e_pos_grid():
    epos_grid = np.zeros((len(p.r1x),len(p.r2x)))
    for i in range(len(p.r1x)):
        epos_grid[i,:] = p.r1x[i] + p.r2x_array
    return -1*epos_grid
epos = construct_e_pos_grid()

mu_e = (M1+M2)/(M1+M2+1)
mu = p.mu
@jit( nopython=True)
def T1(el1):
    out = np.zeros(r1_dim+2, dtype=np.complex128)
    out[1:-1] = el1
    return (-1/(2*mu_e*p.e_spacing**2))*np.diff(out,n=2)
@jit( nopython=True)
def T2(el2):
    out = np.zeros(r2_dim+2, dtype=np.complex128)
    out[1:-1] = el2
    return (-1/(2*mu_e*p.e_spacing**2))*np.diff(out,n=2)
@jit( nopython=True)
def TR(R):
    out = np.zeros(R_dim+2, dtype=np.complex128)
    out[1:-1] = R
    return (-1/(2*mu*p.n_spacing**2))*np.diff(out,n=2)

@jit(nopython=True,nogil=True)
def V1s_calc(pos_e2,pos_R,num_trajs):
    pot_out = np.zeros((num_trajs,r1_dim))
    pos_out = np.zeros((num_trajs,r1_dim))
    for i in range(num_trajs):
        pot_out[i,:] = (1.0/(np.sqrt( (r1x_a - pos_e2[i])**2 + Cr2)) \
                        - 1.0/(np.sqrt( (r1x_a - mu*pos_R[i]/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (r1x_a + mu*pos_R[i]/M2)**2 + Cr2))
        pos_out[i,:] = r1x_a+pos_e2[i]
    return pot_out, pos_out

@jit(nopython=True,nogil=True)
def V2s_calc(pos_e1,pos_R, num_trajs):
    pot_out = np.zeros((num_trajs,r2_dim))
    pos_out = np.zeros((num_trajs,r2_dim))
    for i in range(num_trajs):
        pot_out[i,:] = (1.0/(np.sqrt( (r2x_a - pos_e1[i])**2 + Cr2)) \
                        - 1.0/(np.sqrt( (r2x_a - mu*pos_R[i]/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (r2x_a + mu*pos_R[i]/M2)**2 + Cr2))
        pos_out[i,:] = r2x_a+pos_e1[i]
    return pot_out,pos_out 

@jit(nopython=True,nogil=True)
def VRs_calc(pos_e1,pos_e2,num_trajs):
    out = np.zeros((num_trajs,R_dim))
    for i in range(num_trajs):
        out[i,:] = (1.0/Rx_a- 1.0/(np.sqrt( (pos_e2[i] - mu*Rx_a/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (pos_e2[i] + mu*Rx_a/M2)**2 + Cr2)\
                        - 1.0/(np.sqrt( (pos_e1[i] - mu*Rx_a/M1)**2 +Cr2))\
                        - 1.0/np.sqrt( (pos_e1[i] + mu*Rx_a/M2)**2 + Cr2))
    return out 
    

#}}}
#{{{ Hamiltonian
#The Hamiltonian operators defined here are actually the time_partial_derivative operators
#i.e. they are -1j*H 
@jit(parallel=True,nogil=True)
def c_Ht(psi_e1,psi_e2, psi_R, pos_e1, pos_e2,pos_R, E_t, V1s, V2s, VRs,num_trajs):
    dt_psi_R = np.zeros((num_trajs, R_dim),dtype=np.complex128) 
    dt_psi_e1 = np.zeros((num_trajs, r1_dim),dtype=np.complex128) 
    dt_psi_e2 = np.zeros((num_trajs, r2_dim),dtype=np.complex128) 
    #If there is a mass imbalence between the atoms there is an additional term
    for i in prange(num_trajs):
        dt_psi_e1[i] = T1(psi_e1[i]) + V1s[i]*psi_e1[i] \
                - E_t*(r1x_a+pos_e2[i])*psi_e1[i]
        dt_psi_e2[i] = T2(psi_e2[i]) + V2s[i]*psi_e2[i] \
                - E_t*(r2x_a+pos_e1[i])*psi_e2[i]
        dt_psi_R[i] = TR(psi_R[i]) + VRs[i]*psi_R[i]
    return -1.0j*dt_psi_e1, -1.0j*dt_psi_e2, -1.0j*dt_psi_R

@jit( parallel=True,nogil=True)
def c_H(psi_e1,psi_e2, psi_R, V1s, V2s, VRs, num_trajs):
    dt_psi_R = np.zeros((num_trajs, R_dim),dtype=np.complex128) 
    dt_psi_e1 = np.zeros((num_trajs, r1_dim),dtype=np.complex128) 
    dt_psi_e2 = np.zeros((num_trajs, r2_dim),dtype=np.complex128) 
    #If there is a mass imbalence between the atoms there is an additional term
    for i in prange(num_trajs):
        dt_psi_e1[i] = T1(psi_e1[i]) + V1s[i]*psi_e1[i] 
        dt_psi_e2[i] = T2(psi_e2[i]) + V2s[i]*psi_e2[i] 
        dt_psi_R[i] = TR(psi_R[i]) + VRs[i]*psi_R[i]
    return -1.0j*dt_psi_e1, -1.0j*dt_psi_e2, -1.0j*dt_psi_R
 

@jit( nogil=True)
def conditional_H_2e(two_e_wfs, psi_R, e1_mesh, e2_mesh,R_mesh):
    dt_psi_R = np.zeros((num_trajs, R_dim)) +0j
    dt_psi_2e = np.zeros((num_trajs, r1_dim, r2_dim)) +0j
    #If there is a mass imbalence between the atoms there is an additional term
    for i in range(num_trajs):
        dt_psi_2e[i] = two_e_T(two_e_wfs[i,:,:]) + V2e[:,:,R_mesh[i]]*two_e_wfs[i,:,:]
        dt_psi_R[i] = f.T_n.dot(psi_R[i]) + VR[e1_mesh[i],e2_mesh[i],:]*psi_R[i]
    return -1.0j*dt_psi_2e, -1.0j*dt_psi_R

@jit( nogil=True)
def conditional_Ht_2e(two_e_wfs, psi_R, e1_mesh, e2_mesh,R_mesh, E_t):
    dt_psi_R = np.zeros((num_trajs, len(Rx_a))) +0j
    dt_psi_2e = np.zeros((num_trajs, len(r1x_a), len(r2x_a))) +0j
    #If there is a mass imbalence between the atoms there is an additional term
    if(M1 == M2):
        for i in range(num_trajs):
            dt_psi_2e[i] = two_e_T(two_e_wfs[i,:,:]) + V2e[:,:,R_mesh[i]]*two_e_wfs[i,:,:] \
                    + E_t*(epos)*two_e_wfs[i,:,:]
            dt_psi_R[i] = f.T_n.dot(psi_R[i]) + VR[e1_mesh[i],e2_mesh[i],:]*psi_R[i]
    else:
        lamb = (M2 - M1)/(M1+M2)
        for i in range(num_trajs):
            dt_psi_2e[i] = two_e_T(two_e_wfs[i,:,:]) + V2e[:,:,R_mesh[i]]*two_e_wfs[i,:,:] \
                    + E_t*(epos)*two_e_wfs[i,:,:]
            dt_psi_R[i] = f.T_n.dot(psi_R[i]) + VR[e1_mesh[i],e2_mesh[i],:]*psi_R[i] \
                    +E_t*lamb*(Rx_a)*psi_R[i]
    return -1.0j*dt_psi_2e, -1.0j*dt_psi_R

#}}}
#{{{RK4 integrators
#{{{Hermitian RK4
@jit( nogil=True)
def RK4(e1_wfs, e2_wfs, R_wfs, e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R):
    #find first derivative of the wavefunctions
    K1 = c_H(e1_wfs, e2_wfs, R_wfs, \
                                    e1_mesh, e2_mesh, R_mesh)
    #find the first derivative of the trajectory positions
    traj_e1_vel_K1, traj_e2_vel_K1 = calc_e_vel(e1_wfs,e2_wfs,e1_mesh,e2_mesh)
    traj_R_vel_K1 = calc_n_vel(R_wfs,R_mesh)

    #find the first derivative of the wavefunctions at the mid point time step,
    #   via projecting the wf according to the derivative at the original position
    K2 = c_H(e1_wfs + p.dt*K1[0]/2, e2_wfs + p.dt*K1[1]/2, \
                       R_wfs + p.dt*K1[2]/2, e1_mesh, e2_mesh,R_mesh)

    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    traj_e1_vel_K2, traj_e2_vel_K2 = calc_e_vel(e1_wfs + p.dt*K1[0]/2, e2_wfs + p.dt*K1[1]/2,\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K2 = calc_n_vel(R_wfs + p.dt*K1[2]/2,R_mesh)

    #Do the third term correction for the wavefunctions    
    K3 = c_H(e1_wfs + p.dt*K2[0]/2, e2_wfs + p.dt*K2[1]/2, \
                       R_wfs + p.dt*K2[2]/2, e1_mesh, e2_mesh,R_mesh)
    #And now the trajectory positions
    traj_e1_vel_K3, traj_e2_vel_K3 = calc_e_vel(e1_wfs + p.dt*K2[0]/2, e2_wfs + p.dt*K2[1]/2,\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K3 = calc_n_vel(R_wfs + p.dt*K2[2]/2,R_mesh)

    #finally get the derivative information at the next timestep
    K4 = c_H(e1_wfs + p.dt*K3[0], e2_wfs + p.dt*K3[1], \
                       R_wfs + p.dt*K3[2], e1_mesh, e2_mesh,R_mesh)
    #And now the trajectory positions
    traj_e1_vel_K4, traj_e2_vel_K4 = calc_e_vel(e1_wfs + p.dt*K3[0], e2_wfs + p.dt*K3[1],\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K4 = calc_n_vel(R_wfs + p.dt*K3[2],R_mesh)

    #now update the global variables of trajectory positions
    pos_e1 = pos_e1 + (p.dt/6)*(traj_e1_vel_K1 + 2*traj_e1_vel_K2 +2*traj_e1_vel_K3 \
                        + traj_e1_vel_K4) 
    pos_e2 = pos_e2 + (p.dt/6)*(traj_e2_vel_K1 + 2*traj_e2_vel_K2 +2*traj_e2_vel_K3 \
                        + traj_e2_vel_K4) 
    pos_R = pos_R + (p.dt/6)*(traj_R_vel_K1 + 2*traj_R_vel_K2 +2*traj_R_vel_K3 \
                        + traj_R_vel_K4) 

    return e1_wfs + (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), \
           e2_wfs + (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]), \
            R_wfs + (p.dt/6)*(K1[2] + 2*K2[2] + 2*K3[2] + K4[2]), pos_e1, pos_e2, pos_R

#Same as the above but time dependent
@jit( nogil=True)
def T_dep_RK4(e1_wfs, e2_wfs, R_wfs, e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R,\
              curr_time_index, E_form):
    E_t = E_form[2*curr_time_index]
    E_t_half = E_form[2*curr_time_index+1]
    E_t_advanced = E_form[2*curr_time_index + 2] 
    K1 = c_Ht(e1_wfs, e2_wfs, R_wfs, \
                                    e1_mesh, e2_mesh, R_mesh, E_t)
    traj_e1_vel_K1, traj_e2_vel_K1 = calc_e_vel(e1_wfs,e2_wfs,e1_mesh,e2_mesh)
    traj_R_vel_K1 = calc_n_vel(R_wfs,R_mesh)
    K2 = c_Ht(e1_wfs + p.dt*K1[0]/2, e2_wfs + p.dt*K1[1]/2, \
                        R_wfs + p.dt*K1[2]/2, e1_mesh, e2_mesh, R_mesh, E_t_half)
    traj_e1_vel_K2, traj_e2_vel_K2 = calc_e_vel(e1_wfs + p.dt*K1[0]/2, e2_wfs + p.dt*K1[1]/2,\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K2 = calc_n_vel(R_wfs + p.dt*K1[2]/2,R_mesh)
    K3 = c_Ht(e1_wfs + p.dt*K2[0]/2, e2_wfs + p.dt*K2[1]/2, \
                        R_wfs + p.dt*K2[2]/2, e1_mesh, e2_mesh, R_mesh, E_t_half)
    traj_e1_vel_K3, traj_e2_vel_K3 = calc_e_vel(e1_wfs + p.dt*K2[0]/2, e2_wfs + p.dt*K2[1]/2,\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K3 = calc_n_vel(R_wfs + p.dt*K2[2]/2,R_mesh)
    K4 = c_Ht(e1_wfs + p.dt*K3[0], e2_wfs + p.dt*K3[1], 
                        R_wfs + p.dt*K3[2], e1_mesh, e2_mesh, R_mesh, E_t_advanced)
    traj_e1_vel_K4, traj_e2_vel_K4 = calc_e_vel(e1_wfs + p.dt*K3[0], e2_wfs + p.dt*K3[1],\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K4 = calc_n_vel(R_wfs + p.dt*K3[2],R_mesh)
    pos_e1 = pos_e1 + (p.dt/6)*(traj_e1_vel_K1 + 2*traj_e1_vel_K2 +2*traj_e1_vel_K3 \
                        + traj_e1_vel_K4) 
    pos_e2 = pos_e2 + (p.dt/6)*(traj_e2_vel_K1 + 2*traj_e2_vel_K2 +2*traj_e2_vel_K3 \
                        + traj_e2_vel_K4) 
    pos_R = pos_R + (p.dt/6)*(traj_R_vel_K1 + 2*traj_R_vel_K2 +2*traj_R_vel_K3 \
                        + traj_R_vel_K4) 
    return e1_wfs + (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), \
           e2_wfs + (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]), \
            R_wfs + (p.dt/6)*(K1[2] + 2*K2[2] + 2*K3[2] + K4[2]), pos_e1, pos_e2, pos_R

#Write in simulataneous RK4 evolution of trajectory positions
@jit( nogil=True)
def RK4_2e(two_e_wfs, R_wfs, e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R):
    #find first derivative of the wavefunctions
    K1 = conditional_H_2e(two_e_wfs, R_wfs, \
                                    e1_mesh, e2_mesh, R_mesh)
    #find the first derivative of the trajectory positions
    traj_2e_vel_K1 = calc_2e_vel(two_e_wfs,e1_mesh,e2_mesh)
    traj_R_vel_K1 = calc_n_vel(R_wfs,R_mesh)

    #find the first derivative of the wavefunctions at the mid point time step,
    #   via projecting the wf according to the derivative at the original position
    K2 = conditional_H_2e(two_e_wfs + p.dt*K1[0]/2, \
                       R_wfs + p.dt*K1[1]/2, e1_mesh, e2_mesh,R_mesh)

    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    traj_2e_vel_K2 =calc_2e_vel(two_e_wfs + p.dt*K1[0]/2,\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K2 = calc_n_vel(R_wfs + p.dt*K1[1]/2,R_mesh)

    #Do the third term correction for the wavefunctions    
    K3 = conditional_H_2e(two_e_wfs + p.dt*K2[0]/2, \
                       R_wfs + p.dt*K2[1]/2, e1_mesh, e2_mesh,R_mesh)
    #And now the trajectory positions
    traj_2e_vel_K3= calc_2e_vel(two_e_wfs + p.dt*K2[0]/2,\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K3 = calc_n_vel(R_wfs + p.dt*K2[1]/2,R_mesh)

    #finally get the derivative information at the next timestep
    K4 = conditional_H_2e(two_e_wfs + p.dt*K3[0], \
                       R_wfs + p.dt*K3[1], e1_mesh, e2_mesh,R_mesh)
    #And now the trajectory positions
    traj_2e_vel_K4 = calc_2e_vel(two_e_wfs + p.dt*K3[0],\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K4 = calc_n_vel(R_wfs + p.dt*K3[1],R_mesh)

    #now update the global variables of trajectory positions
    pos_e1 = pos_e1 + (p.dt/6)*(traj_2e_vel_K1[0] + 2*traj_2e_vel_K2[0] +2*traj_2e_vel_K3[0] \
                        + traj_2e_vel_K4[0]) 
    pos_e2 = pos_e2 + (p.dt/6)*(traj_2e_vel_K1[1] + 2*traj_2e_vel_K2[1] +2*traj_2e_vel_K3[1] \
                        + traj_2e_vel_K4[1]) 
    pos_R = pos_R + (p.dt/6)*(traj_R_vel_K1 + 2*traj_R_vel_K2 +2*traj_R_vel_K3 \
                        + traj_R_vel_K4) 

    return two_e_wfs + (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), \
            R_wfs + (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]), pos_e1, pos_e2, pos_R

@jit( nogil=True)
def T_dep_RK4_2e(two_e_wfs, R_wfs, e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R,\
                 curr_time_index, E_form):
    E_t = E_form[2*curr_time_index]
    E_t_half = E_form[2*curr_time_index+1]
    E_t_advanced = E_form[2*curr_time_index + 2] 
    #find first derivative of the wavefunctions
    K1 = conditional_Ht_2e(two_e_wfs, R_wfs, \
                                    e1_mesh, e2_mesh, R_mesh, E_t)
    #find the first derivative of the trajectory positions
    traj_2e_vel_K1 = calc_2e_vel(two_e_wfs,e1_mesh,e2_mesh)
    traj_R_vel_K1 = calc_n_vel(R_wfs,R_mesh)

    #find the first derivative of the wavefunctions at the mid point time step,
    #   via projecting the wf according to the derivative at the original position
    K2 = conditional_Ht_2e(two_e_wfs + p.dt*K1[0]/2, \
                       R_wfs + p.dt*K1[1]/2, e1_mesh, e2_mesh,R_mesh, E_t_half)

    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    traj_2e_vel_K2 =calc_2e_vel(two_e_wfs + p.dt*K1[0]/2,\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K2 = calc_n_vel(R_wfs + p.dt*K1[1]/2,R_mesh)

    #Do the third term correction for the wavefunctions    
    K3 = conditional_Ht_2e(two_e_wfs + p.dt*K2[0]/2, \
                       R_wfs + p.dt*K2[1]/2, e1_mesh, e2_mesh,R_mesh, E_t_half)
    #And now the trajectory positions
    traj_2e_vel_K3= calc_2e_vel(two_e_wfs + p.dt*K2[0]/2,\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K3 = calc_n_vel(R_wfs + p.dt*K2[1]/2,R_mesh)

    #finally get the derivative information at the next timestep
    K4 = conditional_Ht_2e(two_e_wfs + p.dt*K3[0], \
                       R_wfs + p.dt*K3[1], e1_mesh, e2_mesh,R_mesh, E_t_advanced)
    #And now the trajectory positions
    traj_2e_vel_K4 = calc_2e_vel(two_e_wfs + p.dt*K3[0],\
                                    e1_mesh,e2_mesh)
    traj_R_vel_K4 = calc_n_vel(R_wfs + p.dt*K3[1],R_mesh)

    #now update the global variables of trajectory positions
    pos_e1 = pos_e1 + (p.dt/6)*(traj_2e_vel_K1[0] + 2*traj_2e_vel_K2[0] +2*traj_2e_vel_K3[0] \
                        + traj_2e_vel_K4[0]) 
    pos_e2 = pos_e2 + (p.dt/6)*(traj_2e_vel_K1[1] + 2*traj_2e_vel_K2[1] +2*traj_2e_vel_K3[1] \
                        + traj_2e_vel_K4[1]) 
    pos_R = pos_R + (p.dt/6)*(traj_R_vel_K1 + 2*traj_R_vel_K2 +2*traj_R_vel_K3 \
                        + traj_R_vel_K4) 

    return two_e_wfs + (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), \
            R_wfs + (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]), pos_e1, pos_e2, pos_R
#}}}
#{{{ 2e ICWF RK4
#Need to remove Ets
@jit( nogil=True)
def RK4_2e_ICWF(two_e_wfs, R_wfs, e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R, C):
    #find first derivative of the wavefunctions
    K1 = conditional_H_2e(two_e_wfs, R_wfs, \
                                    e1_mesh, e2_mesh, R_mesh)

    #find the first derivative of the trajectory positions
    e1_K1, e2_K1, R_K1 = ICWF_vel(C,two_e_wfs, R_wfs, e1_mesh, e2_mesh, R_mesh)
    C_K1 = C_dot(C,two_e_wfs,R_wfs,e1_mesh,e2_mesh,R_mesh)

    #find the first derivative of the wavefunctions at the mid point time step,
    #   via projecting the wf according to the derivative at the original position
    K2 = conditional_H_2e(two_e_wfs + p.dt*K1[0]/2, \
                       R_wfs + p.dt*K1[1]/2, e1_mesh, e2_mesh,R_mesh)

    

    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    e1_K2, e2_K2, R_K2= ICWF_vel(C,two_e_wfs+ p.dt*K1[0]/2, R_wfs+ p.dt*K1[1]/2, e1_mesh, e2_mesh, R_mesh)
    C_K2 = C_dot(C,two_e_wfs + p.dt*K1[0]/2,R_wfs + p.dt*K1[1]/2,e1_mesh,e2_mesh,R_mesh)

    #Do the third term correction for the wavefunctions    
    K3 = conditional_H_2e(two_e_wfs + p.dt*K2[0]/2, \
                       R_wfs + p.dt*K2[1]/2, e1_mesh, e2_mesh,R_mesh)
    
    #And now the trajectory positions
    e1_K3, e2_K3, R_K3= ICWF_vel(C,two_e_wfs+ p.dt*K2[0]/2, R_wfs+ p.dt*K2[1]/2, e1_mesh, e2_mesh, R_mesh)
    C_K3 = C_dot(C,two_e_wfs + p.dt*K2[0]/2,R_wfs + p.dt*K2[1]/2,e1_mesh,e2_mesh,R_mesh)

    #finally get the derivative information at the next timestep
    K4 = conditional_H_2e(two_e_wfs + p.dt*K3[0], \
                       R_wfs + p.dt*K3[1], e1_mesh, e2_mesh,R_mesh)
    #And now the trajectory positions
    e1_K4, e2_K4, R_K4 = ICWF_vel(C,two_e_wfs+ p.dt*K3[0], R_wfs+ p.dt*K3[1], e1_mesh, e2_mesh, R_mesh)
    C_K4 = C_dot(C,two_e_wfs + p.dt*K3[0],R_wfs + p.dt*K3[1],e1_mesh,e2_mesh,R_mesh)

    #now update the global variables of trajectory positions
    pos_e1 = pos_e1 + (p.dt/6)*(e1_K1 + 2*e1_K2 +2*e1_K3 \
                        + e1_K4) 
    pos_e2 = pos_e2 + (p.dt/6)*(e2_K1 + 2*e2_K2 +2*e2_K3 \
                        + e2_K4) 
    pos_R = pos_R + (p.dt/6)*(R_K1 + 2*R_K2 +2*R_K3 \
                        + R_K4)
    C += (p.dt/6)*(C_K1 + 2*C_K2 + 2*C_K3 + C_K4) 

    return two_e_wfs + (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), \
            R_wfs + (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]), pos_e1, pos_e2, pos_R, C

@jit( nogil=True)
def T_dep_RK4_2e_ICWF(two_e_wfs, R_wfs, e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R,\
                     curr_time_index, E_form, C):
    E_t = E_form[2*curr_time_index]
    E_t_half = E_form[2*curr_time_index+1]
    E_t_advanced = E_form[2*curr_time_index + 2] 

    #find first derivative of the wavefunctions
    K1 = conditional_Ht_2e(two_e_wfs, R_wfs, \
                                    e1_mesh, e2_mesh, R_mesh, E_t)

    #find the first derivative of the trajectory positions
    e1_K1, e2_K1, R_K1 = ICWF_vel(C,two_e_wfs, R_wfs, e1_mesh, e2_mesh, R_mesh)
    C_K1 = C_dot_t(C,two_e_wfs,R_wfs,e1_mesh,e2_mesh,R_mesh, E_t)

    #find the first derivative of the wavefunctions at the mid point time step,
    #   via projecting the wf according to the derivative at the original position
    K2 = conditional_Ht_2e(two_e_wfs + p.dt*K1[0]/2, \
                       R_wfs + p.dt*K1[1]/2, e1_mesh, e2_mesh,R_mesh, E_t_half)

    

    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    e1_K2, e2_K2, R_K2= ICWF_vel(C,two_e_wfs+ p.dt*K1[0]/2, R_wfs+ p.dt*K1[1]/2, \
                    e1_mesh, e2_mesh, R_mesh)
    C_K2 = C_dot_t(C + p.dt*C_K1/2 ,two_e_wfs + p.dt*K1[0]/2,R_wfs + p.dt*K1[1]/2,\
                    e1_mesh,e2_mesh,R_mesh, E_t_half)

    #Do the third term correction for the wavefunctions    
    K3 = conditional_Ht_2e(two_e_wfs + p.dt*K2[0]/2, \
                       R_wfs + p.dt*K2[1]/2, e1_mesh, e2_mesh,R_mesh, E_t_half)
    
    #And now the trajectory positions
    e1_K3, e2_K3, R_K3= ICWF_vel(C,two_e_wfs+ p.dt*K2[0]/2, R_wfs+ p.dt*K2[1]/2, \
                e1_mesh, e2_mesh, R_mesh)
    C_K3 = C_dot_t(C + p.dt*C_K2/2,two_e_wfs + p.dt*K2[0]/2,R_wfs + p.dt*K2[1]/2,\
                e1_mesh,e2_mesh,R_mesh, E_t_half)

    #finally get the derivative information at the next timestep
    K4 = conditional_Ht_2e(two_e_wfs + p.dt*K3[0], \
                       R_wfs + p.dt*K3[1], e1_mesh, e2_mesh,R_mesh, E_t_advanced)
    #And now the trajectory positions
    e1_K4, e2_K4, R_K4 = ICWF_vel(C,two_e_wfs+ p.dt*K3[0], R_wfs+ p.dt*K3[1], \
                    e1_mesh, e2_mesh, R_mesh)
    C_K4 = C_dot_t(C+p.dt*C_K3,two_e_wfs + p.dt*K3[0],R_wfs + p.dt*K3[1],\
                e1_mesh,e2_mesh,R_mesh, E_t_advanced)
    #now update the global variables of trajectory positions
    pos_e1 = pos_e1 + (p.dt/6)*(e1_K1 + 2*e1_K2 +2*e1_K3 \
                        + e1_K4) 
    pos_e2 = pos_e2 + (p.dt/6)*(e2_K1 + 2*e2_K2 +2*e2_K3 \
                        + e2_K4) 
    pos_R = pos_R + (p.dt/6)*(R_K1 + 2*R_K2 +2*R_K3 \
                        + R_K4)
    C += (p.dt/6)*(C_K1 + 2*C_K2 + 2*C_K3 + C_K4) 

    return two_e_wfs + (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), \
            R_wfs + (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]), pos_e1, pos_e2, pos_R, C
#}}}
#{{{ 1e ICWF RK4
@jit( nopython=True,parallel=True,nogil=True)
def super_RK4(el1,el2,R,e1_mesh,e2_mesh,R_mesh,pos_e1,pos_e2,pos_R,C):
    num_trajs = el1.shape[0]
    psi_R_K = np.zeros((4,num_trajs, R_dim),dtype=np.complex128) 
    psi_e1_K = np.zeros((4,num_trajs, r1_dim),dtype=np.complex128) 
    psi_e2_K = np.zeros((4,num_trajs, r2_dim),dtype=np.complex128)
    pos_R_K = np.zeros((4,num_trajs, R_dim),dtype=np.complex128) 
    pos_e1_K = np.zeros((4,num_trajs, r1_dim),dtype=np.complex128)  
    pos_e2_K = np.zeros((4,num_trajs, r2_dim),dtype=np.complex128)
    C_K =np.zeros((4,num_trajs),dtype=np.complex128)
 
    pos_e1_K[0],pos_e2_K[0],pos_R_K[0] = ICWF_vel(C,el1,el2,R,e1_mesh,e2_mesh,R_mesh)
    for i in prange(num_trajs):
        psi_e1_K[0,i] = -1j*(T1(psi_e1[i]) + V1[:,e2_mesh[i],R_mesh[i]]*psi_e1[i] \
                + E_t*epos[:,e2_mesh[i]]*psi_e1[i])
        psi_e2_K[0,i] = -1j*(T2(psi_e2[i]) + V2[e1_mesh[i],:,R_mesh[i]]*psi_e2[i] \
                + E_t*epos[e1_mesh[i],:]*psi_e2[i])
        psi_R_K[0,i] = -1j*(TR(psi_R[i]) + VR[e1_mesh[i],e2_mesh[i],:]*psi_R[i])
 
        C_K[0,i] = C_dot_1e_t(C,el1,el2,R,V1[:,e2_mesh,R_mesh],V2[e1_mesh,:,R_mesh],VR[e1_mesh,e2_mesh,:])
    e1_buff = psi_e1 + p.dt*psi_e1_K[0]/2
    e2_buff = psi_e1 + p.dt*psi_e2_K[0]/2
    R_buff = psi_e1 + p.dt*psi_R_K[0]/2
    C_buff = C + p.dt*C_K[0]/2
    pos_e1_K[1],pos_e2_K[1],pos_R_K[1] = ICWF_1e_vel(C_buff,e1_buff,e2_buff,R_buff,e1_mesh,e2_mesh,R_mesh, num_trajs)
    for i in prange(num_trajs):
        psi_e1_K[1,i] = -1j*(T1(e1_buff[i]) + V1[:,e2_mesh[i],R_mesh[i]]*e1_buff[i] \
                + E_t*epos[:,e2_mesh[i]]*e1_buff[i])
        psi_e2_K[1,i] = -1j*(T2(e2_buff[i]) + V2[e1_mesh[i],:,R_mesh[i]]*e2_buff[i] \
                + E_t*epos[e1_mesh[i],:]*e2_buff[i])
        psi_R_K[1,i] = -1j*(TR(R_buff[i]) + VR[e1_mesh[i],e2_mesh[i],:]*R_buff[i])
 
        C_K[1,i] = C_dot_1e_t(C_buff,e1_buff,e2_buff,R_buff,V1[:,e2_mesh,R_mesh],V2[e1_mesh,:,R_mesh],VR[e1_mesh,e2_mesh,:])
    e1_buff = psi_e1 + p.dt*psi_e1_K[1]/2
    e2_buff = psi_e1 + p.dt*psi_e2_K[1]/2
    R_buff = psi_e1 + p.dt*psi_R_K[1]/2
    C_buff = C + p.dt*C_K[1]/2
    pos_e1_K[2],pos_e2_K[2],pos_R_K[2] = ICWF_1e_vel(C_buff,e1_buff,e2_buff,R_buff,e1_mesh,e2_mesh,R_mesh, num_trajs)
    for i in prange(num_trajs):
        psi_e1_K[2,i] = -1j*(T1(e1_buff[i]) + V1[:,e2_mesh[i],R_mesh[i]]*e1_buff[i] \
                + E_t*epos[:,e2_mesh[i]]*e1_buff[i])
        psi_e2_K[2,i] = -1j*(T2(e2_buff[i]) + V2[e1_mesh[i],:,R_mesh[i]]*e2_buff[i] \
                + E_t*epos[e1_mesh[i],:]*e2_buff[i])
        psi_R_K[2,i] = -1j*(TR(R_buff[i]) + VR[e1_mesh[i],e2_mesh[i],:]*R_buff[i])
 
        C_K[2,i] = C_dot_1e_t(C_buff,e1_buff,e2_buff,R_buff,V1[:,e2_mesh,R_mesh],V2[e1_mesh,:,R_mesh],VR[e1_mesh,e2_mesh,:])
    e1_buff = psi_e1 + p.dt*psi_e1_K[2]
    e2_buff = psi_e1 + p.dt*psi_e2_K[2]
    R_buff = psi_e1 + p.dt*psi_R_K[2]
    C_buff = C + p.dt*C_K[2]
    pos_e1_K[3],pos_e2_K[3],pos_R_K[3] = ICWF_1e_vel(C_buff,e1_buff,e2_buff,R_buff,e1_mesh,e2_mesh,R_mesh, num_trajs)
    for i in prange(num_trajs):
        psi_e1_K[3,i] = -1j*(T1(e1_buff[i]) + V1[:,e2_mesh[i],R_mesh[i]]*e1_buff[i] \
                + E_t*epos[:,e2_mesh[i]]*e1_buff[i])
        psi_e2_K[3,i] = -1j*(T2(e2_buff[i]) + V2[e1_mesh[i],:,R_mesh[i]]*e2_buff[i] \
                + E_t*epos[e1_mesh[i],:]*e2_buff[i])
        psi_R_K[3,i] = -1j*(TR(R_buff[i]) + VR[e1_mesh[i],e2_mesh[i],:]*R_buff[i])
 
        C_K[3,i] = C_dot_1e_t(C_buff,e1_buff,e2_buff,R_buff,V1[:,e2_mesh,R_mesh],V2[e1_mesh,:,R_mesh],VR[e1_mesh,e2_mesh,:])

    pos_e1 = pos_e1 + (p.dt/6)*(pos_e1_K[0] + 2*pos_e1_K[1] +2*pos_e1_K[2] \
                        + pos_e1_K[3]) 
    pos_e2 = pos_e2 + (p.dt/6)*(pos_e2_K[0] + 2*pos_e2_K[1] +2*pos_e2_K[2] \
                        + pos_e2_K[3]) 
    pos_R = pos_R + (p.dt/6)*(pos_R_K[0] + 2*pos_R_K[1] +2*pos_R_K[2] \
                        + pos_R_K[3])
        
    return el1 + (p.dt/6)*(psi_e1_K[0] + 2*psi_e1_K[1] + 2*psi_e1_K[2] + psi_e1_K[3]), \
            el2 + (p.dt/6)*(psi_e2_K[0] + 2*psi_e2_K[1] + 2*psi_e2_K[2] + psi_e2_K[3]), \
            R + (p.dt/6)*(psi_R_K[0] + 2*psi_R_K[1] + 2*psi_R_K[2] + psi_R_K[3]), pos_e1, pos_e2, pos_R, \
            C + (p.dt/6)*(C_K[0] + 2*C_K[1] + 2*C_K[2] + C_K[3])


@jit( nogil=True)
def RK4_1e_ICWF(el1,el2,R,e1_mesh,e2_mesh,R_mesh,pos_e1,pos_e2,pos_R,C):
    e1_mesh_buff = np.zeros(num_trajs)
    e2_mesh_buff = np.zeros(num_trajs) 
    R_mesh_buff = np.zeros(num_trajs) 

    pos_e1_buff = np.zeros(num_trajs)
    pos_e2_buff = np.zeros(num_trajs)
    pos_R_buff = np.zeros(num_trajs)

    C_buff = np.zeros(num_trajs,dtype=np.complex128)

    el1_buff = np.zeros((num_trajs, r1_dim),dtype=np.complex128)
    el2_buff = np.zeros((num_trajs, r2_dim),dtype=np.complex128)
    R_buff = np.zeros((num_trajs, R_dim),dtype=np.complex128)

    V1s,epos1 = V1s_calc(pos_e2,pos_R, num_trajs)
    V2s,epos2 = V2s_calc(pos_e1,pos_R, num_trajs)
    VRs = VRs_calc(pos_e1,pos_e2, num_trajs)

    #find first derivative of the wavefunctions
    K1 = c_H(el1, el2, R, V1s, V2s, VRs, num_trajs)


    #find the first derivative of the trajectory positions
    e1_K1, e2_K1, R_K1 = ICWF_1e_vel(C,el1,el2, R, e1_mesh, e2_mesh, R_mesh, num_trajs)

    C_K1 = C_dot_1e(C,el1,el2,R,V1s, V2s, VRs)


    pos_e1_buff = pos_e1 + p.dt*e1_K1/2
    pos_e2_buff = pos_e2 + p.dt*e2_K1/2
    pos_R_buff = pos_R + p.dt*R_K1/2

    e1_mesh_buff  = f.find_index(r1x_a,pos_e1_buff)
    e2_mesh_buff  = f.find_index(r2x_a,pos_e2_buff)
    R_mesh_buff  = f.find_index(Rx_a, pos_R_buff)

    el1_buff = el1 + p.dt*K1[0]/2
    el2_buff = el2 + p.dt*K1[1]/2
    R_buff = R + p.dt*K1[2]/2

    C_buff = C+p.dt*C_K1/2

    V1s,epos1 = V1s_calc(pos_e2_buff,pos_R_buff, num_trajs)
    V2s,epos2 = V2s_calc(pos_e1_buff,pos_R_buff, num_trajs)
    VRs = VRs_calc(pos_e1_buff,pos_e2_buff, num_trajs)

    #find the first derivative of the wavefunctions at the mid point time step,
    #   via projecting the wf according to the derivative at the original position
    K2 = c_H(el1_buff, el2_buff,R_buff, V1s, V2s, VRs, num_trajs)
    
    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    e1_K2, e2_K2, R_K2= ICWF_1e_vel(C_buff,el1_buff, el2_buff, R_buff, e1_mesh_buff, e2_mesh_buff, R_mesh_buff, num_trajs)
    C_K2 = C_dot_1e(C_buff, el1_buff,el2_buff,R_buff,V1s,V2s,VRs)

    pos_e1_buff = pos_e1 + p.dt*e1_K2/2
    pos_e2_buff = pos_e2 + p.dt*e2_K2/2
    pos_R_buff = pos_R + p.dt*R_K2/2

    e1_mesh_buff  = f.find_index(r1x_a,pos_e1_buff)
    e2_mesh_buff  = f.find_index(r2x_a,pos_e2_buff)
    R_mesh_buff  = f.find_index(Rx_a, pos_R_buff)

    el1_buff = el1 + p.dt*K2[0]/2
    el2_buff = el2 + p.dt*K2[1]/2
    R_buff = R + p.dt*K2[2]/2

    C_buff = C+p.dt*C_K2/2

    V1s,epos1 = V1s_calc(pos_e2_buff,pos_R_buff, num_trajs)
    V2s,epos2 = V2s_calc(pos_e1_buff,pos_R_buff, num_trajs)
    VRs = VRs_calc(pos_e1_buff,pos_e2_buff, num_trajs)

    K3 = c_H(el1_buff, el2_buff,R_buff, V1s, V2s, VRs, num_trajs)
    
    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    e1_K3, e2_K3, R_K3= ICWF_1e_vel(C_buff,el1_buff, el2_buff, R_buff,  e1_mesh_buff, e2_mesh_buff, R_mesh_buff, num_trajs)
    C_K3 = C_dot_1e(C_buff, el1_buff,el2_buff,R_buff,V1s,V2s,VRs)

    pos_e1_buff = pos_e1 + p.dt*e1_K3
    pos_e2_buff = pos_e2 + p.dt*e2_K3
    pos_R_buff = pos_R + p.dt*R_K3

    e1_mesh_buff  = f.find_index(r1x_a,pos_e1_buff)
    e2_mesh_buff  = f.find_index(r2x_a,pos_e2_buff)
    R_mesh_buff  = f.find_index(Rx_a, pos_R_buff)

    el1_buff = el1 + p.dt*K3[0]
    el2_buff = el2 + p.dt*K3[1]
    R_buff = R + p.dt*K3[2]

    C_buff = C+p.dt*C_K3

    V1s,epos1 = V1s_calc(pos_e2_buff,pos_R_buff, num_trajs)
    V2s,epos2 = V2s_calc(pos_e1_buff,pos_R_buff, num_trajs)
    VRs = VRs_calc(pos_e1_buff,pos_e2_buff, num_trajs)

    K4 = c_H(el1_buff, el2_buff,R_buff, V1s, V2s, VRs, num_trajs)
    
    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    e1_K4, e2_K4, R_K4= ICWF_1e_vel(C_buff,el1_buff, el2_buff, R_buff,  e1_mesh_buff, e2_mesh_buff, R_mesh_buff, num_trajs)
    C_K4 = C_dot_1e(C_buff, el1_buff,el2_buff,R_buff,V1s,V2s,VRs)

    #now update the global variables of trajectory positions
    pos_e1 = pos_e1 + (p.dt/6)*(e1_K1 + 2*e1_K2 +2*e1_K3 \
                        + e1_K4) 
    pos_e2 = pos_e2 + (p.dt/6)*(e2_K1 + 2*e2_K2 +2*e2_K3 \
                        + e2_K4) 
    pos_R = pos_R + (p.dt/6)*(R_K1 + 2*R_K2 +2*R_K3 \
                        + R_K4)
    print('pos_e1[0] = ', pos_e1[0])
    print('vel_e1[0] = ', (e1_K1[0] + 2*e1_K2[0] + 2*e1_K3[0] + e1_K4[0]))
    print('pos_e2[0] = ', pos_e2[0])
    print('vel_e2[0] = ', (e2_K1[0] + 2*e2_K2[0] + 2*e2_K3[0] + e2_K4[0]))
    print('pos_R[0] = ', pos_R[0])
    print('vel_R[0] = ', (R_K1[0] + 2*R_K2[0] + 2*R_K3[0] + R_K4[0]))
    print('\n')
    
    return el1 + (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), \
            el2 + (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]), \
            R + (p.dt/6)*(K1[2] + 2*K2[2] + 2*K3[2] + K4[2]), pos_e1, pos_e2, pos_R, \
            C + (p.dt/6)*(C_K1 + 2*C_K2 + 2*C_K3 + C_K4)

@jit( nogil=True)
def T_dep_RK4_1e_ICWF(el1,el2,R,e1_mesh,e2_mesh,R_mesh,pos_e1,pos_e2,pos_R, \
                    curr_time_index, E_form,C):
    E_t = E_form[2*curr_time_index]
    E_t_half = E_form[2*curr_time_index+1]
    E_t_adv = E_form[2*curr_time_index + 2]

    e1_mesh_buff  = np.zeros(num_trajs) 
    e2_mesh_buff  = np.zeros(num_trajs)
    R_mesh_buff  = np.zeros(num_trajs)

    pos_e1_buff = np.zeros(num_trajs)
    pos_e2_buff = np.zeros(num_trajs)
    pos_R_buff = np.zeros(num_trajs)

    C_buff = np.zeros(num_trajs,dtype=np.complex128)

    el1_buff = np.zeros((num_trajs, r1_dim),dtype=np.complex128)
    el2_buff = np.zeros((num_trajs, r2_dim),dtype=np.complex128)
    R_buff = np.zeros((num_trajs, R_dim),dtype=np.complex128)

    V1s,epos1 = V1s_calc(pos_e2,pos_R, num_trajs)
    V2s,epos2 = V2s_calc(pos_e1,pos_R, num_trajs)
    VRs = VRs_calc(pos_e1,pos_e2, num_trajs)

    #find first derivative of the wavefunctions
    K1 = c_Ht(el1, el2, R, pos_e1, pos_e2, pos_R, E_t, V1s, V2s, VRs, num_trajs)
    #find the first derivative of the trajectory positions
    e1_K1, e2_K1, R_K1 = ICWF_1e_vel(C,el1,el2, R, e1_mesh, e2_mesh, R_mesh, num_trajs)
    #find the ansatz constants
    C_K1,M1 = C_dot_1e_t(C,el1,el2,R,V1s,V2s, VRs, E_t,epos1,epos2, epos)

    pos_e1_buff = pos_e1 + p.dt*e1_K1/2
    pos_e2_buff = pos_e2 + p.dt*e2_K1/2
    pos_R_buff = pos_R + p.dt*R_K1/2

    e1_mesh_buff  = f.find_index(r1x_a,pos_e1_buff)
    e2_mesh_buff  = f.find_index(r2x_a,pos_e2_buff)
    R_mesh_buff  = f.find_index(Rx_a, pos_R_buff)

    el1_buff = el1 + p.dt*K1[0]/2
    el2_buff = el2 + p.dt*K1[1]/2
    R_buff = R + p.dt*K1[2]/2

    C_buff = C+p.dt*C_K1/2

    V1s,epos1 = V1s_calc(pos_e2_buff,pos_R_buff, num_trajs)
    V2s,epos2 = V2s_calc(pos_e1_buff,pos_R_buff, num_trajs)
    VRs = VRs_calc(pos_e1_buff,pos_e2_buff, num_trajs)

    K2 = c_Ht(el1_buff, el2_buff, R_buff, \
              pos_e1_buff , pos_e2_buff ,pos_R_buff ,E_t_half, V1s, V2s, VRs, num_trajs)
    
    #Do the same for the trajectory positions, evaluating the velocities at this midpoint
    e1_K2, e2_K2, R_K2= ICWF_1e_vel(C_buff,el1_buff, el2_buff, \
                            R_buff, e1_mesh_buff , e2_mesh_buff , R_mesh_buff, num_trajs)
    C_K2,M2 = C_dot_1e_t(C_buff,el1_buff, el2_buff, R_buff,\
                    V1s,V2s,VRs, E_t_half,epos1,epos2, epos)
        

    pos_e1_buff = pos_e1 + p.dt*e1_K2/2
    pos_e2_buff = pos_e2 + p.dt*e2_K2/2
    pos_R_buff = pos_R + p.dt*R_K2/2

    e1_mesh_buff  = f.find_index(r1x_a,pos_e1_buff)
    e2_mesh_buff  = f.find_index(r2x_a,pos_e2_buff)
    R_mesh_buff  = f.find_index(Rx_a, pos_R_buff)

    el1_buff = el1 + p.dt*K2[0]/2
    el2_buff = el2 + p.dt*K2[1]/2
    R_buff = R + p.dt*K2[2]/2

    C_buff = C+p.dt*C_K2/2

    V1s,epos1 = V1s_calc(pos_e2_buff,pos_R_buff, num_trajs)
    V2s,epos2 = V2s_calc(pos_e1_buff,pos_R_buff, num_trajs)
    VRs = VRs_calc(pos_e1_buff,pos_e2_buff, num_trajs)

    #Do the third term correction for the wavefunctions    
    K3 = c_Ht(el1_buff, el2_buff, \
                       R_buff, pos_e1_buff , pos_e2_buff ,pos_R_buff ,E_t_half,V1s, V2s, VRs, num_trajs)
    
    #And now the trajectory positions
    e1_K3, e2_K3, R_K3= ICWF_1e_vel(C_buff,el1_buff, el2_buff,\
                                R_buff, e1_mesh_buff , e2_mesh_buff, R_mesh_buff, num_trajs)
    C_K3,M3 = C_dot_1e_t(C_buff,el1_buff, el2_buff, R_buff,\
                    V1s,V2s,VRs, E_t_half,epos1,epos2, epos)

    pos_e1_buff = pos_e1 + p.dt*e1_K3
    pos_e2_buff = pos_e2 + p.dt*e2_K3
    pos_R_buff = pos_R + p.dt*R_K3

    e1_mesh_buff  = f.find_index(r1x_a,pos_e1_buff)
    e2_mesh_buff  = f.find_index(r2x_a,pos_e2_buff)
    R_mesh_buff  = f.find_index(Rx_a, pos_R_buff)

    el1_buff = el1 + p.dt*K3[0]
    el2_buff = el2 + p.dt*K3[1]
    R_buff = R + p.dt*K3[2]

    C_buff = C+p.dt*C_K3

    V1s,epos1 = V1s_calc(pos_e2_buff,pos_R_buff, num_trajs)
    V2s,epos2 = V2s_calc(pos_e1_buff,pos_R_buff, num_trajs)
    VRs = VRs_calc(pos_e1_buff,pos_e2_buff, num_trajs)

    #finally get the derivative information at the next timestep
    K4 = c_Ht(el1_buff, el2_buff, \
                       R_buff, pos_e1_buff , pos_e2_buff ,pos_R_buff ,E_t_half,V1s, V2s, VRs, num_trajs)
    
    #And now the trajectory positions
    e1_K4, e2_K4, R_K4= ICWF_1e_vel(C_buff,el1_buff, el2_buff,\
                                R_buff, e1_mesh_buff , e2_mesh_buff, R_mesh_buff, num_trajs)
    C_K4,M4 = C_dot_1e_t(C_buff,el1_buff, el2_buff, R_buff,\
                    V1s,V2s,VRs, E_t_adv,epos1,epos2, epos)


    #now update the global variables of trajectory positions
    pos_e1 = pos_e1 + (p.dt/6)*(e1_K1 + 2*e1_K2 +2*e1_K3 \
                        + e1_K4) 
    pos_e2 = pos_e2 + (p.dt/6)*(e2_K1 + 2*e2_K2 +2*e2_K3 \
                        + e2_K4) 
    pos_R = pos_R + (p.dt/6)*(R_K1 + 2*R_K2 +2*R_K3 \
                        + R_K4)

    print('pos_e1[0] = ', pos_e1[0])
    print('vel_e1[0] = ', (e1_K1[0] + 2*e1_K2[0] + 2*e1_K3[0] + e1_K4[0]))
    print('pos_e2[0] = ', pos_e2[0])
    print('vel_e2[0] = ', (e2_K1[0] + 2*e2_K2[0] + 2*e2_K3[0] + e2_K4[0]))
    print('pos_R[0] = ', pos_R[0])
    print('vel_R[0] = ', (R_K1[0] + 2*R_K2[0] + 2*R_K3[0] + R_K4[0]))
    print('\n')

    #print('el1 increment ', np.average(np.abs( (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]))))
    #print('el2 increment ', np.average(np.abs( (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]))))
    #print('R increment ', np.average(np.abs( (p.dt/6)*(K1[2] + 2*K2[2] + 2*K3[2] + K4[2]))))
    #print('C avg increment ', np.average(np.abs((p.dt/6)*(C_K1 + 2*C_K2 + 2*C_K3 + C_K4))))
    """
    print('C1 avg increment ', np.average(np.abs((p.dt/6)*C_K1)))
    print('C2 avg increment ', np.average(np.abs((p.dt/3)*C_K2)))
    print('C3 avg increment ', np.average(np.abs((p.dt/2)*C_K3)))
    print('C4 avg increment ', np.average(np.abs((p.dt/6)*C_K4)))
    print('C max increment ', np.abs((p.dt/6)*(C_K1 + 2*C_K2 + 2*C_K3 + C_K4)).max())
    print('Average pinv(M1) = %.1E'%Decimal(np.average(np.abs(M1))))
    print('Max pinv(M1) = %.1E'%Decimal(np.abs(M1).max()))
    print('Average pinv(M2) = %.1E'%Decimal(np.average(np.abs(M2))))
    print('Max pinv(M2) = %.1E'%Decimal(np.abs(M2).max()))
    print('Average pinv(M3) = %.1E'%Decimal(np.average(np.abs(M3))))
    print('Max pinv(M3) = %.1E'%Decimal(np.abs(M3).max()))
    print('Average pinv(M4) = %.1E'%Decimal(np.average(np.abs(M4))))
    print('Max pinv(M4) = %.1E'%Decimal(np.abs(M4).max()))
    """
    #print('pos_e1 increment ',np.average(np.abs((p.dt/6)*(e1_K1 + 2*e1_K2 +2*e1_K3 + e1_K4))))
    #print('pos_e2 increment ',np.average(np.abs((p.dt/6)*(e2_K1 + 2*e2_K2 +2*e2_K3 + e2_K4))))
    #print('pos_R increment ',np.average(np.abs((p.dt/6)*(R_K1 + 2*R_K2 +2*R_K3 + R_K4))))

    return el1 + (p.dt/6)*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]), \
            el2 + (p.dt/6)*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1]), \
            R + (p.dt/6)*(K1[2] + 2*K2[2] + 2*K3[2] + K4[2]), pos_e1, pos_e2, pos_R,\
            C + (p.dt/6)*(C_K1 + 2*C_K2 + 2*C_K3 + C_K4)

#}}}
#}}}
#{{{Propagators
#{{{ 1e cwf
#Want the entire dimensionality of the mesh to be (num_traj, dim_e) 
# or (num_traj, dim_n)
#NOTE that H is defined above as -1j*H AKA the time derivative
@jit( nogil=True)
def propagate_hermitian_cwf(psi0, tf_tot, tf_laser, Unique, plotDensity):
    #call("rm ./dump/2e/*",shell=True)
    call(['cp', 'params.py', './dump/1e/params_README'])
    call(['cp', 'run.py', './dump/1e/run_README'])

    #.5dt because for RK4 need E value at half timestep increments
    #this means that E_form has twice as many elements as there are universal timesteps
    E_form = f.shape_E(tf_laser, .5*p.dt, p.Amplitude, p.form)
    e1_mesh, e2_mesh, R_mesh = initialize_mesh(p.threshold,psi0,Unique, num_trajs)
    e1_wfs = psi0[:,e2_mesh, R_mesh].transpose()
    e2_wfs = psi0[e1_mesh, :, R_mesh]
    R_wfs = psi0[e1_mesh, e2_mesh, :]

    pos_e1 = r1x_a[e1_mesh]
    pos_e2 = r2x_a[e2_mesh]
    pos_R = Rx_a[R_mesh]
    """
    if (plotDensity==True):
        plt.ion()
        fig1, (ax_e, ax_n, ax_e_mesh, ax_n_mesh) = plt.subplots(4,1)
        plt.pause(.0001)
        fig_rho_e, = ax_e.plot(p.r1x, cwf_electronic_density(e1_wfs,e2_wfs),'.')
        fig_rho_n, = ax_n.plot(p.Rx,cwf_nuclear_density(R_wfs),'.')
        fig_e_mesh, =ax_e_mesh.plot(r1x_a[e1_mesh],r2x_a[e2_mesh], '.')
        fig_n_mesh, =ax_n_mesh.plot(Rx_a[R_mesh],'.')
    """
    curr_time_index = 0
    curr_time = curr_time_index*p.dt
    while (curr_time < tf_tot):
        if(curr_time < tf_laser-p.dt):
            e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R = T_dep_RK4(e1_wfs, e2_wfs, R_wfs, \
                                      e1_mesh, e2_mesh, R_mesh,pos_e1, pos_e2, pos_R,\
                                      curr_time_index, E_form)
        else: 
            e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R = RK4(e1_wfs, e2_wfs, R_wfs, \
                        e1_mesh,e2_mesh, R_mesh,\
                        pos_e1, pos_e2, pos_R)

        curr_time_index += 1
        curr_time = curr_time_index*p.dt
        #positions are now update in the RK4 method
        e1_mesh  = f.find_index(r1x_a, pos_e1)
        e2_mesh = f.find_index(r2x_a, pos_e2 )
        R_mesh = f.find_index(Rx_a, pos_R )

        if(curr_time_index%p.plot_step==0):
            np.save('./dump/1e/e1-cwf-'+"{:.2f}".format(curr_time), e1_wfs)
            np.save('./dump/1e/e2-cwf-'+"{:.2f}".format(curr_time), e2_wfs)
            np.save('./dump/1e/R-cwf-'+"{:.2f}".format(curr_time), R_wfs)
            print("{:.1f}".format(100*curr_time/tf_tot)+'% done')
            """
            if(plotDensity==True):
                plt.pause(.00001)
                fig_rho_e.set_ydata(cwf_electronic_density(e1_wfs,e2_wfs))
                fig_rho_n.set_ydata(cwf_nuclear_density(R_wfs))
                fig_e_mesh.set_ydata(r2x_a[e2_mesh])
                fig_e_mesh.set_xdata(r1x_a[e1_mesh])
                fig_n_mesh.set_ydata(Rx_a[R_mesh])
                ax_e_mesh.relim()
                ax_e_mesh.autoscale()
                ax_n_mesh.relim()
                ax_n_mesh.autoscale()
                ax_e.relim()
                ax_e.autoscale()
                ax_n.relim()
                ax_n.autoscale()
                fig1.suptitle('Time ' + str(curr_time)) #+ ' Amplitude ' + str(p.Amplitude))
                fig1.canvas.draw()
                fig1.canvas.flush_events
            """ 
    plt.show() 
#}}}
#{{{ 2e cwf
#NOTE that H is defined above as -1j*H AKA the time derivative
@jit( nogil=True)
def propagate_2e_hermitian_cwf(psi0, tf_tot, tf_laser, Unique, plotDensity):
    #.5dt because for RK4 need E value at half timestep increments
    #this means that E_form has twice as many elements as there are universal timesteps
    E_form = f.shape_E(tf_laser, .5*p.dt, p.Amplitude, p.form)
    e1_mesh, e2_mesh, R_mesh = initialize_2e_mesh(p.threshold,psi0,Unique)

    #call("rm ./dump/2e/*",shell=True)
    call(['cp', 'params.py', './dump/2e/README'])
    call(['cp', 'run.py', './dump/2e/run_README'])

    two_e_wfs = psi0[:,:, R_mesh].transpose()
    R_wfs = psi0[e1_mesh, e2_mesh, :]
    pos_e1 = r1x_a[e1_mesh]
    pos_e2 = r2x_a[e2_mesh]
    pos_R = Rx_a[R_mesh]

    if (plotDensity==True):
        plt.ion()
        fig1, (ax_e, ax_n, ax_e_mesh, ax_n_mesh) = plt.subplots(4,1)
        plt.pause(.0001)
        fig_rho_e, = ax_e.plot(p.r1x, cwf_2e_density(two_e_wfs),'.')
        fig_rho_n, = ax_n.plot(p.Rx,cwf_nuclear_density(R_wfs),'.')
        fig_e_mesh, =ax_e_mesh.plot(r1x_a[e1_mesh],r2x_a[e2_mesh], '.')
        fig_n_mesh, =ax_n_mesh.plot(Rx_a[R_mesh],'.')

    curr_time_index = 0
    curr_time = curr_time_index*p.dt
    while (curr_time < tf_tot):
        if(curr_time < tf_laser-p.dt):
            two_e_wfs, R_wfs, pos_e1, pos_e2, pos_R = T_dep_RK4_2e(two_e_wfs, R_wfs, e1_mesh, \
                                      e2_mesh, R_mesh, pos_e1, pos_e2, pos_R,\
                                      curr_time_index, E_form)
        else:
            two_e_wfs, R_wfs, pos_e1, pos_e2, pos_R = RK4_2e(two_e_wfs, R_wfs, \
                            e1_mesh,e2_mesh, R_mesh,\
                            pos_e1, pos_e2, pos_R)

        curr_time_index += 1
        curr_time = curr_time_index*p.dt

        e1_mesh  = f.find_index(r1x_a, pos_e1)
        e2_mesh = f.find_index(r2x_a, pos_e2 )
        R_mesh = f.find_index(Rx_a, pos_R )

        if(curr_time_index%p.plot_step==0):
            np.save('./dump/2e/'+'2e-cwf-'+str(num_trajs)+'-'+"{:.6f}".format(curr_time), two_e_wfs)
            np.save('./dump/2e/'+'R-cwf-'+str(num_trajs)+'-'+"{:.6f}".format(curr_time), R_wfs)
            print("{:.1f}".format(100*curr_time/tf_tot)+'% done')
            """
            if(plotDensity==True):
                plt.pause(.001)
                fig_rho_e.set_ydata(cwf_2e_density(two_e_wfs))
                if(BHF==False):
                    fig_rho_n.set_ydata(cwf_nuclear_density(R_wfs))
                if(BHF==True):
                    fig_rho_n.set_ydata(cwf_nuclear_density_BHF(R_wfs))
                fig_e_mesh.set_ydata(r2x_a[e2_mesh])
                fig_e_mesh.set_xdata(r1x_a[e1_mesh])
                fig_n_mesh.set_ydata(Rx_a[R_mesh])
                ax_e.relim()
                ax_e.autoscale()
                ax_n.relim()
                ax_n.autoscale()
                ax_e_mesh.relim()
                ax_e_mesh.autoscale()
                ax_n_mesh.relim()
                ax_n_mesh.autoscale()
                fig1.suptitle('Time ' + str(curr_time)) #+ ' Amplitude ' + str(p.Amplitude))
                fig1.canvas.draw()
                fig1.canvas.flush_events
            """
    plt.show()

#}}}

#}}}
#{{{ Ansatz
#{{{initialization and tools
@jit( nogil=True)
def initialize_C(psi, psi_el, psi_R, num_trajs):
    num_trajs = el1.shape[0]
    M = np.zeros((num_trajs,num_trajs)) + 0j
    G = np.zeros(num_trajs) + 0j
    C = np.zeros(num_trajs) + 0j
    r_integral = np.zeros(R_dim) + 0j
    for i in range(num_trajs):
        integral = 0j
        for Ri in range(R_dim):
            #r_integral[Ri] = np.sum(np.sum(np.conj(psi_el[i,:,:])*psi[:,:,Ri],0),0)
            for r2i in range(r2_dim):
                integral += np.sum(np.conj(psi_el[i,:,r2i])*np.conj(psi_R[i,Ri])*psi[:,r2i,Ri])
        G[i] = integral
        for j in range(i,num_trajs):
            M[i,j] = np.sum(np.sum(psi_el[i,:,:]*np.conj(psi_el[j,:,:])))*np.sum(np.conj(psi_R[j])*psi_R[i])
            M[j,i] = np.conj(M[i,j])
    G *= p.n_spacing*p.e_spacing**2
    M *= p.n_spacing*p.e_spacing**2
   
    C = np.dot(np.linalg.pinv(M),G)
    return C
@jit( nogil=True)
def initialize_C_1e(psi, psi_el1, psi_el2, psi_R, num_trajs):
    G = np.zeros(num_trajs,dtype=np.complex128)
    C = np.zeros(num_trajs,dtype=np.complex128)
    M = (np.conj(psi_el1) @ psi_el1.transpose())\
        *(np.conj(psi_el2) @ psi_el2.transpose())\
        *(np.conj(psi_R) @ psi_R.transpose())\
        *p.n_spacing*p.e_spacing**2
    for i in range(num_trajs):
        integral = 0j
        for Ri in range(R_dim):
            for r2i in range(r2_dim):
                integral += np.dot(np.conj(psi_el1[i])*np.conj(psi_el2[i,r2i])*np.conj(psi_R[i,Ri]),psi[:,r2i,Ri])

        G[i] = integral
    G *= p.n_spacing*p.e_spacing**2
    C = np.linalg.pinv(M).dot(G)
    #C = np.dot(np.linalg.pinv(M),G)
    return C

@jit( nogil=True)
def initialize_C_slater(psi, psi_el1, psi_el2, psi_R, num_trajs):
    G = np.zeros(num_trajs) + 0j
    C = np.zeros(num_trajs) + 0j
    M =np.dot(np.conj(psi_R),psi_R.transpose())\
        *p.n_spacing*p.e_spacing**2
    for i in range(num_trajs):
        for j in range(num_trajs):
            M[i,j] *= (.5)*np.sum(np.sum( (np.outer(np.conj(psi_el1[i]),np.conj(psi_el2[i]))+np.outer(np.conj(psi_el2[i]),np.conj(psi_el1[i])))*(np.outer(psi_el1[j],psi_el2[j])+np.outer(psi_el2[j],psi_el1[j])) ))
        integral = 0j
        for Ri in range(R_dim):
            for r2i in range(r2_dim):
                integral += np.sum(np.conj(psi_el1[i])*np.conj(psi_el2[i,r2i])*np.conj(psi_R[i,Ri])*psi[:,r2i,Ri])
                integral += np.sum(np.conj(psi_el1[i,r2i])*np.conj(psi_el2[i])*np.conj(psi_R[i,Ri])*psi[:,r2i,Ri])

        G[i] = (1/np.sqrt(2))*integral
    G *= p.n_spacing*p.e_spacing**2
    C = np.linalg.pinv(M).dot(G) 
    #C = np.dot(np.linalg.pinv(M),G)
    return C

@jit( parallel=True,nogil=True)
def flatten_cwf(el1,el2,R):
    dim1 = el1.shape[1]
    dim2= el2.shape[1]
    dimR = R.shape[1]
    out = np.ones((el1.shape[0],dim1*dim2*dimR))+0j
    for t in prange(out.shape[0]):
        for i in prange(out.shape[1]):
            out[t,i] = el1[t,int(np.floor(i/(dimR*dim2)))%dim1]*el2[t,int(np.floor(i/dimR))%dim2]*R[t,i%(dimR)]
    return out

@njit( parallel=True,nogil=True)
def recon_psi_1e(C,el1_wfs,el2_wfs,R_wfs, num_trajs):
    psi_recon = np.zeros((num_trajs, r1_dim, r2_dim, R_dim),dtype=np.complex128)
    for i in prange(num_trajs):
        for R_wfsi in prange(R_dim):
            for r2_wfsi in range(r2_dim):
                psi_recon[i,:,r2_wfsi,R_wfsi] = C[i]*el1_wfs[i]*el2_wfs[i,r2_wfsi]*R_wfs[i,R_wfsi]
    return np.sum(psi_recon,0)

@njit( parallel=True,nogil=True)
def recon_psi_slater(C,el1_wfs,el2_wfs,R_wfs, num_trajs):
    psi_recon = np.zeros((num_trajs, r1_dim, r2_dim, R_dim),dtype=np.complex128)
    for i in prange(num_trajs):
        for R_wfsi in prange(R_dim):
            for r2_wfsi in range(r2_dim):
                psi_recon[i,:,r2_wfsi,R_wfsi] = el1_wfs[i]*el2_wfs[i,r2_wfsi]
                psi_recon[i,:,r2_wfsi,R_wfsi] += el1_wfs[i,r2_wfsi]*el1_wfs[i]
                psi_recon[i,:,r2_wfsi,R_wfsi] *= (1/np.sqrt(2))*C[i]*R_wfs[i,R_wfsi]
    return np.sum(psi_recon,0)

@njit( parallel=True,nogil=True)
def recon_psi(C,el_wfs,R_wfs):
    psi_recon = np.zeros((num_trajs,el_wfs.shape[1], el_wfs.shape[2], R_wfs.shape[1]),dtype=np.complex128) 
    for i in prange(num_trajs):
        for R_wfsi in prange(R_wfs.shape[1]):
            psi_recon[i,:,:,R_wfsi] = C[i]*el_wfs[i]*R_wfs[i,R_wfsi]
    return np.sum(psi_recon,0)

#}}}
#{{{ 2e cwf C_dots NEED TO BE UPDATED
#This double counts the interactions by using Vfull!!!!
@njit( parallel=True,nogil=True)
def C_dot(Ct,psi_el,psi_R, e1_mesh, e2_mesh, R_mesh):
    num_trajs = psi_el.shape[0]
    M_e = np.zeros((num_trajs,num_trajs),dtype=np.complex128) 
    M_R = np.zeros((num_trajs,num_trajs),dtype=np.complex128) 
    M = np.zeros((num_trajs,num_trajs),dtype=np.complex128) 
    W = np.zeros((num_trajs,num_trajs),dtype=np.complex128)
    W1 = np.zeros((num_trajs,num_trajs),dtype=np.complex128) 
    W2 = np.zeros((num_trajs,num_trajs),dtype=np.complex128)
    #because of the way that numba's parallelization works, it can't call numpy.sum twice on 
    # complex arrays, thus there need to be intermediate arrays 
    sum_array = np.zeros((psi_el[0,0].shape[0]),dtype=np.complex128)
    R_array = np.zeros(psi_R[0].shape[0])+0j
    for i in prange(num_trajs):
        for j in prange(i,num_trajs):
            #integrate over one electron cooerdinate
            sum_array = np.sum(psi_el[j,:,:]*np.conj(psi_el[i,:,:]),0)
            #integrate over the other one
            M_e[i,j] = np.sum(sum_array)
            #integrate over nuclear coordinates
            M_R[i,j] = np.sum(np.conj(psi_R[i])*psi_R[j])
            #product of the integrals
            M[i,j] = M_e[i,j]*M_R[i,j]
    for i in prange(num_trajs):
        for j in prange(num_trajs):
            if(j<i):
                M[i,j] = np.conj(M[j,i])
                M_e[i,j] = np.conj(M_e[j,i])
                M_R[i,j] = np.conj(M_R[j,i])
            ##Full potential integral
            #   integrate potential w/rt one electron coordinate
            for Ri in range(R_array.shape[0]):
                sum_array = np.sum(psi_el[j]*np.conj(psi_el[i])*Vfull[:,:,Ri],0)
                #   integrate w/rt the other electron coordinate
                R_array[Ri] = np.sum(sum_array)
            #   integrate w/rt the nuclear coordinates
            W[i,j] = np.sum(R_array*psi_R[j]*np.conj(psi_R[i]))

            ##electron potential Intgral
            #   integrate w/rt one electron coordinate
            sum_array = np.sum(psi_el[j,:,:]*np.conj(psi_el[i,:,:])*V2e[R_mesh[i],:,:],0)
            # integrate e/rt the other electron coordinate and multiply by nuclear CWF norm
            W1[i,j] = np.sum(sum_array)*M_R[i,j]

            ##Nuclear Potential Integral
            W2[i,j] = M_e[i,j]*np.sum(psi_R[j]*np.conj(psi_R[i])*VR[e1_mesh[i],e2_mesh[i],:])

    M*=p.e_spacing**2*p.n_spacing
    W*=p.e_spacing**2*p.n_spacing
    W1*=p.e_spacing**2*p.n_spacing
    W2*=p.e_spacing**2*p.n_spacing

    return (-1j*np.dot(np.linalg.pinv(M),np.dot(W-W1-W2,Ct)))

        

#time dependent version of the above, for external laser field
@njit( parallel=True,nogil=True)
def C_dot_t(Ct,psi_el,psi_R, e1_mesh, e2_mesh, R_mesh,Et):
    num_trajs = psi_el.shape[0]
    M_e = np.zeros((num_trajs,num_trajs),dtype=np.complex128) 
    M_R = np.zeros((num_trajs,num_trajs),dtype=np.complex128) 
    M = np.zeros((num_trajs,num_trajs),dtype=np.complex128) 
    W = np.zeros((num_trajs,num_trajs),dtype=np.complex128)
    W1 = np.zeros((num_trajs,num_trajs),dtype=np.complex128) 
    W2 = np.zeros((num_trajs,num_trajs),dtype=np.complex128)
    #because of the way that numba's parallelization works, it can't call numpy.sum twice on 
    # complex arrays, thus there need to be intermediate arrays 
    sum_array = np.zeros((psi_el[0,0].shape[0]),dtype=np.complex128)
    R_array = np.zeros(psi_R[0].shape[0])+0j
    for i in prange(num_trajs):
        for j in prange(i,num_trajs):
            #integrate over one electron cooerdinate
            sum_array = np.sum(psi_el[j,:,:]*np.conj(psi_el[i,:,:]),0)
            #integrate over the other one
            M_e[i,j] = np.sum(sum_array)
            #integrate over nuclear coordinates
            M_R[i,j] = np.sum(np.conj(psi_R[i])*psi_R[j])
            #product of the integrals
            M[i,j] = M_e[i,j]*M_R[i,j]
    for i in prange(num_trajs):
        for j in prange(num_trajs):
            if(j<i):
                M[j,i] = np.conj(M[i,j])
                M_e[j,i] = np.conj(M_e[i,j])
                M_R[j,i] = np.conj(M_R[i,j])
            ##Full potential integral
            #   integrate potential w/rt one electron coordinate
            for Ri in range(R_array.shape[0]):
                sum_array = np.sum(psi_el[i]*np.conj(psi_el[j])*(Vfull[:,:,Ri]+Et*epos),0)
                #   integrate w/rt the other electron coordinate
                R_array[Ri] = np.sum(sum_array)
            #   integrate w/rt the nuclear coordinates
            W[i,j] = np.sum(R_array*psi_R[i]*np.conj(psi_R[j]))

            ##electron potential Intgral
            #   integrate w/rt one electron coordinate
            sum_array = np.sum(psi_el[i,:,:]*np.conj(psi_el[j,:,:])*V2e[R_mesh[i],:,:],0)
            # integrate e/rt the other electron coordinate and multiply by nuclear CWF norm
            W1[i,j] = np.sum(sum_array)*M_R[i,j]

            ##Nuclear Potential Integral
            W2[i,j] = M_e[i,j]*np.sum(psi_R[i]*np.conj(psi_R[j])*VR[e1_mesh[i],e2_mesh[i],:])

    M*=p.e_spacing**2*p.n_spacing
    W*=p.e_spacing**2*p.n_spacing
    W1*=p.e_spacing**2*p.n_spacing
    W2*=p.e_spacing**2*p.n_spacing

    return (-1j*np.dot(np.linalg.pinv(M),np.dot(W-W1-W2,Ct)))
#}}}
@jit(  nogil=True)
def C_dot_1e(Ct,el1,el2,R,V1s, V2s, VRs):
    M_e1 = np.dot(np.conj(el1),el1.transpose())
    M_e2 = np.dot(np.conj(el2),el2.transpose())
    M_R = np.dot(np.conj(R),R.transpose())
    M = M_e1*M_e2*M_R
    W = M_e1*M_e2*np.dot(np.conj(R),(R*Rinv_flat).transpose())
    for i in range(num_trajs):
        W[i,:] += M_e1[i,:]*np.sum(np.tile(np.conj(el2[i]),(num_trajs,1))\
                 *el2*np.dot(np.tile(np.conj(R[i]),(num_trajs,1))*R,Ve2R.transpose()),1)
        W[i,:] += M_e2[i,:]*np.sum(np.tile(np.conj(el1[i]),(num_trajs,1))\
                 *el1*np.dot(np.tile(np.conj(R[i]),(num_trajs,1))*R,Ve1R.transpose()),1)
        W[i,:] += M_R[i,:]*np.sum(np.tile(np.conj(el1[i]),(num_trajs,1))\
                 *el1*np.dot(np.tile(np.conj(el2[i]),(num_trajs,1))*el2,(Vee).transpose()),1)
    W1 = np.dot(np.conj(el1),(el1*V1s).transpose())*M_e2*M_R
    W2 = np.dot(np.conj(el2),(el2*V2s).transpose())*M_e1*M_R
    WR = np.dot(np.conj(R),(R*VRs).transpose())*M_e1*M_e2

    M *=p.e_spacing**2*p.n_spacing
    W *=p.e_spacing**2*p.n_spacing
    W1 *=p.e_spacing**2*p.n_spacing
    W2 *=p.e_spacing**2*p.n_spacing
    WR *=p.e_spacing**2*p.n_spacing
    
    return (-1j*np.dot(np.linalg.pinv(M),np.dot(W-W1-W2-WR,Ct)))

#Time dependent version of the above
#epos1 refers to the slices of epos that electron 1 sees from the electron 2 trajectories,
    #thus is epos[:,e2_mesh]
@jit(  nogil=True)
def C_dot_1e_t(Ct,el1,el2,R,V1s, V2s, VRs,Et,epos1, epos2, epos):
    M_e1 = np.dot(np.conj(el1),el1.transpose())
    M_e2 = np.dot(np.conj(el2),el2.transpose())
    M_R = np.dot(np.conj(R),R.transpose())
    M = M_e1*M_e2*M_R
    W = M_e1*M_e2*np.dot(np.conj(R),(R*Rinv_flat).transpose())
    for i in range(num_trajs):
        W[i,:] += M_e1[i,:]*np.sum(np.tile(np.conj(el2[i]),(num_trajs,1))\
                 *el2*np.dot(np.tile(np.conj(R[i]),(num_trajs,1))*R,Ve2R.transpose()),1)
        W[i,:] += M_e2[i,:]*np.sum(np.tile(np.conj(el1[i]),(num_trajs,1))\
                 *el1*np.dot(np.tile(np.conj(R[i]),(num_trajs,1))*R,Ve1R.transpose()),1)
        W[i,:] += M_R[i,:]*np.sum(np.tile(np.conj(el1[i]),(num_trajs,1))\
                 *el1*np.dot(np.tile(np.conj(el2[i]),(num_trajs,1))*el2,(Vee+Et*epos).transpose()),1)
    W1 = np.dot(np.conj(el1),(el1*(V1s-Et*epos1)).transpose())*M_e2*M_R
    W2 = np.dot(np.conj(el2),(el2*(V2s-Et*epos2)).transpose())*M_e1*M_R
    WR = np.dot(np.conj(R),(R*VRs).transpose())*M_e1*M_e2

    M *=p.e_spacing**2*p.n_spacing
    W *=p.e_spacing**2*p.n_spacing
    W1 *=p.e_spacing**2*p.n_spacing
    W2 *=p.e_spacing**2*p.n_spacing
    WR *=p.e_spacing**2*p.n_spacing

    """
    print('M average = ', np.average(np.abs(M)))
    print('M max = ', np.abs(M).max())
    print('W average = ', np.average(np.abs(W)))
    print('W max = ', np.abs(W).max())
    print('W1 average = ', np.average(np.abs(W1)))
    print('W1 max = ', np.abs(W1).max())
    print('W2 average = ', np.average(np.abs(W2)))
    print('W2 max = ', np.abs(W2).max())
    print('WR average = ', np.average(np.abs(WR)))
    print('WR max = ', np.abs(WR).max())
    """

    return (-1j*np.dot(np.linalg.pinv(M),np.dot(W-W1-W2-WR,Ct))),np.linalg.pinv(M)
#@jit(nogil=True)
def post_RK4_sieve(C,el1,el2,R,pos_e1,pos_e2,pos_R, mesh_e1, mesh_e2, mesh_R):
    global num_trajs, Rinv_flat
    #get indices of cwf slices to remove
    to_remove = [int(round(np.abs(pos_e1-x).argmin())) for x in pos_e1 if x > r1x_a[-1]]
    to_remove += [int(round(np.abs(pos_e1-x).argmin())) for x in pos_e1 if x < r1x_a[0]]
    to_remove+=[ int(round(np.abs(pos_e2-x).argmin())) for x in pos_e2 if x > r2x_a[-1]]
    to_remove+=[int(round(np.abs(pos_e2-x).argmin())) for x in pos_e2 if x < r2x_a[0]]
    to_remove+=[int(round(np.abs(pos_R-x).argmin())) for x in pos_R if x > Rx_a[-1]]
    to_remove+=[int(round(np.abs(pos_R-x).argmin())) for x in pos_R if x < Rx_a[0]]
    to_remove = list(set(to_remove))
    if len(to_remove)>0:
        psi_recon = recon_psi_1e(C,el1,el2,R,num_trajs)
        psi_recon = psi_recon/np.sqrt(np.sum(np.sum(np.sum(np.abs(psi_recon)**2)))*p.e_spacing**2*p.n_spacing)
        num_trajs = num_trajs - len(to_remove)
        Rinv_flat = np.delete(Rinv_flat,to_remove,axis=0)
        mesh_e1 = np.delete(mesh_e1,to_remove)
        mesh_e2 = np.delete(mesh_e2,to_remove)
        mesh_R = np.delete(mesh_R,to_remove)
        pos_e1 = np.delete(pos_e1,to_remove)
        pos_e2 = np.delete(pos_e2,to_remove)
        pos_R = np.delete(pos_R,to_remove)
        el1 = psi_recon[:,mesh_e2, mesh_R].transpose()
        el2 = psi_recon[mesh_e1, :, mesh_R]
        R = psi_recon[mesh_e1, mesh_e2, :]
        el1,el2,R = normalize_cwf(el1,el2,R,num_trajs)
        C = initialize_C_1e(psi_recon,el1,el2,R,num_trajs)
        """
        #mesh_e1, mesh_e2, mesh_R = initialize_mesh(p.threshold,psi_recon,False)
        pos_e1 = r1x_a[mesh_e1]
        pos_e2 = r2x_a[mesh_e2]
        pos_R = Rx_a[mesh_R]
        C = np.delete(C,to_remove)
        #el1 = np.delete(el1,to_remove, axis=0)
        #el2 = np.delete(el2,to_remove, axis=0)
        #R = np.delete(R,to_remove, axis=0)
        el1 = psi_recon[:,mesh_e2, mesh_R].transpose()
        el2 = psi_recon[mesh_e1,:, mesh_R]
        R = psi_recon[mesh_e1,mesh_e2, :]
        """
    else:
        el1,el2,R = normalize_cwf(el1,el2,R,num_trajs)

    return C, el1, el2, R, pos_e1, pos_e2, pos_R, mesh_e1, mesh_e2, mesh_R

@jit(nogil=True)
def resample(C,el1,el2,R):
    psi_recon = recon_psi_1e(C,el1,el2,R,num_trajs)
    psi_recon = psi_recon/np.sqrt(np.sum(np.sum(np.sum(np.abs(psi_recon)**2)))*p.e_spacing**2*p.n_spacing)
    e1_mesh, e2_mesh,R_mesh = initialize_mesh(p.threshold,psi_recon,False, num_trajs)
    pos_e1 = p.r1x_array[e1_mesh]
    pos_e2 = p.r2x_array[e2_mesh]
    pos_R = p.Rx_array[R_mesh]
    el1 = psi_recon[:,e2_mesh,R_mesh].transpose()
    el2 = psi_recon[e1_mesh,:,R_mesh]
    R = psi_recon[e1_mesh, e2_mesh,:]
    el1,el2,R = normalize_cwf(el1,el2,R,num_trajs)
    C = initialize_C_1e(psi_recon,el1,el2,R,num_trajs)
    return el1,el2,R,pos_e1,pos_e2,pos_R,e1_mesh,e2_mesh,R_mesh,C 

@jit( nogil=True)
def ICWF_propagate(psi0,tf_tot,tf_laser, wf_type):
    r1x_a = p.r1x_array
    r2x_a = p.r2x_array
    Rx_a = p.Rx_array
    Cr2 = p.Cr2
    mu = p.mu
    M1 = p.M1
    M2 = p.M2
    call(['cp', 'params.py', './dump/1e/params_README'])
    call(['cp', 'run.py', './dump/1e/run_README'])
    E_form = f.shape_E(tf_laser, .5*p.dt, p.Amplitude, p.form)
    new_mesh = True
    if(new_mesh==True):
        e1_mesh, e2_mesh, R_mesh = initialize_mesh(p.threshold,psi0,False,num_trajs)
        np.save('./dump/1e/init_e1_mesh',e1_mesh)
        np.save('./dump/1e/init_e2_mesh',e2_mesh)
        np.save('./dump/1e/init_R_mesh',R_mesh)
    if(new_mesh!=True):
        e1_mesh = np.load('./dump/1e/init_e1_mesh.npy')
        e2_mesh = np.load('./dump/1e/init_e2_mesh.npy')
        R_mesh  = np.load('./dump/1e/init_R_mesh.npy')
    if(wf_type=='2e'):
        two_e_wfs = psi0[:,:,R_mesh].transpose()
    if(wf_type=='1e'):
        e1_wfs = psi0[:,e2_mesh, R_mesh].transpose()
        e2_wfs = psi0[e1_mesh, :, R_mesh]
    R_wfs = psi0[e1_mesh, e2_mesh, :]
    e1_wfs,e2_wfs,R_wfs = normalize_cwf(e1_wfs,e2_wfs,R_wfs,num_trajs)

    pos_e1 = r1x_a[e1_mesh]
    pos_e2 = r2x_a[e2_mesh]
    pos_R = Rx_a[R_mesh]
 
    curr_time_index = 0
    curr_time = curr_time_index*p.dt
    if(wf_type=='2e'):
        C = initialize_C(psi0,two_e_wfs,R_wfs,num_trajs)
        while (curr_time < tf_tot):
            if(curr_time < tf_laser-p.dt):
                two_e_wfs, R_wfs, pos_e1, pos_e2, pos_R = T_dep_RK4_2e_ICWF(two_e_wfs, R_wfs, e1_mesh, \
                                          e2_mesh, R_mesh, pos_e1, pos_e2, pos_R,\
                                          curr_time_index, E_form, C)
            else:
                two_e_wfs, R_wfs, pos_e1, pos_e2, pos_R = RK4_2e_ICWF(two_e_wfs, R_wfs, \
                                e1_mesh,e2_mesh, R_mesh,\
                                pos_e1, pos_e2, pos_R,C)
            curr_time_index += 1
            curr_time = curr_time_index*p.dt
            #positions are now update in the RK4 method
            e1_mesh  = f.find_index(r1x_a, pos_e1)
            e2_mesh = f.find_index(r2x_a, pos_e2 )
            R_mesh = f.find_index(Rx_a, pos_R )
            if(curr_time_index%p.psi_save_interval==0):
                np.save('./dump/1e/2e-cwf-'+"{:.6f}".format(curr_time), two_e_wfs)
                np.save('./dump/1e/R-cwf-'+"{:.6f}".format(curr_time), R_wfs)
                np.save('./dump/1e/ansatz'+"{:.6f}".format(curr_time), recon_psi(C,two_e_wfs,R_wfs))
                print("{:.1f}".format(100*curr_time/tf_tot)+'% done')
    if(wf_type=='1e'):
        C = initialize_C_1e(psi0,e1_wfs,e2_wfs,R_wfs,num_trajs)
        norms = [print_norm(C,e1_wfs,e2_wfs,R_wfs), print_norm(C,e1_wfs,e2_wfs,R_wfs)]
        norm_i = 0
        while (curr_time < tf_tot):
            if(curr_time < tf_laser-p.dt):
                e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R, C = T_dep_RK4_1e_ICWF(e1_wfs, e2_wfs, R_wfs,\
                                e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R, \
                                curr_time_index, E_form, C)
            else:
                e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R, C = RK4_1e_ICWF(e1_wfs, e2_wfs, R_wfs, \
                                e1_mesh,e2_mesh, R_mesh, pos_e1, pos_e2, pos_R,C)
            e1_mesh  = f.find_index(r1x_a, pos_e1)
            e2_mesh = f.find_index(r2x_a, pos_e2 )
            R_mesh = f.find_index(Rx_a, pos_R )
            #e1_wfs,e2_wfs,R_wfs = normalize_cwf(e1_wfs,e2_wfs,R_wfs, num_trajs)
            C, e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R, e1_mesh, e2_mesh, R_mesh= post_RK4_sieve(C, e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R, e1_mesh, e2_mesh, R_mesh)
            curr_time_index += 1
            curr_time = curr_time_index*p.dt
            print('time=',curr_time)
            #if(curr_time_index%200 == 0):
            #    e1_wfs, e2_wfs, R_wfs, pos_e1,pos_e2, pos_R, e1_mesh, e2_mesh, R_mesh, C = resample(C,e1_wfs,e2_wfs,R_wfs)
            if(curr_time_index%p.psi_save_interval==0):
                dummy = f.normalize(recon_psi_1e(C,e1_wfs,e2_wfs,R_wfs, num_trajs))
                norms[norm_i] = print_norm(C,e1_wfs,e2_wfs,R_wfs)
                norm_i+=1
                norm_i=norm_i%2
                print('num_trajs = ', num_trajs)
                np.save('./dump/1e/e1-cwf-'+"{:.6f}".format(curr_time), e1_wfs)
                np.save('./dump/1e/e2-cwf-'+"{:.6f}".format(curr_time), e2_wfs)
                np.save('./dump/1e/R-cwf-'+"{:.6f}".format(curr_time), R_wfs)
                np.save('./dump/1e/ansatz'+"{:.6f}".format(curr_time), recon_psi_1e(C,e1_wfs,e2_wfs,R_wfs, num_trajs))
                np.save('./dump/1e/e1_mesh-'+"{:.6f}".format(curr_time),e1_mesh)
                np.save('./dump/1e/e2_mesh-'+"{:.6f}".format(curr_time),e2_mesh)
                np.save('./dump/1e/R_mesh-'+"{:.6f}".format(curr_time),R_mesh)
                np.save('./dump/1e/pos_e1-'+"{:.6f}".format(curr_time),pos_e1)
                np.save('./dump/1e/pos_e2-'+"{:.6f}".format(curr_time),pos_e2)
                np.save('./dump/1e/pos_R-'+"{:.6f}".format(curr_time),pos_R)
                np.save('./dump/1e/C-'+"{:.6f}".format(curr_time),C)
               
                print("{:.1f}".format(100*curr_time/tf_tot)+'% done')

def ICWF_restart_propagate(tf_tot,tf_laser, wf_type,t_init, e1_wfs,e2_wfs,R_wfs, pos_e1,pos_e2, pos_R, e1_mesh, e2_mesh,R_mesh,C):
    global num_trajs, Rinv_flat
    num_trajs = e1_wfs.shape[0]
    Rinv_flat = np.array([1/p.Rx_array for x in range(num_trajs)])
    init_num_trajs = p.num_trajs
    r1x_a = p.r1x_array
    r2x_a = p.r2x_array
    Rx_a = p.Rx_array
    Cr2 = p.Cr2
    mu = p.mu
    M1 = p.M1
    M2 = p.M2
    E_form = f.shape_E(tf_laser, .5*p.dt, p.Amplitude, wf_type)
    curr_time_index = int(t_init/p.dt)
    curr_time = curr_time_index*p.dt
    while (curr_time < tf_tot):
        if(curr_time < tf_laser-p.dt):
            e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R, C = T_dep_RK4_1e_ICWF(e1_wfs, e2_wfs, R_wfs,\
                            e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R, \
                            curr_time_index, E_form, C)
            #dummy = f.normalize(recon_psi_1e(C,e1_wfs,e2_wfs,R_wfs,e1_wfs.shape[0]))
        else:
            e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R, C = RK4_1e_ICWF(e1_wfs, e2_wfs, R_wfs, \
                            e1_mesh,e2_mesh, R_mesh, pos_e1, pos_e2, pos_R,C)
            #plt.plot(np.abs(C))

        e1_mesh  = f.find_index(r1x_a, pos_e1)
        e2_mesh = f.find_index(r2x_a, pos_e2 )
        R_mesh = f.find_index(Rx_a, pos_R )
        C, e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R, e1_mesh, e2_mesh, R_mesh= post_RK4_sieve(C, e1_wfs, e2_wfs, R_wfs, pos_e1, pos_e2, pos_R, e1_mesh, e2_mesh, R_mesh)
        curr_time_index += 1
        curr_time = curr_time_index*p.dt
        #print('num_trajs = ', num_trajs)
        #print('time = '+str(curr_time))
        #positions are now update in the RK4 method
        #print('\n')
        #if(curr_time > 60.6 and curr_time < 60.63):
            #plt.plot(f.integrate_over_electrons(recon_psi_1e(C,e1_wfs,e2_wfs,R_wfs,num_trajs)))
            #plt.show()
        #    plt.pcolormesh(np.sum(np.abs(recon_psi_1e(C,e1_wfs,e2_wfs,R_wfs,num_trajs))**2*p.n_spacing,2))
        #    plt.show()
            #plt.plot(pos_e1,'.', label='r1 '+str(curr_time))
            #plt.plot(pos_e2,'.', label='r2 '+str(curr_time))
            #plt.plot(pos_R,'.', label='R '+str(curr_time))
            #plt.plot(np.abs(C), '.', label='C '+str(curr_time))
        #if(curr_time == 60.63):
        #    plt.legend()
        #    plt.show()
        if(curr_time_index%p.psi_save_interval==0):
            np.save('./dump/1e/e1-cwf-'+"{:.6f}".format(curr_time), e1_wfs)
            np.save('./dump/1e/e2-cwf-'+"{:.6f}".format(curr_time), e2_wfs)
            np.save('./dump/1e/R-cwf-'+"{:.6f}".format(curr_time), R_wfs)
            np.save('./dump/1e/ansatz'+"{:.6f}".format(curr_time), recon_psi_1e(C,e1_wfs,e2_wfs,R_wfs, num_trajs))
            np.save('./dump/1e/e1_mesh-'+"{:.6f}".format(curr_time),e1_mesh)
            np.save('./dump/1e/e2_mesh-'+"{:.6f}".format(curr_time),e2_mesh)
            np.save('./dump/1e/R_mesh-'+"{:.6f}".format(curr_time),R_mesh)
            np.save('./dump/1e/pos_e1-'+"{:.6f}".format(curr_time),pos_e1)
            np.save('./dump/1e/pos_e2-'+"{:.6f}".format(curr_time),pos_e2)
            np.save('./dump/1e/pos_R-'+"{:.6f}".format(curr_time),pos_R)
            np.save('./dump/1e/C-'+"{:.6f}".format(curr_time),C)

        
#}}}
def initialize_test(psi, num_trajs):
    e1_mesh, e2_mesh, R_mesh = initialize_mesh(p.threshold,psi,True, num_trajs)
    pos_e1 = p.r1x_array[e1_mesh]
    pos_e2 = p.r1x_array[e2_mesh]
    pos_R = p.r1x_array[R_mesh]
    el = psi[:,:,R_mesh].transpose()
    R = psi[e1_mesh, e2_mesh,:]
    C = initialize_C(psi,el,R,num_trajs)
    return e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R, el, R, C

def initialize_1e_test(psi, num_trajs):
    e1_mesh, e2_mesh, R_mesh = initialize_mesh(p.threshold,psi,False, num_trajs)
    pos_e1 = p.r1x_array[e1_mesh]
    pos_e2 = p.r2x_array[e2_mesh]
    pos_R = p.Rx_array[R_mesh]
    el1 = psi[:,e2_mesh,R_mesh].transpose()
    el2 = psi[e1_mesh,:,R_mesh]
    R = psi[e1_mesh, e2_mesh,:]
    el1,el2,R = normalize_cwf(el1,el2,R, num_trajs)
    C = initialize_C_1e(psi,el1,el2,R,num_trajs)
    return e1_mesh, e2_mesh, R_mesh, pos_e1, pos_e2, pos_R, el1, el2,R, C

@jit(nopython=True,nogil=True)
def normalize_cwf(el1,el2,R,num_trajs):
    for i in range(num_trajs):
        el1[i] = el1[i]/np.sqrt(np.sum(np.abs(el1[i])**2)*p.e_spacing)
        el2[i] = el2[i]/np.sqrt(np.sum(np.abs(el2[i])**2)*p.e_spacing)
        R[i] = R[i]/np.sqrt(np.sum(np.abs(R[i])**2)*p.n_spacing)
    return el1, el2, R

@jit(nogil=True)
def restart_from_time(t,data_dir):
    """ restart_from_time(time, string_directory_with_data)
    loads the conditional wavefunctions, positions, corresponding mesh, C and ansatz wf. from a given time.
    Returns as C, el1, el2, R, pos_e1, pos_e2, pos_R, mesh_e1, mesh_e2, mesh_R, ansatz
    """
    el1 = np.load(data_dir+'e1-cwf-'+"{:.6f}".format(t)+'.npy')
    el2 = np.load(data_dir+'e2-cwf-'+"{:.6f}".format(t)+'.npy')
    R = np.load(data_dir+'R-cwf-'+"{:.6f}".format(t)+'.npy')
    pos_e1 = np.load(data_dir+'pos_e1-'+"{:.6f}".format(t)+'.npy')
    pos_e2 = np.load(data_dir+'pos_e2-'+"{:.6f}".format(t)+'.npy')
    pos_R = np.load(data_dir+'pos_R-'+"{:.6f}".format(t)+'.npy')
    mesh_e1 = np.load(data_dir+'e1_mesh-'+"{:.6f}".format(t)+'.npy')
    mesh_e2 = np.load(data_dir+'e2_mesh-'+"{:.6f}".format(t)+'.npy')
    mesh_R = np.load(data_dir+'R_mesh-'+"{:.6f}".format(t)+'.npy')
    C = np.load(data_dir+'C-'+"{:.6f}".format(t)+'.npy')
    ansatz = np.load(data_dir+'ansatz'+"{:.6f}".format(t)+'.npy')
    return C, el1, el2, R, pos_e1,pos_e2,pos_R,mesh_e1,mesh_e2,mesh_R,ansatz 
