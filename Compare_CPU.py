import numpy as np
import scipy.linalg as lin

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import time
import random

import PWM_Toolbox as tb

#%% controller
g = 0.03 * (2*np.pi) #unit: 2pi*GHz
Delta = 0. * (2*np.pi) #unit: 2pi*GHz
eta = -0.2 * (2*np.pi) #unit: 2pi*GHz
xi = 0.1 * (2*np.pi) #unit: 2pi*GHz

tau = 1 #unit: ns
mode = 'Trotter'

#%% single qubit operator
N_level = 3
Id = np.eye(N_level)

sigma_p = np.zeros((N_level,N_level), dtype=complex)
sigma_m = np.zeros((N_level,N_level), dtype=complex)
for num_1 in range(1,N_level):
    sigma_p[num_1,num_1-1] = np.sqrt(num_1)
    sigma_m[num_1-1,num_1] = np.sqrt(num_1)

sigma_x = sigma_p + sigma_m
sigma_y = -1j*( sigma_p - sigma_m )
sigma_z = sigma_p @ sigma_m

#%% CPU
N_rep = 10
N_q_list = np.linspace(1,7,7, dtype=int)
N_c_list = np.linspace(1,17,17, dtype=int)
M = 1 

t_PWM = []
t_PWC = []
for N_q in N_q_list:
    #%% Hamiltonian
    dim = N_level**N_q
    H_d = np.zeros((dim, dim), dtype=complex)
    H_c = []
    for num_q in range(0,N_q):
        H_z = sigma_p @ sigma_m
        H_nl = (eta/2) * (sigma_p @ sigma_p @ sigma_m @ sigma_m)
        H_x = sigma_x
        H_y = sigma_y
        H_i = g * np.kron(sigma_x, sigma_x)
    
        for num_1 in range(0,num_q):
            H_z = np.kron(Id, H_z)
            H_nl = np.kron(Id, H_nl)
            H_x = np.kron(Id, H_x)
            H_y = np.kron(Id, H_y)
            if (num_q+2 <= N_q):
                H_i = np.kron(Id, H_i)
                
        for num_2 in range(num_q+1, N_q):
            H_z = np.kron(H_z, Id)
            H_nl = np.kron(H_nl, Id)
            H_x = np.kron(H_x, Id)
            H_y = np.kron(H_y, Id)
            if (num_2 >= num_q+2):
                H_i = np.kron(H_i, Id)
            
        if (num_q==N_q-1):    
            H_d += H_nl
        else:
            H_d += H_nl + H_i
    
        H_c.append(H_z)
        H_c.append(H_x)
        H_c.append(H_y)
    
    for N_c in N_c_list:
        Hc = H_c[:N_c]
        while N_c > len(Hc):
            Hc.append(Hc[-1])
            
        xi = np.min(xi) * np.ones(N_c)
        L_list, R_list, H_list = tb.HamiltonianDiagonalization(H_d, Hc, xi, mode=mode)
    
        #%% propagation
        T = M * tau
        um = np.min(xi/10) * (2*np.random.rand(N_c,M)-1)
        
        tic1 = time.perf_counter_ns()
        # tic1 = time.process_time_ns()
        for num_rep in range(0,N_rep):
            Ut, U = tb.TimeEvolution(H_list, L_list, R_list, um, xi, tau, mode=mode, histroy=False)
        toc1 = time.perf_counter_ns()
        # toc1 = time.process_time_ns()
        t_PWM.append(toc1-tic1)
        
        tic2 = time.perf_counter_ns()
        # tic2 = time.process_time_ns()
        for num_rep in range(0,N_rep):
            Ut, U = tb.Ref_TimeEvolution(H_d, Hc, um, tau)
        toc2 = time.perf_counter_ns()
        # toc2 = time.process_time_ns()
        t_PWC.append(toc2-tic2)
        
        print([N_q, N_c])

t_PWM = np.array(t_PWM)
t_PWM = t_PWM.reshape(len(N_q_list), len(N_c_list))
t_PWC = np.array(t_PWC)
t_PWC = t_PWC.reshape(len(N_q_list), len(N_c_list))

t_CPU = t_PWM/t_PWC
print([np.mean(t_CPU),np.min(t_CPU),np.max(t_CPU)])
np.save('CPU_save.npy', [N_q_list, N_c_list, t_CPU])

#%% plots
N_q_list, N_c_list, t_CPU = np.load('CPU_save.npy', allow_pickle=True)

fig = plt.figure(figsize=(3.5,2.0))
gr = plt.GridSpec(1,1, wspace=0.0, hspace=0.0)
ax3 = plt.subplot(gr[0,0])

N_q_plot = np.linspace(N_q_list[0]-0.5,N_q_list[-1]+.5, len(N_q_list)+1, dtype=float)
N_c_plot = np.linspace(N_c_list[0]-0.5,N_c_list[-1]+.5, len(N_c_list)+1, dtype=float)

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
im = ax3.pcolormesh(N_q_plot, N_c_plot, np.log10(t_CPU).transpose(), cmap='bwr', norm=norm)

ax3.plot(N_q_plot,N_q_plot, color='black', linewidth=0.5)
ax3.plot(N_q_plot,2*N_q_plot, color='black', linewidth=0.5, linestyle='--')
ax3.plot(N_q_plot,3*N_q_plot, color='black', linewidth=0.75, linestyle=':')

ax3.set_xlim(N_q_plot[0],N_q_plot[-1])
ax3.set_ylim(N_c_plot[0],N_c_plot[-1])

ax3.set_xticks([1,2,3,4,5,6,7])
ax3.set_yticks([1,5,9,13,17])
ax3.set_xlabel(r'Number of artificial atoms $N$')
ax3.set_ylabel(r'Number of control freedom $N$')

# fig.align_ylabels()
plt.savefig('Fig_CPU.pdf', bbox_inches='tight') 
plt.show()