import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin

def HamiltonianDiagonalization(H0, Hc, xi, mode='PWM'):
    if (mode=='Trotter'):
        H_list = {'0': H0}
        for num_1 in range(0,len(xi)):
            H_list.update({'+'+str(num_1+1): xi[num_1]*Hc[num_1]})
            H_list.update({'-'+str(num_1+1): -xi[num_1]*Hc[num_1]})
    else:
        H_list = {'0': H0}
        for num_1 in range(0,len(xi)):
            H_list2 = {}
            for Hc_key, Hc_value in H_list.items():
                H_list2.update({Hc_key+'+'+str(num_1+1): Hc_value + xi[num_1]*Hc[num_1]})
                H_list2.update({Hc_key+'-'+str(num_1+1): Hc_value - xi[num_1]*Hc[num_1]})
            H_list.update( H_list2 )
        
    L_list3 = {}
    R_list3 = {}
    H_list3 = {}
    for H_key, H_value in H_list.items():
        Hs, Ls = lin.eigh(H_value)
        L_list3.update({H_key:Ls})
        R_list3.update({H_key:Ls.conjugate().transpose()})
        H_list3.update({H_key:Hs})
    return [L_list3, R_list3, H_list3]

def TimeAssignment(um, xi, tau, mode='PWM'):
    t_list = {}
    for num_1 in range(0,len(um)):
        tm = um[num_1]*tau/xi[num_1]
        if np.sign(tm) > 0:
            t_list.update({'+'+str(num_1+1):np.abs(tm)})
        elif np.sign(tm) < 0:
            t_list.update({'-'+str(num_1+1):np.abs(tm)})
    
    if (mode=='Trotter'):
        t_list2 = {'0':tau/2} 
        while len(t_list) != 0:
            t_max = max(t_list.values())
            key_max = max(t_list, key=t_list.get)
            t_list2.update({key_max: t_max/2})
            del t_list[key_max]
    else:
        t_list2 = {} 
        t_max = tau
        name = '0'
        name_list = {'0':0}
        while len(t_list) != 0:
            t_max_new = max(t_list.values())
            t_list2.update({name: (t_max-t_max_new)/2})
            
            t_max = t_max_new
            key_max =  max(t_list, key=t_list.get)
            
            name_list.update({key_max: np.abs(int(key_max))})
            name = ''.join( sorted(name_list, key=name_list.get) )
            del t_list[key_max]
        t_list2.update({name: t_max/2})
    return t_list2
    
def PixelEvolution(H_list, L_list, R_list, um, xi, tau, mode='PWM'):
    t_list = TimeAssignment(um, xi, tau, mode=mode)
    Up = L_list['0'] @ np.diag(np.exp(-1j*H_list['0']*t_list['0'])) @ R_list['0']
    Um1 = np.copy(Up)
    Um2 = np.copy(Up)
    del t_list['0']
    
    for t_key, t_value in t_list.items():
        Up = L_list[t_key] @ np.diag(np.exp(-1j*H_list[t_key]*t_value)) @ R_list[t_key]
        Um1 = Um1 @ Up
        Um2 = Up @ Um2
    return (Um1 @ Um2)
 
def TimeEvolution(H_list, L_list, R_list, u, xi, tau, mode='PWM', histroy=True):
    U = PixelEvolution(H_list, L_list, R_list, np.array(u)[:,0], xi, tau, mode=mode)
    if histroy:
        U_list = [U]

    for num_m in range(1,len(np.array(u)[0])):
        Um = PixelEvolution(H_list, L_list, R_list, np.array(u)[:,num_m], xi, tau, mode=mode)
        U = Um @ U
        if histroy:
            U_list.append(Um)

    if histroy:
        U_out = [U, U_list]
    else:
        U_out = [U, 'No Histroy']
    return U_out

#%% reference 
def Ref_PixelEvolution(H0, Hc, um, tau):
    H = H0
    for num_1 in range(0, len(um)):
        H = H + um[num_1]*Hc[num_1]
    Um = lin.expm( -1j*tau*H )
    return Um

def Ref_TimeEvolution(H0, Hc, u, tau, histroy=True):
    U = Ref_PixelEvolution(H0, Hc, np.array(u)[:,0], tau)
    if histroy:
        U_list = [U]
        
    for num_m in range(1,len(np.array(u)[0])):
        Um = Ref_PixelEvolution(H0, Hc, np.array(u)[:,num_m], tau)
        U = Um @ U
        
        if histroy:
            U_list.append(Um)
    
    if histroy:
        U_out = [U, U_list]
    else:
        U_out = [U, 'No Histroy']    
    return U_out
    
#%% pulse width -> waveform, pulse train
def u2s(u, xi, tau, M):
    tau_list = u/xi
    w_list = np.abs(tau_list)
    sgn_list = np.sign(tau_list)
    
    tp = []
    tm = []
    for m in range(0,M):
        tp.append((m+1/2)*tau - w_list[m]/2)
        tm.append((m+1/2)*tau + w_list[m]/2)
     
    t_list = np.linspace(0,tau*M,10000)
    s_list = np.zeros(10000)
    u_list = np.zeros(10000)
    num_2 = 0
    num_3 = 0
    for num_1 in range(0,10000):
        if (t_list[num_1] >= tp[num_2]):
            if (t_list[num_1] < tm[num_2]):
                s_list[num_1] = xi*sgn_list[num_2]
            if (t_list[num_1] >= tm[num_2]) and (num_2<M-2):
                num_2 += 1
        
        if (t_list[num_1] >= tau*num_3):
            u_list[num_1] = u[num_3]
            if (t_list[num_1] >= tau*(num_3+1)):
                num_3 += 1
        
    return [t_list, s_list, u_list]
