import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6})
import sympy as sym
import matplotlib.gridspec as gridspec
from qutip import *

# ORIGINAL
amp = 1
freq = 2.0 # MHz

N = 20 # number of periods
T = 20/freq # us
M = int(20) # time intervals per period
tau = (1/freq)/M

nr = int(2E3) # points per tau
t_list = np.arange(0,T,tau/nr)
dt = t_list[1]-t_list[0]
x_t = amp * np.sin(2*np.pi*freq*t_list+np.pi/4)

fft_size = len(t_list)
f_list = np.fft.fftfreq(fft_size,dt)
x_f  = np.fft.fft(x_t)/fft_size

# AWG
t = sym.symbols('t')
y0 = []
Area_list = []
for n1 in range(0,N*M):
    ta = n1*tau
    tb = (n1+1)*tau
    t_middle = (ta+tb)/2
    
    Area = sym.Integral( amp * sym.sin(2*np.pi*freq*t+np.pi/4), (t, ta, tb)).evalf()
    Area_list.append(float(Area))
    
    for n2 in range(0, nr):
        y0.append(np.sin(t_middle))
            
y0f  = np.fft.fft(y0)/fft_size        

# 3-LEVEL PWM
t = sym.symbols('t')
y1 = []
Am = 1.0
for n1 in range(0,N*M):
    ta = n1*tau
    tb = (n1+1)*tau
    t_middle = (ta+tb)/2
    
    Area = Area_list[n1]
    t_width = Area/Am
    rio = int(t_width/tau * nr)
    for n2 in range(0, nr):
        if ( n2 >= int(nr/2 - np.abs(rio)/2) ) & ( n2 < int(nr/2 + np.abs(rio)/2) ):
            y1.append(np.sign(rio)*Am)
        else:
            y1.append(0.0)
            
y1f  = np.fft.fft(y1)/fft_size

# GAUSSIAN
t = sym.symbols('t')
y2 = np.zeros(fft_size, dtype=float)
Am = 1.0
for n1 in range(0,N*M):
    ta = n1*tau
    tb = (n1+1)*tau
    t_middle = (ta+tb)/2

    Area = Area_list[n1]
    t_width = Area/Am
    if np.abs(t_width) > 1E-8:
        y2 += np.sign(t_width)*Am * np.exp( -np.pi/(float(t_width)**2) * (t_list - t_middle)**2 ) 

y2f  = np.fft.fft(y2)/fft_size

# 2-LEVEL PWM 
t = sym.symbols('t')
y3 = []
Am = 1.0
for n1 in range(0,20*M):
    ta = n1*tau
    tb = (n1+1)*tau
    t_middle = (ta+tb)/2
    
    Area = Area_list[n1]
    t_width = Area/(2*Am) + tau/2
    rio = int(t_width/tau * nr)
    
    for n2 in range(0, nr):
        if ( n2 >= int(nr/2 - np.abs(rio)/2) ) & ( n2 < int(nr/2 + np.abs(rio)/2) ):
            y3.append(np.sign(rio)*Am)
        else:
            y3.append(-np.sign(rio)*Am)
                
y3f  = np.fft.fft(y3)/fft_size

#%% plots
Omega_c = freq * (M-1)

# schematic
fig = plt.figure(figsize=(3.5,4))
gr = plt.GridSpec(2,1, wspace=0., hspace=0.35, height_ratios=[3,4])
gr1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = gr[0])
gr2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gr[1], hspace=0.05)

ax1 = plt.subplot(gr1[0,0])
ax2 = plt.subplot(gr2[0,0])
ax3 = plt.subplot(gr2[1,0])

t_plot = t_list * 1E3
T_c = 1E3/freq

ax1.plot(t_plot,x_t,'b',label=r'$u_1(t)$', zorder=3, linewidth=1., linestyle='-.')
ax1.plot(t_plot,y1,'g',label=r'$s_1(t)$', zorder=1, linewidth=1.)
ax1.plot(t_plot,y2,'r',label=r'$s_1^{\rm G}(t)$', alpha=0.75, zorder=2, linewidth=1.25, linestyle='--')
ax1.set_xlim(0,T_c)
ax1.set_ylim(-1.1,1.1)
ax1.set_yticks(np.arange(-1.0, 1.1, step=0.5))
ax1.set_xlabel('Time, t (ns)')
# ax1.set_ylabel('Amplitude')
ax1.legend(loc='upper right')

ax2.plot(f_list, np.real(x_f),'b',label=r'$\mathcal{F}[u_1(t)]$', alpha=0.75, zorder=3, linewidth=1., linestyle='-.')
ax2.plot(f_list, np.real(y1f),'g',label=r'$\mathcal{F}[s_1(t)]$', zorder=1, linewidth=1.)
ax2.plot(f_list, np.real(y2f),'r',label=r'$\mathcal{F}[s_1^{\rm G}(t)]$', alpha=0.75, zorder=2, linewidth=1.25, linestyle='--')
ax2.set_xscale('symlog')
ax2.set_xlim(-1E3, 1E3)
ax2.set_xticks([])

ax3.plot(f_list, np.imag(x_f),'b',label=r'$\mathcal{F}[u_1(t)]$', alpha=0.75, zorder=3, linewidth=1., linestyle='-.')
ax3.plot(f_list, np.imag(y1f),'g',label=r'$\mathcal{F}[s_1(t)]$', zorder=1, linewidth=1.)
ax3.plot(f_list, np.imag(y2f),'r',label=r'$\mathcal{F}[s_1^{\rm G}(t)]$', alpha=0.75, zorder=2, linewidth=1.25, linestyle='--')
ax3.set_xscale('symlog')
ax3.set_xlim(-1E3, 1E3)
ax3.set_xlabel(r'Frequency, $\omega/2\pi$ (MHz)')

ax2.set_ylim(-0.41,0.41)
ax3.set_ylim(-0.41,0.41)

ax2.fill_between([Omega_c, 1E2*Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax2.fill_between([-1E2*Omega_c, -Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax3.fill_between([Omega_c, 1E2*Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax3.fill_between([-1E2*Omega_c, -Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)

fig.text(0.0, 0.75, 'Amplitude', va='center', rotation='vertical')
fig.text(0.0, 0.31, 'Fourier components', va='center', rotation='vertical')
fig.text(0.88, 0.46, 'Real part', ha='right') 
fig.text(0.88, 0.27, 'Imaginary part', ha='right') 

fig.align_ylabels()
plt.savefig('Fig_schematic.pdf',bbox_inches='tight')
plt.show()  

# 2-level
fig = plt.figure(figsize=(3.5,4))
gr = plt.GridSpec(2,1, wspace=0., hspace=0.35, height_ratios=[3,4])
gr1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = gr[0])
gr2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gr[1], hspace=0.05)

ax1 = plt.subplot(gr1[0,0])
ax2 = plt.subplot(gr2[0,0])
ax3 = plt.subplot(gr2[1,0])

t_plot = t_list * 1E3
T_c = 1E3/freq

ax1.plot(t_plot,x_t,'b',label=r'$u_1(t)$', zorder=3, linewidth=1., linestyle='-.')
ax1.plot(t_plot,y3,'g',label=r'$s_1(t)$', zorder=1, linewidth=1.)
ax1.set_xlim(0,T_c)
ax1.set_ylim(-1.1,1.1)
ax1.set_yticks(np.arange(-1.0, 1.1, step=0.5))
ax1.set_xlabel('Time, t (ns)')
ax1.legend(loc='upper right')

ax2.plot(f_list, np.real(x_f),'b',label=r'$\mathcal{F}[u_1(t)]$', alpha=0.75, zorder=3, linewidth=1., linestyle='-.')
ax2.plot(f_list, np.real(y3f),'g',label=r'$\mathcal{F}[s_1(t)]$', zorder=1, linewidth=1.)
ax2.set_xscale('symlog')
ax2.set_xlim(-1E3, 1E3)
ax2.set_xticks([])

ax3.plot(f_list, np.imag(x_f),'b',label=r'$\mathcal{F}[u_1(t)]$', alpha=0.75, zorder=3, linewidth=1., linestyle='-.')
ax3.plot(f_list, np.imag(y3f),'g',label=r'$\mathcal{F}[s_1(t)]$', zorder=1, linewidth=1.)
ax3.set_xscale('symlog')
ax3.set_xlim(-1E3, 1E3)
ax3.set_xlabel(r'Frequency, $\omega/2\pi$ (MHz)')

ax2.set_ylim(-0.41,0.41)
ax3.set_ylim(-0.41,0.41)

ax2.fill_between([Omega_c, 1E2*Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax2.fill_between([-1E2*Omega_c, -Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax3.fill_between([Omega_c, 1E2*Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax3.fill_between([-1E2*Omega_c, -Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)

fig.text(0.0, 0.75, 'Amplitude', va='center', rotation='vertical')
fig.text(0.0, 0.31, 'Fourier components', va='center', rotation='vertical')
fig.text(0.88, 0.46, 'Real part', ha='right') 
fig.text(0.88, 0.27, 'Imaginary part', ha='right') 

fig.align_ylabels()
plt.savefig('Fig_two_level.pdf',bbox_inches='tight')
plt.show() 

# Gaussian
fig = plt.figure(figsize=(3.5,4))
gr = plt.GridSpec(2,1, wspace=0., hspace=0.35, height_ratios=[3,4])
gr1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = gr[0])
gr2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gr[1], hspace=0.05)

ax1 = plt.subplot(gr1[0,0])
ax2 = plt.subplot(gr2[0,0])
ax3 = plt.subplot(gr2[1,0])

t_plot = t_list * 1E3
T_c = 1E3/freq

ax1.plot(t_plot,x_t,'b',label=r'$u_1(t)$', zorder=3, linewidth=1., linestyle='-.')
ax1.plot(t_plot,y2,'g',label=r'$s_1(t)$', zorder=1, linewidth=1.)
ax1.set_xlim(0,T_c)
ax1.set_ylim(-1.1,1.1)
ax1.set_yticks(np.arange(-1.0, 1.1, step=0.5))
ax1.set_xlabel('Time, t (ns)')
ax1.legend(loc='upper right')

ax2.plot(f_list, np.real(x_f),'b',label=r'$\mathcal{F}[u_1(t)]$', alpha=0.75, zorder=3, linewidth=1., linestyle='-.')
ax2.plot(f_list, np.real(y2f),'g',label=r'$\mathcal{F}[s_1(t)]$', zorder=1, linewidth=1.)
ax2.set_xscale('symlog')
ax2.set_xlim(-1E3, 1E3)
ax2.set_xticks([])

ax3.plot(f_list, np.imag(x_f),'b',label=r'$\mathcal{F}[u_1(t)]$', alpha=0.75, zorder=3, linewidth=1., linestyle='-.')
ax3.plot(f_list, np.imag(y2f),'g',label=r'$\mathcal{F}[s_1(t)]$', zorder=1, linewidth=1.)
ax3.set_xscale('symlog')
ax3.set_xlim(-1E3, 1E3)
ax3.set_xlabel(r'Frequency, $\omega/2\pi$ (MHz)')

ax2.set_ylim(-0.41,0.41)
ax3.set_ylim(-0.41,0.41)

ax2.fill_between([Omega_c, 1E2*Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax2.fill_between([-1E2*Omega_c, -Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax3.fill_between([Omega_c, 1E2*Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)
ax3.fill_between([-1E2*Omega_c, -Omega_c],-1.1,1.1, facecolor="grey", alpha=0.5)

fig.text(0.0, 0.75, 'Amplitude', va='center', rotation='vertical')
fig.text(0.0, 0.31, 'Fourier components', va='center', rotation='vertical')
fig.text(0.88, 0.46, 'Real part', ha='right') 
fig.text(0.88, 0.27, 'Imaginary part', ha='right') 

fig.align_ylabels()
plt.savefig('Fig_gaussian.pdf',bbox_inches='tight')
plt.show()  