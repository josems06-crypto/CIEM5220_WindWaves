# ================================================
#         CIEM5220 - Random Vibrations
# Moving vehicle on random road profile (ISO 8608)

# ================================================

# focus here became the plotting 
# see RandomVibRoad for focus on physics

import numpy as np
#import sympy as sp
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from fem_tools import xx, xx, xx

# __ Vehicle parameters ___________________________________________

M= 1500.0               # mass [kg]
K= 20000.0              # suspension stiffness per acle [N/m]
C= 1500.0               # suspension damping per axle [N*s/m]
V= 25                           # Speed [m/s] (~90 km/h)
L= 3                            # Wheelbase [m]

#(speed sweep)

speeds=[10,25,40]  
colors=['blue', 'orange', 'green']

# (to zoom in on plot)
omega_0=np.sqrt(2*K/M)
k_res=omega_0/V
print(f"Natural frequency: {omega_0:.2f} rad/s= {omega_0/(2*np.pi):.2f} Hz")
print(f"Resonance at k={k_res:.3f} rad/m")

# __ Road PSd (ISO 8608) __________________________________________

Gd=64e-6                        #  [m^3/rad]
k0=1.0                          # Reference wave-number               

k=np.linspace(0.05, 50, 1000)                     # rad/m
dk=k[1]-k[0] 

S_w0=Gd*(k0/k)**2
F_amp=np.sqrt(2*dk*S_w0/np.pi)

# __ Road realization _____________________________________________

rng=np.random.default_rng(seed=42)
phi=rng.uniform(0, 2*np.pi, size=len(k))
x=np.linspace(0, 500, 5000)
w0=np.zeros(len(x)) 

for i in range(len(k)):
    w0+=F_amp[i]*np.cos(k[i]*x+phi[i])

#__ Transfer function _____________________________________________

def H_sdof(k, V):
    omega = k*V                     # temporal frequency [rad/m * m/s = rad/s]
    K_bar=K+1j*omega*C              # complex stiffness  [N/m]
    num = K_bar*(1+ np.exp(1j*k*L)) # [N/m] - road input amplitude
    den= -M * V**2 * k**2 + 2*K + 2j*omega*C # [N/m] - dynamic stiffness
    return num/den
H_k=H_sdof(k, V)

#__ Response PSD and variance _____________________________________

S_w= S_w0*np.abs(H_k)**2               # [m^2/(rad/m)]     

sigma2_w=(V/np.pi)*np.trapezoid(S_w,k) # [m^2]
sigma_w=np.sqrt(sigma2_w)              # [m] multiply by 1000 for mm

print(f"Response std dev: sigma_w= {sigma_w*1000:.2f} mm")
print(f"Response variance: zigma^2 = {sigma2_w*1e6:.2f} mm^2")


#__ Plotting ______________________________________________________



fig, axes=plt.subplots(4,1,figsize=(7,6))

# Top: road profile
axes[0].plot(x, w0*1000)
axes[0].set_xlim(0,100)
axes[0].set_xlabel("Position x [m]")
axes[0].set_ylabel("Road elevation w0 [mm]")
axes[0].set_title("Road profile realization - ISO 8608 Class B")
axes[0].grid(True)

k_plot=np.logspace(np.log10(0.05), np.log10(50), 1000) # for the loglog
axes[1].loglog(k_plot, S_w0)
axes[1].set_xlabel("Wave-number k [rad/m]")
axes[1].set_ylabel("PSD S_w0 [m^2/(rad/m)]")
axes[1].set_title("Road roughness PSD - ISO 8608")
axes[1].grid(True, which="both")
axes[1].set_xlim(0,5)
axes[1].axvline(k_res, color='red', linestyle='--', linewidth=1.2,
           label=f"Resonance k={k_res:.2f} rad/m")
axes[1].legend()

axes[2].semilogy(k, np.abs(H_k)**2)
axes[2].set_xlabel("Wave-number k [rad/m]")
axes[2].set_ylabel("|H(k)|^2 [-]")
axes[2].set_title("SDOF Transfer Function |H(k)|^2")
axes[2].grid(True,which="both")
axes[2].set_xlim(0,5)
axes[2].axvline(k_res, color='red', linestyle='--', linewidth=1.2,
           label=f"Resonance k={k_res:.2f} rad/m")
axes[2].legend()

#axes[3].semilogy(k, S_w0, label="Road input S_w0")
axes[3].semilogy(k, S_w0, 'k--', label="Road input S_w0") #speed sweep
for V_i, col in zip(speeds, colors):
    omega_0_1=np.sqrt(2*K/M)
    k_res_i=omega_0_1/V_i
    H_i = H_sdof(k, V_i)
    S_w_i = S_w0*np.abs(H_i)**2
    sigma_i = np.sqrt((V_i/np.pi)*np.trapezoid(S_w_i, k))
    axes[3].semilogy(k, S_w_i, color=col,
                 label=f"V={V_i} m/s, zigma={sigma_i*1000:.1f} mm")
    axes[3].axvline(k_res_i, color=col, linestyle=':', linewidth=1)

axes[3].semilogy(k, S_w, label="Vehicle response S_w")
axes[3].set_xlabel("Wave-number k [rad/m]")
axes[3].set_ylabel("PSD [m^2/(rad/m)]")
axes[3].set_title("Step 4 - Response PSD")
axes[3].legend()
axes[3].grid(True, which="both")
axes[3].set_xlim(0,5)
#axes[3].axvline(k_res, color='red', linestyle='--', linewidth=1.2,
           #label=f"Resonance k={k_res:.2f} rad/m")
axes[3].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
