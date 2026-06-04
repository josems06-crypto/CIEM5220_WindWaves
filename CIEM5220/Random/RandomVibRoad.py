# ================================================
#         CIEM5220 - Random Vibrations
# Moving vehicle on random road profile (ISO 8608)

# ================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from fem_tools import xx, xx, xx

# __ Vehicle parameters ___________________________________________

M= 1500.0               # mass [kg]
K= 29600.0              # suspension stiffness per axle [N/m]
C= 1500.0               # suspension damping per axle [N*s/m]
V= 25.0                 # Speed [m/s] (~90 km/h)
L= 3                    # Wheelbase [m]

J=3000.0                # Moment of inertia [kg*m^2]

omega_0=np.sqrt(2*K/M)
f_0=omega_0/(2*np.pi)
k_res=omega_0/V

print(f"Natural frequency: {omega_0:.3f} rad/s={f_0:.3f} Hz")
print(f"Resonance wave-number: {k_res:.3f} rad/m")

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

#__ Response PSD and variance _________________________________________________

S_w= S_w0*np.abs(H_k)**2               # [m^2/(rad/m)]     

sigma2_w=(1/np.pi)*np.trapezoid(S_w,k) # [m^2]
sigma_w=np.sqrt(sigma2_w)              # [m] multiply by 1000 for mm

print(f"sigma_w={sigma_w*1000:.2f}mm")

#__ Time domain response_______________________________________________________
t=np.linspace(0,5000,100000) # 100 seconds, 10 000 points
w_t=np.zeros(len(t))

for i in range(len(k)):
    omega_i = k[i]*V
    amp_i= F_amp[i]*np.abs(H_k[i])
    phase_i=phi[i]+np.angle(H_k[i])
    w_t+=amp_i*np.cos(omega_i*t+phase_i)

print(f"\nTime domain response:")
print(f"max|w(t)|={np.max(np.abs(w_t))*1000:.2f}mm")
print(f" std w(t)= {np.std(w_t)*1000:.2f} mm")
print(f" PSD sigma= {sigma_w*1000:.2f} mm <-- should match std")

amps=F_amp*np.abs(H_k)
sigma_direct=np.sqrt(0.5*np.sum(amps**2))
print(f"Direct sum sigma= {sigma_direct*1000:.2f} mm")

#__ Speed sweep  _______________________________________________________________

print("\nSpeed sweep:")
for V_i in [10, 25, 40, 100, 200]:
    H_i = H_sdof(k, V_i)
    S_w_i= S_w0*np.abs(H_i)**2
    sigma_i=np.sqrt((1/np.pi)*np.trapezoid(S_w_i, k))
    print(f" V={V_i:>4} m/s -> sigma={sigma_i*1000:.2f} mm")

sigmas=[]
v_range=np.linspace(5, 300, 200)
for V_i in v_range:
    H_i= H_sdof(k, V_i)
    S_w_i= S_w0*np.abs(H_i)**2
    sigma_i=np.sqrt((1/np.pi)*np.trapezoid(S_w_i, k))
    sigmas.append(sigma_i*1000)

v_peak=v_range[np.argmax(sigmas)]
print(f"Worst speed: {v_peak:.1f} m/s = {v_peak*3.6:.1f} km/h")
print(f"Peak sigma: {max(sigmas):.2f} mm")

# __ 2-DOF ________________________________________________________________

def H_2dof(k, V):
    omega=k*V
    K_bar=K+1j*omega*C

    a11=-M*V**2 * k**2 + 2*K_bar
    a22=-J*V**2 * k**2 + L**2*K_bar/2
    b1=K_bar*(1+np.exp(1j*k*L))
    b2=L*K_bar*(1-np.exp(1j*k*L))/2
    H_w=b1/a11
    H_theta=b2/a22
    return H_w, H_theta

H_w, H_theta=H_2dof(k,V)

S_w_2= S_w0*np.abs(H_w)**2
S_w_theta=S_w0*np.abs(H_theta)**2

sigma_2=np.sqrt((1/np.pi)*np.trapezoid(S_w_2, k))
sigma_theta=np.sqrt((1/np.pi)*np.trapezoid(S_w_theta, k))

print(f"max |a11| = {np.max(np.abs(-M*V**2*k**2+2*(K+1j*k*V*C))):.2f}")
print(f"max |a22| = {np.max(np.abs(-J*V**2*k**2+(L**2/2)*(K+1j*k*V*C))):.4f}")
print(f"max |b1| = {np.max(np.abs((K+1j*k*V*C)*(1+np.exp(1j*k*L)))):.2f}")
print(f"max |b2| = {np.max(np.abs((L/2)*(K+1j*k*V*C)*(1-np.exp(1j*k*L)))):.4f}")

print(f"Bounce sigma={sigma_2*1000:.2f} mm")
print(f"Pitch sigma={sigma_theta*1000:.4f} mrad")
print(f"Pitch sigma in degrees={np.degrees(sigma_theta):.4f} deg")

#__ Acceleration at the driver seat__________________________________________

omega_arr = k * V
H_seat = H_w + (L/4) * H_theta

S_a= omega_arr**4*S_w0*np.abs(H_seat)**2
sigma_a=np.sqrt((1/np.pi)*np.trapezoid(S_a, k))

print(f"Driver acceleration sigma={sigma_a:4f} m/s^2")

#__ Probability of exceedence _______________________________________________

a_crit=0.5       #[m/s^2] ISO 2631 comfort threshold

P_exceed=0.5*(1-erf(a_crit/(sigma_a*np.sqrt(2))))

print(f"Critical acceleration: {a_crit} m/s^2")
print(f"P(a>a_crit)={P_exceed:.4f} ({P_exceed*100:.2f}%)")

print("\nExceedance probabilities:")

for a_c in [0.315, 0.5, 1.0, 2.0]:
    P=0.5*(1-erf(a_c/(sigma_a*np.sqrt(2))))
    print(f" P(a > {a_c:.3f} m/s^2)={P*100:.2f}%")