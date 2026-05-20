# ================================================
#         CIEM5220 - Random Vibrations
#   I.   Stationary proof (KT spectrum)
#   II.  Bridge Prelude under eartquake excitation
#       - Travelling wave input (Kanai, Tajimi)
#       - Displacement transformation
#       - Modal analysis & response variance
#   III. Moving vehicle & driver comfort (ISO 2631)
#
#   Scenario: Canyon bridge, Atama desert, Chile
#   Seismic input: Kanai- Tajimi / 2007 Tocopilla
#   Reference: Metrikine, CIEM5220 Random L5-L7
#
# =====================================================
import numpy as np
import matplotlib.pyplot as plt
#
#__ Frequency axis ____________________________________

omega = np.linspace(0.1, 50, 2000) # [rad/s]
dw=omega[1]-omega[0]

#__ Kanai-Tajimi parameters (soft Atacama alluvium)___

omega_g = 6.0     # soil filter frequency [rad/s] (-1 Hz)
zeta_g = 0.30     # soil damping ratio [-]
S0= 0.02          # white noice intesity of the bedrock excitation

def S_KT(S0, w_g, Chi_g, w):
    num=w_g**4+4*Chi_g**2*w_g**2*w**2
    den=(w_g**2-w**2)**2+4*Chi_g**2*w_g**2*w**2
    return S0*num/den

S_KT_vals=S_KT(S0, omega_g, zeta_g, omega)

#__ Amplitudes (deterministic, from PSD) ______________

F_amp=np.sqrt(2*dw*S_KT_vals)

#plt.plot(omega, S_KT_vals)
#plt.axvline(omega_g, color='r', ls='--')
#plt.show()

rng=np.random.default_rng(seed=42)
phi=rng.uniform(0,2*np.pi, size=len(omega))

t=np.linspace(0, 30, 5000)
a_g=np.zeros(len(t)) 

for i in range(len(omega)):
    a_g+=F_amp[i]*np.cos(omega[i]*t+phi[i]) 
    # WHere a_g is ground acceleration ü_g(t)

#plt.plot(t, a_g)
#plt.show()

# statistical check
sigma_theoretical=np.sqrt(np.trapezoid(S_KT_vals, omega))
sigma_signal=np.std(a_g)

print(f"Theoretical sigma: {sigma_theoretical:.3f} m/s^2")
print(f"Signal sigma: {sigma_signal:.3f} m/s^2")

sigma_direct=np.sqrt(0.5*np.sum(F_amp**2))
print(f"Direct from amplitudes: {sigma_direct:.3f} m/s^2")

# To make sure signal sigma converges towards theoretical
for T in [30, 100, 500, 2000]:
    t_test=np.linspace(0,T, T*100)
    a_test=np.zeros(len(t_test))
    for i in range(len(omega)):
        a_test+= F_amp[i]*np.cos(omega[i]*t_test+phi[i])
    print(f"T={T:>5}s: sigma={np.std(a_test):.4f} m/s^2")


# Rayleigh wave velocity for typical rock/soil is roughly 500-3000 m/s. 
# For the Atacama alluvium a value is around c_R= 800 m/s
# With a bridge 200 m that gives a time delay of:
# dt=200/800=0.25 seconds (for the support further away)

#
#         ===============================
#         T                             T
#       support A                   support B
#        x=0                           x=L
#
#  - - (--(((*)))--) - - - > - - - > - - - - - - - - - - 
#     Eartquake source.    wave travelling at c_R
# wave hits support A first, wave hits support b at dt= L/c_R later
#
# If G is shear modulus and rho density
# c_R≈0.92*sqrt(G/rho)  
# => high G (e.g rock)-> fast wave
# => low G (e.g mud)-> slow wave

#__ Bridge & wave parameters___________________________________

L_b= 200.0       # Bridge span [m]
c_R=200.0        # Rayleigh wave velocity [m/s] - stiff Atacama alluvium
EI= 4e10         # Flexural stiffness [N/m^2]
rho_A=2500       # Mass per unit length [kg/m]

#__ Time delay between supports ________________________________

dt_wave=L_b/c_R
print(f"Wave travel time across bridge: {dt_wave:.3f} s")
print(f"Phase shift at omega_g: {omega_g*dt_wave:.3f} rad")

#__ Support signals (travelling wave)__________
# w_e(0,t)-- left support, wave arrives first
# w_e(L,t)-- right support, delayed by L/c_R

w_left=np.zeros(len(t))
w_right=np.zeros(len(t))

for i in range(len(omega)):
    # left support: no delay
    w_left+=F_amp[i]*np.cos(omega[i]*t+phi[i])

    # right support: phase shifted by omega*L/c_R
    phase_delay= omega[i]*L_b/c_R
    w_right+=F_amp[i]*np.cos(omega[i]*t+phi[i]-phase_delay)

print(f"Max w_left: {np.max(np.abs(w_left))*1000:.1f} mm")
print(f"Max w_right: {np.max(np.abs(w_right))*1000:.1f} mm")

x=np.linspace(0, L_b, 200)   # spatial points along bridge

# reshape for broadcasting
# w_left and w_right are (5000,) -- time
# x terms are (200,) -- space
# result will be (200, 5000)

w_tilde_bc=(w_left[np.newaxis, :]*((L-x[:,np.newaxis])/L_b)+w_right[np.newaxis,:]*(x[:, np.newaxis]/L_b))
