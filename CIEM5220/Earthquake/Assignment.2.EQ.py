# ================================================
#         CIEM5220 - Seismology
# Seismic waves through subsoil on structure
# ================================================

import numpy as np
import matplotlib.pyplot as plt

A=6
B=6
C=1
D=1
E=2
F=5
G=7

M=850*1e3          # kg     ## WHERE F=5 (8F0)
J=300*1e6          # kg m2
rho_t=7850         # kg/m3
Et=210e9           # Pa
epsylon=0.001 
D_out_1=9.1        # m
D_inner_1=8.96     # m
D_out_2=7.1        # m
D_inner_2=7.0      # m
F_w=2e5            # N
H_t= 140+A         # m
H_w=30+F           # m
H_1=2+B            # m
H_2=58+G           # m
H_3=38+F           # m
G_1=150e6          # Pa     ## WHERE F=5 (1F0)
rho_1=1650         # kg/m3 
eta_1=1010         # Ns/m2  ## WHERE C=1 (10C0)
G_2=110e6          # Pa     ## WHERE D=1 (1D0)
rho_2=1850         # kg/m3
eta_2=1020         # Ns/m2  ## WHERE E=2 (10E0)
G_3=120e6          # Pa     ## WHERE E=2 (1E0)
rho_3=1900         # kg/m3
eta_3=1010         # Ns/m2  ## WHERE D=1 (10D0)

# Wave speeds, where stiffer=faster, denser=slower

c_s_1=np.sqrt(G_1/rho_1) # stiff, light
c_s_2=np.sqrt(G_2/rho_2) # softer, dense      danger
c_s_3=np.sqrt(G_3/rho_3) # soft, denser       danger

# A soft, dense soil has a very slow wave speed.
# The low G means the particles slide easily 
# high rho becuase its saturated with water
# waves crawl through it, energy gets trapped 
# => layer resonates violently

# When a wave travels from fast rock into slow soft soil
# its lika car hitting mud. But energy must be conserved
# so if the wave slows down but the enrgy has nowhere to go
# the amplitude has to increase. 


#======== EQUATION OF MOTION ==== single layer, no damping

#rho*d2u/dt2-G*d2u/dz2=0
# Divide by rho: d2u/dt2=cs^2*d2u/dz2
# This is the wave equation. cs is the wave speed

# Assume harmonic solution: u(z, t)=U(z)*exp(i*omega*t)
# Substituting: -omega^2*U=cs^2*d2U/dz2
# rearranging: d2U/dz2+k^2*U=0
# where k= omega/cs (the wavenumber)

# General solution: U(z)=A1*exp(-i*k*z)+A2*exp(+i*k*z)
# These are two waves: one going DOWN, one going UP

#======== NATURAL FREQUENCIES ===== each layer independently
# For a single layer: free surface top, fixed base
# fm = (2m-1)*cs/(4H)
# m= 1, 2, 3, ...
# This is the quarter- wavelength resonance condition

modes=np.arange(1, 11) # first 10 modes

f_layer1=(2*modes-1)*c_s_1/(4*H_1)
f_layer2=(2*modes-1)*c_s_2/(4*H_2)
f_layer3=(2*modes-1)*c_s_3/(4*H_3)

print("Layer 1 first 3 natural frequencies (Hz):", f_layer1[:3])
print("Layer 2 first 3 natural frequencies (Hz):", f_layer2[:3])
print("Layer 3 first 3 natural frequencies (Hz):", f_layer3[:3])

# Layer 1 first mode: 9.4 Hz - very high frequency, stiff and thin (8m)
# Layer 2 first mode: 0.94 Hz - much lower, thick layer (65m)
# Layer 3 first mode: 1.46 Hz - similar range, also thich (43m)

print(f"cs1={c_s_1:.2f} m/s")
print(f"cs2={c_s_2:.2f} m/s")
print(f"cs3={c_s_3:.2f} m/s")

modes=np.arange(1,11)
z1=np.linspace(0,H_1, 200) # depth array for layer 1

fig, axes= plt.subplots(2, 5, figsize=(15, 8))
axes=axes.flatten()

for i, m in enumerate(modes):
    k_m=(2*m-1)*np.pi/(2*H_1)  # wavenumber for layer 1
    U_m=np.cos(k_m*z1)         # mode shape
    axes[i].plot(U_m, -z1)     # negative z so depth goes downward
    axes[i].axvline(0,color='k', lw=0.5)
    axes[i].set_title(f"Mode{m}")
    axes[i].set_xlabel("U(norm.)")

plt.suptitle("Mode shapes - Layer 1")
plt.tight_layout()
plt.show()

def build_matrix(omega):

    # complex wave speeds including damping
    cs1_star=np.sqrt((G_1+1j*omega*eta_1)/rho_1)
    cs2_star=np.sqrt((G_2+1j*omega*eta_2)/rho_2)
    cs3_star=np.sqrt((G_3+1j*omega*eta_3)/rho_3)

    #k1=omega/c_s_1
    #k2=omega/c_s_2
    #k3=omega/c_s_3

    k1=omega/cs1_star
    k2=omega/cs2_star
    k3=omega/cs3_star

    M=np.zeros((5,5), dtype=complex)

    # Row 0: displacement continuity at interface 1-2
    # 2A1*cos(k1*H1)=A2+B2
    M[0,0]=2*np.cos(k1*H_1) # coefficient of A1
    M[0,1]=-1 # coefficient of A2
    M[0,2]=-1 # coefficient of B2

    # Row 1: stress continuity at interface 1-2
    # -2A1*G1*k1*sin(k1H1)+ik2*G2*A2-ik2*G2*B2=0
    M[1,0]=-2*G_1*k1*np.sin(k1*H_1)
    M[1,1]=1j*k2*G_2
    M[1,2]=-1j*k2*G_2

    # Row 2: displacement continuity at interface 2-3
    M[2,1]=np.exp(-1j*k2*H_2)
    M[2,2]=np.exp(1j*k2*H_2)
    M[2,3]=-1
    M[2,4]=-1

    # Row 3: stress continuity at interface 2-3
    M[3,1]=-1j*k2*G_2*np.exp(-1j*k2*H_2)
    M[3,2]=1j*k2*G_2*np.exp( 1j*k2*H_2)
    M[3,3]=1j*k3*G_3
    M[3,4]=-1j*k3*G_3

    # Row 4: bedrock - zero displacement
    M[4, 3]=np.exp(-1j*k3*H_3)
    M[4, 4]=np.exp(1j*k3*H_3)

    return M

omega_scan=np.linspace(0.1,200,50000)
det_vals=np.array([np.abs(np.linalg.det(build_matrix(w)))
                   for w in omega_scan])
plt.plot(omega_scan, det_vals)
plt.yscale('log')
plt.xlabel('omega (rad/s)')
plt.ylabel('|det(M)')
plt.title('Determinant scan - minima are natural frequencies')
plt.grid(True)
plt.show()

# Find sign changes = zero crossings = natural frequencies
det_real=np.real(np.array([np.linalg.det(build_matrix(w))
                           for w in omega_scan]))

sign_changes=np.where(np.diff(np.sign(det_real)))[0]

nat_freqs_rad=[]
for idx in sign_changes:
    lo, hi= omega_scan[idx], omega_scan[idx+1]
    # bisection to refine
    for _ in range(50):
        mid=0.5*(lo+hi)
        if np.real(np.linalg.det(build_matrix(lo)))*\
           np.real(np.linalg.det(build_matrix(mid)))<0:
            hi=mid
        else:
            lo=mid
    nat_freqs_rad.append(0.5*(lo+hi))

nat_freqs_rad=np.array(nat_freqs_rad[:10])
nat_freqs_hz=nat_freqs_rad/(2*np.pi)

print("Natural frequencies (rad/s):", np.round(nat_freqs_rad, 3))
print("Natural frequencies (Hz):", np.round(nat_freqs_hz, 3))

# The matrix equation, has a non trivial solution
# meaning the soil acutally vibrates
# only when the matrix M is singular
# i.e cannot be inverted, i.e det(M)=0
# so scanning for frequencies where free vibration is possible

# when converting first freq to Hz, we get 3.42/2pi=0.544 
# close to layer 2s individual freq of 0.928 Hz
# layer 2 dominates because its the thickest layer

n=100 # points per layer

# local coordinates
z1=np.linspace(0,H_1,n)
z2=np.linspace(0,H_2,n)
z3=np.linspace(0,H_3,n)

# global depth for plotting
z1_global=z1
z2_global=z2+H_1
z3_global=z3+H_1+H_2

z_global=np.concatenate([z1_global, z2_global, z3_global])

omega_n=nat_freqs_rad[0]

fig, axes=plt.subplots(2, 5, figsize=(16,9))
axes=axes.flatten()

for i, omega_n in enumerate(nat_freqs_rad):
    k1=omega_n/c_s_1
    k2=omega_n/c_s_2
    k3=omega_n/c_s_3

    M=build_matrix(omega_n)

    # RHS= -M[:,0]*1 (A1=1)
    rhs=-M[1:,0] 
    lhs=M[1:,1:]

    coeffs=np.linalg.solve(lhs, rhs)
    A1=1
    A2, B2, A3, B3=coeffs

    U1=np.real(2*A1*np.cos(k1*z1))
    U2=np.real(A2*np.exp(-1j*k2*z2)+B2*np.exp(1j*k2*z2))
    U3=np.real(A3*np.exp(-1j*k3*z3)+B3*np.exp(1j*k3*z3))

    U=np.concatenate([U1, U2, U3])
    U=U/np.max(np.abs(U)) # normalise

    axes[i].plot(U, -z_global, 'b', lw=1.5)
    axes[i].axvline(0, color='k', lw=0.5)
    axes[i].axhline(-H_1, color='r', lw=0.8, ls='--')
    axes[i].axhline(-(H_1+H_2), color='g', lw=0.8, ls='--')
    axes[i].set_title(f"Mode {i+1}\nf ={nat_freqs_hz[i]:.3f} Hz")
    axes[i].set_xlabel("U (norm)")
    if i==0:
        axes[i].set_ylabel('Depth(m)')
    axes[i].grid(True, alpha=0.3)

plt.suptitle("First 10 mode shapes - 3 layer soil system", fontsize=13)
plt.tight_layout()
plt.show()

def compute_FRF(omega):
    M=build_matrix(omega)
    rhs=np.array([0,0,0,0,1], dtype=complex)

    x=np.linalg.solve(M, rhs)
    A1=x[0]
    A2=x[1]
    B2=x[2]
    A3=x[3]
    B3=x[4]

    return A1, A2, B2, A3, B3

omega_arr=np.linspace(0.1, 150, 5000)
H_surface=[]

for omega in omega_arr:
    k1=omega/c_s_1
    A1, A2, B2, A3, B3=compute_FRF(omega)
    U_surface=2*A1*np.cos(k1*0) # z=0 at surface
    H_surface.append(U_surface)
H_surface=np.array(H_surface)

H_layer3=[]

for omega in omega_arr:
    k2=omega/np.sqrt((G_2+1j*omega*eta_2)/rho_2)
    A1, A2, B2, A3, B3=compute_FRF(omega)
    U_top3=A2*np.exp(-1j*k2*H_2)+B2*np.exp(1j*k2*H_2)
    H_layer3.append(U_top3)
H_layer3=np.array(H_layer3)


fig, ax = plt.subplots(2, 2, figsize=(14, 8))
# Amplitude

ax[0,0].plot(omega_arr/(2*np.pi), np.abs(H_surface))
ax[0,0].set_title('|H|-Top of Layer 1 (surface)')
ax[0,0].set_xlabel('f (Hz)')
ax[0,0].set_ylabel('|H| (-)')
ax[0,0].set_yscale('log') 
ax[0,0].grid(True)
for fn in nat_freqs_hz:
    ax[0,0].axvline(fn, color='r', lw=0.5, ls='--', alpha=0.5)

ax[0,1].plot(omega_arr/(2*np.pi), np.abs(H_layer3))
ax[0,1].set_title('|H|-Top of Layer 3 (z=73)')
ax[0,1].set_xlabel('f(Hz)')
ax[0,1].set_yscale('log')
ax[0,1].set_ylabel('|H| (-)')
ax[0,1].grid(True)
for fn in nat_freqs_hz:
    ax[0,1].axvline(fn, color='r', lw=0.5, ls='--', alpha=0.5)

# Phases

ax[1,0].plot(omega_arr/(2*np.pi), np.angle(H_surface, deg=True))
ax[1,0].set_title('Phase - Top of Layer 1 (surface)')
ax[1,0].set_xlabel('f (Hz)')
ax[1,0].set_ylabel('Phase (degrees)')
ax[1,0].grid(True)

ax[1,1].plot(omega_arr/(2*np.pi), np.angle(H_surface, deg=True))
ax[1,1].set_title('Phase -Top of Layer 3 (surface)')
ax[1,1].set_xlabel('f (Hz)')
ax[1,1].set_ylabel('Phase (degrees)')
ax[1,1].grid(True)

plt.suptitle('Transfer Function H(z,w) - two locations', fontsize=13)
plt.tight_layout()
plt.show()

print(f"omega range:{omega_arr[0]:.2f} to {omega_arr[-1]:2f} rad/s")
print(f"frequency range: {omega_arr[0]/(2*np.pi):.2f} to {omega_arr[-1]/(2*np.pi):.2f} Hz")
print(f"number of point: {len(omega_arr)}")

M_test=build_matrix(nat_freqs_rad[0])
print(f"Condition number: {np.linalg.cond(M_test):.2e}")

print(omega_arr[0]/(2*np.pi), omega_arr[-1]/(2*np.pi))

# From other scrip JONSWAP
U10_LC2=19.0 # m/s
F_fetch=180e3 # m
g=9.81
gamma=3.3 # JONSWAP peak enhancement factor
# has to be checked if these numbers are what they are looking for

# derived wave parameters LC2
Tp_LC2=0.286*(g*F_fetch/U10_LC2**2)**0.33*U10_LC2/g
Hs_LC2=0.0016*np.sqrt(g*F_fetch/U10_LC2**2)*U10_LC2**2/g
omega_p_LC2=2*np.pi/Tp_LC2

print(f"LC2: Tp={Tp_LC2:.2f} s, Hs={Hs_LC2:.2f} m, omega_p={omega_p_LC2:.3f} rad/s")

# ==== LOAD SEISMIC DATA =====
import os

seismic_file= '/Users/rooj/Desktop/funk/pyjive/CIEM5220/Earthquake/NL.G192..ALL.2018-01-08.dat'
data=np.loadtxt(seismic_file)

t_seis=data[:,0] # time
a_NS=data[:,1] # North-South acceleration [m/s^2]
a_EW=data[:,2] # East-West acceleration [m/s^2]
a_V=data[:,3] # Vertical acceleration [m/s^2]

dt=t_seis[1]-t_seis[0]
print(f"dt={dt:.4f} s")
print(f"Duration ={t_seis[-1]:.2f} s")
print(f"Max NS={np.max(np.abs(a_NS)):.6f} m/s^2")
print(f"Max EW = {np.max(np.abs(a_EW)):.6f} m/s^2")
print(f"Max V= {np.max(np.abs(a_V)):.6f} m/s^2")

# Quick plot to see the signal
fig, axes_s=plt.subplots(3, 1, figsize=(12, 8))
axes_s[0].plot(t_seis, a_NS)
axes_s[0].set_ylabel('NS[m/s^2]')
axes_s[0].grid(True)
axes_s[1].plot(t_seis, a_EW)
axes_s[1].set_ylabel('EW [m/s^2]')
axes_s[1].grid(True)
axes_s[2].plot(t_seis, a_V)
axes_s[2].set_ylabel('V[m/s^2]')
axes_s[2].set_xlabel('Time [s]')
axes_s[2].grid(True)
plt.suptitle('Groningen ground motion - NL.G192 2018-01-08')
plt.tight_layout()
plt.show()