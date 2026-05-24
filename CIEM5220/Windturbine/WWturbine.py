# ================================================
# CIEM5220 - Wave and Wind loads
#     Waterbound Windturbine 
# ================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from CIEM5220.Windturbine.dynamics_tools import shape_functions, element_stiffness, element_mass, compute_f1
from scipy.linalg import eigh
from scipy.optimize import brentq

x=sp.Symbol('x')

# Euler-Bernoulli        |<--- qmax=12kN
#                        |<--
#                      __|<-_____                       
#            
# Lenght of beam is L  
# 
# Length of one element is le 
     
# ====== GEOMETRY & MATERIAL ========
D0 = 5                          # [m]  
t=0.08                          # [m]  
Di = D0-2*t                     # [m]
A = np.pi/4*(D0**2-Di**2)       # [m^2]
I = np.pi/64*(D0**4-Di**4)      # [m^-4]
rho = 7850                      # steel [kg/m^3] ##
E = 210e9                       # [Pa]

EI=E*I
rhoA=rho*A

Hw=36.0                         # [m]
Ha=108.0                        # [m]
L=Hw+Ha                         # [m]  

M_nacelle=6e5       # [kg] 

# ====== ROTOR PARAMETERS ========
Omega_max=7.6                   # [rpm]
f_1P=Omega_max/60               # [Hz]
f_3P=3*f_1P                     # [Hz]
f_target=1.1*f_1P               # [Hz]

print(f"1P frequency: {f_1P:.4f} Hz")
print(f"3P frequency: {f_3P:.4f} Hz")
print(f"Target f1 (>=1.1*1P): {f_target:.4f} Hz")

# ====== DIAMETER TUNING =======

Dt_ratio=90
D0_range=np.arange(3.0, 10.0, 0.01) # sweep diameters [m]

elements=12
le=L/elements
elem_w=int(Hw/le)        # elements in submerged zone
elem_a=elements-elem_w   # elements in air zone 

nodes=elements+1
DOF=2             # (w, theta)
ndof=nodes*DOF

for D0_try in D0_range:
    f1_try=compute_f1(elements, ndof, Dt_ratio, D0_try, E, rho, Hw, Ha, L, elem_w, elem_a, le, M_nacelle)
    if f1_try>= f_target:
        D0=D0_try
        t=D0/Dt_ratio
        break

print(f"\n Diameter Tuning")
print(f"D/t ratio: {Dt_ratio}")
print(f"D0= {D0:.3f} m")
print(f"t= {t:.4f}m")
print(f"f1 achieved= {f1_try:.4f} Hz")
print(f"f_target= {f_target:.4f} Hz")
print(f"Margin above 1P = {(f1_try/f_1P-1)*100:.1f}%")

Di=D0-2*t
A=np.pi/4*(D0**2-Di**2)
I=np.pi/64*(D0**4-Di**4)
EI=E*I
rhoA=rho*A

# ====== FE MODEL SETUP ========

B = shape_functions(le)
Ke = element_stiffness(le, EI)
Me = element_mass(le, rhoA)

# ======= LOADS ========
q_wind=12e3                     # [N/m]
q_current=1e3                   # [N/m]

# ===== WIND & WAVE ENVIRONMENTS ======

g=9.81
F_fetch=180e3                   # fetch [m]
z0=0.001                        # roughness length, open sea [m]
f1=0.1397

# --- Load case parameters ---
T1=1/f1
U10_LC2=19.0
Tp_LC2=0.286*(g*F_fetch/U10_LC2**2)**0.33*U10_LC2/g
Hs_LC2=0.0016*np.sqrt(g*F_fetch/U10_LC2**2)*U10_LC2**2/g

Tp_LC1=T1
U10_LC1=brentq(
    lambda U: 0.286*(g*F_fetch/U**2)**0.33*U/g-Tp_LC1,
    1.0, 50.0
)
Hs_LC1=0.0016*np.sqrt(g*F_fetch/U10_LC1**2)*U10_LC1**2/g
print(f"\n--- Load Cases ---")
print(f"LC1: U10={U10_LC1:.2f} m/s, Tp={Tp_LC1:.2f} s, Hs={Hs_LC1:.2f} m")
print(f"LC2: U10={U10_LC2:.2f} m/s, Tp={Tp_LC2:.2f} s, Hs={Hs_LC2:.2f} m")

# --- Node heights above seabed ---
# nodes go from z=0 (mudline) to z=H (hub)
# wind acts on air nodes only (above waterline)
z_nodes=np.array([i*le for i in range(nodes)]) # height from mudline
z_above=z_nodes-Hw # height above waterline

# --- Mean wind profile at each node ---
def mean_wind(U10, z, z0):
    # only valid above waterline (z_above > 0)
    z_pos=np.maximum(z,z0) # aboud log(0)
    return U10*np.log(z_pos/z0)/np.log(10.0/z0)

U_mean_LC1=np.where(z_above > 0, mean_wind(U10_LC1, z_above, z0), 0.0)
U_mean_LC2=np.where(z_above > 0, mean_wind(U10_LC2, z_above, z0), 0.0)

fig, ax=plt.subplots(figsize=(6, 8))
ax.plot(U_mean_LC1, z_nodes, linewidth=2, label='LC1')
ax.plot(U_mean_LC2, z_nodes, linewidth=2, label='LC2')
ax.axhline(Hw, color='blue', linestyle='--', linewidth=0.8, label='Waterline')
ax.set_xlabel('Wind speed [m/s]')
ax.set_ylabel('Height z [m]')
ax.set_title('Mean wind profile')
plt.tight_layout()
plt.legend()
plt.show()

# ==== KAIMAL SPECTRUM =====
# Frequency axis
N_freq=1000          # Number of frequency points
f_max=2*1.2122       #  twice second natural frequency [Hz]
f_axis=np.linspace(0.01, f_max, N_freq)  # [Hz], avoid f=0

# Turbulance parameters (IEC standard, offshore)
I_u=0.06   # Turbulence intensity, offshore [-]
L_u=180.0  # Turbulence length scale [m]


# ======= ASSEMBLY ========
K =np.zeros((ndof,ndof))
M =np.zeros((ndof,ndof))
f_global=np.zeros(ndof)

# submerged elements - constant current load
for e in range(elem_w):
    p=q_current
    fe=np.array([
    float(sp.integrate(Ni*q_current,(x,0,le))) for Ni in B
    ])
    dofs=[2*e, 2*e+1, 2*e+2, 2*e+3]
    for i in range(4):
        f_global[dofs[i]] += fe[i]
        for j in range(4):
            K[dofs[i], dofs[j]] += Ke[i,j]
            M[dofs[i], dofs[j]] += Me[i,j]
# air elements - triangular wind load (zero at base, max at top)
for e in range(elem_a):
    p_bottom=q_wind*(e*le)/Ha
    p_top=q_wind*((e+1)*le)/Ha
    p=p_bottom*(1-x/le)+p_top*(x/le)
    fe=np.array([
    float(sp.integrate(Ni*p,(x,0,le))) for Ni in B
    ])
    dofs=[2*(elem_w+e), 2*(elem_w+e)+1, 2*(elem_w+e)+2, 2*(elem_w+e)+3]
    for i in range(4):
        f_global[dofs[i]] += fe[i]
        for j in range(4):
            K[dofs[i], dofs[j]] += Ke[i,j]
            M[dofs[i], dofs[j]] += Me[i,j]

# ===== BOUNDARY CONDITIONS =======
# clamped base: remove DOFs 0 and 1
free=np.arange(2, ndof)
K_free=K[np.ix_(free,free)]
M_free=M[np.ix_(free,free)]
f_free=f_global[free]

# ====== SANITY CHECK ========
# Uniform load over full length, no nacelle mass
# Analytical: u_tip = qL^4/8EI, M_mudline= qL^2/2
q_check=1e3             # [N/m] uniform load
f_check=np.zeros(ndof)

for e in range(elements):
    fe=np.array([
        float(sp.integrate(Ni*q_check, (x, 0, le))) for Ni in B
    ])
    dofs=[2*e, 2*e+1, 2*e+2, 2*e+3]
    for i in range(4):
        f_check[dofs[i]]+=fe[i]
    
f_check_free=f_check[free]
u_check=np.linalg.solve(K_free, f_check_free)

u_tip_FE= u_check[-2]

u_check_full = np.zeros(ndof)
u_check_full[free]=u_check
u_e0 = u_check_full[0:4]

M_mudline_FE = EI*sum(
    float(sp.diff(Ni, x, 2).subs(x, 0))*u_e0[i]
    for i, Ni in enumerate(B)
)
u_tip_analytical=q_check*L**4/(8*EI)
M_mudline_analytical=q_check*L**2/2

print(f"\n--- Sanity Check (uniform load q={q_check} N/m, no nacelle)---")
print(f"Tip deflection FE: {u_tip_FE:.6f} m")
print(f"Tip deflection Analytical: {u_tip_analytical:.6f} m")
print(f"Mudline moment FE: {M_mudline_FE/1e6:.3f} MNm")
print(f"Mudline moment Analytical: {M_mudline_analytical/1e6:.3f} MNm")

#========= + NACELLE MASS ===========
M[ndof-2, ndof-2]+=M_nacelle
M_free=M[np.ix_(free, free)]

#====== STATIC SOLVE =======

u=np.linalg.solve(K_free, f_free)

print(f"\n--- Static Analysis ---")
print(f"Tip deflection: {u[-2]:.6f} m")
print(f" Tip rotation: {u[-1]:.6f} rad")

# ==== EIGENVALUE ANALYSIS =======
eigenvalues, eigenvectors=eigh(K_free, M_free)
omega=np.sqrt(eigenvalues)
freq=omega/(2*np.pi)

print(f"\n---Natural frequencies ---")
print(f"Mode 1: {freq[0]:.4f} Hz (target >={f_target:.4f} Hz)")
print(f"Mode 2: {freq[1]:.4f} Hz")
print(f"Soft-stiff window: {f_1P:.4f}-{f_3P:.4f} Hz")

# ====== RAYLEIGH DAMPING ========
#Borth modes have 2% critical damping as per assignment

zeta1=0.02           # damping ratio mode 1
zeta2=0.02           # damping ratio mode 2
w1 = omega[0]        # first natural frequency [rad/s]
w2 = omega[1]        # second natural frequency [rad/s]

alpha = 2*w1*w2*(zeta1*w2-zeta2*w1)/(w2**2-w1**2)
beta= 2*(zeta2*w2-zeta1*w1)/(w2**2-w1**2)

C_free=alpha*M_free+beta*K_free

print(f"\n--- Rayleigh Damping ---")
print(f"alpha={alpha:.6f} (mass-proportional)")
print(f"beta={beta:.6f} (stiffness-proportional)")

# ======= POST-PROCESSING =========

# Mode shapes

fig3, ax=plt.subplots(figsize=(6, 8))

# height coordinates of each node
z_nodes=np.array([i*le for i in range(nodes)])

for mode in range(2):
    # extract lateral displacement DOFs (every other DOF starting from 0)
    phi=eigenvectors[:,mode]  # full free DOF eigenvector

    # lateral displacements are even indices of free DOFs
    phi_lateral=phi[0::2]  # every other entry starting from 0

    # prepend zero for clambed base
    phi_plot=np.concatenate([[0], phi_lateral])

    ax.plot(phi_plot, z_nodes, label=f'Mode {mode+1}: {freq[mode]:.4f} Hz')

ax.axhline(Hw, color= 'blue', linestyle='--', linewidth=0.8, label='Waterline')
ax.axvline(0,color='k', linewidth=0.5)
ax.set_xlabel('Normalised displacement')
ax.set_ylabel('Height z [m]')
ax.set_title('Mode shapes')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

omega_range=np.linspace(0.01, 4*w2,500) #frequency range [rad/s]
zeta_total=alpha/(2*omega_range)+beta*omega_range/2
zeta_mass = alpha/(2*omega_range)  # mass term - dominates at low frequencies
zeta_stiff=beta*omega_range/2    # stiffness term - dominates at high frequencies

fig2, ax = plt.subplots(figsize=(9,5))

ax.plot(omega_range/(2*np.pi), zeta_total*100, 'k', linewidth=2, label='Total')
ax.plot(omega_range/(2*np.pi), zeta_mass*100, 'b--', linewidth=1.5,label= 'Mass contribution')
ax.plot(omega_range/(2*np.pi), zeta_stiff*100, 'r--', linewidth=1.5, label='Stiffness contribution')

ax.axvline(w1/(2*np.pi), color='gray', linestyle=':', linewidth=1)
ax.axvline(w2/(2*np.pi), color='gray', linestyle=':', linewidth=1)
ax.axvline(f_1P, color='green', linestyle=':', linewidth=1.5, label='1P rotor')
ax.axvline(f_3P, color='orange', linestyle=':', linewidth=1.5, label='3P blade passing')

ax.annotate(f'Mode 1\n{w1/(2*np.pi):.2f} Hz',
            xy=(w1/(2*np.pi), zeta1*100), xytext=(w1/(2*np.pi)+0.1, zeta1*100+0.3),
            fontsize=8, color='gray')
ax.annotate(f'Mode 2\n{w2/(2*np.pi):.2f} Hz',
            xy=(w2/(2*np.pi), zeta2*100), xytext=(w2/(2*np.pi)+0.1, zeta2*100+0.3),
            fontsize=8, color='gray')

ax.set_xlim(0,4*w2/(2*np.pi))
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Damping ratio [%]')
ax.set_title('Rayleigh Damping')
ax.legend()
ax.grid(True)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.show()

#plotting moment and shear diagrams

u_full=np.zeros(ndof) 
u_full[free]=u       

#storage before the loop
z_plot, M_plot, V_plot, u_plot=[], [], [], []

for e in range(elements):
    dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]
    u_e = u_full[dofs]
    z_base=e*le if e < elem_w else Hw + (e-elem_w)*le

    for xp in np.linspace(0, le, 20):
        Mval = EI*sum(float(sp.diff(Ni, x, 2).subs(x, xp))*u_e[i]
                    for i, Ni in enumerate(B))
        Vval=EI*sum(float(sp.diff(Ni, x, 3).subs(x, xp))*u_e[i]
                    for i, Ni in enumerate(B))
        uval = sum(float(Ni.subs(x,xp))*u_e[i]
                    for i, Ni in enumerate(B))
        z_plot.append(z_base+xp)
        M_plot.append(Mval)
        V_plot.append(Vval)
        u_plot.append(uval)

fig,axes= plt.subplots(1,3,figsize=(12, 8))

for ax in axes:
    ax.axhline(Hw, color= 'blue', linestyle= '--', linewidth=0.8, label='waterline')

axes[0].plot(u_plot, z_plot, 'k')
axes[0].set_xlabel('Deflection [m]')
axes[0].set_ylabel('Height z [m]')
axes[0].set_title('Deflection')
axes[0].grid(True)

axes[1].plot(M_plot, z_plot, 'r')
axes[1].set_xlabel('Bending moment [Nm]')
axes[1].set_title('Bending Moment')
axes[1].axvline(0,color='k',linewidth=0.5)
axes[1].grid(True)

axes[2].plot([v/1e3 for v in V_plot], z_plot, 'g')
axes[2].set_xlabel('Shear force [kN]')
axes[2].set_title('Shear Force')
axes[2].axvline(0,color='k',linewidth=0.5)
axes[2].grid(True)

plt.tight_layout(pad=3.0)
plt.legend()
plt.show()

print(f"Maximum bending moment: {max(abs(m) for m in M_plot)/1e6:.1f} MNm")
print(f"Maximum shear force: {max(abs(v) for v in V_plot)/1e3:.1f} kN")

print(f"A={A:.6f} m^2")
print(f"I={I:.6f} m^4")

g=9.81
F_fetch=180e3  # fetch [m]
T1=1/freq[0]   # first natural peiod [s]

# LC2 - forward
U10_LC2=19.0
Tp_LC2=0.286*(g*F_fetch/U10_LC2**2)**0.33*U10_LC2/g
Hs_LC2=0.0016*np.sqrt(g*F_fetch/U10_LC2**2)*U10_LC2**2/g

# LC1 - invert Tp formula for U10

Tp_LC1=T1
U10_LC1=brentq(
    lambda U: 0.286*(g*F_fetch/U**2)**0.33*U/g-Tp_LC1, 
    1.0, 50.0
)
Hs_LC1=0.0016*np.sqrt(g*F_fetch/U10_LC1**2)*U10_LC1**2/g

print(f"LC1: U10={U10_LC1:.2f} m/s, Tp={Tp_LC1:.2f} s, Hs={Hs_LC1:.2f} m")
print(f"LC2: U10={U10_LC2:.2f} m/s, Tp={Tp_LC2:.2f} s, Hs={Hs_LC2:.2f} m")
