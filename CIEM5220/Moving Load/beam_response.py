import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#--- Beam parameters---
L=10.0       # Length [m]
EI=1e6       # bending stiffness [Nm^2]
rho_A=100.0  # mass per unit length [kg/m]
F0=1000.0    # load magnitude [N]

#--- Load speed ---
V=125.66      # [m/s] to change

#--- Number of modes to include ---
n_modes=4

#--- Natural frequencies ---
omega= np.array([
    (m*np.pi/L)**2*np.sqrt(EI/rho_A)
    for m in range(1, n_modes+1)
])

print("Natural frequencies [rad/s]:")
for m, w in enumerate(omega, start=1):
    print(f"Mode {m}: w={w:.3f}")

#--- Time array (load trravels from x=0 to x=L)---
t_end=L/V
t=np.linspace(0,t_end,1000)

#---Solve each modal ODE---
q=np.zeros((n_modes, len(t))) # each row = one mode's time history

for m in range(1, n_modes + 1):
    omega_m=omega[m-1]
    Omega_m=m*np.pi*V/L
    forcing_amplitude=2*F0/(rho_A*L)

    def modal_ode(t, y):
        # y[0]=q_m, y[1]=dq_m/dt
        dydt=[
            y[1],
            -omega_m**2*y[0]+forcing_amplitude*np.sin(Omega_m*t)
        ]
        return dydt
    sol=solve_ivp(modal_ode, [0, t_end], [0,0], t_eval=t)
    q[m-1, :]=sol.y[0]
print("Modal ODEs solved.")

#--- Redonstruct beam deflection w(x, t) ---
x=np.linspace(0,L,200)

# w is a 2D array: rows=positions, colomns=time steps
w=np.zeros((len(x), len(t)))

for m in range(1, n_modes+1):
    phi_m=np.sin(m*np.pi*x/L)   # mode shape, shape (len(x),)
    w+=np.outer(phi_m, q[m-1, :]) # add contribution of mode m

# --- Plot 1: midspan deflection over time ---
mid_idx=len(x)//2

plt.figure(figsize=(8,4))
plt.plot(t, w[mid_idx, :])
plt.xlabel('Time [s]')
plt.ylabel('Deflection [m]')
plt.title(f'Midspan deflection, V={V}m/s')
plt.grid(True)
plt.tight_layout()
plt.show()

#--Plot 2: beam shape at a specific moment --
t_mid_idx=len(t)//2

plt.figure(figsize=(8,4))
plt.plot(x, w[:,t_mid_idx])
plt.xlabel('x [m]')
plt.ylabel('Deflection [m]')
plt.title(f'Beam shape at t= {t[t_mid_idx]:.2f} s')
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()

#-- Critical velocities --
for m in range(1, n_modes+1):
    V_cr=omega[m-1]*L/(m*np.pi)
    print(f" Mode {m}: V_cr = {V_cr:.2f} m/s")

# Plot individual mode contributions at midspan --
quarter_idx=len(x)//4

plt.figure(figsize=(8,5))

for m in range(1, n_modes +1):
    phi_m_mid = np.sin(m*np.pi*x[quarter_idx]/L)
    contribution= phi_m_mid*q[m-1,:]
    plt.plot(t, q[m-1, :], label= f'Mode {m}')

plt.xlabel('Time[s]')
plt.ylabel('Deflection [m]')
plt.title(f'Individual mode contributions at 1/4 span, V=V_cr2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
