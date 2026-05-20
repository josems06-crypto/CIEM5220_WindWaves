# CANONICAL 
# moving oscillator
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def eom(t, y, omega, omega_v, chi, M, rhoA, L, V, g):

    # --- Unpack state vector ---
    q1 = y[0]
    q2 = y[1]
    u  = y[2]
    dq1= y[3]
    dq2= y[4]
    du = y[5]

    # --- B(t): is the oscillator on the beam? ---
    if t<L/V:
        B =1
        phi1=np.sin(np.pi*V*t/L)   # mode 1 shape at load position
        phi2=np.sin(2*np.pi*V*t/L) # mode 2 shape at load position
    else:
        B = 0
        phi1= 0
        phi2= 0

    #--- Stiffness matrix C (time-varying) ---
    C= np.array([
        [omega[0]**2+chi*omega_v**2*phi1**2*B, chi*omega_v**2*phi2*phi1*B, -chi*omega_v**2*phi1*B],
        [chi*omega_v**2*phi1*phi2*B, omega[1]**2+chi*omega_v**2*phi2**2*B, -chi*omega_v**2*phi2*B],
        [-omega_v**2*phi1*B, -omega_v**2*phi2*B, omega_v**2]
    ])

    # --- Displacement vector D ---
    D=np.array([q1, q2, u])

    # --- Forcing vector E ---
    Q0=M*g
    E=np.array([
        2*Q0*phi1/(rhoA*L),
        2*Q0*phi2/(rhoA*L),
        0.0
    ])

#---Accelerations: D_ddot = E - C @ D ---
    D_ddot = E - C @ D

# --- Return full state derivative ---
    return [dq1, dq2, du, D_ddot[0], D_ddot[1], D_ddot[2]]

#--- Parameters ---
L=25         #  [m]
rhoA=2303    #  [kg/m]
EI=8.31e9    #  [Nm^2]
K=1595e3     #  [N/m]
M=5750       #  [kg]
V=36         #  [m/s]
g=9.81       #  [m/s^2]
Q0=M*g       #  static load [N]

#--- Oscillator ---
omega_v = np.sqrt(K/M) # natural frequency of oscillator [rad/s]
chi = 2*M/(rhoA*L)     # mass ratio, dimensionless

#--- Beam natural frequencies (2 modes) ---


n_modes=2
omega=np.array([
    (mode*np.pi/L)**2*np.sqrt(EI/rhoA)
    for mode in range(1, n_modes +1)
])

print(f"Osscillator natural frequency: omega_v= {omega_v:.2f} rad/s")
print(f"Mass ratio chi={chi:.4f}")
print(f"Modal bar mass m_bar={rhoA*L/2:.1f} kg")
for m, w in enumerate(omega, start=1):
    Omega_m=m*np.pi*V/L
    beta_m=Omega_m/w
    print(f"Mode {m}: omega= {w:.2f} rad/s, Omega= {Omega_m:.2f} rad/s, beta= {beta_m:.3f}")


# --- Time span ---
t_end = 2* L/V  # run twice the passage time to see free vibration
t=np.linspace(0, t_end, 2000)

# --- initial conditions ---
y0 = [0, 0, 0, 0, 0, 0] # everything at rest

# --- Solve ---
sol=solve_ivp(
    fun=lambda t, y: eom(t, y, omega, omega_v, chi, M, rhoA, L, V, g), 
    t_span=[0,t_end], 
    y0=y0,
    t_eval=t, 
    method='RK45', 
    rtol=1e-8,
    atol=1e-10
)

q1=sol.y[0]
q2=sol.y[1]
u=sol.y[2]

print("Solver done. Success:", sol.success)

# --- Reconstruct beam midspan deflection ---
x_mid=L/2
w_mid=q1*np.sin(np.pi*x_mid/L)+q2*np.sin(2*np.pi*x_mid/L)
x_quarter=L/4
w_quarter=q1*np.sin(np.pi*x_quarter/L)+q2*np.sin(2*np.pi*x_quarter/L)
t_passage=L/V # moment oscillator leaves beam

# --- Plot 1 : midspan deflection ---
plt.figure(figsize=(8, 4))
plt.plot(t, w_mid, label='Midspan (x=L/2)')
plt.plot(t, w_quarter, label='Quarter-span (x=L/4)')
plt.axvline(t_passage, color= 'gray', linestyle= '--', label= 'Oscillator exits beam')
plt.xlabel('Time [s]')
plt.ylabel('Midspan deflection [m]')
plt.title('Beam midspan deflection - moving oscillator')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#--- Plot 2 : oscillator displacement ---
plt.figure(figsize=(8,4))
plt.plot(t, u)
plt.axvline(t_passage, color='gray', linestyle='--', label='Oscillator exits beam')
plt.xlabel('Time [s]')
plt.ylabel('Oscillator displacement[m]')
plt.title('Oscilllator displacement u_d(t)')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

F0=1000.0    # load magnitude [N]

# problem 2 for comparison
omega_v=np.sqrt(K/M)
chi=2*M/(rhoA*L)

F0=M*g
n_modes_p2=2
omega_p2=np.array([
    (mode*np.pi/L)**2*np.sqrt(EI/rhoA)
    for mode in range(1, n_modes_p2+1)
])

t_end_p2=2*L/V
t_p2=np.linspace(0,t_end_p2, 2000)
q_p2=np.zeros((n_modes_p2, len(t_p2)))

for mode in range(1, n_modes_p2+1):
    omega_m=omega_p2[mode-1]
    Omega_m=mode*np.pi*V/L
    amp=2*F0/(rhoA*L)

    def make_ode(om, Om, a, passage):
        def modal_ode(t, y):
            force=a*np.sin(Om*t)*(1 if t< passage else 0)
            return [y[1], -om**2*y[0]+force]
        return modal_ode
    sol_p2=solve_ivp(
        make_ode(omega_m, Omega_m, amp, L/V),
        [0, t_end_p2], [0, 0],
        t_eval=t_p2, rtol=1e-8, atol=1e-10
    )
    q_p2[mode -1, :]=sol_p2.y[0]

# reconstruct midspan
w_mid_p2=(q_p2[0]*np.sin(np.pi*(L/2)/L) +
          q_p2[1]*np.sin(2*np.pi*(L/2)/L))

# --- Comparison plot --- 
plt.figure(figsize=(8, 4))
plt.plot(t, w_mid, label='Moving oscillator')
plt.plot(t_p2, w_mid_p2, linestyle='--', label='Moving load (Problem 2)')
plt.axvline(L/V, color='gray', linestyle=':', label='Oscillator exits beam')
plt.xlabel('Time [s]')
plt.ylabel('Midspan deflection [m]')
plt.title(f'Moving oscillator vs moving load -V = {V} m/s')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()