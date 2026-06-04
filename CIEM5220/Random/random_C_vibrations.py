"""
=============================================================================
CIEM5220 – Random Vibrations: Complete Worked Example
Moving Load on Random Road Profile  (ISO 8608 roughness)
=============================================================================

WORKFLOW  (follows Lectures 1–3 exactly)
  STEP 1  – Define the road roughness PSD  S_w0(k)   [ISO 8608, L1/L2]
  STEP 2  – Generate a random road realization w0(x)  [L1 – spectral method]
  STEP 3  – SDOF vehicle: transfer function H(ω)      [L2 – spectral approach]
  STEP 4  – Response PSD  S_w(ω)  and variance σ²     [L2]
  STEP 5  – Time-domain response  w(t)                [L2 – realisation]
  STEP 6  – Extend to 2-DOF (bounce + pitch)          [L3]
  STEP 7  – Acceleration PSD at driver seat           [L3]
  STEP 8  – Probability of exceedance                 [L1/L3]
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erf

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c8cfe8",
    "axes.titlecolor":  "#e8ecff",
    "xtick.color":      "#8890b0",
    "ytick.color":      "#8890b0",
    "grid.color":       "#2a2d3d",
    "grid.linewidth":   0.6,
    "text.color":       "#c8cfe8",
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4d",
    "font.family":      "sans-serif",
    "font.size":        10,
})
CYAN   = "#00d4ff"
GREEN  = "#00ff9f"
ORANGE = "#ff8c42"
PURPLE = "#c77dff"
RED    = "#ff4d6d"
YELLOW = "#ffd166"
WHITE  = "#e8ecff"

# =============================================================================
# PARAMETERS  – all adjustable
# =============================================================================

# --- Vehicle ---
M  = 1500.0        # total mass [kg]
J  = 2000.0        # moment of inertia [kg·m²]
L  = 3.0           # wheelbase [m]
K  = 20_000.0      # suspension stiffness each axle [N/m]  (KR = KF = K)
C  = 1_500.0       # suspension damping each axle [N·s/m]
V  = 25.0          # vehicle speed [m/s]  (~90 km/h)

# --- ISO 8608 road roughness ---
# S_w0(k) = Gd * (k0/k)^2   k in [rad/m]
# Class B road: Gd ≈ 64e-6 m³/rad  at k0=1 rad/m
Gd = 64e-6         # road roughness coefficient [m³/rad]  (ISO class B)
k0 = 1.0           # reference wave-number [rad/m]

# --- Simulation ---
N_k  = 2048        # number of wave-number samples for road generation
k_min = 0.05       # [rad/m]
k_max = 50.0       # [rad/m]

# --- Critical acceleration (comfort limit) ---
a_crit = 0.5       # [m/s²]  e.g. ISO 2631 comfort threshold

# =============================================================================
# STEP 1 – Road roughness PSD  S_w0(k)
# =============================================================================

k_arr = np.linspace(k_min, k_max, N_k)  # wave-number array

def S_road(k):
    """ISO 8608 one-sided PSD  [m²/(rad/m)]"""
    return Gd * (k0 / k)**2

S_w0 = S_road(k_arr)

# =============================================================================
# STEP 2 – Random road realization  w0(x)  via spectral method  (L1)
#   F(t) = Σ F̃(Ωn) cos(Ωn·t + φn),   F̃n = sqrt(2·ΔΩ·S/π)
# =============================================================================

dk   = k_arr[1] - k_arr[0]
Famp = np.sqrt(2 * dk * S_w0 / np.pi)  # amplitude of each harmonic

rng  = np.random.default_rng(seed=42)
phi  = rng.uniform(0, 2*np.pi, size=N_k)   # random phases

# spatial coordinate (road seen by front axle)
x_road = np.linspace(0, 200, 4000)   # 200 m road section

w0_x = np.zeros(len(x_road))
for i, (ki, Fi, ph) in enumerate(zip(k_arr, Famp, phi)):
    w0_x += Fi * np.cos(ki * x_road + ph)

# =============================================================================
# STEP 3 – SDOF vehicle transfer function  H(ω)  (L2)
#   mẍ + 2Cẋ + 2Kx = K̄(w0(Vt) + w0(Vt+L))
#   simplified: symmetric vehicle → only bounce, road input at front axle
#
#   In the frequency / wave-number domain (ω = k·V):
#   H_SDOF(k) = (K + ikVC)(1 + e^{ikL}) / (-mV²k² + 2K + 2ikVC)
# =============================================================================

def H_sdof(k):
    """SDOF bounce transfer function (wave-number domain)."""
    omega = k * V
    K_bar = K + 1j * omega * C          # complex stiffness
    num   = K_bar * (1 + np.exp(1j * k * L))
    den   = -M * V**2 * k**2 + 2*K + 2j*omega*C
    return num / den

H_k = H_sdof(k_arr)

# =============================================================================
# STEP 4 – Response PSD and variance  σ²  (L2)
#   S_w(k) = S_w0(k) · |H(k)|²
#   σ² = (1/π) ∫₀^∞ S_w(ω) dω  = (V/π) ∫ S_w(k) dk   [change of var]
# =============================================================================

S_w_sdof = S_w0 * np.abs(H_k)**2

sigma2_sdof = (V / np.pi) * np.trapezoid(S_w_sdof, k_arr)
sigma_sdof  = np.sqrt(sigma2_sdof)

print("=" * 60)
print("SDOF RESULTS")
print(f"  Response std dev  σ_w  = {sigma_sdof*1000:.3f} mm")
print(f"  Response variance σ²_w = {sigma2_sdof*1e6:.4f} mm²")

# =============================================================================
# STEP 5 – Time-domain response  w(t)  via same spectral method  (L2)
# =============================================================================

t_arr  = x_road / V          # time array for vehicle travelling at speed V
phi2   = rng.uniform(0, 2*np.pi, size=N_k)   # new random phases for response

# Road input at front axle: w0(Vt)
w0_t_front = np.zeros(len(t_arr))
w0_t_rear  = np.zeros(len(t_arr))
w_resp     = np.zeros(len(t_arr))

for i, (ki, ph) in enumerate(zip(k_arr, phi)):
    omega_i = ki * V
    A_road  = Famp[i]
    w0_t_front += A_road * np.cos(omega_i * t_arr + ph)
    w0_t_rear  += A_road * np.cos(omega_i * t_arr + ph - ki * L)

    # response amplitude from |H| and a random phase shift
    A_resp = Famp[i] * np.abs(H_k[i])
    phase_resp = np.angle(H_k[i]) + ph
    w_resp += A_resp * np.cos(omega_i * t_arr + phase_resp)

# =============================================================================
# STEP 6 – 2-DOF vehicle: bounce (w) + pitch (θ)  (L3)
#
#  Matrix system in wave-number domain:
#  [a11  a12] [w̃]   =  w̃0(k) [b1]
#  [a21  a22] [θ̃]              [b2]
# =============================================================================

def transfer_2dof(k):
    """Returns H_w(k) and H_theta(k) for full 2-DOF system (L3 notation)."""
    omega  = k * V
    KR_bar = K + 1j * omega * C
    KF_bar = K + 1j * omega * C   # symmetric: KR = KF = K, CR = CF = C

    a11 = -M  * V**2 * k**2 + KR_bar + KF_bar
    a12 = (L/2) * (KR_bar - KF_bar)   # = 0 for symmetric vehicle
    a21 = a12
    a22 = -J * V**2 * k**2 + (L/2)**2 * (KR_bar + KF_bar)

    b1  = KR_bar + KF_bar * np.exp(1j * k * L)
    b2  = (L/2) * (KR_bar - KF_bar * np.exp(1j * k * L))

    det = a11 * a22 - a12**2

    H_w     = (b1 * a22 - b2 * a12) / det
    H_theta = (-b1 * a21 + b2 * a11) / det
    return H_w, H_theta

H_w2, H_th2 = transfer_2dof(k_arr)

S_w2    = S_w0 * np.abs(H_w2)**2
S_th2   = S_w0 * np.abs(H_th2)**2

sigma2_w2  = (V / np.pi) * np.trapezoid(S_w2,  k_arr)
sigma2_th2 = (V / np.pi) * np.trapezoid(S_th2, k_arr)

print("\n2-DOF RESULTS")
print(f"  Bounce  σ_w   = {np.sqrt(sigma2_w2)*1000:.3f} mm")
print(f"  Pitch   σ_θ   = {np.sqrt(sigma2_th2)*1000:.4f} mrad")

# =============================================================================
# STEP 7 – Acceleration PSD at driver seat  (L3)
#   Seat at 3L/4 from rear  →  a_vert = ẅ + (L/4)·θ̈
#   ã_vert(ω) = -(kV)² · w̃0(k) · (H_w + L/4 · H_θ)
#
#   S_a(ω) = (ω⁴/V²) · S_w0(ω/V) · |H_w(ω/V) + L/4·H_θ(ω/V)|²
# =============================================================================

omega_arr = k_arr * V   # frequency array [rad/s]

H_seat = H_w2 + (L / 4) * H_th2   # combined transfer function at seat

# acceleration PSD (the ω⁴/V² factor comes from double time-differentiation)
S_a = (omega_arr**4 / V**2) * S_w0 * np.abs(H_seat)**2

sigma2_a = (1 / np.pi) * np.trapezoid(S_a, omega_arr)
sigma_a  = np.sqrt(sigma2_a)

print(f"  Acceleration σ_a = {sigma_a:.4f} m/s²")

# =============================================================================
# STEP 8 – Probability of exceedance  (L1 / L3)
#   P(a > a_crit) = ½ · erfc(a_crit / (σ_a · √2))
# =============================================================================

def prob_exceed(a_c, sigma):
    """Gaussian probability of exceedance (zero-mean process)."""
    return 0.5 * (1 - erf(a_c / (sigma * np.sqrt(2))))

P_exceed = prob_exceed(a_crit, sigma_a)

print(f"\nProbability of exceedance")
print(f"  a_crit = {a_crit} m/s²")
print(f"  P(a > a_crit) = {P_exceed:.4f}  ({P_exceed*100:.2f} %)")
print("=" * 60)

# =============================================================================
# PLOTTING – 4-panel overview
# =============================================================================

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel helpers ─────────────────────────────────────────────────────────────
def styled_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=8, framealpha=0.7)

# ── 1. Road profile + vehicle path ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(x_road, w0_x * 1000, color=CYAN, lw=0.8, alpha=0.85, label="Road profile $w_0(x)$")
ax1.axhline(0, color=WHITE, lw=0.5, ls="--", alpha=0.4)
# mark front and rear axle at t=0
ax1.axvline(0,   color=GREEN,  lw=1.2, ls=":", label="Front axle (t=0)")
ax1.axvline(L,   color=ORANGE, lw=1.2, ls=":", label="Rear axle  (t=0)")
styled_ax(ax1,
          "STEP 2 – Random Road Profile  (ISO 8608, Class B, realisation)",
          "Position  x  [m]", "Elevation  $w_0$  [mm]")
ax1.set_xlim(x_road[0], x_road[-1])

# annotate PSD class
ax1.text(0.98, 0.95,
         f"$G_d$ = {Gd*1e6:.0f}×10⁻⁶ m³/rad   V = {V} m/s",
         transform=ax1.transAxes, ha="right", va="top",
         fontsize=8.5, color=YELLOW,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d27", alpha=0.8))

# ── 2. Road PSD  S_w0(k) ─────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.loglog(k_arr, S_w0, color=CYAN, lw=1.8, label="$S_{w_0}(k)$ ISO 8608")
ax2.loglog(k_arr, Gd * (k0/k_arr)**2, color=WHITE, lw=0.7, ls="--", alpha=0.4,
           label="$G_d(k_0/k)^2$ theory")
styled_ax(ax2,
          "STEP 1 – Road PSD  $S_{w_0}(k)$",
          "Wave-number  $k$  [rad/m]",
          "PSD  [m²/(rad/m)]")

# ── 3. |H(k)|² – SDOF vs 2-DOF bounce ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.semilogy(k_arr, np.abs(H_k)**2,    color=GREEN,  lw=1.8, label="|H(k)|² SDOF")
ax3.semilogy(k_arr, np.abs(H_w2)**2,   color=ORANGE, lw=1.8, label="|H_w(k)|² 2-DOF bounce")
ax3.semilogy(k_arr, np.abs(H_th2)**2,  color=PURPLE, lw=1.4, ls="--",
             label="|H_θ(k)|² 2-DOF pitch")
styled_ax(ax3,
          "STEP 3/6 – Transfer Functions",
          "Wave-number  $k$  [rad/m]",
          "$|H(k)|^2$")
ax3.set_xlim(k_min, k_max)

# ── 4. Response PSD  S_w(ω) ──────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.semilogy(omega_arr, S_w_sdof, color=GREEN,  lw=1.8, label="$S_w$ SDOF")
ax4.semilogy(omega_arr, S_w2,     color=ORANGE, lw=1.8, label="$S_w$ 2-DOF bounce")
ax4.semilogy(omega_arr, S_a,      color=RED,    lw=1.4, ls="--",
             label="$S_a$ driver accel")
styled_ax(ax4,
          "STEP 4/7 – Response & Acceleration PSD",
          "Frequency  $\\omega$  [rad/s]",
          "PSD  [m²·s/rad]")
ax4.set_xlim(0, 60)

# ── 5. Time-domain response ───────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
t_plot = t_arr[:800]
ax5.plot(t_plot, w0_t_front[:800]*1000, color=CYAN,  lw=0.9, alpha=0.7,
         label="Road input  $w_0(Vt)$  [mm]")
ax5.plot(t_plot, w_resp[:800]*1000,     color=GREEN, lw=1.3,
         label="SDOF response  $w(t)$  [mm]")
ax5.fill_between(t_plot,
                  -2*sigma_sdof*1000, 2*sigma_sdof*1000,
                  color=GREEN, alpha=0.08, label="±2σ band")
styled_ax(ax5,
          "STEP 5 – Time-Domain Response  (SDOF)",
          "Time  $t$  [s]", "Displacement  [mm]")

# ── 6. Probability of exceedance (Gaussian tail) ─────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])

a_vals = np.linspace(-4*sigma_a, 4*sigma_a, 500)
pdf_a  = (1 / (sigma_a * np.sqrt(2*np.pi))) * np.exp(-0.5 * (a_vals/sigma_a)**2)

ax6.plot(a_vals, pdf_a, color=RED, lw=2, label=f"$p(a)$,  σ={sigma_a:.3f} m/s²")

# shade exceedance region
mask = a_vals >= a_crit
ax6.fill_between(a_vals[mask], pdf_a[mask], color=RED, alpha=0.35,
                 label=f"P(a>{a_crit}) = {P_exceed*100:.2f}%")
ax6.axvline(a_crit, color=YELLOW, lw=1.4, ls="--",
            label=f"$a_{{crit}}$ = {a_crit} m/s²")
ax6.axvline(-a_crit, color=YELLOW, lw=1.4, ls="--", alpha=0.5)

styled_ax(ax6,
          "STEP 8 – Probability of Exceedance",
          "Acceleration  $a$  [m/s²]", "PDF  $p(a)$")

# ── master title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "CIEM5220 – Random Vibrations  |  Vehicle on Random Road  (ISO 8608 Class B)",
    fontsize=13, fontweight="bold", color=WHITE, y=0.98
)

plt.savefig("/mnt/user-data/outputs/random_vibrations.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nFigure saved.")
plt.close()
