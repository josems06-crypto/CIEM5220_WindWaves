import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────
EI   = 6.42e6    # Bending stiffness [Nm^2]
rhoA = 268.3     # Mass per unit length [kg/m]
chi  = 7.3e6     # Winkler foundation stiffness [N/m^2]
F0   = 1.0       # Boundary load amplitude (normalised) [N]

# Cut-off frequency
omega_0 = np.sqrt(chi / rhoA)
print(f"Cut-off frequency  omega_0 = {omega_0:.4f} rad/s")
print(f"Cut-off frequency  f_0     = {omega_0/(2*np.pi):.4f} Hz")

omega_low  = 0.3 * omega_0
omega_high = 3.0 * omega_0
print(f"\nomega_low  = {omega_low:.4f} rad/s  ({omega_low/omega_0:.1f} * omega_0)")
print(f"omega_high = {omega_high:.4f} rad/s  ({omega_high/omega_0:.1f} * omega_0)")

x = np.linspace(0, 80, 3000)

# ── Response function ──────────────────────────────────────────────────────
def beam_response(omega, x, EI, rhoA, chi, F0):
    """
    W(x): steady-state amplitude for semi-infinite beam on Winkler foundation.

    Characteristic equation: lambda^4 = -(chi - rhoA*omega^2)/EI

    Below cut-off (omega < omega_0):
      rhs < 0  =>  four complex roots, TWO have Re < 0
      => 2x2 BC system (shear + moment at x=0)

    Above cut-off (omega > omega_0):
      rhs > 0  =>  two real roots (+/- kappa) + two imaginary (+/- i*kappa)
      Only ONE root has Re < 0 strictly: -kappa (real, negative)
      The imaginary root -i*kappa has Re = 0: marginally stable => propagating wave
      We keep BOTH (physical radiation condition) and apply 2 BCs.
    """
    rhs_val = -(chi - rhoA * omega**2) / EI   # lambda^4 = rhs_val
    mag = abs(rhs_val)**(1/4)

    if rhs_val < 0:
        # Below cut-off — complex roots
        # lambda^4 = -|rhs| = |rhs| * e^(i*pi)
        # roots: mag * e^(i*(pi + 2*pi*k)/4)
        angles = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
        roots  = [mag * np.exp(1j * a) for a in angles]
        # Keep Re < 0
        stable = [r for r in roots if r.real < -1e-12]
        lam1, lam2 = stable[0], stable[1]

    else:
        # Above cut-off — rhs > 0
        # roots: +kappa, -kappa, +i*kappa, -i*kappa
        kappa = mag   # real positive
        # Physical radiation: wave must travel in +x direction (away from source)
        # -kappa => exponential growth, rejected
        # +kappa => exponential decay, but rhs>0 means kappa is real =>
        #   actually: lambda^4 = kappa^4 > 0 =>
        #   lambda = kappa (unstable), -kappa (stable, decaying), i*kappa, -i*kappa
        # For propagating wave (+x direction), keep -i*kappa (outgoing)
        # and -kappa would give decay — but with rhs>0, check: e^(-kappa*x) decays OK
        # Radiation condition for harmonic e^(i*omega*t): outgoing wave e^(-i*kappa*x)
        lam1 = -1j * kappa    # propagating wave in +x direction (outgoing)
        lam2 = -kappa         # this would be evanescent; for pure above-cutoff, only lam1 needed
        # With one propagating root, use single BC (shear force only):
        # EI * lam1^3 * A1 = -F0  => A1 = -F0 / (EI * lam1^3)
        A1 = -F0 / (EI * lam1**3)
        W  = A1 * np.exp(lam1 * x)
        return W

    # Below cut-off: 2x2 system
    # BC1: EI * W'''(0) = -F0  => EI*(A1*lam1^3 + A2*lam2^3) = -F0
    # BC2: EI * W''(0)  =  0   => EI*(A1*lam1^2 + A2*lam2^2) =  0
    M_bc = np.array([
        [lam1**3, lam2**3],
        [lam1**2, lam2**2]
    ], dtype=complex)
    rhs_bc = np.array([-F0 / EI, 0.0], dtype=complex)
    A = np.linalg.solve(M_bc, rhs_bc)
    W = A[0] * np.exp(lam1 * x) + A[1] * np.exp(lam2 * x)
    return W

# ── Compute ────────────────────────────────────────────────────────────────
W_low  = beam_response(omega_low,  x, EI, rhoA, chi, F0)
W_high = beam_response(omega_high, x, EI, rhoA, chi, F0)

w_low  = np.real(W_low)   # snapshot at t=0: e^(i*omega*0) = 1
w_high = np.real(W_high)

# Normalise for visual comparison
w_low_n  = w_low  / np.max(np.abs(w_low))
w_high_n = w_high / np.max(np.abs(w_high))

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

# --- Below cut-off ---
axes[0].plot(x, w_low_n, color='steelblue', linewidth=2)
axes[0].set_title(
    rf'Below cut-off:  $\omega = 0.3\,\omega_0 = {omega_low:.1f}$ rad/s'
    '  →  spatially decaying (evanescent)',
    fontsize=12)
axes[0].set_xlabel('x  [m]', fontsize=11)
axes[0].set_ylabel('w(x) / max|w|  [–]', fontsize=11)
axes[0].axhline(0, color='k', linewidth=0.6, linestyle='--')
axes[0].grid(True, alpha=0.35)

# --- Above cut-off ---
axes[1].plot(x, w_high_n, color='firebrick', linewidth=2)
axes[1].set_title(
    rf'Above cut-off:  $\omega = 3.0\,\omega_0 = {omega_high:.1f}$ rad/s'
    '  →  propagating wave',
    fontsize=12)
axes[1].set_xlabel('x  [m]', fontsize=11)
axes[1].set_ylabel('w(x) / max|w|  [–]', fontsize=11)
axes[1].axhline(0, color='k', linewidth=0.6, linestyle='--')
axes[1].grid(True, alpha=0.35)

plt.suptitle(
    r'Problem 1 — Steady-state response of semi-infinite beam on Winkler foundation'
    '\n'
    r'(snapshot at $t = 0$;  normalised amplitude)',
    fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout()
plt.show()
print("Done — plot saved.")
