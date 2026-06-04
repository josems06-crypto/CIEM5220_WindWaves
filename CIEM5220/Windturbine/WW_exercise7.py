
import numpy as np

#exercise

M=np.array([[2, 1], [1, 1]])
K=np.array([[20, 0],[0, 10] ])

vals, vecs=np.linalg.eig(np.linalg.inv(M)@K)
omegas=np.sqrt(np.sort(vals))
print("omega=", omegas)
print("f=", omegas/(2*np.pi))
print("Mode shapes:\n", vecs)

# For the Rayleigh
zeta=0.05
omega1, omega2= omegas
alpha=2*zeta*omega1*omega2/(omega1+omega2)
beta=2*zeta/(omega1+omega2)
C=alpha*M+beta*K
print("alpha=\n", alpha)
print("beta=\n", beta)
print("C=\n", C)

import matplotlib.pyplot as plt
SO, psi= 0.1, 0.9
freqs=np.linspace(0.01, 2, 1000)
psd4, psd5= [], []

for f in freqs:
    omega=2*np.pi*f
    H=np.linalg.inv(K-omega**2*M+1j*omega*C)
    S=SO*np.eye(2)
    Sc=SO*np.array([[1, psi], [psi, 1]])
    psd4.append(np.real((H[1,:] @ S @ H[1,:].conj())))
    psd5.append(np.real((H[1,:] @ S @ H[1,:].conj())))
# assuming forces are uncorrelated

plt.plot(freqs, psd4, label='psi=0')
plt.plot(freqs, psd5, label='psi=0.9')
plt.xlabel('f [Hz]'); plt.ylabel('Stheta2'); plt.legend()
plt.show()

psi=0.9
SF_corr=SO*np.array([[1, psi], [psi, 1]])
Stheta2_corr=np.zeros(len(freqs))

for i, f in enumerate(freqs):
    w=2*np.pi*f
    H=np.linalg.inv(K-omega**2*M+1j*omega*C)

    Stheta_mat=H @ SF_corr @ H.conj().T
    Stheta2_corr[i]=np.real(Stheta_mat[1,1])

# Variance
var_corr=np.trapezoid(Stheta2_corr, freqs)
print(f"zimga^2theta_2 (corr, psi=0.9)={var_corr:.4f} rad^2")

#plot both together to see the difference
fig, ax= plt.subplots(figsize=(9,4))
ax.plot(freqs, Stheta_uncorr, label='Psi=0 (uncorrelated)', color='steelblue')
ax.plot(freqs, Stheta2_uncorr, label='Psi=0 (uncorrelated)', color='steelblue')
