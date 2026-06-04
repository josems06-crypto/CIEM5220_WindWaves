import numpy as np
import matplotlib.pyplot as plt

# Frequency ratio: forcing frequency / natural frequency
beta=np.linspace(0,2,1000)

# Damping ratios to compare
zeta_values=[0.0, 0.1, 0.2, 0.5]

plt.figure(figsize=(8, 5))

for zeta in zeta_values:
    DAF=1/np.sqrt((1-beta**2)**2+(2*zeta*beta)**2)
    plt.plot(beta, DAF, label=f'zeta={zeta}')

plt.axvline(x=1, color='gray', linestyle='--', label='beta=1 (resonance)')
plt.xlabel('beta=omega/wn')
plt.ylabel('DAF')
plt.title('Dynamic Amplification Factor')
plt.legend()
plt.ylim(0, 6)
plt.grid(True)
plt.tight_layout()
plt.show()