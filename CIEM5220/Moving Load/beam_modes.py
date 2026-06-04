import numpy as np
import matplotlib.pyplot as plt

L=1.0 # Beam leangth (normalised)
x=np.linspace(0,L,500)

plt.figure(figsize=(8, 5))

for m in range(1, 5):
    phi=np.sin(m*np.pi*x/L)
    plt.plot(x, phi, label=f'Mode {m}')

plt.xlabel('x/L')
plt.ylabel('phi(x)')
plt.title('Mode shapes of simply supported beam')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()
