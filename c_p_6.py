import numpy as np
import matplotlib as plt
from scipy.linalg import eigh

# K*phi=omega^2*M*phi 

E=210e9         # Pa
rho=7850        # kg/m^3
b=0.05          # m
h=0.1           # m
I=(b*h**3)/12   # m^4
A=b*h           # m^2

k1=200e6    # N/m
k2=75e6     # N/m
ms=200      # kg
Le=0.6       # m

def beam_element_matrices(E, I, rho, A, Le):
    Kb=(E*I/Le**3)*np.array([[12, 6*Le, -12, 6*Le],
                   [6*Le, 4*Le**2, -6*Le, 2*Le],
                   [-12, -6*Le, 12, -6*Le],
                   [6*Le, 2*Le**2, -6*Le, 4*Le**2]])
    Mb=(rho*A*Le/420)*np.array([[156, 22*Le, 54, -13*Le],
                       [22*Le, 4*Le**2, 13*Le, -3*Le**2],
                       [54, 13*Le, 156, -22*Le],
                       [-13*Le, -3*Le**2, -22*Le, 4*Le**2]])
    return Kb, Mb

Kb, Mb = beam_element_matrices(E, I, rho, A, Le)
print(Kb)
print(Mb)

print(Kb.shape)  # returns (4, 4)
print(Kb(0,0))