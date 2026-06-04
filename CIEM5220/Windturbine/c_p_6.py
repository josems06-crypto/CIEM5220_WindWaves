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
L=0.6       # m

def beam_element_matrices(E, I, rho, A, Le):
    