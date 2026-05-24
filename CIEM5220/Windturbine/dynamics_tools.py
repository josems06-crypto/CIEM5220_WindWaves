import numpy as np
import sympy as sp
from scipy.special import erf
from scipy.linalg import eigh

def shape_functions(le):

    """
    Returns Euler - Bernoulli shape functions 
    for element of length le. Symbolic in x (sympy)
    Call as B= shape_functions(le)
    """

    x= sp.Symbol('x')

    N1=1-3*(x/le)**2 + 2*(x/le)**3
    N2=x*(1 - x/le)**2
    N3=3*(x/le)**2 - 2*(x/le)**3
    N4=(x**2/le)*(x/le - 1)
    
    return [N1, N2, N3, N4]

def element_stiffness(le, EI):
    """
    Returns Euler - Bernoulli stiffness matrix
    length le, EI=Emodulus*inertia 
    """
    return EI/le**3*np.array([
    [12,      6*le,   -12,   6*le ],
    [6*le, 4*le**2, -6*le, 2*le**2], 
    [-12,    -6*le,    12,  -6*le ],
    [6*le, 2*le**2, -6*le, 4*le**2],
    ])

def element_mass(le, rhoA):
    """
    Returns Euler - Bernoulli mass matrix
    length le, rhoA=mass per meter*Area
    """
    return rhoA*(le/420)*np.array([       
    [156,    22*le,      54,   -13*le        ],
    [22*le,  4*le**2,   13*le,   3*le**2     ],
    [54,     13*le,      156,   -22*le       ],
    [-13*le, -3*le**2, -22*le, 4*le**2       ],
    ])

def H_sdof(k, V, K, C, L, M):
    """ 
    Returns the SDOF vehicle 
    transfer function
    """
    omega = k*V                     # temporal frequency [rad/m * m/s = rad/s]
    K_bar=K+1j*omega*C              # complex stiffness  [N/m]
    num = K_bar*(1+ np.exp(1j*k*L)) # [N/m] - road input amplitude
    den= -M * V**2 * k**2 + 2*K + 2j*omega*C # [N/m] - dynamic stiffness
    return num/den

def H_2dof(k, V, K, M, J, L, C):
    """ 
    Returns the MDOF _vehicle_
    transfer function 
    OBS simplified because; 
    K_bar_R = K_bar_F
    i.e. symmetric case
    """
    omega=k*V
    K_bar=K+1j*omega*C

    a11=-M*V**2 * k**2 + 2*K_bar
    a22=-J*V**2 * k**2 + L**2*K_bar/2
    b1=K_bar*(1+np.exp(1j*k*L))
    b2=L*K_bar*(1-np.exp(1j*k*L))/2
    H_w=b1/a11
    H_theta=b2/a22
    return H_w, H_theta

def prob_exceedance(x_crit, sigma):
    """ 
    Gaussian probability of exceedence
    for zero-mean process.

    P(|x|>x_crit) for one-sided (x_crit>0 only)

    Parameters:
        x_crit: float - threshold value
        sigma : float  - standard deviation of the process
    Returns:
        P     : float - probability [0,1]
    """
    return 0.5*(1-erf(x_crit/(sigma*np.sqrt(2))))

#def sigma_from_psd(S,k):

def S_jonswap(alpha, omega, omega_p, gamma):
    """ 
    JONSWAP for irregular waves 
    """
    g=9.81 # gravity [m/s^2]
    #Pierson-Moskowitz base shape
    S_PM= (alpha*g**2/omega**5)*np.exp(-1.25*(omega_p/omega)**4)
    # specral width
    sigma=np.where(omega<= omega_p, 0.07, 0.09)
    # peak enhancement exponent
    r=np.exp(-(omega-omega_p)**2/(2*sigma**2*omega_p**2))
    
    return S_PM*gamma**r

from scipy.optimize import brentq

def wave_numbers(omega_arr, h):
    """ 
    Solve dispersion relation 
    omega^2=g*k*tanh(k*h)
    for each frequency in omega_arr.
    
    Parameters:
        omega_arr : array of frequencies [rad/s]
        h         : water depth [m]
    Returns:
        k_arr     : array of wave numbers [rad/m]
    
    """
    g=9.81
    k_arr=np.zeros(len(omega_arr))
    for i, omega_i in enumerate(omega_arr):
        k_arr[i]=brentq(
            lambda k: omega_i**2-g*k*np.tanh(k*h),
            1e-6, 1000
        )
    return k_arr

def S_KT(S0, w_g, Chi_g, w):
    """ 
     Defines the Kanai-Tajimi spectrum
    """
    num=w_g**4+4*Chi_g**2*w_g**2*w**2
    den=(w_g**2-w**2)**2+4*Chi_g**2*w_g**2*w**2
    return S0*num/den

def compute_f1(elements, ndof, Dt_ratio, D0_try, E, rho, Hw, Ha, L, elem_w, elem_a, le, M_nacelle):
    t_try=D0_try/Dt_ratio
    Di_try=D0_try-2*t_try
    A_try=np.pi/4*(D0_try**2-Di_try**2)
    I_try=np.pi/64*(D0_try**4-Di_try**4)
    EI_try=E*I_try
    rhoA_try=rho*A_try

    Ke_try=element_stiffness(le, EI_try)
    Me_try=element_mass(le, rhoA_try)

    K_try=np.zeros((ndof, ndof))
    M_try=np.zeros((ndof, ndof))

    for e in range(elements):
        dofs=[2*e, 2*e+1, 2*e+2, 2*e+3]
        for i in range(4):
            for j in range(4):
                K_try[dofs[i], dofs[j]] += Ke_try[i,j]
                M_try[dofs[i], dofs[j]]+= Me_try[i,j]

    M_try[ndof-2, ndof-2] += M_nacelle
    free=np.arange(2, ndof)
    K_free_try=K_try[np.ix_(free, free)]
    M_free_try=M_try[np.ix_(free, free)]

    eigenvalues, _ = eigh(K_free_try, M_free_try)
    omega_try=np.sqrt(eigenvalues)
    return omega_try[0]/(2*np.pi)

