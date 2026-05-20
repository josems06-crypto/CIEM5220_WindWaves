# ================================================
#         CIEM5220 - Seismology
# Seismic waves and earthqiake occurence
# ================================================
import os
os.chdir("/Users/rooj/Desktop/funk/pyjive/CIEM5220/Earthquake")
import numpy as np
import matplotlib.pyplot as plt

#__ Velocity model ______________________________
Vp1, Vs1= 4.0, 2.2  # Sediments (0-5 km)
Vp2, Vs2= 6.2, 3.6  # Crust (5-35 km)
Vp3, Vs3= 7.9, 4.5  # Mantle (35+ km)
Moho=35             # [km]

h=16 # focal depth (km)
z=Moho-h     # depth from focus to Moho

# Task 1: ray parameters

p_Pn=1/Vp3
p_Sn=1/Vs3
print(f"p_Pn={p_Pn:.4f} s/km")
print(f"p_Sn={p_Sn:.4f} s/km")

# Task 2: ctitical angles and minimum distances
ic_P=np.arccos(Vp2/Vp3)
ic_S=np.arccos(Vs2/Vs3)

Xmin_P=2*z*np.tan(ic_P)
Xmin_S=2*z*np.tan(ic_S)

Tmin_P=2*z/(Vp2*np.cos(ic_P))+h/Vp2
Tmin_S=2*z/(Vs2*np.cos(ic_S))+h/Vs2

print(f"\nXmin_Pn= {Xmin_P:.1f} km, T={Tmin_P:.2f} s")
print(f"Xmin_Sn= {Xmin_S:.1f} km, T={Tmin_S:.2f} s")

# Task 3: traveltime curves

X=np.linspace(0, 500, 1000)

def T_Pn(X, h):
    z= Moho-h
    ic=np.arccos(Vp2/Vp3)
    Xmin=2*z*np.tan(ic)
    T=X/Vp3+2*z*np.cos(ic)/Vp2+h/Vp2
    T[X<Xmin]=np.nan # Wave doesn't exist before Xmin
    return T

def T_Sn(X, h):
    z= Moho-h
    ic=np.arccos(Vs2/Vs3)
    Xmin=2*z*np.tan(ic)
    T=X/Vs3+2*z*np.cos(ic)/Vs2+h/Vs2
    T[X<Xmin]=np.nan
    return T

Tp=T_Pn(X.copy(), h)
Ts=T_Sn(X.copy(), h)

#PLT
# Slopes 
slope_P=1/Vp3
slope_S=1/Vs3
print(f"\nSlope Pn = {slope_P:.4f} s/km -> V_mantle_P={Vp3} km/s")
print(f"Slope Sn={slope_S:.4f} s/km -> V_mantle_S= {Vs3} km/s")

""" plt.figure(figsize=(8,5))
plt.plot(X, Tp, label='Pn', color='steelblue', linewidth=2)
plt.plot(X, Ts, label='Sn', color='tomato', linewidth=2)
plt.xlabel("Epicentral distance (km)")
plt.ylabel("Traveltime (s)")
plt.title("Pn and Sn traveltime curves (focal depth 16 km)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() """


# Task 4: traveltimes at 487 km
X_pk=np.array([487.0])
Tp_pk= T_Pn(X_pk, h)[0]
Ts_pk=T_Sn(X_pk, h)[0]
print(f"\nAt X=487 km:")
print(f" T_Pn={Tp_pk:.1f} s ({Tp_pk/60:.2f} min)")
print(f" T_Sn= {Ts_pk:.1f} s ({Ts_pk/60:.2f} min)")


#__ Load the earthquake catalog ______________

data= np.loadtxt("List_M.txt")

years=data[:,0].astype(int)
months=data[:,1].astype(int)
days=data[:,2].astype(int)
mags=data[:,3]
idx=mags.argmax()

print(f"Total events: {len(mags)}")
print(f"Max magnitude: {mags[idx]} on {years[idx]}-{months[idx]}-{days[idx]}")

#__ Task 8 - Remove strongest event and build G-R plto
mags_filtered=mags[mags<9.1]
Years_Span=40
M_values=np.arange(4.0, 9.1, 0.1)
N_values=np.array([np.sum(mags_filtered>= m)/ Years_Span for m in M_values])

valid=N_values>0
M_valid=M_values[valid]
logN_valid=np.log10(N_values[valid])

""" plt.figure(figsize=(8,5))
plt.scatter(M_valid, logN_valid, color='green', zorder=3)
plt.xlabel("Magnitude M")
plt.ylabel("log10 N (events/year)")
plt.title("Gutenberg-Richter relation - Japan Trench 1979-2019")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
 """
# Task 9 - linear regression
fit_mask=(M_valid >= 4.5) & (M_valid <= 8)

coeffs=np.polyfit(M_valid[fit_mask], logN_valid[fit_mask], 1)
b_val=-coeffs[0]
a_val=coeffs[1]
print(coeffs)
print(f"a={a_val:.3f}")
print(f"b={b_val:.3f}")

# Task 10 - add fit line to plot
fit_line=a_val-b_val*M_values

""" plt.figure(figsize=(8,5))
plt.scatter(M_valid, logN_valid, color='orange', zorder=3, label='Observed')
plt.plot(M_values, fit_line, color='blue', linewidth=2, label=f'Fit: a={a_val:.3f}, b={b_val:.3f}')
plt.xlabel("Magnitude M")
plt.ylabel("log10 N (events/year)")
plt.title("Gutenberg-Righter relation - Japan Trench 1979-2019")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
 """
# Task 11 - return period and probability
M_target = 9.0
t=80

logN_90=a_val-b_val*M_target
N_90=10**logN_90
T=1/N_90

P_exact=1-np.exp(-t/T)
P_approx=t/T

print(f"logN(M>=9.0)= {logN_90:.3f}")
print(f"N per year= {N_90:.5f}")
print(f"Return period T= {T:.1f} years")
print(f"P(80yr) exact = {P_exact*100:.1f}%")
print(f"P(80yr) approx = {P_approx*100:.1f}%")
