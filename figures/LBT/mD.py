import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
# NLO and LO QCD mD2
Nc, Nf = 3., 3.
pf_g = 4.*np.pi/3.*(Nc+Nf/2.)
alpha0 = 4.*np.pi/(11. - 2./3.*Nf)
Lambda2 = 0.2**2
Q2cut_l = -Lambda2*np.exp(alpha0)
Q2cut_h = Lambda2*np.exp(np.pi*np.tan(np.pi*(0.5-1./alpha0)))
def alpha_s(Q2):
	if Q2 < Q2cut_l:
		return alpha0/np.log(-Q2/Lambda2)
	elif Q2 <= Q2cut_h:
		return 1.0;
	else:
		return alpha0*(0.5 - np.arctan(np.log(Q2/Lambda2)/np.pi)/np.pi)
def mm2_LO_eq(m2, T, mu):
	scale = -m2*mu
	return (alpha_s(scale)*0.482*4.*np.pi)**2 - m2/T/T
def naive_LO_mm2(T, mu):
	scale = -(mu*np.pi*2.*T)**2
	return (alpha_s(scale)*0.482*4.*np.pi)**2*T*T
naive_LO_mm2 = np.vectorize(naive_LO_mm2)
def mm2_LO_sf(T, mu):	
	return brentq(mm2_LO_eq, 0.01, 100., args=(T, mu))
mm2_LO_sf = np.vectorize(mm2_LO_sf)
def mD2_LO_eq(m2, T, mu):
	scale = -m2*mu
	return alpha_s(scale)*6.*np.pi - m2/T/T
def naive_LO_mD2(T, mu):
	Q2 = -(mu*2.*np.pi*T)**2
	return alpha_s(Q2)*6.*np.pi*T*T
naive_LO_mD2 = np.vectorize(naive_LO_mD2)
def mD2_LO_sf(T, mu):	
	return brentq(mD2_LO_eq, 0.01, 100., args=(T, mu))
mD2_LO_sf = np.vectorize(mD2_LO_sf)
def mD2_NLO_eq(m2, T, mu):
	scale = -m2*mu
	mDLO = mD2_LO_sf(T, mu)**0.5
	mmLO = mm2_LO_sf(T, mu)**0.5
	logterm = np.log(2.*mDLO/mmLO) - 0.5
	return alpha_s(scale)*6.*np.pi*(1. + np.sqrt(alpha_s(scale)*6./np.pi)*logterm)**2 \
			- m2/T/T
def naive_NLO_mD2(T, mu):
	scale = -(mu*2.*np.pi*T)**2
	mDLO = naive_LO_mD2(T, mu)**0.5
	mmLO = naive_LO_mm2(T, mu)**0.5
	logterm = np.log(2.*mDLO/mmLO) - 0.5
	return alpha_s(scale)*6.*np.pi*(1. + np.sqrt(alpha_s(scale)*6./np.pi)*logterm)**2*T*T
naive_NLO_mD2 = np.vectorize(naive_NLO_mD2)
def mD2_NLO_sf(T, mu):	
	return brentq(mD2_NLO_eq, -10, 100., args=(T, mu))
mD2_NLO_sf = np.vectorize(mD2_NLO_sf)

T = np.linspace(0.15, 0.6, 100)
mD_LO = [naive_LO_mD2(T,mu)**0.5 for mu in [0.5, 1.0, 2.0]]
mD_sf = [mD2_LO_sf(T, mu)**0.5 for mu in [0.5, 1.0, 2.0]]

for lo, sf, c in zip(mD_LO, mD_sf, 'rgb'):
	plt.plot(T, lo/T, '-', color=c)
	plt.plot(T, sf/T, '--', color=c)

plt.show()




