import numpy as np
import scipy.special as sp
import csv
import matplotlib.pyplot as plt

global PI
global MU0
global BOLTZ
global COMPJ
PI = np.pi               # value of PI
MU0 = 4*PI*1e-7 #(H/m)
BOLTZ = 1.38064852e-23 #(J/K)
COMPJ=np.lib.scimath.sqrt(-1)
def eta(n):
    e=[1.0000, 0.5000,0.6466,0.6864,0.6854,0.6667,0.7778,0.7328,
       0.6895,0.6878,0.7148,0.7392,0.7245,0.7474,0.7339,0.7512,
       0.7403,0.7611,0.8034,0.7623]
    return e[n-1]
def H(r,I,Hm):
    Htheta=I/2/PI/r
    return Htheta, Hm*np.ones(np.shape(r))
def skin_depth(sigma, omega, mu):
    return np.sqrt(2/sigma/omega/mu)
def noise(R,tempK):
    return 2*np.sqrt(BOLTZ*tempK*R)
def Rdc(sigma, r,L):
    return L/sigma/r/r/PI
def Z(theta, mu, sigma, r, N1, N2, L):
    reff=r*np.sqrt(N1)#effective radius of bundle
    R=(1-COMPJ)*reff*Rdc(sigma, reff, L)/2/skin_depth(sigma, omega, MU0)
    Zzz=R*(np.sqrt(mu)*np.cos(theta)*np.cos(theta)+np.sin(theta)*np.sin(theta))
    Zzphi=-R*(2*PI*reff)*1/L*(np.sqrt(mu)-1)*np.cos(theta)*np.sin(theta)
    Zphiz=R*(2*PI*reff)*N2/L*(np.sqrt(mu)-1)*np.cos(theta)*np.sin(theta)
    Zphiphi=-R*(2*PI*reff)*(2*PI*reff)*1/L*N2/L*(np.sqrt(mu)*np.sin(theta)*np.sin(theta)+np.cos(theta)*np.cos(theta))
    return Zzz, Zzphi, Zphiz, Zphiphi
def Demag(d,L):
    return (d/L)*(d/L)*(2*np.log(L/d)-0.46)
def B(H,Hc,mui,Bs, sign):
    #sign=+1 if increasing
    #sign=-1 if decreasing
    Ht = 2*Bs/PI/(mui)/MU0
    return 2*Bs/PI*np.arctan((H-sign*Hc)/Ht)
def mua(mur,D):
    return mur/(1+D*(mur-1))

#Constants:
tempK = 300 #(K)
r_aw = 50e-6 #(m)
L = 10e-3 #length of one side
n_aw = 4 #(#)
Nsense = 100
r_cu = 40e-6
sigma_cu = 1/(1.724e-8)# (siemens/m)
sigma_core = 1/(130e-6/100)
Bs = 0.8#(T)
mu_i = 15000

Hc = 4 #(A/m)
Hm = 1e-12/MU0
Hk = Bs/mu_i/MU0
Hearth = 0e-6/MU0
Iamp = 20e-3
Idc = 2.5e-3
f = 50e6#(Hz)
period = 1/f
nGrid = 100
time = np.linspace(0,2*period,nGrid)
dt = 2*period/(nGrid-1)
omega = 2*PI*f#(rad/s)
cos_omegat = np.cos(omega*time)
sin_omegat = np.sin(omega*time)

Idrive = Iamp*cos_omegat+Idc
delta = skin_depth(sigma_core, omega, mu_i*MU0)
H_phi_dc,H_z_dc  = H(r_aw,Idc,Hm+Hearth)
M_phi_dc = B(H_phi_dc, Hc, mu_i,Bs, 0)/MU0
M_z_dc = B(H_z_dc, Hc, mu_i,Bs, 1)/MU0

D = Demag(2*r_aw,L)
mu_a=mua(mu_i,D)
theta = np.arctan(M_phi_dc/M_z_dc)
Zzz, Zzphi, Zphiz, Zphiphi=Z(theta, mu_a, sigma_core, r_aw, n_aw, Nsense, L)

V=Iamp*np.abs(Zphiz)*np.cos(omega*time+np.angle(Zphiz))

plt.subplot(2, 1, 1)
plt.plot(time, Idrive, 'r.-')
plt.title('Current and Voltage: GMI, 4 wire core')
plt.ylabel('Iw (A)')
plt.xlim(0,2*period)

plt.subplot(2, 1, 2)
plt.plot(time,V , 'r.-')
plt.ylabel('Vc (V)')
plt.xlabel('time (s)')
plt.xlim(0,2*period)


plt.show()

                        
        
