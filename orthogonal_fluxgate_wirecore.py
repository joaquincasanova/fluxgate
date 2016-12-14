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
def Rdrive(L,w,sigma,r):
    return (2*L+w)/sigma/PI/r/r
def Rsense(N,sigma,Dsense,r):
    return (N*PI*Dsense)/sigma/r/r/PI
def Ldrive(mur, w, r,L):    
    return MU0*mur*np.arccosh(w/2/r)*L/PI
def Lsense(L,N,mu,Acore,Asense):
    return MU0*N*N*(mu*Acore+(Asense-Acore))/L
def Demag(d,L):
    return (d/L)*(d/L)*(2*np.log(L/d)-0.46)
def B(H,Hc,mui,Bs, sign):
    #sign=+1 if increasing
    #sign=-1 if decreasing
    Ht = 2*Bs/PI/(mui)/MU0
    return 2*Bs/PI*np.arctan((H-sign*Hc)/Ht)
def dBdH(H,Hc,mui,Bs, sign):
    #sign=+1 if increasing
    #sign=-1 if decreasing
    Ht = 2*Bs/PI/mui/MU0
    return 2*Bs/PI/(1+(H-sign*Hc)/Ht*(H-sign*Hc)/Ht)/Ht
def mua(mur,D):
    return mur/(1+D*(mur-1))
def Pdiss(B,H,period,Volume):
    return np.sum(B*np.gradient(H)/period)*Volume

ac_levels=50
dc_levels=250
loop_n=ac_levels*dc_levels
nGrid = 100
L = 25.4e-3 #length of one side
tempK = 300 #(K)
r_cu = 101e-6
sigma_cu = 1/(1.724e-8)# (siemens/m)

Ndrive = 16# (#) total
Nsense = 100

Idrive_wire = np.zeros([loop_n,nGrid])
Bz_wire = np.zeros([loop_n,nGrid])
Hz_wire = np.zeros([loop_n,nGrid])
Btheta_wire = np.zeros([loop_n,nGrid])
Htheta_wire = np.zeros([loop_n,nGrid])
Vwire = np.zeros([loop_n,nGrid])
timewire = np.zeros([loop_n,nGrid])
FOM=np.zeros([ac_levels,dc_levels])
index=np.zeros([ac_levels,dc_levels])
loop_index=0

#Wire core

for ac_index in range(1,ac_levels+1):
    for dc_index in range(1,dc_levels+1):
        print ac_index, dc_index, loop_index
        sigma_core = 1/(142e-6/100)
        Bs = 0.8#(T)
        mu_i = 15000.
        Hc = 4 #(A/m)
        Hm = (10e-12)/MU0
        Hearth = 0#45e-6/MU0
        Idc = 1e-3*dc_index#(A)
        Iamp = 1e-3*ac_index
        f = 100e3#(Hz)
        period = 1/f #(s)

        time = np.linspace(0,2*period,nGrid)
        dt = 2*period/(nGrid-1)

        omega = 2*PI*f#(rad/s)
        cos_omegat = np.cos(omega*time)
        sin_omegat = np.sin(omega*time)
        sign = np.sign(sin_omegat)

        Idrive = Iamp*cos_omegat+Idc
        dIdrivedt = omega*Iamp*sin_omegat

        r_aw = 15e-6
        n_aw = 16
        r_eff = np.sqrt(n_aw)*r_aw
        OD = 3./16.*25.4e-3
        Dsense = OD
        w=1./16.*25.4e-3
        Atot = n_aw*PI*(r_aw*r_aw)
        delta = skin_depth(sigma_core, omega, mu_i*MU0)
        Acore = n_aw*PI*(r_aw*r_aw-(r_aw-delta)*(r_aw-delta))
        delta_eff = (r_eff-np.sqrt(r_eff*r_eff-Acore/PI))
        r_loop = (r_eff-(r_eff-delta_eff))/np.log(r_eff/(r_eff-delta_eff))#effective mean radius
        Acorewire=Acore
        #approximate wire bundle with one wire r_eff, delta_eff
        print "delta, delta_eff, r_aw, r_eff, r_loop: {}, {}, {}, {}, {}".format(delta,delta_eff,r_aw,r_eff,r_loop)
        Asense = Dsense*Dsense*PI/4
        Aloop = PI*((r_loop)*(r_loop)-(r_eff-delta_eff)*(r_eff-delta_eff))#area for current at field evaluation depth r_loop
        print "Acore, Asense, Aloop: {},{},{}".format(Acore,Asense,Aloop)

        D = Demag(np.sqrt(n_aw)*r_aw/eta(n_aw)*2,L)
        Dwire=D
        mu_a=mua(mu_i,D)
        print "Demag: {}".format(D)    
        R_sense = Rsense(Nsense,sigma_cu,Dsense,r_cu)
        R_drive = Rdrive(L,w,sigma_core,np.sqrt(Acore/PI))
        en_sense = noise(R_sense,tempK)
        en_drive = noise(R_drive,tempK)
        L_drive = Ldrive(mu_i, w, np.sqrt(n_aw)*r_aw,L)
        L_sense = Lsense(L,Nsense,mu_i,Acore,Asense)
        print "R sense: {}".format(R_sense)
        print "R drive: {}".format(R_drive)
        print "L sense: {}".format(L_sense)
        print "L drive: {}".format(L_drive)
        print "en sense: {}".format(en_sense)
        print "en drive: {}".format(en_drive)
        Htheta,Hz  = H(r_eff-delta_eff/2,Idrive*Aloop/Acore,Hm+Hearth)
        B_theta = B(Htheta, Hc, mu_i,Bs, sign)
        mu_z = B_theta/(Htheta-sign*Hc)/MU0
        B_z = Hm*mu_z/(1+Dwire*mu_z)*MU0
        Phys=Pdiss(B_theta,Htheta,2*period,Acore*L)
        print "Pdiss: {}".format(Phys)
        Phi = 2*Acorewire*Nsense*B_z
        V=-np.gradient(Phi)/np.gradient(time)
        Psig=np.max(V)*np.max(V)/R_sense/2
        FOM[ac_index-1,dc_index-1]=Psig
        index[ac_index-1,dc_index-1]=loop_index
        print "Signal power/Hysteresis: {}".format(Psig/Phys)    
        Bz_wire[loop_index,:]=B_z
        Hz_wire[loop_index,:]=Hz
        Btheta_wire[loop_index,:]=B_theta
        Htheta_wire[loop_index,:]=Htheta

        Idrive_wire[loop_index,:]=Idrive
        Vwire[loop_index,:] = V
        timewire[loop_index,:] = time
        loop_index = loop_index+1

best=np.argmax(FOM)
best=index[np.unravel_index(best,[ac_levels,dc_levels])]

plt.subplot(3, 1, 1)
plt.plot(np.transpose(timewire[best,:]), np.transpose(Idrive_wire[best,:]),'-')
plt.title('Current and Voltage: Orthogonal, 10pT flux')
plt.xlim(0,2*period)
plt.ylabel('Iex (A)')

plt.subplot(3, 1, 2)
plt.plot(np.transpose(Htheta_wire[best,:]), np.transpose(Btheta_wire[best,:]),'-')
plt.ylabel('Btheta (T)')

plt.xlabel('Htheta (A/M)')

plt.subplot(3, 1, 3)
plt.plot(np.transpose(timewire[best,:]), np.transpose(Vwire[best,:]),'-')
plt.ylabel('Vsense')
plt.xlim(0,2*period)

plt.xlabel('time (s)')

plt.show()
              

                        
        
