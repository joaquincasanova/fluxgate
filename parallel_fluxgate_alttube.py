import numpy as np
import scipy.special as sp
import csv
import matplotlib.pyplot as plt

global PI
global MU0
global BOLTZ

PI = np.pi               # value of PI
MU0 = 4*PI*1e-7 #(H/m)
BOLTZ = 1.38064852e-23 #(J/K)

def eta(n):
    e=[1.0000, 0.5000,0.6466,0.6864,0.6854,0.6667,0.7778,0.7328,
       0.6895,0.6878,0.7148,0.7392,0.7245,0.7474,0.7339,0.7512,
       0.7403,0.7611,0.8034,0.7623]
    return e[n-1]
def H(I,N,L,Hm):
    Hz=N*I/L+Hm
    return Hz
def skin_depth(sigma, omega, mu):
    return np.sqrt(2/sigma/omega/mu)
def noise(R,tempK):
    return 2*np.sqrt(BOLTZ*tempK*R)
def Rcoil(N,sigma,r,D):
    A = PI*r*r
    return N*PI*D/sigma/A
def Lcoil(L,N,mu,Acore,Asense):
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

ac_levels=100
dc_levels=100
loop_n=ac_levels*dc_levels
nGrid = 100
L = 25.4e-3 #length of one side
tempK = 300 #(K)
r_cu = 101e-6
sigma_cu = 1/(1.724e-8)# (siemens/m)

Ndrive = 200# (#) total
Nsense = 100

Idrive_tube = np.zeros([loop_n,nGrid])
Bz1_tube = np.zeros([loop_n,nGrid])
Hz1_tube = np.zeros([loop_n,nGrid])
Vtube1 = np.zeros([loop_n,nGrid])
Bz2_tube = np.zeros([loop_n,nGrid])
Hz2_tube = np.zeros([loop_n,nGrid])
Vtube2 = np.zeros([loop_n,nGrid])
timetube = np.zeros([loop_n,nGrid])
FOM=np.zeros([ac_levels,dc_levels])
index=np.zeros([ac_levels,dc_levels])
loop_index=0
#Tube core
#Tube core
for ac_index in range(1,ac_levels+1):
    for dc_index in range(1,dc_levels+1):
        #print ac_index, dc_index, loop_index
        sigma_core = 1/(142e-6/100)
        Bs = 1.18#(T)
        mu_i = 15000
        Hc = 1 #(A/m)
        Hm = (1e-12)/MU0
        Hearth = 50e-6/MU0
        Idc =100e-3*dc_index/dc_levels#(A)
        Iamp = Idc*ac_index/ac_levels
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

        delta = skin_depth(sigma_core, omega, mu_i*MU0)
#        print "Skin depth: {}".format(delta)
        Thick = 2*22e-6
        OD = 1.59e-3+2*Thick
        ID = 1.59e-3
        Ddrive = OD
        Dsense = .1875*25.44/1000
        Acore = PI*(OD*OD-ID*ID)/4
        Adrive = PI*(OD*OD)/4
        Asense = Dsense*Dsense*PI/4
        D = Demag(OD,L)-Demag(ID,L)
        Acoretube=Acore
        Dtube=D

        HzL1  = H(Idrive,Ndrive,L,Hm+Hearth)
        HzR1  = H(-Idrive,Ndrive,L,Hm+Hearth)
        mu_a=mua(mu_i,D)
        BzL1 = B(HzL1,Hc,mu_a,Bs, sign)
        BzR1 = B(HzR1,Hc,mu_a,Bs, -sign)

        HzL2  = H(Idrive,Ndrive,L,Hearth)
        HzR2  = H(-Idrive,Ndrive,L,Hearth)
        mu_a=mua(mu_i,D)
        BzL2 = B(HzL2,Hc,mu_a,Bs, sign)
        BzR2 = B(HzR2,Hc,mu_a,Bs, -sign)

        w = OD+4*r_cu #separation
        #print "Acore, Asense: {},{}".format(Acore,Asense)
        Phi1 = Acoretube*Nsense*BzL1+Acoretube*Nsense*BzR1
        Phi2 = Acoretube*Nsense*BzL2+Acoretube*Nsense*BzR2          
        en_sense = noise(Rcoil(Nsense,sigma_cu,r_cu,Dsense),tempK)
        en_drive = noise(Rcoil(Ndrive,sigma_cu,r_cu,Ddrive),tempK)
        R_sense = Rcoil(Nsense,sigma_cu,r_cu,Dsense)
        R_drive = Rcoil(Ndrive,sigma_cu,r_cu,Ddrive)
        L_sense = Lcoil(L,Nsense,mu_i,2*Acore,Asense)
        L_drive = Lcoil(2*L,Ndrive,mu_i,Acore,Adrive)
        Phys=Pdiss(BzL1,HzL1,2*period,Acore*L)+Pdiss(BzR1,HzR1,2*period,Acore*L)
        #print "Demag: {}".format(D)
        #print "en sense: {}".format(en_sense)
        #print "en drive: {}".format(en_drive)
        #print "R sense: {}".format(R_sense)
        #print "R drive: {}".format(R_drive)
        #print "L sense: {}".format(L_sense)
        #print "L drive: {}".format(L_drive)
        #print "Pdiss: {}".format(Phys)
        V1=-np.gradient(Phi1)/np.gradient(time)
        V2=-np.gradient(Phi2)/np.gradient(time)
        
        Psig=np.max(V1-V2)*np.max(V1-V2)/R_sense/2
        FOM[ac_index-1,dc_index-1]=Psig/Phys
        index[ac_index-1,dc_index-1]=loop_index
        #print "Signal power/Hysteresis: {}".format(Psig/Phys)
        Bz1_tube[loop_index,:]=BzL1
        Hz1_tube[loop_index,:]=HzL1
        Idrive_tube[loop_index,:]=Idrive
        Vtube1[loop_index,:] = V1
        Bz2_tube[loop_index,:]=BzL2
        Hz2_tube[loop_index,:]=HzL2
        Idrive_tube[loop_index,:]=Idrive
        Vtube2[loop_index,:] = V2

        timetube[loop_index,:] = time
        loop_index = loop_index+1

best=np.argmax(FOM)
best=index[np.unravel_index(best,[ac_levels,dc_levels])]

plt.subplot(4, 1, 1)
plt.plot(np.transpose(timetube[best,:]), np.transpose(Idrive_tube[best,:]),'-')
plt.title('Current and Voltage: Parallel, 1pT flux')
plt.xlim(0,2*period)
plt.ylabel('Iex (A)')

plt.subplot(4, 1, 2)
plt.plot(np.transpose(Hz1_tube[best,:]), np.transpose(Bz1_tube[best,:]),'-b',np.transpose(Hz2_tube[best,:]), np.transpose(Bz2_tube[best,:]),'-r')
plt.ylabel('Bz (T)')

plt.xlabel('Hz (A/M)')


plt.subplot(4, 1, 3)
plt.plot(np.transpose(timetube[best,:]), np.transpose(Vtube1[best,:]),'-b',np.transpose(timetube[best,:]), np.transpose(Vtube2[best,:]),'-r')
plt.ylabel('Vsense')
plt.xlim(0,2*period)

plt.subplot(4, 1, 4)
plt.plot(np.transpose(timetube[best,:]), np.transpose(Vtube1[best,:])-np.transpose(Vtube2[best,:]),'-b')

plt.ylabel('Vsense')
plt.xlim(0,2*period)


plt.xlabel('time (s)')

plt.show()
                        
        
