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

#Constants:
R = 7e-2 #m
n = 2 #turns
Bm = 1e-12 #T
I = 1e-12*(5*np.sqrt(5)/8)*R/(MU0*n) #A
print "I: ",I
nGrid = 100
x = np.linspace(0,R,nGrid)
B1 = MU0*n*I*R*R/2*(1/np.power(R*R+x*x,1.5))
B2 = MU0*n*I*R*R/2*(1/np.power(R*R+(R-x)*(R-x),1.5))
B=B1+B2
plt.subplot(1, 1, 1)
plt.plot(x, B, 'r.-')
plt.title('Helmholtz flux')
plt.ylabel('B (T)')
plt.xlabel('x (m)')
plt.xlim(0,R)


plt.show()

                        
        
