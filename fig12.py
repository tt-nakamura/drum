# reference:
#  T. Betcke and L. N. Trefethen
#   "Computations of Eigenvalue Avoidance in Planar Domains"
#   Proceedings of the Applied Mathematics and Mechanics 4 (2004) 634

import numpy as np
import matplotlib.pyplot as plt
from PolygonDrum import PolygonDrum

L = np.geomspace(0.5, 2, 101)

plt.figure(figsize=(6.4, 4))

plt.subplot(1,2,1)

mn = [(1,1), (1,2), (2,1), (1,3), (3,1),
      (2,2), (3,3), (1,4), (4,1)]

for m,n in mn: 
    l = np.pi**2*((m*L)**2 + (n/L)**2)
    plt.semilogx(L, l, 'k')

plt.axis([0.5,2,0,100])
plt.xlabel('L', fontsize=14)
plt.ylabel('eigen value', fontsize=14)

##############################################
plt.subplot(1,2,2)

p = 0.2
n = 5
lm = np.empty((len(L), n))

for i,l in enumerate(L):
    v = [0, l, l-p+1j/l, 1j/l]
    pd = PolygonDrum(v, lmin=20, lmax=135, lstep=0.8)
    print(l, pd.eigen_val)
    lm[i] = pd.eigen_val[:n]

plt.semilogx(L, lm, 'k')
plt.axis([0.5,2,0,100])
plt.xlabel('L', fontsize=14)

plt.tight_layout()
plt.savefig('fig12.eps')
plt.show()
