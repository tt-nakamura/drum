import numpy as np
import matplotlib.pyplot as plt
from PolygonDrum import PolygonDrum

v = [0, 1j, -1+1j, -1-2j, -2j, -1j, 1-1j, 1-2j, 2-2j, 2+1j, 1+1j, 1]

pd = PolygonDrum(v, lmax=42, N=40, M=40, smin=1e-4, rmin=1e-12)

x,y = np.meshgrid(np.linspace(-1,2,33), np.linspace(-2,1,33))
levels = np.linspace(-1,1,21)

plt.figure(figsize=(6,2))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

i=0; n=1
while n<=12 and i < len(pd.eigen_val):
    j=0
    while n<=12 and j < pd.multiplicity[i]:
        f = pd.eval(x+y*1j,i,j)
        plt.subplot(2,6,n);
        plt.contourf(x,y,f,levels,cmap='RdBu_r')
        pd.plot_boundary('k', lw=1)
        plt.axis('equal')
        plt.axis('off')
        plt.axis([-1.02,2.02,-2.02,1.02])
        plt.text(0.5, 1.05,
                 '$\lambda_{%d}$ = %.4f' %(i+1, pd.eigen_val[i]),
                 va='bottom', ha='center', fontsize=6)
        n += 1
        j += 1
    i += 1

plt.savefig('fig10.eps')
plt.show()
