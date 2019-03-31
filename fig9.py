import numpy as np
import matplotlib.pyplot as plt
from PolygonDrum import PolygonDrum

v1 = [-1-1j, 1-1j, 1-3j, 3-1j, 3+1j, -1+1j, -1+3j, -3+1j]
v2 = [1+1j, -1+1j, -1+3j, -3+3j, -3+1j, 1-3j, 1-1j, 3-1j]

pd1 = PolygonDrum(v1, lmax=17, N=40, M=40, rmin=1e-12)
pd2 = PolygonDrum(v2, lmax=17, N=48, M=48, rmin=1e-12)

x,y = np.meshgrid(np.linspace(-3,3,33), np.linspace(-3,3,33))
levels = np.linspace(-1,1,21)

plt.figure(figsize=(6,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

i=0; n=1
while n<=12 and i < len(pd1.eigen_val):
    j=0
    while n<=12 and j < pd1.multiplicity[i]:
        f = pd1.eval(x+y*1j,i,j)
        plt.subplot(4,6,2*n-1);
        plt.contourf(x,y,f,levels,cmap='RdBu_r')
        pd1.plot_boundary('k', lw=1)
        plt.axis('equal')
        plt.axis('off')
        plt.axis([-3.03,3.03,-3.03,3.03])
        plt.text(0, 3.05,
                 '$\lambda_{%d}$ = %.4f' %(i+1, pd1.eigen_val[i]),
                 va='bottom', ha='center', fontsize=6)

        f = pd2.eval(x+y*1j,i,j)
        plt.subplot(4,6,2*n);
        plt.contourf(x,y,f,levels,cmap='RdBu_r')
        pd2.plot_boundary('k', lw=1)
        plt.axis('equal')
        plt.axis('off')
        plt.axis([-3.03,3.03,-3.03,3.03])
        plt.text(0, 3.05,
                 '$\lambda_{%d}$ = %.4f' %(i+1, pd2.eigen_val[i]),
                 va='bottom', ha='center', fontsize=6)
        n += 1
        j += 1
    i += 1

plt.savefig('fig9.eps')
plt.show()
