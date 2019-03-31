import numpy as np
import matplotlib.pyplot as plt
from PolygonDrum import PolygonDrum

v = [np.exp(t*2*np.pi*1j) for t in np.arange(6)/6]

pd = PolygonDrum(v, lmax=72, rmin=1e-8)

x,y = np.meshgrid(np.linspace(-1,1,33), np.linspace(-1,1,33))
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
        plt.axis([-1,1,-1,1])
        plt.text(0, 0.88,
                 '$\lambda_{%d}$ = %.4f' %(i+1, pd.eigen_val[i]),
                 va='bottom', ha='center', fontsize=6)
        n += 1
        j += 1
    i += 1

plt.savefig('fig11.eps')
plt.show()
