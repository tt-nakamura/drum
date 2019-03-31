import numpy as np
import matplotlib.pyplot as plt
from Ldrum import Ldrum

ld = Ldrum(lmax=130)
x,y = np.meshgrid(np.linspace(-1,1,33), np.linspace(-1,1,33))
levels = np.linspace(-1,1,21)

plt.figure(figsize=(6,4))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

i=0; n=1
while n<=24 and i < len(ld.eigen_val):
    j=0
    while n<=24 and j < ld.multiplicity[i]:
        f = ld.eval(x,y,i,j)
        plt.subplot(4,6,n); n+=1
        plt.contourf(x,y,f,levels,cmap='RdBu_r')
        ld.plot_boundary('k', lw=1)
        plt.axis('equal')
        plt.axis('off')
        plt.axis([-1.02,1.02,-1.02,1.02])
        plt.text(0, 1.03,
                 '$\lambda_{%d}$ = %.4f' %(i+1, ld.eigen_val[i]),
                 va='bottom', ha='center', fontsize=6)
        j += 1
    i += 1

plt.savefig('fig5.eps')
plt.show()
