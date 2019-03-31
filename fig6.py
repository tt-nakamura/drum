import numpy as np
import matplotlib.pyplot as plt
from EllipLdrum import EllipLdrum

a,b = 2,1
eld = EllipLdrum(a,b, lmax=65)
x,y = np.meshgrid(np.linspace(-a,a,41), np.linspace(-b,b,41))
levels = np.linspace(-1,1,21)

plt.figure(figsize=(3, 4.5))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

i=0; n=1
while n<=18 and i < len(eld.eigen_val):
    j=0
    while n<=18 and j < eld.multiplicity[i]:
        f = eld.eval(x,y,i,j)
        plt.subplot(6,3,n); n+=1
        plt.contourf(x,y,f,levels,cmap='RdBu_r')
        eld.plot_boundary('k', lw=1)
        plt.axis('equal')
        plt.axis('off')
        plt.axis([-a-0.02,a+0.02,-b-0.03,b+0.03])
        plt.text(0, b+0.1,
                 '$\lambda_{%d}$ = %.4f' %(i+1, eld.eigen_val[i]),
                 va='bottom', ha='center', fontsize=6)
        j += 1
    i += 1

plt.savefig('fig6.eps')
plt.show()
