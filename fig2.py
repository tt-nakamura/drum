import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros

mn = [(0,0), (1,0), (1,1), (1,2), (2,3), (2,4)]
x,y = np.meshgrid(np.linspace(-1,1,41), np.linspace(-1,1,41))
levels = np.linspace(-1,1,21)
circle = np.exp(np.linspace(0,2*np.pi,51)*1j)

plt.figure(figsize=(3,2))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

for i,(m,n) in enumerate(mn):
    plt.subplot(2,3,i+1)
    xmn = jn_zeros(n,m+1)
    r,t = np.sqrt(x**2+y**2), np.arctan2(y,x)
    z = jn(n, r*xmn[m]) * np.cos(n*t)
    z /= np.max(z)
    z[r>1] = np.nan
    plt.contourf(x,y,z,levels,cmap='RdBu_r')
    plt.plot(np.real(circle), np.imag(circle), 'k', lw=1)
    plt.axis('equal')
    plt.axis('off')
    plt.axis([-1.04,1.04,-1.04,1.04])
    plt.text(0, 1.03, '(m,n) = (%d,%d)' %(m,n),
             va='bottom', ha='center', fontsize=6)

plt.savefig('fig2.eps')
plt.show()
