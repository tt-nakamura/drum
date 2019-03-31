import numpy as np
import matplotlib.pyplot as plt

mn = [(1,1), (2,1), (2,3)]
x,y = np.meshgrid(np.linspace(0,1,33), np.linspace(0,1,33))
levels = np.linspace(-1,1,21)

plt.figure(figsize=(3,1))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

for i,(m,n) in enumerate(mn):
    plt.subplot(1,3,i+1)
    z = np.sin(np.pi*m*x) * np.sin(np.pi*n*y)
    plt.contourf(x,y,z,levels,cmap='RdBu_r')
    plt.plot([0,1,1,0,0],[0,0,1,1,0],'k',lw=1)
    plt.axis('equal')
    plt.axis('off')
    plt.axis([-0.02,1.02,-0.02,1.02])
    plt.text(0.5, 1.03, '(m,n) = (%d,%d)' %(m,n), fontsize=6,
             va='bottom', ha='center')

plt.savefig('fig1.eps')
plt.show()
