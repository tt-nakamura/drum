import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

x = [0, 0, -1, -1, 1, 1, 0]
y = [0, 1, 1, -1, -1, 0, 0]
M = 64
K = 64

np.random.seed(100)

t1 = 1.5*np.pi*(np.arange(M) + 0.5)/M
r1 = 1/np.fmax(np.abs(np.sin(t1)), np.abs(np.cos(t1)))
t2 = 1.5*np.pi*np.random.rand(K)
r2 = 1/np.fmax(np.abs(np.sin(t2)), np.abs(np.cos(t2)))
r2 *= np.random.rand(K)
z1 = r1 * np.exp(t1*1j)*1j
z2 = r2 * np.exp(t2*1j)*1j

plt.figure(figsize=(6,6))

plt.plot(x, y, 'k')
plt.plot(np.real(z1), np.imag(z1), 'k.')
plt.plot(np.real(z2), np.imag(z2), 'ko', fillstyle='none')

plt.text(x[0], y[0], 'P$_1$', va='bottom', ha='left', fontsize=16)
plt.text(x[1], y[1], 'P$_2$', va='bottom', ha='left', fontsize=16)
plt.text(x[2], y[2], 'P$_3$', va='bottom', ha='right', fontsize=16)
plt.text(x[3], y[3], 'P$_4$', va='top', ha='right', fontsize=16)
plt.text(x[4], y[4], 'P$_5$', va='top', ha='left', fontsize=16)
plt.text(x[5], y[5], 'P$_6$', va='bottom', ha='left', fontsize=16)

plt.axis('off')
plt.axis('equal')
plt.axis([-1.01,1.01,-1.01,1.01])

plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()

