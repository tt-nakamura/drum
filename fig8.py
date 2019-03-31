import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from scipy.optimize import minimize_scalar
from PolygonDrum import PolygonDrum

v1 = [-1-1j, 1-1j, 1-3j, 3-1j, 3+1j, -1+1j, -1+3j, -3+1j]
v2 = [1+1j, -1+1j, -1+3j, -3+3j, -3+1j, 1-3j, 1-1j, 3-1j]

pd1 = PolygonDrum(v1, lmax=10, lstep=0.08)
pd2 = PolygonDrum(v2, lmax=10, lstep=0.08)

print(pd1.eigen_val)
print(pd2.eigen_val)

plt.figure(figsize=(6.4, 3.6))

plt.plot(pd1.lm, pd1.svmin, 'k', label='GWW1')
plt.plot(pd2.lm, pd2.svmin, 'k--', label='GWW2')
plt.axis([1,10,0,0.5])
plt.xlabel('$\lambda$', fontsize=14)
plt.ylabel(r'minimun singular value of $\tilde Q$', fontsize=14)
plt.legend()

plt.tight_layout()
plt.savefig('fig8.eps')
plt.show()
