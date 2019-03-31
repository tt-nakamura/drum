import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from scipy.optimize import minimize_scalar
from Ldrum import Ldrum

ld = Ldrum(lstep=0.08)

plt.figure(figsize=(6.4, 3.6))

plt.plot(ld.lm, ld.svmin, 'k')
plt.axis([0,25,0,1])
plt.xlabel('$\lambda$', fontsize=14)
plt.ylabel(r'minimun singular value of $\tilde Q$', fontsize=14)

plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()
