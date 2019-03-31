# reference:
#  T. Betcke and L. N. Trefethen
#   "Reviving the Method of Particular Solutions"
#   SIAM Review 47 (2005) 469

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from scipy.optimize import minimize_scalar

def basis_func(lm, k, r, th):
    return np.sin(k*th) * jn(k, lm**0.5 * r)

class EllipLdrum:
    def __init__(self, a, b, lmin=1, lmax=25, lstep=0.2,
                 N=32, M=64, K=64, smin=1e-3):
        """
        a = semi major axis
        b = semi minor axis
        [lmin, lmax] = search range of eigen values
        lstep = searching step of eigen values
        N = number of basis functions
        M = number of sample points on boundary
        K = number of internal sample points
        smin = minimum signular value to detect eigen value
        """
        t1 = 1.5*np.pi*(np.arange(M) + 0.5)/M
        r1 = 1/np.sqrt((np.sin(t1)/a)**2 + (np.cos(t1)/b)**2)
        t2 = 1.5*np.pi*np.random.rand(K)
        r2 = 1/np.sqrt((np.sin(t2)/a)**2 + (np.cos(t2)/b)**2)
        r2 *= np.random.rand(K)
        t = np.hstack((t1,t2))[:,np.newaxis]
        r = np.hstack((r1,r2))[:,np.newaxis]

        k = 2/3*(np.arange(N) + 1)

        def ldfun(lm):
            A = basis_func(lm, k, r, t)

            Q,R = np.linalg.qr(A)
            _,S,V = np.linalg.svd(Q[:M])

            self.R = R
            self.S = S
            self.V = V
            return S[-1]
    
        ldfunv = np.vectorize(ldfun)
        L = int((lmax-lmin)/lstep)
        lm = np.linspace(lmin, lmax, L+1)
        S = ldfunv(lm)

        J = np.arange(1,L)
        J = J[(S[J] < S[J-1]) & (S[J] < S[J+1])]
        self.eigen_val = []
        self.multiplicity = []
        self.coeff = []

        for j in J:
            res = minimize_scalar(ldfun, (lm[j-1], lm[j], lm[j+1]))
            m = np.count_nonzero(self.S < smin)
            if m==0: continue
            c = np.linalg.solve(self.R, self.V[-m:].T)

            self.eigen_val.append(res.x)
            self.multiplicity.append(m)
            self.coeff.append(c.T)

        self.n_bases = N
        self.lm = lm
        self.svmin = S
        self.a = a
        self.b = b

    def eval(self,x,y,i=0,j=0):
        """
        input:
          (x,y) = array of points to evaluate eigen func
          i = eigen value index (i-th smallest)
          j = eigen func index in i-th degenerate eigen value
        return:
          eigen func normalized so that max = 1
        """
        if i not in range(len(self.eigen_val)):
            print('i not in range'); exit()

        lm = self.eigen_val[i]
        c = self.coeff[i][j]

        t = (np.arctan2(y,x) - np.pi/2) % (2*np.pi)
        D = (t <= 1.5*np.pi)
        D &= ((x/self.a)**2 + (y/self.b)**2 <= 1)
        r = np.sqrt(x[D]**2 + y[D]**2)
        r = r[:,np.newaxis]
        t = t[D][:,np.newaxis]

        f = np.empty_like(x)
        k = np.arange(self.n_bases) + 1

        f[D] = np.dot(basis_func(lm, 2/3*k, r, t), c)
        f[~D] = np.nan
        return f / np.nanmax(np.abs(f))

    def plot_boundary(self, *a, **k):
        """
        a = argemuents passed to plt.plot
        k = keyword arguments passed to plt.plot
        """
        t = np.linspace(np.pi/2, np.pi*2, 61)
        plt.plot(self.a*np.cos(t), self.b*np.sin(t), *a, **k)
        plt.plot([self.a,0,0],[0,0,self.b], *a, **k)
