# reference:
#  T. Betcke and L. N. Trefethen
#   "Reviving the Method of Particular Solutions"
#   SIAM Review 47 (2005) 469

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
from scipy.optimize import minimize_scalar
from scipy.linalg import qr, svd
from matplotlib.path import Path

def basis_func(lm, k, r, th):
    return np.sin(k*th) * jn(k, lm**0.5 * r)

class PolygonDrum:
    def __init__(self, v, lmin=1, lmax=25, lstep=0.2,
                 N=32, M=32, K=128, smin=1e-3, rmin=0):
        """
        v = array of vertices as complex numbers
        [lmin, lmax] = search range of eigen values
        lstep = searching step of eigen values
        N = number of basis functions
        M = number of sample points on one edge
        K = number of points scattered within polygon
        smin = minimum signular value to detect eigen value
        rmin = threshold for pivoting QR decomposition
               (pivoting is not used if rmin=0)
        """
        p = Path([(z.real, z.imag) for z in v])
        v = np.array(v)
        e1 = np.roll(v,-1) - v
        e2 = np.roll(v,1) - v

        a = 1/(np.angle(e2/e1)/np.pi % 2)
        u = np.isclose(a, np.around(a))
        u = np.arange(len(v))[~u]
        if len(u)==0: u = np.arange(len(v))

        self.path = p
        self.vertex = v
        self.pivot = u
        self.angle = a

        x = (np.arange(M) + 1)/(M+1)
        x = (1 - np.cos(np.pi*x))/2
        z = (v + x[:,np.newaxis]*e1).T.reshape(-1)
        M = len(z)

        x1,x2,y1,y2 = self.get_extents()
        x = x1 + (x2-x1)*np.random.rand(5*K)
        y = y1 + (y2-y1)*np.random.rand(5*K)
        D = np.stack((x,y), axis=-1)
        D = p.contains_points(D)

        z = np.hstack((z, (x+y*1j)[D][:K]))
        r,t = self.z2rt(z)
        r = r[...,np.newaxis]
        t = t[...,np.newaxis]

        ka = ((np.arange(N) + 1)*a[u,np.newaxis])

        def pdfun(lm):
            A = basis_func(lm, ka, r, t)

            Q,R,P = qr(A.reshape(len(z),-1),
                       mode='economic', pivoting=True)
            dR = np.abs(np.diag(R))
            K = np.count_nonzero(dR > rmin*dR[0])

            _,S,V = svd(Q[:M,:K])

            self.R = R[:K,:K]
            self.P = P[:K]
            self.S = S
            self.V = V
            return S[-1]
    
        pdfunv = np.vectorize(pdfun)
        L = int((lmax-lmin)/lstep)
        lm = np.linspace(lmin, lmax, L+1)
        S = pdfunv(lm)

        J = np.arange(1,L)
        J = J[(S[J] < S[J-1]) & (S[J] < S[J+1])]
        self.eigen_val = []
        self.multiplicity = []
        self.coeff = []

        for j in J:
            res = minimize_scalar(pdfun, (lm[j-1], lm[j], lm[j+1]))
            m = np.count_nonzero(self.S < smin)
            if m==0: continue
            c = np.zeros((len(u)*N, m))
            c[self.P] = np.linalg.solve(self.R, self.V[-m:].T)

            self.eigen_val.append(res.x)
            self.multiplicity.append(m)
            self.coeff.append(c.T)

        self.n_bases = N
        self.lm = lm
        self.svmin = S

    def eval(self,z,i=0,j=0):
        """
        input:
          z = array of points (as complex numbers) to evaluate eigen func
          i = eigen value index (i-th smallest)
          j = eigen func index in i-th degenerate eigen value
        return:
          eigen func normalized so that max = 1
        """
        if i not in range(len(self.eigen_val)):
            print('i not in range'); exit()

        lm = self.eigen_val[i]
        c = self.coeff[i][j]
        a = self.angle[self.pivot, np.newaxis]

        D = np.dstack((np.real(z), np.imag(z))).reshape(-1,2)
        D = self.path.contains_points(D).reshape(z.shape)

        r,t = self.z2rt(z[D])
        r = r[...,np.newaxis]
        t = t[...,np.newaxis]

        f = np.empty_like(z, dtype=np.float)
        k = np.arange(self.n_bases) + 1

        A = basis_func(lm, k*a, r, t)
        f[D] = np.dot(A.reshape(len(r),-1), c)
        f[~D] = np.nan

        return f / np.nanmax(np.abs(f))

    def get_extents(self):
        [[x1,y1],[x2,y2]] = self.path.get_extents().get_points()
        return (x1,x2,y1,y2)

    def z2rt(self,z):
        """ transform z to polar coordinates (r,theta) """
        v = self.vertex
        u = self.pivot
        e1 = np.roll(v,-1) - v

        w = z[...,np.newaxis] - v[u]
        r = np.abs(w)
        t = np.angle(w/(e1[u]))

        tmin = np.angle((v[:,np.newaxis] - v[u])/e1[u])
        for (i,j) in enumerate(u):
            tmin[:,i] = np.roll(tmin[:,i], -j)

        tmin = np.min(np.unwrap(tmin[1:], axis=0), axis=0)
        t = (t - tmin)%(2*np.pi) + tmin

        return r,t

    def plot_boundary(self, *a, **k):
        """
        a = argemuents passed to plt.plot
        k = keyword arguments passed to plt.plot
        """
        v = np.append(self.vertex, self.vertex[0])
        plt.plot(np.real(v), np.imag(v), *a, **k)
