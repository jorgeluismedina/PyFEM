
import numpy as np
from itertools import product


"""
Return Gauss points and weights 
for Gauss quadrature in 1D.

Parameters
----------
npts : (int) : Number of quadrature points.

Returns
-------
wts : (ndarray) : Weights for the Gauss-Legendre quadrature.
pts : (ndarray) : Points for the Gauss-Legendre quadrature.
"""
def gauss_legendre(npts):
    quadrature_data = {2:{'pts': [-0.577350269189625764, 0.577350269189625764],
                          'wts': [1.00000000000000000, 1.00000000000000000]},

                       3:{'pts': [-0.774596669241483377, 0, 0.774596669241483377],
                          'wts': [0.555555555555555556, 0.888888888888888889,
                                  0.555555555555555556]},

                       4:{'pts': [-0.861136311594052575, -0.339981043584856265,
                                  0.339981043584856265, 0.861136311594052575],
                          'wts': [0.347854845137453857, 0.652145154862546143,
                                  0.652145154862546143, 0.347854845137453857]},

                       5:{'pts': [-0.906179845938663993, -0.538469310105683091, 0,
                                  0.538469310105683091, 0.906179845938663993],
                          'wts': [0.236926885056189088, 0.478628670499366468,
                                  0.568888888888888889, 0.478628670499366468,
                                  0.236926885056189088]},

                       6:{'pts': [-0.932469514203152028, -0.661209386466264514,
                                  -0.238619186083196909, 0.238619186083196909,
                                  0.661209386466264514, 0.932469514203152028],
                          'wts': [0.171324492379170345, 0.360761573048138608,
                                  0.467913934572691047, 0.467913934572691047,
                                  0.360761573048138608, 0.171324492379170345]},

                       7:{'pts': [-0.949107912342758525, -0.741531185599394440,
                                  -0.405845151377397167, 0, 0.405845151377397167,
                                  0.741531185599394440, 0.949107912342758525],
                          'wts': [0.129484966168869693, 0.279705391489276668,
                                  0.381830050505118945, 0.417959183673469388,
                                  0.381830050505118945, 0.279705391489276668,
                                  0.129484966168869693]},

                       8:{'pts': [-0.960289856497536232, -0.796666477413626740,
                                  -0.525532409916328986, -0.183434642495649805,
                                  0.183434642495649805, 0.525532409916328986,
                                  0.796666477413626740, 0.960289856497536232],
                          'wts': [0.101228536290376259, 0.222381034453374471,
                                  0.313706645877887287, 0.362683783378361983,
                                  0.362683783378361983, 0.313706645877887287,
                                  0.222381034453374471, 0.101228536290376259]},

                       9:{'pts': [-0.968160239507626090, -0.836031107326635794,
                                  -0.613371432700590397, -0.324253423403808929, 0,
                                  0.324253423403808929, 0.613371432700590397,
                                  0.836031107326635794, 0.968160239507626090],
                          'wts': [0.0812743883615744120, 0.180648160694857404,
                                  0.260610696402935462, 0.312347077040002840,
                                  0.330239355001259763, 0.312347077040002840,
                                  0.260610696402935462, 0.180648160694857404,
                                  0.0812743883615744120]},

                       10:{'pts': [-0.973906528517171720, -0.865063366688984511,
                                   -0.679409568299024406, -0.433395394129247191,
                                   -0.148874338981631211, 0.148874338981631211,
                                   0.433395394129247191, 0.679409568299024406,
                                   0.865063366688984511, 0.973906528517171720],
                           'wts': [0.0666713443086881376, 0.149451349150580593,
                                   0.219086362515982044, 0.269266719309996355,
                                   0.295524224714752870, 0.295524224714752870,
                                   0.269266719309996355, 0.219086362515982044,
                                   0.149451349150580593, 0.0666713443086881376]}}
    if npts not in range(2, 11):
        raise ValueError("The number of points should be in [2, 10]")
    points, weights = quadrature_data[npts].values()
    return points, weights




class Gauss_Legendre(): #primero simetrico
    def __init__(self, npts, ndim):
        self.get_points_weights(npts, ndim)
        self.dim = ndim

    """
    Returns points and weights for Gauss quadrature in
    an ND hypercube using products from one-dimensional quadrature scheme.

    Parameters
    ----------
    npts : (int) : Number of sample points.

    Returns
    -------
    nd_wts : (ndarray) : Weights for the Gauss-Legendre quadrature.
    nd_pts : (ndarray) : Points for the Gauss-Legendre quadrature.
    """
    def get_points_weights(self, npts, ndim):
        pts, wts = gauss_legendre(npts)
        nd_pts = np.array(list(product(pts, repeat=ndim)))
        nd_wts = np.prod(np.array(list(product(wts, repeat=ndim))), axis=1)
        self.points = nd_pts
        self.weights = nd_wts
        self.npoin = self.points.shape[0]
    
    """
    Compute an integral of a function over a hypercube using Gaussian quadrature of order n.
    Return the (definite) integral of func(x1,x2,...) from x1=x2=...=xn = [-1..1] 

    Parameters
    ----------
    func : (callable) : A Python vector-valued function or method to integrate.
    npts : (int, optional): Number of sampling points. Default is 5.
    ndim : (int, optional): Number of dimensions of the hypercube. Default is 2 (quadrlateral).

    Returns
    -------
    val : (float, array-like) : Gaussian quadrature approximation to the integral.
    """
    def integrate(self, func):
        return  np.sum(self.weights * func(*self.points.T), axis=-1)