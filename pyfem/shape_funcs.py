
import numpy as np

# r = xi
# s = eta
# t = zeta

# FUNCIONES DE FORMA EN SENTIDO ANTIHORARIO
# COLOCAR CONECTIVIDADES EN SENTIDO ANTIHORARIO
def shape_quad4(r, s):
    omr = 1 - r
    oms = 1 - s
    opr = 1 + r
    ops = 1 + s

    shape = np.empty(4 ,dtype=float)
    shape[0] = 0.25 * omr * oms
    shape[1] = 0.25 * opr * oms
    shape[2] = 0.25 * opr * ops
    shape[3] = 0.25 * omr * ops

    return shape

def deriv_quad4(r, s):
    f1 = 0.25 * (1 - r)
    f2 = 0.25 * (1 + r)
    f3 = 0.25 * (1 - s)
    f4 = 0.25 * (1 + s)

    deriv = np.empty((2,4), dtype=float)
    deriv[0,0] = -f3
    deriv[0,1] =  f3
    deriv[0,2] =  f4
    deriv[0,3] = -f4
    deriv[1,0] = -f1
    deriv[1,1] = -f2
    deriv[1,2] =  f2
    deriv[1,3] =  f1

    return deriv



# FUNCIONES DE FORMA EN SENTIDO ANTIHORARIO
# COLOCAR CONECTIVIDADES EN SENTIDO ANTIHORARIO
def shape_quad8(r, s):
    omr = 1 - r
    opr = 1 + r
    oms = 1 - s
    ops = 1 + s
    omrr = 1 - r*r
    omss = 1 - s*s

    shape = np.empty(4 ,dtype=float)
    shape[0] = 0.25 * omr  * oms * (-r - s - 1) 
    shape[1] = 0.50 * omr  * omss
    shape[2] = 0.25 * omr  * ops * (-r - s - 1)
    shape[3] = 0.50 * omrr * ops
    shape[4] = 0.25 * opr  * ops * ( r + s - 1)
    shape[5] = 0.50 * opr  * omss
    shape[6] = 0.25 * opr  * oms * ( r - s - 1)
    shape[7] = 0.50 * omrr * oms

    return shape

def deriv_quad8(r, s):
    omr = 1 - r
    opr = 1 + r
    oms = 1 - s
    ops = 1 + s
    omrr = 1 - r*r
    omss = 1 - s*s

    deriv = np.empty((2,8), dtype=float)
    deriv[0,0] =  0.25 * oms * (2*r + s)
    deriv[0,1] = -0.50 * omss
    deriv[0,2] =  0.25 * ops * (2*r - s)
    deriv[0,3] = -r * ops
    deriv[0,4] =  0.25 * ops * (2*r + s)
    deriv[0,5] =  0.50 * omss
    deriv[0,6] =  0.25 * oms * (2*r - s)
    deriv[0,7] = -r * oms

    deriv[1,0] =  0.25 * omr * (r + 2*s)
    deriv[1,1] = -s * omr
    deriv[1,2] = -0.25 * omr * (r - 2*s)
    deriv[1,3] =  0.50 * omrr
    deriv[1,4] =  0.25 * opr * (r + 2*s)
    deriv[1,5] = -s * opr
    deriv[1,6] = -0.25 * opr * (r - 2*s)
    deriv[1,7] = -0.50 * omrr

    return deriv



# FUNCIONES DE FORMA EN SENTIDO ANTIHORARIO
# COLOCAR CONECTIVIDADES EN SENTIDO ANTIHORARIO

def shape_tri3(r, s):
    shape = np.empty(3, dtype=float)
    shape[0] = 1 - r - s
    shape[1] = r
    shape[2] = s

    return shape

def deriv_tri3():
    deriv = np.empty((2,3), dtype=float)

    deriv[0,0] = -1
    deriv[0,1] =  1
    deriv[0,2] =  0
    deriv[1,0] = -1
    deriv[1,1] =  0
    deriv[1,2] =  1

    return deriv



# FUNCIONES DE FORMA SIGUIENDO REGLA DE LA MANO DERECHA
# COLOCAR CONECTIVIDADES SEGUN EL EJE ETA (s) DE REFERENCIA

def shape_hex8(r, s, t):
    omr = 1 - r
    opr = 1 + r
    oms = 1 - s
    ops = 1 + s
    omt = 1 - t
    opt = 1 + t

    shape = np.empty(8, dtype=float)
    shape[0] = 0.125 * omr * oms * omt
    shape[1] = 0.125 * omr * oms * opt
    shape[2] = 0.125 * opr * oms * opt
    shape[3] = 0.125 * opr * oms * omt
    shape[4] = 0.125 * omr * ops * omt
    shape[5] = 0.125 * omr * ops * opt
    shape[6] = 0.125 * opr * ops * opt
    shape[7] = 0.125 * opr * ops * omt

    return shape

def deriv_hex8(r, s, t):
    omr = 1 - r
    opr = 1 + r
    oms = 1 - s
    ops = 1 + s
    omt = 1 - t
    opt = 1 + t

    deriv = np.empty((3,8), dtype=float)

    f1 = 0.125 * oms * omt
    f2 = 0.125 * oms * opt
    f3 = 0.125 * ops * omt
    f4 = 0.125 * ops * opt

    deriv[0,0] = -f1
    deriv[0,1] = -f2
    deriv[0,2] =  f2
    deriv[0,3] =  f1
    deriv[0,4] = -f3
    deriv[0,5] = -f4
    deriv[0,6] =  f4
    deriv[0,7] =  f3

    f5 = 0.125 * omr * omt
    f6 = 0.125 * omr * opt
    f7 = 0.125 * opr * omt
    f8 = 0.125 * opr * opt

    deriv[1,0] = -f5
    deriv[1,1] = -f6
    deriv[1,2] = -f8
    deriv[1,3] = -f7
    deriv[1,4] =  f5
    deriv[1,5] =  f6
    deriv[1,6] =  f8
    deriv[1,7] =  f7

    f9  = 0.125 * omr * oms
    f10 = 0.125 * omr * ops
    f11 = 0.125 * opr * oms
    f12 = 0.125 * opr * ops

    deriv[2,0] = -f9
    deriv[2,1] =  f9
    deriv[2,2] =  f11
    deriv[2,3] = -f11
    deriv[2,4] = -f10
    deriv[2,5] =  f10
    deriv[2,6] =  f12
    deriv[2,7] = -f12

    return deriv



def bubble_hex8(r, s, t):
    bubble = np.empty(3, dtype=float)
    bubble[1] = 1 - r*r
    bubble[2] = 1 - s*s
    bubble[3] = 1 - t*t
    return bubble

def deriv_bubble_hex8(r, s, t):
    deriv_bubble = np.zeros((3,3), dtype=float)
    deriv_bubble[1,1] = -2*r
    deriv_bubble[2,2] = -2*s
    deriv_bubble[3,3] = -2*t
    return deriv_bubble
