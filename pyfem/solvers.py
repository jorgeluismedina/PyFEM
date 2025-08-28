
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import scipy as sp


def check_symmetric(a, rtol=1e-6, atol=1e-5):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_conver(resid, force, tol):
    resid = sp.linalg.norm(resid)
    force = sp.linalg.norm(force)
    ratio = resid/force
    return ratio <= tol


def solve_linear_static(model): #estatic

    model.set_restraints()
    fixd_dof = model.fixd_dof
    free_dof = model.free_dof

    glob_stiff = model.assemb_global_stiff()
    glob_loads = model.assemb_global_loads()
    #glob_loads = model.glob_loads
    glob_disps = model.glob_disps

    # Reduccion del sistema
    stiff_ff = glob_stiff[np.ix_(free_dof, free_dof)]
    stiff_sf = glob_stiff[np.ix_(fixd_dof, free_dof)]
    glob_loads -= glob_stiff[:,fixd_dof] @ glob_disps[fixd_dof] # desplazamientos impuestos -> fuerzas

    # Resolucion del Sistema
    #if check_symmetric(stiff_ff):
    #free_disps = sp.linalg.solve(stiff_ff, glob_loads[free_dof], assume_a = 'sym')
    free_disps = cho_solve(cho_factor(stiff_ff), glob_loads[free_dof])
    glob_disps[free_dof] = free_disps
    glob_react = stiff_sf @ free_disps - glob_loads[fixd_dof]

    return glob_disps, glob_react

def vibration_modes(model):

    model.set_restraints()
    #fixd_dof = model.fixd_dof
    free_dof = model.free_dof
    K = model.assemb_global_stiff()
    M = model.assemb_global_mass()
    Kff = K[np.ix_(free_dof, free_dof)]
    Mff = M[np.ix_(free_dof, free_dof)]

    # Calculo de los autovectores y autovalores
    w2, Phi = sp.linalg.eig(Kff, Mff)

    # Ordenamiento de autovalores de manera creciente
    iw = w2.argsort()
    w2 = w2[iw]
    Phi = Phi[:,iw]

    #Frecuencias naturales
    wk = np.sqrt(np.real(w2))
    fk = wk/2/np.pi # [Hz]

    return fk, Phi
 
#'''
def finite_differences(model, t0, tf, dt, x0, v0, a0):

    sol = []
    nsteps = (tf - t0) / dt

    M = model.assemb_global_mass()
    K = model.assemb_global_stiff()
    F = model.assemb_global_loads()
    C = M + K

    aux = F - C@v0 - K@x0
    a0 = sp.linalg.solve(M, aux, assume_a='sym')
    x_back = dt*dt*a0 / 2 - dt*v0 + x0
    x_cent = np.zeros_like(x0)

    for ti in range(nsteps):
        MC = M / (dt*dt) + C / (2*dt)
        aux2 = F - K
        x_ford = sp.linal.solve(MC, F)


    #v = x0
    #a

#'''

def tangencial_stiff(model, facto, nincs=100, max_iter=10, tol=0.01):
    displacements = np.zeros((nincs+1, model.ndofs))
    applied_loads = np.zeros((nincs+1, model.ndofs))
    free_dof = model.fr_dof
    tfact = 0.0

    astif = model.assemb_global_stiff() #stiffness matrix (H)
    disps = np.zeros(free_dof.shape[0]) #displacement guess (phi)
    tload = model.re_tload #total force (f)

    last_step = 0
    for step in range(nincs):
        if tfact >= 1.0: # tfact es el ratio total de carga que se ha aplicado hasta ahora
            last_step = step
            break
        
        print(f"load step: {step+1}")
        force = facto*tload
        unbalance = False

        for elem in model.elems:
            print(elem.yielded)

        for ite in range(max_iter): 
            resid = astif @ disps + force # residual (psi)

            if check_conver(resid, force, tol):
                print(f"    convergence reached in iteration: {ite}")
                if ite>=3:
                    facto = facto*0.5
                break

            unbalance = True
            # phi = phi + delta_phi
            disps += -sp.linalg.solve(astif, resid, assume_a='sym')
            astif = model.update_global_stiff(disps)
        
        if not unbalance:
            astif = model.update_global_stiff(disps)

        displacements[step+1][free_dof] = disps
        applied_loads[step+1][free_dof] = force
        tfact += facto
    
    cum_displacements = np.cumsum(displacements[:last_step+1], axis=0)
    cum_applied_loads = np.cumsum(applied_loads[:last_step+1], axis=0)
    return cum_displacements, cum_applied_loads
