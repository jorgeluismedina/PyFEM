
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import scipy as sp
from scipy.sparse.linalg import spsolve



def solve_linear_static(model): # dense matrix

    model.set_restraints()
    fixd_dof = model.fixd_dof
    free_dof = model.free_dof

    glob_stiff = model.assemb_global_stiff()
    glob_loads = model.assemb_global_loads()
    glob_disps = model.assemb_global_disps()

    # Reduccion del sistema
    stiff_ff = glob_stiff[np.ix_(free_dof, free_dof)]
    stiff_sf = glob_stiff[np.ix_(fixd_dof, free_dof)]
    glob_loads -= glob_stiff[:,fixd_dof] @ glob_disps[fixd_dof] # desplazamientos impuestos -> fuerzas

    # Resolucion del Sistema
    # Cholesky solo va a funcionar cuando la matriz sea Sym-Pos-definite
    # los nodos tienen que estar en sentido antihorario para que sea SPD
    free_disps = cho_solve(cho_factor(stiff_ff), glob_loads[free_dof])
    glob_disps[free_dof] = free_disps
    glob_react = stiff_sf @ free_disps - glob_loads[fixd_dof]

    return glob_disps, glob_react


def solve_linear_static2(model): # dense matrix

    model.set_restraints()
    fixd_dof = model.fixd_dof
    free_dof = model.free_dof

    glob_stiff = model.assemb_global_stiff_sparse() # sparse
    glob_loads = model.assemb_global_loads() # dense
    glob_disps = model.assemb_global_disps() # dense

    # Reduccion del sistema
    stiff_ff = glob_stiff[free_dof, :][:, free_dof].tocsc() # a formato csc para resolucion
    stiff_sf = glob_stiff[fixd_dof, :][:, free_dof]
    glob_loads -= glob_stiff[:, fixd_dof].dot(glob_disps[fixd_dof]) # desplazamientos impuestos -> fuerzas

    # Resolucion del Sistema
    free_disps = spsolve(stiff_ff, glob_loads[free_dof])
    glob_disps[free_dof] = free_disps
    glob_react = stiff_sf.dot(free_disps) - glob_loads[fixd_dof]

    return glob_disps, glob_react



def check_symmetric(a, rtol=1e-6, atol=1e-5):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_conver(resid, force, tol):
    resid = sp.linalg.norm(resid)
    force = sp.linalg.norm(force)
    ratio = resid/force
    return ratio <= tol


def vibration_modes(Kff, Mff):
    # Calculo de los autovectores y autovalores
    w2, Phi = sp.linalg.eig(Kff, Mff)

    # Ordenamiento de autovalores de manera creciente
    iw = w2.argsort()
    w2 = w2[iw]
    Phi = Phi[:,iw]

    #Frecuencias naturales
    wk = np.sqrt(np.real(w2))
    fk = wk/2/np.pi # [Hz]

    return fk, wk, Phi
 
#'''
def finite_differences(model, tspan, F, dt):
    # Matrices del sistema
    model.set_restraints()
    free_dof = model.free_dof
    M0 = model.assemb_global_mass()
    K0 = model.assemb_global_stiff()
    F0 = model.assemb_global_loads()
    K = K0[np.ix_(free_dof, free_dof)]
    M = M0[np.ix_(free_dof, free_dof)]
    #F = F0[free_dof]
    fk, wk, Phi = vibration_modes(K, M)
    wki = wk[0];  zti = 0.01
    wkj = wk[1];  ztj = 0.01

    alpha = np.linalg.solve([[1/(2*wki), wki/2], 
                             [1/(2*wkj), wkj/2]], [zti, ztj])
    C = alpha[0]*M + alpha[1]*K
    
    # Iteraciones
    dt2 = dt*dt
    nsteps = int(tspan / dt) 
    #aux = F - C@v0 - K@x0 # velocidad y desplazamiento inicial = 0
    a0 = sp.linalg.solve(M, F[0][free_dof], assume_a='sym') # F <-- aux
    x_back = a0 * dt2 / 2 #- dt*v0 + x0
    x_cent = np.zeros(len(free_dof))
    sol = [x_cent]

    for step in range(nsteps):
        MC = M / (dt2) + C / (2*dt)
        aux = F[step][free_dof] - (K - 2/dt2 * M) @ x_cent - (M/dt2 - C*(2*dt)) @ x_back
        x_ford = sp.linalg.solve(MC, aux)
        sol.append(x_ford)
        x_back = x_cent
        x_cent = x_ford

    t = np.linspace(0, tspan, nsteps)
    return t, sol

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
