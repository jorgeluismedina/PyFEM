
import numpy as np
import scipy as sp


def check_symmetric(a, rtol=1e-10, atol=1e-11):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_conver(resid, force, tol):
    resid = sp.linalg.norm(resid)
    force = sp.linalg.norm(force)
    ratio = resid/force
    return ratio <= tol


def tangencial_stiff(struct, facto, nincs=100, max_iter=10, tol=0.01):
    displacements = np.zeros((nincs+1, struct.ndofs))
    applied_loads = np.zeros((nincs+1, struct.ndofs))
    free_dof = struct.fr_dof
    tfact = 0.0

    astif = struct.assemb_global_stiff() #stiffness matrix (H)
    disps = np.zeros(free_dof.shape[0]) #displacement guess (phi)
    tload = struct.re_tload #total force (f)

    for step in range(nincs):
        if tfact >= 1.0:
            break
        
        print(f"load step: {step+1}")
        force = facto*tload
        unbalance = False

        for elem in struct.elems:
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
            astif = struct.update_global_stiff(disps)
        
        if not unbalance:
            astif = struct.update_global_stiff(disps)

        displacements[step+1][free_dof] = disps
        applied_loads[step+1][free_dof] = force
        tfact += facto
    
    cum_displacements = np.cumsum(displacements[:step+1], axis=0)
    cum_applied_loads = np.cumsum(applied_loads[:step+1], axis=0)
    return cum_displacements, cum_applied_loads
