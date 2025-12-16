
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import AreaSection
from pyfem.solvers import solve_linear_static



# Dimensiones de la placa
B = 4 #[m]
A = 4 #[m]
t = 0.1 #[m]
E = 2.08e10 #[N/m2]
nu = 0.3


# Mallado
xdivitions = 6
ydivitions = 6
nnelem = xdivitions * ydivitions

xnodes_l = xdivitions + 1
ynodes_l = ydivitions + 1
nnods = xnodes_l * ynodes_l

x = np.linspace(0,B,xnodes_l)
y = np.linspace(0,A,ynodes_l)
idx_nodes = np.arange(0,nnods).reshape((xnodes_l, ynodes_l))

X, Y = np.meshgrid(x,y)
coordinates = np.hstack([X.ravel()[:,None], 
                         Y.ravel()[:,None]])


# Materiales
concrete = Material(elast=E, poiss=nu, dense=1.0,
                    constitutive_model="plane_stress") #[N/m2]
materials = [concrete]

# Secciones
sect1 = AreaSection(thick=t) #[m2]
sections = [sect1]



# Creacion de Modelo
mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Elementos y conectividades
for ex in range(xdivitions):
    for ey in range(ydivitions):
        elem_grid = idx_nodes[ex:ex+2, ey:ey+2]
        conec = [elem_grid[0,0], elem_grid[0,1], 
                 elem_grid[1,1], elem_grid[1,0]]
        mod.add_area_element('DKQ', concrete, sect1, conec)



# si se restringe el desplazamiento de un borde Y 
# se debe restringir el giro sobre el eje X y visceversa
inf_boundary = [1,0,0]
sup_boundary = [1,0,0]
izq_boundary = [1,0,0]
der_boundary = [1,0,0]

# Condiciones de borde inferior (y = -ylim)
for node_tag in idx_nodes[0,:]:
    mod.add_node_restraint(node_tag, inf_boundary)

# Condiciones de borde superior (y = ylim)
for node_tag in idx_nodes[-1,:]:
    mod.add_node_restraint(node_tag, sup_boundary)

# Condiciones de borde izquierdo
for node_tag in idx_nodes[1:-2, 0]:
    mod.add_node_restraint(node_tag, izq_boundary)

# Condiciones de borde derecho
for node_tag in idx_nodes[1:-2, -1]:
    mod.add_node_restraint(node_tag, der_boundary)


# Cargas distribuidas
nnelem = len(mod.elems)
for elem_tag in range(nnelem):
    mod.add_elem_load(elem_tag, [-7000]) #[N/m2]


#print(np.sum(mod.elems[0].loads.reshape((8,3)), axis=0))
#print(mod.assemb_global_stiff().shape)
#print(mod.assemb_global_stiff())


#print(mod.elems[0].area)
#print(mod.elems[0].coord)
#print(mod.elems[0].loads)
#print(mod.elems[0].loads.reshape(8,3))
#print(mod.assemb_global_loads().reshape((mod.nnods, mod.ndofn)))

'''
print(mod.elems[0].coord)
print('D matrices')
print(mod.elems[0].dfmat[0], '\n')
print(mod.elems[0].dsmat[0], '\n')
print('\n')
print('B matrices')
print(mod.elems[0].bfmat[0], '\n')
print(mod.elems[0].bsmat[0], '\n')
print('\n')
print('K matrix')
print(mod.elems[0].stiff.shape, '\n')


#'''
# Solucion
glob_disps, reactions = solve_linear_static(mod)
print('Desplazamientos')
print(glob_disps.reshape((mod.nnods, mod.ndofn)), '\n')
#print('Reacciones')
#print(reactions, '\n')

dz = glob_disps.reshape((mod.nnods, mod.ndofn))[:,0] #desp en z
abs_dz = np.abs(dz)
print('Max. desplazamiento: {:5f} [m]'.format(np.max(abs_dz)))
z0 = np.zeros_like(dz)
z1 = z0 + dz*150



# Esfuerzos
mod.calculate_stresses(glob_disps)
#print(mod.elems[0].gfstress)


coords3D_i = np.hstack([mod.coord, z0[:,None]])
coords3D_f = np.hstack([mod.coord, z1[:,None]])


fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

for elem in mod.elems:
    #ecoord_i = coords3D_i[elem.conec]
    ecoord_f = coords3D_f[elem.conec]
    #eplotcoord_i = np.vstack([ecoord_i, ecoord_i[0]])
    eplotcoord_f = np.vstack([ecoord_f, ecoord_f[0]])
    ex0, ey0, ez1 = eplotcoord_f.T
    #ax.plot(ex0, ey0, np.zeros_like(ex0), color='gray', ls='--', linewidth=1.0, alpha=0.5)
    ax.plot(ex0, ey0, ez1, 'b-', linewidth=1.0, alpha=0.8)

ax.axis('equal')
plt.show()
#'''