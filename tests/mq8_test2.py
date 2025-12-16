
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
E = 10920 #[N/m2]
nu = 0.3


# Mallado
xdivitions = 8
ydivitions = 8
nnelem = xdivitions * ydivitions

xnodes_l = 2 * xdivitions + 1
ynodes_l = 2 * ydivitions + 1
nnods = xnodes_l * ynodes_l

x = np.linspace(0,B,xnodes_l)
y = np.linspace(0,A,ynodes_l)
idx_nodes = np.arange(0,nnods).reshape((xnodes_l, ynodes_l))
real_nodes = np.ones_like(idx_nodes, dtype=bool)
real_nodes[1::2,1::2] = False
#print(xnodes_l, ynodes_l)

X, Y = np.meshgrid(x,y)
coordinates = np.hstack([X.ravel()[:,None], 
                         Y.ravel()[:,None]])[real_nodes.ravel()]

real_idx_nodes_p = np.arange(0,coordinates.shape[0])
idx_nodes[real_nodes] = real_idx_nodes_p




# Materiales
concrete = Material(elast=E, poiss=nu, dense=0.0,
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
        elem_grid = idx_nodes[2*ex:2*ex+3, 2*ey:2*ey+3]
        conec = [elem_grid[0,0], elem_grid[0,1], elem_grid[0,2], elem_grid[1,2], 
                 elem_grid[2,2], elem_grid[2,1], elem_grid[2,0], elem_grid[1,0]]
        mod.add_area_element('MQ8', concrete, sect1, conec)



# Si se restringe, un borde tangente al x
# se restringe la rotacion en torno al eje y, theta_y
# pero en la formulacion del elemento se debe restringir theta_x
#inf_boundary = [1,1,0] # [1,1,0]
#sup_boundary = [1,1,0] # [1,1,0]
#izq_boundary = [1,0,1] # [1,0,1]
#der_boundary = [1,0,1] # [1,0,1]

# Todo Empotrado
inf_boundary = [1,1,1] # [1,1,0]
sup_boundary = [1,1,1] # [1,1,0]
izq_boundary = [1,1,1] # [1,0,1]
der_boundary = [1,1,1] # [1,0,1]

# Condiciones de borde inferior (y = -ylim)
for node_tag in idx_nodes[0, 1:-1]:
    mod.add_node_restraint(node_tag, inf_boundary)

# Condiciones de borde superior (y = ylim)
for node_tag in idx_nodes[-1, 1:-1]:
    mod.add_node_restraint(node_tag, sup_boundary)

# Condiciones de borde izquierdo
for node_tag in idx_nodes[1:-1, 0]:
    mod.add_node_restraint(node_tag, izq_boundary)

# Condiciones de borde derecho
for node_tag in idx_nodes[1:-1, -1]:
    mod.add_node_restraint(node_tag, der_boundary)

# Condiciones esquinas
#mod.add_node_restraint(idx_nodes[ 0, 0], [1,0,0])
#mod.add_node_restraint(idx_nodes[-1, 0], [1,0,0])
#mod.add_node_restraint(idx_nodes[ 0,-1], [1,0,0])
#mod.add_node_restraint(idx_nodes[-0,-1], [1,0,0])

# Todo empotrado
mod.add_node_restraint(idx_nodes[ 0, 0], [1,1,1])
mod.add_node_restraint(idx_nodes[-1, 0], [1,1,1])
mod.add_node_restraint(idx_nodes[ 0,-1], [1,1,1])
mod.add_node_restraint(idx_nodes[-0,-1], [1,1,1])


# Cargas distribuidas
nnelem = len(mod.elems)
for elem_tag in range(nnelem):
    #mod.add_elem_load(elem_tag, [-7000]) #[N/m2]
    mod.add_elem_load(elem_tag, [-1])


#print(np.sum(mod.elems[0].loads.reshape((8,3)), axis=0))
#print(mod.assemb_global_stiff().shape)
#print(mod.assemb_global_stiff())


#print(mod.elems[0].area)
#print(mod.elems[0].coord)
#print(mod.elems[0].loads)
#print(mod.elems[0].loads.reshape(8,3))
#print(mod.elems[0].conec)
#print(mod.assemb_global_loads().reshape((mod.nnods, mod.ndofn)))



#'''
# Solucion
glob_disps, reactions = solve_linear_static(mod)

dz = glob_disps.reshape((mod.nnods, mod.ndofn))[:,0] #desp en z
abs_dz = np.abs(dz)
print('Max. displacement: {:5f} [m]'.format(np.max(abs_dz)))
z0 = np.zeros_like(dz)
z1 = z0 + dz*2



# Esfuerzos
mod.calculate_stresses(glob_disps)
#print(mod.elems[0].gfstress)


coords3D_i = np.hstack([mod.coord, z0[:,None]])
coords3D_f = np.hstack([mod.coord, z1[:,None]])

inf_nodes = coords3D_i[idx_nodes[0, 1:-1]]
sup_nodes = coords3D_i[idx_nodes[-1, 1:-1]]
izq_nodes = coords3D_i[idx_nodes[1:-1, 0]]
der_nodes = coords3D_i[idx_nodes[1:-1, -1]]


fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

for elem in mod.elems:
    #ecoord_i = coords3D_i[elem.conec]
    ecoord_f = coords3D_f[elem.conec]
    #eplotcoord_i = np.vstack([ecoord_i, ecoord_i[0]])
    eplotcoord_f = np.vstack([ecoord_f, ecoord_f[0]])
    ex0, ey0, ez1 = eplotcoord_f.T
    #ax.plot(ex0, ey0, np.zeros_like(ex0), color='gray', ls='--', linewidth=1.0, alpha=0.5)
    ax.plot(ex0, ey0, ez1, 'b-', linewidth=1.0, alpha=0.8)

#for n, coor in enumerate(coords3D_f):
#    ax.text(coor[0]+0.05, coor[1]+0.05, coor[2]+0.05, str(n), 
#            fontsize=7, ha='left', va='bottom')

#ax.scatter(*coordinates.T, c='b', s=1.0)
ax.scatter(*inf_nodes.T, c='r', s=6.0)
ax.scatter(*sup_nodes.T, c='lime', s=6.0)
ax.scatter(*izq_nodes.T, c='cyan', s=6.0)
ax.scatter(*der_nodes.T, c='k', s=6.0)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.axis('equal')
plt.show()
#'''
