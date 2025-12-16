
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
B = 2 #[cm]
A = 2 #[cm]
t = 0.1 #[cm]
E = 10920 #[kg/cm2]
nu = 0.3


# Mallado
xdivitions = 4
ydivitions = 4
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
Z = np.zeros(nnods)
coordinates = np.column_stack((X.ravel(), Y.ravel(), Z))[real_nodes.ravel()]

real_idx_nodes_p = np.arange(0,coordinates.shape[0])
idx_nodes[real_nodes] = real_idx_nodes_p




# Materiales
material = Material(elast=E, poiss=nu, dense=0.0,
                    constitutive_model="plane_stress") #[N/m2]
materials = [material]

# Secciones
sect1 = AreaSection(thick=t) #[m2]
sections = [sect1]


# Creacion de Modelo
mod = Model(ndofn=6)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
# Elementos y conectividades
for ex in range(xdivitions):
    for ey in range(ydivitions):
        elem_grid = idx_nodes[2*ex:2*ex+3, 2*ey:2*ey+3]
        conec = [elem_grid[0,0], elem_grid[0,1], elem_grid[0,2], elem_grid[1,2], 
                 elem_grid[2,2], elem_grid[2,1], elem_grid[2,0], elem_grid[1,0]]
        mod.add_area_element('ShellMQ8', material, sect1, conec)



# Añadir restricciones
# Si se restringe, un borde tangente al x
# se restringe la rotacion en torno al eje y, theta_y
# pero en la formulacion del elemento se debe restringir theta_x
#---------------u  v  w  rx ry rz
sup_boundary = [1, 1, 1, 1, 1, 0]
inf_boundary = [0, 1, 0, 0, 1, 0]
izq_boundary = [1, 1, 1, 1, 1, 0]
der_boundary = [1, 0, 0, 1, 0, 0]

sup_bound = idx_nodes[ 0, 1:-1]
inf_bound = idx_nodes[-1, 1:-1]
izq_bound = idx_nodes[ 1:-1, 0]
der_bound = idx_nodes[ 1:-1,-1]

print( idx_nodes[-1,-1])
print( idx_nodes[0,-1])

# Condiciones de borde inferior (y = -ylim)
for node_tag in sup_bound:
    mod.add_node_restraint(node_tag, inf_boundary)

# Condiciones de borde superior (y = ylim)
for node_tag in inf_bound:
    mod.add_node_restraint(node_tag, sup_boundary)

# Condiciones de borde izquierdo
for node_tag in izq_bound:
    mod.add_node_restraint(node_tag, izq_boundary)

# Condiciones de borde derecho
for node_tag in der_bound:
    mod.add_node_restraint(node_tag, der_boundary)

# Condiciones esquinas
mod.add_node_restraint(idx_nodes[ 0, 0], [1, 1, 1, 1, 1, 0])
mod.add_node_restraint(idx_nodes[-1, 0], [1, 1, 1, 1, 1, 0])
mod.add_node_restraint(idx_nodes[-1,-1], [1, 1, 1, 1, 1, 0])
mod.add_node_restraint(idx_nodes[ 0,-1], [1, 1, 0, 1, 1, 0])




# Cargas distribuidas
nnelem = len(mod.elems)
for elem_tag in range(nnelem):
    mod.add_elem_load(elem_tag, [-1]) #[N/m2]

#print(mod.elems[0].loads)


# Solucion
glob_disps, reactions = solve_linear_static(mod)
node_disps = glob_disps.reshape((mod.nnods, mod.ndofn))[:,:3]
node_rots = glob_disps.reshape((mod.nnods, mod.ndofn))[:,3:]
print('Max. displacement: {:5f} [cm]'.format(np.min(node_disps[:,-1])))
print(node_disps[:,-1][8])

esc = 2
def_coords = coordinates + node_disps * esc


#'''
fig3, ax3 = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

for elem in mod.elems:
    ecoord = coordinates[elem.conec]

    eplotcoord = np.vstack([ecoord, ecoord[0]])
    ax3.plot(*eplotcoord.T, 'b-', linewidth=1.0, alpha=0.8)
    ax3.quiver(*elem.mid_point, *elem.local_axes[0], length=0.2, arrow_length_ratio=0.2, color='red')#colors=['red', 'lime', 'cyan']
    ax3.quiver(*elem.mid_point, *elem.local_axes[1], length=0.2, arrow_length_ratio=0.2, color='lime')
    ax3.quiver(*elem.mid_point, *elem.local_axes[2], length=0.2, arrow_length_ratio=0.2, color='cyan')

for n, coor in enumerate(coordinates):
    ax3.text(coor[0], coor[1], coor[2]+0.05, str(n), 
            fontsize=7, ha='left', va='bottom')
    
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")
ax3.axis('equal')




fig4, ax4 = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

ax4.scatter(*def_coords.T, c='b', s=3.0)
for elem in mod.elems:
    ecoord = def_coords[elem.conec]

    eplotcoord = np.vstack([ecoord, ecoord[0]])
    ax4.plot(*eplotcoord.T, 'b-', linewidth=1.0, alpha=0.8)

sup_nodes = def_coords[sup_bound]
inf_nodes = def_coords[inf_bound]
izq_nodes = def_coords[izq_bound]
der_nodes = def_coords[der_bound]
#ax4.scatter(*sup_nodes.T, c='lime', s=6.0)
#ax4.scatter(*inf_nodes.T, c='r', s=6.0)
#ax4.scatter(*izq_nodes.T, c='cyan', s=6.0)
#ax4.scatter(*der_nodes.T, c='k', s=6.0)

ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.set_zlabel("Z")
ax4.axis('equal')




CA_bound = idx_nodes[0, :]
BA_bound = idx_nodes[:,-1]

fig5, ax5 = plt.subplots(1, 2, figsize=(12,5))
ax5[0].plot(CA_bound, node_disps[:,2][CA_bound], 'bo-', lw=1, ms=5.0)
ax5[0].set_ylim(-0.35, 0)
ax5[0].set_xlabel("CA nodes", fontsize=18)
ax5[0].set_ylabel("z displacement [cm]", fontsize=18)
ax5[0].set_xticks(CA_bound)
ax5[0].grid(True)

ax5[1].plot(np.arange(9), node_disps[:,2][BA_bound], 'bo-', lw=1, ms=5.0)
ax5[1].set_ylim(-0.35, 0)
ax5[1].set_xlabel("BA nodes", fontsize=18)
#ax5[1].set_ylabel("z displacement [cm]", fontsize=20)
ax5[1].set_xticks(np.arange(9))
ax5[1].set_xticklabels(BA_bound)
ax5[1].grid(True)


fig6, ax6 = plt.subplots(1, 2, figsize=(12,5))
ax6[0].plot(CA_bound, node_rots[:,0][CA_bound], 'ro-', lw=1, ms=5.0)
#ax6[0].set_ylim(-0.3, 0)
ax6[0].set_xlabel("CA nodes", fontsize=18)
ax6[0].set_ylabel("rotation around x [rad]", fontsize=18)
ax6[0].set_xticks(CA_bound)
ax6[0].grid(True)

ax6[1].plot(np.arange(9), node_rots[:,1][BA_bound], 'ro-', lw=1, ms=5.0)
#ax6[1].set_ylim(0, 0.3)
ax6[1].set_xlabel("BA nodes", fontsize=18)
ax6[1].set_ylabel("rotation around y [rad]", fontsize=18)
ax6[1].set_xticks(np.arange(9))
ax6[1].set_xticklabels(BA_bound)
ax6[1].grid(True)

fig6.tight_layout()



plt.show()
#'''