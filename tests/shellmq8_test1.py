
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import AreaSection
from pyfem.solvers import solve_linear_static, check_symmetric, detailed_check_SPD

# Materiales
concrete = Material(elast=1, poiss=0.0, dense=0.0,
                    constitutive_model="plane_stress") #[KN/m2]
materials = [concrete]

# Secciones
sect1 = AreaSection(thick=1) #[m2]
sections = [sect1]


l = 9
h = 3
l_divitions = 6
h_divitions = 1
l_nnods = 2 * l_divitions + 1
h_nnods = 2 * h_divitions + 1
nnods0 = l_nnods * h_nnods

idx_nodes = np.arange(0,nnods0).reshape((h_nnods, l_nnods))
real_nodes = np.ones_like(idx_nodes, dtype=bool)
real_nodes[1::2,1::2] = False


x = np.linspace(0, l, l_nnods)
z = np.linspace(0, h, h_nnods)
X, Z = np.meshgrid(x,z)
Y = np.zeros_like(X)
coordinates = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[real_nodes.ravel()]
#coordinates = np.hstack([X.ravel()[:,None], 
#                         Z.ravel()[:,None]])[real_nodes.ravel()]

real_idx_nodes_p = np.arange(0,coordinates.shape[0])
idx_nodes[real_nodes] = real_idx_nodes_p

print(idx_nodes)


# Materiales
material = Material(elast=1, poiss=0.0, dense=0.0,
                    constitutive_model="plane_stress") 
materials = [material]

# Secciones
sect1 = AreaSection(thick=1) 
sections = [sect1]





# Creacion de Modelo
mod = Model(ndofn=6)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
for i in range(h_divitions):
    for j in range(l_divitions):
        elem_grid = idx_nodes[2*i:2*i+3, 2*j:2*j+3]
        conec = [elem_grid[0,0], elem_grid[0,1], elem_grid[0,2], elem_grid[1,2], 
                 elem_grid[2,2], elem_grid[2,1], elem_grid[2,0], elem_grid[1,0]]
        
        mod.add_area_element('ShellMQ8', material, sect1, conec)

# Añadir restricciones
fixed_nodes = idx_nodes[:,0]
print(fixed_nodes)
#---------------------------------------u  v  w  rx ry rz
mod.add_node_restraint(fixed_nodes[0], [1, 1, 0, 1, 0, 1])
mod.add_node_restraint(fixed_nodes[1], [1, 1, 1, 1, 0, 1])
mod.add_node_restraint(fixed_nodes[2], [1, 1, 0, 1, 0, 1])


# Añadir fuerzas nodales
mod.add_node_load(idx_nodes[:,-1][1], [0, 0, -1, 0, 0, 0])


#print(mod.elems[0].stiff)
#for elem in mod.elems:
    #print(elem.damat)
    #print(elem.conec)
    #print(check_symmetric(mod.elems[0].stiff))
    #print(mod.elems[0].stiff)
    #print(elem.local_axes)
    #print(check_symmetric(elem.stiff))
    #detailed_check_SPD(elem.stiff)

#mod.set_restraints()
#glob_stiff = mod.assemb_global_stiff()
#glob_loads = mod.assemb_global_loads()


# Reduccion del sistema
#stiff_ff = glob_stiff[np.ix_(mod.free_dof, mod.free_dof)]
#loads_f = glob_loads[mod.free_dof]


#plt.imshow(stiff_ff)
#plt.spy(-stiff_ff, precision=1e-5)
#plt.show()

# Resolucion del sistema
# Resolucion del sistema 
glob_disps, reactions = solve_linear_static(mod)
node_disps = glob_disps.reshape((mod.nnods, mod.ndofn))

esc = 0.008
def_coords = coordinates + node_disps[:,:3] * esc



print('Desplazamiento en z max:')
print(np.min(node_disps[:,2]))
print(node_disps[:,2][19])

fig, ax1 = plt.subplots(1,1, subplot_kw={'projection': '3d'})

#for i, coord in enumerate(def_coords):
#    ax1.text(coord[0]+0.05, coord[1]+0.05, coord[2]+0.05, str(i), fontsize=7, ha='left', va='bottom')

ax1.scatter(*def_coords.T, c='b', s=3.0)
for elem in mod.elems:
    def_ecoord = def_coords[elem.conec]
    plot_def_ecoord = np.vstack([def_ecoord, def_ecoord[0]])
    ax1.plot(*plot_def_ecoord.T, 'b-', linewidth=1.0, alpha=0.8)

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.axis("equal")


alter_idx = []
for col in range(idx_nodes.shape[1]):
    if col%2 == 0:
        alter_idx.extend(idx_nodes[:,col])
    else:
        alter_idx.extend(idx_nodes[:,col][0::2])


fig, ax2 = plt.subplots(1,1, figsize=(15, 6))
ax2.plot(idx_nodes[real_nodes].ravel(), node_disps[:,2][alter_idx], 'bo-', lw=1, ms=5.0)
ax2.set_xlim(0, 33)
ax2.set_ylim(-120, 0)
ax2.set_xlabel("Mesh nodes", fontsize=20)
ax2.set_ylabel("z displacement [cm]", fontsize=20)
ax2.set_xticks(idx_nodes[real_nodes].ravel())
ax2.set_yticks(np.arange(0,-130,-10))
ax2.grid(True)


plt.show()