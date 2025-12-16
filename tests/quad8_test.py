
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
from pyfem.gauss_quad import gauss_nd

#print(gauss_nd(2,2))

# Materiales
concrete = Material(elast=2.57e7, poiss=0.2, dense=0.0,
                    constitutive_model="plane_stress") #[KN/m2]
materials = [concrete]

# Secciones
sect1 = AreaSection(thick=0.2) #[m2]
sections = [sect1]

coordinates = np.array([(-1.0,-1.0),
                        ( 0.0,-1.0),
                        ( 1.0,-1.0),
                        ( 1.0, 0.0),
                        ( 1.0, 1.0),
                        ( 0.0, 1.0),
                        (-1.0, 1.0),
                        (-1.0, 0.0)])


# Creacion de Modelo
mod = Model(ndofn=2)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
mod.add_area_element('Quad8', concrete, sect1, [0, 1, 2, 3, 4, 5, 6, 7])

# Añadir apoyos
mod.add_node_restraint(0, [1, 1])
mod.add_node_restraint(1, [1, 1])
mod.add_node_restraint(2, [1, 1])

# Cargas puntuales
mod.add_node_load(4, [0, 250])
mod.add_node_load(5, [0, 500])
mod.add_node_load(6, [0, 250])

# Cargas distribuidas
#mod.add_elem_load(2, [[0,0,500,0], [0,0,0,0]]) #carga en el tercer lado
#mod.add_elem_load(3, [[0,0,500,0], [0,0,0,0]])

#print(mod.elems[0].loads)
print(mod.assemb_global_loads())

print(mod.elems[0].area)
#print(mod.elems[0].bmatx)
#print(mod.elems[0].mater.dmatx)
#mod.set_restraints()
#print(mod.assemb_global_stiff()[np.ix_(mod.free_dof, mod.free_dof)])



#'''

# Solucion
glob_disps, reactions = solve_linear_static(mod)
node_disps = glob_disps.reshape((mod.nnods, mod.ndofn))

print('Desplazamientos')
print(node_disps)

mod.calculate_stresses(glob_disps)
print('Esfuerzos Elementos')
for elem in mod.elems:
    #print(elem.area)
    print(elem.stress.get_cartesians())
    #print(elem.stress.get_principals())
    #print(elem.stress.get_von_mises())
#'''

esc = 500
def_coords = coordinates + node_disps * esc

fig1 = plt.figure(1)

for elem in mod.elems:
    ecoord = def_coords[elem.conec]
    eplotcoord = np.vstack([ecoord, ecoord[0]])
    plt.plot(*eplotcoord.T, 'bo-', ms=3.0)

plt.axis('equal')
plt.grid(True)

plt.show()