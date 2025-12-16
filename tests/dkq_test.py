
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


# Materiales
concrete = Material(elast=2.08e10, poiss=0.25, dense=1.0,
                    constitutive_model="plane_stress") #[KN/m2]
materials = [concrete]

# Secciones
sect1 = AreaSection(thick=0.1) #[m2]
sections = [sect1]

coordinates = np.array([(-1.0,-1.0),
                        ( 0.0,-1.0),
                        ( 1.0,-1.0),
                        (-1.0, 0.0),
                        ( 0.0, 0.0),
                        ( 1.0, 0.0),
                        (-1.0, 1.0),
                        ( 0.0, 1.0),
                        ( 1.0, 1.0)])


# Creacion de Modelo
mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
mod.add_area_element('DKQ', concrete, sect1, [0, 1, 4, 3])
mod.add_area_element('DKQ', concrete, sect1, [1, 2, 5, 4])
mod.add_area_element('DKQ', concrete, sect1, [4, 5, 8, 7])
mod.add_area_element('DKQ', concrete, sect1, [3, 4, 7, 6])

# Añadir apoyos
mod.add_node_restraint(0, [1, 1, 1])
mod.add_node_restraint(1, [1, 1, 1])
mod.add_node_restraint(2, [1, 1, 1])
mod.add_node_restraint(3, [1, 1, 1])
mod.add_node_restraint(5, [1, 1, 1])
mod.add_node_restraint(6, [1, 1, 1])
mod.add_node_restraint(7, [1, 1, 1])
mod.add_node_restraint(8, [1, 1, 1])

# Cargas distribuidas
mod.add_elem_load(0, [-7000]) #[N/m2]
mod.add_elem_load(1, [-7000]) #[N/m2]
mod.add_elem_load(2, [-7000]) #[N/m2]
mod.add_elem_load(3, [-7000]) #[N/m2]

print(mod.elems[0].dmatx)
print(mod.elems[0].area)
print(mod.elems[0].bmatx)
#print(mod.elems[3].loads)
#print(mod.assemb_global_loads())

#print(mod.elems[0].bmatx)
#print(mod.elems[0].mater.dmatx)
#mod.set_restraints()
#print(mod.assemb_global_stiff()[np.ix_(mod.free_dof, mod.free_dof)])



#'''

# Solucion
glob_disps, reactions = solve_linear_static(mod)

print('Desplazamientos')
print(glob_disps.reshape((mod.nnods, mod.ndofn)))