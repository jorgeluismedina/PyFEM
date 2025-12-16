

import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import FrameSection2D
from pyfem.solvers import solve_linear_static


# Materiales
steel = Material(elast=2.1e8, poiss=0.25, dense=1.0,
                 constitutive_model="elastic3D") #[KN/m2]
materials = [steel]

# Secciones
lado = 200 # [mm]
sect1 = FrameSection2D(xarea=lado**2/(1000**2), inrt3=lado**4/12 /(1000**4)) #[m2]
sections = [sect1]

coordinates = np.array([[0.0, 0.0],
                        [0.0, 2.0],
                        [5.0, 3.0],
                        [5.0, 0.0]])

# Creacion de Modelo
mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)

# Añadir elementos
mod.add_frame_element('Frame2D', steel, sect1, [0, 1])
mod.add_frame_element('Frame2D', steel, sect1, [1, 2])
mod.add_frame_element('Frame2D', steel, sect1, [2, 3])

# Añadir apoyos
mod.add_node_restraint(0, [1, 1, 1])
mod.add_node_restraint(3, [1, 1, 1])

# Cargas distribuidas
mod.add_elem_load(1, [1, 4, 4, 1])

#print(mod.elems[1].loads.reshape((2,3)))
# Solucion
glob_disps, reactions = solve_linear_static(mod)
print(glob_disps.reshape((mod.nnods, mod.ndofn)), '\n')
#print(reactions, '\n')

mod.calculate_forces(glob_disps)
print(mod.elems[0].forces)
print(mod.elems[1].forces)
print(mod.elems[2].forces)