
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import plotly.io as pio
pio.renderers.default = "chrome"

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import FrameSection
from pyfem.plotting import print_matrix, plot_3dmodel
from pyfem.solvers import solve_linear_static

# Materiales
steel = Material(elast=2.0e8, poiss=0.2, dense=1.0) #[KN/m2]
materials = [steel]

# Secciones
sect1 = FrameSection(xarea=0.00139345, inrt3=8.358e-5) #[m2]
sections = [sect1]

# Coordenadas   
coordinates = np.array([[0.0, 0.0, 0.0], 
                        [0.0, 2.0, 0.0], 
                        [2.5, 0.0, 1.7], 
                        [2.5, 2.0, 1.7],
                        [3.5, 1.0, 0.0],
                        [5.5, 1.0, 1.7]])

# Creacion de Modelo
mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
mod.add_element(0, [0, 1], sect1, steel, 'Truss3D')
mod.add_element(1, [0, 4], sect1, steel, 'Truss3D')
mod.add_element(2, [1, 4], sect1, steel, 'Truss3D')
mod.add_element(3, [0, 2], sect1, steel, 'Truss3D')
mod.add_element(4, [0, 3], sect1, steel, 'Truss3D')
mod.add_element(5, [1, 3], sect1, steel, 'Truss3D')
mod.add_element(6, [2, 3], sect1, steel, 'Truss3D')
mod.add_element(7, [2, 4], sect1, steel, 'Truss3D')
mod.add_element(8, [3, 4], sect1, steel, 'Truss3D')
mod.add_element(9, [2, 5], sect1, steel, 'Truss3D')
mod.add_element(10, [4, 5], sect1, steel, 'Truss3D')
mod.add_element(11, [3, 5], sect1, steel, 'Truss3D')

# Añadir apoyos
mod.add_node_restraint(0, [1, 1, 1])
mod.add_node_restraint(1, [1, 1, 1])
mod.add_node_restraint(4, [0, 0, 1])

# Cargas nodales
mod.add_node_load(5, [0.0, 0.0, -100.0])

# Solucion
glob_disps, reactions = solve_linear_static(mod)

print('Desplazamientos')
print_matrix(glob_disps*1000, 2, floatfmt=".3e")

print('Reacciones')
print_matrix(reactions, 2, floatfmt=".3e")

mod.calculate_forces(glob_disps)
print('Fuerzas internas')
print([elem.force for elem in mod.elems])

fig = plot_3dmodel(mod)
#fig.show()
fig.write_html("tests/3dtruss1.html", include_plotlyjs="cdn", full_html=True)