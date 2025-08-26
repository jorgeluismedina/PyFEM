

import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
#import plotly.io as pio
#pio.renderers.default = "chrome"

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import FrameSection
from pyfem.plotting import print_matrix, plot_2dmodel
from pyfem.solvers import solve_linear_static

# Materiales
steel = Material(elast=2.0e8, poiss=0.2, dense=1.0) #[KN/m2]
materials = [steel]

# Secciones
sect1 = FrameSection(xarea=1, inrt3=8.358e-5) #[m2]
sections = [sect1]

mod = Model(ndofn=2)
mod.add_materials(materials)
mod.add_sections(sections)

nodos = []
elements = []
# LECTURA DE DATOS
with open(r'D:\ProyectosPy\PyFEM\tests\Torre_mesh.msh', 'r') as file:
    lines = file.readlines()

part_coordinates = False
part_elements = False

for line in lines:
    line = line.strip()

    if line.startswith('Coordinates'):
        part_coordinates = True
        continue
    elif line.startswith('End Coordinates'):
        part_coordinates = False
        mod.add_nodes(np.array(nodos))
        continue
    elif line.startswith('Elements'):
        part_elements = True
        continue
    elif line.startswith('End Elements'):
        part_elements = False
        continue 

    if part_coordinates:
        data = line.split()
        x = float(data[1])
        y = float(data[2])
        nodos.append([x,y])

    elif part_elements:
        data = line.split()
        tag = int(data[0])-1
        ni = int(data[1])-1
        nj = int(data[2])-1
        mod.add_element(tag, [ni, nj], sect1, steel, 'Truss2D')
        #elements.append([tag, ni, nj])
        

#mod.add_nodes(np.array(nodos))
#for element in elements:
#    tag, ni, nj = element
#    mod.add_element(tag, [ni, nj], sect1, steel, 'Truss2D')

mod.add_node_restraint(0, [1, 1])
mod.add_node_restraint(5, [1, 1])

mod.add_node_load(91, [-100.0, 0.0]) #KN


#SOLUCION
glob_disps, reactions = solve_linear_static(mod)

print('Desplazamientos')
print_matrix(glob_disps*1000, 2, floatfmt=".3e")

print('Reacciones')
print_matrix(reactions, 2, floatfmt=".3e")

mod.calculate_forces(glob_disps)
#print('Fuerzas internas')
#print([elem.force for elem in mod.elems])
print(len(mod.elems))
print(mod.assemb_global_mass())
plt.spy(mod.assemb_global_mass())

#fig = plot_2dmodel(mod)
plt.show()

