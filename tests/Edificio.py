
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
from pyfem.solvers import solve_linear_static, vibration_modes

# Materiales
# Para problemas en dinamica el modulo de elasticidad tiene que estar en [N/m2]
steel = Material(elast=210e9, poiss=0.2, dense=7850) #[KN/m2] [] [Kg/m3]
materials = [steel]

# Secciones
a = 0.2
b = 0.5
sect1 = FrameSection(xarea=0.0062, inrt3=0.00013) #[m2]
sections = [sect1]

mod = Model(ndofn=3)
mod.add_materials(materials)
mod.add_sections(sections)

nodos = []
# LECTURA DE DATOS
with open(r'D:\ProyectosPy\PyFEM\tests\Torre_vib.msh', 'r') as file:
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
        mod.add_element(tag, [ni, nj], sect1, steel, 'Frame22D')
        #elements.append([tag, ni, nj])
        

#mod.add_nodes(np.array(nodos))
#for element in elements:
#    tag, ni, nj = element
#    mod.add_element(tag, [ni, nj], sect1, steel, 'Truss2D')

mod.add_node_restraint(20, [1, 1, 1])
mod.add_node_restraint(21, [1, 1, 1])
#mod.add_node_restraint(109, [1, 1, 1])
#mod.add_node_restraint(111, [1, 1, 1])
#mod.add_node_restraint(113, [1, 1, 1])
#mod.add_node_restraint(115, [1, 1, 1])
#mod.add_node_restraint(116, [1, 1, 1])

mod.add_node_load(14, [-100.0, 0.0, 0.0]) #KN

#print(mod.elems[0].mmatx())
#print(mod.elems[0].kmatx())
print(mod.assemb_global_stiff())
print(mod.assemb_global_mass())
#SOLUCION

fk, Phi = vibration_modes(mod)
print('Frecuencias')
print(fk)

#fig = plot_2dmodel(mod)
#plt.show()