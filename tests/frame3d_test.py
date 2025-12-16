
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import FrameSection3D
from pyfem.solvers import solve_linear_static


# Materiales
steel = Material(elast=2.1e10, poiss=0.25, dense=1.0,
                 constitutive_model="elastic3D") #[N/m2]
materials = [steel]

# Secciones
b = 0.2 # [m]
h = 0.3 # [m]
A = b*h
Iz = b*h**3/12
Iy = h*b**3/12
beta = 1/3 - 0.21 * b/h * (1 - 1/12 * (b/h)**4)
J = beta * b**3 * h
print(A)
print(Iz)
print(Iy)
print(beta)
print(J)

sect1 = FrameSection3D(xarea=A, inrt3=Iz, inrt2=Iy, torsion_const=J, section_direc=90) #[m2]
sections = [sect1]

coordinates = np.array([[0.0, 0.0, 0.0],
                        [10.0, 5.0, 3.0]])

# Creacion de Modelo
mod = Model(ndofn=6)
mod.add_nodes(coordinates)
mod.add_materials(materials)

# Añadir elementos
mod.add_frame_element('Frame3D', steel, sect1, [0, 1])

# Añadir apoyos
mod.add_node_restraint(0, [1, 1, 1, 1, 1, 1])

# Añadir cargas nodales
mod.add_node_load(1, [0, 0, -10000, 0, 0, 0]) #[N]


glob_disps, reactions = solve_linear_static(mod)
print(glob_disps.reshape((mod.nnods, mod.ndofn)), '\n')


fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
elem = mod.elems[0]
ax.plot(*elem.coord.T, color='b')
ax.plot(*np.vstack([elem.mid_point, elem.mid_point+elem.local_axes[0]]).T, color='red')
ax.plot(*np.vstack([elem.mid_point, elem.mid_point+elem.local_axes[1]]).T, color='lime')
ax.plot(*np.vstack([elem.mid_point, elem.mid_point+elem.local_axes[2]]).T, color='cyan')
ax.xaxis.set_pane_color('white')  # Transparent x-pane
ax.axis('equal')
plt.show()