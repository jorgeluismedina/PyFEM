
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

# Materiales
concrete = Material(elast=2.08e10, poiss=0.3, dense=1.0,
                    constitutive_model="plane_stress") #[KN/m2]
materials = [concrete]

# Secciones
sect1 = AreaSection(thick=0.1) #[m2]
sections = [sect1]



# coordenadas
coordinates = np.array([(-1.0,-1.0),
                        (-0.5,-1.0),
                        ( 0.0,-1.0),
                        ( 0.5,-1.0),
                        ( 1.0,-1.0),
                        (-1.0,-0.5),
                        ( 0.0,-0.5),
                        ( 1.0,-0.5),
                        (-1.0, 0.0),
                        (-0.5, 0.0),
                        ( 0.0, 0.0),
                        ( 0.5, 0.0),
                        ( 1.0, 0.0),
                        (-1.0, 0.5),
                        ( 0.0, 0.5),
                        ( 1.0, 0.5),
                        (-1.0, 1.0),
                        (-0.5, 1.0),
                        ( 0.0, 1.0),
                        ( 0.5, 1.0),
                        ( 1.0, 1.0)])

# Creacion de Modelo
mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
mod.add_area_element('MQ8', concrete, sect1, [0, 1, 2, 6, 10, 9, 8, 5])
mod.add_area_element('MQ8', concrete, sect1, [2, 3, 4, 7, 12, 11, 10, 6])
mod.add_area_element('MQ8', concrete, sect1, [10, 11, 12, 15, 20, 19, 18, 14])
mod.add_area_element('MQ8', concrete, sect1, [8, 9, 10, 14, 18, 17, 16, 13])

# Añadir apoyos
mod.add_node_restraint(0, [1, 0, 0])
mod.add_node_restraint(1, [1, 0, 0])
mod.add_node_restraint(2, [1, 0, 0])
mod.add_node_restraint(3, [1, 0, 0])
mod.add_node_restraint(4, [1, 0, 0])
mod.add_node_restraint(5, [1, 0, 0])
mod.add_node_restraint(7, [1, 0, 0])
mod.add_node_restraint(8, [1, 0, 0])
mod.add_node_restraint(12, [1, 0, 0])
mod.add_node_restraint(13, [1, 0, 0])
mod.add_node_restraint(15, [1, 0, 0])
mod.add_node_restraint(16, [1, 0, 0])
mod.add_node_restraint(17, [1, 0, 0])
mod.add_node_restraint(18, [1, 0, 0])
mod.add_node_restraint(19, [1, 0, 0])
mod.add_node_restraint(20, [1, 0, 0])


# Cargas distribuidas
nnelem = len(mod.elems)
for elem_tag in range(nnelem):
    mod.add_elem_load(elem_tag, [-7000])
    print(mod.elems[elem_tag].conec)


print(mod.elems[0].area)
print(mod.elems[0].coord,'\n')

print(gauss_nd(2,2), '\n')
print(mod.elems[0].bfmat,'\n')
print(mod.elems[0].bsmat)
#print(mod.elems[0].loads)
#print(mod.elems[0].loads.reshape(8,3))
#print(mod.assemb_global_loads().reshape((mod.nnods, mod.ndofn)))

'''
print(mod.elems[0].area)
print('D matrices')
print(mod.elems[0].dfmat[0], '\n')
print(mod.elems[0].dsmat[0], '\n')
print('\n')
print('B matrices')
print(mod.elems[0].bfmat[0], '\n')
print(mod.elems[0].bsmat[0], '\n')
print('\n')
print('K matrix')
print(mod.elems[0].stiff.shape, '\n')

'''
#'''
# Solucion
glob_disps, reactions = solve_linear_static(mod)
#print('Desplazamientos')
#print(glob_disps.reshape((mod.nnods, mod.ndofn)))


#'''
dz = glob_disps.reshape((mod.nnods, mod.ndofn))[:,0] #desp en z
abs_dz = np.abs(dz)
print('Max. displacement: {:5f} [m]'.format(np.max(abs_dz)))
z0 = np.zeros_like(dz)
z1 = z0 + dz*300

x0, y0 = mod.coord.T

#plt.figure(1)
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.scatter(x0, y0, z0, color='blue', label='undeformed')
ax.scatter(x0, y0, z1, color='red', label='deformed')
ax.axis('equal')
ax.legend()
#plt.show()
#'''

