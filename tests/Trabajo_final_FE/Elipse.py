
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import scipy as sp
import pickle
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import AreaSection
from pyfem.plotting import print_matrix, plot_2dmodel, plot_matrix
from pyfem.solvers import solve_linear_static, vibration_modes, finite_differences
from pyfem.togid import to_gid_results

# Materiales
# Para problemas en dinamica el modulo de elasticidad tiene que estar en [N/m2]
steel = Material(elast=1, poiss=0.3, dense=1) #[N/m2] [] [Kg/m3]
materials = [steel]

# Secciones
sect1 = AreaSection(thick=0.01) #[m]
sections = [sect1]

mod = Model(ndofn=2)
mod.add_materials(materials)
mod.add_sections(sections)

nodos = []
# LECTURA DE DATOS
with open(r'D:\Maestria UFRGS\Elementos Finitos\TrabajoFinalEF\ElipseQuad4.gid\ElipseQuad4.dat', 'r') as file:
    lines = file.readlines()

part_coordinates = False
part_elements = False
part_restraints = False
part_nod_loads = False

for line in lines:
    line = line.strip()

    if line.startswith('# Coordinates'):
        part_coordinates = True
        continue
    elif line.startswith('# End Coordinates'):
        part_coordinates = False
        mod.add_nodes(np.array(nodos))
        print('number of nodes: {0}'.format(mod.nnods))
        print('number of degrees of freedom: {0}'.format(mod.ndofs))
        continue
    elif line.startswith('# Elements'):
        part_elements = True
        continue
    elif line.startswith('# End Elements'):
        part_elements = False
        print('number of elements: {0}'.format(len(mod.elems)))
        continue 
    elif line.startswith('# Restraints'):
        part_restraints = True
        continue
    elif line.startswith('# End Restraints'):
        part_restraints = False
        continue
    elif line.startswith('# Nodal Loads'):
        part_nod_loads = True
        continue
    elif line.startswith('# End Nodal Loads'):
        part_nod_loads = False


    if part_coordinates:
        data = line.split()
        x = float(data[1])
        y = float(data[2])
        nodos.append([x,y])

    elif part_elements:
        data = line.split()
        tag = int(data[0])-1
        etype = int(data[1])
        matid = int(data[2])-1
        if etype==3: #Quad4 en GiD
            n1 = int(data[3])-1
            n2 = int(data[4])-1
            n3 = int(data[5])-1
            n4 = int(data[6])-1
            mod.add_element(tag, [n1, n2, n3, n4], 
                            sections[matid], materials[matid], 'Quad4')
        elif etype==2: #Tri3 en GiD
            n1 = int(data[3])-1
            n2 = int(data[4])-1
            n3 = int(data[5])-1
            mod.add_element(tag, [n3, n2, n1], #para que lo lea antihorario
                            sections[matid], materials[matid], 'Tri3')
        
    elif part_restraints:
        data = line.split()
        tag = int(data[0])-1
        boolx = int(data[1])
        booly = int(data[2])
        mod.add_node_restraint(tag, [boolx, booly])

    elif part_nod_loads:
        data = line.split()
        tag = int(data[0])-1
        forcex = float(data[1])
        forcey = float(data[2])
        mod.add_node_load(tag, [forcex, forcey])


glob_disps, reactions = solve_linear_static(mod)
node_disps = glob_disps.reshape((mod.nnods, mod.ndofn))
#node_reacs = reactions.reshape((mod.nnods, mod.ndofn))

mod.calculate_stresses(glob_disps)
node_cart_stresses, node_prin_stresses, node_vs_stresses = mod.calculate_node_stresses()

majoraxq_node1 = 653 -1
majoraxq_node2 = 864 -1
majoraxq_node3 = 1270 -1
majoraxq_node4 = 1867 -1
majoraxq_node5 = 5 -1

majoraxt_node1 = 646 -1
majoraxt_node2 = 869 -1
majoraxt_node3 = 1294 -1
majoraxt_node4 = 1910 -1
majoraxt_node4 = 2818 -1

#print(node_cart_stresses[majoraxt_node1])
#print(node_prin_stresses[majoraxt_node1])
#print(node_vs_stresses[majoraxt_node1])



to_gid_results(r"D:\Maestria UFRGS\Elementos Finitos\TrabajoFinalEF\ElipseQuad4.gid\ElipseQuad4", 
               node_disps, node_cart_stresses, node_prin_stresses, node_vs_stresses)

#'''
root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#ruta_pickle1 = os.path.join(root, "Trabajo_final_FE", "resultadosT3_4.pkl")
ruta_pickle2 = os.path.join(root, "Trabajo_final_FE", "resultadosQ4_4.pkl")

resultados2 = (node_cart_stresses, node_prin_stresses, node_vs_stresses)
with open(ruta_pickle2, "wb") as f:
    pickle.dump(resultados2, f)
#'''