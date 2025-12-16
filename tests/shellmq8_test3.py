
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import meshio
import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import AreaSection
from pyfem.solvers import solve_linear_static

# ---------- CONFIGURACIONES -----------
MSH_FILE = "shell3.msh"


# ---------- LECTURA DE MALLA -----------
mesh = meshio.read(MSH_FILE)
coordinates = mesh.points.copy()

# ---------- DEFINICION DE MATERIALES -----------
E = 10.6e6 #[psi]
nu = 0.333333 
material = Material(elast=E, poiss=nu, dense=0.0, constitutive_model="plane_stress")
materials = [material]


# ---------- DEFINICION DE SECCIONES -----------
t = 0.13 #0.13[pul]
section = AreaSection(thick=t) #[m2]
sections = [section]



# ---------- CREACION DEL MODELO -----------
mod = Model(ndofn=6)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

#nelems_per_group = []
#shell_conectivities = []

# ---------- ANADIR ELEMENTOS -----------
for cell_block in mesh.cells:
    # Filtrar solo cuadriláteros (elementos shell)
    if "quad8" in cell_block.type:
        elem_type = cell_block.type
        #nelems = len(cell_block.data)
        for e_conectivity in cell_block.data:
            cor = e_conectivity[:4]
            sid = e_conectivity[4:]
            #ne_conectivity = [cor[3], sid[3], cor[0], sid[0],
            #                  cor[1], sid[1], cor[2], sid[2]]
            #ne_conectivity = [cor[2], sid[2], cor[1], sid[1],
            #                  cor[0], sid[0], cor[3], sid[3]]
            ne_conectivity = [cor[0], sid[0], cor[1], sid[1],
                              cor[2], sid[2], cor[3], sid[3]]
            mod.add_area_element('ShellMQ8', material, section, ne_conectivity)
        
print(mod.elems[0].area)
#print(mod.elems[0].area)

# ---------- ANADIR RESTRICCIONES DE BORDE -----------

# Base
tot_fixed = [1, 1, 1, 1, 1, 0]
suport1 = [1, 0, 0, 1, 0, 0]
suport2 = [0, 1, 0, 0, 1, 0]


tot_fixed_nodes = [4,114,113,115,5,119,116,120,117,121,118,122,6,124,123,125,7]
for tag in tot_fixed_nodes:
    mod.add_node_restraint(tag, tot_fixed)

# Simetria respecto al eje x
xsymm_nodes = [20,18,19,2,17,13,16,12,15,11,14,1,10,8,9,0]
for tag in xsymm_nodes:
    mod.add_node_restraint(tag, suport2)

# Simetria respecto al eje y
ysymm_nodes = [7,112,100,111,99,110,98,109,97,108,96,107,
               95,106,94,105,93,104,92,103,91,102,90,101]
for tag in ysymm_nodes:
    mod.add_node_restraint(tag, suport1)

# Esquina de simetria
mod.add_node_restraint(3, [1,1,0,1,1,0])
#print(mod.fixd_nodes)


# ---------- ANADIR CARGAS NODALES -----------
mod.add_node_load(70, [0,0,-58.35,0,0,0])



# ---------- SOLUCION SISTEMA -----------
glob_disps, reactions = solve_linear_static(mod)
node_disps = glob_disps.reshape((mod.nnods, mod.ndofn))[:,:3]
#print('Max. displacement: {:5f} [cm]'.format(np.min(node_disps[:,-1])))

esc = 500
def_coords = coordinates + node_disps * esc


#'''
fig3, ax3 = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

for elem in mod.elems:
    ecoord = coordinates[elem.conec]

    eplotcoord = np.vstack([ecoord, ecoord[0]])
    ax3.plot(*eplotcoord.T, 'b-', linewidth=1.0, alpha=0.8)
    #ax3.quiver(*elem.mid_point, *elem.local_axes[0], length=0.2, arrow_length_ratio=0.2, color='red')#colors=['red', 'lime', 'cyan']
    #ax3.quiver(*elem.mid_point, *elem.local_axes[1], length=0.2, arrow_length_ratio=0.2, color='lime')
    #ax3.quiver(*elem.mid_point, *elem.local_axes[2], length=0.2, arrow_length_ratio=0.2, color='cyan')

#for n, coor in enumerate(coordinates):
#    ax3.text(coor[0], coor[1], coor[2]+0.05, str(n), 
#            fontsize=7, ha='left', va='bottom')

#mod.nodal_loads
#ax3.quiver(*coordinates[70].T, *elem.local_axes[0], length=0.2, arrow_length_ratio=0.2, color='red')
ax3.scatter(*coordinates[tot_fixed_nodes].T, c='lime', s=6.0)
ax3.scatter(*coordinates[xsymm_nodes].T, c='r', s=6.0)
ax3.scatter(*coordinates[ysymm_nodes].T, c='cyan', s=6.0)
ax3.scatter(*coordinates[3].T, c='k', s=6.0)
ax3.scatter(*coordinates[70].T, c='k', s=10.0)
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")
ax3.axis('equal')
#'''


#'''
fig4, ax4 = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

for elem in mod.elems:
    ecoord = def_coords[elem.conec]

    eplotcoord = np.vstack([ecoord, ecoord[0]])
    ax4.plot(*eplotcoord.T, 'b-', linewidth=1.0, alpha=0.8)

ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.set_zlabel("Z")
ax4.axis('equal')




sectionAB = [3,20,18,19,2,17,13,16,12,15,11,14,1,10,8,9,0]
sectionCD = [1,55,44,56,45,57,46,58,47,59,48,60,49,61,50,62,51,63,52,64,53,65,54,66,5]
sectionEF = [0,32,21,33,22,34,23,35,24,36,25,37,26,38,27,39,28,40,29,41,30,42,31,43,4]

fig5, ax5 = plt.subplots(1, 1, figsize=(10,5))
nnodAB = np.arange(len(sectionAB))
ax5.plot(nnodAB, node_disps[:,2][sectionAB]*1000, 'bo-', ms=3.0)
ax5.set_xlabel("Section AB nodes", fontsize=12)
ax5.set_ylabel("Z displacement [pul*0.001]", fontsize=12)
ax5.set_xticks(nnodAB)
ax5.set_xticklabels(sectionAB)
ax5.grid(True)
#'''

fig51, ax51 = plt.subplots(1, 1, figsize=(10,5))
ax51.plot(nnodAB, node_disps[:,0][sectionAB]*1000, 'bo-', ms=3.0)
ax51.set_xlabel("Section AB nodes", fontsize=12)
ax51.set_ylabel("X displacement [pul * 0.001]", fontsize=12)
ax51.set_xticks(nnodAB)
ax51.set_xticklabels(sectionAB)
ax51.grid(True)



'''
fig6, ax6 = plt.subplots(1, 1, figsize=(10,5))
nnodCD = np.arange(len(sectionCD))
ax6.plot(nnodCD, node_disps[:,0][sectionCD]*1000, 'bo-', ms=3.0)
ax6.set_xlabel("Edge CD nodes", fontsize=12)
ax6.set_ylabel("X displacement [pul * 0.001]", fontsize=12)
ax6.set_xticks(nnodCD)
ax6.set_xticklabels(sectionCD)
ax6.grid(True)


fig7, ax7 = plt.subplots(1, 1, figsize=(10,5))
nnodEF = np.arange(len(sectionEF))
ax7.plot(nnodEF, node_disps[:,0][sectionEF]*1000, 'bo-', ms=3.0)
ax7.set_xlabel("Edge EF nodes", fontsize=12)
ax7.set_ylabel("X displacement [pul * 0.001]", fontsize=12)
ax7.set_xticks(nnodEF)
ax7.set_xticklabels(sectionEF)
ax7.grid(True)

'''



plt.show()