
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
from .nodeclass import Node
from .constructors import get_constructor


'''
GLOSARIO DE VARIABLES
ndofn = Numero de grados de libertad por nodo
nnods = Numero de nodos en la estructura
ndofs = Numero de grados de libertad de la estructura
elems = Lista de todos los elementos finitos
'''


class Model():
    def __init__(self, ndofn):
        self.ndofn = ndofn 
        #self.nodes = []
        self.elems = []
        #self.nelem = None

        self.fixd_nodes = []
        self.restraints = []
        self.nodal_loads = []
        self.loaded_nodes = []
        self.loaded_elems = []

    def add_materials(self, materials): # Son necesarios ahora?
        self.mater = materials

    def add_sections(self, sections): # Son necesarios ahora?
        self.sections = sections

    # NODOS
    # Añadir todo uno por uno? los nodos creo que no

    #def add_node(self, tag, coord, ndof):
    #    self.nodes.append(Node(tag, coord, ndof))
    #    self.coord.append(coord)

    #def set_nodes(self):
    #    self.all_coords = np.array([node.coord for node in self.nodes])
    #    self.all_dof = np.array([node.dof for node in self.nodes])
    #    self.nnods = self.all_coords.shape[0]
    #    self.ndofs = self.ndofn * self.nnods
    #    # ver si quitarlos y que se obtengan con un return
    #    self.glob_disps = np.zeros(self.ndofs)
    #    self.glob_loads = np.zeros(self.ndofs)

    def add_nodes(self, coordinates):
        self.coord = coordinates
        self.nnods = coordinates.shape[0]
        self.ndofs = self.ndofn * self.nnods
        self.all_dof = np.arange(self.ndofs, dtype=int)
        self.all_nodof = np.reshape(self.all_dof, (self.nnods, self.ndofn))
        # ver si quitarlos y que se obtengan con un return
        self.glob_disps = np.zeros(self.ndofs)
        self.glob_loads = np.zeros(self.ndofs)

    def add_node_restraint(self, tag, restraints):
        self.fixd_nodes.append(tag)
        self.restraints.append(restraints)

    def set_restraints(self):
        dofs, vals = self.assemb_global_vec(self.fixd_nodes, self.restraints)
        self.fixd_dof = list(dofs[vals.astype(bool)])
        self.free_dof = list(np.setdiff1d(self.all_dof, self.fixd_dof))

    def add_node_load(self, tag, loads): 
        self.loaded_nodes.append(tag)
        self.nodal_loads.append(loads)

    # ELEMENTOS
    def add_element(self, tag, conec, section, material, etype):
        constructor = get_constructor(etype)
        if constructor is None:
            raise ValueError(f"Not supported element type: {etype}")
        
        conec = np.array(conec)
        dof   = self.all_nodof[conec].flatten()
        coord = self.coord[conec]
        self.elems.append(constructor(conec, dof, coord, section, material))

    def add_elem_load(self, elem_tag, loads):
        self.loaded_elems.append(elem_tag)
        self.elems[elem_tag].add_loadss(*loads)

    def clear_elements(self):
        self.elems.clear()
    
    # FUNCIONES
    def assemb_global_vec(self, node_tag, values):
        dof = self.all_nodof[node_tag].flatten()
        vals = np.array(values).flatten()
        return dof, vals
          
    
    def assemb_global_loads(self):
        glob_loads = np.zeros(self.ndofs)

        if self.loaded_nodes:
            dofs, vals = self.assemb_global_vec(self.loaded_nodes, self.nodal_loads)
            glob_loads[dofs] = vals

        if self.loaded_elems:
            for id_elem in self.loaded_elems:
                glob_loads[self.elems[id_elem].dof] += self.elems[id_elem].loads
            
        return glob_loads
    
    def impose_displacements(self, node_tag, imp_disps):
        dofs, vals = self.assemb_global_vec(node_tag, imp_disps)
        self.glob_disps[dofs] = vals

    #Adaptar para sparse matrix (ensamblar la matriz global a partir de elem.dof_row y elem.dof_col)
    def assemb_global_stiff(self):
        glob_stiff = np.zeros((self.ndofs, self.ndofs))
        for elem in self.elems:
            glob_stiff[np.ix_(elem.dof, elem.dof)] += elem.stiff 

        return glob_stiff
    
    def assemb_global_mass(self):
        glob_mass = np.zeros((self.ndofs, self.ndofs))
        for elem in self.elems:
            glob_mass[np.ix_(elem.dof, elem.dof)] += elem.mass 

        return glob_mass
    
    def calculate_forces(self, glob_disps):
        for elem in self.elems:
            edisp = glob_disps[elem.dof]
            elem.calculate_forces(edisp)

    def calculate_stresses(self, glob_disps):
        for elem in self.elems:
            edisp = glob_disps[elem.dof]
            elem.calculate_stress(edisp)
    
    
    def calculate_node_stresses(self):
        areas = np.zeros(self.nnods)
        area_cart_stress = np.zeros((self.nnods, 3)) # [A*sx, A*sy, A*txy] de momento solo para plane stress sz=0
        area_prin_stress = np.zeros((self.nnods, 3))
        area_von_mises_stress = np.zeros(self.nnods)
        for elem in self.elems:
            areas[elem.conec] += elem.area
            area_cart_stress[elem.conec] += elem.area * elem.stress.get_cartesians().T
            area_prin_stress[elem.conec] += elem.area * elem.stress.get_principals().T
            area_von_mises_stress[elem.conec] += elem.area * elem.stress.get_von_mises().T

        cart_stress = area_cart_stress / areas[:,None]
        prin_stress = area_prin_stress / areas[:,None]
        von_mises_stress = area_von_mises_stress / areas
        return cart_stress, prin_stress, von_mises_stress
            



    def update_global_stiff(self, re_disps):
        gl_stiff = np.zeros((self.ndofs, self.ndofs))
        gl_disps = np.zeros(self.ndofs)
        gl_disps[self.free_dof] = re_disps
        for elem in self.elems:
            edisp = gl_disps[elem.dof]
            dstrs = elem.calc_stress(edisp)
            elem.update_stiff(dstrs)
            gl_stiff[np.ix_(elem.dof, elem.dof)] += elem.stiff  
        re_stiff = gl_stiff[np.ix_(self.free_dof, self.free_dof)]
        return re_stiff




        


