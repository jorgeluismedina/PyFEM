
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
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
        self.elems = []

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
    def add_nodes(self, coordinates):
        #Considerar construir un array grande de dofs y que sean dato de entrada para construir un elemento
        self.coord = coordinates
        self.nnods = coordinates.shape[0]
        self.ndofs = self.ndofn * self.nnods
        self.all_dof = list(np.arange(self.ndofs, dtype=int))
        # ver si quitarlos y que se obtengan con un return
        self.glob_disps = np.zeros(self.ndofs)
        self.glob_loads = np.zeros(self.ndofs)

    def add_node_restraint(self, tag, restraints):
        self.fixd_nodes.append(tag)
        self.restraints.append(restraints)

    def add_node_load(self, tag, loads): 
        self.loaded_nodes.append(tag)
        self.nodal_loads.append(loads)

    # ELEMENTOS
    def add_element(self, tag, nodes, section, material, etype):
        constructor = get_constructor(etype)
        if constructor is None:
            raise ValueError(f"Not supported element type: {etype}")
        
        coord = self.coord[nodes]
        nodes = np.array(nodes)
        self.elems.append(constructor(nodes, coord, section, material))

    def add_elem_load(self, tag, loads):
        self.loaded_elems.append(tag)
        self.elems[tag].add_loadss(*loads)

    def clear_elements(self):
        self.elems.clear()

    
    # FUNCIONES
    def assemb_global_vec(self, nodes, values):
        ndofn = self.ndofn
        nodes = np.array(nodes) # lista de nodos
        vals = np.array(values).flatten() # valores
        dofs = np.tile(nodes[:,None]*ndofn, ndofn) + np.arange(ndofn)
        dofs = dofs.astype(int).flatten() # dofs con asignacion
        return dofs, vals #nodes, dofs, vals
        
    def set_restraints(self):
        dofs, vals = self.assemb_global_vec(self.fixd_nodes, self.restraints)
        self.fixd_dof = list(dofs[vals.astype(bool)])
        self.free_dof = list(np.setdiff1d(self.all_dof, self.fixd_dof))
    
    def assemb_global_loads(self):
        glob_loads = np.zeros(self.ndofs)

        if self.loaded_nodes:
            dofs, vals = self.assemb_global_vec(self.loaded_nodes, self.nodal_loads)
            glob_loads[dofs] = vals

        if self.loaded_elems:
            for id_elem in self.loaded_elems:
                glob_loads[self.elems[id_elem].dof] += self.elems[id_elem].loads
            
        return glob_loads
    
    def impose_displacements(self, tag, imp_disps):
        dofs, vals = self.assemb_global_vec(tag, imp_disps)
        self.glob_disps[dofs] = vals

    #Adaptar para sparse matrix (ensamblar la matriz global a partir de elem.dof_row y elem.dof_col)
    def assemb_global_stiff(self):
        glob_stiff = np.zeros((self.ndofs, self.ndofs))
        for elem in self.elems:
            glob_stiff[np.ix_(elem.dof, elem.dof)] += elem.stiff 

        return glob_stiff
    
    def calculate_forces(self, glob_disps):
        for elem in self.elems:
            edisp = glob_disps[elem.dof]
            elem.calculate_forces(edisp)

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




        


