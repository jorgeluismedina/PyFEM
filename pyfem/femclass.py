
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy import sparse
from .nodeclass import Node
from .constructors import frame_constructor, area_constructor, solid_constructor


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

        self.fixd_nodes = [] #tags
        self.restraints = [] #values

        self.loaded_nodes = [] #tags
        self.nodal_loads = [] #values

        self.imposed_disp_nodes = [] #tags
        self.imposed_disps = [] #values

        self.loaded_elems = [] #tags

    def add_materials(self, materials): # Son necesarios ahora?
        self.mater = materials

    def add_sections(self, sections): # Son necesarios ahora?
        self.sections = sections

    def assemb_global_vec(self, node_tag, values):
        dof = self.all_nodof[node_tag].flatten()
        vals = np.array(values).flatten()
        return dof, vals

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


    def add_frame_element(self, etype, material, section, conec):
        coord = self.coord[conec]
        dof   = self.all_nodof[conec].ravel()
        self.elems.append(frame_constructor(etype, material, section, coord, conec, dof))

    def add_area_element(self, etype, material, section, conec):
        coord = self.coord[conec]
        dof   = self.all_nodof[conec].ravel()
        self.elems.append(area_constructor(etype, material, section, coord, conec, dof))

    def add_solid_element(self, etype, material, conec):
        coord = self.coord[conec]
        dof   = self.all_nodof[conec].ravel()
        self.elems.append(solid_constructor(etype, material, coord, conec, dof))

        

    def add_elem_load(self, elem_tag, loads):
        self.loaded_elems.append(elem_tag)
        #self.elems[elem_tag].add_loadss(*loads)
        self.elems[elem_tag].add_loads(*loads)

    def clear_elements(self):
        self.elems.clear()
          
    
    def assemb_global_loads(self):
        glob_loads = np.zeros(self.ndofs)

        if self.loaded_nodes:
            dofs, vals = self.assemb_global_vec(self.loaded_nodes, self.nodal_loads)
            glob_loads[dofs] = vals

        if self.loaded_elems:
            for id_elem in self.loaded_elems:
                glob_loads[self.elems[id_elem].dof] += self.elems[id_elem].loads
            
        return glob_loads
    
    
    def add_node_disp(self, tag, imposed_disps): 
        self.imposed_disp_nodes.append(tag)
        self.imposed_disps.append(imposed_disps)

    def assemb_global_disps(self):
        glob_disps = np.zeros(self.ndofs)

        if self.imposed_disp_nodes:
            dofs, vals = self.assemb_global_vec(self.imposed_disp_nodes, self.imposed_disps)
            glob_disps[dofs] = vals

        return glob_disps

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
    
    
    def assemb_global_stiff_sparse(self):
        rows = []
        cols = []
        vals = []

        for elem in self.elems:
            dof = elem.dof
            n = dof.size
            stiff = elem.stiff
            rows.append(np.repeat(dof, n))
            cols.append(np.tile(dof, n))
            vals.append(stiff.ravel())
        
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)

        glob_stiff = coo_matrix((vals, (rows, cols)), shape=(self.ndofs, self.ndofs)).tocsr()
        return glob_stiff


        

    
    
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




        


