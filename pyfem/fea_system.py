
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
from .elem_constructors import get_elem_constructor


'''
GLOSARIO DE VARIABLES
ndofn = Numero de grados de libertad por nodo
nnods = Numero de nodos en la estructura
ndofs = Numero de grados de libertad de la estructura
elems = Lista de todos los elementos finitos
al_dof = Indice de grados de libertad de la estructura
fr_dof = Indice de grados de libertad libres de la estructura
fx_dof = Indice de grados de libertad fijados de la estructura
gl_stiff = Matriz de rigidez global
re_stiff = Matriz de rigidez reducida
gl_tload = Vector global de cargas aplicadas totales
re_tload = Vector reducido de cargas aplicadas totales
gl_disps = Vector de desplazamientos global
'''


class Structure():
    def __init__(self, ndofn):
        self.ndofn = ndofn 
        self.mater = []
        self.elems = []

    def add_nodes(self, coordinates):
        #Considerar construir un array grande de dofs y que sean dato de entrada para construir un elemento
        self.coord = coordinates
        self.nnods = coordinates.shape[0]
        self.ndofs = self.ndofn * self.nnods
        self.al_dof = np.arange(self.ndofs)

    def add_materials(self, materials):
        self.mater = materials

    def add_elements(self, element_data):
        # data[0] El tipo de elemento está en el primer elemento de data (idea de chatGPT)
        elem_type = element_data[:,-1][0]
        constructor = get_elem_constructor(elem_type)
        self.elems.extend(constructor(element_data, self.coord, self.mater))

    def assemble_vector(self, vec_data, get_dof=False):
        vector = np.zeros(self.ndofs)
        nodes = vec_data[:,0] #lista de nodos
        nodof = vec_data[:,1] #dof de los nodos
        value = vec_data[:,2] #valor asignado a cada dof
        asdof = (self.ndofn * nodes + nodof).astype(int) #lista de dof seleccionados
        vector[asdof] = value
        if get_dof:
            return vector, asdof
        else:
            return vector
        
    def set_restraints(self, restraints):
        self.gl_disps, self.fx_dof = self.assemble_vector(restraints, get_dof=True)
        self.fr_dof = np.setdiff1d(self.al_dof, self.fx_dof)
    
    def set_total_loads(self, tot_loads):
        self.gl_tload = self.assemble_vector(tot_loads)
        self.re_tload = self.gl_tload[self.fr_dof]
    

    #Adaptar para sparse matrix (ensamblar la matriz global a partir de elem.dof_row y elem.dof_col)
    def assemb_global_stiff(self):
        gl_stiff = np.zeros((self.ndofs, self.ndofs))
        for elem in self.elems:
            gl_stiff[np.ix_(elem.dof, elem.dof)] += elem.stiff 

        re_stiff = gl_stiff[np.ix_(self.fr_dof, self.fr_dof)]
        return re_stiff#, gl_stiff
 
    def update_global_stiff(self, re_disps):
        gl_stiff = np.zeros((self.ndofs, self.ndofs))
        gl_disps = np.zeros(self.ndofs)
        gl_disps[self.fr_dof] = re_disps
        for elem in self.elems:
            edisp = gl_disps[elem.dof]
            dstrs = elem.calc_stress(edisp)
            elem.update_elem(dstrs)
            gl_stiff[np.ix_(elem.dof, elem.dof)] += elem.stiff  
        re_stiff = gl_stiff[np.ix_(self.fr_dof, self.fr_dof)]
        return re_stiff




        


