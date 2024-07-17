
import numpy as np
import scipy as sp
from scipy import linalg
from scipy.sparse import coo_matrix
from itertools import product
from finite_elements import Quad4, Bar2D
from elem_constructors import get_elem_constructor


elem_type = {0: Bar2D,
             1: Quad4}

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
gl_loads = Vector de cargas global
re_stiff = Matriz de rigidez reducida
re_loads = Vector de cargas reducido
gl_disps = Vector de desplazamientos global
'''


class Structure():
    def __init__(self, element_data, coordinates, materials, ndofn):
        self.ndofn = ndofn 
        self.nnods = coordinates.shape[0]
        self.ndofs = self.ndofn * self.nnods
        self.al_dof = np.arange(self.ndofs)
        #Considerar construir un array grande de dofs y que sean dato de entrada para construir un elemento

        self.elems = []
        self.gl_stiff = None
        self.gl_loads = None
        self.gl_disps = None

        self.assemble_elements(element_data, coordinates, materials)
    
    def assemble_elements(self, element_data, coordinates, materials): #element_data es una lista de arrays
        for data in element_data:
            elem_type = data[:,-1][0]  # data[0] El tipo de elemento está en el primer elemento de data (idea de chatGPT)
            constructor = get_elem_constructor(elem_type)
            if constructor:
                self.elems.extend(constructor(data, coordinates, materials))

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
    
    def set_loads(self, fix_loads, inc_loads=None, nincs=None):
        fixloads = self.assemble_vector(fix_loads)
        if inc_loads is not None:
            incloads = self.assemble_vector(inc_loads)/nincs
            incloads = np.tile(incloads,(nincs,1))
            self.gl_loads = np.vstack([fixloads, incloads])
            self.re_loads = self.gl_loads[:,self.fr_dof] 
        else:
            self.gl_loads = fixloads
            self.re_loads = self.gl_loads[self.fr_dof]

    #Adaptar para sparse matrix (ensamblar la matriz global a partir de elem.dof_row y elem.dof_col)
    def assemble_stiff_mat(self):
        self.gl_stiff = np.zeros((self.ndofs, self.ndofs))
        for elem in self.elems:
            ix_edof = np.ix_(elem.dof, elem.dof)
            self.gl_stiff[ix_edof] += elem.stiff

        ix_freedof = np.ix_(self.fr_dof, self.fr_dof)
        self.re_stiff = self.gl_stiff[ix_freedof]    
    
    def solve_system(self):
        self.gl_disps[self.fr_dof] = sp.linalg.solve(self.re_stiff, self.re_loads, assume_a='sym')
        #ne_stiff = self.gl_stiff[np.ix_(self.fx_dof, self.al_dof)]
        #self.reacts[self.fx_dof] = ne_stiff @ self.gl_disps - self.gl_loads[self.fx_dof]

    def get_element_stresses(self, gl_disps):
        for elem in self.elems:
            edisp = gl_disps[elem.dof]
            elem.get_stress(edisp)

    
    #FUNCIONES PARA APLICAR METODO DE LA RIGIDEZ TANGENCIAL
    #Se retorna la matriz de rigidez tangente reducida
    def retan_stiff(self, re_disps):
        gl_stiff = np.zeros((self.ndofs, self.ndofs))
        gl_disps = np.zeros(self.ndofs)
        gl_disps[self.fr_dof] = re_disps
        for elem in self.elems:
            edisp = gl_disps[elem.dof]
            elem.get_stress(edisp)
            gl_stiff[np.ix_(elem.dof, elem.dof)] += elem.stiff  
        re_stiff = gl_stiff[np.ix_(self.fr_dof, self.fr_dof)]
        return re_stiff

    #Devuelve los desplazamientos de los dofs libres
    def direct_solve(self, re_stiff, re_loads):
        gl_disps = np.zeros(self.ndofs)
        gl_disps[self.fr_dof] = sp.linalg.solve(re_stiff, re_loads, assume_a='sym')
        return gl_disps


        


