
import numpy as np
from .finite_elements import Quad4, Bar2D, Bar1D

def construct_bar1d(data, coordinates, materials):
    # ********** data ***********
    # [n1, n2, area, idmat, etype]
    nelem = data.shape[0]
    areas = data[:,-3]
    nodes = data[:,:-3].astype(int)
    idmat = data[:,-2].astype(int)
    new_coords = coordinates[nodes]
    #elem_mater = materials[idmat]
    elements = [Bar1D(nodes[i], new_coords[i], 
                      areas[i], materials[idmat[i]]) for i in range(nelem)]
    return elements


def construct_bar2d(data, coordinates, materials):
    # ********** data ***********
    # [n1, n2, area, idmat, etype]
    nelem = data.shape[0]
    areas = data[:,-3]
    nodes = data[:,:-3].astype(int)
    idmat = data[:,-2].astype(int)
    coord = coordinates[nodes]
    #elem_mater = materials[idmat]
    elements = [Bar2D(nodes[i], coord[i], 
                      areas[i], materials[idmat[i]]) for i in range(nelem)]
    return elements
 
def construct_quad4(data, coordinates, materials):
    # ************* data ****************
    # [n1, n2, n3, n4 thick, idmat, etype]
    nelem = data.shape[0]
    thicks = data[:,-3]
    nodes = data[:,:-3].astype(int)
    idmat = data[:,-2].astype(int)
    new_coords = coordinates[nodes]
    elem_mater = materials[idmat]
    elements = [Quad4(nodes[i], new_coords[i], 
                      thicks[i], elem_mater[i]) for i in range(nelem)]
    return elements 



def get_elem_constructor(elem_type):
    element_constructors = {0: construct_bar1d, 
                            1: construct_bar2d,
                            2: construct_quad4}
    
    return element_constructors.get(elem_type)
