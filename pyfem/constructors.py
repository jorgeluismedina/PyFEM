
import numpy as np
from .elements.bar1d import Bar1D
from .elements.truss2d import Truss2D
from .elements.truss3d import Truss3D
from .elements.frame2d import Frame2D
from .elements.frame22d import Frame22D
from .elements.tri3 import Tri3
from .elements.quad4 import Quad4
from .elements.quad8 import Quad8


def construct_bar1d(nodes, coord, section, material):
    return Bar1D(nodes, coord, section, material)

def construct_truss2d(nodes, coord, section, material):
    return Truss2D(nodes, coord, section, material)

def construct_truss3d(nodes, coord, section, material):
    return Truss3D(nodes, coord, section, material)

def construct_frame2d(nodes, coord, section, material):
    return Frame2D(nodes, coord, section, material)

def construct_frame22d(nodes, coord, section, material):
    return Frame22D(nodes, coord, section, material)

def construct_tri3(nodes, coord, section, material):
    return Tri3(nodes, coord, section, material)

def construct_quad4(nodes, coord, section, material):
    return Quad4(nodes, coord, section, material)

def construct_quad8(nodes, coord, section, material):
    return Quad8(nodes, coord, section, material)





def get_constructor(elem_type):
    element_constructors = {'Bar1D': construct_bar1d,
                            'Truss2D': construct_truss2d,
                            'Truss3D': construct_truss3d,
                            'Frame2D': construct_frame2d,
                            'Frame22D': construct_frame22d,
                            'Tri3': construct_tri3,
                            'Quad4': construct_quad4,
                            'Quad8': construct_quad8}
    
    return element_constructors.get(elem_type)
