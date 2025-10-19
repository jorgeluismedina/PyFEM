
import numpy as np
from .elements.bar1d import Bar1D
from .elements.truss2d import Truss2D
from .elements.truss3d import Truss3D
from .elements.frame2d import Frame2D
from .elements.frame22d import Frame22D
from .elements.tri3 import Tri3
from .elements.quad4 import Quad4
from .elements.quad8 import Quad8
from .elements.hex8 import Hex8



def frame_constructor(etype, mater, section, coord, conec, dof):
    if etype == 'Bar1D':
        return Bar1D(mater, section, coord, conec, dof)
    
    elif etype == 'Truss2D':
        return Truss2D(mater, section, coord, conec, dof)
    
    elif etype == 'Truss3D':
        return Truss3D(mater, section, coord, conec, dof)
    
    elif etype == 'Frame2D':
        return Frame2D(mater, section, coord, conec, dof)
    
    elif etype == 'Frame22D':
        return Frame22D(mater, section, coord, conec, dof)
    
    else:
        raise ValueError(f"Not supported element type: {etype}")


    
def area_constructor(etype, mater, section, coord, conec, dof):
    if etype == 'Tri3':
        return Tri3(mater, section, coord, conec, dof)
    
    elif etype == 'Quad4':
        return Quad4(mater, section, coord, conec, dof)
    
    elif etype == 'Quad8':
        return Quad8(mater, section, coord, conec, dof)
    
    else:
        raise ValueError(f"Not supported element type: {etype}")




def solid_constructor(etype, mater, coord, conec, dof):
    if etype == 'Hex8':
        return Hex8(mater, coord, conec, dof)
    
    else:
        raise ValueError(f"Not supported element type: {etype}")


