
import numpy as np

class FrameSection:
    def __init__(self, xarea, inrt3):
        self.xarea = xarea
        self.inrt3 = inrt3

class AreaSection:
    def __init__(self, thick):
        self.xarea = thick