import numpy as np

class Filter():
    def __init__(self, wn = 2, zeta = 0.7):
        self.wn = wn
        self.zeta = zeta
    
    def derivatives(self, t, state, accel_cmd):
        x1, x2 = state
        
        x1dot = -2*self.zeta*self.wn*x1 - self.wn**2*x2 + self.wn**2*accel_cmd
        x2dot = x1
        
        return np.array([x1dot, x2dot])
    
    def update(self, t, state, accel_cmd):
        self.x1, self.x2 = state
        
        return self.x1 