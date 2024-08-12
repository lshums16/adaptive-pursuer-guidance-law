import numpy as np
from tc_guidance_laws import GuidanceLaw
from filter import Filter

class Derivatives():
    def __init__(self, filter_cmd = False, target_accel_type = "constant", target_accel = -25., target_accel_amp = -10., accel_sat = 3.):
        gravity = 9.81
        self.accel_sat = accel_sat*gravity
        
        self.guidance = GuidanceLaw()
        if filter_cmd:
            self.filter_cmd = True
            self.filter = Filter()
        else:
            self.filter_cmd = False
        if target_accel_type != "constant" and target_accel_type != "weave":
            raise ValueError("target_accel_type must be \"constant\" or \"weave\"")    
            
        self.target_accel_type = target_accel_type
        self.target_accel = target_accel
        if target_accel_type == "weave":
            self.target_accel_amp = target_accel_amp
        
    def pronav_derivatives(self, t, x):
        seeker = x[:4]
        target = x[4:8]
        
        accel_cmd = self.guidance.ProNav(seeker, target)
        
        if self.filter_cmd:
            filter = x[8:]
            accel_cmd, filter_dot = self.filter_accel_cmd(t, filter, accel_cmd)
        
        sat_accel_cmd = self.saturate(accel_cmd)
        
        seeker_dot = self.get_deriv(seeker, sat_accel_cmd)
        
        target_dot = self.calc_target_derivatives(target, t)
            
        if self.filter_cmd:
            return np.concatenate((seeker_dot, target_dot, filter_dot))
        else:
            return np.concatenate((seeker_dot, target_dot))

    def FTCG_derivatives(self, t, x):
        seeker = x[:4]
        target = x[4:8]
        
        accel_cmd = self.guidance.FTCG(seeker, target)
        
        if self.filter_cmd:
            filter = x[8:]
            accel_cmd, filter_dot = self.filter_accel_cmd(t, filter, accel_cmd)
        
        sat_accel_cmd = self.saturate(accel_cmd)
        
        seeker_dot = self.get_deriv(seeker, sat_accel_cmd)
        
        target_dot = self.calc_target_derivatives(target, t)
            
        if self.filter_cmd:
            return np.concatenate((seeker_dot, target_dot, filter_dot))
        else:
            return np.concatenate((seeker_dot, target_dot))
    
    def AFTNG_derivatives(self, t, x):
        seeker = x[:4]
        chi = x[4:6]
        target = x[6:10]
        
        accel_cmd, chi_dot = self.guidance.AFTNG(seeker, chi, target)
        
        if self.filter_cmd:
            filter = x[10:]
            accel_cmd, filter_dot = self.filter_accel_cmd(t, filter, accel_cmd)
        
        sat_accel_cmd = self.saturate(accel_cmd)
        
        seeker_dot = self.get_deriv(seeker, sat_accel_cmd)
        
        chi_dot = np.zeros(2)
        
        target_dot = self.calc_target_derivatives(target, t)
            
        if self.filter_cmd:
            return np.concatenate((seeker_dot, chi_dot, target_dot, filter_dot))
        else:
            return np.concatenate((seeker_dot, chi_dot, target_dot))

    def S_AFTNG_derivatives(self, t, x):
        seeker = x[:4]
        chi = x[4]
        target = x[5:9]
        
        accel_cmd, chi_dot = self.guidance.S_AFTNG(seeker, chi, target)
        
        if self.filter_cmd:
            filter = x[9:]
            accel_cmd, filter_dot = self.filter_accel_cmd(t, filter, accel_cmd)
        
        sat_accel_cmd = self.saturate(accel_cmd)
        
        seeker_dot = self.get_deriv(seeker, sat_accel_cmd)
        
        seeker_dot = np.concatenate((seeker_dot, np.array([chi_dot])))
        
        target_dot = self.calc_target_derivatives(target, t)
            
        if self.filter_cmd:
            return np.concatenate((seeker_dot, target_dot, filter_dot))
        else:
            return np.concatenate((seeker_dot, target_dot))

    def filter_accel_cmd(self, t, filter_state, accel_cmd):
        filter_dot = self.filter.derivatives(t, filter_state, accel_cmd)
        if abs(filter_state[1]) > self.accel_sat:
            filtered_cmd = np.sign(filter_state[1])*self.accel_sat
        else:
            filtered_cmd = filter_state[1]
        return filtered_cmd, filter_dot
    
    def saturate(self, accel_cmd):
        if np.isscalar(accel_cmd):
            if abs(accel_cmd) > self.accel_sat:
                return np.sign(accel_cmd)*self.accel_sat
            else:
                return accel_cmd
        else:
            new_commands = np.zeros(len(accel_cmd))
            for i in range(len(accel_cmd)):
                new_commands[i] = self.saturate(accel_cmd[i])
            return new_commands
        
    def get_deriv(self, state, accel_cmd):
        vel = state[2]
        crs_angle = state[3]
        
        x_dot = vel*np.cos(crs_angle)
        y_dot = vel*np.sin(crs_angle)
        V_dot =  0
        crs_angle_dot = accel_cmd/vel
        
        return np.array([x_dot, y_dot, V_dot, crs_angle_dot])
    
    def calc_target_derivatives(self, target, t):
        if self.target_accel_type == "constant":
            target_dot = self.get_deriv(target, self.target_accel)
        elif self.target_accel_type == "weave":
            target_dot = self.get_deriv(target, self.target_accel + self.target_accel_amp*np.sin(t))
            
        return target_dot
