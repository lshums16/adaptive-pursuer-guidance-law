import numpy as np

class GuidanceLaw():
    def __init__(self):
        self.seeker_qdot_PNG = []
        self.seeker_qdot_FTCG = []
        self.seeker_qdot_S_AFTNG = []
    
    def get_coeffs(self, seeker, target):
        seeker_pos = seeker[:2]
        seeker_vel = seeker[2]
        seeker_crs_angle = seeker[3]
        
        target_pos = target[:2]
        target_vel = target[2]
        target_crs_angle = target[3]
        
        rel_vel = target_vel*np.array([np.cos(target_crs_angle), np.sin(target_crs_angle)]) - seeker_vel*np.array([np.cos(seeker_crs_angle), np.sin(seeker_crs_angle)])
        Vr = np.linalg.norm(rel_vel)
        LOS = target_pos - seeker_pos
        R = np.linalg.norm(LOS)
        q = np.arctan2(LOS.item(1), LOS.item(0))

        qdot = np.cross(LOS, rel_vel)/np.inner(LOS, LOS)
        Vq = qdot*R
        
        return q, Vr, Vq, R, seeker_crs_angle 
     
    def ProNav(self, seeker, target):
        N = 4
        
        q, Vr, Vq, R, seeker_crs_angle = self.get_coeffs(seeker, target)
        # Vq = -np.linalg.norm(rel_vel - np.inner(rel_vel, LOS)/np.inner(LOS, LOS)*LOS)
        # qdot = Vq/R
        
        
        A_cos = 1/np.cos(q - seeker_crs_angle)*(N*Vr*Vq/R)
        A = N*Vr*Vq/R
        
        # Experiment
        # Rm = np.array([[np.cos(seeker_crs_angle), -np.sin(seeker_crs_angle), 0], [np.sin(seeker_crs_angle), np.cos(seeker_crs_angle), 0], [0, 0, 1]])
        # LOS = np.concatenate((target[:2] - seeker[:2], np.array([0])))
        # target_crs_angle = target[3]
        
        # Vt_vec = target[2]*np.array([np.cos(target_crs_angle), np.sin(target_crs_angle), 0])
        # Vm_vec = seeker[2]*np.array([np.cos(seeker_crs_angle), np.sin(seeker_crs_angle), 0])
        # Vr_vec = Vt_vec - Vm_vec
        
        # omega = np.cross(LOS, Vr_vec)/np.dot(LOS, LOS)
        
        # A_test =  -N*np.linalg.norm(Vr_vec)*np.cross(Vm_vec/np.linalg.norm(Vm_vec), omega) # TODO: why is this not orthogonal to the missile velocity vector? 
        # A_test = Rm.T@A_test
        # if not np.isclose(A_test.item(1), A):
        #     raise ValueError("A_test and A are not equal")
        
        return A_cos
    
    def S_AFTNG(self, seeker, chi, target):
        q, Vr, Vq, R, seeker_crs_angle = self.get_coeffs(seeker, target)
        
        k1 = 1
        L_AT = 50
        L_AdotT = 25
        eta = 10
        alpha = 5
        qdot = Vq/R
        
        if not np.isinf(np.exp(alpha*qdot)):
            k0 = L_AdotT*(L_AT + abs(chi) + eta)*(np.exp(alpha*qdot) - 1)/(np.exp(alpha*qdot) + 1)/k1
        else:
            k0 = L_AdotT*(L_AT + abs(chi) + eta)/k1
        
        A = 1/np.cos(q - seeker_crs_angle)*(Vq*Vr/R + k0 + chi)
        
        chi_dot = k1*np.sign(Vq)
        
        return A, chi_dot
       
    def FTCG(self, seeker, target):
        q, Vr, Vq, R, seeker_crs_angle = self.get_coeffs(seeker, target)
        
        N = 3
        epsilon = 0.005
        eta = 0.4
        c = 30
        f = 25
        qdot = Vq/R
        
        # Vq = np.linalg.norm(rel_vel - np.inner(rel_vel, LOS)/np.inner(LOS, LOS)*LOS)
        # qdot = Vq/R
        
        A = 1/np.cos(q - seeker_crs_angle)*(N*Vr*Vq/R + f*np.min([1, qdot/epsilon])  + c*abs(qdot)**eta*np.sign(qdot))
        
        return A

    def AFTNG(self, seeker, chi, target):
        q, Vr, Vq, R, seeker_crs_angle = self.get_coeffs(seeker, target)
        
        k1 = 3
        k2 = 2
        k3 = 0.01
        k4 = 0.1
        gamma = 0.5
        
        # Vq = np.linalg.norm(rel_vel - np.inner(rel_vel, LOS)/np.inner(LOS, LOS)*LOS)
        # qdot = Vq/R
        
        A = 1/np.cos(q - seeker_crs_angle)*(-Vr*Vq/R + k1*Vq*abs(Vq)**(-gamma) + k2*chi[0] + k3*chi[1])
        
        chi1_dot = Vq/abs(Vq)
        if chi[1] != 0:
            chi2_dot = k2*k3*abs(chi[1])*Vq/abs(Vq) - k4*chi[1]/abs(chi[1])
        else:
            chi2_dot = -k2*k3*abs(chi[1])*Vq/abs(Vq)
        return A, np.array([chi1_dot, chi2_dot])

