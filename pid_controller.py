import numpy as np

class PIDController:
    def __init__(self,
                 Kp_r=1.5, Ki_r=0.3, Kd_r=2.5,
                 Kp_z=3.0, Ki_z=0.5, Kd_z=1.0,
                 Kp_th=3.0, Ki_th=0.2, Kd_th=0.5,
                 gravity=9.81,
                 integrator_limit=1.0,
                 a_r_limit=3.0,
                 dt=0.1,
                 der_filter_alpha=0.8):
        self.Kp_r, self.Ki_r, self.Kd_r = Kp_r, Ki_r, Kd_r
        self.Kp_z, self.Ki_z, self.Kd_z = Kp_z, Ki_z, Kd_z
        self.Kp_th, self.Ki_th, self.Kd_th = Kp_th, Ki_th, Kd_th
        self.gravity = gravity
        self.int_lim = integrator_limit
        self.a_r_limit = a_r_limit
        self.dt = dt
        self.alpha_r = der_filter_alpha
        self.de_r_filt = 0.0
        self.I_r = self.I_z = self.I_th = 0.0
        self.e_r_prev = self.e_z_prev = self.e_th_prev = 0.0

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

    def reset(self):
        self.I_r = self.I_z = self.I_th = 0.0
        self.e_r_prev = self.e_z_prev = self.e_th_prev = 0.0
        self.de_r_filt = 0.0

    def compute_control(self, state, p_ref, v_ref, tangent, dref=None):
        _, _, z, r, vz, theta = state

        th_ref = np.arctan2(tangent[1], tangent[0])
        t_xy = np.array([tangent[0], tangent[1], 0.0])
        t_xy /= (np.linalg.norm(t_xy) + 1e-8)
        v_xy = np.array([v_ref[0], v_ref[1], 0.0])
        r_ref = np.dot(v_xy, t_xy)

        dr = dref.get('dr', 0.0) if dref else 0.0
        dvz = dref.get('dvz', 0.0) if dref else 0.0
        dth = dref.get('dtheta', 0.0) if dref else 0.0

        # radial PID error correction
        e_r = r_ref - r
        de_r = (e_r - self.e_r_prev) / self.dt
        self.de_r_filt = self.alpha_r * self.de_r_filt + (1 - self.alpha_r) * de_r
        self.e_r_prev  = e_r
        self.I_r += e_r * self.dt
        self.I_r = np.clip(self.I_r, -self.int_lim, self.int_lim)
        a_r = self.Kp_r*e_r + self.Ki_r*self.I_r + self.Kd_r*self.de_r_filt + dr
        a_r = np.clip(a_r, -self.a_r_limit, self.a_r_limit)

        # altitude PID error correction
        e_z = p_ref[2] - z
        de_z = (e_z - self.e_z_prev) / self.dt
        self.I_z += e_z * self.dt
        self.I_z = np.clip(self.I_z, -self.int_lim, self.int_lim)
        a_z = self.Kp_z*e_z + self.Ki_z*self.I_z + self.Kd_z*de_z + dvz + self.gravity
        self.e_z_prev = e_z

        # heading PID error correction
        e_th = self._wrap_angle(th_ref - theta)
        de_th = (e_th - self.e_th_prev) / self.dt
        self.I_th += e_th * self.dt
        self.I_th = np.clip(self.I_th, -self.int_lim, self.int_lim)
        omega = self.Kp_th*e_th + self.Ki_th*self.I_th + self.Kd_th*de_th + dth
        self.e_th_prev = e_th

        return np.array([a_r, a_z, omega], dtype=float)
