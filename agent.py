# agent.py
import numpy as np

class Drone:
    def __init__(self, initial_state: np.ndarray, gravity: float = 9.81):  # [x, y, z, r, v_z, theta]
        self.state   = np.array(initial_state, dtype=float)
        self.gravity = gravity

    def _dynamics(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        x, y, z, r, vz, theta = state
        a_r, a_z, omega       = u

        dx = np.zeros_like(state)
        dx[0] = r * np.cos(theta)         # dot{x}
        dx[1] = r * np.sin(theta)         # dot{y}
        dx[2] = vz                        # dot{z}
        dx[3] = a_r                       # dot{r}
        dx[4] = a_z - self.gravity        # dot{v_z}, gravity pulls down
        dx[5] = omega                     # dot{theta}
        return dx

    def _linearize(self, x: np.ndarray, u: np.ndarray, eps: float = 1e-5):
        # we linearize dynam. as pid is for linear systems but in practice no problem
        n, m = x.size, u.size
        f0 = self._dynamics(x, u)
        A  = np.zeros((n, n))
        B  = np.zeros((n, m))

        #lineartize wrt x
        for i in range(n):
            dx = np.zeros(n); dx[i] = eps
            A[:, i] = (self._dynamics(x + dx, u) - f0) / eps
        #lineartize wrt u
        for j in range(m):
            du = np.zeros(m); du[j] = eps
            B[:, j] = (self._dynamics(x, u + du) - f0) / eps
        
        c = f0 - A @ x - B @ u
        return A, B, c

    def update_state(self, control_input: np.ndarray, dt: float):
        u = np.array(control_input, dtype=float)
        x = self.state.copy()
        A, B, c = self._linearize(x, u)
        x_dot_lin = A @ x + B @ u + c
        self.state = x + x_dot_lin * dt

    def get_state(self) -> np.ndarray:
        return self.state
