import numpy as np
from core.attitude import quat_kinematics
from models.actuators import BoomArray, Magnetorquer

class WiggleSat:
    def __init__(self, config, init_state):
        self.J_base = config['J']
        self.Booms = BoomArray(config['booms'])
        self.Magnetorquers = Magnetorquer(config['magnetorquers'])

        self.controller = None
        self.init_state = init_state
        self.state = init_state

        self.J = self.J_base.copy()
        self.J_inv = np.linalg.inv(self.J)
        self.J_dot = np.zeros((3, 3))
        self.dJ_dtheta = np.zeros((3, 3, 3))  # dJ_dtheta[i] = dJ/dtheta_i

        self.use_motor = config['motor_dynamics']

    def update_inertia(self, theta, theta_dot=None):
        """
        Updates the inertia matrix based on the current boom angles, in base body frame.
        If theta_dot is given, also updates self.dJ_dtheta (dJ/dtheta_i for each boom) and
        self.J_dot = sum_i theta_dot_i * dJ/dtheta_i (time derivative of J in the body frame).
        """
        B = self.Booms
        J_total = self.J_base.copy()
        dJ_dtheta = np.zeros((3, 3, 3))
        for i in range(3):
            r_com, r_tip, J_rod_body = B.geometry(i, theta[i])
            J_rod_parallel = B.m_rods[i] * (np.dot(r_com, r_com) * np.eye(3) - np.outer(r_com, r_com))
            J_tip_parallel = B.m_tips[i] * (np.dot(r_tip, r_tip) * np.eye(3) - np.outer(r_tip, r_tip))

            J_total += J_rod_body + J_rod_parallel + J_tip_parallel

            if theta_dot is not None:
                # Each boom point mass moves as r = r_hinge + R p, so dr/dtheta = a x (r - r_hinge).
                a = B.axes[i]
                a_hat = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
                dJ_dtheta[i] += a_hat @ J_rod_body - J_rod_body @ a_hat  # rotating rod-about-own-COM term
                for m, r in ((B.m_rods[i], r_com), (B.m_tips[i], r_tip)):
                    v = np.cross(a, r - B.r_hinges[i])
                    dJ_dtheta[i] += m * (2.0 * np.dot(r, v) * np.eye(3) - np.outer(v, r) - np.outer(r, v))

        self.J = J_total
        if theta_dot is not None:
            self.dJ_dtheta = dJ_dtheta
            self.J_dot = np.einsum('i,ijk->jk', np.asarray(theta_dot, float), dJ_dtheta)
        else:
            self.J_dot = np.zeros((3, 3))
        self.J_inv = np.linalg.inv(J_total)
        return J_total

    def angular_momentum(self, state):
        """
        Total angular momentum about the chassis CoM, in the body frame:
            H = J(theta) w + sum_i g_i(theta_i) * theta_dot_i
        With no external torque this is conserved in the inertial frame (so |H| is constant).
        """
        w, theta, theta_dot = state[4:7], state[7:10], state[10:13]
        self.update_inertia(theta)
        H = self.J @ w
        for i in range(3):
            H = H + self.Booms.momentum_vector(i, theta[i]) * theta_dot[i]
        return H

    def kinetic_energy(self, state):
        """
        Rotational kinetic energy about the chassis CoM:
            T = 1/2 w.J(theta).w + sum_i theta_dot_i * g_i.w + 1/2 sum_i J_boom_i * theta_dot_i^2
        The hinge torques are the only thing that does work on it: dT/dt = sum_i tau_i * theta_dot_i.
        """
        w, theta, theta_dot = state[4:7], state[7:10], state[10:13]
        self.update_inertia(theta)
        T = 0.5 * w @ self.J @ w + 0.5 * np.sum(self.Booms.inertias * theta_dot**2)
        for i in range(3):
            T += theta_dot[i] * (self.Booms.momentum_vector(i, theta[i]) @ w)
        return T

    def state_derivative(self, t, state, u_boom, u_mag, B_body=np.array([0.0,0.0,0.0])):
        """
        Computes ODE derivatives.
        state: [q(4), w(3), theta_booms(3), thetadot_booms(3)]
        control inputs:
            u_boom: motor_dynamics=False -> 3 boom hinge accelerations [rad/s^2] (prescribed motion)
                    motor_dynamics=True  -> 3 commanded boom angles [rad] for the stepper motors
            u_mag (3 dipole moments)

        The chassis CoM is treated as the (inertial) reference point, i.e. the shift of the system
        CoM as the booms swing is neglected. Ignoring orbit dynamics for now.
        """
        q = state[:4] / np.linalg.norm(state[:4])
        w = state[4:7]
        theta = state[7:10]
        theta_dot = state[10:13]
        B = self.Booms

        self.update_inertia(theta, theta_dot)
        tau_mag = self.Magnetorquers.compute_torque(u_mag, B_body) # placeholder for B_body for now

        if self.use_motor:
            # Hinge torques are given, so the chassis and boom accelerations are coupled:
            #   [ J    G ] [ w_dot      ]   [ tau_ext - J_dot w - sum_i g_i' theta_dot_i^2 - w x H ]
            #   [ G^T  D ] [ theta_ddot ] = [ tau_motor_i + 1/2 w.(dJ/dtheta_i).w                 ]
            # G = [g_1 g_2 g_3], D = diag(J_boom). The second row is Lagrange's equation for theta_i;
            # theta_ddot = tau/J_boom alone would ignore that the hinge rides on an accelerating chassis.
            tau_motor = B.motor_torques(theta, theta_dot, u_boom)
            G = np.column_stack([B.momentum_vector(i, theta[i]) for i in range(3)])
            H = self.J @ w + G @ theta_dot
            rhs_w = tau_mag - self.J_dot @ w - np.cross(w, H)
            for i in range(3):
                rhs_w = rhs_w - B.momentum_vector_rate(i, theta[i]) * theta_dot[i] ** 2
            rhs_theta = tau_motor + 0.5 * np.einsum('j,ijk,k->i', w, self.dJ_dtheta, w)
            M = np.block([[self.J, G], [G.T, np.diag(B.inertias)]])
            sol = np.linalg.solve(M, np.concatenate([rhs_w, rhs_theta]))
            dw_dt, theta_ddot = sol[:3], sol[3:]
        else:
            theta_ddot, tau_reaction = B.compute_reaction_torque(theta, theta_dot, u_boom, w)
            tau_total = tau_reaction + tau_mag
            # Rotational Dynamics (Euler's multi-body equations)
            dw_dt = self.J_inv @ (tau_total - np.cross(w, self.J @ w) - self.J_dot @ w)  # J_dot term: J varies with boom angle

        dq_dt = quat_kinematics(q, w)

        return np.concatenate([dq_dt, dw_dt, theta_dot, theta_ddot])
