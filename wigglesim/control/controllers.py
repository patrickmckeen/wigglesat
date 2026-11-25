class Controller:
    def u(self,t,x):
        pass

class DirectInputController(Controller):
    '''takes in input to booms (u_b) and magnetorquers (u_m) directly.
    u_b is hinge accelerations [rad/s^2], or commanded angles [rad] when motor_dynamics=True.'''
    def __init__(self, u_b, u_m):
        self.u_b = u_b
        self.u_m = u_m

    def u(self, t, x):
        return self.u_b, self.u_m
