"""
tracking.py — turn a trajectory reference into a backend command, per mode.

    position : forward q_ref directly; the servo's internal loop tracks it
               (how a Dynamixel in position mode is normally driven).
    velocity : qd_ref feed-forward + PI on position error  -> rad/s
    effort   : J * qdd_ref feed-forward + PID on position error -> Nm
               (current control; the integrator absorbs unmodelled gravity)

Only this class knows about modes, so adding a mode (e.g. impedance) is a
local change.
"""

import numpy as np

from planar_arm_control.hardware_interface import ControlMode, JointCommand
from planar_arm_control.pid import PID

DEFAULT_GAINS = {
    ControlMode.VELOCITY: dict(kp=[8.0, 8.0, 8.0], ki=[2.0, 2.0, 2.0],
                               kd=[0.0, 0.0, 0.0],
                               output_limit=[2.0, 2.5, 3.0],
                               integral_limit=0.5),
    # kp ~ J * w^2, kd ~ 2 * zeta * w * J with w ~ 17 rad/s, zeta ~ 1.
    # The integral term carries gravity, so it gets real (but bounded) authority.
    ControlMode.EFFORT: dict(kp=[900.0, 300.0, 90.0], ki=[3000.0, 1000.0, 300.0],
                             kd=[120.0, 40.0, 12.0],
                             output_limit=[120.0, 50.0, 15.0],
                             integral_limit=[80.0, 30.0, 8.0]),
}


class TrackingController:
    def __init__(self, mode, inertia=(3.0, 1.0, 0.3), gains=None):
        self.mode = ControlMode(mode)
        self.inertia = np.asarray(inertia, float)
        gains = gains or DEFAULT_GAINS
        self.pid = PID(**gains[self.mode]) if self.mode in gains else None

    def reset(self):
        if self.pid is not None:
            self.pid.reset()

    def compute(self, q_ref, qd_ref, qdd_ref, q_meas, dt):
        q_ref = np.asarray(q_ref, float)
        if self.mode == ControlMode.POSITION:
            return JointCommand(self.mode, q_ref)
        if self.mode == ControlMode.VELOCITY:
            u = self.pid.update(q_ref, q_meas, dt, feedforward=qd_ref)
            return JointCommand(self.mode, u)
        u = self.pid.update(q_ref, q_meas, dt,
                            feedforward=self.inertia * np.asarray(qdd_ref),
                            setpoint_rate=qd_ref)
        return JointCommand(self.mode, u)
