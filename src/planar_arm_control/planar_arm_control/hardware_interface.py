"""
hardware_interface.py — the seam between the planner/controller and "the arm".

The controller only ever talks to an `ArmBackend`: it reads joint state and
writes a `JointCommand` in one of three modes. Swapping simulation for real
servos (e.g. Dynamixel) means writing one new backend class; the planner,
trajectory generator, PID loops, ROS interfaces and GUI stay untouched.

    ControllerNode ──JointCommand──▶ ArmBackend.write()
                   ◀──JointReading── ArmBackend.read()

Backends provided:
    SimulatedArm     — a simple dynamic model (inertia, damping, gravity,
                       servo lag, hard stops). Default.
    DynamixelBackend — skeleton showing where the SDK calls would go.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ControlMode(str, Enum):
    POSITION = "position"   # command = joint angle (rad); servo closes the loop
    VELOCITY = "velocity"   # command = joint velocity (rad/s)
    EFFORT = "effort"       # command = joint torque (Nm) ~ motor current


@dataclass
class JointCommand:
    mode: ControlMode
    values: np.ndarray


@dataclass
class JointReading:
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ArmBackend(ABC):
    """Abstract hardware abstraction layer for an n-joint arm."""

    supported_modes = ()

    @abstractmethod
    def read(self) -> JointReading:
        """Latest measured joint state."""

    @abstractmethod
    def write(self, command: JointCommand) -> None:
        """Send a command. Must be non-blocking (called from the control loop)."""

    def step(self, dt: float) -> None:
        """Advance time. Simulations integrate here; hardware does nothing."""

    def set_mode(self, mode: ControlMode) -> None:
        if mode not in self.supported_modes:
            raise ValueError(f"{type(self).__name__} does not support {mode.value}")

    def enable(self) -> None:
        """Torque on."""

    def disable(self) -> None:
        """Torque off / safe stop."""

    def close(self) -> None:
        """Release ports and resources."""


class SimulatedArm(ArmBackend):
    """Decoupled per-joint dynamics with gravity, for a vertical planar arm.

        J_i * qdd_i = tau_i - b_i * qd_i - g_i(q)

    Position and velocity modes model the servo's internal loop as a
    first-order lag, like a Dynamixel in position/velocity mode.
    Hard joint limits act as mechanical stops.
    """

    supported_modes = (ControlMode.POSITION, ControlMode.VELOCITY,
                       ControlMode.EFFORT)

    def __init__(self, link_lengths, joint_limits, q0,
                 inertia=(3.0, 1.0, 0.3), damping=(2.0, 1.0, 0.3),
                 link_masses=(1.0, 0.7, 0.4), gravity=9.81,
                 max_velocity=(2.0, 2.5, 3.0), max_effort=(120.0, 50.0, 15.0),
                 position_time_constant=0.05, velocity_time_constant=0.03,
                 noise_std=0.0, seed=None):
        self.link_lengths = np.asarray(link_lengths, float)
        self.joint_limits = np.asarray(joint_limits, float)
        self.q = np.asarray(q0, float).copy()
        self.qd = np.zeros_like(self.q)
        self.tau = np.zeros_like(self.q)
        self.inertia = np.asarray(inertia, float)
        self.damping = np.asarray(damping, float)
        self.link_masses = np.asarray(link_masses, float)
        self.gravity = gravity
        self.max_velocity = np.asarray(max_velocity, float)
        self.max_effort = np.asarray(max_effort, float)
        self.tau_pos = position_time_constant
        self.tau_vel = velocity_time_constant
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.mode = ControlMode.POSITION
        self.command = self.q.copy()
        self.enabled = True

    # -- ArmBackend API ----------------------------------------------------
    def set_mode(self, mode):
        super().set_mode(mode)
        self.mode = mode
        # Bumpless switch: hold the current state in the new mode.
        self.command = (self.q.copy() if mode == ControlMode.POSITION
                        else np.zeros_like(self.q))

    def write(self, command):
        if command.mode != self.mode:
            self.set_mode(command.mode)
        self.command = np.asarray(command.values, float).copy()

    def read(self):
        noise = (self.rng.normal(0.0, self.noise_std, self.q.shape)
                 if self.noise_std > 0 else 0.0)
        return JointReading(self.q + noise, self.qd.copy(), self.tau.copy())

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.qd[:] = 0.0

    # -- physics -----------------------------------------------------------
    def gravity_torque(self, q):
        """Torque each joint must supply to hold the arm still against gravity."""
        angles = np.cumsum(q)
        joint_x = np.concatenate(([0.0], np.cumsum(self.link_lengths * np.cos(angles))))
        com_x = joint_x[:-1] + 0.5 * self.link_lengths * np.cos(angles)
        tau = np.zeros_like(q)
        for i in range(len(q)):
            tau[i] = np.sum(self.link_masses[i:] * self.gravity
                            * (com_x[i:] - joint_x[i]))
        return tau

    def step(self, dt):
        if dt <= 0.0 or not self.enabled:
            return
        if self.mode == ControlMode.POSITION:
            target = np.clip(self.command, self.joint_limits[:, 0], self.joint_limits[:, 1])
            qd_cmd = (target - self.q) / self.tau_pos
            self._velocity_servo(qd_cmd, dt)
        elif self.mode == ControlMode.VELOCITY:
            self._velocity_servo(self.command, dt)
        else:
            self.tau = np.clip(self.command, -self.max_effort, self.max_effort)
            qdd = (self.tau - self.damping * self.qd
                   - self.gravity_torque(self.q)) / self.inertia
            self.qd += qdd * dt
            self.qd = np.clip(self.qd, -self.max_velocity, self.max_velocity)
            self.q += self.qd * dt
        self._hard_stops()

    def _velocity_servo(self, qd_cmd, dt):
        qd_cmd = np.clip(qd_cmd, -self.max_velocity, self.max_velocity)
        alpha = min(1.0, dt / self.tau_vel)
        self.qd += alpha * (qd_cmd - self.qd)
        self.q += self.qd * dt
        # Report the torque the servo would be producing (useful telemetry).
        self.tau = self.gravity_torque(self.q) + self.damping * self.qd

    def _hard_stops(self):
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        hit = (self.q < lo) | (self.q > hi)
        self.q = np.clip(self.q, lo, hi)
        self.qd[hit] = 0.0


class DynamixelBackend(ArmBackend):
    """Skeleton for a real Dynamixel X-series arm (not exercised in this task).

    Mapping a JointCommand onto the Protocol 2.0 control table:
        POSITION -> Operating Mode 3, Goal Position (addr 116), 4096 ticks/rev
        VELOCITY -> Operating Mode 1, Goal Velocity (addr 104), 0.229 rpm/unit
        EFFORT   -> Operating Mode 0, Goal Current  (addr 102), ~2.69 mA/unit
    Reads use one GroupSyncRead of Present Current/Velocity/Position
    (addr 126..135) per cycle; writes use one GroupSyncWrite, so a control tick
    costs two bus transactions regardless of joint count.
    Changing Operating Mode requires torque off, so set_mode() disables first.
    """

    supported_modes = (ControlMode.POSITION, ControlMode.VELOCITY,
                       ControlMode.EFFORT)

    def __init__(self, port="/dev/ttyUSB0", baudrate=1_000_000, ids=(1, 2, 3),
                 zero_offsets=(2048, 2048, 2048), directions=(1, 1, 1)):
        try:
            import dynamixel_sdk  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "dynamixel_sdk is not installed; use backend:=sim") from exc
        self.port, self.baudrate, self.ids = port, baudrate, ids
        self.zero_offsets = np.asarray(zero_offsets)
        self.directions = np.asarray(directions)
        raise NotImplementedError("Hardware backend left as a documented stub.")

    def read(self):
        raise NotImplementedError

    def write(self, command):
        raise NotImplementedError


def make_backend(name, **kwargs):
    if name == "sim":
        return SimulatedArm(**kwargs)
    if name == "dynamixel":
        return DynamixelBackend()
    raise ValueError(f"Unknown backend '{name}' (expected 'sim' or 'dynamixel')")
