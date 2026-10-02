"""
pid.py — vectorised PID for joint-space trajectory tracking.

One gain set per joint. Includes the bits that matter on real hardware:
a derivative term on the velocity error when the reference rate is known
(trajectory tracking) or on the measurement otherwise (no derivative kick
on setpoint steps), anti-windup
(the integral *term* is clamped to `integral_limit`, in output units, so its
authority is explicit), output saturation, and an optional feed-forward
term supplied by the caller (reference velocity or inertia * acceleration).
"""

import numpy as np


class PID:
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=None):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        n = self.kp.size
        self.output_limit = (None if output_limit is None
                             else np.broadcast_to(np.asarray(output_limit, float), (n,)))
        self.integral_limit = (None if integral_limit is None
                               else np.broadcast_to(np.asarray(integral_limit, float), (n,)))
        self.reset(n)

    def reset(self, n=None):
        n = n if n is not None else self.kp.size
        self.integral = np.zeros(n)
        self.prev_measurement = None
        self.last_error = np.zeros(n)

    def update(self, setpoint, measurement, dt, feedforward=0.0,
               setpoint_rate=None):
        setpoint = np.asarray(setpoint, dtype=float)
        measurement = np.asarray(measurement, dtype=float)
        error = setpoint - measurement
        self.last_error = error

        if dt <= 0.0:
            return np.asarray(feedforward, float) + self.kp * error

        self.integral += error * dt
        if self.integral_limit is not None:
            # Clamp ki * integral to +/- integral_limit (skip joints with ki = 0).
            ki_safe = np.where(self.ki > 0, self.ki, 1.0)
            bound = np.where(self.ki > 0, self.integral_limit / ki_safe, 0.0)
            self.integral = np.clip(self.integral, -bound, bound)

        if self.prev_measurement is None:
            d_meas = np.zeros_like(measurement)
        else:
            d_meas = (measurement - self.prev_measurement) / dt
        self.prev_measurement = measurement
        # Without a reference rate this is plain derivative-on-measurement.
        d_error = (np.asarray(setpoint_rate, float) - d_meas
                   if setpoint_rate is not None else -d_meas)

        out = (np.asarray(feedforward, float)
               + self.kp * error
               + self.ki * self.integral
               + self.kd * d_error)
        if self.output_limit is not None:
            out = np.clip(out, -self.output_limit, self.output_limit)
        return out
