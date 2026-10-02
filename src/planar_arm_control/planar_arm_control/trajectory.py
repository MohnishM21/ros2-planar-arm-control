"""
trajectory.py — time-parameterised joint-space trajectories.

A quintic (5th-order) polynomial per joint with zero start/end velocity and
acceleration. All joints share one duration so they start and finish together
(synchronised motion), which keeps the end-effector path predictable.

Why quintic over trapezoidal: it is C2-continuous (no acceleration steps),
so there is no jerk spike at the blend points. That matters on real servos,
where acceleration discontinuities show up as current spikes and audible
"clunks". The cost is a ~25% higher peak velocity for the same duration,
which we account for when choosing the duration.

Pure numpy, no ROS — so it can be unit-tested and reused by any backend.
"""

import numpy as np

# For s(tau) = 10tau^3 - 15tau^4 + 6tau^5 on tau in [0, 1]:
QUINTIC_PEAK_VEL = 1.875                  # max |ds/dtau|   at tau = 0.5
QUINTIC_PEAK_ACC = 10.0 / np.sqrt(3.0)    # max |d2s/dtau2| ~= 5.7735


class QuinticTrajectory:
    """Synchronised point-to-point quintic trajectory in joint space."""

    def __init__(self, q_start, q_goal, duration):
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_goal = np.asarray(q_goal, dtype=float)
        if self.q_start.shape != self.q_goal.shape:
            raise ValueError("q_start and q_goal must have the same shape")
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        self.duration = float(duration)
        self.delta = self.q_goal - self.q_start

    @staticmethod
    def min_duration(q_start, q_goal, max_velocity, max_acceleration,
                     floor=0.5):
        """Shortest duration that keeps every joint within vel/acc limits."""
        delta = np.abs(np.asarray(q_goal, float) - np.asarray(q_start, float))
        dmax = float(np.max(delta)) if delta.size else 0.0
        t_vel = QUINTIC_PEAK_VEL * dmax / max_velocity
        t_acc = np.sqrt(QUINTIC_PEAK_ACC * dmax / max_acceleration)
        return max(floor, t_vel, t_acc)

    def sample(self, t):
        """Return (q, qd, qdd) at time t (clamped to [0, duration])."""
        T = self.duration
        tau = min(max(t / T, 0.0), 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
        return (self.q_start + s * self.delta,
                ds * self.delta,
                dds * self.delta)

    def finished(self, t):
        return t >= self.duration

    def sample_positions(self, n=100):
        """Evenly spaced positions along the path (for safety validation)."""
        return [self.sample(t)[0]
                for t in np.linspace(0.0, self.duration, n)]


class TrajectoryChain:
    """
    A sequence of trajectories executed back to back.

    Used to pass through a safe waypoint; looks like a single trajectory to
    the controller.
    """

    def __init__(self, segments):
        if not segments:
            raise ValueError("TrajectoryChain needs at least one segment")
        self.segments = list(segments)
        self.duration = sum(seg.duration for seg in self.segments)
        self.q_goal = self.segments[-1].q_goal

    def sample(self, t):
        for seg in self.segments[:-1]:
            if t < seg.duration:
                return seg.sample(t)
            t -= seg.duration
        return self.segments[-1].sample(t)

    def finished(self, t):
        return t >= self.duration

    def sample_positions(self, n=100):
        return [self.sample(t)[0]
                for t in np.linspace(0.0, self.duration, n)]
