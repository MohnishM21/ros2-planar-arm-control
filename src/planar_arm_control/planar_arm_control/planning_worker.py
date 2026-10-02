"""
planning_worker.py — run MotionPlanner calls in a separate process.

IK is CPU-heavy Python. Run in a thread of the controller process, it holds
the GIL and starves the 100 Hz control loop: measured worst-case loop stalls
were 19 ms for a plain pick-and-place and 142 ms with straight-line approach
moves. A worker process has its own interpreter and GIL, so planning can
take as long as it needs while the control loop keeps its timing. The
calling thread just blocks on a future (releasing the GIL while it waits).
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from planar_arm_control.motion_planner import MotionPlanner
from planar_arm_control.planar_arm import PlanarArm

_planner = None


def _init(link_lengths):
    global _planner
    _planner = MotionPlanner(PlanarArm(link_lengths))


def _call(limits, method, args):
    _planner.max_velocity, _planner.max_acceleration = limits
    return getattr(_planner, method)(*args)


class PlanningWorker:
    """Proxy for MotionPlanner methods, executed in one worker process."""

    def __init__(self, link_lengths, max_velocity, max_acceleration):
        # "spawn": forking a process that already runs DDS threads is unsafe.
        context = multiprocessing.get_context("spawn")
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=context,
                                        initializer=_init, initargs=(link_lengths,))
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.pool.submit(_call, (max_velocity, max_acceleration),
                         "project_target", ((0.0, 1.0),)).result()   # warm up

    def __getattr__(self, method):
        def call(*args, timeout=30.0):
            limits = (self.max_velocity, self.max_acceleration)
            return self.pool.submit(_call, limits, method, args).result(timeout)
        return call

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
