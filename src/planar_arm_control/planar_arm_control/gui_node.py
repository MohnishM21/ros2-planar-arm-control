#!/usr/bin/env python3
"""
gui_node.py — PyQt5 / PyQtGraph operator console for the planar arm.

Threading model (the graded part)
    Main thread : Qt event loop. Owns every widget. Redraws on a QTimer at
                  `render_hz`, independent of how fast ROS messages arrive.
    ROS thread  : a SingleThreadedExecutor spinning GuiNode in a daemon thread.
                  Callbacks never touch widgets. They either
                    - write the latest sample into TelemetryBuffer (a lock-
                      protected ring buffer the render timer reads), or
                    - emit a Qt signal on RosBridge for one-off events
                      (service responses, action feedback/results). Signals
                      emitted from a non-Qt thread are queued onto the GUI
                      thread automatically.
    Outgoing commands are issued with call_async / send_goal_async, which
    return immediately, so a slow controller never freezes the window.

    joint_states ─▶ [ROS thread] ─▶ TelemetryBuffer ─▶ QTimer redraw ─▶ plots
    responses    ─▶ [ROS thread] ─▶ RosBridge signal ─▶ slot (GUI thread)
    buttons      ─▶ [GUI thread] ─▶ call_async ─────▶ controller
"""

import os
import signal
import sys
import threading
import time
from collections import deque

# pyqtgraph picks the first Qt binding it finds; pin it to PyQt5 so its
# widgets and our QApplication come from the same library.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
from PyQt5 import QtCore, QtWidgets  # noqa: E402  (must precede pyqtgraph)
import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
import rclpy  # noqa: E402
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from planar_arm_control.planar_arm import PlanarArm
from planar_arm_interfaces.action import PickPlace
from planar_arm_interfaces.msg import ArmStatus
from planar_arm_interfaces.srv import MoveToTarget

LINK_LENGTHS = [3.0, 2.0, 1.5]
HISTORY_S = 15.0
JOINT_COLORS = ["#e6194b", "#3cb44b", "#4363d8"]


# ==========================================================================
# ROS side
# ==========================================================================
class TelemetryBuffer:
    """Thread-safe hand-off from the ROS thread to the render timer."""

    def __init__(self, maxlen=3000):
        self.lock = threading.Lock()
        self.t = deque(maxlen=maxlen)
        self.q = deque(maxlen=maxlen)
        self.q_ref = deque(maxlen=maxlen)
        self.err = deque(maxlen=maxlen)
        self.latest_q = None
        self.latest_status = None
        self.msg_count = 0

    def add_joint_state(self, q):
        with self.lock:
            self.latest_q = q
            self.msg_count += 1

    def add_status(self, status):
        with self.lock:
            self.latest_status = status
            self.t.append(time.monotonic())
            self.q.append(np.degrees(status_measured(status)))
            self.q_ref.append(np.degrees(status.desired_position))
            self.err.append(np.degrees(status.tracking_error))

    def snapshot(self):
        with self.lock:
            return dict(t=np.array(self.t), q=np.array(self.q),
                        q_ref=np.array(self.q_ref), err=np.array(self.err),
                        latest_q=self.latest_q, status=self.latest_status,
                        msg_count=self.msg_count)


def status_measured(status):
    return np.asarray(status.desired_position) - np.asarray(status.tracking_error)


class RosBridge(QtCore.QObject):
    """Signals carrying ROS events into the Qt thread."""
    log = QtCore.pyqtSignal(str, str)                 # level, text
    move_response = QtCore.pyqtSignal(object)
    action_feedback = QtCore.pyqtSignal(object)
    action_done = QtCore.pyqtSignal(bool, str)
    action_running = QtCore.pyqtSignal(bool)


class GuiNode(Node):
    def __init__(self, buffer, bridge):
        super().__init__("gui_node")
        self.buffer = buffer
        self.bridge = bridge
        self.goal_handle = None

        self.create_subscription(JointState, "joint_states", self.on_joint_state, 10)
        self.create_subscription(ArmStatus, "arm_status", self.on_status, 10)
        self.move_client = self.create_client(MoveToTarget, "move_to_target")
        self.home_client = self.create_client(Trigger, "home")
        self.stop_client = self.create_client(Trigger, "stop")
        self.param_client = self.create_client(
            SetParameters, "controller_node/set_parameters")
        self.pick_place_client = ActionClient(self, PickPlace, "pick_place")

    # -- subscriptions (ROS thread) ------------------------------------------
    def on_joint_state(self, msg):
        self.buffer.add_joint_state(np.asarray(msg.position, float))

    def on_status(self, msg):
        self.buffer.add_status(msg)

    # -- commands (called from the GUI thread; all non-blocking) --------------
    def _ready(self, client, name):
        if not client.service_is_ready():
            self.bridge.log.emit("error", f"{name} unavailable — is controller_node running?")
            return False
        return True

    def move_to(self, x, y):
        if not self._ready(self.move_client, "/move_to_target"):
            return
        req = MoveToTarget.Request(x=float(x), y=float(y), duration=0.0)
        fut = self.move_client.call_async(req)
        fut.add_done_callback(lambda f: self.bridge.move_response.emit(f.result()))

    def trigger(self, which):
        client = {"home": self.home_client, "stop": self.stop_client}[which]
        if not self._ready(client, f"/{which}"):
            return
        fut = client.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self.bridge.log.emit(
            "info" if f.result().success else "warn", f"{which}: {f.result().message}"))

    def set_mode(self, mode):
        if not self._ready(self.param_client, "set_parameters"):
            return
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=mode)
        req = SetParameters.Request(parameters=[Parameter(name="control_mode", value=value)])
        fut = self.param_client.call_async(req)

        def done(f):
            r = f.result().results[0]
            if r.successful:
                self.bridge.log.emit("info", f"control mode -> {mode}")
            else:
                self.bridge.log.emit("warn", f"mode change refused: {r.reason}")
        fut.add_done_callback(done)

    def pick_place(self, pick, place, return_home):
        if not self.pick_place_client.server_is_ready():
            self.bridge.log.emit("error", "/pick_place action server unavailable")
            return
        goal = PickPlace.Goal()
        goal.pick.x, goal.pick.y = map(float, pick)
        goal.place.x, goal.place.y = map(float, place)
        goal.return_home = return_home
        fut = self.pick_place_client.send_goal_async(
            goal, feedback_callback=lambda fb: self.bridge.action_feedback.emit(fb.feedback))
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, fut):
        handle = fut.result()
        if not handle.accepted:
            self.bridge.log.emit("warn", "pick-and-place rejected (arm busy)")
            return
        self.goal_handle = handle
        self.bridge.action_running.emit(True)
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, fut):
        result = fut.result().result
        self.goal_handle = None
        self.bridge.action_running.emit(False)
        self.bridge.action_done.emit(result.success, result.message)

    def cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.bridge.log.emit("warn", "cancel requested")


class RosThread:
    """Spins the node in a daemon thread until shutdown()."""

    def __init__(self, node):
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self.thread = threading.Thread(target=self._spin, name="ros-spin", daemon=True)

    def _spin(self):
        try:
            self.executor.spin()
        except Exception:  # executor shut down while waiting
            pass

    def start(self):
        self.thread.start()

    def shutdown(self):
        self.executor.shutdown(timeout_sec=1.0)
        self.thread.join(timeout=2.0)


# ==========================================================================
# Qt side
# ==========================================================================
def spin_box(value, lo=-8.0, hi=8.0):
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(2)
    box.setSingleStep(0.1)
    box.setValue(value)
    return box


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, node, buffer, bridge, render_hz=30.0):
        super().__init__()
        self.node = node
        self.buffer = buffer
        self.arm = PlanarArm(LINK_LENGTHS)
        self.trail = deque(maxlen=400)
        self.last_msg_count = 0
        self.last_rate_check = time.monotonic()
        self.msg_rate = 0.0
        self.frame_times = deque(maxlen=60)
        self.banner_source = None

        self.setWindowTitle("Planar Arm Operator Console")
        self.resize(1400, 860)
        pg.setConfigOptions(antialias=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self._build_arm_view(), 6)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self._build_plots(), 5)
        right.addWidget(self._build_controls(), 4)
        layout.addLayout(right, 6)

        bridge.log.connect(self.log)
        bridge.move_response.connect(self.on_move_response)
        bridge.action_feedback.connect(self.on_action_feedback)
        bridge.action_done.connect(self.on_action_done)
        bridge.action_running.connect(self.on_action_running)

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.timeout.connect(self.render)
        self.render_timer.start(int(1000 / render_hz))

    # -- layout --------------------------------------------------------------
    def _build_arm_view(self):
        view = pg.PlotWidget(title="Arm (live)")
        view.setAspectLocked(True)
        view.showGrid(x=True, y=True, alpha=0.2)
        view.setMinimumWidth(560)
        view.setXRange(-7.8, 7.8, padding=0)
        view.setYRange(-1, 7.5, padding=0)
        view.getPlotItem().vb.disableAutoRange()   # keep the workspace framed
        view.setLabel("bottom", "x")
        view.setLabel("left", "y")
        reach = sum(LINK_LENGTHS)
        th = np.linspace(0, np.pi, 200)
        view.plot(reach * np.cos(th), reach * np.sin(th),
                  pen=pg.mkPen("#888", style=QtCore.Qt.DashLine))
        ground = pg.LinearRegionItem([-1.0, 0.0], orientation="horizontal",
                                     movable=False, brush=pg.mkBrush(120, 90, 60, 60))
        view.addItem(ground)
        view.addLine(y=0, pen=pg.mkPen("#a07040", width=2))

        self.trail_item = view.plot(pen=pg.mkPen((255, 200, 0, 120), width=1))
        self.ghost_item = view.plot(pen=pg.mkPen((150, 150, 255, 120), width=3,
                                                 style=QtCore.Qt.DashLine))
        self.arm_item = view.plot(pen=pg.mkPen("#4fc3f7", width=6), symbol="o",
                                  symbolSize=12, symbolBrush="#ffffff")
        self.pick_item = view.plot(pen=None, symbol="s", symbolSize=14,
                                   symbolBrush=None, symbolPen=pg.mkPen("#3cb44b", width=2))
        self.place_item = view.plot(pen=None, symbol="s", symbolSize=14,
                                    symbolBrush=None, symbolPen=pg.mkPen("#4363d8", width=2))
        self.projection_line = view.plot(pen=pg.mkPen("#ff5555", width=2,
                                                      style=QtCore.Qt.DotLine))
        self.requested_item = view.plot(pen=None, symbol="x", symbolSize=16,
                                        symbolBrush="#ff5555", symbolPen="#ff5555")
        self.object_item = view.plot(pen=None, symbol="s", symbolSize=11,
                                     symbolBrush="#ffd600")
        self.gripper_item = view.plot(pen=None, symbol="t1", symbolSize=16)
        view.scene().sigMouseClicked.connect(self.on_arm_view_clicked)
        self.arm_view = view
        return view

    def _build_plots(self):
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        self.angle_plot = pg.PlotWidget(title="Joint angles (deg) — solid: measured, dashed: reference")
        self.angle_plot.addLegend(offset=(5, 5))
        self.angle_plot.showGrid(x=True, y=True, alpha=0.2)
        self.error_plot = pg.PlotWidget(title="Tracking error (deg)")
        self.error_plot.showGrid(x=True, y=True, alpha=0.2)
        self.error_plot.setXLink(self.angle_plot)
        self.error_plot.getPlotItem().vb.disableAutoRange(axis=pg.ViewBox.YAxis)
        self.angle_curves, self.ref_curves, self.err_curves = [], [], []
        for i, c in enumerate(JOINT_COLORS):
            self.angle_curves.append(self.angle_plot.plot(pen=pg.mkPen(c, width=2), name=f"q{i+1}"))
            self.ref_curves.append(self.angle_plot.plot(
                pen=pg.mkPen(c, width=1, style=QtCore.Qt.DashLine)))
            self.err_curves.append(self.error_plot.plot(pen=pg.mkPen(c, width=2)))
        for p in (self.angle_plot, self.error_plot):
            p.setLabel("bottom", "time (s)")
        lay.addWidget(self.angle_plot, 3)
        lay.addWidget(self.error_plot, 2)
        return box

    def _build_controls(self):
        panel = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(panel)
        grid.setContentsMargins(0, 0, 0, 0)

        # Banner for warnings such as projected targets.
        self.banner = QtWidgets.QLabel()
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("background:#ff9800;color:black;padding:6px;"
                                  "border-radius:4px;font-weight:bold;")
        self.banner.hide()
        grid.addWidget(self.banner, 0, 0, 1, 2)

        # -- commands
        cmd = QtWidgets.QGroupBox("Commands")
        f = QtWidgets.QGridLayout(cmd)
        self.pick_x, self.pick_y = spin_box(4.0), spin_box(2.0)
        self.place_x, self.place_y = spin_box(-3.0), spin_box(3.0)
        self.move_x, self.move_y = spin_box(7.0), spin_box(3.0)
        rows = [("Pick", self.pick_x, self.pick_y),
                ("Place", self.place_x, self.place_y),
                ("Move to", self.move_x, self.move_y)]
        for r, (label, bx, by) in enumerate(rows):
            f.addWidget(QtWidgets.QLabel(label), r, 0)
            f.addWidget(QtWidgets.QLabel("x"), r, 1)
            f.addWidget(bx, r, 2)
            f.addWidget(QtWidgets.QLabel("y"), r, 3)
            f.addWidget(by, r, 4)
        self.return_home = QtWidgets.QCheckBox("return home after place")
        f.addWidget(self.return_home, 3, 0, 1, 5)

        self.pp_button = QtWidgets.QPushButton("▶ Run pick && place")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        move_button = QtWidgets.QPushButton("Move to target")
        home_button = QtWidgets.QPushButton("Home")
        stop_button = QtWidgets.QPushButton("■ STOP")
        stop_button.setStyleSheet("background:#c62828;color:white;font-weight:bold;")
        self.pp_button.clicked.connect(self.send_pick_place)
        self.cancel_button.clicked.connect(self.node.cancel)
        move_button.clicked.connect(
            lambda: self.node.move_to(self.move_x.value(), self.move_y.value()))
        home_button.clicked.connect(lambda: self.node.trigger("home"))
        stop_button.clicked.connect(lambda: self.node.trigger("stop"))
        buttons = QtWidgets.QHBoxLayout()
        for b in (self.pp_button, self.cancel_button, move_button, home_button, stop_button):
            buttons.addWidget(b)
        f.addLayout(buttons, 4, 0, 1, 5)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setFormat("%p%  idle")
        f.addWidget(self.progress, 5, 0, 1, 5)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Control mode"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["position", "velocity", "effort"])
        self.mode_combo.activated[str].connect(self.node.set_mode)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        mode_row.addWidget(QtWidgets.QLabel("Tip: click the arm view to set 'Move to'"))
        f.addLayout(mode_row, 6, 0, 1, 5)
        grid.addWidget(cmd, 1, 0)

        # -- telemetry
        tel = QtWidgets.QGroupBox("Telemetry")
        form = QtWidgets.QFormLayout(tel)
        self.tel = {}
        for key in ("State", "Mode", "Phase", "End-effector", "Joints (deg)",
                    "Target", "Gripper", "Control loop", "Link"):
            lbl = QtWidgets.QLabel("—")
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.tel[key] = lbl
            form.addRow(key, lbl)
        grid.addWidget(tel, 1, 1)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        grid.addWidget(self.log_view, 2, 0, 1, 2)
        return panel

    # -- slots (GUI thread) -------------------------------------------------
    @QtCore.pyqtSlot(str, str)
    def log(self, level, text):
        color = {"info": "#cccccc", "warn": "#ffb74d", "error": "#ef5350"}.get(level, "#ccc")
        stamp = time.strftime("%H:%M:%S")
        self.log_view.appendHtml(f'<span style="color:{color}">[{stamp}] {text}</span>')

    def show_banner(self, text, source="command"):
        # source: "command" (sticky until the next command) or "status"
        # (mirrors the controller's target_projected flag).
        self.banner_source = source
        self.banner.setText(text)
        self.banner.show()

    def hide_banner(self):
        self.banner_source = None
        self.banner.hide()

    def send_pick_place(self):
        self.hide_banner()
        pick = (self.pick_x.value(), self.pick_y.value())
        place = (self.place_x.value(), self.place_y.value())
        self.log("info", f"pick-and-place: pick {pick} → place {place}")
        self.node.pick_place(pick, place, self.return_home.isChecked())

    def on_move_response(self, resp):
        if not resp.accepted:
            self.show_banner(f"⛔ Move rejected: {resp.message}")
            self.log("error", f"move rejected: {resp.message}")
            return
        p = resp.planned_target
        if not resp.reachable:
            self.show_banner(f"⚠ {resp.message}. Moving to the closest reachable point.")
            self.log("warn", resp.message)
        else:
            self.hide_banner()
        self.log("info", f"moving to ({p.x:.3f}, {p.y:.3f}) in {resp.duration:.2f} s")

    def on_action_feedback(self, fb):
        self.progress.setValue(int(fb.progress * 100))
        self.progress.setFormat(f"%p%  {fb.phase}")

    def on_action_running(self, running):
        self.pp_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def on_action_done(self, success, message):
        self.progress.setFormat("%p%  " + ("done" if success else "failed"))
        if success:
            self.log("info", f"pick-and-place succeeded: {message}")
        else:
            self.log("error", f"pick-and-place failed: {message}")
            if "unreachable" in message:
                self.show_banner(f"⚠ {message}")

    def on_arm_view_clicked(self, event):
        pos = self.arm_view.getPlotItem().vb.mapSceneToView(event.scenePos())
        self.move_x.setValue(pos.x())
        self.move_y.setValue(pos.y())

    # -- render loop (GUI thread, fixed rate) ------------------------------
    def render(self):
        now = time.monotonic()
        self.frame_times.append(now)
        snap = self.buffer.snapshot()
        if now - self.last_rate_check >= 1.0:
            self.msg_rate = (snap["msg_count"] - self.last_msg_count) / (now - self.last_rate_check)
            self.last_msg_count = snap["msg_count"]
            self.last_rate_check = now
        fps = ((len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
               if len(self.frame_times) > 1 else 0.0)
        self.tel["Link"].setText(f"/joint_states {self.msg_rate:.0f} Hz · GUI {fps:.0f} fps")

        q = snap["latest_q"]
        st = snap["status"]
        if q is None or st is None:
            self.tel["State"].setText("waiting for controller…")
            return

        pts = np.array(self.arm.forward_kinematics(q))
        self.arm_item.setData(pts[:, 0], pts[:, 1])
        ghost = np.array(self.arm.forward_kinematics(st.desired_position))
        self.ghost_item.setData(ghost[:, 0], ghost[:, 1])
        self.trail.append(pts[-1])
        trail = np.array(self.trail)
        self.trail_item.setData(trail[:, 0], trail[:, 1])

        self.gripper_item.setData([pts[-1, 0]], [pts[-1, 1]],
                                  symbolBrush="#ff7043" if st.gripper_closed else "#9e9e9e")
        if st.has_object:
            self.object_item.setData([st.object_position.x], [st.object_position.y])
        else:
            self.object_item.setData([], [])
        self.pick_item.setData([self.pick_x.value()], [self.pick_y.value()])
        self.place_item.setData([self.place_x.value()], [self.place_y.value()])

        req, act = st.requested_target, st.active_target
        if st.target_projected:
            self.requested_item.setData([req.x], [req.y])
            self.projection_line.setData([req.x, act.x], [req.y, act.y])
            if self.banner_source is None:
                # Commands from any client (CLI, scripts) surface here too.
                self.show_banner(f"⚠ Requested ({req.x:.2f}, {req.y:.2f}) is outside the "
                                 f"workspace. Driving to the closest reachable point "
                                 f"({act.x:.3f}, {act.y:.3f}).", source="status")
        else:
            self.requested_item.setData([], [])
            self.projection_line.setData([], [])
            if self.banner_source == "status":
                self.hide_banner()

        # Time-series plots.
        t = snap["t"]
        if len(t) > 1:
            mask = t >= t[-1] - HISTORY_S
            tt = t[mask] - t[-1]
            for i in range(3):
                self.angle_curves[i].setData(tt, snap["q"][mask, i])
                self.ref_curves[i].setData(tt, snap["q_ref"][mask, i])
                self.err_curves[i].setData(tt, snap["err"][mask, i])
            # Symmetric y-range with a floor so numerical noise isn't magnified.
            span = max(0.5, 1.2 * float(np.max(np.abs(snap["err"][mask]))))
            self.error_plot.setYRange(-span, span, padding=0)

        # Telemetry labels.
        ee = st.end_effector
        self.tel["State"].setText(st.state)
        self.tel["Mode"].setText(st.control_mode)
        if self.mode_combo.currentText() != st.control_mode and not self.mode_combo.view().isVisible():
            self.mode_combo.setCurrentText(st.control_mode)
        self.tel["Phase"].setText(st.phase or "—")
        self.tel["End-effector"].setText(f"x = {ee.x:+.3f}   y = {ee.y:+.3f}")
        self.tel["Joints (deg)"].setText("  ".join(f"{a:+7.2f}" for a in np.degrees(q)))
        tgt = f"({act.x:+.3f}, {act.y:+.3f})"
        if st.target_projected:
            tgt += f"  ⚠ projected from ({req.x:.2f}, {req.y:.2f})"
        self.tel["Target"].setText(tgt)
        self.tel["Target"].setStyleSheet("color:#ff9800;" if st.target_projected else "")
        grip = "closed" if st.gripper_closed else "open"
        self.tel["Gripper"].setText(grip + (" · holding object" if st.gripper_closed and st.has_object else ""))
        self.tel["Control loop"].setText(
            f"period {st.control_period_mean_ms:.2f} ms · max jitter {st.control_jitter_max_ms:.2f} ms")


def main(args=None):
    rclpy.init(args=args)
    app = QtWidgets.QApplication(sys.argv)

    buffer = TelemetryBuffer()
    bridge = RosBridge()
    node = GuiNode(buffer, bridge)
    ros = RosThread(node)
    window = MainWindow(node, buffer, bridge)
    window.show()
    ros.start()

    # Qt's C++ loop starves Python signal handlers. Route SIGINT/SIGTERM to
    # app.quit() (rclpy's own handler would only shut ROS down and leave the
    # window open), and wake the interpreter periodically so they can run.
    # Also quit if the ROS context goes away for any other reason.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.quit())
    watchdog = QtCore.QTimer()
    watchdog.timeout.connect(lambda: None if rclpy.ok() else app.quit())
    watchdog.start(200)

    exit_code = app.exec_()
    ros.shutdown()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
