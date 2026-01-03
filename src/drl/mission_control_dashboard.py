#!/usr/bin/env python3
"""
Mission Control Dashboard
- ROS Noetic + PyQt5 + Multithreading safe UI
- Camera preview (OpenCV -> QImage)
- LiDAR scatter (pyqtgraph)
- Process control for Gazebo, DRL agent, and mission manager
"""
import os
import signal
import subprocess
import sys
from typing import List, Optional

import cv2
import numpy as np
import pyqtgraph as pg
import rospy
from cv_bridge import CvBridge
from PyQt5 import QtCore, QtGui, QtWidgets
from sensor_msgs.msg import Image, LaserScan
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandTOL, CommandTOLRequest


class RosWorker(QtCore.QThread):
    image_signal = QtCore.pyqtSignal(np.ndarray)
    lidar_signal = QtCore.pyqtSignal(object)
    log_signal = QtCore.pyqtSignal(str)
    ready_signal = QtCore.pyqtSignal()

    def __init__(self, camera_topic: str, lidar_topic: str, parent=None):
        super().__init__(parent)
        self.camera_topic = camera_topic
        self.lidar_topic = lidar_topic
        self._running = True
        self._subs = []
        self._bridge = CvBridge()

    def run(self):
        try:
            if not rospy.core.is_initialized():
                rospy.init_node("mission_control_dashboard", anonymous=True, disable_signals=True)
            self.log_signal.emit("ROS node initialized")
            cam_sub = rospy.Subscriber(self.camera_topic, Image, self._camera_cb, queue_size=1)
            lidar_sub = rospy.Subscriber(self.lidar_topic, LaserScan, self._lidar_cb, queue_size=1)
            self._subs = [cam_sub, lidar_sub]
            self.ready_signal.emit()
            rate = rospy.Rate(50)
            while self._running and not rospy.is_shutdown():
                rate.sleep()
        except Exception as exc:
            self.log_signal.emit(f"ROS worker error: {exc}")
        finally:
            for sub in self._subs:
                try:
                    sub.unregister()
                except Exception:
                    pass
            self.log_signal.emit("ROS worker stopped")

    def _camera_cb(self, msg: Image):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_signal.emit(cv_img)
        except Exception as exc:
            self.log_signal.emit(f"Camera callback error: {exc}")

    def _lidar_cb(self, msg: LaserScan):
        try:
            self.lidar_signal.emit(msg)
        except Exception as exc:
            self.log_signal.emit(f"LiDAR callback error: {exc}")

    def stop(self):
        self._running = False
        if not rospy.is_shutdown():
            try:
                rospy.signal_shutdown("GUI exit")
            except Exception:
                pass
        self.wait(1500)


class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mission Control Dashboard")
        self.resize(1400, 800)

        self.launch_proc: Optional[subprocess.Popen] = None
        self.train_proc: Optional[subprocess.Popen] = None
        self.mission_proc: Optional[subprocess.Popen] = None
        self.procs: List[subprocess.Popen] = []
        self.proc_readers: List[_ProcessReader] = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        self._build_ui(central)
        self._setup_ros_worker()

        self.set_mode_srv = None
        self.arm_srv = None
        self.takeoff_srv = None

    def _build_ui(self, parent: QtWidgets.QWidget):
        main_layout = QtWidgets.QHBoxLayout(parent)

        left = QtWidgets.QVBoxLayout()
        self.btn_launch = QtWidgets.QPushButton("Gazebo Başlat")
        self.btn_train = QtWidgets.QPushButton("Eğitimi Başlat (drl_agent_node.py)")
        self.btn_mission = QtWidgets.QPushButton("Görevi Başlat (mission_manager.py)")
        self.btn_takeoff = QtWidgets.QPushButton("Kalkış (Takeoff)")
        self.btn_land = QtWidgets.QPushButton("İniş (Land)")
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)

        for btn in [self.btn_launch, self.btn_train, self.btn_mission, self.btn_takeoff, self.btn_land]:
            btn.setMinimumHeight(40)

        left.addWidget(self.btn_launch)
        left.addWidget(self.btn_train)
        left.addWidget(self.btn_mission)
        left.addWidget(self.btn_takeoff)
        left.addWidget(self.btn_land)
        left.addWidget(QtWidgets.QLabel("Durum Log"))
        left.addWidget(self.log_box, stretch=1)

        mid = QtWidgets.QVBoxLayout()
        mid.addWidget(QtWidgets.QLabel("Kamera Yayını"))
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #111;")
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        mid.addWidget(self.camera_label)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Engel Haritası (LiDAR)"))
        self.lidar_plot = pg.PlotWidget()
        self.lidar_plot.setAspectLocked(True)
        self.lidar_plot.showGrid(x=True, y=True, alpha=0.25)
        self.scatter = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(0, 200, 0, 180))
        self.origin = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(220, 50, 50, 200))
        self.origin.setData([{"pos": (0.0, 0.0)}])
        self.lidar_plot.addItem(self.scatter)
        self.lidar_plot.addItem(self.origin)
        right.addWidget(self.lidar_plot, stretch=1)

        main_layout.addLayout(left, 2)
        main_layout.addLayout(mid, 3)
        main_layout.addLayout(right, 3)

        self.btn_launch.clicked.connect(self.start_launch)
        self.btn_train.clicked.connect(self.start_training)
        self.btn_mission.clicked.connect(self.start_mission)
        self.btn_takeoff.clicked.connect(self.cmd_takeoff)
        self.btn_land.clicked.connect(self.cmd_land)

    def _setup_ros_worker(self):
        self.ros_worker = RosWorker("/camera/rgb/image_raw", "/scan")
        self.ros_worker.image_signal.connect(self._on_camera_frame, QtCore.Qt.QueuedConnection)
        self.ros_worker.lidar_signal.connect(self._on_lidar_scan, QtCore.Qt.QueuedConnection)
        self.ros_worker.log_signal.connect(self._append_log, QtCore.Qt.QueuedConnection)
        self.ros_worker.ready_signal.connect(self._on_ros_ready, QtCore.Qt.QueuedConnection)
        self.ros_worker.start()

    @QtCore.pyqtSlot()
    def _on_ros_ready(self):
        self._append_log("ROS worker ready, waiting for MavROS services...")
        self.set_mode_srv = self._wait_service("/mavros/set_mode", SetMode)
        self.arm_srv = self._wait_service("/mavros/cmd/arming", CommandBool)
        self.takeoff_srv = self._wait_service("/mavros/cmd/takeoff", CommandTOL)

    def _wait_service(self, name, srv_type):
        try:
            rospy.wait_for_service(name, timeout=3.0)
            self._append_log(f"Service ready: {name}")
            return rospy.ServiceProxy(name, srv_type)
        except Exception:
            self._append_log(f"Service not available: {name}")
            return None

    def start_launch(self):
        if self.launch_proc and self.launch_proc.poll() is None:
            self._append_log("Gazebo zaten çalışıyor")
            return
        cmd = ["roslaunch", "drl", "hybrid_system.launch"]
        self.launch_proc = self._start_process(cmd, "Gazebo")

    def start_training(self):
        if self.train_proc and self.train_proc.poll() is None:
            self._append_log("Eğitim süreci zaten çalışıyor")
            return
        cmd = ["python3", "drl_agent_node.py"]
        self.train_proc = self._start_process(cmd, "DRL Agent")

    def start_mission(self):
        if self.mission_proc and self.mission_proc.poll() is None:
            self._append_log("Görev yöneticisi zaten çalışıyor")
            return
        cmd = ["python3", "mission_manager.py"]
        self.mission_proc = self._start_process(cmd, "Mission Manager")

    def cmd_takeoff(self):
        if not self.set_mode_srv or not self.arm_srv or not self.takeoff_srv:
            self._append_log("MavROS servisleri hazır değil")
            return
        try:
            mode_req = SetModeRequest(custom_mode="GUIDED")
            mode_resp = self.set_mode_srv(mode_req)
            self._append_log(f"GUIDED modu: {mode_resp.mode_sent}")

            arm_req = CommandBoolRequest(value=True)
            arm_resp = self.arm_srv(arm_req)
            self._append_log(f"Arm sonucu: {arm_resp.success}")

            tol_req = CommandTOLRequest(altitude=2.0, latitude=0.0, longitude=0.0, min_pitch=0.0, yaw=0.0)
            tol_resp = self.takeoff_srv(tol_req)
            self._append_log(f"Kalkış komutu: {tol_resp.success}")
        except Exception as exc:
            self._append_log(f"Kalkış hatası: {exc}")

    def cmd_land(self):
        if not self.set_mode_srv:
            self._append_log("LAND için servis hazır değil")
            return
        try:
            req = SetModeRequest(custom_mode="LAND")
            resp = self.set_mode_srv(req)
            self._append_log(f"LAND komutu gönderildi: {resp.mode_sent}")
        except Exception as exc:
            self._append_log(f"İniş hatası: {exc}")

    def _start_process(self, cmd: List[str], label: str) -> subprocess.Popen:
        self._append_log(f"Başlatılıyor [{label}]: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)
        self.procs.append(proc)
        reader = _ProcessReader(proc, label)
        reader.log_signal.connect(self._append_log, QtCore.Qt.QueuedConnection)
        reader.finished.connect(lambda: self._cleanup_reader(reader))
        self.proc_readers.append(reader)
        reader.start()
        return proc

    def _cleanup_reader(self, reader):
        try:
            self.proc_readers.remove(reader)
        except ValueError:
            pass

    @QtCore.pyqtSlot(np.ndarray)
    def _on_camera_frame(self, cv_img: np.ndarray):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.camera_label.width(), self.camera_label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.camera_label.setPixmap(pix)

    @QtCore.pyqtSlot(object)
    def _on_lidar_scan(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        angles = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        spots = [{"pos": (float(x), float(y))} for x, y in zip(xs, ys)]
        self.scatter.setData(spots)

    @QtCore.pyqtSlot(str)
    def _append_log(self, text: str):
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        try:
            self.ros_worker.stop()
        except Exception:
            pass

        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
        event.accept()


class _ProcessReader(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)

    def __init__(self, proc: subprocess.Popen, label: str):
        super().__init__()
        self.proc = proc
        self.label = label

    def run(self):
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            self.log_signal.emit(f"[{self.label}] {line.rstrip()}")
        try:
            self.proc.stdout.close()
        except Exception:
            pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    dash = DashboardWindow()
    dash.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
