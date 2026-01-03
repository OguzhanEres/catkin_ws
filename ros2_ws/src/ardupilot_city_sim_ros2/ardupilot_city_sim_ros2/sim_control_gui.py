#!/usr/bin/env python3

import os
import queue
import signal
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


class SimControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ArduPilot City Sim Control (ROS2)")
        self.root.geometry("520x480")

        self.processes = {}
        self.log_queue = queue.Queue()

        self._build_ui()
        self._poll_log_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)

        self.gazebo_gui_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_frame, text="Gazebo GUI", variable=self.gazebo_gui_var).pack(fill=tk.X, pady=3)
        self.ardupilot_wipe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_frame, text="Paramlari Sifirla (wipe)", variable=self.ardupilot_wipe_var).pack(fill=tk.X, pady=3)

        self._add_button(btn_frame, "Gazebo Baslat", self._start_gazebo)
        self._add_button(btn_frame, "Gazebo Durdur", lambda: self._stop_process("gazebo"))
        self._add_button(btn_frame, "ArduPilot Baslat", self._start_ardupilot)
        self._add_button(btn_frame, "ArduPilot Durdur", lambda: self._stop_process("ardupilot"))
        self._add_button(btn_frame, "Training Baslat (ROS1)", self._start_training_ros1)
        self._add_button(btn_frame, "Training Durdur", self._stop_training)

        self.log = ScrolledText(main, height=12, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _add_button(self, parent, text, command):
        btn = ttk.Button(parent, text=text, command=command)
        btn.pack(fill=tk.X, pady=3)

    def _log(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _poll_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _start_process(self, key, cmd):
        proc = self.processes.get(key)
        if proc and proc.poll() is None:
            self._log(f"{key} already running.")
            return

        self._log(f"Starting {key}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        self.processes[key] = proc
        threading.Thread(target=self._read_output, args=(key, proc), daemon=True).start()

    def _stop_process(self, key: str):
        proc = self.processes.get(key)
        if not proc or proc.poll() is not None:
            self.log_queue.put(f"[{key}] not running.")
            return

        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pass

        def _signal(sig, label):
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)
                else:
                    proc.send_signal(sig)
                self.log_queue.put(f"[{key}] {label} sent.")
            except OSError:
                self.log_queue.put(f"[{key}] {label} failed.")

        _signal(signal.SIGINT, "SIGINT")
        deadline = time.time() + 6.0
        while time.time() < deadline:
            if proc.poll() is not None:
                self.log_queue.put(f"[{key}] stopped.")
                return
            time.sleep(0.2)

        _signal(signal.SIGTERM, "SIGTERM")
        deadline = time.time() + 4.0
        while time.time() < deadline:
            if proc.poll() is not None:
                self.log_queue.put(f"[{key}] stopped.")
                return
            time.sleep(0.2)

        _signal(signal.SIGKILL, "SIGKILL")
        self.log_queue.put(f"[{key}] forced kill.")

    def _read_output(self, key, proc):
        if not proc.stdout:
            return
        for line in proc.stdout:
            self.log_queue.put(f"[{key}] {line.rstrip()}")
        self.log_queue.put(f"[{key}] process exited.")

    def _start_gazebo(self):
        cmd = ["ros2", "launch", "ardupilot_city_sim_ros2", "city_sim_gazebo.launch.py"]
        if not self.gazebo_gui_var.get():
            cmd.append("gui:=false")
        self._start_process("gazebo", cmd)

    def _start_ardupilot(self):
        param_file = os.path.expanduser("~/catkin_ws/src/ardupilot_city_sim/config/ardupilot_override.parm")
        wipe_arg = " -w" if self.ardupilot_wipe_var.get() else ""
        cmd = [
            "bash",
            "-lc",
            f"cd $HOME/ardupilot/ArduCopter && ../Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris -I0 --out=127.0.0.1:14550 --add-param-file={param_file}{wipe_arg}",
        ]
        self._start_process("ardupilot", cmd)

    def _start_training_ros1(self):
        self._start_process("training_ros1", ["roslaunch", "drl", "drl_train.launch"])

    def _stop_training(self):
        for key in ["training_ros1"]:
            if key in self.processes:
                self._stop_process(key)

    def _on_close(self):
        for key, proc in list(self.processes.items()):
            if proc and proc.poll() is None:
                self._stop_process(key)
        self.root.destroy()


def main():
    root = tk.Tk()
    _ = SimControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

