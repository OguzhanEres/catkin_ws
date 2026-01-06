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
        self.root.title("PX4 City Sim Control")
        self.root.geometry("720x640")

        self.processes = {}
        self.log_queue = queue.Queue()
        self.status_items = {}
        self._status_cache = {}
        self._status_refresh_running = False

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
        self.px4_wipe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_frame, text="PX4 Paramlari Sifirla (wipe)", variable=self.px4_wipe_var).pack(fill=tk.X, pady=3)

        self._add_button(btn_frame, "Gazebo Baslat", self._start_gazebo)
        self._add_button(btn_frame, "Gazebo Durdur", lambda: self._stop_process("gazebo"))
        self._add_button(btn_frame, "PX4 Baslat", self._start_px4)
        self._add_button(btn_frame, "PX4 Durdur", self._stop_px4)
        self._add_button(btn_frame, "Baglanti (Connect)", self._start_connect)
        self._add_button(btn_frame, "Baglanti Durdur", lambda: self._stop_process("connect"))

        epoch_row = ttk.Frame(btn_frame)
        epoch_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(epoch_row, text="Epoch Sayisi:").pack(side=tk.LEFT)
        self.epoch_entry = ttk.Entry(epoch_row, width=8)
        self.epoch_entry.insert(0, "10")
        self.epoch_entry.pack(side=tk.LEFT, padx=6)

        self._add_button(btn_frame, "Training Baslat - Exploration", self._start_training_exploration)
        self._add_button(btn_frame, "Training Baslat - Tracking", self._start_training_tracking)
        self._add_button(btn_frame, "Training Durdur", self._stop_training)

        status_frame = ttk.LabelFrame(main, text="Status Checks")
        status_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))

        status_top = ttk.Frame(status_frame)
        status_top.pack(fill=tk.X)
        self.status_time_var = tk.StringVar(value="Last refresh: never")
        ttk.Button(status_top, text="Refresh Status", command=self._refresh_status_async).pack(side=tk.LEFT)
        ttk.Label(status_top, textvariable=self.status_time_var).pack(side=tk.LEFT, padx=10)

        self.status_tree = ttk.Treeview(
            status_frame,
            columns=("status", "detail"),
            show="tree headings",
            height=10,
        )
        self.status_tree.heading("#0", text="Check")
        self.status_tree.heading("status", text="Status")
        self.status_tree.heading("detail", text="Detail")
        self.status_tree.column("#0", width=220, anchor=tk.W)
        self.status_tree.column("status", width=80, anchor=tk.W)
        self.status_tree.column("detail", width=360, anchor=tk.W)
        self.status_tree.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.status_tree.tag_configure("ok", background="#e8f5e9")
        self.status_tree.tag_configure("warn", background="#fff3e0")
        self.status_tree.tag_configure("fail", background="#ffebee")

        self.log = ScrolledText(main, height=12, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self._init_status_items()
        self._refresh_status_async()

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

    def _init_status_items(self):
        self.status_checks = [
            ("ros_master", "ROS master", self._check_ros_master),
            ("ros_pkg_path", "ROS_PACKAGE_PATH", self._check_ros_package_path),
            ("node_gazebo", "Node /gazebo", lambda: self._check_node("/gazebo")),
            ("node_mavros", "Node /mavros", lambda: self._check_node("/mavros")),
            ("svc_set_mode", "Service /mavros/set_mode", lambda: self._check_service("/mavros/set_mode")),
            ("svc_set_state", "Service /gazebo/set_model_state", lambda: self._check_service("/gazebo/set_model_state")),
            ("topic_clock", "Topic /clock", lambda: self._check_topic_msg("/clock")),
            ("topic_model_states", "Topic /gazebo/model_states", lambda: self._check_topic_msg("/gazebo/model_states")),
            ("topic_local_pose", "Topic /mavros/local_position/pose", lambda: self._check_topic_msg("/mavros/local_position/pose")),
            ("topic_gps_raw", "Topic /mavros/global_position/raw/fix", lambda: self._check_topic_msg("/mavros/global_position/raw/fix")),
            ("topic_agent_gps", "Topic /agent/gps_vec", lambda: self._check_topic_msg("/agent/gps_vec")),
            ("topic_route_raw", "Type /agent/route_raw", lambda: self._check_topic_type("/agent/route_raw", "nav_msgs/Path")),
            ("topic_route_smoothed", "Type /agent/route_smoothed", lambda: self._check_topic_type("/agent/route_smoothed", "nav_msgs/Path")),
            ("param_model_name", "Param /ppo_lstm_trainer/model_name", lambda: self._check_param("/ppo_lstm_trainer/model_name")),
            ("param_gps_topic", "Param /sensor_vectorizer/gps_topic", lambda: self._check_param("/sensor_vectorizer/gps_topic")),
            ("proc_px4", "Process px4", lambda: self._check_process("px4")),
            ("proc_gzserver", "Process gzserver", lambda: self._check_process("gzserver")),
        ]
        for key, label, _func in self.status_checks:
            item = self.status_tree.insert("", tk.END, text=label, values=("...", ""), tags=("warn",))
            self.status_items[key] = item

    def _refresh_status_async(self):
        if self._status_refresh_running:
            return
        self._status_refresh_running = True
        threading.Thread(target=self._refresh_status, daemon=True).start()

    def _refresh_status(self):
        self._status_cache = {}
        results = []
        for key, _label, func in self.status_checks:
            status, detail = func()
            results.append((key, status, detail))

        def _apply():
            for key, status, detail in results:
                tag = "ok" if status == "OK" else "warn" if status == "WARN" else "fail"
                self.status_tree.item(self.status_items[key], values=(status, detail), tags=(tag,))
            self.status_time_var.set("Last refresh: " + time.strftime("%H:%M:%S"))
            self._status_refresh_running = False

        self.root.after(0, _apply)

    def _run_cmd(self, cmd, timeout=2.0):
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            return False, detail if detail else "error"
        return True, result.stdout.strip()

    def _get_nodes(self):
        if "nodes" in self._status_cache:
            return self._status_cache["nodes"]
        ok, out = self._run_cmd(["rosnode", "list"])
        nodes = out.splitlines() if ok and out else []
        self._status_cache["nodes"] = (ok, nodes, out)
        return self._status_cache["nodes"]

    def _get_services(self):
        if "services" in self._status_cache:
            return self._status_cache["services"]
        ok, out = self._run_cmd(["rosservice", "list"])
        services = out.splitlines() if ok and out else []
        self._status_cache["services"] = (ok, services, out)
        return self._status_cache["services"]

    def _check_ros_master(self):
        ok, nodes, detail = self._get_nodes()
        if not ok:
            return "FAIL", detail
        return "OK", f"{len(nodes)} nodes"

    def _check_ros_package_path(self):
        path = os.environ.get("ROS_PACKAGE_PATH", "")
        if "catkin_ws" in path:
            return "OK", "workspace sourced"
        return "WARN", path if path else "empty"

    def _check_node(self, name: str):
        ok, nodes, detail = self._get_nodes()
        if not ok:
            return "FAIL", detail
        return ("OK", "running") if name in nodes else ("WARN", "missing")

    def _check_service(self, name: str):
        ok, services, detail = self._get_services()
        if not ok:
            return "FAIL", detail
        return ("OK", "available") if name in services else ("WARN", "missing")

    def _check_topic_msg(self, topic: str):
        ok, detail = self._run_cmd(["rostopic", "echo", "-n", "1", topic], timeout=2.5)
        return ("OK", "message received") if ok else ("WARN", detail)

    def _check_topic_type(self, topic: str, expected: str):
        ok, out = self._run_cmd(["rostopic", "info", topic], timeout=2.0)
        if not ok:
            return "WARN", out
        msg_type = ""
        for line in out.splitlines():
            if line.startswith("Type:"):
                msg_type = line.split(":", 1)[1].strip()
                break
        if not msg_type:
            return "WARN", "type not found"
        if msg_type == expected:
            return "OK", msg_type
        return "WARN", msg_type

    def _check_param(self, name: str):
        ok, out = self._run_cmd(["rosparam", "get", name], timeout=2.0)
        return ("OK", out) if ok else ("WARN", out)

    def _check_process(self, token: str):
        ok, out = self._run_cmd(["pgrep", "-f", token], timeout=1.5)
        if ok and out:
            return "OK", "running"
        return "WARN", "not found"

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
        cmd = ["roslaunch", "ardupilot_city_sim", "city_sim_gazebo.launch"]
        if not self.gazebo_gui_var.get():
            cmd.append("gui:=false")
        self._start_process("gazebo", cmd)

    def _start_px4(self):
        running, detail = self._is_px4_running()
        if running:
            self._log("PX4 already running. Stop it first (PX4 Durdur).")
            if detail:
                self._log(detail)
            return
        wipe_cmd = ""
        if self.px4_wipe_var.get():
            wipe_cmd = "rm -f $HOME/PX4-Autopilot/eeprom/parameters_* && "
        cmd = [
            "bash",
            "-lc",
            wipe_cmd
            + "cd $HOME/PX4-Autopilot && "
            + "export PX4_SIM_MODEL=iris && export PX4_SIM_HOST_ADDR=127.0.0.1 && "
            + "./build/px4_sitl_default/bin/px4 ./build/px4_sitl_default/etc "
            + "-s etc/init.d-posix/rcS -t ./test_data",
        ]
        self._start_process("px4", cmd)

    def _stop_px4(self):
        self._stop_process("px4")
        running, detail = self._is_px4_running()
        if running and detail:
            self._log("PX4 still running; killing external PX4 process.")
            self._log(detail)
            subprocess.run(["pkill", "-f", "px4_sitl_default/bin/px4"], check=False)
            time.sleep(0.5)
        running, detail = self._is_px4_running()
        if running:
            self._log("PX4 still running; close the other terminal or kill it manually.")
            if detail:
                self._log(detail)
        else:
            self._log("PX4 stopped.")

    def _is_px4_running(self):
        result = subprocess.run(
            ["pgrep", "-a", "-f", "px4_sitl_default/bin/px4"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
        return False, ""

    def _start_connect(self):
        self._start_process("connect", ["roslaunch", "ardupilot_city_sim", "mavros_connect.launch"])

    def _start_training_exploration(self):
        self._start_training("exploration")

    def _start_training_tracking(self):
        self._start_training("tracking")

    def _start_training(self, mode: str):
        try:
            epochs = int(self.epoch_entry.get().strip())
        except ValueError:
            epochs = 10
        self._start_process(
            f"training_{mode}",
            ["roslaunch", "drl", "drl_train.launch", f"mode:={mode}", f"epochs:={epochs}"],
        )

    def _stop_training(self):
        stopped = False
        for key, proc in list(self.processes.items()):
            if not key.startswith("training_"):
                continue
            if proc and proc.poll() is None:
                self._stop_process(key)
                stopped = True
        if not stopped:
            self.log_queue.put("No running training process found.")

    def _on_close(self):
        for key, proc in list(self.processes.items()):
            if proc and proc.poll() is None:
                self._stop_process(key)
        self.root.destroy()


def main():
    root = tk.Tk()
    gui = SimControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
