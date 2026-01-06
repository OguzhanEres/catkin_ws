#!/usr/bin/env python3
"""
Hybrid target detection and tracking node for UAV autonomy (SEARCH / TRACK / REDETECT).
- Detects with YOLOv5, tracks with OpenCV KCF/CSRT.
- Publishes bbox info on /target/bbox: [cx_norm, cy_norm, distance_m, width_norm, height_norm].
"""
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


STATE_SEARCH = "SEARCH"
STATE_TRACK = "TRACK"
STATE_REDETECT = "REDETECT"


class HybridTrackerNode:
    def __init__(self):
        rospy.init_node("hybrid_tracker", anonymous=False)
        self.bridge = CvBridge()

        # Initialize frame_lock BEFORE subscriber to prevent race condition
        self.frame_lock = threading.Lock()
        self.last_frame = None

        self.det_conf = rospy.get_param("~det_conf", 0.35)
        self.det_model_path = rospy.get_param("~det_model_path", "yolov8n.pt")
        self.tracker_type = rospy.get_param("~tracker_type", "KCF")  # Options: KCF, CSRT
        self.focal_px = rospy.get_param("~focal_px", 525.0)
        self.target_real_height = rospy.get_param("~target_real_height", 0.3)
        self.redetect_score_drop = rospy.get_param("~redetect_score_drop", 0.25)
        self.redetect_period = rospy.get_param("~redetect_period", 1.0)
        self.device = rospy.get_param("~device", "cpu")  # 'cpu' or 'cuda'

        self.detector = self._load_detector(self.det_model_path)
        self.tracker = None
        self.state = STATE_SEARCH
        self.last_det_time = rospy.Time.now()
        self.last_det_score = 0.0

        # Create subscriber AFTER all attributes are initialized
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self._image_cb, queue_size=1)
        self.target_pub = rospy.Publisher("/target/bbox", Float32MultiArray, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration.from_sec(0.05), self._process_frame)  # 20 Hz

    def _load_detector(self, model_path: str):
        try:
            rospy.loginfo(f"Loading YOLOv5 detector from {model_path} on {self.device}...")
            model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
            model.conf = self.det_conf
            model.to(self.device)
            if self.device == 'cuda':
                model.half()  # Use half precision for GPU
            return model
        except Exception as exc:  # pragma: no cover - runtime dependency
            rospy.logerr(f"Failed to load detector: {exc}")
            return None

    def _image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:  # pragma: no cover
            rospy.logwarn(f"CV bridge error: {exc}")
            return
        with self.frame_lock:
            self.last_frame = frame

    def _process_frame(self, _event):
        with self.frame_lock:
            frame = None if self.last_frame is None else self.last_frame.copy()
        if frame is None:
            return

        bbox, score = self._run_state_machine(frame)
        if bbox is None:
            return

        cx, cy, w, h = self._bbox_center(bbox)
        img_h, img_w = frame.shape[:2]
        cx_norm = cx / float(img_w)
        cy_norm = cy / float(img_h)
        w_norm = w / float(img_w)
        h_norm = h / float(img_h)
        distance = self._estimate_distance(h)

        msg = Float32MultiArray()
        msg.data = [cx_norm, cy_norm, distance, w_norm, h_norm]
        self.target_pub.publish(msg)

    def _run_state_machine(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        if self.state == STATE_SEARCH:
            bbox, score = self._detect(frame)
            if bbox is not None:
                self._init_tracker(frame, bbox)
                self.last_det_time = rospy.Time.now()
                self.last_det_score = score
                self.state = STATE_TRACK
                rospy.loginfo("State -> TRACK")
            return bbox, score

        if self.state == STATE_TRACK:
            if self.last_det_score < self.redetect_score_drop:
                rospy.loginfo("Score low; forcing REDETECT")
            else:
                bbox, score = self._update_tracker(frame)
                if bbox is not None:
                    return bbox, score
            self.state = STATE_REDETECT
            rospy.loginfo("Tracker lost target; State -> REDETECT")

        if self.state == STATE_REDETECT:
            bbox, score = self._detect(frame)
            if bbox is not None:
                self._init_tracker(frame, bbox)
                self.last_det_time = rospy.Time.now()
                self.last_det_score = score
                self.state = STATE_TRACK
                rospy.loginfo("State -> TRACK")
                return bbox, score
            self.state = STATE_SEARCH
            rospy.loginfo("State -> SEARCH")
            return None, 0.0

        self.state = STATE_SEARCH
        return None, 0.0

    def _update_tracker(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        force_redetect = (rospy.Time.now() - self.last_det_time).to_sec() > self.redetect_period
        if self.tracker is None:
            return None, 0.0
        ok, bbox = self.tracker.update(frame)
        if ok and not force_redetect:
            return tuple(map(int, bbox)), self.last_det_score
        return None, 0.0

    def _detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        if self.detector is None:
            rospy.logwarn_once("Detector unavailable; skipping detections")
            return None, 0.0
        # TODO: replace with YOLOv8 inference; this is a placeholder using torch hub output format.
        results = self.detector(frame)
        if results is None:
            return None, 0.0
        try:
            df = results.pandas().xyxy[0]
        except Exception:
            return None, 0.0
        if df.empty:
            return None, 0.0
        row = df.iloc[0]
        bbox = (int(row["xmin"]), int(row["ymin"]), int(row["xmax"] - row["xmin"]), int(row["ymax"] - row["ymin"]))
        score = float(row["confidence"])
        return bbox, score

    def _init_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        tracker_ctor = cv2.TrackerKCF_create if self.tracker_type.upper() == "KCF" else cv2.TrackerCSRT_create
        self.tracker = tracker_ctor()
        self.tracker.init(frame, bbox)

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0, w, h

    def _estimate_distance(self, bbox_height_px: float) -> float:
        if bbox_height_px <= 1e-3:
            return float("inf")
        return (self.target_real_height * self.focal_px) / bbox_height_px


if __name__ == "__main__":
    try:
        node = HybridTrackerNode()
        rospy.loginfo("Hybrid tracker node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
