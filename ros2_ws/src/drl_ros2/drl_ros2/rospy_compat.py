"""
Very small rospy-like compatibility shim for ROS 2 (rclpy).

It is intentionally minimal: enough to run the ported scripts with small edits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

_NODE: Optional[Node] = None
_THROTTLE_LAST: Dict[str, float] = {}


def _require_node() -> Node:
    if _NODE is None:
        raise RuntimeError("rospy_compat node not initialized; call init_node() first")
    return _NODE


def init_node(name: str, anonymous: bool = False) -> None:
    del anonymous
    global _NODE
    if not rclpy.ok():
        rclpy.init()
    if _NODE is None:
        _NODE = rclpy.create_node(name)


def get_param(name: str, default: Any = None) -> Any:
    node = _require_node()
    param_name = name[1:] if name.startswith("~") else name.lstrip("/")
    if not node.has_parameter(param_name):
        node.declare_parameter(param_name, default)
    return node.get_parameter(param_name).value


def sleep(seconds: float) -> None:
    time.sleep(max(0.0, float(seconds)))


def is_shutdown() -> bool:
    return not rclpy.ok()


class ROSInterruptException(Exception):
    pass


def spin() -> None:
    rclpy.spin(_require_node())


def shutdown() -> None:
    global _NODE
    if _NODE is not None:
        _NODE.destroy_node()
        _NODE = None
    if rclpy.ok():
        rclpy.shutdown()


def loginfo(msg: str, *args: Any) -> None:
    _require_node().get_logger().info(msg % args if args else msg)


def logwarn(msg: str, *args: Any) -> None:
    _require_node().get_logger().warning(msg % args if args else msg)


def _throttle_key(msg: str, args: Any) -> str:
    return f"{msg}|{args!r}"


def logwarn_throttle(period_s: float, msg: str, *args: Any) -> None:
    now = time.time()
    key = _throttle_key(msg, args)
    last = _THROTTLE_LAST.get(key, 0.0)
    if now - last >= float(period_s):
        _THROTTLE_LAST[key] = now
        logwarn(msg, *args)


def loginfo_throttle(period_s: float, msg: str, *args: Any) -> None:
    now = time.time()
    key = _throttle_key(msg, args)
    last = _THROTTLE_LAST.get(key, 0.0)
    if now - last >= float(period_s):
        _THROTTLE_LAST[key] = now
        loginfo(msg, *args)


@dataclass(frozen=True)
class Duration:
    _duration: RclpyDuration

    @staticmethod
    def from_sec(seconds: float) -> "Duration":
        return Duration(RclpyDuration(seconds=float(seconds)))

    def to_sec(self) -> float:
        return self._duration.nanoseconds / 1e9


class Time:
    __slots__ = ("_ns",)

    def __init__(self, secs: int = 0, nsecs: int = 0):
        self._ns = int(secs) * 1_000_000_000 + int(nsecs)

    @staticmethod
    def now() -> "Time":
        return Time.from_nanoseconds(int(_require_node().get_clock().now().nanoseconds))

    @staticmethod
    def from_nanoseconds(ns: int) -> "Time":
        t = Time()
        t._ns = int(ns)
        return t

    @staticmethod
    def from_msg(stamp: TimeMsg) -> "Time":
        return Time(int(stamp.sec), int(stamp.nanosec))

    def to_msg(self) -> TimeMsg:
        msg = TimeMsg()
        msg.sec = int(self._ns // 1_000_000_000)
        msg.nanosec = int(self._ns % 1_000_000_000)
        return msg

    def __sub__(self, other: "Time") -> Duration:
        return Duration.from_sec((self._ns - other._ns) / 1e9)

    def __add__(self, other: Duration) -> "Time":
        return Time.from_nanoseconds(self._ns + int(other._duration.nanoseconds))

    def __le__(self, other: "Time") -> bool:
        return self._ns <= other._ns

    def __lt__(self, other: "Time") -> bool:
        return self._ns < other._ns

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Time) and self._ns == other._ns


class Rate:
    def __init__(self, hz: float):
        self._period = 1.0 / max(1e-6, float(hz))

    def sleep(self) -> None:
        time.sleep(self._period)


class Publisher:
    def __init__(self, topic: str, msg_type: Any, queue_size: int = 1, latch: bool = False):
        node = _require_node()
        qos = QoSProfile(depth=int(queue_size))
        if latch:
            qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            qos.reliability = ReliabilityPolicy.RELIABLE
        self._pub = node.create_publisher(msg_type, topic, qos)

    def publish(self, msg: Any) -> None:
        self._pub.publish(msg)


class Subscriber:
    def __init__(self, topic: str, msg_type: Any, callback: Callable[[Any], None], queue_size: int = 1):
        node = _require_node()
        self._sub = node.create_subscription(msg_type, topic, callback, int(queue_size))


class Timer:
    def __init__(self, duration: Duration, callback: Callable[[Any], None]):
        node = _require_node()
        self._timer = node.create_timer(float(duration.to_sec()), lambda: callback(None))


class ServiceException(Exception):
    pass


class _ServiceProxy:
    def __init__(self, name: str, srv_type: Type[Any]):
        node = _require_node()
        self._client = node.create_client(srv_type, name)
        self._srv_type = srv_type
        self._name = name
        self._client.wait_for_service(timeout_sec=10.0)

    def __call__(self, **kwargs: Any) -> Any:
        req = self._srv_type.Request()
        for k, v in kwargs.items():
            setattr(req, k, v)
        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(_require_node(), future, timeout_sec=10.0)
        if future.result() is None:
            raise ServiceException(f"Service call failed: {self._name}")
        return future.result()


def wait_for_service(_name: str) -> None:
    return


def ServiceProxy(name: str, srv_type: Type[Any]) -> _ServiceProxy:
    return _ServiceProxy(name, srv_type)

