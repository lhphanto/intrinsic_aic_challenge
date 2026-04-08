"""
Gymnasium-compatible environment for the AIC robot arm simulation.

This wraps the ROS2 interfaces into a standard Gymnasium env with:
  - reset(): calls /env/reset service, returns initial observation
  - step(action): sends action to robot, returns (obs, reward, terminated, truncated, info)

The observation and action spaces match the AIC controller interface.

Usage:
    import gymnasium as gym
    from aic_training.aic_gym_env import AICGymEnv

    env = AICGymEnv()
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    JointMotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3, Wrench
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String, Header
from std_srvs.srv import SetBool, Trigger

logger = logging.getLogger(__name__)


# ---- Observation keys (matching aic_robot_aic_controller.py) ----

OBS_KEYS = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
    "tcp_velocity.linear.x",
    "tcp_velocity.linear.y",
    "tcp_velocity.linear.z",
    "tcp_velocity.angular.x",
    "tcp_velocity.angular.y",
    "tcp_velocity.angular.z",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
    "wrench.force.x",
    "wrench.force.y",
    "wrench.force.z",
    "wrench.torque.x",
    "wrench.torque.y",
    "wrench.torque.z",
]

# 6 DOF Cartesian velocity action: [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]
ACTION_KEYS = [
    "linear.x",
    "linear.y",
    "linear.z",
    "angular.x",
    "angular.y",
    "angular.z",
]


@dataclass
class AICGymEnvConfig:
    """Configuration for the AIC Gym environment."""

    # Control mode: "cartesian" or "joint"
    control_mode: str = "cartesian"

    # Frame for Cartesian commands
    frame_id: str = "gripper/tcp"

    # Control frequency (Hz) - how fast step() runs
    control_freq_hz: float = 10.0

    # Max episode steps (0 = unlimited, you handle termination yourself)
    max_episode_steps: int = 0

    # Velocity limits for action space (Cartesian mode)
    max_linear_vel: float = 0.25  # m/s
    max_angular_vel: float = 2.0  # rad/s

    # Velocity limits for action space (Joint mode)
    max_joint_vel: float = 1.0  # rad/s

    # Stiffness and damping defaults for Cartesian mode
    cartesian_stiffness: list[float] = field(
        default_factory=lambda: [85.0, 85.0, 85.0, 85.0, 85.0, 85.0]
    )
    cartesian_damping: list[float] = field(
        default_factory=lambda: [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]
    )

    # Stiffness and damping defaults for Joint mode
    joint_stiffness: list[float] = field(
        default_factory=lambda: [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    )
    joint_damping: list[float] = field(
        default_factory=lambda: [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
    )

    # Whether to include images in observations (adds significant overhead)
    include_images: bool = False

    # Image scaling factor (if include_images=True)
    image_scale: float = 0.25

    # Timeout for reset service call (seconds)
    reset_timeout_sec: float = 60.0

    # Node name
    node_name: str = "aic_gym_env"

    # Whether env.reset() should use a randomly sampled trial config
    random_reset: bool = False


class AICGymEnv(gym.Env):
    """Gymnasium environment wrapping the AIC robot arm simulation."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[AICGymEnvConfig] = None):
        super().__init__()
        self.config = config or AICGymEnvConfig()
        self._step_count = 0
        self._step_period = 1.0 / self.config.control_freq_hz
        self._current_task_info: dict[str, Any] = {}

        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()

        self._node = Node(self.config.node_name)
        self._node.get_logger().info("Initializing AICGymEnv...")

        # --- Define spaces ---
        n_obs = len(OBS_KEYS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float64,
        )

        if self.config.control_mode == "cartesian":
            n_act = 6  # linear xyz + angular xyz
            low = np.array(
                [-self.config.max_linear_vel] * 3
                + [-self.config.max_angular_vel] * 3,
                dtype=np.float64,
            )
            high = -low
        else:
            n_act = 6  # 6 joint velocities
            low = np.full(n_act, -self.config.max_joint_vel, dtype=np.float64)
            high = -low

        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

        # --- ROS2 subscriptions ---
        self._last_controller_state: Optional[ControllerState] = None
        self._last_joint_states: Optional[JointState] = None
        self._last_wrench: Optional[WrenchStamped] = None
        self._fts_tare_offset_force = np.zeros(3)
        self._fts_tare_offset_torque = np.zeros(3)
        self._last_insertion_event: float = 0.0

        self._node.create_subscription(
            ControllerState, "/aic_controller/controller_state",
            self._controller_state_cb, 10,
        )
        self._node.create_subscription(
            JointState, "/joint_states",
            self._joint_states_cb, qos_profile_sensor_data,
        )
        self._node.create_subscription(
            WrenchStamped, "/fts_broadcaster/wrench",
            self._wrench_cb, 10,
        )
        self._node.create_subscription(
            String, "/scoring/insertion_event",
            self._insertion_event_cb, 10,
        )

        # --- ROS2 publishers ---
        self._motion_update_pub = self._node.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10,
        )
        self._joint_motion_update_pub = self._node.create_publisher(
            JointMotionUpdate, "/aic_controller/joint_commands", 10,
        )

        # --- ROS2 service clients ---
        self._change_target_mode_client = self._node.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode",
        )
        self._reset_env_client = self._node.create_client(
            SetBool, "/env/reset",
        )

        # --- Start spinner ---
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Wait for connections
        self._node.get_logger().info("Waiting for controller services...")
        self._change_target_mode_client.wait_for_service()
        self._node.get_logger().info("Controller service available.")

        # Set control mode
        self._set_control_mode()

        # Wait for first observation
        self._node.get_logger().info("Waiting for first observation...")
        t0 = time.monotonic()
        while self._last_controller_state is None or self._last_joint_states is None:
            time.sleep(0.1)
            if time.monotonic() - t0 > 30.0:
                raise TimeoutError("No observation received within 30s")
        self._node.get_logger().info("AICGymEnv ready.")

    # ---- Gymnasium API ----

    def reset(self, *, seed=None, options=None):
        """Reset the environment. Calls the /env/reset service."""
        super().reset(seed=seed, options=options)
        self._step_count = 0
        self._last_insertion_event = 0.0

        # Call reset service (data=True → random reset from config trials)
        if self._reset_env_client.wait_for_service(timeout_sec=5.0):
            req = SetBool.Request()
            req.data = self.config.random_reset
            future = self._reset_env_client.call_async(req)

            # Wait for result
            t0 = time.monotonic()
            while not future.done():
                time.sleep(0.1)
                if time.monotonic() - t0 > self.config.reset_timeout_sec:
                    self._node.get_logger().error("Reset service timed out")
                    break

            if future.done():
                result = future.result()
                if result and not result.success:
                    self._node.get_logger().error(f"Reset failed: {result.message}")
                    self._current_task_info = {}
                else:
                    # response.message contains JSON-encoded task details on success
                    try:
                        self._current_task_info = json.loads(result.message) if result else {}
                    except (json.JSONDecodeError, TypeError):
                        self._current_task_info = {}
                    self._node.get_logger().info(
                        f"Environment reset successful | task_info={self._current_task_info}"
                    )
        else:
            self._node.get_logger().warn(
                "/env/reset service not available. "
                "Is env_resetter.py running in the distrobox?"
            )

        # Wait briefly for new observations after reset
        time.sleep(1.0)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: NDArray[np.float64]):
        """Execute one control step."""
        self._step_count += 1

        # Clip action to bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Send action
        if self.config.control_mode == "cartesian":
            self._send_cartesian_action(action)
        else:
            self._send_joint_action(action)

        # Wait for control period
        time.sleep(self._step_period)

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        # Compute reward (override this in subclass for your task)
        reward = self._compute_reward(obs, action, info)

        # Check termination (override this in subclass for your task)
        terminated = self._check_terminated(obs, info)
        truncated = self._check_truncated(obs, info)

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up ROS2 resources."""
        self._executor.shutdown()
        self._node.destroy_node()

    # ---- Override these for your task ----

    def _compute_reward(self, obs: NDArray, action: NDArray, info: dict) -> float:
        """Compute reward. Override in subclass.

        Default: +1.0 on insertion event, 0.0 otherwise.
        """
        if info.get("insertion_event", 0.0) > 0.5:
            return 1.0
        return 0.0

    def _check_terminated(self, obs: NDArray, info: dict) -> bool:
        """Check if episode should terminate (task success/failure).
        Override in subclass.

        Default: terminate on insertion event.
        """
        return info.get("insertion_event", 0.0) > 0.5

    def _check_truncated(self, obs: NDArray, info: dict) -> bool:
        """Check if episode should be truncated (time limit, safety).
        Override in subclass.

        Default: truncate at max_episode_steps if configured.
        """
        if self.config.max_episode_steps > 0:
            return self._step_count >= self.config.max_episode_steps
        return False

    # ---- Internal ----

    def _controller_state_cb(self, msg: ControllerState):
        self._last_controller_state = msg
        tare = msg.fts_tare_offset.wrench
        self._fts_tare_offset_force = np.array([tare.force.x, tare.force.y, tare.force.z])
        self._fts_tare_offset_torque = np.array([tare.torque.x, tare.torque.y, tare.torque.z])

    def _joint_states_cb(self, msg: JointState):
        self._last_joint_states = msg

    def _wrench_cb(self, msg: WrenchStamped):
        self._last_wrench = msg

    def _insertion_event_cb(self, msg: String):
        self._last_insertion_event = 1.0 if msg.data else 0.0

    def _set_control_mode(self):
        req = ChangeTargetMode.Request()
        req.target_mode.mode = (
            TargetMode.MODE_CARTESIAN
            if self.config.control_mode == "cartesian"
            else TargetMode.MODE_JOINT
        )
        future = self._change_target_mode_client.call_async(req)
        t0 = time.monotonic()
        while not future.done():
            time.sleep(0.05)
            if time.monotonic() - t0 > 5.0:
                self._node.get_logger().error("Failed to set control mode")
                return
        result = future.result()
        if result and not result.success:
            self._node.get_logger().error("Failed to set control mode")
        else:
            self._node.get_logger().info(
                f"Control mode set to {self.config.control_mode}"
            )

    def _get_observation(self) -> NDArray[np.float64]:
        """Build flat observation vector from latest ROS2 messages."""
        obs = np.zeros(len(OBS_KEYS), dtype=np.float64)

        cs = self._last_controller_state
        js = self._last_joint_states
        wr = self._last_wrench

        if cs is None or js is None:
            return obs

        tcp = cs.tcp_pose
        vel = cs.tcp_velocity
        err = cs.tcp_error
        pos = js.position

        i = 0
        # TCP pose (7)
        obs[i] = tcp.position.x; i += 1
        obs[i] = tcp.position.y; i += 1
        obs[i] = tcp.position.z; i += 1
        obs[i] = tcp.orientation.x; i += 1
        obs[i] = tcp.orientation.y; i += 1
        obs[i] = tcp.orientation.z; i += 1
        obs[i] = tcp.orientation.w; i += 1

        # TCP velocity (6)
        obs[i] = vel.linear.x; i += 1
        obs[i] = vel.linear.y; i += 1
        obs[i] = vel.linear.z; i += 1
        obs[i] = vel.angular.x; i += 1
        obs[i] = vel.angular.y; i += 1
        obs[i] = vel.angular.z; i += 1

        # Joint positions (7 - UR5e has 6 arm joints + 1 gripper)
        for j in range(min(7, len(pos))):
            obs[i] = pos[j]; i += 1
        # Pad if fewer joints reported
        i = 13 + 7

        # Wrench (6) - tare-corrected
        if wr is not None:
            obs[i] = wr.wrench.force.x - self._fts_tare_offset_force[0]; i += 1
            obs[i] = wr.wrench.force.y - self._fts_tare_offset_force[1]; i += 1
            obs[i] = wr.wrench.force.z - self._fts_tare_offset_force[2]; i += 1
            obs[i] = wr.wrench.torque.x - self._fts_tare_offset_torque[0]; i += 1
            obs[i] = wr.wrench.torque.y - self._fts_tare_offset_torque[1]; i += 1
            obs[i] = wr.wrench.torque.z - self._fts_tare_offset_torque[2]; i += 1

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Build info dict with additional data not in observation vector."""
        info: dict[str, Any] = {
            "step": self._step_count,
            "insertion_event": self._last_insertion_event,
            "task": self._current_task_info,
        }

        cs = self._last_controller_state
        if cs is not None:
            info["tcp_error"] = list(cs.tcp_error)

        return info

    def _send_cartesian_action(self, action: NDArray):
        """Send a Cartesian velocity command."""
        msg = MotionUpdate()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self.config.frame_id

        msg.velocity = Twist()
        msg.velocity.linear.x = float(action[0])
        msg.velocity.linear.y = float(action[1])
        msg.velocity.linear.z = float(action[2])
        msg.velocity.angular.x = float(action[3])
        msg.velocity.angular.y = float(action[4])
        msg.velocity.angular.z = float(action[5])

        msg.target_stiffness = np.diag(self.config.cartesian_stiffness).flatten()
        msg.target_damping = np.diag(self.config.cartesian_damping).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        self._motion_update_pub.publish(msg)

    def _send_joint_action(self, action: NDArray):
        """Send a joint velocity command."""
        msg = JointMotionUpdate()
        msg.target_state.velocities = action.tolist()
        msg.target_stiffness = self.config.joint_stiffness
        msg.target_damping = self.config.joint_damping
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

        self._joint_motion_update_pub.publish(msg)
