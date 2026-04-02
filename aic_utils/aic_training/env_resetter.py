#!/usr/bin/env python3
"""
Environment reset node for RL training.

This node runs INSIDE the distrobox/docker container where simulation_interfaces
is available. It provides a /env/reset service (std_srvs/Trigger) that performs
the full reset sequence:

1. Delete cable entity from Gazebo
2. Deactivate aic_controller
3. Reset robot joints to home positions
4. Reactivate aic_controller
5. Tare force/torque sensor
6. Re-spawn cable at initial pose (relative to current gripper position)
7. Wait for robot joints to stabilize

Usage (inside distrobox):
    source /ws_aic/install/setup.bash
    python3 env_resetter.py [--ros-args -p use_sim_time:=true]
"""

import math
import subprocess
import sys
import time
from threading import Event

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from aic_engine_interfaces.srv import ResetJoints
from controller_manager_msgs.srv import SwitchController
from simulation_interfaces.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import tf2_ros


# ---- Default Configuration ----

# Home joint positions (from ur_gz.urdf.xacro initial_positions)
HOME_JOINT_POSITIONS = {
    "shoulder_pan_joint": -0.1597,
    "shoulder_lift_joint": -1.3542,
    "elbow_joint": -1.6648,
    "wrist_1_joint": -1.6933,
    "wrist_2_joint": 1.5710,
    "wrist_3_joint": 1.4110,
}

# Cable spawn config: offset relative to gripper TCP
CABLE_GRIPPER_OFFSET = {"x": 0.0, "y": 0.015385, "z": 0.04245}
CABLE_ORIENTATION = {"roll": 0.4432, "pitch": -0.4838, "yaw": 1.3303}
CABLE_TYPE = "sfp_sc_cable"
ATTACH_CABLE_TO_GRIPPER = True

# Entity names managed by the resetter
CABLE_ENTITY_NAME = "cable_0"

# Timeouts
SERVICE_TIMEOUT_SEC = 10.0
STABILIZE_TIMEOUT_SEC = 10.0
JOINT_VELOCITY_THRESHOLD = 1e-3


def rpy_to_quaternion(roll: float, pitch: float, yaw: float):
    """Convert roll/pitch/yaw to quaternion (x, y, z, w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def generate_cable_sdf(cable_type: str, attach_to_gripper: bool) -> str:
    """Run xacro to generate cable SDF string."""
    try:
        # Find the aic_description share directory
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "aic_description", "--share"],
            capture_output=True, text=True, check=True,
        )
        aic_desc_share = result.stdout.strip()
        xacro_file = f"{aic_desc_share}/urdf/cable.sdf.xacro"

        cmd = [
            "xacro", xacro_file,
            f"attach_cable_to_gripper:={'true' if attach_to_gripper else 'false'}",
            f"cable_type:={cable_type}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"xacro failed: {e.stderr}") from e


class EnvResetter(Node):
    """ROS2 node that resets the simulation environment for RL training."""

    def __init__(self):
        super().__init__("env_resetter")

        # Declare parameters
        self.declare_parameter("cable_type", CABLE_TYPE)
        self.declare_parameter("attach_cable_to_gripper", ATTACH_CABLE_TO_GRIPPER)
        self.declare_parameter("cable_entity_name", CABLE_ENTITY_NAME)
        self.declare_parameter("cable_offset_x", CABLE_GRIPPER_OFFSET["x"])
        self.declare_parameter("cable_offset_y", CABLE_GRIPPER_OFFSET["y"])
        self.declare_parameter("cable_offset_z", CABLE_GRIPPER_OFFSET["z"])
        self.declare_parameter("cable_roll", CABLE_ORIENTATION["roll"])
        self.declare_parameter("cable_pitch", CABLE_ORIENTATION["pitch"])
        self.declare_parameter("cable_yaw", CABLE_ORIENTATION["yaw"])
        self.declare_parameter("delete_task_board", False)
        self.declare_parameter("gripper_frame", "gripper/tcp")

        self._cb_group = ReentrantCallbackGroup()

        # Service clients
        self._reset_joints_client = self.create_client(
            ResetJoints, "/scoring/reset_joints", callback_group=self._cb_group,
        )
        self._switch_controller_client = self.create_client(
            SwitchController, "/controller_manager/switch_controller",
            callback_group=self._cb_group,
        )
        self._spawn_entity_client = self.create_client(
            SpawnEntity, "/gz_server/spawn_entity", callback_group=self._cb_group,
        )
        self._delete_entity_client = self.create_client(
            DeleteEntity, "/gz_server/delete_entity", callback_group=self._cb_group,
        )
        self._tare_ft_client = self.create_client(
            Trigger, "/aic_controller/tare_force_torque_sensor",
            callback_group=self._cb_group,
        )

        # TF for gripper pose lookup
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Joint state subscriber (for stabilization check)
        self._joints_stable = Event()
        self._joint_states_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, 10,
        )

        # Pre-generate cable SDF
        cable_type = self.get_parameter("cable_type").value
        attach = self.get_parameter("attach_cable_to_gripper").value
        self.get_logger().info(f"Generating cable SDF (type={cable_type}, attach={attach})...")
        self._cable_sdf = generate_cable_sdf(cable_type, attach)
        self.get_logger().info(f"Cable SDF generated ({len(self._cable_sdf)} bytes)")

        # Reset service
        self._reset_service = self.create_service(
            Trigger, "/env/reset", self._reset_callback,
            callback_group=self._cb_group,
        )

        # Track whether this is the first reset (skip delete on first call
        # if entities were spawned by the launch file)
        self._first_reset = True

        self.get_logger().info("EnvResetter ready. Call /env/reset to reset environment.")

    def _joint_states_cb(self, msg: JointState):
        if not msg.velocity:
            return
        if all(abs(v) < JOINT_VELOCITY_THRESHOLD for v in msg.velocity):
            self._joints_stable.set()

    def _wait_for_service(self, client, name: str) -> bool:
        if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_SEC):
            self.get_logger().error(f"Service {name} not available")
            return False
        return True

    def _call_service(self, client, request, name: str):
        """Synchronous service call with timeout."""
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_SEC)
        if not future.done():
            self.get_logger().error(f"Service {name} timed out")
            return None
        return future.result()

    def _delete_entity(self, entity_name: str) -> bool:
        if not self._wait_for_service(self._delete_entity_client, "delete_entity"):
            return False
        req = DeleteEntity.Request()
        req.entity = entity_name
        resp = self._call_service(self._delete_entity_client, req, "delete_entity")
        if resp is None:
            return False
        # Result code 1 = OK in simulation_interfaces
        if resp.result.result != 1:
            self.get_logger().warn(
                f"Delete entity '{entity_name}' returned: {resp.result.error_message}"
            )
            return False
        self.get_logger().info(f"Deleted entity '{entity_name}'")
        return True

    def _switch_controllers(self, activate: list, deactivate: list) -> bool:
        if not self._wait_for_service(
            self._switch_controller_client, "switch_controller"
        ):
            return False
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.BEST_EFFORT
        resp = self._call_service(
            self._switch_controller_client, req, "switch_controller"
        )
        if resp is None or not resp.ok:
            self.get_logger().error("Failed to switch controllers")
            return False
        return True

    def _reset_joints(self) -> bool:
        if not self._wait_for_service(self._reset_joints_client, "reset_joints"):
            return False
        req = ResetJoints.Request()
        req.joint_names = list(HOME_JOINT_POSITIONS.keys())
        req.initial_positions = list(HOME_JOINT_POSITIONS.values())
        resp = self._call_service(self._reset_joints_client, req, "reset_joints")
        if resp is None or not resp.success:
            self.get_logger().error(f"Failed to reset joints: {getattr(resp, 'message', 'timeout')}")
            return False
        self.get_logger().info("Joints reset to home positions")
        return True

    def _tare_ft_sensor(self) -> bool:
        if not self._wait_for_service(self._tare_ft_client, "tare_ft"):
            return False
        req = Trigger.Request()
        resp = self._call_service(self._tare_ft_client, req, "tare_ft")
        if resp is None or not resp.success:
            self.get_logger().error("Failed to tare F/T sensor")
            return False
        self.get_logger().info("F/T sensor tared")
        return True

    def _get_gripper_pose(self):
        """Get current gripper TCP pose in world frame."""
        gripper_frame = self.get_parameter("gripper_frame").value
        try:
            t = self._tf_buffer.lookup_transform("world", gripper_frame, rclpy.time.Time())
            return t.transform.translation
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def _spawn_cable(self) -> bool:
        if not self._wait_for_service(self._spawn_entity_client, "spawn_entity"):
            return False

        # Get gripper position for cable placement
        gripper_pos = self._get_gripper_pose()
        if gripper_pos is None:
            return False

        offset_x = self.get_parameter("cable_offset_x").value
        offset_y = self.get_parameter("cable_offset_y").value
        offset_z = self.get_parameter("cable_offset_z").value
        roll = self.get_parameter("cable_roll").value
        pitch = self.get_parameter("cable_pitch").value
        yaw = self.get_parameter("cable_yaw").value
        entity_name = self.get_parameter("cable_entity_name").value

        qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)

        req = SpawnEntity.Request()
        req.name = entity_name
        req.allow_renaming = True
        req.resource_string = self._cable_sdf
        req.entity_namespace = ""
        req.initial_pose.header.frame_id = "world"
        req.initial_pose.pose.position.x = gripper_pos.x + offset_x
        req.initial_pose.pose.position.y = gripper_pos.y + offset_y
        req.initial_pose.pose.position.z = gripper_pos.z + offset_z
        req.initial_pose.pose.orientation.x = qx
        req.initial_pose.pose.orientation.y = qy
        req.initial_pose.pose.orientation.z = qz
        req.initial_pose.pose.orientation.w = qw

        resp = self._call_service(self._spawn_entity_client, req, "spawn_entity")
        if resp is None or resp.result.result != 1:
            err = getattr(resp, "result", None)
            self.get_logger().error(
                f"Failed to spawn cable: {err.error_message if err else 'timeout'}"
            )
            return False

        self.get_logger().info(f"Cable spawned as '{resp.entity_name}'")
        return True

    def _wait_for_stabilization(self) -> bool:
        """Wait for robot joints to settle (velocities near zero)."""
        self._joints_stable.clear()
        settled = self._joints_stable.wait(timeout=STABILIZE_TIMEOUT_SEC)
        if not settled:
            self.get_logger().warn("Robot did not stabilize within timeout")
        return settled

    def _reset_callback(self, request, response):
        """Full environment reset sequence."""
        self.get_logger().info("=== Starting environment reset ===")
        success = True
        message_parts = []

        # Step 1: Delete cable (skip on first reset if launched via entrypoint.sh)
        entity_name = self.get_parameter("cable_entity_name").value
        self.get_logger().info(f"Step 1: Deleting cable '{entity_name}'...")
        if not self._delete_entity(entity_name):
            message_parts.append("cable deletion failed (may not exist)")
            # Not fatal - cable might not exist yet

        # Step 2: Delete task board if configured
        if self.get_parameter("delete_task_board").value:
            self.get_logger().info("Step 1b: Deleting task board...")
            self._delete_entity("task_board")

        # Step 3: Deactivate controller
        self.get_logger().info("Step 2: Deactivating aic_controller...")
        if not self._switch_controllers([], ["aic_controller"]):
            success = False
            message_parts.append("controller deactivation failed")

        # Step 4: Reset joints to home
        if success:
            self.get_logger().info("Step 3: Resetting joints to home...")
            if not self._reset_joints():
                success = False
                message_parts.append("joint reset failed")

        # Step 5: Reactivate controller
        self.get_logger().info("Step 4: Reactivating aic_controller...")
        if not self._switch_controllers(["aic_controller"], []):
            success = False
            message_parts.append("controller reactivation failed")

        # Step 6: Tare F/T sensor (before cable attachment adds weight)
        if success:
            self.get_logger().info("Step 5: Taring F/T sensor...")
            if not self._tare_ft_sensor():
                message_parts.append("FT tare failed")
                # Not fatal

        # Step 7: Re-spawn cable
        if success:
            self.get_logger().info("Step 6: Spawning cable...")
            if not self._spawn_cable():
                success = False
                message_parts.append("cable spawn failed")

        # Step 8: Wait for stabilization
        if success:
            self.get_logger().info("Step 7: Waiting for stabilization...")
            self._wait_for_stabilization()

        # Step 9: Re-spawn task board if we deleted it
        if self.get_parameter("delete_task_board").value:
            self.get_logger().info("Step 8: Task board re-spawn not implemented (static)")
            # Task board is typically static and doesn't need re-spawning
            # If you need dynamic task board reset, implement spawn_task_board here

        self._first_reset = False

        if success:
            msg = "Environment reset complete"
            if message_parts:
                msg += f" (warnings: {'; '.join(message_parts)})"
            self.get_logger().info(f"=== {msg} ===")
        else:
            msg = f"Environment reset failed: {'; '.join(message_parts)}"
            self.get_logger().error(f"=== {msg} ===")

        response.success = success
        response.message = msg
        return response


def main(args=None):
    rclpy.init(args=args)
    node = EnvResetter()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
