#!/usr/bin/env python3
"""
Environment reset node for RL training.

This node runs INSIDE the distrobox/docker container where simulation_interfaces
is available. It provides a /env/reset service (std_srvs/SetBool) that performs
the full reset sequence:

  data=False (deterministic):
    1. Delete cable entity from Gazebo
    2. Delete task board (only if delete_task_board param is True)
    3. Deactivate aic_controller
    4. Reset robot joints to home positions
    5. Reactivate aic_controller
    6. Tare force/torque sensor
    7. Re-spawn cable at fixed pose (from ROS2 params)
    8. Wait for robot joints to stabilize

  data=True (random, requires config_path param):
    1. Delete cable entity from Gazebo
    2. Delete task board
    3. Deactivate aic_controller
    4. Reset robot joints to home positions
    5. Reactivate aic_controller
    6. Tare force/torque sensor
    7. Re-spawn task board with randomly sampled trial scene
       (NIC rails, SC rails, mount rails, board pose)
    8. Re-spawn cable with randomly sampled trial cable config
    9. Wait for robot joints to stabilize

Usage (inside distrobox):
    source /ws_aic/install/setup.bash
    python3 env_resetter.py [--ros-args -p use_sim_time:=true]
"""

import math
import random
import subprocess
import sys
import time
from threading import Event

import yaml

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from aic_engine_interfaces.srv import ResetJoints
from controller_manager_msgs.srv import SwitchController
from simulation_interfaces.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import SetBool, Trigger
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
STABILIZE_TIMEOUT_SEC = 20.0
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


def generate_task_board_urdf(task_board_cfg: dict, limits: dict) -> str:
    """Run xacro to generate task board URDF from a trial's scene.task_board config.

    Maps YAML keys to xacro args:
      nic_rail_N       → nic_card_mount_N_*   (N = 0..4)
      sc_rail_N        → sc_port_N_*          (N = 0..1)
      lc_mount_rail_N  → lc_mount_rail_N_*    (N = 0..1)
      sfp_mount_rail_N → sfp_mount_rail_N_*   (N = 0..1)
      sc_mount_rail_N  → sc_mount_rail_N_*    (N = 0..1)

    Translations are clamped to the limits from task_board_limits in the config.
    """
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "aic_description", "--share"],
            capture_output=True, text=True, check=True,
        )
        aic_desc_share = result.stdout.strip()
        xacro_file = f"{aic_desc_share}/urdf/task_board.urdf.xacro"

        pose = task_board_cfg.get("pose", {})

        nic_lim = limits.get("nic_rail", {})
        nic_min = float(nic_lim.get("min_translation", -0.0215))
        nic_max = float(nic_lim.get("max_translation", 0.0234))

        sc_lim = limits.get("sc_rail", {})
        sc_min = float(sc_lim.get("min_translation", -0.06))
        sc_max = float(sc_lim.get("max_translation", 0.055))

        mount_lim = limits.get("mount_rail", {})
        mount_min = float(mount_lim.get("min_translation", -0.09425))
        mount_max = float(mount_lim.get("max_translation", 0.09425))

        args = [
            f"x:={pose.get('x', 0.15)}",
            f"y:={pose.get('y', -0.2)}",
            f"z:={pose.get('z', 1.14)}",
            f"roll:={pose.get('roll', 0.0)}",
            f"pitch:={pose.get('pitch', 0.0)}",
            f"yaw:={pose.get('yaw', 3.1415)}",
        ]

        # NIC rails (0-4) → nic_card_mount_N
        for i in range(5):
            cfg = task_board_cfg.get(f"nic_rail_{i}", {})
            present = cfg.get("entity_present", False)
            args.append(f"nic_card_mount_{i}_present:={'true' if present else 'false'}")
            if present:
                ep = cfg.get("entity_pose", {})
                t = max(nic_min, min(nic_max, float(ep.get("translation", 0.0))))
                args += [
                    f"nic_card_mount_{i}_translation:={t}",
                    f"nic_card_mount_{i}_roll:={ep.get('roll', 0.0)}",
                    f"nic_card_mount_{i}_pitch:={ep.get('pitch', 0.0)}",
                    f"nic_card_mount_{i}_yaw:={ep.get('yaw', 0.0)}",
                ]

        # SC rails (0-1) → sc_port_N
        for i in range(2):
            cfg = task_board_cfg.get(f"sc_rail_{i}", {})
            present = cfg.get("entity_present", False)
            args.append(f"sc_port_{i}_present:={'true' if present else 'false'}")
            if present:
                ep = cfg.get("entity_pose", {})
                t = max(sc_min, min(sc_max, float(ep.get("translation", 0.0))))
                args += [
                    f"sc_port_{i}_translation:={t}",
                    f"sc_port_{i}_roll:={ep.get('roll', 0.0)}",
                    f"sc_port_{i}_pitch:={ep.get('pitch', 0.0)}",
                    f"sc_port_{i}_yaw:={ep.get('yaw', 0.0)}",
                ]

        # Mount rails: lc_mount, sfp_mount, sc_mount (0-1 each)
        for rail_type in ("lc_mount", "sfp_mount", "sc_mount"):
            for i in range(2):
                key = f"{rail_type}_rail_{i}"
                cfg = task_board_cfg.get(key, {})
                present = cfg.get("entity_present", False)
                args.append(f"{key}_present:={'true' if present else 'false'}")
                if present:
                    ep = cfg.get("entity_pose", {})
                    t = max(mount_min, min(mount_max, float(ep.get("translation", 0.0))))
                    args += [
                        f"{key}_translation:={t}",
                        f"{key}_roll:={ep.get('roll', 0.0)}",
                        f"{key}_pitch:={ep.get('pitch', 0.0)}",
                        f"{key}_yaw:={ep.get('yaw', 0.0)}",
                    ]

        cmd = ["xacro", xacro_file] + args
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"xacro failed: {e.stderr}") from e


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
        self.declare_parameter("config_path", "")

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

        # Pre-generate cable SDFs for all known cable types
        attach = self.get_parameter("attach_cable_to_gripper").value
        self._cable_sdfs: dict[str, str] = {}
        for ct in ("sfp_sc_cable", "sfp_sc_cable_reversed"):
            self.get_logger().info(f"Generating cable SDF (type={ct}, attach={attach})...")
            try:
                self._cable_sdfs[ct] = generate_cable_sdf(ct, attach)
                self.get_logger().info(
                    f"Cable SDF '{ct}' generated ({len(self._cable_sdfs[ct])} bytes)"
                )
            except RuntimeError as e:
                self.get_logger().warn(f"Could not pre-generate SDF for '{ct}': {e}")

        # Default SDF (used when not doing random reset)
        default_ct = self.get_parameter("cable_type").value
        if default_ct not in self._cable_sdfs:
            self.get_logger().info(f"Generating default cable SDF (type={default_ct})...")
            self._cable_sdfs[default_ct] = generate_cable_sdf(default_ct, attach)

        # Load trial configs and task board limits from sample_config.yaml for random reset
        self._trials: list[dict] = []
        self._task_board_limits: dict = {}
        config_path = self.get_parameter("config_path").value
        if config_path:
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                self._trials = list(cfg.get("trials", {}).values())
                self._task_board_limits = cfg.get("task_board_limits", {})
                self.get_logger().info(
                    f"Loaded {len(self._trials)} trials from {config_path}"
                )
            except Exception as e:
                self.get_logger().warn(f"Failed to load config '{config_path}': {e}")

        # Reset service (SetBool: data=True → random reset, data=False → deterministic)
        self._reset_service = self.create_service(
            SetBool, "/env/reset", self._reset_callback,
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
        """Synchronous service call with timeout.

        Uses a busy-wait instead of rclpy.spin_until_future_complete because
        the node is already being spun by a MultiThreadedExecutor. Calling
        spin_until_future_complete from within an executor callback causes
        the service response to never be sent back.
        """
        future = client.call_async(request)
        t0 = time.monotonic()
        while not future.done():
            time.sleep(0.05)
            if time.monotonic() - t0 > SERVICE_TIMEOUT_SEC:
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

    def _spawn_cable(
        self,
        offset_x: float = None,
        offset_y: float = None,
        offset_z: float = None,
        roll: float = None,
        pitch: float = None,
        yaw: float = None,
        entity_name: str = None,
        cable_type: str = None,
    ) -> bool:
        if not self._wait_for_service(self._spawn_entity_client, "spawn_entity"):
            return False

        # Fall back to ROS parameters for any value not explicitly provided
        if offset_x is None:
            offset_x = self.get_parameter("cable_offset_x").value
        if offset_y is None:
            offset_y = self.get_parameter("cable_offset_y").value
        if offset_z is None:
            offset_z = self.get_parameter("cable_offset_z").value
        if roll is None:
            roll = self.get_parameter("cable_roll").value
        if pitch is None:
            pitch = self.get_parameter("cable_pitch").value
        if yaw is None:
            yaw = self.get_parameter("cable_yaw").value
        if entity_name is None:
            entity_name = self.get_parameter("cable_entity_name").value
        if cable_type is None:
            cable_type = self.get_parameter("cable_type").value

        cable_sdf = self._cable_sdfs.get(cable_type)
        if cable_sdf is None:
            self.get_logger().error(f"No pre-generated SDF for cable type '{cable_type}'")
            return False

        # Get gripper position for cable placement
        gripper_pos = self._get_gripper_pose()
        if gripper_pos is None:
            return False

        qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)

        req = SpawnEntity.Request()
        req.name = entity_name
        req.allow_renaming = True
        req.resource_string = cable_sdf
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

    def _spawn_task_board(self, task_board_cfg: dict) -> bool:
        """Delete and re-spawn the task board using the given scene config dict."""
        if not self._wait_for_service(self._spawn_entity_client, "spawn_entity"):
            return False

        try:
            urdf = generate_task_board_urdf(task_board_cfg, self._task_board_limits)
        except RuntimeError as e:
            self.get_logger().error(f"Task board xacro generation failed: {e}")
            return False

        pose = task_board_cfg.get("pose", {})
        qx, qy, qz, qw = rpy_to_quaternion(
            float(pose.get("roll", 0.0)),
            float(pose.get("pitch", 0.0)),
            float(pose.get("yaw", 3.1415)),
        )

        req = SpawnEntity.Request()
        req.name = "task_board"
        req.allow_renaming = True
        req.resource_string = urdf
        req.entity_namespace = ""
        req.initial_pose.header.frame_id = "world"
        req.initial_pose.pose.position.x = float(pose.get("x", 0.15))
        req.initial_pose.pose.position.y = float(pose.get("y", -0.2))
        req.initial_pose.pose.position.z = float(pose.get("z", 1.14))
        req.initial_pose.pose.orientation.x = qx
        req.initial_pose.pose.orientation.y = qy
        req.initial_pose.pose.orientation.z = qz
        req.initial_pose.pose.orientation.w = qw

        resp = self._call_service(self._spawn_entity_client, req, "spawn_entity")
        if resp is None or resp.result.result != 1:
            err = getattr(resp, "result", None)
            self.get_logger().error(
                f"Failed to spawn task board: {err.error_message if err else 'timeout'}"
            )
            return False

        self.get_logger().info(f"Task board spawned as '{resp.entity_name}'")
        return True

    def _wait_for_stabilization(self) -> bool:
        """Wait for robot joints to settle (velocities near zero)."""
        self._joints_stable.clear()
        settled = self._joints_stable.wait(timeout=STABILIZE_TIMEOUT_SEC)
        if not settled:
            self.get_logger().warning("Robot did not stabilize within timeout")
        return settled

    def _pick_random_trial(self) -> dict | None:
        """Pick a random trial from the loaded config. Returns the full trial dict,
        or None if no trials are loaded."""
        if not self._trials:
            self.get_logger().warn(
                "Random reset requested but no trials loaded (set config_path parameter)"
            )
            return None
        return random.choice(self._trials)

    def _cable_config_from_trial(self, trial: dict) -> dict | None:
        """Extract cable kwargs for _spawn_cable() from a trial dict."""
        cables = trial.get("scene", {}).get("cables", {})
        if not cables:
            self.get_logger().warn("Trial has no cables section, using defaults")
            return None
        cable_name, cable_cfg = next(iter(cables.items()))
        pose = cable_cfg.get("pose", {})
        offset = pose.get("gripper_offset", {})
        return {
            "offset_x": float(offset.get("x", CABLE_GRIPPER_OFFSET["x"])),
            "offset_y": float(offset.get("y", CABLE_GRIPPER_OFFSET["y"])),
            "offset_z": float(offset.get("z", CABLE_GRIPPER_OFFSET["z"])),
            "roll": float(pose.get("roll", CABLE_ORIENTATION["roll"])),
            "pitch": float(pose.get("pitch", CABLE_ORIENTATION["pitch"])),
            "yaw": float(pose.get("yaw", CABLE_ORIENTATION["yaw"])),
            "entity_name": cable_name,
            "cable_type": cable_cfg.get("cable_type", CABLE_TYPE),
        }

    def _reset_callback(self, request: SetBool.Request, response: SetBool.Response):
        """Full environment reset sequence.

        request.data=True  → randomly pick a trial from sample_config.yaml; resets
                             the task board scene (NIC/SC rails, mount positions, board
                             pose) AND the cable configuration.
        request.data=False → use fixed default parameters (cable only; task board is
                             left as-is unless delete_task_board param is set).
        """
        random_reset = request.data
        self.get_logger().info(
            f"=== Starting environment reset (random={random_reset}) ==="
        )

        trial: dict | None = None
        cable_override: dict | None = None
        task_board_cfg: dict | None = None

        if random_reset:
            trial = self._pick_random_trial()
            if trial:
                cable_override = self._cable_config_from_trial(trial)
                task_board_cfg = trial.get("scene", {}).get("task_board")
                self.get_logger().info(
                    f"Random reset: cable='{cable_override['entity_name'] if cable_override else 'default'}' "
                    f"type='{cable_override['cable_type'] if cable_override else 'default'}', "
                    f"task_board_pose={task_board_cfg.get('pose') if task_board_cfg else 'none'}"
                )

        success = True
        message_parts = []

        # Step 1: Delete cable
        entity_name = (
            cable_override["entity_name"]
            if cable_override
            else self.get_parameter("cable_entity_name").value
        )
        self.get_logger().info(f"Step 1: Deleting cable '{entity_name}'...")
        if not self._delete_entity(entity_name):
            message_parts.append("cable deletion failed (may not exist)")
            # Not fatal - cable might not exist yet

        # Step 2: Delete task board
        #   - always for random reset (we'll re-spawn with new scene config)
        #   - only when delete_task_board param is True for deterministic reset
        if random_reset or self.get_parameter("delete_task_board").value:
            self.get_logger().info("Step 2: Deleting task board...")
            if not self._delete_entity("task_board"):
                message_parts.append("task board deletion failed (may not exist)")
                # Not fatal

        # Step 3: Deactivate controller
        self.get_logger().info("Step 3: Deactivating aic_controller...")
        if not self._switch_controllers([], ["aic_controller"]):
            success = False
            message_parts.append("controller deactivation failed")

        # Step 4: Reset joints to home
        if success:
            self.get_logger().info("Step 4: Resetting joints to home...")
            if not self._reset_joints():
                success = False
                message_parts.append("joint reset failed")

        # Step 5: Reactivate controller
        self.get_logger().info("Step 5: Reactivating aic_controller...")
        if not self._switch_controllers(["aic_controller"], []):
            success = False
            message_parts.append("controller reactivation failed")

        # Step 6: Tare F/T sensor (before cable/board add weight)
        if success:
            self.get_logger().info("Step 6: Taring F/T sensor...")
            if not self._tare_ft_sensor():
                message_parts.append("FT tare failed")
                # Not fatal

        # Step 7: Spawn task board
        #   - for random reset: spawn with new scene config from trial
        #   - for deterministic reset: no-op (task board stays as launched)
        if success and random_reset and task_board_cfg:
            self.get_logger().info("Step 7: Spawning task board from trial config...")
            if not self._spawn_task_board(task_board_cfg):
                success = False
                message_parts.append("task board spawn failed")

        # Step 8: Spawn cable
        if success:
            self.get_logger().info("Step 8: Spawning cable...")
            spawn_ok = (
                self._spawn_cable(**cable_override)
                if cable_override
                else self._spawn_cable()
            )
            if not spawn_ok:
                success = False
                message_parts.append("cable spawn failed")

        # Step 9: Wait for stabilization
        if success:
            self.get_logger().info("Step 9: Waiting for stabilization...")
            self._wait_for_stabilization()

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
