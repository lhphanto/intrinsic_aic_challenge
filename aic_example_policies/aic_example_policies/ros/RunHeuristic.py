#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
RunHeuristic: Vision-guided 3-stage insertion policy using PortKeypointNet.

Stage 1 APPROACH — move TCP above the port using keypoint 3D reconstruction.
                   Mostly XY correction; large Z offset kept for safety.
Stage 2 DESCEND  — once XY-aligned, slowly lower TCP while re-centring XY
                   from keypoint estimates.  Force sensor guards against
                   excessive lateral misalignment.
Stage 3 (done)   — insertion declared when TCP has descended past the
                   connector-type-specific depth threshold.

3-D reconstruction:
  The keypoint model outputs xy_norm normalised against the saved image size
  (cam_info.width * IMAGE_SCALE × cam_info.height * IMAGE_SCALE).
  This simplifies to:  u_full_res = xy_norm[0] * cam_info.width
  which is the full-resolution pixel coordinate expected by the P matrix.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms as T

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time
from tf2_ros import TransformException

# ---------------------------------------------------------------------------
# Configuration — update paths before running
# ---------------------------------------------------------------------------

KEYPOINT_CHECKPOINT_PATH: Path | None = Path(
    "/ws_aic/checkpoints/0512_keypoint1_best.pt"
)

_KEYPOINT_CAM_ORDER = ["left_camera", "center_camera", "right_camera"]

# Must match dataset.py: saved image size = full-res * IMAGE_SCALE
IMAGE_SCALE = 0.25

# ---------------------------------------------------------------------------
# Stage parameters
# ---------------------------------------------------------------------------

_APPROACH_Z_OFFSET  = 0.06    # m above port face to start approach
_FINE_Z_OFFSET      = 0.012   # m above port face — transition to descent
_APPROACH_Z_TAPER   = 0.005   # m reduction per converged step

_MAX_XY_STEP        = 0.012   # m max lateral displacement per cycle
_MAX_Z_STEP         = 0.008   # m max vertical displacement per cycle
_DESCEND_STEP       = 0.001   # m lowered per cycle during descent
_DESCEND_XY_STEP    = 0.006   # m max lateral correction during descent

_XY_ALIGN_THRESH    = 0.006   # m — consider XY aligned below this
_Z_APPROACH_THRESH  = 0.008   # m — consider Z-target reached below this

_FORCE_ABORT_N      = 18.0    # N total force — hard stop
_FORCE_LATERAL_N    = 8.0     # N lateral force — misaligned → back off
_BACKOFF_M          = 0.005   # m to raise TCP when lateral force is too high

_APPROACH_STEPS     = 350
_DESCEND_STEPS      = 350
_CONTROL_HZ         = 10.0

# How far below the port face to descend before declaring insertion success
_INSERTION_DEPTH: dict[str, float] = {
    "sfp_port_0":   0.045,   # SFP: ~45 mm insertion depth
    "sfp_port_1":   0.045,
    "sc_port_base": 0.015,   # SC: ~15 mm
}

# Minimum model confidence to accept a keypoint detection
_CONF_VIS_MIN  = 0.55
_CONF_PRES_MIN = 0.55

# Search behaviour when the port is not detected
# After _NO_DETECT_PATIENCE consecutive misses, the TCP cycles through
# _SEARCH_PATTERN offsets (relative to the TCP position when search started),
# dwelling _SEARCH_DWELL frames at each position before advancing.
# As soon as the port is detected, the search state resets automatically.
_NO_DETECT_PATIENCE = 5
_SEARCH_DWELL       = 3
_SEARCH_PATTERN: list[tuple[float, float, float]] = [
    ( 0.000,  0.000,  0.030),  # lift up 30 mm — reduce self-occlusion
    ( 0.020,  0.000,  0.030),  # lift + shift right
    (-0.020,  0.000,  0.030),  # lift + shift left
    ( 0.000,  0.020,  0.030),  # lift + shift forward
    ( 0.000, -0.020,  0.030),  # lift + shift backward
    ( 0.000,  0.000,  0.060),  # lift more if still not found
]

# Preprocessing applied to each camera image before the keypoint model.
# Matches dataset.py LEROBOT_TRANSFORM: input must be (C, H, W) float32 in [0, 1].
_KP_TRANSFORM = T.Compose([
    T.Resize((256, 256), antialias=True),
    T.CenterCrop((224, 224)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion → 3×3 rotation matrix (float32)."""
    x, y, z, w = q_xyzw.astype(np.float64)
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def _tf_to_rt(tf_stamped):
    """Extract R (3×3 float32) and t (3,) float32 from a TF stamped transform."""
    tr = tf_stamped.transform
    R = _quat_to_rot(np.array([
        tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w,
    ]))
    t = np.array([tr.translation.x, tr.translation.y, tr.translation.z], dtype=np.float32)
    return R, t


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class RunHeuristic(Policy):
    """Heuristic 3-stage SFP/SC insertion policy using PortKeypointNet.

    Requires:
      - A trained PortKeypointNet checkpoint at KEYPOINT_CHECKPOINT_PATH.
      - TF frames: base_link and camera optical frames (from the running stack).
      - Observation messages with CameraInfo (camera_info included in Observation.msg).
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"RunHeuristic: using device {self.device}")
        self._keypoint_model = self._load_keypoint_model()
        self._tare_force: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_keypoint_model(self):
        if KEYPOINT_CHECKPOINT_PATH is None or not KEYPOINT_CHECKPOINT_PATH.exists():
            self.get_logger().warn(
                f"RunHeuristic: keypoint model not found at {KEYPOINT_CHECKPOINT_PATH}"
            )
            return None
        from aic_training.keypoint_model import PortKeypointNet
        ckpt  = torch.load(KEYPOINT_CHECKPOINT_PATH, map_location="cpu")
        model = PortKeypointNet(pretrained=False)
        model.load_state_dict(ckpt.get("model", ckpt))
        model.eval().to(self.device)
        self.get_logger().info(
            f"RunHeuristic: loaded keypoint model from {KEYPOINT_CHECKPOINT_PATH}"
        )
        return model

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _ros_image_to_tensor(ros_img) -> torch.Tensor:
        """sensor_msgs/Image → (C, H, W) uint8 RGB tensor (no resize)."""
        img = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(
            ros_img.height, ros_img.width, -1
        ).copy()
        if ros_img.encoding in ("bgr8", "bgra8"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

    def _run_keypoint_model(
        self, images: dict[str, torch.Tensor]
    ) -> dict | None:
        """Run PortKeypointNet.  Returns the raw output dict or None if no model."""
        if self._keypoint_model is None:
            return None
        imgs_list = []
        for cam in _KEYPOINT_CAM_ORDER:
            img_f = images[cam].float() / 255.0          # (C, H, W) in [0, 1]
            imgs_list.append(_KP_TRANSFORM(img_f).unsqueeze(0).to(self.device))
        with torch.no_grad():
            return self._keypoint_model(imgs_list)

    # ------------------------------------------------------------------
    # 3-D reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_port_3d(
        self,
        target_module: str,
        port_name: str,
        pred: dict,
        obs_msg: Observation,
    ) -> np.ndarray | None:
        """Return the port's (x, y, z) in base_link frame, or None on failure.

        Picks the camera with the highest joint (vis × pres) confidence.
        Falls back gracefully if TF lookup fails or confidence is too low.
        """
        from aic_training.keypoint_constants import OUTPUT_KEYS

        cam_infos = {
            "left_camera":   obs_msg.left_camera_info,
            "center_camera": obs_msg.center_camera_info,
            "right_camera":  obs_msg.right_camera_info,
        }

        conf_vis  = pred["conf_visible"][0].cpu()  # (36,)
        conf_pres = pred["conf_present"][0].cpu()  # (36,)
        xy_pred   = pred["xy"][0].cpu()             # (36, 2)
        log_dist  = pred["log_dist"][0].cpu()       # (36,)

        # Find the detection with highest confidence for this (entity, port) pair
        best_idx, best_cam, best_score = None, None, 0.0
        for out_idx, (entity, port, cam) in enumerate(OUTPUT_KEYS):
            if entity != target_module or port != port_name:
                continue
            cv = float(conf_vis[out_idx])
            cp = float(conf_pres[out_idx])
            if cv < _CONF_VIS_MIN or cp < _CONF_PRES_MIN:
                continue
            score = cv * cp
            if score > best_score:
                best_score, best_idx, best_cam = score, out_idx, cam

        if best_idx is None:
            return None

        cam_info = cam_infos.get(best_cam)
        if cam_info is None:
            return None

        P  = np.array(cam_info.p).reshape(3, 4)
        fx = float(P[0, 0])
        fy = float(P[1, 1])
        cx = float(P[0, 2])
        cy = float(P[1, 2])

        # xy_norm is normalised against the saved image size (cam_info dims × IMAGE_SCALE).
        # This means: u_full = xy_norm[0] * cam_info.width  (see module docstring).
        u_full = float(xy_pred[best_idx, 0]) * cam_info.width
        v_full = float(xy_pred[best_idx, 1]) * cam_info.height
        dist   = math.exp(float(log_dist[best_idx]))

        # Back-project to 3-D ray in camera frame, scale to measured distance
        d_x = (u_full - cx) / fx
        d_y = (v_full - cy) / fy
        d_z = 1.0
        scale = dist / math.sqrt(d_x*d_x + d_y*d_y + d_z*d_z)
        point_cam = np.array([d_x * scale, d_y * scale, d_z * scale], dtype=np.float32)

        # Transform camera-frame point to base_link
        camera_frame = cam_info.header.frame_id
        try:
            tf_s = self._parent_node._tf_buffer.lookup_transform(
                "base_link", camera_frame, Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed for {camera_frame}: {ex}")
            return None

        R, trans = _tf_to_rt(tf_s)
        return R @ point_cam + trans  # (3,) in base_link

    # ------------------------------------------------------------------
    # Force sensor helpers
    # ------------------------------------------------------------------

    def _get_wrench(self, obs_msg: Observation) -> np.ndarray:
        """Return tare-corrected wrench as (6,) [fx, fy, fz, tx, ty, tz] in N / N·m."""
        w = obs_msg.wrist_wrench.wrench
        raw = np.array([
            w.force.x,  w.force.y,  w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ], dtype=np.float32)
        if self._tare_force is None:
            self._tare_force = raw.copy()
            self.get_logger().info(
                f"RunHeuristic: force tare set to |F|={float(np.linalg.norm(raw[:3])):.2f} N"
            )
        return raw - self._tare_force

    @staticmethod
    def _force_mag(w: np.ndarray) -> float:
        return float(np.linalg.norm(w[:3]))

    @staticmethod
    def _lateral_force(w: np.ndarray) -> float:
        """XY force magnitude — proxy for misalignment."""
        return float(math.sqrt(w[0]*w[0] + w[1]*w[1]))

    # ------------------------------------------------------------------
    # Motion helper
    # ------------------------------------------------------------------

    def _move_toward(
        self,
        tcp_pos: np.ndarray,
        tcp_quat: np.ndarray,
        target_pos: np.ndarray,
        move_robot,
        max_xy: float = _MAX_XY_STEP,
        max_z:  float = _MAX_Z_STEP,
        stiffness: list | None = None,
        damping:   list | None = None,
    ) -> None:
        """Clamp XY and Z displacement separately, then send a pose target."""
        d = target_pos - tcp_pos

        # Clamp lateral (XY)
        xy_mag = math.sqrt(d[0]*d[0] + d[1]*d[1])
        if xy_mag > max_xy:
            scale = max_xy / xy_mag
            d[0] *= scale
            d[1] *= scale

        # Clamp vertical (Z)
        d[2] = float(np.clip(d[2], -max_z, max_z))

        new_pos = tcp_pos + d
        kw: dict = {}
        if stiffness is not None:
            kw["stiffness"] = stiffness
        if damping is not None:
            kw["damping"] = damping

        self.set_pose_target(
            move_robot=move_robot,
            pose=Pose(
                position=Point(
                    x=float(new_pos[0]), y=float(new_pos[1]), z=float(new_pos[2])
                ),
                orientation=Quaternion(
                    x=float(tcp_quat[0]), y=float(tcp_quat[1]),
                    z=float(tcp_quat[2]), w=float(tcp_quat[3]),
                ),
            ),
            **kw,
        )

    # ------------------------------------------------------------------
    # Shared image extraction
    # ------------------------------------------------------------------

    def _get_images(self, obs_msg: Observation) -> dict[str, torch.Tensor]:
        return {
            "left_camera":   self._ros_image_to_tensor(obs_msg.left_image),
            "center_camera": self._ros_image_to_tensor(obs_msg.center_image),
            "right_camera":  self._ros_image_to_tensor(obs_msg.right_image),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        log = self.get_logger()
        log.info(
            f"RunHeuristic.insert_cable()  target={task.target_module_name}  "
            f"port={task.port_name}"
        )

        if self._keypoint_model is None:
            log.error("RunHeuristic: no keypoint model loaded — cannot run")
            return False

        target_module = task.target_module_name
        port_name     = task.port_name
        insert_depth  = _INSERTION_DEPTH.get(port_name, 0.030)

        # Tare force sensor from the very first observation
        self._tare_force = None
        obs_msg = get_observation()
        self._get_wrench(obs_msg)  # captures tare

        def _tcp(obs: Observation):
            cs = obs.controller_state
            p, q = cs.tcp_pose.position, cs.tcp_pose.orientation
            return (
                np.array([p.x, p.y, p.z], dtype=np.float32),
                np.array([q.x, q.y, q.z, q.w], dtype=np.float32),
            )

        # ==============================================================
        # Stage 1: Approach
        # Move TCP above the port XY position with a decreasing Z offset.
        # ==============================================================
        send_feedback("Stage 1: approaching port")
        log.info("Stage 1: Approach")

        port_pos: np.ndarray | None = None
        z_offset = _APPROACH_Z_OFFSET

        # Search state — activated after _NO_DETECT_PATIENCE consecutive misses
        no_detect_count  = 0
        search_origin: np.ndarray | None = None  # TCP pos when search started
        search_pat_idx   = 0
        search_dwell_cnt = 0

        for step in range(_APPROACH_STEPS):
            obs_msg = get_observation()
            wrench  = self._get_wrench(obs_msg)
            tcp_pos, tcp_quat = _tcp(obs_msg)

            if self._force_mag(wrench) > _FORCE_ABORT_N:
                log.error(
                    f"Force abort in approach: |F|={self._force_mag(wrench):.1f} N"
                )
                return False

            # Re-estimate port position every step
            pred = self._run_keypoint_model(self._get_images(obs_msg))
            detected = False
            if pred is not None:
                pos = self._reconstruct_port_3d(target_module, port_name, pred, obs_msg)
                if pos is not None:
                    port_pos = pos
                    detected = True

            if not detected:
                no_detect_count += 1
                if no_detect_count < _NO_DETECT_PATIENCE:
                    # Brief patience window — just wait
                    self.sleep_for(1.0 / _CONTROL_HZ)
                    continue

                # --- Search mode: lift Z and try small XY offsets ---
                if search_origin is None:
                    search_origin    = tcp_pos.copy()
                    search_pat_idx   = 0
                    search_dwell_cnt = 0
                    log.info(
                        f"  Step {step}: port not detected for {no_detect_count} frames "
                        f"— starting search from ({search_origin[0]:.3f}, "
                        f"{search_origin[1]:.3f}, {search_origin[2]:.3f})"
                    )

                dx, dy, dz = _SEARCH_PATTERN[search_pat_idx]
                search_target = search_origin + np.array([dx, dy, dz], dtype=np.float32)
                self._move_toward(tcp_pos, tcp_quat, search_target, move_robot)
                log.info(
                    f"  Step {step}: search [{search_pat_idx}] "
                    f"offset=({dx:.3f}, {dy:.3f}, {dz:.3f})"
                )

                search_dwell_cnt += 1
                if search_dwell_cnt >= _SEARCH_DWELL:
                    search_pat_idx   = (search_pat_idx + 1) % len(_SEARCH_PATTERN)
                    search_dwell_cnt = 0

                self.sleep_for(1.0 / _CONTROL_HZ)
                continue

            # Port detected — reset search state
            no_detect_count  = 0
            search_origin    = None
            search_pat_idx   = 0
            search_dwell_cnt = 0

            target = np.array([port_pos[0], port_pos[1], port_pos[2] + z_offset],
                               dtype=np.float32)

            xy_err = math.sqrt(
                (tcp_pos[0] - target[0])**2 + (tcp_pos[1] - target[1])**2
            )
            z_err = abs(tcp_pos[2] - target[2])

            if xy_err < _XY_ALIGN_THRESH and z_err < _Z_APPROACH_THRESH:
                # Gradually reduce Z offset to transition toward fine alignment
                if z_offset > _FINE_Z_OFFSET:
                    z_offset = max(_FINE_Z_OFFSET, z_offset - _APPROACH_Z_TAPER)
                    log.info(
                        f"  Step {step}: converged — reducing z_offset to {z_offset*1000:.0f} mm"
                    )
                else:
                    log.info(
                        f"Stage 1 done at step {step}  xy_err={xy_err*1000:.1f} mm"
                    )
                    break

            if step % 30 == 0:
                log.info(
                    f"  Step {step}: port=({port_pos[0]:.3f}, {port_pos[1]:.3f}, {port_pos[2]:.3f})  "
                    f"xy_err={xy_err*1000:.1f} mm  z_off={z_offset*1000:.0f} mm"
                )

            self._move_toward(tcp_pos, tcp_quat, target, move_robot)
            self.sleep_for(1.0 / _CONTROL_HZ)
        else:
            log.error("Stage 1 timed out")
            return False

        if port_pos is None:
            log.error("Port never detected")
            return False

        # ==============================================================
        # Stage 2: Descend
        # Slowly lower Z while re-centering on port XY from keypoints.
        # Lateral force guard: back off when the connector is misaligned.
        # ==============================================================
        send_feedback("Stage 2: descending for insertion")
        log.info("Stage 2: Descend")

        # Z command starts at current TCP position (already near port + FINE_Z_OFFSET)
        z_cmd = float(tcp_pos[2])
        port_face_z = float(port_pos[2])

        for step in range(_DESCEND_STEPS):
            obs_msg = get_observation()
            wrench  = self._get_wrench(obs_msg)
            tcp_pos, tcp_quat = _tcp(obs_msg)

            f_mag = self._force_mag(wrench)
            f_lat = self._lateral_force(wrench)

            if f_mag > _FORCE_ABORT_N:
                log.error(f"Force abort in descent: |F|={f_mag:.1f} N at step {step}")
                return False

            # Lateral force too high → misaligned, back off a few mm
            if f_lat > _FORCE_LATERAL_N:
                z_cmd += _BACKOFF_M
                log.warn(
                    f"  Step {step}: lateral F={f_lat:.1f} N — backing off to z={z_cmd:.4f}"
                )
                target = np.array([port_pos[0], port_pos[1], z_cmd], dtype=np.float32)
                self._move_toward(tcp_pos, tcp_quat, target, move_robot,
                                  max_xy=_DESCEND_XY_STEP, max_z=_BACKOFF_M)
                self.sleep_for(1.0 / _CONTROL_HZ)
                continue

            # Re-estimate XY position for ongoing centering
            pred = self._run_keypoint_model(self._get_images(obs_msg))
            if pred is not None:
                pos = self._reconstruct_port_3d(target_module, port_name, pred, obs_msg)
                if pos is not None:
                    port_pos = pos
                    port_face_z = float(pos[2])

            # Advance Z downward
            z_cmd -= _DESCEND_STEP
            depth = port_face_z - z_cmd  # positive = below face

            target = np.array([port_pos[0], port_pos[1], z_cmd], dtype=np.float32)
            self._move_toward(
                tcp_pos, tcp_quat, target, move_robot,
                max_xy=_DESCEND_XY_STEP,
                max_z=_DESCEND_STEP * 2,
                # Softer in Z (index 2) for compliant insertion
                stiffness=[90.0, 90.0, 40.0, 50.0, 50.0, 50.0],
                damping   =[50.0, 50.0, 30.0, 20.0, 20.0, 20.0],
            )

            if depth >= insert_depth:
                log.info(
                    f"Stage 2 done: depth={depth*1000:.1f} mm >= "
                    f"{insert_depth*1000:.0f} mm  |F|={f_mag:.1f} N"
                )
                break

            if step % 20 == 0:
                log.info(
                    f"  Step {step}: depth={depth*1000:.1f}/{insert_depth*1000:.0f} mm  "
                    f"|F|={f_mag:.1f} N  lat={f_lat:.1f} N"
                )

            self.sleep_for(1.0 / _CONTROL_HZ)
        else:
            log.warn("Stage 2 timed out without reaching insertion depth")
            return False

        # ==============================================================
        # Done
        # ==============================================================
        send_feedback("Insertion complete")
        log.info("RunHeuristic: insertion complete — settling 1 s")
        self.sleep_for(1.0)
        return True
