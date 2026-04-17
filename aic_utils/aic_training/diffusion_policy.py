"""
Diffusion policy for the AIC cable-insertion task.

Architecture overview:
  observation (images + robot state + task info)  [obs_horizon steps]
      │
      ├─ ImageEncoder    → obs_horizon × num_cameras image tokens  (512-d each, ResNet18)
      ├─ StateEncoder    → obs_horizon              state tokens   (256-d each, MLP)
      └─ TaskEncoder     → 1                        task token     (32-d, one of 12 combos)
      │
      └─► condition token sequence  (keys & values for cross-attention)
                │
  noisy_action ─┤                                            ┌── DiTLayer (self-attn)
  + timestep    └─► action tokens (queries) ────────────────┤   DiTLayer (cross-attn → cond)
    embedding                                                └── × n_layers
                                                               │
                                                               ▼
                                                         predicted noise
                                                    (used by DDPM/DDIM sampler)

Task integer encodings (from aic_robot_aic_controller.py):

    TASK_TARGET_MODULE_ENCODING:
        "nic_card_mount_0" → 0,  "nic_card_mount_1" → 1,  "nic_card_mount_2" → 2,
        "nic_card_mount_3" → 3,  "nic_card_mount_4" → 4,  "sc_port_0"        → 5,
        "sc_port_1"        → 6,  "sc_port_2"        → 7,  "sc_port_3"        → 8,
        "sc_port_4"        → 9

    TASK_PORT_NAME_ENCODING:
        "sfp_port_0" → 0,  "sfp_port_1" → 1,  "sc_port_base" → 2
"""

import torch
import torch.nn as nn
import torchvision
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Constants (must match aic_robot_aic_controller.py)
# ---------------------------------------------------------------------------

NUM_TARGET_MODULES = 10   # nic_card_mount_{0..4} + sc_port_{0..4}
NUM_PORT_NAMES = 3        # sfp_port_0, sfp_port_1, sc_port_base

# Robot state vector layout:
#   tcp_pose.position    (3)  — x, y, z  [meters]
#   tcp_pose.orientation (4)  — qx, qy, qz, qw  [unit quaternion]
#   tcp_velocity.linear  (3)  — vx, vy, vz  [m/s]
#   tcp_velocity.angular (3)  — wx, wy, wz  [rad/s]
#   joint_positions      (7)  — 6 arm joints + 1 gripper  [rad]
#   wrench.force         (3)  — fx, fy, fz  [N]  (tare-corrected)
#   wrench.torque        (3)  — tx, ty, tz  [Nm] (tare-corrected)
ROBOT_STATE_DIM = 26  # 3+4+3+3+7+3+3

# ResNet18 output dimension (after removing the fc layer)
RESNET_FEATURE_DIM = 512


# ---------------------------------------------------------------------------
# Vision encoder helpers
# ---------------------------------------------------------------------------

def get_resnet(name: str, weights=None, **kwargs) -> nn.Module:
    """Initialize a standard ResNet vision encoder without its final fc layer.

    Args:
        name:    "resnet18", "resnet34", or "resnet50"
        weights: "IMAGENET1K_V1" for pretrained ImageNet weights (default), None for random init
    Returns:
        ResNet with fc replaced by nn.Identity().
        Output dim: 512 for resnet18/34, 2048 for resnet50.
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Identity()
    return resnet


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace all submodules selected by predicate with the output of func."""
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)

    # verify all targeted modules have been replaced
    remaining = [
        k for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(remaining) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16,
) -> nn.Module:
    """Replace all BatchNorm2d layers with GroupNorm.

    IMPORTANT: must be called before using the encoder with EMA, otherwise
    training performance will degrade significantly.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features,
        ),
    )
    return root_module


# ---------------------------------------------------------------------------
# Image Encoder
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """Encodes images from all three cameras using a single shared ResNet18.

    The same ResNet18 backbone is applied independently to each camera view,
    then the per-camera features are concatenated.

    Input:  images dict {camera_key: (B, obs_horizon, C, H, W)}
    Output: (B, obs_horizon, num_cameras * RESNET_FEATURE_DIM)

    The obs_horizon dimension allows the policy to condition on a short
    history of frames (same pattern as the reference diffusion policy demo).
    When obs_horizon=1 the output is (B, 1, num_cameras * 512).
    """

    CAMERA_KEYS = ("left_camera", "center_camera", "right_camera")

    def __init__(
        self,
        resnet_name: str = "resnet18",
        weights="IMAGENET1K_V1",
        features_per_group: int = 16,
    ):
        super().__init__()

        # Single shared backbone — all cameras go through the same weights
        backbone = get_resnet(resnet_name, weights=weights)
        # Replace BatchNorm with GroupNorm so the encoder is compatible with EMA
        self.backbone = replace_bn_with_gn(backbone, features_per_group=features_per_group)

        self.feature_dim = RESNET_FEATURE_DIM          # 512 for resnet18
        self.num_cameras = len(self.CAMERA_KEYS)
        self.output_dim  = self.num_cameras * self.feature_dim  # 3 * 512 = 1536

    def forward(self, images: dict) -> torch.Tensor:
        """
        Args:
            images: {camera_key: (B, obs_horizon, C, H, W)}
                    pixel values normalised to [0, 1]

        Returns:
            features: (B, obs_horizon, num_cameras * RESNET_FEATURE_DIM)
        """
        B, T = next(iter(images.values())).shape[:2]

        per_camera = []
        for key in self.CAMERA_KEYS:
            img = images[key]                          # (B, T, C, H, W)
            # Merge batch and time so the backbone sees (B*T, C, H, W)
            img_flat = img.flatten(end_dim=1)          # (B*T, C, H, W)
            feat = self.backbone(img_flat)             # (B*T, 512)
            feat = feat.reshape(B, T, self.feature_dim)  # (B, T, 512)
            per_camera.append(feat)

        # Concatenate along feature axis: (B, T, num_cameras * 512)
        return torch.cat(per_camera, dim=-1)


# ---------------------------------------------------------------------------
# Diffusion Timestep Encoder
# ---------------------------------------------------------------------------

class DiffusionTimestepEncoder(nn.Module):
    """Encodes the diffusion timestep t into a sinusoidal + MLP embedding.

    Standard in DDPM / Diffusion Policy literature.
    Produces a vector that is added to (or concatenated with) the noisy
    action before passing through the noise prediction network.

    Input:  (B,) integer timestep tensor
    Output: (B, timestep_embed_dim) float32 tensor
    """

    def __init__(self, timestep_embed_dim: int = 128):
        super().__init__()
        self.timestep_embed_dim = timestep_embed_dim
        assert timestep_embed_dim % 2 == 0, "timestep_embed_dim must be even"

        # TODO: optionally replace the MLP with a larger network or
        #       use learned frequency embeddings instead of sinusoidal ones
        self.mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional embedding for scalar timesteps.

        Args:
            t: (B,) integer or float timestep
        Returns:
            emb: (B, timestep_embed_dim)
        """
        half = self.timestep_embed_dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=t.device)
            * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )                                    # (half,)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, timestep_embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) diffusion timestep indices (integers 0 … T-1)
        Returns:
            emb: (B, timestep_embed_dim)
        """
        sin_emb = self._sinusoidal_embedding(t)   # (B, timestep_embed_dim)
        return self.mlp(sin_emb)                  # (B, timestep_embed_dim)


# ---------------------------------------------------------------------------
# Task Encoder
# ---------------------------------------------------------------------------

class TaskEncoder(nn.Module):
    """Encodes task identity as a single token from 12 valid (module, port) combinations.

    Valid combinations (from aic_robot_aic_controller.py encodings):

        task_idx  target_module        port_name
        ────────  ───────────────────  ──────────────
          0       nic_card_mount_0 (0)  sfp_port_0 (0)
          1       nic_card_mount_0 (0)  sfp_port_1 (1)
          2       nic_card_mount_1 (1)  sfp_port_0 (0)
          3       nic_card_mount_1 (1)  sfp_port_1 (1)
          4       nic_card_mount_2 (2)  sfp_port_0 (0)
          5       nic_card_mount_2 (2)  sfp_port_1 (1)
          6       nic_card_mount_3 (3)  sfp_port_0 (0)
          7       nic_card_mount_3 (3)  sfp_port_1 (1)
          8       nic_card_mount_4 (4)  sfp_port_0 (0)
          9       nic_card_mount_4 (4)  sfp_port_1 (1)
         10       sc_port_0        (5)  sc_port_base (2)
         11       sc_port_1        (6)  sc_port_base (2)

    Input:  target_module (B,) int, port_name (B,) int
    Output: (B, 1, task_embed_dim) — single task token
    """

    NUM_TASKS = 12

    # Lookup table: _TASK_INDEX[target_module_int, port_name_int] → task_idx
    # Shape (NUM_TARGET_MODULES, NUM_PORT_NAMES), -1 for invalid combinations.
    _LOOKUP = torch.full((NUM_TARGET_MODULES, NUM_PORT_NAMES), fill_value=-1, dtype=torch.long)
    # nic_card_mount_{0..4} × {sfp_port_0, sfp_port_1}
    for _m in range(5):
        _LOOKUP[_m, 0] = _m * 2      # sfp_port_0
        _LOOKUP[_m, 1] = _m * 2 + 1  # sfp_port_1
    # sc_port_0 and sc_port_1 × sc_port_base
    _LOOKUP[5, 2] = 10
    _LOOKUP[6, 2] = 11

    def __init__(self, task_embed_dim: int = 32):
        super().__init__()
        self.task_embed_dim = task_embed_dim
        self.embedding = nn.Embedding(self.NUM_TASKS, task_embed_dim)
        # Register as buffer so it moves with .to(device)
        self.register_buffer("lookup", self._LOOKUP.clone())

    def forward(
        self,
        target_module: torch.Tensor,
        port_name: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_module: (B,) int64 — index from TASK_TARGET_MODULE_ENCODING
            port_name:     (B,) int64 — index from TASK_PORT_NAME_ENCODING
        Returns:
            task_token: (B, 1, task_embed_dim)
        """
        task_idx = self.lookup[target_module, port_name]   # (B,)
        return self.embedding(task_idx).unsqueeze(1)        # (B, 1, task_embed_dim)


# ---------------------------------------------------------------------------
# Diffusion Transformer (noise prediction network)
# ---------------------------------------------------------------------------

class DiTLayer(nn.Module):
    """Single Diffusion Transformer layer.

    Each action token:
      1. Self-attends to other action tokens (captures inter-step dependencies).
      2. Cross-attends to the condition token sequence
         (image tokens + state tokens + task token).
      3. Passes through a feed-forward network.

    Pre-norm (LayerNorm before each sub-layer) for training stability.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, pred_horizon, d_model)  — noisy action tokens
            cond: (B, num_cond_tokens, d_model) — condition token sequence
        Returns:
            x:    (B, pred_horizon, d_model)
        """
        # 1. Self-attention over action tokens
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + self.drop(x)

        # 2. Cross-attention: action tokens query the condition sequence
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(query=x, key=cond, value=cond)
        x = residual + self.drop(x)

        # 3. Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + self.drop(x)

        return x


class DiffusionTransformer(nn.Module):
    """Transformer-based noise prediction network for diffusion policy.

    Condition token sequence (keys/values for cross-attention):
      ┌─────────────────────────────────────────────────────────────────┐
      │  image tokens   (obs_horizon × num_cameras tokens, d_model each) │
      │  state tokens   (obs_horizon tokens, d_model each)               │
      │  task token     (1 token for the combined task identity, d_model)  │
      └─────────────────────────────────────────────────────────────────┘
    Total condition tokens: obs_horizon × (num_cameras + 1) + 1

    Query tokens (action sequence):
      noisy_action projected to d_model + diffusion timestep embedding added.
    """

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        # input projection dims — must match encoder output dims
        img_token_dim: int = RESNET_FEATURE_DIM,   # 512 per camera per timestep
        state_token_dim: int = 256,                 # hidden_dim // 2
        task_token_dim: int = 32,
        timestep_embed_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        # --- Input projections: each condition source → d_model ---
        self.img_proj   = nn.Linear(img_token_dim,   d_model)
        self.state_proj = nn.Linear(state_token_dim, d_model)
        self.task_proj  = nn.Linear(task_token_dim,  d_model)  # applied to each of the 2 task tokens

        # --- Action input projection ---
        # noisy action tokens + timestep embedding are summed after projection
        self.action_proj    = nn.Linear(action_dim,        d_model)
        self.timestep_proj  = nn.Linear(timestep_embed_dim, d_model)

        # --- Transformer layers ---
        self.layers = nn.ModuleList([
            DiTLayer(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # --- Output projection: d_model → action_dim ---
        self.out_proj = nn.Linear(d_model, action_dim)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep_emb: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        task_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action:  (B, pred_horizon, action_dim)
            timestep_emb:  (B, timestep_embed_dim)
            img_tokens:    (B, obs_horizon * num_cameras, RESNET_FEATURE_DIM)
            state_tokens:  (B, obs_horizon, state_token_dim)
            task_token:    (B, 1, task_token_dim)

        Returns:
            noise_pred: (B, pred_horizon, action_dim)
        """
        # --- Build condition token sequence ---
        img_cond   = self.img_proj(img_tokens)     # (B, obs_horizon*num_cameras, d_model)
        state_cond = self.state_proj(state_tokens)  # (B, obs_horizon, d_model)
        task_cond  = self.task_proj(task_token)     # (B, 1, d_model)
        # cond: (B, obs_horizon*num_cameras + obs_horizon + 1, d_model)
        cond = torch.cat([img_cond, state_cond, task_cond], dim=1)

        # --- Build action token sequence ---
        x = self.action_proj(noisy_action)          # (B, pred_horizon, d_model)
        # Add timestep embedding broadcast over pred_horizon
        t = self.timestep_proj(timestep_emb)        # (B, d_model)
        x = x + t.unsqueeze(1)                      # (B, pred_horizon, d_model)

        # --- Transformer layers ---
        for layer in self.layers:
            x = layer(x, cond)

        x = self.norm_out(x)                        # (B, pred_horizon, d_model)
        return self.out_proj(x)                     # (B, pred_horizon, action_dim)


# ---------------------------------------------------------------------------
# Diffusion Policy (top-level module)
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """Diffusion policy for the AIC cable-insertion task.

    Combines image, robot state, and task embeddings as a condition token
    sequence, then uses a Diffusion Transformer to predict the noise added
    to the action at each denoising step.

    Inputs at each denoising step:
        images          dict[str, (B, obs_horizon, C, H, W)]  — one entry per camera
        robot_state     (B, obs_horizon, ROBOT_STATE_DIM)     — tcp_pose(7), tcp_velocity(6), joint_positions(7), wrench(6)
        target_module   (B,)  int                              — task target module index
        port_name       (B,)  int                              — task port name index
        noisy_action    (B, pred_horizon, action_dim)          — action corrupted with noise
                        action layout: [x, y, z, qx, qy, qz, qw]
                        where (x,y,z) is tcp_pose.position and (qx,qy,qz,qw) is tcp_pose.orientation
        timestep        (B,)  int                              — diffusion step index

    Output:
        noise_pred      (B, pred_horizon, action_dim)          — predicted noise

    Condition token sequence fed into cross-attention:
        obs_horizon × num_cameras  image tokens   (one per camera per obs step)
        obs_horizon                state tokens   (one per obs step)
        1                          task token     (one of 12 valid task combinations)
    """

    def __init__(
        self,
        obs_horizon: int = 2,
        action_dim: int = 7,            # 7-DOF TCP pose: 3 position (x,y,z) + 4 orientation (qx,qy,qz,qw)
        timestep_embed_dim: int = 128,
        task_embed_dim: int = 32,
        robot_state_dim: int = ROBOT_STATE_DIM,
        hidden_dim: int = 512,          # robot state encoder hidden size; state token dim = hidden_dim // 2
        # Diffusion Transformer hyperparameters
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        # Vision backbone
        resnet_name: str = "resnet18",
        resnet_weights="IMAGENET1K_V1",
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.action_dim  = action_dim

        # --- Image encoder (single shared ResNet18 for all three cameras) ---
        self.image_encoder = ImageEncoder(
            resnet_name=resnet_name,
            weights=resnet_weights,
        )
        # Produces (B, obs_horizon, num_cameras * RESNET_FEATURE_DIM)
        # We split this back to per-camera tokens inside forward()

        # --- Diffusion timestep encoder ---
        self.timestep_encoder = DiffusionTimestepEncoder(
            timestep_embed_dim=timestep_embed_dim
        )

        # --- Task encoder → single task token ---
        self.task_encoder = TaskEncoder(task_embed_dim=task_embed_dim)

        # --- Robot state encoder → one state token per obs step ---
        # Input layout (ROBOT_STATE_DIM=26):
        #   tcp_pose.position (3), tcp_pose.orientation (4),
        #   tcp_velocity.linear (3), tcp_velocity.angular (3),
        #   joint_positions (7), wrench.force (3), wrench.torque (3)
        # TODO: add input normalisation (mean/std from dataset stats)
        state_token_dim = hidden_dim // 2
        self.state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_token_dim),
        )

        # --- Diffusion Transformer (noise prediction network) ---
        self.noise_pred_net = DiffusionTransformer(
            action_dim=action_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            img_token_dim=RESNET_FEATURE_DIM,   # 512 — one token per camera per obs step
            state_token_dim=state_token_dim,
            task_token_dim=task_embed_dim,
            timestep_embed_dim=timestep_embed_dim,
            dropout=dropout,
        )

    def forward(
        self,
        images: dict,
        robot_state: torch.Tensor,
        target_module: torch.Tensor,
        port_name: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise given noisy action and all conditioning inputs.

        Args:
            images:        {camera_key: (B, obs_horizon, C, H, W)} — normalised to [0, 1]
            robot_state:   (B, obs_horizon, ROBOT_STATE_DIM)
            target_module: (B,) int64
            port_name:     (B,) int64
            noisy_action:  (B, pred_horizon, action_dim)
            timestep:      (B,) int64 diffusion step

        Returns:
            noise_pred: (B, pred_horizon, action_dim)
        """
        B = noisy_action.shape[0]
        num_cameras = len(ImageEncoder.CAMERA_KEYS)

        # --- Image tokens ---
        # image_encoder returns (B, obs_horizon, num_cameras * 512)
        img_feat = self.image_encoder(images)
        # Reshape to (B, obs_horizon * num_cameras, 512) — one token per camera per step
        img_tokens = img_feat.reshape(B, self.obs_horizon * num_cameras, RESNET_FEATURE_DIM)

        # --- State tokens: one per obs step ---
        # robot_state: (B, obs_horizon, ROBOT_STATE_DIM)
        state_tokens = self.state_encoder(
            robot_state.flatten(end_dim=1)                   # (B*obs_horizon, ROBOT_STATE_DIM)
        ).reshape(B, self.obs_horizon, -1)                   # (B, obs_horizon, state_token_dim)

        # --- Task token: single embedding for the combined task identity ---
        task_token = self.task_encoder(
            target_module, port_name
        )                                                    # (B, 1, task_embed_dim)

        # --- Diffusion timestep embedding ---
        timestep_emb = self.timestep_encoder(timestep)       # (B, timestep_embed_dim)

        # --- Predict noise via Diffusion Transformer ---
        return self.noise_pred_net(
            noisy_action=noisy_action,
            timestep_emb=timestep_emb,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            task_token=task_token,
        )                                                    # (B, pred_horizon, action_dim)
