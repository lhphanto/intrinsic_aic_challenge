# aic_training - Environment Reset for RL Training

Provides a Gymnasium-compatible environment wrapper and reset utilities for
training policies in the AIC Gazebo simulation.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Training Loop (your machine / pixi env)             │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  AICGymEnv (Gymnasium wrapper)               │    │
│  │    .reset()  → calls EnvResetter via ROS2    │    │
│  │    .step()   → sends action, reads obs       │    │
│  └──────────────────────────────────────────────┘    │
│                         │ ROS2 topics/services       │
├─────────────────────────┼────────────────────────────┤
│  Simulation (distrobox / docker)                     │
│    - Gazebo (gz-sim)                                 │
│    - aic_controller                                  │
│    - /scoring/reset_joints service                   │
│    - /gz_server/spawn_entity & delete_entity         │
│    - /controller_manager/switch_controller           │
└──────────────────────────────────────────────────────┘
```

## Components

### `env_resetter.py`
ROS2 node that performs the full environment reset sequence. Exposes a
`/env/reset` service (`std_srvs/srv/SetBool`).

**Deterministic reset** (`request.data = False`, default):
1. Delete cable from Gazebo
2. Delete task board (only if `delete_task_board` ROS2 param is `True`)
3. Deactivate `aic_controller`
4. Reset robot joints to home via `/scoring/reset_joints`
5. Reactivate `aic_controller`
6. Tare force/torque sensor
7. Re-spawn cable at fixed pose (from ROS2 parameters)
8. Wait for robot to stabilize

**Random reset** (`request.data = True`, requires `config_path` param):
1. Delete cable from Gazebo
2. Delete task board
3. Deactivate `aic_controller`
4. Reset robot joints to home via `/scoring/reset_joints`
5. Reactivate `aic_controller`
6. Tare force/torque sensor
7. Re-spawn task board with randomly sampled trial scene:
   - Task board world pose (x, y, z, roll, pitch, yaw)
   - NIC card presence and position on each of 5 rails (Zone 1)
   - SC port presence and position on each of 2 rails (Zone 2)
   - LC/SFP/SC mount presence and position on each of 4 pick rails (Zones 3–4)
   - All translations clamped to `task_board_limits` from the config
8. Re-spawn cable with the trial's cable type, gripper offset, and orientation
9. Wait for robot to stabilize

The config must follow the structure of `aic_engine/config/sample_config.yaml`.

**Must run inside the distrobox** (needs `simulation_interfaces`).

**Must run inside the distrobox** (needs `simulation_interfaces`).

### `aic_gym_env.py`
Gymnasium environment that wraps the AIC robot controller. Runs outside
distrobox in the pixi environment. Calls `/env/reset` for resets.

The `AICGymEnvConfig` dataclass has a `random_reset` field (default `False`).
When `True`, each call to `env.reset()` sends `data=True` to the reset service,
triggering a randomly sampled trial configuration.

### `reset_env.sh`
Shell script to launch the resetter node inside distrobox.

## Usage

### 1. Start the simulation (in distrobox)

```bash
distrobox enter -r aic_eval
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.785 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.08 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false launch_rviz:=false
```

### 2. Start the reset service (in distrobox, separate terminal)

> **Note:** If `source /ws_aic/install/setup.bash` gives a "No such file or directory" error, the
> workspace hasn't been built yet. Run `colcon build --symlink-install` from `/ws_aic` first.
>
> **Note:** `export RMW_IMPLEMENTATION=rmw_zenoh_cpp` is required to fix a "service not available"
> issue that occurs with the default RMW implementation.

Deterministic reset (fixed cable pose):
```bash
distrobox enter -r aic_eval
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
source /ws_aic/install/setup.bash
/usr/bin/python3 /home/lhphanto/ws_aic/src/aic/aic_utils/aic_training/env_resetter.py
```

Random reset (requires a config file):
```bash
distrobox enter -r aic_eval
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
source /ws_aic/install/setup.bash
/usr/bin/python3 /home/lhphanto/ws_aic/src/aic/aic_utils/aic_training/env_resetter.py \
    --ros-args -p config_path:=/home/lhphanto/ws_aic/src/aic/aic_engine/config/sample_config.yaml
```

### 3. Run your training (in pixi env)

Deterministic reset (default):
```python
from aic_training.aic_gym_env import AICGymEnv

env = AICGymEnv()
obs, info = env.reset()

for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

Random reset (samples a trial from `sample_config.yaml` on each reset):
```python
from aic_training.aic_gym_env import AICGymEnv, AICGymEnvConfig

env = AICGymEnv(AICGymEnvConfig(random_reset=True))
obs, info = env.reset()

for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()  # each reset picks a new random trial
```
