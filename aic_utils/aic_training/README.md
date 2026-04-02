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
ROS2 node that performs the full environment reset sequence:
1. Delete cable entity from Gazebo
2. Deactivate `aic_controller`
3. Reset robot joints to home via `/scoring/reset_joints`
4. Reactivate `aic_controller`
5. Tare force/torque sensor
6. Re-spawn cable at initial pose (relative to gripper)
7. Wait for robot to stabilize

Exposes a `/env/reset` service (`std_srvs/srv/Trigger`) that the training
loop can call.

**Must run inside the distrobox** (needs `simulation_interfaces`).

### `aic_gym_env.py`
Gymnasium environment that wraps the AIC robot controller. Runs outside
distrobox in the pixi environment. Calls `/env/reset` for resets.

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

```bash
distrobox enter -r aic_eval
source /ws_aic/install/setup.bash
python3 /path/to/aic_training/env_resetter.py
```

### 3. Run your training (in pixi env)

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
