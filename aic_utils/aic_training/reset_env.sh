#!/bin/bash
# Launch the environment resetter node inside the distrobox container.
#
# Usage:
#   ./reset_env.sh                     # default config
#   ./reset_env.sh --cable_type sfp_sc_cable_reversed
#
# This script must be run from within the distrobox (distrobox enter -r aic_eval)
# OR it will automatically enter the distrobox.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESETTER_SCRIPT="$SCRIPT_DIR/env_resetter.py"

# Check if we're inside the distrobox by testing for simulation_interfaces
if python3 -c "import simulation_interfaces" 2>/dev/null; then
    echo "Running inside distrobox environment."
    source /ws_aic/install/setup.bash 2>/dev/null || true
    exec python3 "$RESETTER_SCRIPT" --ros-args -p use_sim_time:=true "$@"
else
    echo "Not inside distrobox. Attempting to enter aic_eval distrobox..."
    exec distrobox enter -r aic_eval -- bash -c "
        source /ws_aic/install/setup.bash 2>/dev/null || true
        python3 '$RESETTER_SCRIPT' --ros-args -p use_sim_time:=true $*
    "
fi
