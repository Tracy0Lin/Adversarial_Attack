#!/usr/bin/env bash
set -euo pipefail
BLIP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
/workspace/projects/couser/environments/attackvlm/bin/python "$BLIP_SCRIPT_DIR/export_detector_models.py"
/workspace/projects/couser/environments/attackvlm/bin/python "$BLIP_SCRIPT_DIR/package_prepared.py"
