#!/usr/bin/env bash
# The included common checkout owns both cloud setup backends and their pins.
set -euo pipefail
repo=$(realpath "$(dirname "$0")/../..")
common=${POLYLOGUE_SINEX_COMMON:-$(dirname "$repo")/polylogue-sinex-common}
if [ ! -f "$common/dev/cloud/setup.sh" ]; then
    echo 'cloud setup: include the polylogue-sinex-common repository in this environment' >&2
    exit 1
fi
exec bash "$common/dev/cloud/setup.sh" "$repo"
