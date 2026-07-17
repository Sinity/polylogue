#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 /path/to/polylogue /path/to/sinex /path/to/swarm-root" >&2
  exit 2
fi

POLYLOGUE=$(realpath "$1")
SINEX=$(realpath "$2")
SWARM=$(realpath -m "$3")
mkdir -p "$SWARM"/{polylogue,sinex,out/{locks,status,contracts,evidence,patches,validation}}

add_lane() {
  local repo=$1 branch=$2 path=$3
  if [[ -e "$path/.git" || -f "$path/.git" ]]; then
    printf 'exists: %s\n' "$path"
    return
  fi
  git -C "$repo" worktree add -b "$branch" "$path" master
}

add_lane "$POLYLOGUE" launch/integration "$SWARM/polylogue-integration"
add_lane "$SINEX" launch/integration "$SWARM/sinex-integration"

for lane in narrative docs claims receipts-fixture tour semantic-contract terminal-renderer web-renderer web-reliability install-media; do
  add_lane "$POLYLOGUE" "launch/$lane" "$SWARM/polylogue/$lane"
done

for lane in narrative demo-contract moment-demo replay-demo outage-demo; do
  add_lane "$SINEX" "launch/$lane" "$SWARM/sinex/$lane"
done

cat > "$SWARM/out/README.txt" <<EOF
Swarm root: $SWARM
Integration branches:
  $SWARM/polylogue-integration
  $SWARM/sinex-integration
Coordination lives under $SWARM/out and must not be committed accidentally.
EOF

printf 'created worktree swarm under %s\n' "$SWARM"
