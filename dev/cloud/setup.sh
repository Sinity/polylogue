#!/usr/bin/env bash
# Prepares a cloud VM (Claude Code on the web, Codex cloud) to build and test
# this repository: the Nix development shell for every later command, Cargo's
# sources, and the link to a sibling polylogue-sinex-common checkout when the
# session has one. It requires root permissions during environment setup while
# the network is open, and is safe to run again. Claude Code clones the
# repository to /home/user/polylogue; Codex starts in the checkout:
#
#     bash /home/user/polylogue/dev/cloud/setup.sh    # Claude Code
#     bash dev/cloud/setup.sh                   # Codex
#
# CLOUD_PREBUILD=1 also builds every test binary, for platforms whose cached
# setup has time for it (Codex).
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v nix >/dev/null 2>&1; then
    # Ubuntu's package comes from mirrors every network preset allows.
    apt-get update -qq
    apt-get install -y -qq nix-bin
    mkdir -p /etc/nix
    printf '%s\n' 'build-users-group =' 'sandbox = false' >>/etc/nix/nix.conf
fi
nix=(nix --extra-experimental-features 'nix-command flakes')

# Every later shell enters the development shell: non-interactive ones through
# BASH_ENV=/etc/devenv.sh in the environment's variables, interactive ones
# through the first line of root's .bashrc, before its non-interactive return.
"${nix[@]}" print-dev-env .#default >/etc/devenv.sh.new
mv /etc/devenv.sh.new /etc/devenv.sh
entry='[ -f /etc/devenv.sh ] && . /etc/devenv.sh'
if ! grep -qxF "$entry" /root/.bashrc 2>/dev/null; then
    { echo "$entry"; cat /root/.bashrc 2>/dev/null || true; } >/root/.bashrc.new
    mv /root/.bashrc.new /root/.bashrc
fi

# The development shell's hook links a sibling polylogue-sinex-common checkout.
set +u
. /etc/devenv.sh
set -u

# Polylogue's Git dependencies come from polylogue-sinex-common and Sinex; a
# session without access to both cannot fetch them.
if ! cargo fetch --locked; then
    echo "setup: cargo fetch failed; a Polylogue session needs the polylogue-sinex-common and sinex repositories" >&2
    exit 1
fi

if [ "${CLOUD_PREBUILD:-0}" = 1 ]; then
    cargo xtask test --no-run
fi
