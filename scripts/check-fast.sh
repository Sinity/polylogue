#!/usr/bin/env bash
# Cheap feedback; full behavioral/package qualification is an integration check.
set -euo pipefail
cargo fmt -p polylogue-model -p polylogue-query -p polylogue-sources \
    -p polylogue-engine -p polylogue-store -p polylogue -p xtask -- --check
cargo run --locked -p xtask -- boundaries
cargo check --locked --workspace --all-targets
