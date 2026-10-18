#!/usr/bin/env bash
# Entry point for older pre-push hooks; the fast check lives in xtask.
exec cargo run --locked -q -p xtask -- fast
