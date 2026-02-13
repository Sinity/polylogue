#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_DIR="/tmp/polylogue-demo"
ENV_FILE="$DEMO_DIR/.env"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Generate animated terminal screencasts for polylogue demos.

Options:
    --synthetic     Use synthetic fixtures (no real exports needed)
    --fixtures-dir  Custom fixtures directory (default: tests/fixtures/real/)
    --tape NAME     Run only a specific tape (e.g., 01-overview)
    --seed-only     Only seed the demo database, don't record tapes
    --skip-seed     Skip seeding, use existing demo database
    --verbose       Verbose output
    -h, --help      Show this help

Examples:
    # Full generation with synthetic data
    ./demos/generate.sh --synthetic

    # Re-record a single tape (uses existing seeded data)
    ./demos/generate.sh --skip-seed --tape 03-search

    # Seed with real data, then record all tapes
    ./demos/generate.sh --fixtures-dir tests/fixtures/real/
EOF
}

SYNTHETIC=""
FIXTURES_DIR=""
TAPE_FILTER=""
SEED_ONLY=false
SKIP_SEED=false
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --synthetic) SYNTHETIC="--synthetic"; shift ;;
        --fixtures-dir) FIXTURES_DIR="--fixtures-dir $2"; shift 2 ;;
        --tape) TAPE_FILTER="$2"; shift 2 ;;
        --seed-only) SEED_ONLY=true; shift ;;
        --skip-seed) SKIP_SEED=true; shift ;;
        --verbose|-v) VERBOSE="-v"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ── Step 1: Seed demo database ──────────────────────────────────────────
if [ "$SKIP_SEED" = false ]; then
    echo "==> Seeding demo database..."
    mkdir -p "$DEMO_DIR"

    # Capture env vars from seed script
    ENV_OUTPUT=$(cd "$PROJECT_ROOT" && python scripts/seed_demo.py \
        --output-dir "$DEMO_DIR" \
        --env-only \
        $SYNTHETIC $FIXTURES_DIR $VERBOSE)

    # Write env file for tape scripts
    echo "$ENV_OUTPUT" > "$ENV_FILE"
    echo "==> Demo environment seeded at $DEMO_DIR"

    if [ "$SEED_ONLY" = true ]; then
        echo ""
        echo "To use this environment:"
        echo "  source $ENV_FILE"
        echo "  polylogue  # shows demo stats"
        exit 0
    fi
else
    if [ ! -f "$ENV_FILE" ]; then
        echo "Error: No demo environment found at $ENV_FILE"
        echo "Run without --skip-seed first to create it."
        exit 1
    fi
    echo "==> Using existing demo environment at $DEMO_DIR"
fi

# ── Step 2: Verify VHS is available ─────────────────────────────────────
if ! command -v vhs &>/dev/null; then
    echo "Error: vhs not found. Install it via:"
    echo "  nix develop  # if using nix (vhs is in devShell)"
    echo "  brew install vhs  # macOS"
    echo "  go install github.com/charmbracelet/vhs@latest"
    exit 1
fi

# ── Step 3: Record tapes ────────────────────────────────────────────────
TAPE_DIR="$SCRIPT_DIR/tapes"
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

TAPES=()
if [ -n "$TAPE_FILTER" ]; then
    TAPE_FILE="$TAPE_DIR/${TAPE_FILTER}.tape"
    if [ ! -f "$TAPE_FILE" ]; then
        echo "Error: Tape not found: $TAPE_FILE"
        echo "Available tapes:"
        ls "$TAPE_DIR"/*.tape 2>/dev/null | xargs -I{} basename {} .tape
        exit 1
    fi
    TAPES=("$TAPE_FILE")
else
    for tape in "$TAPE_DIR"/*.tape; do
        [ -f "$tape" ] && TAPES+=("$tape")
    done
fi

if [ ${#TAPES[@]} -eq 0 ]; then
    echo "No tape files found in $TAPE_DIR"
    exit 1
fi

echo "==> Recording ${#TAPES[@]} tape(s)..."
for tape in "${TAPES[@]}"; do
    name=$(basename "$tape" .tape)
    echo "  Recording: $name"
    # Run VHS from project root so relative output paths work
    (cd "$PROJECT_ROOT" && vhs "$tape") || {
        echo "  Warning: Failed to record $name (continuing...)"
    }
done

# ── Step 4: Copy key GIFs for README ────────────────────────────────────
ASSETS_DIR="$PROJECT_ROOT/docs/assets"
mkdir -p "$ASSETS_DIR"

# Copy the overview GIF as the hero demo
if [ -f "$OUTPUT_DIR/01-overview.gif" ]; then
    cp "$OUTPUT_DIR/01-overview.gif" "$ASSETS_DIR/demo-overview.gif"
fi

echo ""
echo "==> Done! Generated GIFs:"
ls -lh "$OUTPUT_DIR"/*.gif 2>/dev/null || echo "  (no GIFs generated — check VHS output above)"
echo ""
echo "GIFs are in: $OUTPUT_DIR/"
echo "README assets copied to: $ASSETS_DIR/"
