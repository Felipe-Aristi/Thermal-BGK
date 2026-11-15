
#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# TBGK - Thermal BGK Lattice Boltzmann
#
# Compile and run the D2V37 Rayleigh-Benard simulation.
# ============================================================

# Project root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ============================================================
# Compilation configuration
# ============================================================

NVCC="${NVCC:-nvcc}"

ARCH="sm_89"
STD="c++20"

SOURCE="src/main.cu"
EXECUTABLE="tbgk"

# ============================================================
# Compile
# ============================================================

echo "============================================"
echo "           Compiling TBGK"
echo "============================================"

"$NVCC" \
    -O3 \
    -std="$STD" \
    -arch="$ARCH" \
    --expt-extended-lambda \
    -lineinfo \
    "$SOURCE" \
    -o "$EXECUTABLE"

echo "Compilation successful."

# ============================================================
# Execute
# ============================================================

echo ""
echo "============================================"
echo "             Running TBGK"
echo "============================================"

./"$EXECUTABLE"

echo ""
echo "Simulation completed."
