#!/usr/bin/env bash
# Push results_tables.tex to Overleaf project 69e4f728bc5a19d240051233
# Usage: bash scripts/push_to_overleaf.sh
# You will be prompted for your Overleaf password.

set -euo pipefail

OVERLEAF_EMAIL="shantipriya.parida@gmail.com"
OVERLEAF_PROJECT_ID="69e4f728bc5a19d240051233"
OVERLEAF_GIT_URL="https://git.overleaf.com/${OVERLEAF_PROJECT_ID}"
CLONE_DIR="/tmp/overleaf_icpr_${OVERLEAF_PROJECT_ID}"
TABLES_SRC="$(cd "$(dirname "$0")/.." && pwd)/paper/results_tables.tex"

echo "=== Overleaf push for project ${OVERLEAF_PROJECT_ID} ==="
echo "Authenticated as: ${OVERLEAF_EMAIL}"
echo ""

# Clone (or pull if already cloned)
if [ -d "${CLONE_DIR}/.git" ]; then
  echo "[1/4] Pulling latest from Overleaf..."
  git -C "${CLONE_DIR}" pull
else
  echo "[1/4] Cloning Overleaf project (enter your Overleaf password when prompted)..."
  git clone "https://${OVERLEAF_EMAIL}@git.overleaf.com/${OVERLEAF_PROJECT_ID}" "${CLONE_DIR}"
fi

echo "[2/4] Copying results_tables.tex..."
cp "${TABLES_SRC}" "${CLONE_DIR}/results_tables.tex"

echo "[3/4] Committing..."
git -C "${CLONE_DIR}" add results_tables.tex
git -C "${CLONE_DIR}" diff --cached --stat
git -C "${CLONE_DIR}" commit -m "Update results tables: real IndicSentiment numbers + clean layout" || echo "(nothing new to commit)"

echo "[4/4] Pushing to Overleaf (enter password again if prompted)..."
git -C "${CLONE_DIR}" push

echo ""
echo "=== Done! Open https://www.overleaf.com/project/${OVERLEAF_PROJECT_ID} to verify ==="
echo ""
echo "To include the tables in your main .tex file, add:"
echo "  \\\\input{results_tables}"
