#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs"
OUT_DIR="./output"
mkdir -p "$LOG_DIR" "$OUT_DIR"

LOG_FILE="$LOG_DIR/run_${TS}.log"

echo "[${TS}] Starting run" | tee -a "$LOG_FILE"
python3 run.py --config config.yaml --save-mode database --load-metadata --performance-report 2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}

if [[ $RC -ne 0 ]]; then
  echo "[${TS}] Run failed with exit code $RC" | tee -a "$LOG_FILE"
  exit $RC
fi

echo "[${TS}] Run completed successfully" | tee -a "$LOG_FILE"


