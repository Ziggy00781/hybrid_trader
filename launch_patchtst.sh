#!/usr/bin/env bash
set -euo pipefail

SESSION="patchtst"
LOGDIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOGFILE="$LOGDIR/train_$TIMESTAMP.log"

mkdir -p "$LOGDIR"

echo "[*] Starting PatchTST training in tmux session: $SESSION"
echo "[*] Log file: $LOGFILE"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[*] Existing session found. Killing it..."
    tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "
    echo '[*] Activating conda environment...' &&
    source ~/.bashrc &&
    conda activate trader &&
    echo '[*] Running training script...' &&
    python -m src.train.train_patchtst 2>&1 | tee \"$LOGFILE\"
"

echo "[*] Training started."
echo "Attach with: tmux attach -t $SESSION"
echo "Detach with: Ctrl+b then d"
echo "View logs:   tail -f $LOGFILE"