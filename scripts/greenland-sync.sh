#!/usr/bin/env bash
#
# greenland-sync.sh — rsync the project to/from the SDB over the SSM tunnel.
# --------------------------------------------------------------------------
# The tunnel (greenland-connect.sh tunnel) must be running in another terminal.
#
# Usage:
#   ./scripts/greenland-sync.sh up      # local -> remote (push code)  [default]
#   ./scripts/greenland-sync.sh down    # remote -> local (pull results/figures)
#   ./scripts/greenland-sync.sh up --dry-run
#
# Excludes caches, git, adapters, logs, and large weights (re-created/downloaded
# on the remote). Mirrors the .gitignore intent.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/greenland-config.sh"

SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $LOCAL_PORT"

EXCLUDES=(
  --exclude ".git/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude ".pytest_cache/"
  --exclude ".venv/"
  --exclude "venv/"
  --exclude "logs/"
  --exclude "adapters/"
  --exclude "wandb/"
  --exclude "*.pt"
  --exclude "*.pth"
  --exclude "*.safetensors"
  --exclude ".DS_Store"
)

DIRECTION="${1:-up}"
shift || true
EXTRA_ARGS=("$@")   # e.g. --dry-run

case "$DIRECTION" in
  up)
    echo ">> Pushing $LOCAL_PROJECT_DIR/ -> $SSH_USER@remote:$REMOTE_PROJECT_DIR/"
    $SSH_CMD "$SSH_USER@localhost" "mkdir -p '$REMOTE_PROJECT_DIR'"
    rsync -avz --progress "${EXCLUDES[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
      -e "$SSH_CMD" \
      "$LOCAL_PROJECT_DIR/" \
      "$SSH_USER@localhost:$REMOTE_PROJECT_DIR/"
    ;;
  down)
    echo ">> Pulling $SSH_USER@remote:$REMOTE_PROJECT_DIR/ -> $LOCAL_PROJECT_DIR/"
    rsync -avz --progress "${EXCLUDES[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
      -e "$SSH_CMD" \
      "$SSH_USER@localhost:$REMOTE_PROJECT_DIR/" \
      "$LOCAL_PROJECT_DIR/"
    ;;
  *)
    echo "Usage: $0 [up|down] [rsync-args...]"; exit 1 ;;
esac

echo ">> Sync ($DIRECTION) done."
