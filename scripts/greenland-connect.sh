#!/usr/bin/env bash
#
# greenland-connect.sh — open the SSM tunnel and/or SSH into the SDB.
# -------------------------------------------------------------------
# Usage:
#   ./scripts/greenland-connect.sh           # auth + open SSM tunnel (default)
#   ./scripts/greenland-connect.sh auth      # auth only
#   ./scripts/greenland-connect.sh tunnel    # open SSM tunnel only (keep open)
#   ./scripts/greenland-connect.sh ssh       # SSH in (run in a 2nd terminal after tunnel)
#
# Credentials from `ada` are temporary — re-run `auth` before each session.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/greenland-config.sh"

auth() {
  "$(dirname "${BASH_SOURCE[0]}")/greenland-auth.sh"
}

tunnel() {
  echo ">> SSM port-forward localhost:$LOCAL_PORT -> $SSM_TARGET:$REMOTE_PORT (keep terminal open)..."
  aws ssm start-session \
    --target "$SSM_TARGET" \
    --document-name AWS-StartPortForwardingSession \
    --parameters "{\"portNumber\":[\"$REMOTE_PORT\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}" \
    --profile "$PROFILE" \
    --region "$REGION"
}

ssh_in() {
  echo ">> SSH to $SSH_USER@localhost:$LOCAL_PORT (tunnel must be running)..."
  ssh -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o ServerAliveInterval=60 \
      -p "$LOCAL_PORT" "$SSH_USER@localhost"
}

case "${1:-all}" in
  auth)   auth ;;
  tunnel) tunnel ;;
  ssh)    ssh_in ;;
  all)    auth; tunnel ;;
  *) echo "Usage: $0 [auth|tunnel|ssh|all]"; exit 1 ;;
esac
