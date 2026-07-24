#!/usr/bin/env bash
#
# greenland-auth.sh — Run from your LOCAL LAPTOP, daily.
# -----------------------------------------------------
# Authenticates with Midway/Isengard and sets up the 'greenland' AWS profile.
# Credentials from `ada` are temporary — re-run before each session.
#
# Usage:  ./scripts/greenland-auth.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/greenland-config.sh"

echo "============================================"
echo " Greenland Daily Auth (Local Laptop)"
echo "============================================"

echo "[1/4] Refreshing Midway credentials..."
mwinit -f
echo "  ✓ Midway OK"

echo "[2/4] Assuming role '$CUSTOMER_ROLE' on account $ACCOUNT (provider: $PROVIDER)..."
ada credentials update \
  --account "$ACCOUNT" \
  --role "$CUSTOMER_ROLE" \
  --provider "$PROVIDER" \
  --once
echo "  ✓ Credentials updated in default profile"

echo "[3/4] Configuring '$PROFILE' AWS profile..."
aws configure set --profile "$PROFILE" source_profile default
aws configure set --profile "$PROFILE" region "$REGION"
aws configure set --profile "$PROFILE" role_arn "$JOB_ROLE_ARN"
echo "  ✓ Profile '$PROFILE' configured"

echo "[4/4] Verifying auth chain (default -> greenland job role)..."
aws sts get-caller-identity --profile "$PROFILE"

echo "============================================"
echo " ✅ Auth complete."
echo "    Next: ./scripts/greenland-connect.sh tunnel"
echo "============================================"
