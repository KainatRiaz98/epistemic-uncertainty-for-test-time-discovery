#!/usr/bin/env bash
#
# greenland-config.sh — shared config for all Greenland scripts.
# ---------------------------------------------------------------
# Sourced by greenland-auth.sh, greenland-connect.sh, greenland-sync.sh.
# To switch instances, edit ONLY the INSTANCE-SPECIFIC block below
# (copy the values from the Greenland Console job JSON).

# ---- Account / role (stable across instances for this initiative) ----
ACCOUNT="703671891219"
CUSTOMER_ROLE="Intern"
PROVIDER="isengard"                 # conduit is denied for this alias; isengard works
PROFILE="greenland"
REGION="us-east-2"
JOB_ROLE_ARN="arn:aws:iam::072510399842:role/greenland-access-37f871283e3e69fdbfe97939a34079a8bfdfdd85"

# ---- INSTANCE-SPECIFIC (edit when you switch jobs) -------------------
# Job: cmohsinm-workspace (EKS: cmohsinm-workspace-74cc1422)
# 1x p4d.24xlarge, 8x A100, us-east-2. Initiative: KiroScienceInterns.
# JobId: 736e0cb8-0a72-4a77-8fd6-a86bed9e9fc8
SSM_TARGET="mi-002ae31a3cbbff0cc"   # SsmManagedInstanceId from job JSON
MAIN_NODE_IP="10.3.62.98"           # MainNodeIP / NodesEniHostIP
LOCAL_PORT="1048"                   # local SSH tunnel port (per-instance to avoid clashes)
# ----------------------------------------------------------------------

# ---- SSH / tunnel (stable) ----
REMOTE_PORT="2222"                  # sshd port inside the pod
SSH_USER="greenland-user"

# ---- Sync (rsync over the tunnel) ----
# Local project root -> remote workspace dir.
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_PROJECT_DIR="/home/greenland-user/epistemic-uncertainty-for-test-time-discovery"
