#!/usr/bin/env bash

# donor_dump_live_usb.sh
# Collect donor PCIe device data on a bare-metal Ubuntu Live USB and store it
# in a portable datastore directory for later templating/build on another
# Linux machine with Vivado installed.
#
# Preconditions (you handle outside this script):
#   - Booted bare-metal Ubuntu with persistence (so packages/venv survive).
#   - Secure Boot disabled or set to trust custom keys so vfio-pci can load.
#   - IOMMU enabled in kernel cmdline: intel_iommu=on iommu=pt OR amd_iommu=on iommu=pt.
#   - Target device present and not passed through to another driver/VM.
#
# Usage:
#   sudo ./scripts/donor_dump_live_usb.sh <BDF> <BOARD> [DATASTORE_DIR]
# Example:
#   sudo ./scripts/donor_dump_live_usb.sh 0000:03:00.0 xupvvh /mnt/pcileech_datastore
#
# Notes:
#   - DATASTORE_DIR defaults to ./pcileech_datastore if not provided.
#   - Run from the repository root. The resulting datastore contains
#     device_context.json and msix_data.json.
#   - After collection, copy the datastore to your build machine and run the
#     normal build flow pointing --datastore at that directory.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sudo $0 <BDF> <BOARD> [DATASTORE_DIR]" >&2
  exit 1
fi

BDF="$1"
BOARD="$2"
DATASTORE="${3:-pcileech_datastore}"

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run as root (needed for PCI config access)." >&2
  exit 1
fi

if [[ ! -d .git || ! -f pcileech.py ]]; then
  echo "Please run this script from the PCILeechFWGenerator repository root." >&2
  exit 1
fi

echo "[prep] Installing minimal dependencies (python3-venv, pciutils)..."
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip pciutils git

echo "[prep] Checking for IOMMU kernel flags (informational)..."
if grep -Eq "(intel|amd)_iommu=on" /proc/cmdline; then
  echo "  IOMMU flag detected in /proc/cmdline."
else
  echo "  WARNING: IOMMU not detected in /proc/cmdline (intel_iommu=on or amd_iommu=on)." >&2
  echo "           Enable it and reboot if VFIO binding fails." >&2
fi

echo "[prep] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "[prep] Installing Python requirements (may take a moment)..."
pip install -r requirements.txt

mkdir -p "$DATASTORE"

echo "[collect] Running host collection for BDF=$BDF → $DATASTORE"
python pcileech.py build \
  --bdf "$BDF" \
  --board "$BOARD" \
  --host-collect-only \
  --datastore "$DATASTORE" \
  --container-mode local

echo "[done] Datastore ready at: $(realpath "$DATASTORE")"
echo "       Files: device_context.json, msix_data.json"
echo "       Transfer this directory to your build machine and run the full build there."
