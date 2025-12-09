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
#   - Run from the repository root. The resulting datastore contains:
#       * device_context.json (PCI config space)
#       * msix_data.json (MSI-X parse)
#       * option_rom.bin + option_rom.json (if readable)
#       * vpd.bin (if readable)
#       * lspci-*.txt (diagnostic context)
#       * resource file copy from sysfs (BAR layout)
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

# Paths
SYSFS_PATH="/sys/bus/pci/devices/$BDF"

echo "[collect] Capturing sysfs metadata snapshot..."
if [[ -d "$SYSFS_PATH" ]]; then
  cp -f "$SYSFS_PATH/resource" "$DATASTORE/resource" 2>/dev/null || true
  lspci -vvnn -s "$BDF" > "$DATASTORE/lspci-vvnn.txt" 2>/dev/null || true
  lspci -k -s "$BDF"    > "$DATASTORE/lspci-k.txt"    2>/dev/null || true
  lspci -xxx -s "$BDF"  > "$DATASTORE/lspci-xxx.txt"  2>/dev/null || true
else
  echo "  WARNING: $SYSFS_PATH not found; device may be absent." >&2
fi

echo "[collect] Running host collection for BDF=$BDF → $DATASTORE"
python pcileech.py build \
  --bdf "$BDF" \
  --board "$BOARD" \
  --host-collect-only \
  --datastore "$DATASTORE" \
  --container-mode local

# Optional: capture VPD if available
echo "[collect] Capturing VPD (if exposed)..."
if [[ -r "$SYSFS_PATH/vpd" ]]; then
  cat "$SYSFS_PATH/vpd" > "$DATASTORE/vpd.bin" 2>/dev/null || true
  echo "  VPD captured to vpd.bin"
else
  echo "  VPD not readable or not present."
fi

# Optional: capture Option ROM if available
echo "[collect] Capturing Option ROM (if exposed)..."
if [[ -w "$SYSFS_PATH/rom" ]]; then
  ROM_PATH="$SYSFS_PATH/rom"
  # Enable ROM read
  echo 1 > "$ROM_PATH" 2>/dev/null || true
  if dd if="$ROM_PATH" of="$DATASTORE/option_rom.bin" status=none bs=4K 2>/dev/null; then
    echo 0 > "$ROM_PATH" 2>/dev/null || true
    # Record metadata
    ROM_SIZE=$(stat -c%s "$DATASTORE/option_rom.bin" 2>/dev/null || echo 0)
    ROM_SHA256=$(sha256sum "$DATASTORE/option_rom.bin" 2>/dev/null | awk '{print $1}')
    cat > "$DATASTORE/option_rom.json" <<EOF
{
  "size_bytes": $ROM_SIZE,
  "sha256": "${ROM_SHA256:-unknown}",
  "source": "$ROM_PATH"
}
EOF
    echo "  Option ROM captured (${ROM_SIZE} bytes)."
  else
    echo 0 > "$ROM_PATH" 2>/dev/null || true
    echo "  Failed to read Option ROM (device or kernel may block access)."
  fi
else
  echo "  Option ROM not exposed or not writable."
fi

echo "[done] Datastore ready at: $(realpath "$DATASTORE")"
echo "       Files: device_context.json, msix_data.json"
echo "              plus any captured: vpd.bin, option_rom.bin/json, lspci-*.txt, resource"
echo "       Transfer this directory to your build machine and run the full build there."
