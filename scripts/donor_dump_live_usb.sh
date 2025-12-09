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
#   sudo ./scripts/donor_dump_live_usb.sh <BDF> <BOARD> [DATASTORE_DIR] [--with-module]
# Example:
#   sudo ./scripts/donor_dump_live_usb.sh 0000:03:00.0 xupvvh /mnt/pcileech_datastore --with-module
#
# Notes:
#   - DATASTORE_DIR defaults to ./pcileech_datastore if not provided.
#   - Run from the repository root. The resulting datastore contains:
#       * device_context.json (PCI config space)
#       * msix_data.json (MSI-X parse)
#       * option_rom.bin + option_rom.json (if readable)
#       * vpd.bin (if readable)
#       * lspci-*.txt (diagnostic context)
#       * sysfs snapshots: resource(+resourceN), config.bin, ids, driver/iommu metadata
#       * donor_info.txt/json from donor_dump.ko when --with-module is used
#   - After collection, copy the datastore to your build machine and run the
#     normal build flow pointing --datastore at that directory.

set -euo pipefail

WITH_MODULE=0
if [[ "${4:-}" == "--with-module" || "${5:-}" == "--with-module" || "${WITH_MODULE:-0}" == "1" ]]; then
  WITH_MODULE=1
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: sudo $0 <BDF> <BOARD> [DATASTORE_DIR] [--with-module]" >&2
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

if [[ $WITH_MODULE -eq 1 ]]; then
  echo "[prep] Installing kernel build deps (headers, build-essential, dkms)..."
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    linux-headers-$(uname -r) build-essential dkms
fi

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
  # Copy individual BAR resource files if present (resource0, resource0_wc, etc.)
  for f in "$SYSFS_PATH"/resource*; do
    base=$(basename "$f")
    cp -f "$f" "$DATASTORE/$base" 2>/dev/null || true
  done

  # Raw config space dump (binary) for completeness
  cp -f "$SYSFS_PATH/config" "$DATASTORE/config.bin" 2>/dev/null || true

  # Basic ID files and metadata
  for f in vendor device subsystem_vendor subsystem_device class revision irq; do
    if [[ -r "$SYSFS_PATH/$f" ]]; then
      cp -f "$SYSFS_PATH/$f" "$DATASTORE/$f" 2>/dev/null || true
    fi
  done

  # Driver and IOMMU info
  {
    echo "driver: $(readlink -f "$SYSFS_PATH/driver" 2>/dev/null || echo none)"
    echo "iommu_group: $(basename $(readlink -f "$SYSFS_PATH/iommu_group" 2>/dev/null || echo none))"
    echo "modalias: $(cat "$SYSFS_PATH/modalias" 2>/dev/null || echo none)"
    echo "numa_node: $(cat "$SYSFS_PATH/numa_node" 2>/dev/null || echo none)"
  } > "$DATASTORE/sysfs_info.txt"

  lspci -vvnn -s "$BDF" > "$DATASTORE/lspci-vvnn.txt" 2>/dev/null || true
  lspci -k -s "$BDF"    > "$DATASTORE/lspci-k.txt"    2>/dev/null || true
  lspci -xxx -s "$BDF"  > "$DATASTORE/lspci-xxx.txt"  2>/dev/null || true
  lspci -xxxx -s "$BDF" > "$DATASTORE/lspci-xxxx.txt" 2>/dev/null || true
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

if [[ $WITH_MODULE -eq 1 ]]; then
  echo "[module] Building donor_dump.ko..."
  make -C src/donor_dump clean
  make -C src/donor_dump

  echo "[module] Loading donor_dump.ko for $BDF..."
  insmod src/donor_dump/donor_dump.ko bdf="$BDF"
  if [[ -r /proc/donor_dump ]]; then
    cat /proc/donor_dump > "$DATASTORE/donor_info.txt" || true
    echo "[module] Captured /proc/donor_dump to donor_info.txt"
    python - "$DATASTORE/donor_info.txt" "$DATASTORE/donor_info.json" <<'PY'
import json, sys
from pathlib import Path

raw = Path(sys.argv[1])
out = Path(sys.argv[2])
data = {}
for line in raw.read_text().splitlines():
    if ':' not in line:
        continue
    k, v = line.split(':', 1)
    data[k.strip()] = v.strip()
out.write_text(json.dumps(data, indent=2))
print(f"Wrote {out}")
PY
  else
    echo "[module] /proc/donor_dump not readable"
  fi

  echo "[module] Unloading donor_dump.ko..."
  rmmod donor_dump || true
fi

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
echo "              plus any captured: vpd.bin, option_rom.bin/json, lspci-*.txt, resource*, config.bin, ids"
echo "              donor_info.txt/json present if --with-module was used"
echo "       Transfer this directory to your build machine and run the full build there."
