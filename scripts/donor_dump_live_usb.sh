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
#   sudo ./scripts/donor_dump_live_usb.sh <BDF> <BOARD> [DATASTORE_DIR] [--with-module] [--nvme-extra]
# Example:
#   sudo ./scripts/donor_dump_live_usb.sh 0000:03:00.0 xupvvh /mnt/pcileech_datastore --with-module --nvme-extra
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
#       * NVMe extras (if --nvme-extra): nvme_id_ctrl.txt, nvme_list.txt, nvme_list_subsys.txt,
#         nvme_fw_log.txt, nvme_smart_log.txt, nvme_error_log.txt, nvme_id_ns_<nsid>.txt per namespace,
#         nvme_bar0.bin, nvme_doorbells.bin, nvme_regs.json, nvme_udev_info.txt, nvme_telemetry_host.bin (best effort)
#   - After collection, copy the datastore to your build machine and run the
#     normal build flow pointing --datastore at that directory.

set -euo pipefail

WITH_MODULE=0
NVME_EXTRA=0

if [[ $# -lt 2 ]]; then
  echo "Usage: sudo $0 <BDF> <BOARD> [DATASTORE_DIR] [--with-module] [--nvme-extra]" >&2
  exit 1
fi

BDF="$1"
BOARD="$2"
DATASTORE="${3:-pcileech_datastore}"

# Flag scan (allows flags in any position after required args)
for arg in "$@"; do
  case "$arg" in
    --with-module) WITH_MODULE=1 ;;
    --nvme-extra)  NVME_EXTRA=1 ;;
  esac
done

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

if [[ $NVME_EXTRA -eq 1 ]]; then
  echo "[prep] Installing nvme-cli for NVMe captures..."
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nvme-cli
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

  if [[ $NVME_EXTRA -eq 1 ]]; then
    echo "[nvme] NVMe extra capture enabled"
    if [[ -d "$SYSFS_PATH" ]]; then
      CLASS_HEX=$(cat "$SYSFS_PATH/class" 2>/dev/null || echo "")
      if [[ "$CLASS_HEX" =~ 010802 || "$CLASS_HEX" =~ 0x010802 ]]; then
        NVME_CTRL_PATH=$(ls -d "$SYSFS_PATH"/nvme/nvme* 2>/dev/null | head -n1 || true)
        NVME_DEV=""
        if [[ -n "$NVME_CTRL_PATH" ]]; then
          NVME_DEV=$(basename "$NVME_CTRL_PATH")
        fi

        if [[ -n "$NVME_DEV" && -e "/dev/$NVME_DEV" ]]; then
          echo "  NOTE: Ensure this NVMe is not the boot/system drive before proceeding."
          echo "  NVMe controller: /dev/$NVME_DEV"
          nvme list -v               > "$DATASTORE/nvme_list.txt"            2>/dev/null || true
          nvme list-subsys           > "$DATASTORE/nvme_list_subsys.txt"     2>/dev/null || true
          nvme id-ctrl "/dev/$NVME_DEV" > "$DATASTORE/nvme_id_ctrl.txt"     2>/dev/null || true
          nvme fw-log "/dev/$NVME_DEV"  > "$DATASTORE/nvme_fw_log.txt"      2>/dev/null || true
          nvme smart-log "/dev/$NVME_DEV" > "$DATASTORE/nvme_smart_log.txt" 2>/dev/null || true
          nvme error-log "/dev/$NVME_DEV" --entries=64 > "$DATASTORE/nvme_error_log.txt" 2>/dev/null || true
          nvme telemetry-log "/dev/$NVME_DEV" --host --output-file "$DATASTORE/nvme_telemetry_host.bin" 2>/dev/null || true

          if udevadm info --query=all --path="/sys/block/${NVME_DEV}" > "$DATASTORE/nvme_udev_info.txt" 2>/dev/null; then
            true
          fi

          for nsdev in /sys/class/nvme/${NVME_DEV}n*; do
            [[ -e "$nsdev" ]] || continue
            nsname=$(basename "$nsdev")
            nsid=${nsname#${NVME_DEV}n}
            [[ -n "$nsid" ]] || continue
            nvme id-ns "/dev/$NVME_DEV" -n "$nsid" > "$DATASTORE/nvme_id_ns_${nsid}.txt" 2>/dev/null || true
          done

          if [[ -r "$SYSFS_PATH/resource0" ]]; then
            echo "  Capturing first 8KB of BAR0 to nvme_bar0.bin"
            dd if="$SYSFS_PATH/resource0" of="$DATASTORE/nvme_bar0.bin" bs=4K count=2 status=none 2>/dev/null || true
            echo "  Capturing 4KB doorbell region (offset 0x1000) to nvme_doorbells.bin"
            dd if="$SYSFS_PATH/resource0" of="$DATASTORE/nvme_doorbells.bin" bs=4K skip=1 count=1 status=none 2>/dev/null || true
            python - "$DATASTORE/nvme_bar0.bin" "$DATASTORE/nvme_regs.json" <<'PY'
      import json, struct, sys
      from pathlib import Path

      bar_path = Path(sys.argv[1])
      out_path = Path(sys.argv[2])
      if not bar_path.exists():
        sys.exit(0)
      data = bar_path.read_bytes()

      def le32(off):
        if off + 4 > len(data):
          return None
        return struct.unpack_from('<I', data, off)[0]

      def le64(off):
        if off + 8 > len(data):
          return None
        return struct.unpack_from('<Q', data, off)[0]

      cap = le64(0)
      vs = le32(8)
      cc = le32(0x14)
      csts = le32(0x1c)
      aqa = le32(0x24)
      asq = le64(0x28)
      acq = le64(0x30)
      cmbloc = le32(0x38)
      cmbsz = le32(0x3c)
      pmrcap = le32(0xE0)
      pmrctl = le32(0xE4)
      pmrsts = le32(0xE8)

      def cap_fields(val):
        if val is None:
          return {}
        return {
          "mqes": (val & 0xFFFF) + 1,
          "cqr": (val >> 16) & 0x1,
          "ams": (val >> 17) & 0x3,
          "to_usec": ((val >> 24) & 0xFF) * 500,
          "dstrd": (val >> 32) & 0xF,
          "css": (val >> 37) & 0xFF,
          "mpsmin": (val >> 48) & 0xF,
          "mpsmax": (val >> 52) & 0xF,
          "pmrs": (val >> 56) & 0x1,
          "cmbs": (val >> 57) & 0x1,
        }

      def cc_fields(val):
        if val is None:
          return {}
        return {
          "en": val & 0x1,
          "css": (val >> 4) & 0x7,
          "mps": (val >> 7) & 0xF,
          "ams": (val >> 11) & 0x7,
          "shn": (val >> 14) & 0x3,
          "iosqes": (val >> 16) & 0xF,
          "iocqes": (val >> 20) & 0xF,
        }

      def csts_fields(val):
        if val is None:
          return {}
        return {
          "rdy": val & 0x1,
          "cfs": (val >> 1) & 0x1,
          "shst": (val >> 2) & 0x3,
        }

      payload = {
        "raw": {
          "cap": f"0x{cap:016x}" if cap is not None else None,
          "vs": f"0x{vs:08x}" if vs is not None else None,
          "cc": f"0x{cc:08x}" if cc is not None else None,
          "csts": f"0x{csts:08x}" if csts is not None else None,
          "aqa": f"0x{aqa:08x}" if aqa is not None else None,
          "asq": f"0x{asq:016x}" if asq is not None else None,
          "acq": f"0x{acq:016x}" if acq is not None else None,
          "cmbloc": f"0x{cmbloc:08x}" if cmbloc is not None else None,
          "cmbsz": f"0x{cmbsz:08x}" if cmbsz is not None else None,
          "pmrcap": f"0x{pmrcap:08x}" if pmrcap is not None else None,
          "pmrctl": f"0x{pmrctl:08x}" if pmrctl is not None else None,
          "pmrsts": f"0x{pmrsts:08x}" if pmrsts is not None else None,
        },
        "parsed": {
          "cap": cap_fields(cap),
          "cc": cc_fields(cc),
          "csts": csts_fields(csts),
          "aq_depths": {
            "admin_sq_entries": ((aqa or 0) & 0xFFF) + 1 if aqa is not None else None,
            "admin_cq_entries": (((aqa or 0) >> 16) & 0xFFF) + 1 if aqa is not None else None,
          },
          "asq": asq,
          "acq": acq,
          "doorbell_stride_bytes": 4 * (1 << cap_fields(cap).get("dstrd", 0)) if cap is not None else None,
        },
      }

      out_path.write_text(json.dumps(payload, indent=2))
      print(f"Wrote {out_path}")
      PY
          else
            echo "  BAR0 not readable; skipping nvme_bar0.bin"
          fi
        else
          echo "  NVMe controller node not found under $SYSFS_PATH; skipping nvme extra." >&2
        fi
      else
        echo "  Device class is not NVMe (010802); skipping nvme extra." >&2
      fi
    else
      echo "  WARNING: $SYSFS_PATH not found; NVMe extra skipped." >&2
    fi
  fi

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
if [[ $NVME_EXTRA -eq 1 ]]; then
  echo "              nvme_* captures present (identify, logs, BAR0/doorbells, regs, telemetry where available)"
fi
echo "       Transfer this directory to your build machine and run the full build there."
