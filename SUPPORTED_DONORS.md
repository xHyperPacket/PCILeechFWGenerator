# Supported Donor Devices (Practical Picks)

This guide lists donor devices that tend to work well with the PCILeech Firmware Generator. It emphasizes availability, clean PCIe behavior, and ease of VFIO/IOMMU isolation. Use add-in cards in a desktop/bench system whenever possible; avoid onboard/boot-critical hardware.

## Quick Top Picks (easy mode)
- Realtek RTL8111/8168/8411 NIC — ultra-common, cheap, simple PCIe layout.
- Intel 82574L / I210 / I350 NICs — well-documented, stable MSI/MSI-X.
- ASMedia or Renesas-based PCIe USB 3.0/3.1 cards — straightforward bridges.
- Elgato or AVerMedia capture cards — accessible, moderately simple BARs.
- LSI/Broadcom HBAs in IT mode (add-in card) — predictable storage-class donor (not your boot disk).

## Network Interface Cards (NICs)
**Why NICs?** Simple PCIe endpoints, standard capabilities, reliable MSI/MSI-X, and abundant supply.
- Realtek RTL8111/8168/8411 (GbE) — plentiful; great for first-time dumps.
- Intel 82574L, 82579LM — mature silicon, stable config, well-documented.
- Intel I210, I350 families — server-grade, clean capability chains.
- Broadcom NetXtreme (various) — generally well-behaved; check IOMMU grouping.
- Mellanox ConnectX-3 / ConnectX-4 — works but more complex; use if experienced.

## Audio Interfaces
**Why audio?** Small BARs, simple capability chains, easy to isolate.
- Creative Sound Blaster (multiple generations) — accessible, well-known layouts.
- ASUS Xonar series — stable PCIe behavior.
- M-Audio interfaces — straightforward register maps.
- Generic PCIe-to-USB audio bridges — minimal PCIe complexity.

## Video / Capture Cards
**Why capture?** Diverse but mostly standard endpoints; good for varied BAR/MSI-X examples.
- AVerMedia Live Gamer series — common, moderate complexity.
- Elgato capture cards — popular, consistent behavior.
- Hauppauge WinTV series — TV-tuner style, usually simple BARs.
- Blackmagic DeckLink series — professional; richer capabilities but solid.

## USB / IO Controllers
**Why USB bridges?** Clean PCIe-to-USB translation, small BARs, easy dumps.
- ASMedia PCIe USB 3.0/3.1 controllers — very common, easy MSI/MSI-X.
- Renesas/NEC PCIe USB controllers — stable, well-supported.
- PCIe Serial/Parallel port cards — tiny BARs, minimal capabilities.
- GPIO / Digital I/O cards — industrial/measurement cards with simple maps.

## Storage Controllers (add-in preferred)
**Why cautious?** Storage can be critical; use non-boot add-in hardware only.
- LSI/Broadcom HBAs (IT mode) — predictable registers; great for storage-class patterns.
- Add-in SATA/AHCI controllers — simple endpoints when on a card.
- Secondary NVMe on a PCIe adapter (NOT your boot/system NVMe) — ensure its own IOMMU group.

## Other Suitable Donors
- PCIe-to-FireWire or niche bridges with simple BARs.
- Older Wi-Fi adapters that expose a clean PCIe function (validate isolation).
- Simple coprocessor/accelerator cards with minimal vendor init.

## Devices to Avoid or De-prioritize
- Boot/system storage (onboard NVMe/SATA) — risky and often entangled in IOMMU groups.
- GPUs — large BARs, complex init, driver entanglement.
- On-board chipset devices sharing IOMMU groups with bridges/root ports.
- Proprietary multi-function SoCs unless you need them and can isolate.

## Practical Selection Tips
- Favor add-in cards you can move between slots to get a clean IOMMU group.
- Check isolation before dumping: `lspci -vvv -s <BDF>` and group membership under `/sys/kernel/iommu_groups/`.
- Keep Secure Boot disabled or enroll keys if you plan to load donor_dump.ko.
- For NVMe donors: use a secondary drive on a PCIe adapter; run NVMe captures before binding to VFIO; do not use your boot disk.

## Rationale
The best donors are standard-compliant PCIe endpoints with:
- Clean, readable 4KB config space and intact capability chains (MSI/MSI-X, PM, PCIe caps).
- Modest BAR sizes that fit FPGA resource and overlay constraints.
- Minimal vendor-specific initialization requirements.
- Stable, documented behavior to reduce guesswork in templating and validation.

Use this list to pick hardware that minimizes friction during donor dump, templating, and FPGA build.
