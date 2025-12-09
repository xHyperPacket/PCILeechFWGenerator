"""Cross-platform `fcntl` compatibility shim.

Provides a minimal `fcntl`-like interface on platforms where the real
module is unavailable (e.g., Windows). The shim registers itself in
``sys.modules['fcntl']`` so plain ``import fcntl`` works after this module
has been imported at least once.
"""
from __future__ import annotations

import sys
import types
import errno

try:  # pragma: no cover - real import path on POSIX
    import fcntl as _fcntl  # type: ignore
    FCNTL_AVAILABLE = True
except ImportError:  # pragma: no cover - Windows / non-POSIX
    shim = types.ModuleType("fcntl")
    shim.LOCK_SH = 1
    shim.LOCK_EX = 2
    shim.LOCK_UN = 8
    shim.LOCK_NB = 4

    def _unsupported(*_args, **_kwargs):
        raise OSError(errno.ENOSYS, "fcntl is not available on this platform")

    shim.flock = _unsupported
    shim.ioctl = _unsupported
    shim.fcntl = _unsupported

    # Export shim as the fcntl module
    sys.modules["fcntl"] = shim
    _fcntl = shim
    FCNTL_AVAILABLE = False
else:
    # Ensure the module is registered (should already be)
    sys.modules.setdefault("fcntl", _fcntl)

fcntl = _fcntl


def ensure_fcntl():
    """Return the platform fcntl module or shim."""
    return _fcntl


__all__ = ["fcntl", "FCNTL_AVAILABLE", "ensure_fcntl"]
