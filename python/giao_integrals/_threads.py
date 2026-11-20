"""Thread policy shared by Python wrappers and native Python entry points."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from functools import lru_cache
from operator import index
from pathlib import Path

import numpy as np

from ._giao_integrals import _build_info, openmp_enabled

_override: int | None = None
_warned_serial = False


def _positive(value, name="threads") -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        n = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if not 1 <= n <= 2**31 - 1:
        raise ValueError(f"{name} must be a positive integer <= 2147483647")
    return n


@lru_cache(maxsize=1)
def physical_core_count() -> int:
    """Count physical cores available to the process; use a safe fallback."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                check=True,
            )
            return max(1, int(result.stdout))
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    if sys.platform.startswith("linux"):
        try:
            cpus = os.sched_getaffinity(0)
            cores = set()
            for cpu in cpus:
                topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
                cores.add(
                    (
                        (topology / "physical_package_id").read_text().strip(),
                        (topology / "core_id").read_text().strip(),
                    )
                )
            if cores:
                return len(cores)
        except (AttributeError, OSError):
            pass
    # Logical cores may oversubscribe SMT machines, so fall back to serial.
    return 1


def set_num_threads(n: int | None) -> None:
    """Set the process-wide GIAO default; None restores environment/core policy.

    Explicit threads= on an integral call takes precedence. Changes apply to
    subsequent calls, not OpenMP regions already in progress. This does not
    change BLAS settings or the environment.
    """
    global _override
    if n is not None:
        n = _positive(n, "n")
        if n > 1 and not openmp_enabled():
            raise ValueError("parallel ERI evaluation requires an OpenMP-enabled build")
    _override = n


def resolve_num_threads(threads: int | None = None) -> int:
    """Resolve one thread budget before releasing the GIL or allocating ERIs."""
    global _warned_serial
    explicit = threads is not None
    if explicit:
        n = _positive(threads)
    elif _override is not None:
        n = _override
    else:
        n = None
        for name in ("GIAO_NUM_THREADS", "OMP_NUM_THREADS"):
            if name in os.environ:
                try:
                    # OMP nested lists are intentionally unsupported: no nesting here.
                    n = _positive(int(os.environ[name]), name)
                except ValueError as error:
                    raise ValueError(f"{name} must be a positive integer") from error
                break
        if n is None:
            n = physical_core_count()
    if n > 1 and not openmp_enabled():
        message = "parallel ERI evaluation requires an OpenMP-enabled build"
        if explicit:
            raise ValueError(message)
        if not _warned_serial:
            warnings.warn(message + "; using threads=1", RuntimeWarning, stacklevel=2)
            _warned_serial = True
        return 1
    return n


def build_info() -> dict:
    """Return version, compiler, OpenMP runtime path/status, and default threads.

    openmp_runtime_path identifies the loaded runtime (including wheel runtimes),
    or None for serial builds. default_threads is the effective GIAO budget;
    physical_cores is the detected physical-core count. BLAS is independent.
    """
    return {
        **_build_info(),
        "default_threads": resolve_num_threads(),
        "physical_cores": physical_core_count(),
    }
