"""Runtime identity and extension safety contract for Polylogue entrypoints."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _cgroup_cpu_quota(
    *,
    v2_path: Path = _CGROUP_V2_CPU_MAX,
    v1_quota_path: Path = _CGROUP_V1_QUOTA,
    v1_period_path: Path = _CGROUP_V1_PERIOD,
) -> int | None:
    """Return the integer CPUs allowed by cgroup quota, if it is bounded."""
    try:
        quota_text, period_text = v2_path.read_text().split()
        if quota_text == "max":
            return None
        quota, period = int(quota_text), int(period_text)
    except (OSError, ValueError):
        try:
            quota = int(v1_quota_path.read_text().strip())
            period = int(v1_period_path.read_text().strip())
        except (OSError, ValueError):
            return None
        if quota <= 0:
            return None
    if period <= 0:
        return None
    return max(1, quota // period)


def available_cpus(
    *,
    cpu_count: int | None = None,
    affinity: int | None = None,
    v2_path: Path = _CGROUP_V2_CPU_MAX,
    v1_quota_path: Path = _CGROUP_V1_QUOTA,
    v1_period_path: Path = _CGROUP_V1_PERIOD,
) -> int | None:
    """Return the smallest credible CPU budget available to this process.

    The cgroup quota is authoritative for CPU admission; affinity and the
    host count provide fallback and upper-bound signals when quota is absent.
    Optional values are dependency seams for callers and tests.
    """
    host = os.cpu_count() if cpu_count is None else cpu_count
    quota = _cgroup_cpu_quota(
        v2_path=v2_path,
        v1_quota_path=v1_quota_path,
        v1_period_path=v1_period_path,
    )
    candidates = [count for count in (host, quota) if count]
    if affinity is None:
        with contextlib.suppress(AttributeError, OSError):
            affinity = len(os.sched_getaffinity(0))
    if affinity:
        candidates.append(affinity)
    return min(candidates) if candidates else None


MINIMUM_PYTHON = (3, 14)
REQUIRED_NATIVE_PACKAGES: tuple[str, ...] = (
    "sqlite_vec",
    "nh3",
    "watchfiles",
)
OPTIONAL_NATIVE_PACKAGES: tuple[str, ...] = ("msgspec",)


class RuntimeContractError(RuntimeError):
    """The process does not satisfy Polylogue's selected runtime contract."""


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    implementation: str
    version: tuple[int, int, int]
    gil_enabled: bool
    abi_flags: str
    executable: str

    @property
    def free_threaded(self) -> bool:
        return self.implementation == "cpython" and self.version[:2] >= MINIMUM_PYTHON and not self.gil_enabled

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "version": ".".join(str(part) for part in self.version),
            "gil_enabled": self.gil_enabled,
            "free_threaded": self.free_threaded,
            "abi_flags": self.abi_flags,
            "executable": self.executable,
        }


@dataclass(frozen=True, slots=True)
class ExtensionProbe:
    name: str
    importable: bool
    origin: str | None
    native_files: tuple[str, ...]
    abi_compatible: bool
    error: str | None = None

    @property
    def safe(self) -> bool:
        return self.importable and self.abi_compatible and self.error is None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "importable": self.importable,
            "origin": self.origin,
            "native_files": list(self.native_files),
            "abi_compatible": self.abi_compatible,
            "safe": self.safe,
            "error": self.error,
        }


def runtime_identity() -> RuntimeIdentity:
    """Return the live interpreter identity without relying on version text."""

    checker = getattr(sys, "_is_gil_enabled", None)
    gil_enabled = True if checker is None else bool(checker())
    version = (int(sys.version_info[0]), int(sys.version_info[1]), int(sys.version_info[2]))
    return RuntimeIdentity(
        implementation=sys.implementation.name,
        version=version,
        gil_enabled=gil_enabled,
        abi_flags=str(getattr(sys, "abiflags", "")),
        executable=sys.executable,
    )


def require_free_threaded_runtime(*, consumer: str) -> RuntimeIdentity:
    """Fail before product work when the selected free-threaded runtime is absent."""

    identity = runtime_identity()
    checker = getattr(sys, "_is_gil_enabled", None)
    if checker is None:
        reason = "CPython free-threading probe sys._is_gil_enabled is unavailable"
    elif identity.implementation != "cpython":
        reason = f"implementation {identity.implementation!r} is not CPython"
    elif identity.version[:2] < MINIMUM_PYTHON:
        reason = f"Python {identity.version[0]}.{identity.version[1]} is older than 3.14"
    elif identity.gil_enabled:
        reason = "the GIL is enabled"
    else:
        incompatible = tuple(
            probe.name
            for probe in probe_extensions()
            if not probe.safe and (probe.name in REQUIRED_NATIVE_PACKAGES or probe.importable)
        )
        if incompatible:
            raise RuntimeContractError(
                f"{consumer} requires free-threaded-compatible extensions; "
                f"failed imports or ABI checks: {', '.join(incompatible)}; refusing to start"
            )
        return identity
    raise RuntimeContractError(
        f"{consumer} requires CPython 3.14 free-threading; {reason}; refusing to start before archive or network work"
    )


def _native_files(module: ModuleType, origin: str | None) -> tuple[str, ...]:
    roots: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        roots.append(Path(module_file).parent)
    if origin and origin not in {"built-in", "frozen"}:
        roots.append(Path(origin).parent)
    suffixes = (".so", ".pyd")
    found: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for suffix in suffixes:
            found.update(str(path) for path in root.glob(f"*{suffix}"))
    return tuple(sorted(found))


def _abi_compatible(native_files: tuple[str, ...]) -> bool:
    """Accept pure/native SQLite extensions and require the active CPython ABI for CPython wheels."""

    soabi = str(sysconfig.get_config_var("SOABI") or "")
    for filename in native_files:
        name = Path(filename).name
        if "cpython-" in name and soabi not in name:
            return False
    return True


def probe_extensions(names: tuple[str, ...] = REQUIRED_NATIVE_PACKAGES) -> tuple[ExtensionProbe, ...]:
    """Import required extension packages and return their concrete safety evidence."""

    probes: list[ExtensionProbe] = []
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None:
            probes.append(ExtensionProbe(name, False, None, (), False, "module not found"))
            continue
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # extension import failures are evidence, not crashes
            probes.append(ExtensionProbe(name, False, spec.origin, (), False, f"{type(exc).__name__}: {exc}"))
            continue
        native_files = _native_files(module, spec.origin)
        probes.append(ExtensionProbe(name, True, spec.origin, native_files, _abi_compatible(native_files)))
    return tuple(probes)


def runtime_report() -> dict[str, object]:
    """Build the machine-readable runtime proof used by devtools and packaging."""

    identity = runtime_identity()
    extensions = probe_extensions()
    extensions_safe = all(
        probe.safe or (probe.name in OPTIONAL_NATIVE_PACKAGES and not probe.importable) for probe in extensions
    )
    return {
        "runtime": identity.to_dict(),
        "extensions": [probe.to_dict() for probe in extensions],
        "free_threaded_assertion": identity.free_threaded,
        "extensions_safe": extensions_safe,
        "pass": identity.free_threaded and extensions_safe,
    }


__all__ = [
    "ExtensionProbe",
    "MINIMUM_PYTHON",
    "OPTIONAL_NATIVE_PACKAGES",
    "REQUIRED_NATIVE_PACKAGES",
    "RuntimeContractError",
    "RuntimeIdentity",
    "probe_extensions",
    "require_free_threaded_runtime",
    "runtime_identity",
    "runtime_report",
]
