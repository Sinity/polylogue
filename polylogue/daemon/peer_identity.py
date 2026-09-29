"""Which local uid owns the TCP peer of a loopback connection.

The web-credential cookie has no port scoping -- RFC 6265 cookies never do --
so a browser also attaches it to a same-host request aimed at a different
local uid's service on another port; that uid's process can then replay the
leaked cookie with forged Host, Origin and Sec-Fetch-Site headers, since none
of those are enforced for a raw (non-browser) client. The kernel's connection
table is the one witness the peer cannot forge: ``/proc/net/tcp`` on Linux,
``lsof`` over this uid's own sockets elsewhere. Every lookup fails closed.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: The kernel's IPv4 TCP connection table.
_PROC_NET_TCP = Path("/proc/net/tcp")
#: The kernel's IPv6 TCP connection table (``--api-host ::1``).
_PROC_NET_TCP6 = Path("/proc/net/tcp6")


def _procfs_endpoint(ip: str, port: int) -> tuple[Path, str] | None:
    """The connection table and key for one endpoint, as the kernel prints them.

    IPv4 addresses are one little-endian 32-bit word; IPv6 addresses are four
    such words, each in host (little-endian) order.
    """
    try:
        address = ipaddress.ip_address(ip.strip("[]"))
    except ValueError:
        return None
    packed = address.packed
    if isinstance(address, ipaddress.IPv6Address):
        # IPv4-mapped peers of a dual-stack socket are listed in tcp6 too.
        words = "".join(bytes(reversed(packed[index : index + 4])).hex() for index in range(0, 16, 4))
        return _PROC_NET_TCP6, f"{words.upper()}:{port:04X}"
    return _PROC_NET_TCP, f"{bytes(reversed(packed)).hex().upper()}:{port:04X}"


def tcp_socket_owner_uid(*, local_ip: str, local_port: int, remote_ip: str, remote_port: int) -> int | None:
    """The uid owning the peer socket ``remote_ip:remote_port -> local_ip:local_port`` (Linux).

    The peer's own table entry has its local address as what we see as
    remote, and its remote address as the socket we accepted on -- whichever
    loopback address the daemon is bound to. ``None`` when the entry or the
    table is unavailable.
    """
    remote = _procfs_endpoint(remote_ip, remote_port)
    local = _procfs_endpoint(local_ip, local_port)
    if remote is None or local is None or remote[0] != local[0]:
        return None
    (table, remote_key), (_table, local_key) = remote, local
    try:
        text = table.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 8 and fields[1] == remote_key and fields[2] == local_key:
            try:
                return int(fields[7])
            except ValueError:
                return None
    return None


def lsof_peer_is_current_uid(*, local_ip: str, local_port: int, remote_ip: str, remote_port: int) -> bool:
    """Whether this uid owns the peer socket, for hosts without procfs (macOS).

    ``lsof -u <uid>`` lists only sockets this uid's processes hold, and the
    peer's own entry is named ``<remote>-><local>``; our accepted socket is
    named the other way round, so it can never satisfy the match. There is
    no deadline: a slow lookup under load must not turn a valid credential
    invalid. A missing ``lsof`` fails closed.
    """
    lsof = shutil.which("lsof") or ("/usr/sbin/lsof" if Path("/usr/sbin/lsof").exists() else None)
    if lsof is None:
        return False
    try:
        result = subprocess.run(
            [lsof, "-nP", "-a", "-u", str(os.getuid()), f"-iTCP@{remote_ip}:{remote_port}", "-Fn"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    peer_name = f"n{remote_ip}:{remote_port}->{local_ip}:{local_port}"
    return any(line.strip() == peer_name for line in result.stdout.splitlines())


def peer_socket_owned_by_current_uid(*, local_ip: str, local_port: int, remote_ip: str, remote_port: int) -> bool:
    """Whether the kernel attributes the TCP peer's socket to this process's uid."""
    if sys.platform.startswith("linux"):
        uid = tcp_socket_owner_uid(
            local_ip=local_ip, local_port=local_port, remote_ip=remote_ip, remote_port=remote_port
        )
        return uid is not None and uid == os.getuid()
    return lsof_peer_is_current_uid(
        local_ip=local_ip, local_port=local_port, remote_ip=remote_ip, remote_port=remote_port
    )


__all__ = ["lsof_peer_is_current_uid", "peer_socket_owned_by_current_uid", "tcp_socket_owner_uid"]
