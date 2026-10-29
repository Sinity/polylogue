//! Who may talk to the HTTP port. Loopback
//! alone does not stop a web page in the operator's browser from reaching
//! the port, so every request is checked:
//!
//! - the `Host` header names this listener (stops DNS rebinding);
//! - the connection's peer belongs to the operator's uid;
//! - every route but sign-in and the assets needs the archive's access
//!   token, as the cookie named after the archive key or a bearer header.

use crate::runtime_dir::{archive_key, runtime_dir};
use std::fs;
use std::io::{self, Read};
use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// The default port of an archive's daemon.
pub const DEFAULT_PORT: u16 = 8765;

/// The port to listen on: `POLYLOGUE_HTTP_PORT` when set; a port the
/// system assigns for an archive moved with `POLYLOGUE_ARCHIVE` (tests,
/// demos, measurements), so it never takes the operator's 8765; otherwise
/// 8765. Clients read the port from the `port` file either way.
pub fn wanted_port() -> u16 {
    if let Some(p) = std::env::var("POLYLOGUE_HTTP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
    {
        return p;
    }
    if std::env::var_os("POLYLOGUE_ARCHIVE").is_some() {
        0
    } else {
        DEFAULT_PORT
    }
}

/// A fresh random token, 32 bytes as hex.
pub fn new_token() -> io::Result<String> {
    let mut bytes = [0u8; 32];
    fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// Writes the token and port files (mode 0600) for clients that open the
/// reader. A new daemon writes a new token, so pages from a previous run
/// must authenticate again.
pub fn write_access_files(dir: &Path, token: &str, port: u16) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    write_private(&dir.join("token"), token)?;
    write_private(&dir.join("port"), &port.to_string())
}

fn write_private(path: &Path, text: &str) -> io::Result<()> {
    polylogue_model::atomic_write::write_atomic(path, text.as_bytes(), None)
}

/// The running reader's authenticated entry point, as written by that
/// archive's daemon. Verify the listener before disclosing its token.
pub fn reader_url(root: &Path) -> io::Result<String> {
    let dir = runtime_dir(&archive_key(root));
    reader_url_in(&dir)
}

pub(crate) fn reader_url_in(dir: &Path) -> io::Result<String> {
    let token = fs::read_to_string(dir.join("token"))?;
    let port = fs::read_to_string(dir.join("port"))?
        .trim()
        .parse::<u16>()
        .map_err(|_| io::Error::other("the reader port file is unreadable"))?;
    if listener_owner(port).is_none() || listener_owner(port) != my_uid() {
        return Err(io::Error::other("the reader listener is unavailable"));
    }
    Ok(format!("http://127.0.0.1:{port}/auth#{}", token.trim()))
}

/// Whether a `Host` header names this listener.
pub fn host_allowed(host: Option<&str>, port: u16) -> bool {
    let Some(host) = host else {
        return false;
    };
    [
        format!("127.0.0.1:{port}"),
        format!("localhost:{port}"),
        format!("[::1]:{port}"),
    ]
    .iter()
    .any(|h| h.eq_ignore_ascii_case(host))
}

/// The cookie that carries the token for this archive. Cookies are not
/// kept apart by port (RFC 6265 §8.5), so the name carries the key.
pub fn cookie_name(key: &str) -> String {
    format!("polylogue_{key}")
}

/// The token from a `Cookie` header, if it carries this archive's cookie.
pub fn cookie_token<'a>(header: Option<&'a str>, name: &str) -> Option<&'a str> {
    header?.split(';').find_map(|part| {
        let (k, v) = part.trim().split_once('=')?;
        (k == name).then_some(v)
    })
}

/// Compares in time independent of where the inputs first differ.
pub fn same_secret(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

/// The uid that owns the peer's end of an accepted connection, from
/// `/proc/net/tcp{,6}`: the entry whose local address is `peer` and whose
/// remote address is `server`, this listener's end. Both ends must match:
/// the kernel keeps closed connections in `TIME_WAIT` under uid 0, and one
/// local port can carry connections to other destinations owned by other
/// users.
pub fn socket_owner(peer: SocketAddr, server: SocketAddr) -> Option<u32> {
    let (file, local, remote) = match (peer, server) {
        (SocketAddr::V4(p), SocketAddr::V4(s)) => ("/proc/net/tcp", v4_hex(p), v4_hex(s)),
        (SocketAddr::V6(p), SocketAddr::V6(s)) => ("/proc/net/tcp6", v6_hex(p), v6_hex(s)),
        _ => return None,
    };
    owner_in(&fs::read_to_string(file).ok()?, &local, &remote).or_else(|| {
        // An IPv4 peer that connected from a dual-stack socket is listed as
        // v4-mapped v6.
        let (SocketAddr::V4(p), SocketAddr::V4(s)) = (peer, server) else {
            return None;
        };
        let mapped = |a: std::net::SocketAddrV4| {
            std::net::SocketAddrV6::new(a.ip().to_ipv6_mapped(), a.port(), 0, 0)
        };
        owner_in(
            &fs::read_to_string("/proc/net/tcp6").ok()?,
            &v6_hex(mapped(p)),
            &v6_hex(mapped(s)),
        )
    })
}

/// The uid owning the socket listening on 127.0.0.1:`port`, the address
/// the reader binds and its URL names, so a client never sends the token to
/// another user's listener or to one on another address.
pub fn listener_owner(port: u16) -> Option<u32> {
    let loopback = std::net::SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, port);
    // A dual-stack socket bound to the v4 address is listed as v4-mapped v6.
    let mapped = std::net::SocketAddrV6::new(loopback.ip().to_ipv6_mapped(), port, 0, 0);
    [
        ("/proc/net/tcp", v4_hex(loopback)),
        ("/proc/net/tcp6", v6_hex(mapped)),
    ]
    .iter()
    .find_map(|(file, local)| listener_in(&fs::read_to_string(file).ok()?, local))
}

/// The uid of the socket listening on `local` in a `/proc/net/tcp` table.
/// State `0A` is LISTEN.
fn listener_in(table: &str, local: &str) -> Option<u32> {
    table.lines().skip(1).find_map(|line| {
        let cols: Vec<&str> = line.split_whitespace().collect();
        (cols.get(1)? == &local && cols.get(3)? == &"0A").then(|| cols.get(7)?.parse().ok())?
    })
}

/// The uid of the live socket with these local and remote addresses in a
/// `/proc/net/tcp` table. State `06` is `TIME_WAIT`, which has no owner.
fn owner_in(table: &str, local: &str, remote: &str) -> Option<u32> {
    table.lines().skip(1).find_map(|line| {
        let cols: Vec<&str> = line.split_whitespace().collect();
        (cols.get(1)? == &local && cols.get(2)? == &remote && cols.get(3)? != &"06")
            .then(|| cols.get(7)?.parse().ok())?
    })
}

fn v4_hex(a: std::net::SocketAddrV4) -> String {
    // The kernel prints the address as a host-order u32 in hex.
    let n = u32::from_ne_bytes(a.ip().octets());
    format!("{n:08X}:{:04X}", a.port())
}

fn v6_hex(a: std::net::SocketAddrV6) -> String {
    let o = a.ip().octets();
    let words: String = o
        .chunks(4)
        .map(|c| format!("{:08X}", u32::from_ne_bytes([c[0], c[1], c[2], c[3]])))
        .collect();
    format!("{words}:{:04X}", a.port())
}

/// The calling process's uid.
pub fn my_uid() -> Option<u32> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("Uid:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Replaces the archive directory's path with `[archive]` in text served
/// to the browser.
pub fn scrub(text: &str, root: &Path) -> String {
    let root = root.to_string_lossy();
    if root.is_empty() || root == "/" {
        return text.to_owned();
    }
    text.replace(root.as_ref(), "[archive]")
}

/// HTTP hides this host path even when it occurs in a stored value or key.
pub(super) fn scrub_value(value: &mut serde_json::Value, root: &Path) {
    match value {
        serde_json::Value::String(text) => *text = scrub(text, root),
        serde_json::Value::Array(values) => {
            values.iter_mut().for_each(|value| scrub_value(value, root))
        }
        serde_json::Value::Object(object) => {
            let original = std::mem::take(object);
            // Original keys keep their names when a transformed key collides.
            let (unchanged, changed): (Vec<_>, Vec<_>) = original
                .into_iter()
                .partition(|(key, _)| scrub(key, root) == *key);
            for (key, mut value) in unchanged.into_iter().chain(changed) {
                scrub_value(&mut value, root);
                let base = scrub(&key, root);
                let mut key = base.clone();
                let mut ordinal = 2;
                while object.contains_key(&key) {
                    key = format!("{base} #{ordinal}");
                    ordinal += 1;
                }
                object.insert(key, value);
            }
        }
        _ => {}
    }
}

/// Only one declared media type is accepted; parameters such as UTF-8 do
/// not change the body format. Duplicate headers cannot select a parser.
pub(super) fn content_type(headers: &axum::http::HeaderMap, expected: &str) -> bool {
    let mut values = headers.get_all(axum::http::header::CONTENT_TYPE).iter();
    let Some(value) = values.next() else {
        return false;
    };
    values.next().is_none()
        && value.to_str().ok().is_some_and(|value| {
            value
                .split(';')
                .next()
                .is_some_and(|value| value.trim().eq_ignore_ascii_case(expected))
        })
}

#[cfg(test)]
mod tests;
