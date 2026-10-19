//! Who may talk to the HTTP port (spec `surfaces/web.md`, Access). Loopback
//! alone does not stop a web page in the operator's browser from reaching
//! the port, so every request is checked:
//!
//! - the `Host` header names this listener (stops DNS rebinding);
//! - the connection's peer belongs to the operator's uid;
//! - every route but sign-in and the assets needs the archive's access
//!   token, as the cookie named after the archive key or a bearer header.

use std::fs;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// The default port of an archive's daemon.
pub const DEFAULT_PORT: u16 = 8765;

/// The first 12 hex digits of BLAKE3 over the archive directory's resolved
/// path: it names the archive's runtime directory and its cookie, so two
/// archives never share a token.
pub fn archive_key(root: &Path) -> String {
    let resolved = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let hash = blake3::hash(resolved.as_os_str().as_encoded_bytes());
    hash.to_hex()[..12].to_owned()
}

/// `$XDG_RUNTIME_DIR/polylogue/<key>/`: the token and the port.
pub fn runtime_dir(key: &str) -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("polylogue").join(key)
}

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
pub fn publish(dir: &Path, token: &str, port: u16) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    write_private(&dir.join("token"), token)?;
    write_private(&dir.join("port"), &port.to_string())
}

fn write_private(path: &Path, text: &str) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)?;
    f.write_all(text.as_bytes())?;
    f.sync_all()?;
    fs::rename(tmp, path)
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

/// The uid that owns the TCP socket at `addr`, from `/proc/net/tcp{,6}`.
/// For an accepted connection, the peer's own socket is the entry whose
/// local address is the peer address.
pub fn socket_owner(addr: SocketAddr) -> Option<u32> {
    let (file, wanted) = match addr {
        SocketAddr::V4(a) => ("/proc/net/tcp", v4_hex(a)),
        SocketAddr::V6(a) => ("/proc/net/tcp6", v6_hex(a)),
    };
    let table = fs::read_to_string(file).ok()?;
    owner_in(&table, &wanted).or_else(|| {
        // An IPv4 peer on a dual-stack listener appears as v4-mapped v6.
        let SocketAddr::V4(a) = addr else {
            return None;
        };
        let mapped = SocketAddr::new(a.ip().to_ipv6_mapped().into(), a.port());
        let SocketAddr::V6(m) = mapped else {
            return None;
        };
        owner_in(&fs::read_to_string("/proc/net/tcp6").ok()?, &v6_hex(m))
    })
}

/// The uid owning the socket listening on loopback `port`, so a client
/// never sends the token to another user's listener.
pub fn listener_owner(port: u16) -> Option<u32> {
    let suffix = format!(":{port:04X}");
    ["/proc/net/tcp", "/proc/net/tcp6"].iter().find_map(|file| {
        let table = fs::read_to_string(file).ok()?;
        table.lines().skip(1).find_map(|line| {
            let cols: Vec<&str> = line.split_whitespace().collect();
            // State 0A is LISTEN.
            (cols.get(1)?.ends_with(&suffix) && cols.get(3)? == &"0A")
                .then(|| cols.get(7)?.parse().ok())?
        })
    })
}

fn owner_in(table: &str, wanted: &str) -> Option<u32> {
    table.lines().skip(1).find_map(|line| {
        let cols: Vec<&str> = line.split_whitespace().collect();
        (cols.get(1)? == &wanted).then(|| cols.get(7)?.parse().ok())?
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn host_must_name_this_listener() {
        assert!(host_allowed(Some("127.0.0.1:8765"), 8765));
        assert!(host_allowed(Some("localhost:8765"), 8765));
        assert!(host_allowed(Some("[::1]:8765"), 8765));
        assert!(!host_allowed(Some("evil.example:8765"), 8765));
        assert!(!host_allowed(Some("127.0.0.1:8766"), 8765));
        assert!(!host_allowed(Some("127.0.0.1"), 8765));
        assert!(!host_allowed(None, 8765));
    }

    #[test]
    fn the_cookie_is_named_after_the_archive() {
        let name = cookie_name("abc123");
        assert_eq!(
            cookie_token(Some("x=1; polylogue_abc123=tok; y=2"), &name),
            Some("tok")
        );
        assert_eq!(cookie_token(Some("polylogue_other=tok"), &name), None);
        assert_eq!(cookie_token(None, &name), None);
    }

    #[test]
    fn keys_differ_by_archive_and_are_twelve_hex_digits() {
        let a = archive_key(Path::new("/tmp/a-archive"));
        let b = archive_key(Path::new("/tmp/b-archive"));
        assert_eq!(a.len(), 12);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn secrets_compare_whole() {
        assert!(same_secret("abc", "abc"));
        assert!(!same_secret("abc", "abd"));
        assert!(!same_secret("abc", "abcd"));
    }

    #[test]
    fn a_local_connection_is_owned_by_this_uid() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (_server, peer) = listener.accept().unwrap();
        assert_eq!(peer, client.local_addr().unwrap());
        assert_eq!(socket_owner(peer), my_uid());
        assert!(my_uid().is_some());
    }

    #[test]
    fn the_archive_path_is_scrubbed() {
        let root = Path::new("/home/x/.local/share/polylogue");
        assert_eq!(
            scrub("file /home/x/.local/share/polylogue/blobs/ab", root),
            "file [archive]/blobs/ab"
        );
    }

    #[test]
    fn publish_writes_private_files() {
        let dir = tempfile::tempdir().unwrap();
        let rt = dir.path().join("k");
        publish(&rt, "tok", 1234).unwrap();
        assert_eq!(fs::read_to_string(rt.join("token")).unwrap(), "tok");
        assert_eq!(fs::read_to_string(rt.join("port")).unwrap(), "1234");
        let mode = fs::metadata(rt.join("token")).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
