//! Publish listening sockets and observe their directory without polling.
use rustix::fs::inotify;
use std::ffi::OsString;
use std::io;
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

/// The longest Unix socket path in bytes (`sun_path` is 108 with its NUL).
const SOCKET_PATH_LIMIT: usize = 107;

/// Uses the same directory entry through an open parent when the full path
/// does not fit in a Unix socket address.
pub(crate) fn with_short_path<T>(
    path: &Path,
    open: impl FnOnce(&Path) -> io::Result<T>,
) -> io::Result<T> {
    if path.as_os_str().as_bytes().len() <= SOCKET_PATH_LIMIT {
        return open(path);
    }
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "socket has no file name"))?;
    let parent = std::fs::File::open(path.parent().unwrap_or_else(|| Path::new(".")))?;
    let alias = PathBuf::from(format!("/proc/self/fd/{}", parent.as_raw_fd())).join(name);
    open(&alias)
}

pub(crate) fn connect(path: &Path) -> io::Result<UnixStream> {
    with_short_path(path, |path| UnixStream::connect(path))
}

/// Publishes a managed MCP address after the archive's own listener is ready.
/// The service owns this path for the lifetime of its private runtime directory.
/// Leaving a dangling link during a stop avoids an old instance unlinking
/// its successor's endpoint.
pub(crate) fn publish_mcp_link(path: &Path, target: &Path) -> io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let directory = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the MCP socket has no parent directory",
        )
    })?;
    let metadata = std::fs::metadata(directory)?;
    if !metadata.is_dir()
        || metadata.permissions().mode() & 0o077 != 0
        || metadata.uid() != rustix::process::geteuid().as_raw()
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "the MCP socket needs a private directory owned by the daemon user",
        ));
    }
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.is_symlink() => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "the MCP socket path is not a symlink",
            ));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let temporary = path.with_file_name(format!(".polylogue-mcp-{}", uuid::Uuid::now_v7()));
    std::os::unix::fs::symlink(target, &temporary)?;
    let result = std::fs::rename(&temporary, path);
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
    result
}

/// The caller owns the path through its archive lock or hosted-session identity.
/// Publish only after bind has also made the socket listen.
pub(crate) fn bind(path: &Path) -> io::Result<UnixListener> {
    // A longer path would be published but no client could connect to it.
    if path.as_os_str().as_bytes().len() > SOCKET_PATH_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "the socket path {} is longer than the {SOCKET_PATH_LIMIT} bytes a Unix socket allows; use a shorter archive directory",
                path.display()
            ),
        ));
    }
    let mut private = path.as_os_str().to_os_string();
    private.push(".binding");
    let private = PathBuf::from(private);
    match std::fs::remove_file(&private) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    let listener = with_short_path(&private, |path| UnixListener::bind(path));
    let result = listener.and_then(|listener| {
        std::fs::rename(&private, path)?;
        Ok(listener)
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&private);
    }
    result
}

pub(crate) struct Watch {
    descriptor: OwnedFd,
    name: OsString,
}

impl Watch {
    pub(crate) fn new(path: &Path) -> io::Result<Self> {
        let name = path
            .file_name()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "socket has no file name"))?
            .to_os_string();
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let descriptor =
            inotify::init(inotify::CreateFlags::CLOEXEC | inotify::CreateFlags::NONBLOCK)?;
        inotify::add_watch(
            &descriptor,
            parent,
            inotify::WatchFlags::CREATE
                | inotify::WatchFlags::MOVED_TO
                | inotify::WatchFlags::CLOSE_WRITE
                | inotify::WatchFlags::DELETE_SELF
                | inotify::WatchFlags::MOVE_SELF
                | inotify::WatchFlags::ONLYDIR,
        )?;
        Ok(Self { descriptor, name })
    }

    /// Drain notifications. Closing the writable archive lock also wakes a
    /// client whose daemon failed before it could publish a socket.
    pub(crate) fn changed(&self) -> io::Result<bool> {
        let mut bytes = [MaybeUninit::uninit(); 4096];
        let mut reader = inotify::Reader::new(&self.descriptor, &mut bytes);
        let mut changed = false;
        loop {
            let event = match reader.next() {
                Ok(event) => event,
                Err(rustix::io::Errno::AGAIN) => return Ok(changed),
                Err(rustix::io::Errno::INTR) => continue,
                Err(error) => return Err(error.into()),
            };
            if event.events().intersects(
                inotify::ReadFlags::DELETE_SELF
                    | inotify::ReadFlags::MOVE_SELF
                    | inotify::ReadFlags::IGNORED,
            ) {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "the socket directory disappeared while waiting",
                ));
            }
            let name = event.file_name().map(|name| name.to_bytes());
            changed |= name == Some(self.name.as_bytes())
                || (name == Some(b"lock")
                    && event.events().contains(inotify::ReadFlags::CLOSE_WRITE))
                || event.events().contains(inotify::ReadFlags::QUEUE_OVERFLOW);
        }
    }
}

impl AsFd for Watch {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.descriptor.as_fd()
    }
}

impl AsRawFd for Watch {
    fn as_raw_fd(&self) -> RawFd {
        self.descriptor.as_raw_fd()
    }
}

#[cfg(test)]
mod tests;
