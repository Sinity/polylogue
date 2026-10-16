//! Foundation process ownership. Unsupported product operations stay unregistered.
use polylogue_engine::owner::{Handle, Owner};
use polylogue_model::types::{ErrorKind, Failure};
use polylogue_store::{ReadHandle, Writer};
pub mod input_cat;
pub mod model_read;
mod output_json;
mod protocol;
pub mod read_cancellation;
pub mod reference_read;
mod session_document;
pub mod session_export;
pub mod session_query;
pub mod session_read;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead, BufReader, Write},
    os::unix::{
        fs::{FileTypeExt, MetadataExt, PermissionsExt},
        net::{UnixListener, UnixStream},
    },
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

pub fn socket_path(archive: &Path, runtime: &Path) -> PathBuf {
    let key = blake3::hash(archive.as_os_str().as_encoded_bytes()).to_hex();
    runtime.join("polylogue").join(&key[..12]).join("socket")
}

pub struct Running {
    owner: Option<Owner>,
    stop: Arc<AtomicBool>,
    clients: Arc<Mutex<HashMap<u64, UnixStream>>>,
    accept: Option<JoinHandle<()>>,
    socket: OwnedSocket,
}
impl Running {
    /// One-shot daemon ownership; no socket, actor or ingestion is started.
    pub fn recover_copy(
        source: &Path,
        destination: &Path,
        cancelled: &AtomicBool,
    ) -> Result<polylogue_store::recovery::RecoveryReport, polylogue_store::Error> {
        polylogue_store::recovery::recover_copy(source, destination, cancelled)
    }
    pub fn start(archive: &Path, runtime: &Path) -> Result<Self, String> {
        Self::start_inner(archive, runtime, false)
    }
    pub fn start_cancellable(
        archive: &Path,
        runtime: &Path,
        cancelled: &AtomicBool,
    ) -> Result<Self, String> {
        Self::start_cancellable_inner(archive, runtime, false, cancelled)
    }
    fn start_inner(archive: &Path, runtime: &Path, fail_after_bind: bool) -> Result<Self, String> {
        Self::start_cancellable_inner(archive, runtime, fail_after_bind, &AtomicBool::new(false))
    }
    fn start_cancellable_inner(
        archive: &Path,
        runtime: &Path,
        fail_after_bind: bool,
        cancelled: &AtomicBool,
    ) -> Result<Self, String> {
        let writer = Writer::open_cancellable(archive, cancelled).map_err(|e| e.to_string())?;
        Self::start_writer(writer, archive, runtime, fail_after_bind)
    }
    pub fn start_observed(
        archive: &Path,
        runtime: &Path,
        source: &Path,
        machine: &str,
        host: &str,
        id: &str,
        seen_at: i64,
        cancelled: &AtomicBool,
    ) -> Result<(Self, polylogue_store::RetainedSource), polylogue_store::Error> {
        let mut writer = Writer::open_cancellable(archive, cancelled)?;
        let retained = writer.acquire_claude_main(source, machine, host, id, seen_at, cancelled)?;
        Ok((
            Self::start_writer(writer, archive, runtime, false)
                .map_err(polylogue_store::Error::Invariant)?,
            retained,
        ))
    }
    fn start_writer(
        writer: Writer,
        archive: &Path,
        runtime: &Path,
        fail_after_bind: bool,
    ) -> Result<Self, String> {
        let reader = Arc::new(Mutex::new(writer.read_handle().map_err(|e| e.to_string())?));
        let archive = fs::canonicalize(archive).map_err(|e| e.to_string())?;
        let socket = socket_path(&archive, runtime);
        if socket.as_os_str().as_encoded_bytes().len() >= 108 {
            return Err(format!(
                "socket address is too long: {} bytes: {}",
                socket.as_os_str().as_encoded_bytes().len(),
                socket.display()
            ));
        }
        let parent = socket.parent().expect("socket has parent");
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
        // Only the archive lock holder may remove a socket left by a dead daemon.
        if let Ok(metadata) = fs::symlink_metadata(&socket) {
            match UnixStream::connect(&socket) {
                Err(error)
                    if error.kind() == io::ErrorKind::ConnectionRefused
                        && metadata.file_type().is_socket() =>
                {
                    fs::remove_file(&socket).map_err(|e| e.to_string())?;
                }
                _ => return Err(format!("endpoint is occupied: {}", socket.display())),
            }
        }
        let listener = UnixListener::bind(&socket)
            .map_err(|e| format!("cannot bind {}: {e}", socket.display()))?;
        let inode = fs::symlink_metadata(&socket)
            .map_err(|e| e.to_string())?
            .ino();
        let socket = OwnedSocket {
            path: socket,
            inode: Some(inode),
        };
        if fail_after_bind {
            return Err("injected startup failure after bind".into());
        }
        fs::set_permissions(&socket.path, fs::Permissions::from_mode(0o600))
            .map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let stop = Arc::new(AtomicBool::new(false));
        let clients = Arc::new(Mutex::new(HashMap::new()));
        let owner = Owner::start(writer, Arc::clone(&stop)).map_err(|e| e.to_string())?;
        let handle = owner.handle();
        let thread_stop = Arc::clone(&stop);
        let thread_clients = Arc::clone(&clients);
        let accept = thread::spawn(move || {
            let mut workers: Vec<JoinHandle<()>> = Vec::new();
            let mut next_client = 0u64;
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let Ok(cancel) = stream.try_clone() else {
                            continue;
                        };
                        let id = next_client;
                        next_client += 1;
                        thread_clients
                            .lock()
                            .expect("client lock")
                            .insert(id, cancel);
                        // Bound simultaneous request decoders and idle sockets.
                        if thread_clients.lock().expect("client lock").len() > 64 {
                            thread_clients.lock().expect("client lock").remove(&id);
                            let _ = stream.shutdown(std::net::Shutdown::Both);
                            continue;
                        }
                        let handle = handle.clone();
                        let archive = archive.clone();
                        let reader = Arc::clone(&reader);
                        let completed = Arc::clone(&thread_clients);
                        workers.retain(|worker| !worker.is_finished());
                        workers.push(thread::spawn(move || {
                            if let Err(error) = serve(stream, &archive, &reader, &handle) {
                                eprintln!("client: {error}");
                            }
                            completed.lock().expect("client lock").remove(&id);
                        }));
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(_) => break,
                }
            }
            for (_, stream) in thread_clients.lock().expect("client lock").drain() {
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
            for worker in workers {
                let _ = worker.join();
            }
        });
        Ok(Self {
            owner: Some(owner),
            stop,
            clients,
            accept: Some(accept),
            socket,
        })
    }
    pub fn socket(&self) -> &Path {
        &self.socket.path
    }
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for stream in self.clients.lock().expect("client lock").values() {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
        self.socket.remove();
        if let Some(owner) = self.owner.take() {
            if let Err(error) = owner.shutdown() {
                eprintln!("writer shutdown: {error}");
            }
        }
    }
}
impl Drop for Running {
    fn drop(&mut self) {
        self.stop();
    }
}

struct OwnedSocket {
    path: PathBuf,
    inode: Option<u64>,
}
impl OwnedSocket {
    fn remove(&mut self) {
        if let Some(inode) = self.inode.take() {
            if fs::symlink_metadata(&self.path).is_ok_and(|m| m.ino() == inode) {
                let _ = fs::remove_file(&self.path);
            }
        }
    }
}
impl Drop for OwnedSocket {
    fn drop(&mut self) {
        self.remove();
    }
}

fn serve(
    mut stream: UnixStream,
    archive: &Path,
    reader: &Mutex<ReadHandle>,
    owner: &Handle,
) -> Result<(), String> {
    let mut line = String::new();
    use std::io::Read;
    // Admission is on request metadata, never a semantic provider-record cap.
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;
    BufReader::new(stream.try_clone().map_err(|e| e.to_string())?)
        .take(65537)
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    if line.len() > 65536 || !line.ends_with('\n') {
        return Err("request metadata exceeds admission or lacks a newline".into());
    }
    let reply = match serde_json::from_str::<Value>(&line) {
        Ok(request) if request == json!({"command":"status"}) => {
            match reader.lock().expect("reader lock").status() {
                Ok(status) => {
                    let notes = if status.recovery.is_some() {
                        vec![polylogue_model::RecoveryQualification::NOTE]
                    } else {
                        vec![]
                    };
                    json!({"outcome":if status.recovery.is_some(){"degraded"}else{"ok"},"recovery":status.recovery,"notes":notes,"missing":notes,"archive":archive,"foundation":status,"capabilities":["schema","status","context-qualified-immutable-retention","observed-claude-startup-v1","masked-retained-input-cat-v1","provider-progress","selected-message-terminal-read","selected-session-json-read","bounded-text-model-read-v1","session-query-list-prose-v1"],"product_path":"partial"})
                }
                Err(error) => {
                    json!({"outcome":"error","error":Failure::new(ErrorKind::Io,error.to_string()),"archive":archive})
                }
            }
        }
        Ok(request) if request["command"] == "input-cat" => {
            return input_cat::serve(&mut stream, &request, reader).map_err(|e| e.to_string());
        }
        Ok(request) if request["command"] == "query" => {
            return session_query::serve(&mut stream, &request, reader).map_err(|e| e.to_string());
        }
        Ok(request) if request["command"] == "model-read" => {
            return model_read::serve(&mut stream, &request, reader).map_err(|e| e.to_string());
        }
        Ok(request) if request["command"] == "session-read" => {
            return session_read::serve(&mut stream, &request, reader).map_err(|e| e.to_string());
        }
        Ok(request) if request["command"] == "read" => {
            return reference_read::serve(&mut stream, &request, reader).map_err(|e| e.to_string());
        }
        Ok(request) => protocol::handle(&request, owner),
        Err(error) => {
            json!({"outcome":"error","error":Failure::new(ErrorKind::InvalidQuery,error.to_string()),"archive":archive})
        }
    };
    // Retained interpretation diagnostics may contain provider-controlled text.
    // Use the same response-boundary policy as normalized JSON documents.
    let mut encoded = output_json::Encoder::new(
        &mut stream,
        std::time::Instant::now() + Duration::from_secs(30),
    );
    encoded.value(&reply).map_err(|e| e.to_string())?;
    encoded.finish().map_err(|e| e.to_string())?;
    stream.write_all(b"\n").map_err(|e| e.to_string())
}

pub fn request(socket: &Path) -> Result<Value, io::Error> {
    request_value(socket, &json!({"command":"status"}))
}
/// Send one request and read one bounded metadata response. Import success means
/// durable raw retention; query readiness is reported independently.
pub fn request_value(socket: &Path, value: &Value) -> Result<Value, io::Error> {
    request_metadata(socket, value, None, None)
}
/// The absolute deadline covers connect, every write/read, framing and decode.
/// No helper thread owns a transport after this call returns or is cancelled.
pub fn request_value_until(
    socket: &Path,
    value: &Value,
    deadline: std::time::Instant,
) -> Result<Value, io::Error> {
    request_metadata(socket, value, Some(deadline), None)
}
/// A cancellable read-only metadata request, sharing the normal decoder.
pub fn request_value_cancellable(
    socket: &Path,
    value: &Value,
    cancellation: read_cancellation::ReadCancellation,
) -> io::Result<Value> {
    request_metadata(
        socket,
        value,
        Some(std::time::Instant::now() + Duration::from_secs(30)),
        Some(cancellation),
    )
}
fn request_metadata(
    socket: &Path,
    value: &Value,
    deadline: Option<std::time::Instant>,
    cancellation: Option<read_cancellation::ReadCancellation>,
) -> Result<Value, io::Error> {
    let bytes =
        serde_json::to_vec(value).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    if bytes.len() >= 65536 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "request metadata exceeds admission",
        ));
    }
    let mut stream = metadata_transport_cancellable(socket, deadline, cancellation)?;
    stream.write_all(&bytes)?;
    stream.write_all(b"\n")?;
    stream.stream.shutdown(std::net::Shutdown::Write)?;
    let mut line = String::new();
    use std::io::Read;
    // Every underlying read uses the SAME absolute deadline; a trickle cannot
    // renew a per-frame or per-request timeout through BufReader/read_line.
    BufReader::new(stream).take(65537).read_line(&mut line)?;
    remaining(deadline)?;
    if line.len() > 65536 || !line.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid response metadata frame",
        ));
    }
    let result =
        serde_json::from_str(&line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    remaining(deadline)?;
    Ok(result)
}
pub(crate) fn metadata_transport_cancellable(
    socket: &Path,
    deadline: Option<std::time::Instant>,
    cancellation: Option<read_cancellation::ReadCancellation>,
) -> io::Result<MetadataTransport> {
    if let Some(c) = &cancellation {
        c.check()?;
    }
    let stream = if deadline.is_some() {
        // socket2 owns nonblocking connect/poll without a blocked helper thread.
        // Remaining time is recomputed after address/socket construction too.
        let address = socket2::SockAddr::unix(socket)?;
        let stream = if cancellation.is_some() {
            loop {
                if let Some(c) = &cancellation {
                    c.check()?;
                }
                remaining(deadline)?;
                let stream =
                    socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None)?;
                stream.set_nonblocking(true)?;
                match stream.connect(&address) {
                    Ok(()) => {
                        stream.set_nonblocking(false)?;
                        break stream;
                    }
                    // For a saturated local Unix listener EAGAIN means no
                    // connection was queued. Retry before sending any bytes,
                    // with a fresh socket because failed state is unspecified.
                    Err(e)
                        if e.kind() == io::ErrorKind::WouldBlock
                            || e.raw_os_error() == Some(libc::EINPROGRESS) =>
                    {
                        thread::sleep(remaining(deadline)?.unwrap().min(Duration::from_millis(10)));
                    }
                    Err(e) => return Err(e),
                }
            }
        } else {
            let stream = socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None)?;
            stream.connect_timeout(&address, remaining(deadline)?.unwrap())?;
            stream
        };
        UnixStream::from(std::os::fd::OwnedFd::from(stream))
    } else {
        UnixStream::connect(socket)?
    };
    remaining(deadline)?;
    if let Some(c) = &cancellation {
        c.attach(&stream)?;
    }
    Ok(MetadataTransport {
        stream,
        deadline,
        cancellation,
    })
}
fn remaining(deadline: Option<std::time::Instant>) -> io::Result<Option<Duration>> {
    deadline
        .map(|deadline| {
            deadline
                .checked_duration_since(std::time::Instant::now())
                .filter(|remaining| !remaining.is_zero())
                .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "metadata request deadline"))
        })
        .transpose()
}
pub(crate) struct MetadataTransport {
    stream: UnixStream,
    deadline: Option<std::time::Instant>,
    cancellation: Option<read_cancellation::ReadCancellation>,
}
impl Drop for MetadataTransport {
    fn drop(&mut self) {
        if let Some(c) = &self.cancellation {
            c.detach();
        }
    }
}
impl io::Read for MetadataTransport {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if let Some(c) = &self.cancellation {
            c.check()?;
        }
        self.stream.set_read_timeout(remaining(self.deadline)?)?;
        let result = io::Read::read(&mut self.stream, bytes);
        remaining(self.deadline)?;
        result
    }
}
impl Write for MetadataTransport {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(c) = &self.cancellation {
            c.check()?;
        }
        self.stream.set_write_timeout(remaining(self.deadline)?)?;
        let result = self.stream.write(bytes);
        remaining(self.deadline)?;
        result
    }
    fn flush(&mut self) -> io::Result<()> {
        remaining(self.deadline)?;
        self.stream.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metadata_write_uses_the_same_absolute_deadline_after_partial_progress() {
        let (stream, _unread) = UnixStream::pair().unwrap();
        socket2::SockRef::from(&stream)
            .set_send_buffer_size(1024)
            .unwrap();
        let started = std::time::Instant::now();
        let mut transport = MetadataTransport {
            stream,
            deadline: Some(started + Duration::from_millis(80)),
            cancellation: None,
        };
        let error = transport.write_all(&vec![b'x'; 65536]).unwrap_err();
        assert!(matches!(
            error.kind(),
            io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
        ));
        assert!(started.elapsed() < Duration::from_secs(1));
    }
    #[test]
    fn saturated_unix_connect_never_blocks_beyond_absolute_deadline() {
        let temp = tempfile::tempdir_in("/tmp").unwrap();
        let path = temp.path().join("socket");
        let listener =
            socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None).unwrap();
        listener
            .bind(&socket2::SockAddr::unix(&path).unwrap())
            .unwrap();
        listener.listen(0).unwrap();
        let _queued = UnixStream::connect(&path).unwrap();
        let started = std::time::Instant::now();
        assert!(
            request_value_until(
                &path,
                &json!({"command":"progress"}),
                started + Duration::from_millis(80)
            )
            .is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
    }
    #[test]
    fn independent_stacks_shutdown_only_owned_clients_and_socket() {
        let temp = tempfile::tempdir().unwrap();
        // Unix socket names have a kernel length limit independent of TMPDIR.
        // Only socket dentries live here; database/blob work stays in managed TMPDIR.
        let runtime_owner = tempfile::tempdir_in("/tmp").unwrap();
        let runtime = runtime_owner.path().join("runtime");
        let mut first = Running::start(&temp.path().join("a"), &runtime).unwrap();
        let second = Running::start(&temp.path().join("b"), &runtime).unwrap();
        assert_eq!(request(first.socket()).unwrap()["product_path"], "partial");
        assert_eq!(request(second.socket()).unwrap()["outcome"], "ok");
        let mut stalled = UnixStream::connect(first.socket()).unwrap();
        stalled.write_all(b"{\"command\":").unwrap();
        let first_path = first.socket().to_owned();
        first.stop();
        assert!(!first_path.exists());
        assert!(request(second.socket()).is_ok());
        let replacement = Running::start(&temp.path().join("a"), &runtime).unwrap();
        drop(first);
        assert!(request(replacement.socket()).is_ok());
    }
    #[test]
    fn startup_failure_removes_only_its_bound_socket_and_releases_writer() {
        let temp = tempfile::tempdir().unwrap();
        let archive = temp.path().join("archive");
        // Unix socket names have a kernel length limit independent of TMPDIR.
        // Only socket dentries live here; database/blob work stays in managed TMPDIR.
        let runtime_owner = tempfile::tempdir_in("/tmp").unwrap();
        let runtime = runtime_owner.path().join("runtime");
        assert!(Running::start_inner(&archive, &runtime, true).is_err());
        let socket = socket_path(&fs::canonicalize(&archive).unwrap(), &runtime);
        assert!(!socket.exists());
        let running = Running::start(&archive, &runtime).unwrap();
        assert!(request(running.socket()).is_ok());
    }
    #[test]
    fn occupied_endpoint_is_never_adopted_or_removed() {
        let temp = tempfile::tempdir().unwrap();
        let archive = temp.path().join("archive");
        let writer = Writer::open(&archive).unwrap();
        drop(writer);
        // Unix socket names have a kernel length limit independent of TMPDIR.
        // Only socket dentries live here; database/blob work stays in managed TMPDIR.
        let runtime_owner = tempfile::tempdir_in("/tmp").unwrap();
        let runtime = runtime_owner.path().join("runtime");
        let socket = socket_path(&fs::canonicalize(&archive).unwrap(), &runtime);
        fs::create_dir_all(socket.parent().unwrap()).unwrap();
        let foreign = UnixListener::bind(&socket).unwrap();
        let inode = fs::symlink_metadata(&socket).unwrap().ino();
        assert!(Running::start(&archive, &runtime).is_err());
        assert_eq!(fs::symlink_metadata(&socket).unwrap().ino(), inode);
        assert!(UnixStream::connect(&socket).is_ok());
        drop(foreign);
    }
    #[test]
    fn replaced_socket_survives_owner_drop() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_owner = tempfile::tempdir_in("/tmp").unwrap();
        let running = Running::start(&temp.path().join("archive"), runtime_owner.path()).unwrap();
        let path = running.socket().to_owned();
        fs::remove_file(&path).unwrap();
        let foreign = UnixListener::bind(&path).unwrap();
        let inode = fs::symlink_metadata(&path).unwrap().ino();
        drop(running);
        assert_eq!(fs::symlink_metadata(&path).unwrap().ino(), inode);
        drop(foreign);
    }
}
