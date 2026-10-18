//! The daemon: it owns the archive writer, serves clients on a Unix socket,
//! and runs the configured collectors. Also the client side of the protocol.
use polylogue_engine::owner::{Handle, Owner};
use polylogue_model::types::{ErrorKind, Failure};
use polylogue_store::{ReadHandle, Writer};
pub mod authored;
mod claude_continuation;
mod codex_continuation;
pub mod input_cat;
pub mod model_read;
pub mod native_tools;
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

fn pending_codex_references(
    reader: &ReadHandle,
    until: std::time::Instant,
    selection: Option<&str>,
) -> Result<Value, polylogue_store::Error> {
    let mut pending = polylogue_engine::codex_discovery::pending(reader, until, selection)?;
    let mut omitted = 0;
    if let Some(references) = pending["references"].as_array_mut() {
        references.retain(|reference| {
            let keep = reference["session"]
                .as_str()
                .is_some_and(|id| output_json::check_reference(id, until).is_ok());
            if !keep {
                omitted += 1;
            }
            keep
        });
        let shown = references.len();
        pending["references_shown"] = json!(shown);
    }
    if omitted > 0 {
        pending["references_omitted"] = pending["references_omitted"]
            .as_u64()
            .map(|count| json!(count + omitted))
            .unwrap_or(Value::Null);
        pending["identity_details_omitted"] = json!(omitted);
        pending["presentation_complete"] = json!(false);
    }
    Ok(pending)
}
/// Drop the per-reference and per-snapshot details from a status reply, keeping
/// the counts. Used when a reply must fit a budget.
pub fn summarize_pending_references(value: &mut Value) {
    let Some(pending) = value.get_mut("pending_references") else {
        return;
    };
    let references = pending["references"].as_array().map_or(0, Vec::len);
    let snapshots = pending["snapshots"].as_array().map_or(0, Vec::len);
    pending["references"] = json!([]);
    pending["snapshots"] = json!([]);
    pending["references_shown"] = json!(0);
    pending["snapshot_details_shown"] = json!(0);
    pending["references_omitted"] = pending["references_omitted"]
        .as_u64()
        .map(|n| json!(n + references as u64))
        .unwrap_or(Value::Null);
    pending["snapshot_details_omitted"] = pending["snapshot_details_omitted"]
        .as_u64()
        .map(|n| json!(n + snapshots as u64))
        .unwrap_or(Value::Null);
    pending["presentation_complete"] = json!(false);
}

pub fn socket_path(archive: &Path, runtime: &Path) -> PathBuf {
    let key = blake3::hash(archive.as_os_str().as_encoded_bytes()).to_hex();
    runtime.join("polylogue").join(&key[..12]).join("socket")
}

/// One Claude Code session file to collect, and who is collecting it.
#[derive(Clone, Debug)]
pub struct ClaudeFile {
    pub path: PathBuf,
    pub machine: String,
    pub host: String,
    pub request_id: String,
    pub seen_at: i64,
}

pub struct Running {
    owner: Option<Owner>,
    collector: Option<JoinHandle<()>>,
    continuation: Arc<Mutex<Option<codex_continuation::Status>>>,
    claude_continuation: Arc<Mutex<Option<claude_continuation::Status>>>,
    stop: Arc<AtomicBool>,
    clients: Arc<Mutex<HashMap<u64, UnixStream>>>,
    accept: Option<JoinHandle<()>>,
    socket: OwnedSocket,
}
impl Running {
    /// Recover a stopped archive into a new directory. Starts no socket or writer.
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
        Self::start_writer(writer, archive, runtime, fail_after_bind, None, false)
    }
    /// Store the Claude Code session file now, then start the daemon. With
    /// `follow`, keep polling the file in chunks of that many bytes.
    pub fn start_claude_file(
        archive: &Path,
        runtime: &Path,
        file: &ClaudeFile,
        follow: Option<u64>,
        cancelled: &AtomicBool,
    ) -> Result<(Self, polylogue_store::StoredSource), polylogue_store::Error> {
        if follow.is_some_and(|bytes| !(65536..=8 * 1024 * 1024).contains(&bytes)) {
            return Err(polylogue_store::Error::ObservedSourceRefused(
                "acquisition quantum must be between 65536 and 8388608 bytes".into(),
            ));
        }
        let ClaudeFile {
            path,
            machine,
            host,
            request_id,
            seen_at,
        } = file;
        let mut writer = Writer::open_cancellable(archive, cancelled)?;
        let stored =
            writer.acquire_claude_main(path, machine, host, request_id, *seen_at, cancelled)?;
        let stored = polylogue_engine::claude_bodies::acquire(
            &mut writer,
            &stored,
            path,
            machine,
            host,
            request_id,
            *seen_at,
            cancelled,
        )?;
        let mut running = Self::start_writer(writer, archive, runtime, false, None, false)
            .map_err(polylogue_store::Error::Invariant)?;
        if let Some(quantum_bytes) = follow {
            *running.claude_continuation.lock().expect("Claude status") =
                Some(claude_continuation::Status::initial(quantum_bytes));
            let handle = running.owner.as_ref().expect("running owner").handle();
            let file = file.clone();
            let stop = Arc::clone(&running.stop);
            let status = Arc::clone(&running.claude_continuation);
            running.collector = Some(
                thread::Builder::new()
                    .name("polylogue-claude-main".into())
                    .spawn(move || {
                        claude_continuation::run(handle, file, quantum_bytes, stop, status)
                    })
                    .map_err(polylogue_store::Error::Io)?,
            );
        }
        Ok((running, stored))
    }
    pub fn start_claude_projects(
        archive: &Path,
        runtime: &Path,
        config: &polylogue_store::configured_claude::Config,
        id: &str,
        cancelled: &AtomicBool,
    ) -> Result<Self, polylogue_store::Error> {
        config.validate()?;
        let writer = Writer::open_cancellable(archive, cancelled)?;
        let mut running = Self::start_writer(writer, archive, runtime, false, None, false)
            .map_err(polylogue_store::Error::Invariant)?;
        let mut initial = claude_continuation::Status::initial(config.acquisition_quantum_bytes);
        initial.mode = "claude_projects";
        initial.state = "discovering";
        initial.projects = Some(claude_continuation::projects::ProjectStatus::initial());
        *running.claude_continuation.lock().expect("Claude status") = Some(initial);
        let handle = running.owner.as_ref().expect("running owner").handle();
        let config = config.clone();
        let id = id.to_owned();
        let stop = Arc::clone(&running.stop);
        let shared = Arc::clone(&running.claude_continuation);
        running.collector = Some(
            thread::Builder::new()
                .name("polylogue-claude-projects".into())
                .spawn(move || claude_continuation::projects::run(handle, config, id, stop, shared))
                .map_err(polylogue_store::Error::Io)?,
        );
        Ok(running)
    }
    pub fn start_codex(
        archive: &Path,
        runtime: &Path,
        config: &polylogue_store::configured_codex::Config,
        id: &str,
        seen_at: i64,
        cancelled: &AtomicBool,
        continue_collecting: bool,
    ) -> Result<(Self, polylogue_store::configured_codex::Acquisition), polylogue_store::Error>
    {
        let mut writer = Writer::open_cancellable(archive, cancelled)?;
        let report = writer.acquire_codex_home(config, id, seen_at, cancelled)?;
        let mut running = Self::start_writer(
            writer,
            archive,
            runtime,
            false,
            Some(report.clone()),
            continue_collecting || config.paginated_history.is_some(),
        )
        .map_err(polylogue_store::Error::Invariant)?;
        if continue_collecting || config.paginated_history.is_some() {
            if let Some(status) = running
                .continuation
                .lock()
                .expect("continuation status")
                .as_mut()
            {
                status.paginated_pending = config.paginated_history.is_some();
                status.paginated_state = if status.paginated_pending {
                    "pending"
                } else {
                    "disabled"
                };
            }
            let handle = running.owner.as_ref().expect("running owner").handle();
            let config = config.clone();
            let request = id.to_owned();
            let stop = Arc::clone(&running.stop);
            let status = Arc::clone(&running.continuation);
            running.collector = Some(
                thread::Builder::new()
                    .name("polylogue-codex-home".into())
                    .spawn(move || {
                        codex_continuation::run(
                            handle,
                            config,
                            request,
                            stop,
                            status,
                            continue_collecting,
                        )
                    })
                    .map_err(polylogue_store::Error::Io)?,
            );
        }
        Ok((running, report))
    }
    fn start_writer(
        writer: Writer,
        archive: &Path,
        runtime: &Path,
        fail_after_bind: bool,
        acquisition: Option<polylogue_store::configured_codex::Acquisition>,
        continue_collecting: bool,
    ) -> Result<Self, String> {
        let acquisition = Arc::new(acquisition);
        let continuation = Arc::new(Mutex::new(
            continue_collecting.then(codex_continuation::Status::initial),
        ));
        let thread_continuation = Arc::clone(&continuation);
        let claude_continuation = Arc::new(Mutex::new(None));
        let thread_claude_continuation = Arc::clone(&claude_continuation);
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
                        let acquisition = Arc::clone(&acquisition);
                        let continuation = Arc::clone(&thread_continuation);
                        let claude_continuation = Arc::clone(&thread_claude_continuation);
                        let completed = Arc::clone(&thread_clients);
                        workers.retain(|worker| !worker.is_finished());
                        workers.push(thread::spawn(move || {
                            let continuation =
                                continuation.lock().expect("continuation status").clone();
                            let claude_continuation =
                                claude_continuation.lock().expect("Claude status").clone();
                            if let Err(error) = serve(
                                stream,
                                &archive,
                                &reader,
                                &handle,
                                acquisition.as_ref().as_ref(),
                                continuation.as_ref(),
                                claude_continuation.as_ref(),
                            ) {
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
            collector: None,
            continuation,
            claude_continuation,
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
        if let Some(collector) = self.collector.take() {
            let _ = collector.join();
        }
        if let Some(owner) = self.owner.take()
            && let Err(error) = owner.shutdown()
        {
            eprintln!("writer shutdown: {error}");
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
        if let Some(inode) = self.inode.take()
            && fs::symlink_metadata(&self.path).is_ok_and(|m| m.ino() == inode)
        {
            let _ = fs::remove_file(&self.path);
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
    acquisition: Option<&polylogue_store::configured_codex::Acquisition>,
    continuation: Option<&codex_continuation::Status>,
    claude_continuation: Option<&claude_continuation::Status>,
) -> Result<(), String> {
    let mut line = String::new();
    use std::io::Read;
    // Limits the request line only; provider records are not limited here.
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;
    BufReader::new(stream.try_clone().map_err(|e| e.to_string())?)
        .take(65537)
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    if line.len() > 65536 || !line.ends_with('\n') {
        return Err("request is larger than 65536 bytes or has no newline".into());
    }
    let reply = match serde_json::from_str::<Value>(&line) {
        Ok(request) if request == json!({"command":"status"}) => {
            let guard = reader.lock().expect("reader lock");
            match guard.status().and_then(|status| {
                pending_codex_references(
                    &guard,
                    std::time::Instant::now() + Duration::from_secs(30),
                    None,
                )
                .map(|pending| (status, pending))
            }) {
                Ok((status, pending)) => {
                    let mut notes = if status.recovery.is_some() {
                        vec![polylogue_model::RecoveryRecord::NOTE]
                    } else {
                        vec![]
                    };
                    if status.unavailable_blob_files > 0 {
                        notes
                            .push("Some stored blob files are missing; reads that need them fail.");
                    }
                    if acquisition.is_some()
                        || (pending["examined_snapshots"]
                            .as_u64()
                            .is_some_and(|n| n > 0)
                            || pending["scan_complete"] == false)
                    {
                        notes.push("Codex sessions found so far may be incomplete.");
                    }
                    json!({"outcome":if notes.is_empty() {"ok"} else {"degraded"},"recovery":status.recovery,"notes":notes,"missing":notes,"archive":archive,"store":status,"acquisition":acquisition,"continuation":continuation,"claude_continuation":claude_continuation,"pending_references":pending})
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
    // Diagnostics can contain provider text, so replies get the same credential
    // masking as session documents.
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
/// Send one request and read its one-line reply.
pub fn request_value(socket: &Path, value: &Value) -> Result<Value, io::Error> {
    request_metadata(socket, value, None, None)
}
/// Like `request_value`, with one deadline covering connect, writes and reads.
pub fn request_value_until(
    socket: &Path,
    value: &Value,
    deadline: std::time::Instant,
) -> Result<Value, io::Error> {
    request_metadata(socket, value, Some(deadline), None)
}
/// Like `request_value_until` with a 30-second deadline, closed early when
/// `cancellation` fires. A write the daemon already accepted stays done.
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
            "request is larger than 65535 bytes",
        ));
    }
    let mut stream = metadata_transport_cancellable(socket, deadline, cancellation)?;
    stream.write_all(&bytes)?;
    stream.write_all(b"\n")?;
    stream.stream.shutdown(std::net::Shutdown::Write)?;
    let mut line = String::new();
    use std::io::Read;
    // Every read uses the same absolute deadline, so a slow trickle of bytes
    // cannot extend it.
    BufReader::new(stream).take(65537).read_line(&mut line)?;
    remaining(deadline)?;
    if line.len() > 65536 || !line.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "reply is larger than 65536 bytes or has no newline",
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
        // socket2 gives a connect with a timeout, without a helper thread.
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
                    // EAGAIN on a Unix socket means the listener's queue is full.
                    // Retry with a fresh socket; a failed one is unusable.
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
mod tests;
