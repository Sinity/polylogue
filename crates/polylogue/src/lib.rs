//! Foundation process ownership. Unsupported product operations stay unregistered.
use polylogue_engine::owner::{Handle, Owner};
use polylogue_model::types::{ErrorKind, Failure};
use polylogue_store::{ReadHandle, Writer};
mod protocol;
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
                    json!({"outcome":"ok","archive":archive,"foundation":status,"capabilities":["schema","status","codex-immutable-retention","provider-progress"],"product_path":"not_implemented"})
                }
                Err(error) => {
                    json!({"outcome":"error","error":Failure::new(ErrorKind::Io,error.to_string()),"archive":archive})
                }
            }
        }
        Ok(request) => protocol::handle(&request, owner),
        Err(error) => {
            json!({"outcome":"error","error":Failure::new(ErrorKind::InvalidQuery,error.to_string()),"archive":archive})
        }
    };
    serde_json::to_writer(&mut stream, &reply).map_err(|e| e.to_string())?;
    stream.write_all(b"\n").map_err(|e| e.to_string())
}

pub fn request(socket: &Path) -> Result<Value, io::Error> {
    request_value(socket, &json!({"command":"status"}))
}
/// Send one request and read one bounded metadata response. Import success means
/// durable raw retention; query readiness is reported independently.
pub fn request_value(socket: &Path, value: &Value) -> Result<Value, io::Error> {
    let mut stream = UnixStream::connect(socket)?;
    let bytes =
        serde_json::to_vec(value).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    if bytes.len() >= 65536 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "request metadata exceeds admission",
        ));
    }
    stream.write_all(&bytes)?;
    stream.write_all(b"\n")?;
    stream.shutdown(std::net::Shutdown::Write)?;
    let mut line = String::new();
    use std::io::Read;
    BufReader::new(stream).take(65537).read_line(&mut line)?;
    if line.len() > 65536 || !line.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid response metadata frame",
        ));
    }
    serde_json::from_str(&line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_stacks_shutdown_only_owned_clients_and_socket() {
        let temp = tempfile::tempdir().unwrap();
        let runtime = temp.path().join("runtime");
        let mut first = Running::start(&temp.path().join("a"), &runtime).unwrap();
        let second = Running::start(&temp.path().join("b"), &runtime).unwrap();
        assert_eq!(
            request(first.socket()).unwrap()["product_path"],
            "not_implemented"
        );
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
        let runtime = temp.path().join("runtime");
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
        let runtime = temp.path().join("runtime");
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
        let running =
            Running::start(&temp.path().join("archive"), &temp.path().join("runtime")).unwrap();
        let path = running.socket().to_owned();
        fs::remove_file(&path).unwrap();
        let foreign = UnixListener::bind(&path).unwrap();
        let inode = fs::symlink_metadata(&path).unwrap().ino();
        drop(running);
        assert_eq!(fs::symlink_metadata(&path).unwrap().ino(), inode);
        drop(foreign);
    }
}
