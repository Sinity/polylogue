//! Unix socket framing and listener lifecycle. Application operations live
//! in `service`; each complete JSON request receives one JSON response.

use super::Request;
use super::service::{self, Daemon, error, handle};
use polylogue_engine::Config;
use polylogue_store::Archive;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;

fn serve_client(stream: UnixStream, d: &Daemon) {
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let mut out = stream;
    for line in BufReader::new(read_half).lines() {
        let Ok(line) = line else {
            return;
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(r) => handle(r, d),
            Err(e) => error("bad_request", e),
        };
        let Ok(mut text) = serde_json::to_string(&response) else {
            return;
        };
        text.push('\n');
        if out.write_all(text.as_bytes()).is_err() {
            return;
        }
    }
}

/// Runs the daemon until it is killed: takes the archive lock, starts the
/// loop and serves clients.
pub fn serve() -> std::io::Result<()> {
    let root = Archive::default_root();
    let (daemon, _lock) = service::start(&root, Config::from_env())?;
    let listener = bind(&daemon)?;
    eprintln!(
        "polylogue: serving {} on {}",
        root.display(),
        daemon.archive.socket().display()
    );
    // The browser's port. If it cannot be served (the port is taken), the
    // socket still serves every other client.
    match crate::web::start(Arc::clone(&daemon)) {
        Ok(port) => eprintln!("polylogue: web reader on http://127.0.0.1:{port}/"),
        Err(e) => eprintln!("polylogue: web reader not started: {e}"),
    }
    listen(&listener, &daemon);
    Ok(())
}

/// Binds the archive's socket. A socket file nobody answers on is left
/// from a stopped daemon; the archive lock held by `daemon` proves no
/// other daemon owns this archive.
pub fn bind(daemon: &Daemon) -> std::io::Result<UnixListener> {
    let socket = daemon.archive.socket();
    if socket.exists() {
        std::fs::remove_file(&socket)?;
    }
    UnixListener::bind(&socket)
}

/// Serves clients on `listener`, one thread each, until it fails.
pub fn listen(listener: &UnixListener, daemon: &Arc<Daemon>) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        let d = Arc::clone(daemon);
        std::thread::spawn(move || serve_client(stream, &d));
    }
}
