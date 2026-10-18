//! Serving requests on the Unix socket: one JSON request per line, one
//! JSON response per line. The daemon holds the archive lock, runs the
//! convergence loop on its own thread, and answers reads from a pool of
//! read connections while the loop writes.

use super::{Request, Response};
use polylogue_engine::{Config, Engine, LoopMsg, Pass};
use polylogue_store::{Archive, ReadPool};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::mpsc::{Sender, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What request handlers share.
pub struct Daemon {
    pub archive: Archive,
    pub reads: ReadPool,
    loop_tx: Mutex<Sender<LoopMsg>>,
    status: Arc<Mutex<Pass>>,
    started: Instant,
}

fn error(kind: &str, message: impl ToString) -> Response {
    Response::Error {
        kind: kind.into(),
        message: message.to_string(),
    }
}

fn ok(data: impl serde::Serialize) -> Response {
    match serde_json::to_value(data) {
        Ok(data) => Response::Ok { data },
        Err(e) => error("internal", e),
    }
}

/// Answers one request. Reads use the read pool; imports go to the loop
/// thread, which owns the writer.
pub fn handle(request: Request, d: &Daemon) -> Response {
    let reader = match d.reads.reader() {
        Ok(r) => r,
        Err(e) => return error("store", e),
    };
    match request {
        Request::Status => {
            let store = match reader.status() {
                Ok(s) => s,
                Err(e) => return error("store", e),
            };
            let last_pass = d.status.lock().map(|p| p.clone()).unwrap_or_default();
            ok(serde_json::json!({
                "archive": d.archive.root,
                "pid": std::process::id(),
                "uptime_seconds": d.started.elapsed().as_secs(),
                "store": store,
                "last_pass": last_pass,
            }))
        }
        Request::List { limit, before } => match reader.list(limit, before) {
            Ok(rows) => ok(rows),
            Err(e) => error("store", e),
        },
        Request::Read {
            session, logical, ..
        } => {
            let s = match reader.session(&session) {
                Ok(Some(s)) => s,
                Ok(None) => return error("not_found", format!("no session {session}")),
                Err(e) => return error("store", e),
            };
            match reader.messages(&session, logical) {
                Ok(messages) => ok(serde_json::json!({ "session": s, "messages": messages })),
                Err(e) => error("store", e),
            }
        }
        Request::Import { paths, .. } => {
            drop(reader);
            let (reply, wait) = sync_channel(1);
            let sent = d
                .loop_tx
                .lock()
                .map(|tx| tx.send(LoopMsg::Import { paths, reply }).is_ok())
                .unwrap_or(false);
            if !sent {
                return error("internal", "the convergence loop has stopped");
            }
            match wait.recv() {
                Ok(Ok(report)) => ok(report),
                Ok(Err(e)) => error("import", e),
                Err(_) => error("internal", "the convergence loop has stopped"),
            }
        }
        other => {
            let op = serde_json::to_value(&other)
                .ok()
                .and_then(|v| v.get("op").and_then(|o| o.as_str()).map(str::to_owned))
                .unwrap_or_default();
            error("not_built", format!("`{op}` is not built yet"))
        }
    }
}

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
    let to_io = |e: &dyn std::fmt::Display| std::io::Error::other(e.to_string());
    let root = Archive::default_root();
    let (archive, _lock) = Archive::open_or_create(&root).map_err(|e| to_io(&e))?;
    let mut engine = Engine::new(Config::from_env(), archive.clone()).map_err(|e| to_io(&e))?;
    let reads = ReadPool::open(&archive).map_err(|e| to_io(&e))?;
    let (tx, rx) = channel();
    let status = Arc::new(Mutex::new(Pass::default()));
    let daemon = Arc::new(Daemon {
        archive: archive.clone(),
        reads,
        loop_tx: Mutex::new(tx.clone()),
        status: Arc::clone(&status),
        started: Instant::now(),
    });
    std::thread::Builder::new()
        .name("polylogue-loop".into())
        .spawn(move || {
            if let Err(e) = engine.run(tx, &rx, &status) {
                eprintln!("polylogue: the convergence loop stopped: {e}");
            }
        })?;

    let socket = archive.socket();
    // A socket file nobody answers on is left from a stopped daemon; the
    // archive lock above proves no other daemon owns this archive.
    if socket.exists() {
        std::fs::remove_file(&socket)?;
    }
    let listener = UnixListener::bind(&socket)?;
    eprintln!("polylogue: serving {} on {}", root.display(), socket.display());
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        let d = Arc::clone(&daemon);
        std::thread::spawn(move || serve_client(stream, &d));
    }
    Ok(())
}
