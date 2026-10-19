//! Serving requests on the Unix socket: one JSON request per line, one
//! JSON response per line. The daemon holds the archive lock, runs the
//! convergence loop on its own thread, and answers reads from a pool of
//! read connections while the loop writes.

use super::{Request, Response};
use crate::briefing::{self, Ask};
use polylogue_engine::{Config, Engine, LoopMsg, Pass};
use polylogue_model::Author;
use polylogue_store::user::{self as user_writes, machine_id};
use polylogue_store::{
    Archive, Delivery, ReadPool, StoreError, UserRequest, UserWrite, WriterSender,
};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::mpsc::{Sender, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// What request handlers share.
pub struct Daemon {
    pub archive: Archive,
    pub reads: ReadPool,
    /// User writes (notes, deliveries, settings) through the one writer.
    writes: WriterSender,
    loop_tx: Mutex<Sender<LoopMsg>>,
    status: Arc<Mutex<Pass>>,
    started: Instant,
}

fn error(kind: &str, message: impl ToString) -> Response {
    Response::Error {
        kind: kind.into(),
        message: message.to_string(),
        detail: None,
    }
}

fn store_error(e: &StoreError) -> Response {
    match e {
        StoreError::Conflict(_) => error("conflict", e),
        StoreError::NotFound(_) => error("not_found", e),
        StoreError::Invalid(_) => error("bad_request", e),
        _ => error("store", e),
    }
}

/// Prepares context and, when a sender is named, records it as a delivery
/// before handing it out: prepared is recorded first, delivered is a
/// separate event the client reports.
fn context(d: &Daemon, request_id: String, ask: &Ask, sender: Option<String>) -> Response {
    if matches!(ask, Ask::SessionStart { .. }) {
        let off = d
            .reads
            .reader()
            .and_then(|r| r.setting("briefing.session_start", &machine_id()))
            .is_ok_and(|v| v.as_deref() == Some("off"));
        if off {
            return ok(serde_json::json!({ "off": true, "empty": true }));
        }
    }
    let prepared = {
        let reader = match d.reads.reader() {
            Ok(r) => r,
            Err(e) => return store_error(&e),
        };
        match briefing::prepare(&reader, ask, &request_id, user_writes::now()) {
            Ok(p) => p,
            Err(e) => return store_error(&e),
        }
    };
    let mut delivery = None;
    if let Some(sender) = sender
        && !prepared.empty
    {
        let write = UserWrite::Deliver(Delivery {
            recipient_kind: if ask.recipient().is_some() {
                "session".into()
            } else {
                "operator".into()
            },
            recipient_id: ask.recipient().map(ToString::to_string),
            sender,
            event: prepared.event.into(),
            cwd: ask.cwd().map(|c| c.to_string_lossy().into_owned()),
            text: prepared.text.clone(),
            refs: prepared.refs.clone(),
            left_out: prepared.left_out.clone(),
        });
        match d.writes.user(UserRequest::now(request_id, write)) {
            Ok(out) => delivery = Some(out.id),
            Err(e) => return store_error(&e),
        }
    }
    ok(serde_json::json!({ "delivery": delivery, "prepared": prepared }))
}

fn write(d: &Daemon, request_id: String, w: UserWrite) -> Response {
    match d.writes.user(UserRequest::now(request_id, w)) {
        Ok(out) => ok(out),
        Err(e) => store_error(&e),
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
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
        Request::Query {
            text,
            cursor,
            page_size,
            why,
            session,
        } => {
            let ask = super::query::Ask {
                text: &text,
                cursor: cursor.as_deref(),
                page_size,
                why,
                session,
                now: now_ms(),
                // The operator's zone and 04:00 day start; the `day.zone`
                // and `day.logical_start` settings come with the settings
                // lane.
                day: polylogue_model::LogicalDay::system(),
                embeddings: false,
            };
            match super::query::run(&reader, ask) {
                Ok(data) => Response::Ok { data },
                Err(f) => Response::Error {
                    kind: f.kind,
                    message: f.message,
                    detail: f.detail,
                },
            }
        }
        Request::Sql { statement } => match reader.query(&statement, &[]) {
            // The connection is read-only: a statement that writes fails.
            Ok(rows) => ok(rows),
            Err(e) => error("sql", e),
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
        Request::Export { sessions, format } => export(&reader, &sessions, &format),
        Request::Excise {
            request_id,
            target,
            reason,
            dry_run,
        } => {
            drop(reader);
            let scope = match polylogue_engine::ExciseScope::parse(&target) {
                Ok(s) => s,
                Err(e) => return error("bad_request", e),
            };
            let request = polylogue_engine::ExciseRequest {
                id: request_id,
                scope,
                reason,
                dry_run,
            };
            let (reply, wait) = sync_channel(1);
            let sent = d
                .loop_tx
                .lock()
                .map(|tx| tx.send(LoopMsg::Excise { request, reply }).is_ok())
                .unwrap_or(false);
            if !sent {
                return error("internal", "the convergence loop has stopped");
            }
            match wait.recv() {
                Ok(Ok(done)) => ok(done),
                Ok(Err(e)) if e.contains("not found") => error("not_found", e),
                Ok(Err(e)) => error("excise", e),
                Err(_) => error("internal", "the convergence loop has stopped"),
            }
        }
        Request::AddNote { request_id, note } => {
            drop(reader);
            write(d, request_id, UserWrite::AddNote { note })
        }
        Request::Withdraw {
            request_id,
            note,
            author,
            reason,
        } => {
            drop(reader);
            write(
                d,
                request_id,
                UserWrite::Withdraw {
                    note,
                    author,
                    reason,
                },
            )
        }
        Request::Notes { filter } => match reader.notes(&filter) {
            Ok(notes) => ok(notes),
            Err(e) => store_error(&e),
        },
        Request::Context {
            request_id,
            ask,
            sender,
        } => {
            drop(reader);
            context(d, request_id, &ask, sender)
        }
        Request::DeliveryEvent {
            delivery,
            what,
            detail,
        } => {
            drop(reader);
            write(
                d,
                user_writes::new_request_id(),
                UserWrite::DeliveryEvent {
                    delivery,
                    what,
                    detail,
                },
            )
        }
        Request::Candidates { cwd, limit } => {
            match crate::continuation::candidates(&reader, &cwd, limit.unwrap_or(10)) {
                Ok(c) => ok(c),
                Err(e) => store_error(&e),
            }
        }
        Request::Continue { session } => match crate::continuation::plan(&reader, &session) {
            Ok(p) => ok(p),
            Err(e) => store_error(&e),
        },
        Request::Setting {
            request_id,
            key,
            value: None,
            machine: _,
        } => {
            let _ = request_id;
            match reader.setting(&key, &machine_id()) {
                Ok(v) => ok(serde_json::json!({ "key": key, "value": v })),
                Err(e) => store_error(&e),
            }
        }
        Request::Setting {
            request_id,
            key,
            value: Some(value),
            machine,
        } => {
            drop(reader);
            write(
                d,
                request_id.unwrap_or_else(user_writes::new_request_id),
                UserWrite::SetSetting {
                    key,
                    value,
                    author: Author::Operator,
                    machine: machine.then(machine_id),
                },
            )
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

/// Whole sessions in the export format (`json`, lossless and read back by
/// `import`) or rendered as Markdown or HTML.
fn export(
    reader: &polylogue_store::Reader<'_>,
    ids: &[polylogue_model::SessionId],
    format: &str,
) -> Response {
    let mut sessions = Vec::new();
    for id in ids {
        match reader.session_export(id) {
            Ok(Some(s)) => sessions.push(s),
            Ok(None) => return error("not_found", format!("no session {id}")),
            Err(e) => return store_error(&e),
        }
    }
    let doc = polylogue_model::export::Document::new(sessions);
    if format == "json" {
        return match serde_json::to_value(&doc) {
            Ok(v) => Response::Ok { data: v },
            Err(e) => error("internal", e),
        };
    }
    let Some(f) = polylogue_render::Format::parse(format) else {
        return error("bad_request", format!("unknown export format {format:?}"));
    };
    let mut items = Vec::new();
    for s in &doc.sessions {
        items.push(polylogue_render::Rendered::SessionHeader(&s.session));
        for m in &s.messages {
            items.push(polylogue_render::Rendered::Message(m));
        }
        for e in &s.events {
            items.push(polylogue_render::Rendered::Event(e));
        }
    }
    let mut out = Vec::new();
    if let Err(e) = polylogue_render::render(&items, f, polylogue_render::View::Full, &mut out) {
        return error("internal", e);
    }
    ok(serde_json::json!({ "format": format, "content": String::from_utf8_lossy(&out) }))
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
    let writes = engine.writer().map_err(|e| to_io(&e))?;
    let reads = ReadPool::open(&archive).map_err(|e| to_io(&e))?;
    let (tx, rx) = channel();
    let status = Arc::new(Mutex::new(Pass::default()));
    let daemon = Arc::new(Daemon {
        archive: archive.clone(),
        reads,
        writes,
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
    eprintln!(
        "polylogue: serving {} on {}",
        root.display(),
        socket.display()
    );
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        let d = Arc::clone(&daemon);
        std::thread::spawn(move || serve_client(stream, &d));
    }
    Ok(())
}
