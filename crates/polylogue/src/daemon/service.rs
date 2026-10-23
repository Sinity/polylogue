//! Application operations and their shared archive state. Socket, HTTP and
//! MCP adapters use the same typed requests; transport framing stays outside.

use super::{Request, Response};
use crate::briefing::{self, Ask};
use polylogue_engine::{Config, Engine, LoopMsg, Pass};
use polylogue_model::Author;
use polylogue_store::user_writes::{self as user_writes, machine_id};
use polylogue_store::{
    Archive, Delivery, ReadPool, StoreError, UserRequest, UserWrite, WriterSender,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// What request handlers share.
pub struct Daemon {
    pub archive: Archive,
    pub reads: ReadPool,
    /// The loop's settings: where the browser host spools captures for
    /// the loop to read.
    pub config: Config,
    /// User writes (notes, deliveries, settings) through the one writer.
    pub(crate) writes: Option<WriterSender>,
    backing: BackingState,
    loop_tx: Mutex<Sender<LoopMsg>>,
    status: Arc<Mutex<Pass>>,
    started: Instant,
    /// The signal rules in force and the ones refused, for `status`.
    pub(crate) signals: Mutex<serde_json::Value>,
    stopping: AtomicBool,
}

pub(crate) fn error(kind: &str, message: impl ToString) -> Response {
    Response::Error {
        kind: kind.into(),
        message: message.to_string(),
        detail: None,
    }
    .masked()
}

pub(crate) fn store_error(e: &StoreError) -> Response {
    match e {
        StoreError::Conflict(_) => error("conflict", e),
        StoreError::ReadPoolClosed => error("daemon_stopping", e),
        StoreError::NotFound(_) => error("not_found", e),
        StoreError::Invalid(_) => error("bad_request", e),
        StoreError::ExcisionFormat { input, format, .. } => Response::Error {
            kind: "unsupported_excision".into(),
            message: e.to_string(),
            detail: Some(serde_json::json!({"input": input, "format": format})),
        },
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
        let writes = match writer(d) {
            Ok(writes) => writes,
            Err(response) => return response,
        };
        match writes.user(UserRequest::now(request_id, write)) {
            Ok(out) => delivery = Some(out.id),
            Err(e) => return store_error(&e),
        }
    }
    ok(serde_json::json!({ "delivery": delivery, "prepared": prepared }))
}

fn writes_paused(d: &Daemon) -> Response {
    match &d.backing {
        BackingState::Blocked {
            path,
            error: reason,
            ..
        } => Response::Error {
            kind: "backing_unavailable".into(),
            message: format!("archive writes are paused: {reason}"),
            detail: Some(serde_json::json!({ "config_path": path, "error": reason })),
        },
        _ => store_error(&StoreError::WriterGone),
    }
}

fn writer(d: &Daemon) -> Result<&WriterSender, Response> {
    d.writes.as_ref().ok_or_else(|| writes_paused(d))
}

fn write(d: &Daemon, request_id: String, w: UserWrite) -> Response {
    let writes = match writer(d) {
        Ok(writes) => writes,
        Err(response) => return response,
    };
    match writes.user(UserRequest::now(request_id, w)) {
        Ok(out) => ok(out),
        Err(e) => store_error(&e),
    }
}

fn browser_refusal(refused: crate::browser::actions::Refused) -> Response {
    Response::Error {
        kind: refused.1.into(),
        message: refused.1.into(),
        detail: Some(serde_json::json!({ "http_status": refused.0 })),
    }
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

pub(crate) fn ok(data: impl serde::Serialize) -> Response {
    match serde_json::to_value(data) {
        Ok(data) => Response::Ok { data },
        Err(e) => error("internal", e),
    }
}

/// Prepare a complete finite feed through the same admission and read pool.
/// The returned unnamed file owns delivery bytes, never a live Reader.
pub(crate) fn change_feed(d: &Daemon, since: Option<&str>) -> Result<std::fs::File, Response> {
    if d.stopping.load(Ordering::Acquire) {
        return Err(error(
            "daemon_stopping",
            "the daemon has stopped accepting requests",
        ));
    }
    let reader = d.reads.reader().map_err(|e| store_error(&e))?;
    let file = tempfile::tempfile_in(&d.archive.root).map_err(|e| error("io", e))?;
    super::query::feed::prepare(&reader, since, now_ms(), file).map_err(|f| {
        Response::Error {
            kind: f.kind,
            message: f.message,
            detail: f.detail,
        }
        .masked()
    })
}

/// Answers one request. Reads use the read pool; imports go to the loop
/// thread, which owns the writer.
pub fn handle(request: Request, d: &Daemon) -> Response {
    handle_inner(request, d).masked()
}

fn handle_inner(request: Request, d: &Daemon) -> Response {
    if d.stopping.load(Ordering::Acquire) {
        return error(
            "daemon_stopping",
            "the daemon has stopped accepting requests",
        );
    }
    if matches!(d.backing, BackingState::Blocked { .. }) && !request.read_only() {
        return writes_paused(d);
    }
    let reader = match d.reads.reader() {
        Ok(r) => r,
        Err(StoreError::ReadPoolClosed) => {
            return error("daemon_stopping", StoreError::ReadPoolClosed);
        }
        Err(e) => return error("store", e),
    };
    match request {
        Request::BrowserEnqueue { action } => {
            let enabled = match reader.setting("browser.actions", &machine_id()) {
                Ok(value) => value.as_deref() == Some("on"),
                Err(error) => return store_error(&error),
            };
            let ctx = match crate::browser::Ctx::at(&d.config, &d.archive) {
                Ok(ctx) => ctx,
                Err(error) => return self::error("store", error),
            };
            match crate::browser::actions::enqueue(&ctx, &action, enabled) {
                Ok((created, action)) => {
                    ok(serde_json::json!({ "ok": true, "created": created, "action": action }))
                }
                Err(refused) => browser_refusal(refused),
            }
        }
        Request::BrowserUpload { bytes } => {
            let ctx = match crate::browser::Ctx::at(&d.config, &d.archive) {
                Ok(ctx) => ctx,
                Err(error) => return self::error("store", error),
            };
            match crate::browser::actions::upload(&ctx, &bytes) {
                Ok(value) => ok(value),
                Err(refused) => browser_refusal(refused),
            }
        }
        Request::Changes { .. } => error(
            "stream_required",
            "the change feed uses the finite socket stream",
        ),
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
                "signals": d.signals.lock().map(|s| s.clone()).unwrap_or_default(),
                "backing": d.backing,
            }))
        }
        Request::Query {
            text,
            cursor,
            page_size,
            why,
            session,
            fields,
        } => {
            let provider = crate::embeddings::Provider::from_settings(&reader, &machine_id())
                .ok()
                .flatten();
            let embeddings = crate::embeddings::available(&reader, provider.as_ref());
            let near = |q: &str| -> Result<String, String> {
                let Some(p) = &provider else {
                    return Ok("[]".into());
                };
                crate::embeddings::nearest_json(&reader, p, q).map_err(|e| e.to_string())
            };
            let ask = super::query::Ask {
                text: &text,
                cursor: cursor.as_deref(),
                page_size,
                why,
                session,
                now: now_ms(),
                // The operator's zone and 04:00 day start; there are no
                // `day.zone` or `day.logical_start` settings yet.
                day: polylogue_model::LogicalDay::system(),
                embeddings,
                near: Some(&near),
                fields: &fields,
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
            session,
            view,
            format,
            logical,
            budget,
            cursor,
        } => read::run(
            &reader,
            &session,
            read::Options {
                view: &view,
                format: &format,
                logical,
                budget,
                cursor: cursor.as_deref(),
            },
        ),

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
        Request::Embed {
            what,
            query,
            limit,
            yes,
        } => crate::embeddings::handle(&reader, &what, query.as_deref(), limit, yes),
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
        Request::Keep {
            request_id,
            text,
            name,
        } => {
            let ask = super::query::Ask {
                text: &text,
                cursor: None,
                page_size: None,
                why: false,
                session: None,
                now: now_ms(),
                day: polylogue_model::LogicalDay::system(),
                embeddings: false,
                near: None,
                fields: &[],
            };
            let kept = match super::query::keep(&reader, ask) {
                Ok(k) => k,
                Err(f) => {
                    return Response::Error {
                        kind: f.kind,
                        message: f.message,
                        detail: f.detail,
                    };
                }
            };
            drop(reader);
            let view = serde_json::json!({ "now": kept["now"] }).to_string();
            write(
                d,
                request_id,
                UserWrite::KeepResult {
                    name,
                    query: text,
                    rows_json: kept.to_string(),
                    view,
                },
            )
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
        Request::Resumable { cwd, limit } => {
            match crate::resume::resumable(&reader, &cwd, limit.unwrap_or(10)) {
                Ok(c) => ok(c),
                Err(e) => store_error(&e),
            }
        }
        Request::Continue { session } => match crate::resume::plan(&reader, &session) {
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

/// Whole sessions in the structured export format (`json`, read back by
/// `import`; credentials masked at the response boundary) or rendered as Markdown or HTML.
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
    let Some(format) = polylogue_render::Format::parse(format) else {
        return error("bad_request", format!("unknown export format {format:?}"));
    };
    match polylogue_render::export(doc, format) {
        Ok(value) => ok(value),
        Err(error) => self::error("internal", error),
    }
}

/// Owns the application workers and writer lock. Transports must stop
/// before this runtime; external daemon references cannot keep workers alive.
pub struct Runtime {
    pub daemon: Arc<Daemon>,
    signals_stop: Option<Sender<()>>,
    signals: Option<JoinHandle<()>>,
    engine: Option<JoinHandle<std::io::Result<()>>>,
    #[cfg(feature = "sinex")]
    publisher: Option<crate::sinex::Publisher>,
    lock: Option<polylogue_store::ArchiveLock>,
}

impl Runtime {
    /// Wake pending read admissions before joining transports or workers.
    /// Admitted readers and writes keep their existing completion semantics.
    pub fn stop_admission(&self) {
        self.daemon.stopping.store(true, Ordering::Release);
        self.daemon.reads.close();
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        self.stop_admission();
        if let Some(stop) = self.signals_stop.take() {
            let _ = stop.send(());
        }
        let mut failure = None;
        if let Some(signals) = self.signals.take()
            && signals.join().is_err()
        {
            failure = Some(std::io::Error::other("signal worker panicked"));
        }
        if let Ok(tx) = self.daemon.loop_tx.lock() {
            let _ = tx.send(LoopMsg::Stop);
        }
        if let Some(engine) = self.engine.take() {
            match engine.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    failure.get_or_insert(error);
                }
                Err(_) => {
                    failure.get_or_insert_with(|| {
                        std::io::Error::other("convergence worker panicked")
                    });
                }
            }
        }
        #[cfg(feature = "sinex")]
        if let Some(mut publisher) = self.publisher.take()
            && let Err(error) = publisher.shutdown()
        {
            failure.get_or_insert(error);
        }
        self.daemon.reads.close();
        self.lock.take();
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: shutdown: {error}");
        }
    }
}

/// The persistence choice or a configuration failure, never a fallback mode.
pub enum Backing {
    Standalone,
    #[cfg(feature = "sinex")]
    Sinex {
        config: polylogue_sinex::Config,
        path: std::path::PathBuf,
    },
    Blocked {
        mode: Option<&'static str>,
        path: std::path::PathBuf,
        error: String,
    },
}

#[derive(serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum BackingState {
    Standalone,
    #[cfg(feature = "sinex")]
    Primary,
    Blocked {
        mode: Option<&'static str>,
        path: std::path::PathBuf,
        error: String,
    },
}

/// Opens the archive and starts owned application workers. Startup errors
/// join any workers already started before releasing the archive lock.
pub fn start(root: &std::path::Path, config: Config, backing: Backing) -> std::io::Result<Runtime> {
    let to_io = |e: &dyn std::fmt::Display| std::io::Error::other(e.to_string());
    let (archive, lock) = Archive::open_or_create(root).map_err(|e| to_io(&e))?;
    let paths = config.clone();
    #[allow(unused_mut)]
    let mut config = config;
    #[cfg(feature = "sinex")]
    let mut publisher = None;
    let backing = match backing {
        Backing::Standalone => BackingState::Standalone,
        Backing::Blocked { mode, path, error } => BackingState::Blocked { mode, path, error },
        #[cfg(feature = "sinex")]
        Backing::Sinex {
            config: sinex,
            path,
        } => match crate::sinex::start(&archive, &mut config, sinex) {
            Ok(started) => {
                publisher = Some(started);
                BackingState::Primary
            }
            Err(error) => BackingState::Blocked {
                mode: Some("primary"),
                path,
                error: error.to_string(),
            },
        },
    };
    let mut engine = if matches!(backing, BackingState::Blocked { .. }) {
        None
    } else {
        Some(Engine::new(config, archive.clone()).map_err(|e| to_io(&e))?)
    };
    let writes = engine
        .as_ref()
        .map(Engine::writer)
        .transpose()
        .map_err(|e| to_io(&e))?;
    let per_reader = paths
        .reader_memory_per_connection()
        .map_err(|e| to_io(&e))?;
    let reads = ReadPool::with_budget(&archive, paths.workers, per_reader * paths.workers as u64)
        .map_err(|e| to_io(&e))?;
    let (tx, rx) = channel();
    let status = Arc::new(Mutex::new(Pass::default()));
    let daemon = Arc::new(Daemon {
        archive,
        reads,
        config: paths,
        writes,
        backing,
        loop_tx: Mutex::new(tx.clone()),
        status: Arc::clone(&status),
        started: Instant::now(),
        signals: Mutex::new(serde_json::Value::Null),
        stopping: AtomicBool::new(false),
    });
    let mut runtime = Runtime {
        daemon,
        signals_stop: None,
        signals: None,
        engine: None,
        #[cfg(feature = "sinex")]
        publisher,
        lock: Some(lock),
    };
    let Some(mut engine) = engine.take() else {
        return Ok(runtime);
    };
    let (ready, wait) = sync_channel(1);
    runtime.engine = Some(
        std::thread::Builder::new()
            .name("polylogue-loop".into())
            .spawn(move || {
                let started = ready.clone();
                let result = engine.run(tx, &rx, &status, || {
                    let _ = started.send(Ok(()));
                });
                if let Err(error) = &result {
                    let _ = ready.try_send(Err(error.to_string()));
                }
                result.map_err(|error| std::io::Error::other(error.to_string()))
            })?,
    );
    wait.recv()
        .map_err(|_| std::io::Error::other("convergence worker exited during startup"))?
        .map_err(std::io::Error::other)?;
    let (stop, stopped) = channel();
    runtime.signals_stop = Some(stop);
    let daemon = Arc::clone(&runtime.daemon);
    runtime.signals = Some(
        std::thread::Builder::new()
            .name("polylogue-signals".into())
            .spawn(move || crate::signals::run(&daemon, &stopped))?,
    );
    Ok(runtime)
}

mod read;

#[cfg(test)]
mod tests;
