//! Application operations and their shared archive state. Socket, HTTP and
//! MCP adapters use the same typed requests; transport framing stays outside.

use super::{ReadTarget, Request, Response};
use crate::briefing::{self, Ask};
use polylogue_engine::{Config, Engine, LoopMsg, LoopStatus};
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
    page_ends: read::page_ends::Cache,
    /// User writes (notes, deliveries, settings) through the one writer.
    pub(crate) writes: Option<WriterSender>,
    mode: ModeState,
    loop_tx: Mutex<Sender<LoopMsg>>,
    status: Arc<Mutex<LoopStatus>>,
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
        StoreError::IncompleteParents(issue) => Response::Error {
            kind: "incomplete_parents".into(),
            message: e.to_string(),
            detail: Some(serde_json::json!({"issue": issue})),
        },
        StoreError::ExcisionRefused { target, .. } => Response::Error {
            kind: "unsupported_excision".into(),
            message: e.to_string(),
            detail: Some(serde_json::json!({"target": target})),
        },
        StoreError::ExcisionInsideZip { input, file } => Response::Error {
            kind: "unsupported_excision".into(),
            message: e.to_string(),
            detail: Some(serde_json::json!({"input": input, "format": "zip", "file": file})),
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
        let _invalidation = d.page_ends.mutation();
        match writes.user(UserRequest::now(request_id, write)) {
            Ok(out) => delivery = Some(out.id),
            Err(e) => return store_error(&e),
        }
    }
    ok(serde_json::json!({ "delivery": delivery, "prepared": prepared }))
}

fn writer(d: &Daemon) -> Result<&WriterSender, Response> {
    d.writes.as_ref().ok_or_else(|| d.mode.write_refusal())
}

fn write(d: &Daemon, request_id: String, w: UserWrite) -> Response {
    let writes = match writer(d) {
        Ok(writes) => writes,
        Err(response) => return response,
    };
    let _invalidation = d.page_ends.mutation();
    match writes.user(UserRequest::now(request_id, w)) {
        Ok(out) => ok(out),
        Err(e) => store_error(&e),
    }
}

pub(crate) fn browser_refusal(refused: crate::browser::actions::Refused) -> Response {
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

/// Refuses new requests once the daemon is stopping.
pub(super) fn accepting(d: &Daemon) -> Result<(), Response> {
    if d.stopping.load(Ordering::Acquire) {
        return Err(error(
            "daemon_stopping",
            "the daemon has stopped accepting requests",
        ));
    }
    Ok(())
}

/// Prepares a complete finite feed through the same checks and read pool.
/// The returned unnamed file holds the bytes to send, never a live Reader.
pub(crate) fn change_feed(d: &Daemon, since: Option<&str>) -> Result<std::fs::File, Response> {
    accepting(d)?;
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
    if let Err(response) = accepting(d) {
        return response;
    }
    if d.mode.blocked() && !request.read_only() {
        return d.mode.write_refusal();
    }
    let read_epoch = d.page_ends.epoch();
    let reader = match d.reads.reader() {
        Ok(r) => r,
        Err(StoreError::ReadPoolClosed) => {
            return error("daemon_stopping", StoreError::ReadPoolClosed);
        }
        Err(e) => return error("store", e),
    };
    match request {
        Request::Attachment {
            message,
            block,
            attachment,
            reveal,
            cursor,
            ..
        } => {
            if reveal {
                return error(
                    "operator_required",
                    "exact attachment bytes require the interactive CLI operator path",
                );
            }
            if cursor.is_some() {
                return error("bad_request", "a cursor continues an exact attachment read");
            }
            match reader.masked_attachment(&message, block, attachment) {
                Ok(resolution) => ok(resolution),
                Err(e) => store_error(&e),
            }
        }
        Request::ChatSend { .. }
        | Request::ChatActions
        | Request::ChatAction { .. }
        | Request::ChatClaim { .. }
        | Request::ChatReport { .. }
        | Request::ChatDecide { .. }
        | Request::ChatAttachment { .. }
        | Request::BackfillStart { .. }
        | Request::BackfillJobs
        | Request::BackfillCancel { .. }
        | Request::BackfillClaim { .. }
        | Request::BackfillReport { .. }
        | Request::Conversations { .. } => browser::handle(request, d, reader),
        Request::Changes { .. } => error(
            "stream_required",
            "the change feed uses the finite socket stream",
        ),
        Request::Status => {
            let store = match reader.status() {
                Ok(s) => s,
                Err(e) => return error("store", e),
            };
            // Read after the store counts: a loop seen idle now has finished
            // the pass that wrote any row those counts include.
            let loop_status = d.status.lock().map(|s| s.clone()).unwrap_or_default();
            ok(serde_json::json!({
                "archive": d.archive.root,
                "pid": std::process::id(),
                "uptime_seconds": d.started.elapsed().as_secs(),
                "store": store,
                "busy": loop_status.busy,
                "passes": loop_status.passes,
                "last_pass": loop_status.last_pass,
                "signals": d.signals.lock().map(|s| s.clone()).unwrap_or_default(),
                "mode": d.mode,
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
            target,
            view,
            format,
            with_parents,
            budget,
            cursor,
        } => {
            let options = read::Options {
                view: &view,
                format: &format,
                with_parents,
                budget,
                cursor: cursor.as_deref(),
            };
            match &target {
                ReadTarget::Session(session) => {
                    read::run(&reader, session, options, &d.page_ends, read_epoch)
                }
                ReadTarget::Document(document) => read::document(&reader, document, options),
                ReadTarget::Event(event) => read::event(&reader, event, options),
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
                return error("internal", "the loop has stopped");
            }
            match wait.recv() {
                Ok(Ok(report)) => ok(report),
                Ok(Err(e)) => error("import", e),
                Err(_) => error("internal", "the loop has stopped"),
            }
        }
        Request::Export { targets, format } => export(&reader, &targets, &format),
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
            let target = match polylogue_engine::ExciseTarget::parse(&target) {
                Ok(target) => target,
                Err(e) => return error("bad_request", e),
            };
            let request = polylogue_engine::ExciseRequest {
                id: request_id,
                target,
                reason,
                dry_run,
            };
            let _invalidation = d.page_ends.mutation();
            let (reply, wait) = sync_channel(1);
            let sent = d
                .loop_tx
                .lock()
                .map(|tx| tx.send(LoopMsg::Excise { request, reply }).is_ok())
                .unwrap_or(false);
            if !sent {
                return error("internal", "the loop has stopped");
            }
            match wait.recv() {
                Ok(Ok(done)) => ok(done),
                Ok(Err(polylogue_engine::EngineError::Store(
                    e @ (StoreError::NotFound(_)
                    | StoreError::ExcisionRefused { .. }
                    | StoreError::ExcisionInsideZip { .. }),
                ))) => store_error(&e),
                Ok(Err(e)) => error("excise", e),
                Err(_) => error("internal", "the loop has stopped"),
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
            let read_at = serde_json::json!({ "now": kept["now"] }).to_string();
            write(
                d,
                request_id,
                UserWrite::KeepResult {
                    name,
                    query: text,
                    rows_json: kept.to_string(),
                    read_at,
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
            match crate::resume::resumable(&reader, &cwd, limit.unwrap_or(10), user_writes::now()) {
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
    }
}

/// Whole sessions in the structured export format (`json`, read back by
/// `import`; credentials masked at the response boundary) or rendered as Markdown or HTML.
fn export(reader: &polylogue_store::Reader<'_>, targets: &[ReadTarget], format: &str) -> Response {
    let mut sessions = Vec::new();
    for id in targets.iter().filter_map(|t| match t {
        ReadTarget::Session(id) => Some(id),
        _ => None,
    }) {
        match reader.session_export(id) {
            Ok(Some(s)) => sessions.push(s),
            Ok(None) => return error("not_found", format!("no session {id}")),
            Err(e) => return store_error(&e),
        }
    }
    let mut documents = Vec::new();
    for id in targets.iter().filter_map(|t| match t {
        ReadTarget::Document(id) => Some(id),
        _ => None,
    }) {
        match reader.document(id) {
            Ok(Some(doc)) => documents.push(doc),
            Ok(None) => return error("not_found", format!("no document {id}")),
            Err(e) => return store_error(&e),
        }
    }
    let mut events = Vec::new();
    for id in targets.iter().filter_map(|t| match t {
        ReadTarget::Event(id) => Some(id),
        _ => None,
    }) {
        match reader.event_export(id) {
            Ok(Some(event)) => events.push(event),
            Ok(None) => return error("not_found", format!("no event {id}")),
            Err(e) => return store_error(&e),
        }
    }
    let doc = polylogue_model::export::Document::new(sessions)
        .with_documents(documents)
        .with_events(events);
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
    lock: Option<polylogue_store::ArchiveLock>,
}

impl Runtime {
    /// Refuses new requests and wakes readers waiting for a reader slot,
    /// before transports and workers are joined. Readers and writes already
    /// running finish as usual.
    pub fn stop_accepting(&self) {
        self.daemon.stopping.store(true, Ordering::Release);
        self.daemon.page_ends.clear();
        self.daemon.reads.close();
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        self.stop_accepting();
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
                    failure.get_or_insert_with(|| std::io::Error::other("loop thread panicked"));
                }
            }
        }
        self.daemon.page_ends.clear();
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

/// Standalone, primary on Sinex, or blocked: a Sinex configuration that
/// cannot be read, or a Sinex that cannot be reached at start; never a
/// fallback to another mode. Only a build with the
/// `sinex` feature reads a Sinex configuration, so only it can be blocked.
pub enum ArchiveMode {
    Standalone,
    #[cfg(feature = "sinex")]
    Primary {
        config: polylogue_sinex::Config,
        path: std::path::PathBuf,
    },
    #[cfg(feature = "sinex")]
    Blocked {
        /// The mode the configuration asked for, when it could be read.
        configured: Option<&'static str>,
        path: std::path::PathBuf,
        error: String,
    },
}

#[derive(serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum ModeState {
    Standalone,
    #[cfg(feature = "sinex")]
    Primary,
    #[cfg(feature = "sinex")]
    Blocked {
        configured: Option<&'static str>,
        path: std::path::PathBuf,
        error: String,
    },
}

impl ModeState {
    /// Writes are paused until the Sinex configuration is fixed or Sinex
    /// can be reached.
    fn blocked(&self) -> bool {
        #[cfg(feature = "sinex")]
        {
            matches!(self, Self::Blocked { .. })
        }
        #[cfg(not(feature = "sinex"))]
        {
            false
        }
    }

    /// The answer to a write the daemon cannot take.
    fn write_refusal(&self) -> Response {
        #[cfg(feature = "sinex")]
        if let Self::Blocked {
            path,
            error: reason,
            ..
        } = self
        {
            return Response::Error {
                kind: "sinex_unavailable".into(),
                message: format!("archive writes are paused: {reason}"),
                detail: Some(serde_json::json!({ "config_path": path, "error": reason })),
            };
        }
        store_error(&StoreError::WriterGone)
    }
}

/// Opens the archive and starts owned application workers. Startup errors
/// join any workers already started before releasing the archive lock.
pub fn start(
    root: &std::path::Path,
    config: Config,
    mode: ArchiveMode,
) -> std::io::Result<Runtime> {
    let to_io = |e: &dyn std::fmt::Display| std::io::Error::other(e.to_string());
    let settings = config.clone();
    let (archive, lock, mode, engine) = match mode {
        ArchiveMode::Standalone => {
            let (archive, lock) = Archive::open_or_create(root).map_err(std::io::Error::other)?;
            let engine = Engine::new(config, archive.clone()).map_err(|e| to_io(&e))?;
            (archive, lock, ModeState::Standalone, Some(engine))
        }
        #[cfg(feature = "sinex")]
        ArchiveMode::Blocked {
            configured,
            path,
            error,
        } => {
            let (archive, lock) = Archive::open_or_create(root).map_err(std::io::Error::other)?;
            let state = ModeState::Blocked {
                configured,
                path,
                error,
            };
            (archive, lock, state, None)
        }
        #[cfg(feature = "sinex")]
        ArchiveMode::Primary {
            config: sinex,
            path,
        } => match crate::sinex::open(root, &sinex)? {
            (archive, lock, Ok(connection)) => {
                let engine =
                    Engine::on_sinex(config, archive.clone(), connection).map_err(|e| to_io(&e))?;
                (archive, lock, ModeState::Primary, Some(engine))
            }
            (archive, lock, Err(error)) => {
                let state = ModeState::Blocked {
                    configured: Some("primary"),
                    path,
                    error,
                };
                (archive, lock, state, None)
            }
        },
    };
    let mut engine = engine;
    let writes = engine
        .as_ref()
        .map(Engine::writer)
        .transpose()
        .map_err(|e| to_io(&e))?;
    let per_reader = settings
        .reader_memory_per_connection()
        .map_err(|e| to_io(&e))?;
    let reads = ReadPool::with_budget(
        &archive,
        settings.workers,
        per_reader * settings.workers as u64,
    )
    .map_err(|e| to_io(&e))?;
    let (tx, rx) = channel();
    // The loop's first pass starts as soon as it runs.
    let status = Arc::new(Mutex::new(LoopStatus {
        busy: engine.is_some(),
        ..LoopStatus::default()
    }));
    let daemon = Arc::new(Daemon {
        archive,
        reads,
        page_ends: read::page_ends::Cache::default(),
        writes,
        mode,
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
        .map_err(|_| std::io::Error::other("loop thread exited during startup"))?
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

mod browser;
mod read;

#[cfg(test)]
mod tests;
