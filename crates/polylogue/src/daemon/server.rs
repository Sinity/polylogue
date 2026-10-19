//! Serving requests on the Unix socket: one JSON request per line, one
//! JSON response per line. The daemon holds the archive lock, runs the
//! convergence loop on its own thread, and answers reads from a pool of
//! read connections while the loop writes.

use super::{Request, Response};
use crate::briefing::{self, Ask};
use polylogue_engine::{Config, Engine, LoopMsg, Pass};
use polylogue_model::Author;
use polylogue_store::user_writes::{self as user_writes, machine_id};
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
    /// `$XDG_STATE_HOME/polylogue` as the loop sees it: where the browser
    /// host spools captures for the loop to read.
    pub state: std::path::PathBuf,
    /// User writes (notes, deliveries, settings) through the one writer.
    writes: WriterSender,
    loop_tx: Mutex<Sender<LoopMsg>>,
    status: Arc<Mutex<Pass>>,
    started: Instant,
    /// The signal rules in force and the ones refused, for `status`.
    signals: Mutex<serde_json::Value>,
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

/// Signal deliveries already recorded, for the evaluator to start from.
fn signal_history(d: &Daemon) -> Vec<crate::signals::Sent> {
    let Ok(r) = d.reads.reader() else {
        return vec![];
    };
    let rows = r
        .query(
            "SELECT at, refs_json FROM a.delivery WHERE sender = 'signal' ORDER BY at",
            &[],
        )
        .unwrap_or_default();
    rows.iter()
        .filter_map(|row| {
            let refs: Vec<String> = serde_json::from_str(row["refs_json"].as_str()?).ok()?;
            let agent = refs.iter().any(|r| r == "interrupt:agent");
            crate::signals::sent_from(&refs, row["at"].as_i64()?, agent)
        })
        .collect()
}

/// Evaluates the signal rules after each change and at each `for`
/// deadline, so a duration fires with no new data (law 140), and records
/// every handoff as a delivery.
fn signal_loop(d: &Daemon) {
    use crate::signals::{self, How, Rule, Unusable};
    use polylogue_model::{LogicalDay, NoteKind};
    let day = LogicalDay::system();
    let mut eval = signals::Evaluator::new(&signal_history(d));
    loop {
        let now = now_ms();
        let Ok(reader) = d.reads.reader() else {
            std::thread::sleep(std::time::Duration::from_secs(30));
            continue;
        };
        let filter = polylogue_store::NoteFilter {
            kinds: vec![NoteKind::Signal],
            ..Default::default()
        };
        let mut rules: Vec<Rule> = vec![];
        let mut unusable: Vec<Unusable> = vec![];
        for note in reader.notes(&filter).unwrap_or_default() {
            match signals::rule(&note) {
                Ok(r) => rules.push(r),
                Err(u) => unusable.push(u),
            }
        }
        let mut matches = std::collections::HashMap::new();
        let mut usable = vec![];
        for rule in rules {
            let ask = super::query::Ask {
                text: &rule.query,
                cursor: None,
                page_size: Some(1_000),
                why: false,
                session: None,
                now,
                day: day.clone(),
                embeddings: false,
                near: None,
                fields: &[],
            };
            match super::query::run(&reader, ask) {
                Ok(v) => {
                    let rows = v["rows"]
                        .as_array()
                        .map(|rows| rows.iter().map(signals::row_key).collect())
                        .unwrap_or_default();
                    matches.insert(rule.id.clone(), rows);
                    usable.push(rule);
                }
                Err(f) => unusable.push(Unusable {
                    id: rule.id.clone(),
                    reason: format!("its query fails: {}", f.message),
                }),
            }
        }
        if let Ok(mut s) = d.signals.lock() {
            *s = signals::summary(&usable, &unusable);
        }
        for h in eval.step(now, &usable, &matches, |t| day.local_hour(t)) {
            let Some(rule) = usable.iter().find(|r| r.id == h.rule) else {
                continue;
            };
            let mut refs = signals::refs(&h.rule, &h.row);
            if h.how == How::Interrupt && !rule.operator {
                refs.push("interrupt:agent".into());
            }
            let id = polylogue_store::user_writes::new_request_id();
            let delivery = Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "signal".into(),
                event: "signal".into(),
                cwd: None,
                text: signals::notice(rule, &h),
                refs,
                left_out: vec![],
            };
            if d.writes
                .user(UserRequest::now(id.clone(), UserWrite::Deliver(delivery)))
                .is_err()
            {
                continue;
            }
            // A notice is shown by `status`, briefings and the change feed;
            // an interrupt needs the notifier `signals.toml` names, which is
            // not built, so it stays a notice and says so.
            let event = match h.how {
                How::Notice => None,
                How::Interrupt => Some((
                    polylogue_store::DeliveryOutcome::NotSent,
                    "no notifier is configured",
                )),
                How::Deferred => Some((
                    polylogue_store::DeliveryOutcome::NotSent,
                    "rate limit: kept for the next briefing",
                )),
            };
            if let Some((what, detail)) = event {
                let _ = d.writes.user(UserRequest::now(
                    polylogue_store::user_writes::new_request_id(),
                    UserWrite::DeliveryEvent {
                        delivery: id,
                        what,
                        detail: Some(detail.into()),
                    },
                ));
            }
        }
        let wait = eval
            .next_due()
            .map_or(30_000, |due| (due - now_ms()).clamp(1_000, 30_000));
        std::thread::sleep(std::time::Duration::from_millis(wait as u64));
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
                "signals": d.signals.lock().map(|s| s.clone()).unwrap_or_default(),
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
            fields,
        } => {
            let provider = crate::embeddings::Provider::from_settings(&reader, &machine_id())
                .ok()
                .flatten();
            let embeddings = crate::embeddings::available(&d.archive, provider.as_ref());
            let near = |q: &str| -> Result<String, String> {
                let Some(p) = &provider else {
                    return Ok("[]".into());
                };
                crate::embeddings::nearest_json(&d.archive, &reader, p, q)
                    .map_err(|e| e.to_string())
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
        } => {
            let s = match reader.session(&session) {
                Ok(Some(s)) => s,
                Ok(None) => return error("not_found", format!("no session {session}")),
                Err(e) => return error("store", e),
            };
            match reader.messages(&session, logical) {
                Ok(messages) => read(&s, messages, &view, &format, budget, cursor.as_deref()),
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
        Request::Embed {
            what,
            query,
            limit,
            yes,
        } => embed(d, &reader, &what, query.as_deref(), limit, yes),
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

/// Embedding status, cost preview and build ([`crate::embeddings`]).
fn embed(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    what: &str,
    query: Option<&str>,
    limit: Option<u64>,
    yes: bool,
) -> Response {
    use crate::embeddings::{self, ProviderKind};
    let provider = match embeddings::Provider::from_settings(reader, &machine_id()) {
        Ok(p) => p,
        Err(e) => return error("bad_request", e),
    };
    if what == "status" {
        let vectors =
            polylogue_store::vectors::Vectors::open_read(&d.archive).and_then(|v| v.status());
        return match vectors {
            Ok(s) => ok(serde_json::json!({
                "provider": provider.as_ref().map(|p| serde_json::json!({
                    "endpoint": p.endpoint, "model": p.model,
                })),
                "vectors": s,
            })),
            Err(e) => store_error(&e),
        };
    }
    let Some(provider) = provider else {
        return error(
            "embeddings_off",
            "embeddings are off; `polylogue setting embed.provider local` (or `voyage`) turns them on",
        );
    };
    // A query limits the work to the sessions it finds.
    let sessions: Option<Vec<String>> = match query.filter(|q| !q.trim().is_empty()) {
        None => None,
        Some(q) => {
            let ask = super::query::Ask {
                text: q,
                cursor: None,
                page_size: Some(1_000_000),
                why: false,
                session: None,
                now: now_ms(),
                day: polylogue_model::LogicalDay::system(),
                embeddings: false,
                near: None,
                fields: &[],
            };
            match super::query::run(reader, ask) {
                Ok(v) if v["unit"] == "sessions" => Some(
                    v["rows"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|r| r["id"].as_str().map(str::to_owned))
                        .collect(),
                ),
                Ok(_) => return error("bad_request", "embed takes a sessions query"),
                Err(f) => {
                    return Response::Error {
                        kind: f.kind,
                        message: f.message,
                        detail: f.detail,
                    };
                }
            }
        }
    };
    let result = match what {
        "preview" => embeddings::preview(&d.archive, reader, &provider, sessions.as_deref(), limit)
            .map(|p| serde_json::to_value(p).unwrap_or_default()),
        "build" => {
            if provider.kind == ProviderKind::Voyage && !yes {
                return error(
                    "confirm",
                    "Voyage is paid: `polylogue embed preview` shows the cost; build with --yes",
                );
            }
            embeddings::build(&d.archive, reader, &provider, sessions.as_deref(), limit)
                .map(|b| serde_json::to_value(b).unwrap_or_default())
        }
        other => {
            return error(
                "bad_request",
                format!("embed {other}: status, preview or build"),
            );
        }
    };
    match result {
        Ok(v) => ok(v),
        Err(e) => error("embed", e),
    }
}

/// One session in a view: as JSON messages, rendered in a format, or, with
/// a token budget, as one page of text for a model with the cursor of the
/// next.
fn read(
    session: &polylogue_model::Session,
    mut messages: Vec<polylogue_model::Message>,
    view: &str,
    format: &str,
    budget: Option<usize>,
    cursor: Option<&str>,
) -> Response {
    let Some(v) = polylogue_render::View::parse(view) else {
        return error("bad_request", format!("unknown view {view:?}"));
    };
    if let Some(budget) = budget {
        let page = polylogue_render::model_pages::page(&messages, v, budget, cursor);
        return ok(serde_json::json!({ "session": session, "text": page.text, "next": page.next }));
    }
    if cursor.is_some() {
        return error("bad_request", "a cursor continues a read with a budget");
    }
    if format == "json" {
        if v != polylogue_render::View::Full {
            for m in &mut messages {
                let kept: Vec<_> = m.blocks.iter().filter(|b| v.keeps(m, b)).cloned().collect();
                m.blocks = kept;
            }
            messages.retain(|m| !m.blocks.is_empty());
        }
        return ok(serde_json::json!({ "session": session, "messages": messages }));
    }
    let Some(f) = polylogue_render::Format::parse(format) else {
        return error("bad_request", format!("unknown format {format:?}"));
    };
    let mut items = vec![polylogue_render::Rendered::SessionHeader(session)];
    items.extend(messages.iter().map(polylogue_render::Rendered::Message));
    let mut out = Vec::new();
    if let Err(e) = polylogue_render::render(&items, f, v, &mut out) {
        return error("internal", e);
    }
    ok(serde_json::json!({ "format": format, "content": String::from_utf8_lossy(&out) }))
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

/// Opens the archive at `root`, starts its convergence loop on a thread of
/// its own and returns the shared daemon state. The lock is held for as
/// long as the returned value lives.
pub fn start(
    root: &std::path::Path,
    config: Config,
) -> std::io::Result<(Arc<Daemon>, polylogue_store::ArchiveLock)> {
    let to_io = |e: &dyn std::fmt::Display| std::io::Error::other(e.to_string());
    let (archive, lock) = Archive::open_or_create(root).map_err(|e| to_io(&e))?;
    let state = config
        .root(polylogue_sources::browser_capture::ROOT)
        .join("polylogue");
    #[allow(unused_mut)]
    let mut config = config;
    #[cfg(feature = "sinex")]
    crate::sinex::start(&archive, &mut config)?;
    let mut engine = Engine::new(config, archive.clone()).map_err(|e| to_io(&e))?;
    let writes = engine.writer().map_err(|e| to_io(&e))?;
    let reads = ReadPool::open(&archive).map_err(|e| to_io(&e))?;
    let (tx, rx) = channel();
    let status = Arc::new(Mutex::new(Pass::default()));
    let daemon = Arc::new(Daemon {
        archive,
        reads,
        state,
        writes,
        loop_tx: Mutex::new(tx.clone()),
        status: Arc::clone(&status),
        started: Instant::now(),
        signals: Mutex::new(serde_json::Value::Null),
    });
    {
        let d = Arc::clone(&daemon);
        std::thread::Builder::new()
            .name("polylogue-signals".into())
            .spawn(move || signal_loop(&d))?;
    }
    std::thread::Builder::new()
        .name("polylogue-loop".into())
        .spawn(move || {
            if let Err(e) = engine.run(tx, &rx, &status) {
                eprintln!("polylogue: the convergence loop stopped: {e}");
            }
        })?;
    Ok((daemon, lock))
}

/// Runs the daemon until it is killed: takes the archive lock, starts the
/// loop and serves clients.
pub fn serve() -> std::io::Result<()> {
    let root = Archive::default_root();
    let (daemon, _lock) = start(&root, Config::from_env())?;
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

#[cfg(test)]
mod tests;
