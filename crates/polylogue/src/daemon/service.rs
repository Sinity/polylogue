//! Application operations and their shared archive state. Socket, HTTP and
//! MCP adapters use the same typed requests; transport framing stays outside.

use super::{ReadTarget, Request, Response};
use crate::briefing::{self, Ask};
use polylogue_engine::{Config, Engine, LoopMsg, LoopStatus};
use polylogue_model::{Author, NewNote, NoteKind};
use polylogue_store::user_writes::{self as user_writes, machine_id};
use polylogue_store::{
    Archive, Delivery, ReadPool, StoreError, UserRequest, UserWrite, WriterSender,
};
use std::collections::HashSet;
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
    pub(crate) warm: super::warm::Warm,
    /// User writes (notes, deliveries, settings) through the one writer.
    pub(crate) writes: Option<WriterSender>,
    mode: ModeState,
    loop_tx: Mutex<Sender<LoopMsg>>,
    pub(crate) status: Arc<Mutex<LoopStatus>>,
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
        let provider = crate::embeddings::Provider::from_settings(&reader, &machine_id())
            .ok()
            .flatten();
        let near = |q: &str| match &provider {
            Some(p) => d.warm.similarity(&reader, p, q),
            None => Err("embeddings are off".into()),
        };
        let near: Option<&super::query::NearFn<'_>> = provider.is_some().then_some(&near);
        match briefing::prepare(&reader, ask, &request_id, user_writes::now(), near) {
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

fn repeated_request(
    reader: &polylogue_store::Reader<'_>,
    id: &str,
    intended: &UserWrite,
) -> Result<Option<UserWrite>, Response> {
    match reader.user_request(id).map_err(|e| store_error(&e))? {
        Some(previous) if intended.intent() == previous.write.intent() => Ok(Some(previous.write)),
        Some(_) => Err(store_error(&StoreError::Conflict(id.into()))),
        None => Ok(None),
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
        Request::Explain {
            subject,
            expression,
            unit,
            page_size,
            cursor,
            session,
        } => {
            let result = if subject == "query" {
                if unit.is_some() || page_size.is_some() || cursor.is_some() {
                    return error(
                        "bad_request",
                        "unit, page_size and cursor apply to declaration pages",
                    );
                }
                let Some(text) = expression.as_deref().filter(|s| !s.trim().is_empty()) else {
                    return error("bad_request", "expression is required for query");
                };
                let provider = crate::embeddings::Provider::from_settings(&reader, &machine_id())
                    .ok()
                    .flatten();
                let embeddings =
                    crate::embeddings::ranking_key(&reader, provider.as_ref()).is_some();
                super::query::explain(
                    &reader,
                    super::query::Ask {
                        text,
                        cursor: None,
                        page_size: None,
                        why: false,
                        session,
                        now: now_ms(),
                        day: None,
                        embeddings,
                        near: None,
                        fields: &[],
                    },
                )
            } else {
                crate::explain::answer(
                    &subject,
                    expression.as_deref(),
                    unit.as_deref(),
                    page_size,
                    cursor.as_deref(),
                )
            };
            match result {
                Ok(data) => ok(data),
                Err(f) => Response::Error {
                    kind: f.kind,
                    message: f.message,
                    detail: f.detail,
                },
            }
        }
        Request::Status => {
            let position = match reader.change_position() {
                Ok(p) => p.to_string(),
                Err(e) => return store_error(&e),
            };
            let counted = d.warm.counts(&reader, &position);
            // Read after the store counts: a loop seen idle now has finished
            // the pass that wrote any row those counts include.
            let loop_status = d.status.lock().map(|s| s.clone()).unwrap_or_default();
            ok(serde_json::json!({
                "archive": d.archive.root,
                "pid": std::process::id(),
                "uptime_seconds": d.started.elapsed().as_secs(),
                "store": counted.as_ref().map(|c| &c.status),
                "counted": counted.as_ref().map_or_else(
                    || serde_json::json!({ "state": "counting" }),
                    |c| serde_json::json!({
                        "position": c.position,
                        "current": c.position == position,
                        "seconds_ago": c.at.elapsed().as_secs_f64(),
                        "seconds_to_count": c.took.as_secs_f64(),
                    }),
                ),
                "busy": loop_status.busy,
                "passes": loop_status.passes,
                "last_pass": loop_status.last_pass,
                "signals": d.signals.lock().map(|s| s.clone()).unwrap_or_default(),
                "notices": crate::signals::recent(&reader, now_ms()),
                "warm": d.warm.summary(),
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
            let position = match reader.change_position() {
                Ok(p) => format!("{p}:{}", polylogue_store::vectors::writes()),
                Err(e) => return store_error(&e),
            };
            let key =
                serde_json::json!([text, cursor, page_size, why, session, fields]).to_string();
            if let Some(data) = d.warm.answer(&key, &position) {
                return Response::Ok { data };
            }
            let provider = crate::embeddings::Provider::from_settings(&reader, &machine_id())
                .ok()
                .flatten();
            let embeddings = crate::embeddings::ranking_key(&reader, provider.as_ref()).is_some();
            let near = |q: &str| match &provider {
                Some(p) => d.warm.similarity(&reader, p, q),
                None => Err("embeddings are off".into()),
            };
            let ask = super::query::Ask {
                text: &text,
                cursor: cursor.as_deref(),
                page_size,
                why,
                session,
                now: now_ms(),
                day: None,
                embeddings,
                near: Some(&near),
                fields: &fields,
            };
            match super::query::run(&reader, ask) {
                Ok(data) => {
                    d.warm.keep_answer(key, position, cursor.is_none(), &data);
                    Response::Ok { data }
                }
                Err(f) => Response::Error {
                    kind: f.kind,
                    message: f.message,
                    detail: f.detail,
                },
            }
        }
        Request::Complete { word } => {
            let position = match reader.change_position() {
                Ok(p) => p.to_string(),
                Err(e) => return store_error(&e),
            };
            match d.warm.completion(&reader, &position) {
                Ok(data) => ok(super::warm::completions(&data, &word)
                    .into_iter()
                    .map(|(value, help)| serde_json::json!({"value": value, "help": help}))
                    .collect::<Vec<_>>()),
                Err(e) => error("store", e),
            }
        }
        Request::Sql { statement } => match reader.query(&statement, &[]) {
            // The connection is read-only: a statement that writes fails.
            Ok(rows) => ok(rows),
            Err(e) => error("sql", e),
        },
        Request::Note { id } => match reader.note(&id) {
            Ok(Some(note)) => ok(note),
            Ok(None) => error("not_found", "the note revision is not available"),
            Err(e) => store_error(&e),
        },
        Request::Outline {
            session,
            with_parents,
            page_size,
            cursor,
        } => read::outline(
            &reader,
            &session,
            with_parents,
            page_size,
            cursor.as_deref(),
        ),
        Request::Read {
            target,
            view,
            format,
            with_parents,
            budget,
            cursor,
            start,
        } => {
            let options = read::Options {
                view: &view,
                format: &format,
                with_parents,
                budget,
                cursor: cursor.as_deref(),
                start: start.as_ref(),
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
        Request::Export {
            targets,
            format,
            user_state,
        } => export(&reader, &targets, &format, user_state),
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
            let intended = UserWrite::KeepResult {
                name: name.clone(),
                query: text.clone(),
                rows_json: String::new(),
                read_at: String::new(),
            };
            match repeated_request(&reader, &request_id, &intended) {
                Ok(Some(previous)) => {
                    drop(reader);
                    return write(d, request_id, previous);
                }
                Err(response) => return response,
                Ok(None) => {}
            }
            let provider = crate::embeddings::Provider::from_settings(&reader, &machine_id())
                .ok()
                .flatten();
            let embeddings = crate::embeddings::ranking_key(&reader, provider.as_ref()).is_some();
            let near = |q: &str| match &provider {
                Some(p) => d.warm.similarity(&reader, p, q),
                None => Err("embeddings are off".into()),
            };
            let ask = super::query::Ask {
                text: &text,
                cursor: None,
                page_size: None,
                why: false,
                session: None,
                now: now_ms(),
                day: None,
                embeddings,
                near: Some(&near),
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
        Request::Save {
            request_id,
            name,
            text,
            author,
        } => {
            let mut note = NewNote {
                kind: NoteKind::SavedQuery,
                key: Some(name.clone()),
                value: Some(text.clone()),
                author,
                targets: vec![],
                relations: vec![],
                revises: None,
                valid_until: None,
            };
            match repeated_request(
                &reader,
                &request_id,
                &UserWrite::SaveQuery { note: note.clone() },
            ) {
                Ok(Some(previous)) => {
                    drop(reader);
                    return write(d, request_id, previous);
                }
                Err(response) => return response,
                Ok(None) => {}
            }
            let revises = match saved_revision(&reader, &name, &text) {
                Ok(revises) => revises,
                Err(response) => return response,
            };
            drop(reader);
            note.revises = revises;
            write(d, request_id, UserWrite::SaveQuery { note })
        }
        Request::Watch {
            request_id,
            query,
            name,
            hold,
            notify,
            author,
        } => {
            if let Some(hold) = &hold
                && crate::signals::duration(hold).is_none()
            {
                return error(
                    "invalid",
                    format!("--for {hold:?} is not a duration such as 20m"),
                );
            }
            // A watch whose query cannot run would only be refused later.
            let ask = super::query::Ask {
                text: &query,
                cursor: None,
                page_size: Some(1),
                why: false,
                session: None,
                now: now_ms(),
                day: None,
                embeddings: false,
                near: None,
                fields: &[],
            };
            if let Err(f) = super::query::run(&reader, ask) {
                return Response::Error {
                    kind: f.kind,
                    message: f.message,
                    detail: f.detail,
                };
            }
            drop(reader);
            let mut rule = serde_json::json!({
                "name": name.clone().unwrap_or_else(|| query.clone()),
                "query": query,
                "fires": "on_new",
                "deliver": "notice",
                "interrupt": notify,
            });
            if let Some(hold) = hold {
                rule["for"] = serde_json::json!(hold);
            }
            let note = NewNote {
                kind: NoteKind::Signal,
                key: name,
                value: Some(rule.to_string()),
                author,
                targets: vec![],
                relations: vec![],
                revises: None,
                valid_until: None,
            };
            write(d, request_id, UserWrite::AddNote { note })
        }
        Request::AddNote { request_id, note } => {
            drop(reader);
            write(d, request_id, UserWrite::AddNote { note })
        }
        Request::AnnotateQuery {
            request_id,
            query,
            mut note,
        } => {
            let intended = UserWrite::AnnotateQuery {
                query: query.clone(),
                note: note.clone(),
            };
            let resolved = match repeated_request(&reader, &request_id, &intended) {
                Ok(Some(previous)) => previous,
                Ok(None) => {
                    note.targets = match query_targets(d, &reader, &query) {
                        Ok(targets) => targets,
                        Err(response) => return response,
                    };
                    UserWrite::AnnotateQuery { query, note }
                }
                Err(response) => return response,
            };
            drop(reader);
            match write(d, request_id, resolved) {
                Response::Ok { mut data } => {
                    let committed = d.reads.reader().and_then(|reader| {
                        reader.note(data["id"].as_str().expect("write result ID"))
                    });
                    match committed {
                        Ok(Some(note)) => data["targets"] = serde_json::json!(note.targets.len()),
                        Ok(None) => return error("not_found", "the committed note is absent"),
                        Err(e) => return store_error(&e),
                    }
                    Response::Ok { data }
                }
                other => other,
            }
        }

        Request::Judge {
            request_id,
            note,
            accepted,
            reason,
        } => {
            let found = match reader.note(&note) {
                Ok(Some(found)) => found,
                Ok(None) => return error("not_found", format!("no note {note}")),
                Err(e) => return store_error(&e),
            };
            if !found.author.is_agent() {
                return error(
                    "bad_request",
                    "only notes agents wrote are accepted or rejected; revise or withdraw the operator's own",
                );
            }
            drop(reader);
            let judgment = polylogue_model::NewNote {
                kind: polylogue_model::NoteKind::Judgment,
                key: Some(if accepted { "accepted" } else { "rejected" }.into()),
                value: reason,
                author: Author::Operator,
                targets: vec![polylogue_model::Target::Note(found.id)],
                relations: vec![],
                revises: None,
                valid_until: None,
            };
            write(d, request_id, UserWrite::AddNote { note: judgment })
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

/// The references of every row `query` returns, for a note on all of
/// them: sessions, messages, documents and notes.
fn query_targets(
    d: &Daemon,
    reader: &polylogue_store::Reader<'_>,
    query: &str,
) -> Result<Vec<polylogue_model::Target>, Response> {
    use polylogue_model::Target;
    let provider = crate::embeddings::Provider::from_settings(reader, &machine_id())
        .ok()
        .flatten();
    let embeddings = crate::embeddings::ranking_key(reader, provider.as_ref()).is_some();
    let near = |q: &str| match &provider {
        Some(provider) => d.warm.similarity(reader, provider, q),
        None => Err("embeddings are off".into()),
    };
    let now = now_ms();
    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    let mut cursor: Option<String> = None;
    loop {
        let ask = super::query::Ask {
            text: query,
            cursor: cursor.as_deref(),
            page_size: Some(1_000),
            why: false,
            session: None,
            now,
            day: None,
            embeddings,
            near: Some(&near),
            fields: &[],
        };
        let page = super::query::run(reader, ask).map_err(|f| Response::Error {
            kind: f.kind,
            message: f.message,
            detail: f.detail,
        })?;
        let unit = page["unit"].as_str().unwrap_or("");
        if !matches!(unit, "sessions" | "messages" | "documents" | "notes")
            || page["shape"] != "rows"
        {
            return Err(error(
                "bad_request",
                format!(
                    "a note needs session, message, document or note rows; the query returned {unit} in another form"
                ),
            ));
        }
        for row in page["rows"].as_array().into_iter().flatten() {
            let Some(id) = row["id"].as_str() else {
                continue;
            };
            let target = match unit {
                "sessions" => polylogue_model::SessionId::parse(id)
                    .map(Target::Session)
                    .ok(),
                "messages" => polylogue_model::MessageId::parse(id)
                    .map(Target::Message)
                    .ok(),
                "documents" => polylogue_model::DocumentId::parse(id)
                    .map(Target::Document)
                    .ok(),
                _ => Some(Target::Note(id.to_owned())),
            };
            if let Some(target) = target
                && seen.insert(target.clone())
            {
                targets.push(target);
            }
        }
        match page["next"].as_str() {
            Some(next) => cursor = Some(next.to_owned()),
            None => break,
        }
    }
    if targets.is_empty() {
        return Err(error(
            "empty",
            "the query returns no rows; nothing was noted",
        ));
    }
    Ok(targets)
}

/// Whole sessions in the structured export format (`json`, read back by
/// `import`; credentials masked at the response boundary) or rendered as Markdown or HTML.
/// With `user_state`, the `json` export also carries every request and
/// excision, which `import` replays.
fn export(
    reader: &polylogue_store::Reader<'_>,
    targets: &[ReadTarget],
    format: &str,
    user_state: bool,
) -> Response {
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
    let mut doc = polylogue_model::export::Document::new(sessions)
        .with_documents(documents)
        .with_events(events);
    if user_state {
        if format != "json" {
            return error("bad_request", "user state is exported only as json");
        }
        let entries = match reader.user_state() {
            Ok(entries) => entries,
            Err(e) => return store_error(&e),
        };
        let entries = match entries
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(entries) => entries,
            Err(e) => return error("internal", e.to_string()),
        };
        doc = doc.with_user_state(entries);
    }
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
    counts_stop: Option<Sender<()>>,
    counts: Option<JoinHandle<()>>,
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
        self.daemon.warm.clear();
        self.daemon.reads.close();
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        self.stop_accepting();
        if let Some(stop) = self.signals_stop.take() {
            let _ = stop.send(());
        }
        if let Some(stop) = self.counts_stop.take() {
            let _ = stop.send(());
            self.daemon.warm.clear();
        }
        if let Some(counts) = self.counts.take()
            && counts.join().is_err()
        {
            eprintln!("polylogue: the count worker panicked");
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
        warm: super::warm::Warm::default(),
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
        counts_stop: None,
        counts: None,
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
    let (stop, stopped) = channel();
    runtime.counts_stop = Some(stop);
    let daemon = Arc::clone(&runtime.daemon);
    runtime.counts = Some(
        std::thread::Builder::new()
            .name("polylogue-counts".into())
            .spawn(move || daemon.warm.count_worker(&daemon.reads, &stopped))?,
    );
    // Load the vectors `near:` ranks by before the first query needs them.
    let daemon = Arc::clone(&runtime.daemon);
    std::thread::Builder::new()
        .name("polylogue-warm".into())
        .spawn(move || {
            let Ok(reader) = daemon.reads.reader() else {
                return;
            };
            if let Ok(Some(provider)) =
                crate::embeddings::Provider::from_settings(&reader, &machine_id())
            {
                let _ = daemon.warm.vectors(&reader, &provider);
            }
        })?;
    Ok(runtime)
}

mod browser;
mod read;

#[cfg(test)]
mod tests;

/// The saved query a `save` revises: the current definition with that
/// name, if any. The name must be one `query:NAME` can say, and the text a
/// query.
fn saved_revision(
    reader: &polylogue_store::Reader<'_>,
    name: &str,
    text: &str,
) -> Result<Option<String>, Response> {
    let usable = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    if !usable {
        return Err(error(
            "invalid",
            format!("{name:?} cannot follow `query:`; use letters, digits, `_`, `-` and `.`"),
        ));
    }
    if let Err(e) = polylogue_query::parse(text) {
        return Err(error(e.kind(), e.to_string()));
    }
    let saved = super::query::saved_queries(reader).map_err(|f| Response::Error {
        kind: f.kind,
        message: f.message,
        detail: f.detail,
    })?;
    match saved.get(name).map(Vec::as_slice) {
        None | Some([]) => Ok(None),
        Some([one]) => Ok(Some(one.revision.clone())),
        Some(several) => Err(error(
            "ambiguous",
            format!(
                "{} saved queries are named {name}: {}; withdraw all but one first",
                several.len(),
                several
                    .iter()
                    .map(|s| format!("note:{}", s.first_note))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )),
    }
}
