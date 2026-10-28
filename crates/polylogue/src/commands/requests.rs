//! The daemon request each command sends, built from its checked values.

use super::{Caller, Command, LIVE, NOTICES, Values};
use crate::briefing::{Ask, StartSource};
use crate::browser::actions::{ChatFile, ChatRequest};
use crate::daemon::{ReadTarget, Request};
use polylogue_model::{Author, NewNote, NoteKind, Target};
use std::path::PathBuf;

type Built = Result<Request, String>;

fn query_request(text: String, page_size: Option<u64>, v: &Values) -> Request {
    Request::Query {
        text,
        cursor: v.text("cursor"),
        page_size,
        why: v.flag("why"),
        session: None,
        fields: v.list("fields"),
    }
}

/// A directory the caller named, made absolute against its working
/// directory; without one, the caller's working directory.
fn cwd(v: &Values, caller: &Caller) -> Result<PathBuf, String> {
    match (v.text("cwd"), &caller.cwd) {
        (Some(dir), Some(base)) => Ok(base.join(dir)),
        (Some(dir), None) if dir.starts_with('/') => Ok(dir.into()),
        (Some(_), None) => Err("`cwd` must be an absolute path".into()),
        (None, Some(base)) => Ok(base.clone()),
        (None, None) => Err("`cwd` is required".into()),
    }
}

fn targets(texts: &[String]) -> Result<Vec<Target>, String> {
    texts
        .iter()
        .map(|t| Target::parse(t).map_err(|e| e.to_string()))
        .collect()
}

fn note_kind(text: &str) -> Result<NoteKind, String> {
    NoteKind::parse(text).ok_or_else(|| format!("unknown note kind {text:?}"))
}

pub fn query(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(query_request(
        v.text("query").unwrap_or_default(),
        v.integer("limit"),
        v,
    ))
}

pub fn status(_: &Command, _: &Values, _: &Caller) -> Built {
    Ok(Request::Status)
}

pub fn complete(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Complete {
        word: v.text("word").unwrap_or_default(),
    })
}

pub fn attachment(_: &Command, v: &Values, caller: &Caller) -> Built {
    let reveal = v.flag("reveal");
    if reveal {
        if caller.surface != super::Surface::Cli || v.text("out").is_none() {
            return Err("exact attachment output requires the CLI with --reveal --out FILE".into());
        }
        crate::daemon::attachment::check_cli()?;
    } else if v.text("out").is_some() || v.text("input").is_some() {
        return Err("--out and --input require --reveal".into());
    }
    let index = |key: &str, default: u64| {
        usize::try_from(v.integer(key).unwrap_or(default))
            .map_err(|_| format!("{key} exceeds address space"))
    };
    let message = v.text("message").ok_or("message is required")?;
    Ok(Request::Attachment {
        message: polylogue_model::MessageId::parse(
            message.strip_prefix("message:").unwrap_or(&message),
        )
        .map_err(|e| e.to_string())?,
        block: index("block", 0)?,
        attachment: index("attachment", 0)?,
        input: v.text("input"),
        reveal,
        length: index("length", 65536)?,
        cursor: None,
    })
}

pub fn import(_: &Command, v: &Values, caller: &Caller) -> Built {
    let base = caller.cwd.clone().unwrap_or_default();
    Ok(Request::Import {
        request_id: v.request_id(),
        paths: v.list("paths").into_iter().map(|p| base.join(p)).collect(),
    })
}

/// The query; without one, the `sessions` unit with its defaults.
pub fn list(_: &Command, v: &Values, _: &Caller) -> Built {
    let text = v
        .text("query")
        .filter(|q| !q.trim().is_empty())
        .unwrap_or_else(|| "sessions".into());
    Ok(query_request(text, v.integer("limit"), v))
}

pub fn read(_: &Command, v: &Values, caller: &Caller) -> Built {
    let target = ReadTarget::parse(&v.text("session").ok_or("`session` is required")?)?;
    let with_parents = v.flag("with_parents");
    match target {
        ReadTarget::Event(_) if with_parents => {
            return Err("an event has no conversation parents".into());
        }
        ReadTarget::Document(_) if with_parents => {
            return Err("documents have no conversation parents".into());
        }
        _ => {}
    }
    let message = v.text("message");
    if message.is_some() && v.flag("latest") {
        return Err("choose either message or latest".into());
    }
    let start = if let Some(message) = message {
        Some(crate::daemon::PageSelection::Message(
            polylogue_model::MessageId::parse(message.strip_prefix("message:").unwrap_or(&message))
                .map_err(|e| e.to_string())?,
        ))
    } else {
        v.flag("latest")
            .then_some(crate::daemon::PageSelection::Latest)
    };
    Ok(Request::Read {
        slices: v.flag("slices"),
        selection: start,
        target,
        view: v.text("view").unwrap_or_else(|| "full".into()),
        format: v
            .text("format")
            .unwrap_or_else(|| if caller.json { "json" } else { "terminal" }.into()),
        with_parents,
        budget: v
            .integer("budget")
            .map(|b| usize::try_from(b).unwrap_or(usize::MAX)),
        cursor: v.text("cursor"),
    })
}

pub fn export(_: &Command, v: &Values, _: &Caller) -> Built {
    let names = v.list("sessions");
    let user_state = names.iter().any(|text| text == "user-state");
    let targets = names
        .iter()
        .filter(|text| *text != "user-state")
        .map(|text| ReadTarget::parse(text))
        .collect::<Result<Vec<_>, _>>()?;
    if targets.is_empty() && !user_state {
        return Err("name at least one session, document or event, or user-state".into());
    }
    Ok(Request::Export {
        targets,
        format: v.text("format").unwrap_or_else(|| "json".into()),
        user_state,
    })
}

/// The author: over MCP always the calling agent; elsewhere the operator
/// unless `agent` is given.
pub fn note_add(_: &Command, v: &Values, caller: &Caller) -> Built {
    let session = || match v.0.get("agent").and_then(|a| a.as_str()) {
        Some(s) => polylogue_model::SessionId::parse(s)
            .map(Some)
            .map_err(|e| e.to_string()),
        None => Ok(None),
    };
    let author = if caller.surface == super::Surface::Mcp || v.0.contains_key("agent") {
        Author::Agent(session()?)
    } else {
        Author::Operator
    };
    noted(
        v,
        NewNote {
            kind: note_kind(&v.text("kind").unwrap_or_else(|| "note".into()))?,
            key: v.text("key"),
            value: v.text("body"),
            author,
            targets: targets(&v.list("targets"))?,
            relations: v
                .text("relations")
                .map(|text| {
                    serde_json::from_str(&text)
                        .map_err(|error| format!("invalid note relations: {error}"))
                })
                .transpose()?
                .unwrap_or_default(),
            revises: v.text("revises"),
            valid_until: v
                .text("valid_until")
                .map(|text| {
                    polylogue_model::Timestamp::parse_rfc3339(&text)
                        .ok_or_else(|| "valid_until needs a time with a UTC offset".to_owned())
                })
                .transpose()?,
        },
    )
}

/// A note on its targets, or on every row of `--query`; one of the two.
fn noted(v: &Values, note: NewNote) -> Built {
    match (v.text("query"), note.targets.is_empty()) {
        (Some(_), false) => Err("give targets or `--query`, not both".into()),
        (None, true) if matches!(note.kind, NoteKind::SavedQuery | NoteKind::Signal) => {
            Ok(Request::AddNote {
                request_id: v.request_id(),
                note,
            })
        }
        (None, true) => Err("give the targets, or `--query` for every row of a query".into()),
        (Some(query), true) => Ok(Request::AnnotateQuery {
            request_id: v.request_id(),
            query,
            note,
        }),
        (None, false) => Ok(Request::AddNote {
            request_id: v.request_id(),
            note,
        }),
    }
}

/// `note accept` and `note reject`: the operator's judgment of an agent's
/// note.
pub fn note_accept(_: &Command, v: &Values, _: &Caller) -> Built {
    judge(v, true)
}

pub fn note_reject(_: &Command, v: &Values, _: &Caller) -> Built {
    judge(v, false)
}

fn judge(v: &Values, accepted: bool) -> Built {
    Ok(Request::Judge {
        request_id: v.request_id(),
        note: v.text("note").unwrap_or_default(),
        accepted,
        reason: v.text("reason"),
    })
}

pub fn note_tag(_: &Command, v: &Values, _: &Caller) -> Built {
    noted(
        v,
        NewNote {
            kind: NoteKind::Tag,
            key: v.text("tag"),
            value: v.text("value"),
            author: Author::Operator,
            targets: targets(&v.list("targets"))?,
            relations: vec![],
            revises: None,
            valid_until: None,
        },
    )
}

pub fn note_withdraw(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Withdraw {
        request_id: v.request_id(),
        note: v.text("note").unwrap_or_default(),
        author: Author::Operator,
        reason: v.text("reason"),
    })
}

/// `note list`: a `notes` query over the given targets and kinds.
pub fn note_list(_: &Command, v: &Values, caller: &Caller) -> Built {
    use crate::daemon::query::notes;
    let mut t = targets(&v.list("targets"))?;
    if v.0.contains_key("cwd") {
        let dir = cwd(v, caller)?;
        for d in crate::briefing::scope_dirs(&dir) {
            t.push(Target::Cwd(d));
        }
        if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
            t.push(Target::Repo(name.to_owned()));
        }
    }
    let mut conditions = Vec::new();
    if !t.is_empty() {
        conditions.push(notes::about(&t));
    }
    let kinds = v
        .list("kind")
        .iter()
        .map(|k| note_kind(k).map(|k| format!("kind:{}", k.as_str())))
        .collect::<Result<Vec<_>, _>>()?;
    if !kinds.is_empty() {
        conditions.push(format!("({})", kinds.join(" OR ")));
    }
    if v.flag("agents") {
        conditions.push("agent:true".into());
    }
    match v.text("judged").as_deref() {
        Some("none") => conditions.push("judged:unknown".into()),
        Some(j) => conditions.push(format!("judged:{j}")),
        None => {}
    }
    if v.flag("history") {
        conditions.push("current:*".into());
    }
    if v.flag("withdrawn") {
        conditions.push("withdrawn:*".into());
    }
    let text = if conditions.is_empty() {
        "notes".to_owned()
    } else {
        format!("notes where {}", conditions.join(" "))
    };
    Ok(query_request(text, v.integer("limit"), v))
}

pub fn brief(_: &Command, v: &Values, caller: &Caller) -> Built {
    Ok(Request::Context {
        request_id: v.request_id(),
        ask: Ask::SessionStart {
            cwd: cwd(v, caller)?,
            session: v.session("session")?,
            source: StartSource::parse(&v.text("source").unwrap_or_default()),
        },
        sender: caller.sender(),
    })
}

pub fn takeover(_: &Command, v: &Values, caller: &Caller) -> Built {
    Ok(Request::Context {
        request_id: v.request_id(),
        ask: Ask::Takeover {
            session: v.session("session")?.ok_or("`session` is required")?,
            for_session: None,
        },
        sender: caller.sender(),
    })
}

/// With a session, its resume commands; without, the sessions in a
/// directory worth resuming.
pub fn resume(_: &Command, v: &Values, caller: &Caller) -> Built {
    Ok(match v.session("session")? {
        Some(session) => Request::Continue { session },
        None => Request::Resumable {
            cwd: cwd(v, caller)?,
            limit: v.integer("limit"),
        },
    })
}

pub fn recall(_: &Command, v: &Values, caller: &Caller) -> Built {
    Ok(Request::Context {
        request_id: v.request_id(),
        ask: Ask::Recall {
            text: v.text("query").unwrap_or_default(),
            cwd: cwd(v, caller).ok(),
            for_session: None,
            cursor: v.text("cursor"),
        },
        sender: caller.sender(),
    })
}

/// A read without a value, a write with one.
pub fn setting(_: &Command, v: &Values, _: &Caller) -> Built {
    let value = v.text("value");
    Ok(Request::Setting {
        request_id: value.as_ref().map(|_| v.request_id()),
        key: v.text("key").unwrap_or_default(),
        value,
        machine: v.flag("machine"),
    })
}

/// `measure WHAT` as the query `QUERY | measure WHAT`.
pub fn measure(c: &Command, v: &Values, _: &Caller) -> Built {
    let name = c.name.trim_start_matches("measure ");
    let what = match v.text("metric").or_else(|| v.text("option")) {
        Some(metric) => format!("{name}:{metric}"),
        None => name.to_owned(),
    };
    let text = match v.text("query").filter(|q| !q.trim().is_empty()) {
        Some(q) => format!("{q} | measure {what}"),
        None => format!("sessions | measure {what}"),
    };
    Ok(query_request(text, v.integer("limit"), v))
}

pub fn work(_: &Command, v: &Values, _: &Caller) -> Built {
    let text = match v.text("query").filter(|q| !q.trim().is_empty()) {
        Some(q) => format!("work where {q}"),
        None => "work".into(),
    };
    Ok(query_request(text, v.integer("limit"), v))
}

pub fn live(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(query_request(LIVE.into(), None, v))
}

pub fn notices(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(query_request(
        NOTICES.into(),
        Some(v.integer("limit").unwrap_or(50)),
        v,
    ))
}

/// `save NAME QUERY`. Over MCP the author is the calling agent.
pub fn save(_: &Command, v: &Values, caller: &Caller) -> Built {
    Ok(Request::Save {
        request_id: v.request_id(),
        name: v.text("name").unwrap_or_default(),
        text: v.text("query").unwrap_or_default(),
        author: if caller.surface == super::Surface::Mcp {
            Author::Agent(None)
        } else {
            Author::Operator
        },
    })
}

/// `watch QUERY`. Over MCP the author is the calling agent.
pub fn watch(_: &Command, v: &Values, caller: &Caller) -> Built {
    Ok(Request::Watch {
        request_id: v.request_id(),
        query: v.text("query").unwrap_or_default(),
        name: v.text("name"),
        hold: v.text("for"),
        notify: v.flag("notify"),
        author: if caller.surface == super::Surface::Mcp {
            Author::Agent(None)
        } else {
            Author::Operator
        },
    })
}

pub fn keep(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Keep {
        request_id: v.request_id(),
        text: v.text("query").unwrap_or_default(),
        name: v.text("name"),
    })
}

pub fn sql(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Sql {
        statement: v.text("statement").unwrap_or_default(),
    })
}

/// `embed status|preview|build`.
pub fn embed(c: &Command, v: &Values, _: &Caller) -> Built {
    let what = c.name.trim_start_matches("embed ");
    let status = what == "status";
    Ok(Request::Embed {
        what: what.to_owned(),
        query: (!status).then(|| v.text("query").unwrap_or_default()),
        limit: v.integer("limit"),
        yes: v.flag("yes"),
    })
}

/// `chat send`: the command line reads `--attach` files here, in the
/// caller's process.
pub fn chat_send(_: &Command, v: &Values, caller: &Caller) -> Built {
    let mut files = Vec::new();
    for path in v.list("attach") {
        let path = match &caller.cwd {
            Some(cwd) => cwd.join(&path),
            None => std::path::PathBuf::from(&path),
        };
        let bytes =
            std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or_else(|| format!("{} names no file", path.display()))?;
        files.push(ChatFile {
            media_type: media_type(&name).to_owned(),
            name,
            bytes,
        });
    }
    Ok(Request::ChatSend {
        request_id: v.request_id(),
        action: ChatRequest {
            provider: v.text("provider").unwrap_or_default(),
            conversation: v.text("conversation"),
            project: v.text("project"),
            text: v.text("text").unwrap_or_default(),
            model: v.text("model").unwrap_or_default(),
            effort: v.text("effort"),
            submit: v.flag("submit"),
        },
        files,
    })
}

/// The media type a file's extension names; the provider checks the bytes.
fn media_type(name: &str) -> &'static str {
    let extension = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    match extension.as_deref() {
        Some("txt" | "md") => "text/plain",
        Some("json") => "application/json",
        Some("pdf") => "application/pdf",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("csv") => "text/csv",
        Some("html") => "text/html",
        _ => "application/octet-stream",
    }
}

pub fn chat_list(_: &Command, _: &Values, _: &Caller) -> Built {
    Ok(Request::ChatActions)
}

/// `chat approve` and `chat decline`.
pub fn chat_decide(c: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::ChatDecide {
        id: v.text("id").unwrap_or_default(),
        approve: c.name == "chat approve",
        by: "operator".into(),
        detail: None,
    })
}

pub fn backfill_start(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::BackfillStart {
        provider: v.text("provider").unwrap_or_default(),
        since: v.text("since"),
    })
}

pub fn backfill_list(_: &Command, _: &Values, _: &Caller) -> Built {
    Ok(Request::BackfillJobs)
}

pub fn backfill_cancel(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::BackfillCancel {
        id: v.text("id").unwrap_or_default(),
    })
}

pub fn outline(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Outline {
        session: polylogue_model::SessionId::parse(
            &v.text("session").ok_or("session is required")?,
        )
        .map_err(|e| e.to_string())?,
        with_parents: v.flag("with_parents"),
        page_size: usize::try_from(v.integer("page_size").unwrap_or(50))
            .map_err(|_| "page_size exceeds address space")?,
        cursor: v.text("cursor"),
    })
}

pub fn changes(_: &Command, v: &Values, _: &Caller) -> Built {
    if v.text("cursor").is_some() || v.0.contains_key("fields") || v.0.contains_key("why") {
        return Err("the change feed takes --since; use a changes query for filtering, fields or ordinary pages".into());
    }
    Ok(Request::Changes {
        since: v.text("since"),
    })
}

pub fn note_read(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Note {
        id: v.text("id").ok_or("id is required")?,
    })
}

pub fn explain(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Explain {
        subject: v.text("subject").ok_or("`subject` is required")?,
        expression: v.text("expression"),
        unit: v.text("unit"),
        page_size: v.integer("page_size"),
        cursor: v.text("cursor"),
        session: v.text("session"),
    })
}

pub fn timeline(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Timeline {
        date: v.text("date").filter(|s| !s.is_empty()),
        across_years: v.flag("across_years"),
        undated: v.flag("undated"),
        cursor: v.text("cursor"),
        page_size: v.integer("limit"),
    })
}
pub fn cost(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Cost {
        month: v.text("month").filter(|s| !s.is_empty()),
        undated: v.flag("undated"),
        unknown_model: v.flag("unknown_model"),
        query: v.text("query").unwrap_or_else(|| "sessions".into()),
        date: v.text("date").filter(|s| !s.is_empty()),
        model: v.text("model").filter(|s| !s.is_empty()),
        cursor: v.text("cursor"),
        page_size: v.integer("limit"),
    })
}
