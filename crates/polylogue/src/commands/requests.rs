//! The daemon request each command sends, built from its checked values.

use super::{Caller, Command, LIVE, NOTICES, Values};
use crate::briefing::{Ask, StartSource};
use crate::daemon::Request;
use polylogue_model::{Author, NewNote, NoteKind, Target};
use polylogue_store::NoteFilter;
use serde_json::Value;
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
    Ok(Request::Read {
        session: v.session("session")?.ok_or("`session` is required")?,
        view: v.text("view").unwrap_or_else(|| "full".into()),
        format: v
            .text("format")
            .unwrap_or_else(|| if caller.json { "json" } else { "terminal" }.into()),
        logical: v.flag("logical"),
        budget: v
            .integer("budget")
            .map(|b| usize::try_from(b).unwrap_or(usize::MAX)),
        cursor: v.text("cursor"),
    })
}

pub fn export(_: &Command, v: &Values, _: &Caller) -> Built {
    let sessions = v
        .list("sessions")
        .iter()
        .map(|s| polylogue_model::SessionId::parse(s).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if sessions.is_empty() {
        return Err("name at least one session".into());
    }
    Ok(Request::Export {
        sessions,
        format: v.text("format").unwrap_or_else(|| "json".into()),
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
    Ok(Request::AddNote {
        request_id: v.request_id(),
        note: NewNote {
            kind: note_kind(&v.text("kind").unwrap_or_else(|| "note".into()))?,
            key: None,
            value: v.text("body"),
            author,
            targets: targets(&v.list("targets"))?,
            relations: vec![],
            revises: v.text("revises"),
            valid_until: None,
        },
    })
}

pub fn note_tag(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::AddNote {
        request_id: v.request_id(),
        note: NewNote {
            kind: NoteKind::Tag,
            key: v.text("tag"),
            value: v.text("value"),
            author: Author::Operator,
            targets: targets(&v.list("targets"))?,
            relations: vec![],
            revises: None,
            valid_until: None,
        },
    })
}

pub fn note_withdraw(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Withdraw {
        request_id: v.request_id(),
        note: v.text("note").unwrap_or_default(),
        author: Author::Operator,
        reason: v.text("reason"),
    })
}

pub fn note_list(_: &Command, v: &Values, caller: &Caller) -> Built {
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
    Ok(Request::Notes {
        filter: NoteFilter {
            targets: t,
            kinds: v
                .list("kind")
                .iter()
                .map(|k| note_kind(k))
                .collect::<Result<_, _>>()?,
            agents: v.flag("agents").then_some(true),
            history: v.flag("history"),
            withdrawn: v.flag("withdrawn"),
            after_seq: None,
            limit: v.integer("limit"),
        },
    })
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
    let what = match v.text("metric") {
        Some(metric) => format!("{name}:{metric}"),
        None => name.to_owned(),
    };
    let text = match v.text("query").filter(|q| !q.trim().is_empty()) {
        Some(q) => format!("{q} | measure {what}"),
        None => format!("sessions | measure {what}"),
    };
    Ok(query_request(text, None, v))
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

pub fn changes(_: &Command, v: &Values, _: &Caller) -> Built {
    if v.text("cursor").is_some() || v.0.contains_key("fields") || v.0.contains_key("why") {
        return Err("the change feed takes --since; use a changes query for filtering, fields or ordinary pages".into());
    }
    Ok(Request::Changes {
        since: v.text("since"),
    })
}

pub fn notices(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(query_request(
        NOTICES.into(),
        Some(v.integer("limit").unwrap_or(50)),
        v,
    ))
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

pub fn excise(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Excise {
        request_id: v.request_id(),
        target: v.text("target").unwrap_or_default(),
        reason: v.text("reason"),
        dry_run: !v.flag("yes"),
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

pub fn browser_action(value: Value) -> Built {
    serde_json::from_value(value)
        .map(|action| Request::BrowserEnqueue { action })
        .map_err(|error| format!("invalid browser action: {error}"))
}

pub fn browser_attachment(bytes: Vec<u8>) -> Request {
    Request::BrowserUpload { bytes }
}
