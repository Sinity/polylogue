//! The daemon request each command sends, built from its checked values.

use super::{Caller, Command, LIVE, NOTICES, Values, WAITING};
use crate::briefing::{Ask, StartSource};
use crate::browser::actions::{ChatFile, ChatRequest};
use crate::daemon::Request;
use std::path::PathBuf;

type Built = Result<Request, String>;

fn query_request(text: String, page_size: Option<u64>, v: &Values) -> Request {
    Request::Query {
        text,
        cursor: v.text("cursor"),
        page_size,
        why: v.flag("why"),
        session: v.text("session"),
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
    let text = v.text("text").unwrap_or_default();
    let position = v
        .integer("position")
        .map(usize::try_from)
        .transpose()
        .map_err(|_| "completion position cannot be represented on this host".to_owned())?
        .unwrap_or(text.len());
    if position > text.len() || !text.is_char_boundary(position) {
        return Err("completion position must be a UTF-8 byte boundary within the query".into());
    }
    Ok(Request::Complete { text, position })
}

pub fn attachment(_: &Command, v: &Values, _: &Caller) -> Built {
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
        length: index("length", 65536)?,
        cursor: v.text("cursor"),
    })
}

pub fn import(_: &Command, v: &Values, caller: &Caller) -> Built {
    let absolute = |path: String| {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            Ok(path)
        } else {
            caller.cwd.as_ref().map(|base| base.join(path)).ok_or_else(||
            "relative import paths require a native caller working directory; supply an absolute path".to_string())
        }
    };
    let paths = v
        .list("paths")
        .into_iter()
        .map(absolute)
        .collect::<Result<Vec<_>, String>>()?;
    if let Some(archive) = v.text("from_archive") {
        if !paths.is_empty() {
            return Err("--from-archive names one archive and takes no other paths".into());
        }
        return Ok(Request::ImportArchive {
            request_id: v.request_id(),
            path: absolute(archive)?,
            dry_run: v.flag("dry_run"),
        });
    }
    if v.flag("dry_run") {
        return Err("--dry-run applies to --from-archive".into());
    }
    Ok(Request::Import {
        request_id: v.request_id(),
        paths,
    })
}

/// A relative manifest path names a file in the caller's directory.
pub fn verify(_: &Command, v: &Values, caller: &Caller) -> Built {
    let manifest = v
        .text("manifest")
        .map(|path| {
            let path = PathBuf::from(path);
            if path.is_absolute() {
                Ok(path)
            } else {
                caller.cwd.as_ref().map(|base| base.join(path)).ok_or_else(|| {
                    "a relative manifest path requires a native caller working directory; supply an absolute path".to_string()
                })
            }
        })
        .transpose()?;
    Ok(Request::Verify { manifest })
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
    if v.integer("page_tokens") == Some(0) {
        return Err("page_tokens must be positive".into());
    }
    use polylogue_model::read::{ReadDelivery, ReadOptions};
    let directory = v.text("to").is_some();
    let mut options = ReadOptions {
        view: v
            .text("view")
            .unwrap_or_else(|| "transcript".into())
            .parse()
            .map_err(|error: polylogue_model::read::ReadError| error.to_string())?,
        output_lines: v.integer("output_lines").unwrap_or(100),
        // A bundle carries inherited prefixes unless they are hidden.
        inherited: directory,
        for_others: v.flag("for_others"),
        variants: v.list("variant"),
        at: v
            .text("at")
            .map(|at| polylogue_model::MessageId::parse(&at))
            .transpose()
            .map_err(|error| error.to_string())?,
        diff: v.text("diff"),
        around: v.integer("around"),
        latest: v.flag("latest"),
        physical: v.flag("physical"),
        ..ReadOptions::default()
    };
    options
        .apply_fold(&v.list("show"), &v.list("hide"))
        .map_err(|error| error.to_string())?;
    options.validate().map_err(|error| error.to_string())?;
    let delivery = if directory {
        ReadDelivery::Stream
    } else {
        v.text("delivery")
            .map(|value| serde_json::from_value(serde_json::Value::String(value)))
            .transpose()
            .map_err(|error| error.to_string())?
            .unwrap_or_default()
    };
    Ok(Request::Read {
        target: v.text("target").ok_or("target is required")?,
        options,
        delivery,
        format: v
            .text("format")
            .unwrap_or_else(|| if caller.json { "json" } else { "terminal" }.into()),
        budget: v
            .integer("page_tokens")
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "page_tokens exceeds address space")?,
        with_blobs: v.flag("with_blobs"),
        cursor: v.text("cursor"),
    })
}

/// Typed operation decoding precedes this builder; decode again at the direct
/// CLI boundary so a structural value cannot become an unchecked write.
/// Only the operation's declared arguments reach the decoder; output options
/// such as `json` and the request ID are not part of the operation.
fn assertion_arguments<T: serde::de::DeserializeOwned>(
    command: &Command,
    v: &Values,
) -> Result<T, String> {
    let declared =
        v.0.iter()
            .filter(|(name, _)| command.args.iter().any(|argument| argument.name == *name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
    serde_json::from_value(serde_json::Value::Object(declared)).map_err(|error| error.to_string())
}
pub fn assert(command: &Command, v: &Values, _: &Caller) -> Built {
    let args: crate::protocol::assertion::Assert = assertion_arguments(command, v)?;
    if args.assertions.is_empty() {
        return Err("assertions must not be empty".into());
    }
    Ok(Request::Assert {
        request_id: v.request_id(),
        assertions: args.assertions,
    })
}
pub fn assert_query(command: &Command, v: &Values, _: &Caller) -> Built {
    let args: crate::protocol::assertion::AssertQuery = assertion_arguments(command, v)?;
    if args.query.trim().is_empty() {
        return Err("query must not be empty".into());
    }
    Ok(Request::AssertQuery {
        request_id: v.request_id(),
        query: args.query,
        assertion: args.assertion,
    })
}
pub fn revise(command: &Command, v: &Values, _: &Caller) -> Built {
    let args: crate::protocol::assertion::Revise = assertion_arguments(command, v)?;
    if args.bases.is_empty() {
        return Err("bases must not be empty".into());
    }
    Ok(Request::Revise {
        request_id: v.request_id(),
        bases: args.bases,
        patch: args.patch,
    })
}
pub fn withdraw(command: &Command, v: &Values, _: &Caller) -> Built {
    let args: crate::protocol::assertion::Withdraw = assertion_arguments(command, v)?;
    if args.targets.is_empty() {
        return Err("targets must not be empty".into());
    }
    Ok(Request::Withdraw {
        request_id: v.request_id(),
        targets: args.targets,
        reason: args.reason,
    })
}

pub fn brief(_: &Command, v: &Values, caller: &Caller) -> Built {
    Ok(Request::Context {
        budget_tokens: Default::default(),
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
        budget_tokens: Default::default(),
        request_id: v.request_id(),
        ask: Ask::Takeover {
            session: v.session("session")?.ok_or("`session` is required")?,
            for_session: v.session("for_session")?,
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
        budget_tokens: Default::default(),
        request_id: v.request_id(),
        ask: Ask::Recall {
            text: v.text("query").unwrap_or_default(),
            cwd: cwd(v, caller).ok(),
            for_session: v.session("for_session")?,
            cursor: v.text("cursor"),
        },
        sender: caller.sender(),
    })
}

/// A read without a value, a write with one.
pub fn setting(_: &Command, v: &Values, _: &Caller) -> Built {
    let value = v.text("value");
    let key = v.text("key");
    if value.is_some() && key.is_none() {
        return Err("a value requires a setting key".into());
    }
    Ok(Request::Setting {
        request_id: value.as_ref().map(|_| v.request_id()),
        key,
        value,
        machine: v.text("machine"),
    })
}

/// `measure WHAT` as the query `QUERY | measure WHAT`.
pub fn measure(c: &Command, v: &Values, _: &Caller) -> Built {
    let name = c.name.trim_start_matches("measure ");
    let what = match v.text("metric").or_else(|| v.text("option")) {
        Some(metric) => format!("{name}:{metric}"),
        None => name.to_owned(),
    };
    let what = polylogue_query::quoted(&what);
    let text = match v.text("query").filter(|q| !q.trim().is_empty()) {
        Some(q) => format!("{q} | measure {what}"),
        None => format!("sessions | measure {what}"),
    };
    Ok(query_request(text, v.integer("limit"), v))
}

pub fn live(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(query_request(LIVE.into(), None, v))
}

pub fn waiting(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(query_request(WAITING.into(), None, v))
}

pub fn notices(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(query_request(
        NOTICES.into(),
        Some(v.integer("limit").unwrap_or(50)),
        v,
    ))
}

/// `save NAME QUERY`.
pub fn save(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Save {
        request_id: v.request_id(),
        name: v.text("name").unwrap_or_default(),
        text: v.text("query").unwrap_or_default(),
        denominator: v.text("denominator"),
    })
}

/// `watch QUERY`.
pub fn watch(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Watch {
        request_id: v.request_id(),
        query: v.text("query").unwrap_or_default(),
        name: v.text("name"),
        hold: v.text("for"),
        notify: v.flag("notify"),
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

pub fn changes(_: &Command, v: &Values, _: &Caller) -> Built {
    if v.text("cursor").is_some() || v.0.contains_key("fields") || v.0.contains_key("why") {
        return Err("the change feed takes --since; use a changes query for filtering, fields or ordinary pages".into());
    }
    Ok(Request::Changes {
        since: v.text("since"),
    })
}

fn session_arg(v: &Values) -> Result<polylogue_model::SessionId, String> {
    polylogue_model::SessionId::parse(&v.text("session").ok_or("a session reference is required")?)
        .map_err(|e| e.to_string())
}

pub fn ping(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Ping {
        request_id: v.request_id(),
        target: v
            .text("target")
            .ok_or("a session reference or @[query] is required")?,
        text: v.text("text").ok_or("a ping needs text")?,
        live: v.flag("live"),
        about: v.text("about"),
    })
}

pub fn answer(_: &Command, v: &Values, _: &Caller) -> Built {
    let approve = match v.text("decision").as_deref() {
        Some("approve") => true,
        Some("deny") => false,
        _ => return Err("answer takes approve or deny".into()),
    };
    Ok(Request::Answer {
        request_id: v.request_id(),
        session: session_arg(v)?,
        approve,
        text: v.text("text"),
        standing: v.flag("standing"),
    })
}

pub fn interrupt(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Interrupt {
        request_id: v.request_id(),
        session: session_arg(v)?,
    })
}

pub fn start(_: &Command, v: &Values, _: &Caller) -> Built {
    let (harness, role) = (v.text("harness"), v.text("role"));
    if harness.is_none() && role.is_none() {
        return Err("start takes --harness or --role".into());
    }
    Ok(Request::Start {
        request_id: v.request_id(),
        harness,
        role,
        cwd: std::path::PathBuf::from(v.text("cwd").ok_or("start takes --cwd")?),
        text: v.text("text").ok_or("start needs the first turn's text")?,
    })
}

pub fn hand_off(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::HandOff {
        request_id: v.request_id(),
        session: session_arg(v)?,
        to: v.text("to").ok_or("hand-off takes --to")?,
        cwd: v.text("cwd").map(std::path::PathBuf::from),
        at: v.text("at"),
        stop_source: v.flag("stop_source"),
        print: v.flag("print"),
    })
}

pub fn follow(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::Follow {
        query: v
            .text("target")
            .ok_or("follow takes a session reference or a sessions query")?,
        marks: None,
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

fn schema_origin(v: &Values) -> Result<Option<String>, String> {
    let origin = v.text("origin");
    if origin.is_some() && v.flag("all") {
        return Err("choose an origin or --all".into());
    }
    Ok(origin)
}

pub fn schema_infer(_: &Command, v: &Values, _: &Caller) -> Built {
    let revision = v
        .integer("revision")
        .map(i64::try_from)
        .transpose()
        .map_err(|_| "revision exceeds signed range")?;
    let request = crate::daemon::schema::Infer {
        origin: schema_origin(v)?,
        inputs: v.list("input").into_iter().collect(),
        revision,
        since: v.text("since"),
        fields: v.list("field").into_iter().collect(),
    };
    request.validate()?;
    Ok(Request::SchemaInfer { selection: request })
}

pub fn schema_unread(_: &Command, v: &Values, _: &Caller) -> Built {
    Ok(Request::SchemaUnread {
        origin: schema_origin(v)?,
    })
}
