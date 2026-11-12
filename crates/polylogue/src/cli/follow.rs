//! `TARGET follow`: a live stream of a session in its reading form, or of a
//! query's new rows. A session prints each new item in the collapsed form a
//! transcript shows, as the session grows; it reads pages through the
//! ordinary `read` operation and continues with each page's `follow` call.
//! A query prints the rows that were not in its earlier answers. Nothing is
//! recorded.
use super::{Archive, fail};
use crate::protocol::follow::{Line, Marks};
use polylogue_model::read::{HeaderKind, ReadPage, ReadyCall};
use std::collections::HashSet;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Duration;

/// How long to wait before asking again when nothing new arrived.
const QUIET_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

pub fn run(
    values: &crate::commands::Values,
    startup: &crate::config::Startup,
    call: &command_surface::RunnableRequest,
) -> ExitCode {
    let json = values.flag("json");
    let output = startup.output;
    let Some(target) = values.text("target") else {
        return fail(
            output,
            json,
            "bad_request",
            "follow needs a session or a query",
            2,
        );
    };
    let archive = Archive::at(&startup.config.archive);
    let session_target = target.starts_with("root:")
        || matches!(
            polylogue_model::Reference::parse(&target),
            Ok(polylogue_model::Reference::Session(_)
                | polylogue_model::Reference::Message(_)
                | polylogue_model::Reference::MessageRange { .. })
        );
    if !session_target {
        if values.flag("until_idle")
            || !values.list("show").is_empty()
            || !values.list("hide").is_empty()
        {
            return fail(
                output,
                json,
                "bad_request",
                "--until-idle, --show and --hide apply to following a session",
                2,
            );
        }
        if target.split_whitespace().next() == Some("sessions") {
            return live_query(values, &target, startup, call);
        }
        return follow_query(&archive, &target, output, json);
    }
    follow_session(&archive, &target, values, output, json)
}

fn follow_session(
    archive: &Archive,
    target: &str,
    values: &crate::commands::Values,
    output: crate::config::OutputConfig,
    json: bool,
) -> ExitCode {
    let until_idle = values.flag("until_idle");
    let mut arguments = serde_json::json!({
        "target": target,
        "format": "json",
        "latest": true,
    });
    for name in ["show", "hide"] {
        let words = values.list(name);
        if !words.is_empty() {
            arguments[name] = serde_json::json!(words);
        }
    }
    let first = command_surface::RunnableRequest {
        operation: "read".into(),
        arguments,
        request_id: format!("follow-{}", uuid::Uuid::now_v7()),
        deadline: command_surface::Deadline::Default,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut call = first;
    let mut encoder = None;
    let mut session = None;
    loop {
        let page: ReadPage = match answer(archive, &call) {
            Ok(page) => page,
            Err((code, message)) => return fail(output, json, &code, &message, 1),
        };
        if session.is_none() {
            session = match &page.document.header.kind {
                HeaderKind::Session { session } => Some(session.id.to_string()),
                // A logical session runs in its newest physical session.
                HeaderKind::Logical {
                    physical_sessions,
                    root,
                    ..
                } => Some(physical_sessions.last().unwrap_or(root).to_string()),
                _ => None,
            };
        }
        let fresh = !page.document.items.is_empty();
        if encoder.is_none() {
            match polylogue_render::Encoder::new(
                &page.document.header,
                polylogue_render::Format::Terminal,
                polylogue_render::EncodeOptions {
                    commands: Some(crate::commands::declarations()),
                    ..Default::default()
                },
                &mut out,
            ) {
                Ok(started) => encoder = Some(started),
                Err(error) => return fail(output, json, "internal", &error.to_string(), 1),
            }
        }
        let started = encoder.as_mut().expect("started above");
        for item in &page.document.items {
            if started.push(item).is_err() {
                return ExitCode::SUCCESS;
            }
        }
        let next: Option<&ReadyCall> = page.document.footer.next.as_ref().or(page
            .document
            .footer
            .follow
            .as_ref());
        let Some(next) = next else {
            return ExitCode::SUCCESS;
        };
        if !fresh && page.document.footer.next.is_none() {
            // Nothing new: with --until-idle, stop once the session's live
            // state is neither working nor waiting; otherwise ask again.
            if until_idle {
                match live_state(archive, session.as_deref()) {
                    Ok(state) if !matches!(state.as_str(), "working" | "waiting") => {
                        return ExitCode::SUCCESS;
                    }
                    Ok(_) => {}
                    Err((code, message)) => return fail(output, json, &code, &message, 1),
                }
            }
            std::thread::sleep(QUIET_WAIT);
        }
        call = next.request.clone();
        let _ = io::stdout().flush();
    }
}

/// The session's live state (`working`, `waiting`, `idle`, `unknown`,
/// `stale` or `ended`), from the query language's `live` field.
fn live_state(archive: &Archive, session: Option<&str>) -> Result<String, (String, String)> {
    let Some(session) = session else {
        return Err((
            "bad_request".into(),
            "--until-idle needs a session reading".into(),
        ));
    };
    let call = command_surface::RunnableRequest {
        operation: "query".into(),
        arguments: serde_json::json!({
            "query": format!("sessions where id:{}", quoted(session)),
            "fields": ["live"],
            "limit": 1,
        }),
        request_id: format!("follow-live-{}", uuid::Uuid::now_v7()),
        deadline: command_surface::Deadline::Default,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let rows: serde_json::Value = answer(archive, &call)?;
    Ok(rows["rows"][0]["live"]
        .as_str()
        .unwrap_or("unknown")
        .to_owned())
}

fn quoted(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}

/// `QUERY follow`: the query's rows, then each row that was not in an
/// earlier answer, until interrupted. The whole answer, every page, is
/// read again only when the archive's read position moved.
fn follow_query(
    archive: &Archive,
    query: &str,
    output: crate::config::OutputConfig,
    json: bool,
) -> ExitCode {
    let mut seen: HashSet<String> = HashSet::new();
    let mut position = serde_json::Value::Null;
    let stdout = io::stdout();
    loop {
        let page = |cursor: Option<&str>| -> Result<serde_json::Value, (String, String)> {
            answer(
                archive,
                &command_surface::RunnableRequest {
                    operation: "query".into(),
                    arguments: serde_json::json!({ "query": query, "cursor": cursor }),
                    request_id: format!("follow-query-{}", uuid::Uuid::now_v7()),
                    deadline: command_surface::Deadline::Default,
                    budget_tokens: Default::default(),
                    cursor: None,
                    fresh: false,
                    format: Default::default(),
                },
            )
        };
        let mut data = match page(None) {
            Ok(data) => data,
            Err((code, message)) => return fail(output, json, &code, &message, 1),
        };
        if data["outcome"] == "measure" {
            return fail(
                output,
                json,
                "bad_request",
                "follow takes a query of rows, not a measure",
                2,
            );
        }
        if data["read_at"] == position {
            std::thread::sleep(QUIET_WAIT);
            continue;
        }
        position = data["read_at"].clone();
        let mut rows = data["rows"].as_array().cloned().unwrap_or_default();
        let mut next = data["next"].as_str().map(str::to_owned);
        while let Some(cursor) = next {
            let more = match page(Some(&cursor)) {
                Ok(more) => more,
                Err((code, message)) => return fail(output, json, &code, &message, 1),
            };
            rows.extend(more["rows"].as_array().cloned().unwrap_or_default());
            next = more["next"].as_str().map(str::to_owned);
        }
        let fresh: Vec<serde_json::Value> = rows
            .into_iter()
            .filter(|row| seen.insert(row_identity(row)))
            .collect();
        if !fresh.is_empty() {
            data["rows"] = serde_json::Value::Array(fresh);
            data["next"] = serde_json::Value::Null;
            let mut out = stdout.lock();
            if polylogue_render::results::plain_terminal(&data, &mut out).is_err()
                || out.flush().is_err()
            {
                return ExitCode::SUCCESS;
            }
        }
        std::thread::sleep(QUIET_WAIT);
    }
}

/// A row is the same row again when its reference is: `ref`, else `id`,
/// else its whole value.
fn row_identity(row: &serde_json::Value) -> String {
    ["ref", "id"]
        .into_iter()
        .find_map(|key| row[key].as_str().map(str::to_owned))
        .unwrap_or_else(|| row.to_string())
}

fn answer<T: serde::de::DeserializeOwned>(
    archive: &Archive,
    call: &command_surface::RunnableRequest,
) -> Result<T, (String, String)> {
    let answer = crate::client::call_ready(archive, call)
        .map_err(|error| ("daemon_unavailable".to_string(), error.to_string()))?;
    match answer.value() {
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|error| ("internal".to_string(), error.to_string())),
        None => Err((
            "read_failed".into(),
            match &answer.outcome {
                command_surface::Outcome::Error(error) => error.message.clone(),
                _ => "the read returned no value".into(),
            },
        )),
    }
}

/// How long one follow request runs before the client renews it.
const REQUEST_SPAN: Duration = Duration::from_secs(60);

fn live_query(
    values: &crate::commands::Values,
    query: &str,
    startup: &crate::config::Startup,
    _: &command_surface::RunnableRequest,
) -> ExitCode {
    let zone = startup
        .config
        .standalone_day()
        .map(|day| day.zone().clone())
        .unwrap_or_else(|_| jiff::tz::TimeZone::system());
    let archive = Archive::at(&startup.config.archive);
    let mut marks: Option<Marks> = None;
    let mut stdout = io::stdout().lock();
    loop {
        let arguments = crate::daemon::service::follow::Arguments {
            target: query.to_owned(),
            marks: marks.clone(),
        };
        let mut request = crate::client::request(
            "follow",
            serde_json::to_value(arguments).expect("follow arguments serialize"),
            polylogue_store::user_writes::new_request_id(),
        );
        request.deadline = command_surface::Deadline::Milliseconds(
            u64::try_from(REQUEST_SPAN.as_millis()).expect("the follow span fits"),
        );
        let mut next: Option<Marks> = None;
        let result =
            crate::client::stream::run_call(&archive, &request, Some(REQUEST_SPAN), |frame| {
                if frame.event == "error" {
                    return Err(io::Error::other(frame.data));
                }
                let line: Line = serde_json::from_str(&frame.data).map_err(io::Error::other)?;
                match line {
                    Line::Marks { marks } => next = Some(marks),
                    line => {
                        if values.flag("json") {
                            serde_json::to_writer(&mut stdout, &line)?;
                            writeln!(stdout)?;
                            stdout.flush()?;
                        } else if let Some(text) = line.text(&zone) {
                            writeln!(stdout, "{}", polylogue_render::terminal::escape(&text))?;
                            stdout.flush()?;
                        }
                    }
                }
                Ok(true)
            });
        match (result, next) {
            (Ok(()), Some(reached)) => marks = Some(reached),
            (Ok(()), None) => {
                return fail(
                    startup.output,
                    true,
                    "io",
                    "the daemon ended the follow without its marks",
                    1,
                );
            }
            (Err(error), _) => {
                return fail(startup.output, true, "io", &error.to_string(), 1);
            }
        }
    }
}
