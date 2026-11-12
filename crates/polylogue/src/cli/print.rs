//! Answers as text for a person: query results as tables (numbered for
//! `@N`), measurements, sessions, briefs, notes and resume plans. Each
//! command names its printer in the command table; `--json` prints the
//! answer itself.

use polylogue_render::shell_quote;
use serde_json::Value;
use std::fmt::Write as _;

fn s(v: &Value) -> String {
    v.as_str().unwrap_or("").to_owned()
}

/// The answer as indented JSON: commands without a text view.
pub fn json(data: &Value, _color: bool) -> String {
    format!("{}\n", polylogue_render::terminal::json(data, true))
}

/// A query's rows as a table, or a measurement in its
/// own text view.
pub fn query(data: &Value, color: bool) -> String {
    use crate::protocol::query::Answer;
    match query_answer(data) {
        Answer::PagedMeasure(measure) => {
            measurement(&measure.measure, &measure.data, measure.next.as_deref())
        }
        Answer::Measure(measure) => measurement(&measure.measure, &measure.data, None),
        Answer::Rows(_) => {
            let mut out = Vec::new();
            let _ = polylogue_render::results::terminal_with_color(data, color, &mut out);
            String::from_utf8_lossy(&out).into_owned()
        }
    }
}

/// Query text for the tour, whose final display handles styling.
pub fn query_plain(data: &Value) -> String {
    match query_answer(data) {
        crate::protocol::query::Answer::Rows(_) => {
            let mut out = Vec::new();
            let _ = polylogue_render::results::plain_terminal(data, &mut out);
            String::from_utf8_lossy(&out).into_owned()
        }
        _ => query(data, false),
    }
}

/// The query operation's declared answer.
fn query_answer(data: &Value) -> crate::protocol::query::Answer {
    serde::Deserialize::deserialize(data).expect("query operation returns its declared answer")
}

/// A measurement in its measure's text view, with the continuation line.
fn measurement(what: &str, data: &Value, next: Option<&str>) -> String {
    let text = crate::daemon::measure::find(what)
        .map_or(polylogue_render::measures::json as fn(&_) -> _, |(m, _)| {
            m.text
        });
    let mut out = text(data);
    if let Some(next) = next {
        if !out.ends_with('\n') {
            out.push('\n');
        }
        let _ = writeln!(out, "{}", polylogue_render::results::continuation(next));
    }
    out
}

/// The same typed read document, rendered for an MCP/model-facing transcript.
pub fn read(data: &Value, _color: bool) -> String {
    let page: polylogue_model::read::ReadPage = serde_json::from_value(data.clone())
        .expect("read operation returns its declared page type");
    let mut bytes = Vec::new();
    polylogue_render::encode_document(
        &page.document,
        polylogue_render::Format::Markdown,
        polylogue_render::EncodeOptions {
            commands: Some(crate::commands::declarations()),
            via: command_surface::Via::Mcp,
            model_facing: true,
            ..Default::default()
        },
        &mut bytes,
    )
    .expect("typed read document writes to a byte vector");
    String::from_utf8(bytes).expect("read encoder writes UTF-8")
}

/// A prepared brief, takeover or recall.
pub fn context(data: &Value, _color: bool) -> String {
    polylogue_render::terminal::escape(&context_text(data))
}

fn context_text(data: &Value) -> String {
    s(&data["prepared"]["text"])
}

/// `continue`: the sessions worth resuming here, or one session's plan.
pub fn resume(data: &Value, _color: bool) -> String {
    polylogue_render::terminal::escape(&resume_text(data))
}

fn resume_text(data: &Value) -> String {
    match data.as_array() {
        Some(sessions) => resumable(sessions),
        None => plan(data),
    }
}

fn resumable(sessions: &[Value]) -> String {
    let mut out = String::new();
    for c in sessions {
        let open = &c["open"];
        let mut why = Vec::new();
        if open["unanswered_prompt"].as_bool() == Some(true) {
            why.push("unanswered prompt".to_owned());
        }
        if open["open_tool_call"].as_bool() == Some(true) {
            why.push("unfinished tool call".to_owned());
        }
        if let Some(t) = open["todo"].as_str() {
            why.push(format!("todo: {t}"));
        }
        if let Some(w) = open["waiting_on"].as_str() {
            why.push(format!("waiting on {w}"));
        }
        for f in c["follow_ups"].as_array().into_iter().flatten() {
            why.push(s(f));
        }
        let _ = writeln!(
            out,
            "{}  {}  {}\n    {}",
            s(&c["session"]["last_activity"]),
            s(&c["session"]["id"]),
            s(&c["session"]["title"]),
            why.join("; ")
        );
    }
    out
}

fn plan(data: &Value) -> String {
    let line = |c: &Value| -> String {
        let mut words = vec![shell_quote(&s(&c["program"]))];
        words.extend(
            c["args"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|a| shell_quote(&s(a))),
        );
        let cmd = words.join(" ");
        match c["cwd"].as_str() {
            Some(d) => format!("cd {} && {cmd}", shell_quote(d)),
            None => cmd,
        }
    };
    let mut out = String::new();
    if !data["harness_command"].is_null() {
        let _ = writeln!(
            out,
            "harness command:    {}",
            line(&data["harness_command"])
        );
    }
    let _ = writeln!(
        out,
        "with context:       {}",
        line(&data["context_assisted"])
    );
    out
}

/// `chat list`: one line per chat action.
pub fn chat_actions(data: &Value, _color: bool) -> String {
    polylogue_render::terminal::escape(&chat_actions_text(data))
}

fn chat_actions_text(data: &Value) -> String {
    let actions = data["actions"].as_array().map_or(&[][..], Vec::as_slice);
    if actions.is_empty() {
        return "no chat actions\n".into();
    }
    let mut out = String::new();
    for a in actions {
        let target = a["conversation"].as_str().unwrap_or("new conversation");
        let _ = writeln!(
            out,
            "{}  {}  {}  {}  {}",
            s(&a["id"]),
            s(&a["status"]),
            s(&a["provider"]),
            target,
            s(&a["updated_at"])
        );
        if let Some(error) = a["last_error"].as_str() {
            let _ = writeln!(out, "    {error}");
        }
    }
    out
}

/// `backfill start`, `list` and `cancel`: one line per job.
pub fn backfill_jobs(data: &Value, _color: bool) -> String {
    polylogue_render::terminal::escape(&backfill_jobs_text(data))
}

fn backfill_jobs_text(data: &Value) -> String {
    let jobs: Vec<&Value> = match data["jobs"].as_array() {
        Some(jobs) => jobs.iter().collect(),
        None => data
            .get("job")
            .filter(|j| !j.is_null())
            .into_iter()
            .collect(),
    };
    if jobs.is_empty() {
        return "no backfill jobs\n".into();
    }
    let mut out = String::new();
    for j in jobs {
        let _ = writeln!(
            out,
            "{}  {}  {}  listed {}, captured {}, unchanged {}, failed {}  {}",
            s(&j["id"]),
            s(&j["provider"]),
            s(&j["status"]),
            j["listed"],
            j["captured"],
            j["unchanged"],
            j["failed"],
            s(&j["updated_at"])
        );
        if let Some(detail) = j["detail"].as_str() {
            let _ = writeln!(out, "    {detail}");
        }
    }
    out
}

/// One escaped diagnostic for local and remote CLI failures. JSON uses its own encoder.
pub(crate) fn diagnostic(
    kind: &str,
    message: &str,
    suggestion: Option<&str>,
    color: bool,
) -> String {
    let label = if kind.is_empty() {
        format!("polylogue: {message}\n")
    } else {
        format!("polylogue: {kind}: {message}\n")
    };
    let mut text = polylogue_render::terminal::heading(&label, color);
    if let Some(suggestion) = suggestion {
        text.push_str(&polylogue_render::terminal::escape(&format!(
            "polylogue: try: {suggestion}"
        )));
        text.push('\n');
    }
    text
}

#[cfg(test)]
mod tests;

/// What `daemon --until-idle` did, for a person.
pub fn idle_summary(summary: &polylogue_engine::RunSummary) -> String {
    let work = &summary.work;
    let mut out = format!(
        "idle after {} pass{} and {} import{} ({:.1} s in passes)\n",
        summary.passes,
        if summary.passes == 1 { "" } else { "es" },
        summary.imports,
        if summary.imports == 1 { "" } else { "s" },
        work.seconds,
    );
    for (name, n) in [
        ("full scans", work.full_scans),
        ("inputs seen", work.inputs_seen),
        ("inputs changed", work.inputs_changed),
        ("bytes read", work.bytes_read),
        ("bytes stored", work.bytes_stored),
        ("messages written", work.messages_written),
        ("sessions derived", work.sessions_derived),
    ] {
        let _ = writeln!(out, "  {name:<17} {n}");
    }
    for root in &work.unreadable_roots {
        let _ = writeln!(
            out,
            "  unreadable: {}",
            polylogue_render::terminal::escape(&root.display().to_string())
        );
    }
    for error in &work.errors {
        let _ = writeln!(
            out,
            "  error: {}",
            polylogue_render::terminal::escape(error)
        );
    }
    out
}

/// `verify`: the counts, then each problem.
pub fn verify(data: &Value, _color: bool) -> String {
    let Ok(found) = serde_json::from_value::<polylogue_store::verify::Verification>(data.clone())
    else {
        return json(data, false);
    };
    let counts = &found.counts;
    let mut out = format!(
        "{} inputs, {} revisions, {} extents, {} observations, {} blobs ({} bytes)\n",
        counts.inputs,
        counts.revisions,
        counts.extents,
        counts.observations,
        counts.blobs,
        counts.blob_bytes
    );
    let _ = writeln!(
        out,
        "blobs checked: {} ({} bytes, {} stored), {} with problems",
        found.blobs.checked, found.blobs.bytes, found.blobs.stored_bytes, found.blobs.mismatched
    );
    for format in &found.formats {
        let _ = writeln!(
            out,
            "  {:<28} {:>8} inputs {:>9} revisions {:>14} bytes",
            format.format, format.inputs, format.revisions, format.revision_bytes
        );
    }
    let classes = |counts: &[polylogue_store::verify::ClassCount]| {
        counts
            .iter()
            .map(|class| format!("{} {}", class.class, class.count))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(out, "observations: {}", classes(&found.observation_classes));
    let _ = writeln!(out, "revisions: {}", classes(&found.revision_kinds));
    for database in &found.databases {
        let _ = writeln!(
            out,
            "{}: {}",
            polylogue_render::terminal::escape(&database.file),
            polylogue_render::terminal::escape(&database.result.join("; "))
        );
    }
    if let Some(manifest) = &found.manifest {
        let _ = writeln!(
            out,
            "manifest: {} revisions hashed from {} bytes",
            manifest.revisions, manifest.bytes_read
        );
    }
    if found.problems.is_empty() {
        out.push_str("no problems\n");
    }
    for problem in &found.problems {
        let kind = serde_json::to_value(problem.kind)
            .ok()
            .and_then(|kind| kind.as_str().map(str::to_owned))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "{kind}: {}: {}",
            polylogue_render::terminal::escape(&problem.subject),
            polylogue_render::terminal::escape(&problem.detail)
        );
    }
    out
}
