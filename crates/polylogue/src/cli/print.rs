//! Answers as text for a person: query results as tables (numbered for
//! `@N`), measurements, sessions, briefs, notes and resume plans. Each
//! command names its printer in the command table; `--json` prints the
//! answer itself.

use polylogue_render::shell_quote;
use polylogue_store::Archive;
use serde_json::Value;
use std::fmt::Write as _;

fn s(v: &Value) -> String {
    v.as_str().unwrap_or("").to_owned()
}

/// The answer as indented JSON: commands without a text view.
pub fn json(data: &Value) -> String {
    format!(
        "{}\n",
        serde_json::to_string_pretty(data).unwrap_or_default()
    )
}

/// A query's rows as a table, remembered for `@N`, or a measurement in its
/// own text view.
pub fn query(data: &Value) -> String {
    if data["outcome"] == "measure" {
        let text = crate::daemon::measure::find(data["measure"].as_str().unwrap_or(""))
            .map_or(polylogue_render::measures::json as fn(&_) -> _, |(m, _)| {
                m.text
            });
        return text(&data["data"]);
    }
    if matches!(data["shape"].as_str(), Some("rows" | "matches")) {
        crate::row_numbers::remember(
            &Archive::default_root(),
            &crate::row_numbers::row_ids(&data["rows"]),
        );
    }
    let mut out = Vec::new();
    let _ = polylogue_render::results::render(data, polylogue_render::Format::Terminal, &mut out);
    String::from_utf8_lossy(&out).into_owned()
}

/// A session in a view, or one budgeted page with the cursor of the next.
pub fn read(data: &Value) -> String {
    let notice = if data["parents"]["complete"] == false && !data["content"].is_string() {
        "Parents incomplete; --json describes the missing links.\n"
    } else {
        ""
    };
    let content = match data["next"].as_str() {
        _ if data["content"].is_string() => s(&data["content"]),
        Some(next) => format!("{}\n(next page: --cursor {next})\n", s(&data["text"])),
        None => s(&data["text"]),
    };
    format!("{notice}{content}")
}

/// A prepared brief, takeover or recall.
pub fn context(data: &Value) -> String {
    if data["off"].as_bool() == Some(true) {
        return "session-start briefs are off (`polylogue setting briefing.session_start on`)\n"
            .into();
    }
    s(&data["prepared"]["text"])
}

/// `continue`: the sessions worth resuming here, or one session's plan.
pub fn resume(data: &Value) -> String {
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
pub fn chat_actions(data: &Value) -> String {
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
pub fn backfill_jobs(data: &Value) -> String {
    let jobs: Vec<&Value> = match data["jobs"].as_array() {
        Some(jobs) => jobs.iter().collect(),
        None => data.get("job").filter(|j| !j.is_null()).into_iter().collect(),
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
