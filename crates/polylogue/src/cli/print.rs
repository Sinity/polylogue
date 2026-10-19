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
    match data["next"].as_str() {
        _ if data["content"].is_string() => s(&data["content"]),
        Some(next) => format!("{}\n(next page: --cursor {next})\n", s(&data["text"])),
        None => s(&data["text"]),
    }
}

/// A prepared brief, takeover or recall.
pub fn context(data: &Value) -> String {
    if data["off"].as_bool() == Some(true) {
        return "session-start briefs are off (`polylogue setting briefing.session_start on`)\n"
            .into();
    }
    s(&data["prepared"]["text"])
}

pub fn notes(data: &Value) -> String {
    let mut out = String::new();
    for n in data.as_array().into_iter().flatten() {
        let targets: Vec<String> = n["targets"]
            .as_array()
            .into_iter()
            .flatten()
            .map(s)
            .collect();
        let label = n["key"]
            .as_str()
            .map_or_else(|| s(&n["kind"]), str::to_owned);
        let _ = writeln!(
            out,
            "{}  [{}] {}  ({}; {})",
            s(&n["id"]),
            label,
            s(&n["value"]),
            targets.join(", "),
            s(&n["author"])
        );
    }
    out
}

/// `continue`: the sessions worth resuming here, or one session's plan.
pub fn resume(data: &Value) -> String {
    match data.as_array() {
        Some(candidates) => resumable(candidates),
        None => plan(data),
    }
}

fn resumable(candidates: &[Value]) -> String {
    let mut out = String::new();
    for c in candidates {
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
    if !data["native"].is_null() {
        let _ = writeln!(out, "native resume:      {}", line(&data["native"]));
    }
    let _ = writeln!(
        out,
        "with context:       {}",
        line(&data["context_assisted"])
    );
    out
}
