//! Visibility and idle-stop choices for live line records.
use crate::protocol::follow::Line;
use serde_json::Value;
use std::io::{self, Write};

pub(crate) fn record(
    client: &crate::config::Client,
    step: &Value,
    data: &str,
    json: bool,
) -> io::Result<bool> {
    let line: Line = serde_json::from_str(data).map_err(io::Error::other)?;
    let kind = match &line {
        Line::State { .. } => "state",
        Line::Message { .. } => "text",
        Line::ToolCall { .. } => "tool",
        Line::Marks { .. } => "marks",
    };
    let matches = |key: &str| {
        step[key]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .any(|word| word == kind || word == "all")
    };
    let has_show = step["show"]
        .as_array()
        .is_some_and(|words| !words.is_empty());
    if (!has_show || matches("show") || kind == "state") && !matches("hide") {
        if json {
            writeln!(
                io::stdout().lock(),
                "{}",
                serde_json::json!({"event":"record","data":line})
            )?;
        } else {
            if let Some(text) = line.text(&client.zone()) {
                let mut output = io::stdout().lock();
                writeln!(output, "{}", polylogue_render::terminal::escape(&text))?;
                output.flush()?;
            }
        }
    }
    Ok(step["until_idle"] == true
        && matches!(&line, Line::Marks { marks }
        if marks.sessions.values().all(|mark| !matches!(mark.state.state.as_str(), "working" | "waiting"))))
}
