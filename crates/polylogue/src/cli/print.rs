//! Answers printed for a person: lists numbered for `@N`, sessions and
//! briefs as text, notes, measurements and query results as tables.

use super::*;

/// Plain terminal output until the renderer (architecture 7, `render`)
/// takes these over: one line per session, one paragraph per message.
pub(super) fn print_plain(request: &Request, data: &serde_json::Value) {
    let s = |v: &serde_json::Value| v.as_str().unwrap_or("").to_owned();
    match request {
        Request::Read { .. } => match data["next"].as_str() {
            _ if data["content"].is_string() => print!("{}", s(&data["content"])),
            Some(next) => println!("{}\n(next page: --cursor {next})", s(&data["text"])),
            None => print!("{}", s(&data["text"])),
        },
        Request::Context { .. } if data["off"].as_bool() == Some(true) => {
            println!(
                "session-start briefs are off (`polylogue setting briefing.session_start on`)"
            );
        }
        Request::Context { .. } => {
            print!("{}", s(&data["prepared"]["text"]));
        }
        Request::Notes { .. } => {
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
                println!(
                    "{}  [{}] {}  ({}; {})",
                    s(&n["id"]),
                    label,
                    s(&n["value"]),
                    targets.join(", "),
                    s(&n["author"])
                );
            }
        }
        Request::Resumable { .. } => {
            for c in data.as_array().into_iter().flatten() {
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
                println!(
                    "{}  {}  {}\n    {}",
                    s(&c["session"]["last_activity"]),
                    s(&c["session"]["id"]),
                    s(&c["session"]["title"]),
                    why.join("; ")
                );
            }
        }
        Request::Continue { .. } => {
            let line = |c: &serde_json::Value| -> String {
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
            if !data["native"].is_null() {
                println!("native resume:      {}", line(&data["native"]));
            }
            println!("with context:       {}", line(&data["context_assisted"]));
        }
        Request::Query { .. } if data["outcome"] == "measure" => {
            let text = crate::daemon::measure::find(data["measure"].as_str().unwrap_or(""))
                .map_or(polylogue_render::measures::json as fn(&_) -> _, |(m, _)| {
                    m.text
                });
            print!("{}", text(&data["data"]));
        }
        Request::Query { .. } => {
            if matches!(data["shape"].as_str(), Some("rows" | "matches")) {
                crate::row_numbers::remember(
                    &Archive::default_root(),
                    &crate::row_numbers::row_ids(&data["rows"]),
                );
            }
            let _ = polylogue_render::results::render(
                data,
                polylogue_render::Format::Terminal,
                &mut std::io::stdout().lock(),
            );
        }
        _ => println!("{}", serde_json::to_string_pretty(data).unwrap_or_default()),
    }
}
