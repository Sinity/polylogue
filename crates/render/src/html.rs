//! HTML content shared by exported documents and the web reader.

use super::{Rendered, View, outcome_label, pieces};
use polylogue_model::{Block, Message};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io;

/// Escape text for HTML content and quoted attributes.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// An ID percent-encoded once as a whole, for a URL path segment.
pub fn encode_path_segment(id: &str) -> String {
    let mut out = String::new();
    for b in id.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// Render content without a document frame. Tool calls and their results
/// share one rendering policy on the command line and in the web reader.
/// Source content is masked before this function filters and pairs messages.
pub fn fragment(items: &[Rendered<'_>], view: View, out: &mut dyn io::Write) -> io::Result<()> {
    let messages: Vec<&Message> = items
        .iter()
        .filter_map(|item| match item {
            Rendered::Message(message) => Some(*message),
            _ => None,
        })
        .collect();
    let results = results_by_call(&messages);
    let mut paired = std::collections::HashSet::new();
    for item in items {
        match item {
            Rendered::SessionHeader { session, title } => {
                write!(
                    out,
                    "<h1>{}</h1><p class=\"id\"><code>{}</code></p>",
                    escape(&super::title(session, *title)),
                    escape(&session.id.to_string())
                )?;
            }
            Rendered::Message(value) => {
                write!(out, "{}", message(value, view, &results, &mut paired))?;
            }
            Rendered::Event(event) if view == View::Full => {
                write!(out, "<p class=\"event\">{}</p>", escape(&event.kind))?;
            }
            Rendered::Event(_) => {}
            Rendered::Notice(notice) => write!(out, "<aside>{}</aside>", escape(notice))?,
        }
    }
    Ok(())
}

pub(crate) fn document(
    items: &[Rendered<'_>],
    view: View,
    out: &mut dyn io::Write,
) -> io::Result<()> {
    writeln!(
        out,
        "<!doctype html><meta charset=\"utf-8\"><article class=\"polylogue\">"
    )?;
    fragment(items, view, out)?;
    writeln!(out, "</article>")
}

struct Outcome<'a> {
    label: String,
    text: &'a str,
}

fn results_by_call<'a>(messages: &[&'a Message]) -> HashMap<String, Outcome<'a>> {
    let mut out = HashMap::new();
    for m in messages {
        for b in &m.blocks {
            if let Block::ToolResult {
                call_id: Some(id),
                outcome,
                output,
                ..
            } = b
            {
                out.insert(
                    id.clone(),
                    Outcome {
                        label: outcome_label(outcome),
                        text: output.as_deref().unwrap_or(""),
                    },
                );
            }
        }
    }
    out
}

fn message(
    m: &Message,
    view: View,
    results: &HashMap<String, Outcome<'_>>,
    paired: &mut std::collections::HashSet<String>,
) -> String {
    let ps = pieces(m, view);
    // Call IDs of this message's blocks, in the order `pieces` keeps them.
    let call_ids: Vec<Option<&String>> = m
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::ToolCall { call_id, .. } | Block::ToolResult { call_id, .. } => {
                Some(call_id.as_ref())
            }
            _ => None,
        })
        .collect();
    let mut calls = call_ids.into_iter();
    let mut inner = String::new();
    for p in &ps {
        match p.kind {
            "text" => {
                let _ = write!(inner, "<div class=\"text\">{}</div>", paragraphs(&p.text));
            }
            "thinking" => {
                let _ = write!(
                    inner,
                    "<details class=\"thinking\"><summary>thinking</summary><pre>{}</pre></details>",
                    escape(&p.text)
                );
            }
            "tool_call" => {
                let id = calls.next().flatten();
                let result = id.and_then(|id| {
                    paired.insert(id.clone());
                    results.get(id)
                });
                let (label, output) = result.map_or(("no result", String::new()), |r| {
                    (r.label.as_str(), r.text.to_owned())
                });
                let class = if label == "ok" {
                    "ok"
                } else if label == "no result" || label == "unknown" {
                    "unknown"
                } else {
                    "error"
                };
                let _ = write!(
                    inner,
                    "<details class=\"tool\"><summary><span class=\"outcome {class}\">{}</span> <b>{}</b> <code>{}</code></summary><pre class=\"input\">{}</pre>{}</details>",
                    escape(label),
                    escape(p.label.as_deref().unwrap_or("tool")),
                    escape(&first_line(&p.text)),
                    escape(&p.text),
                    if output.is_empty() {
                        String::new()
                    } else {
                        format!("<pre class=\"output\">{}</pre>", escape(&output))
                    }
                );
            }
            "tool_result" => {
                let id = calls.next().flatten();
                if id.is_some_and(|id| paired.contains(id)) {
                    continue;
                }
                let _ = write!(
                    inner,
                    "<details class=\"tool result\"><summary><span class=\"outcome\">{}</span> result</summary><pre class=\"output\">{}</pre></details>",
                    escape(p.label.as_deref().unwrap_or("")),
                    escape(&p.text)
                );
            }
            kind => {
                let _ = write!(
                    inner,
                    "<div class=\"{kind}\">[{}{}] {}</div>",
                    escape(kind),
                    p.label
                        .as_deref()
                        .map(|l| format!(" {}", escape(l)))
                        .unwrap_or_default(),
                    escape(&p.text)
                );
            }
        }
    }
    if inner.is_empty() {
        return String::new();
    }
    let role = super::who(m);
    format!(
        "<section class=\"message {role}\" id=\"m-{}\" tabindex=\"-1\"><h2>{role}{}{}</h2>{inner}</section>",
        encode_path_segment(&m.id.to_string()),
        m.at.map(|t| format!(" <time>{}</time>", escape(&t.to_string())))
            .unwrap_or_default(),
        m.model
            .as_deref()
            .map(|x| format!(" <span class=\"model\">{}</span>", escape(x)))
            .unwrap_or_default(),
    )
}

fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or_default();
    if line.chars().count() > 120 {
        format!("{}…", line.chars().take(119).collect::<String>())
    } else {
        line.to_owned()
    }
}

fn paragraphs(text: &str) -> String {
    text.split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .map(|p| format!("<p>{}</p>", escape(p).replace('\n', "<br>")))
        .collect()
}

#[cfg(test)]
mod tests;
