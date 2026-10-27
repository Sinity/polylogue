//! HTML content shared by exported documents and the web reader.

use super::{Rendered, View, outcome_label, rendered_blocks};
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

/// Render a structured page already masked by the shared pager. A page never
/// hydrates hidden tool output; continuation is an ordinary reader link.
pub fn page_slices(
    slices: &[crate::model_pages::BlockSlice],
    out: &mut dyn io::Write,
) -> io::Result<()> {
    if slices.first().is_some_and(|slice| slice.from > 0) {
        write!(
            out,
            "<p class=\"meta\">This page starts within a source block.</p>"
        )?;
    }
    let mut current = None;
    for slice in slices {
        let id = slice.message.as_ref().map(|m| &m.id);
        if id != current {
            if current.is_some() {
                write!(out, "</section>")?;
            }
            if let Some(message) = &slice.message {
                let encoded = encode_path_segment(&message.id.to_string());
                write!(
                    out,
                    "<section class=\"message {}\" id=\"m-{encoded}\" tabindex=\"-1\" data-reference=\"message:{}\"><h2><a href=\"/m/{encoded}\">{}</a> <span class=\"meta\">{}</span></h2>",
                    message.role.as_str(),
                    escape(&message.id.to_string()),
                    message.role.as_str(),
                    escape(
                        &message
                            .at
                            .map_or_else(|| "Time unknown".into(), |at| at.to_string())
                    )
                )?;
                write!(
                    out,
                    "<p class=\"meta\">Written by {}{}</p>",
                    message.written_by.as_str(),
                    message
                        .model
                        .as_deref()
                        .map(|model| format!(" · {}", escape(model)))
                        .unwrap_or_default()
                )?;
            }
            current = id;
        }
        let label = slice.label.as_deref().unwrap_or(&slice.kind);
        write!(
            out,
            "<div class=\"block {}\" data-block=\"{}\" data-from=\"{}\" data-until=\"{}\">",
            escape(&slice.kind),
            slice.block,
            slice.from,
            slice.until
        )?;
        if slice.kind == "text" {
            write!(out, "<div class=\"text\">{}</div>", escape(&slice.text))?;
        } else if slice.kind == "tool_call" {
            let outcome = slice.outcome.map_or("unknown", |outcome| outcome.as_str());
            write!(
                out,
                "<details class=\"tool\"><summary><span class=\"outcome {outcome}\">{outcome}</span> <b>{}</b> <code>{}</code></summary><pre>{}</pre></details>",
                escape(label),
                escape(&first_line(&slice.text)),
                escape(&slice.text)
            )?;
        } else if matches!(slice.kind.as_str(), "tool_result" | "thinking") {
            write!(
                out,
                "<details class=\"{}\"><summary>{}</summary><pre>{}</pre></details>",
                if slice.kind == "thinking" {
                    "thinking"
                } else {
                    "tool result"
                },
                escape(label),
                escape(&slice.text)
            )?;
        } else {
            write!(
                out,
                "<p class=\"meta\">{}</p><pre>{}</pre>",
                escape(label),
                escape(&slice.text)
            )?;
        }
        if slice.more {
            write!(
                out,
                "<span class=\"meta\">Continues on the next page</span>"
            )?;
        }
        write!(out, "</div>")?;
    }
    if current.is_some() {
        write!(out, "</section>")?;
    }
    Ok(())
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
            Rendered::Document(doc) => super::documents::html(doc, view, out)?,
            Rendered::Message(value) => {
                write!(out, "{}", message(value, view, &results, &mut paired))?;
            }
            Rendered::Event(event) if view == View::Full => super::events::html(event, out)?,
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
    let ps = rendered_blocks(m, view);
    // Call IDs of this message's blocks, in the order `rendered blocks` keeps them.
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
