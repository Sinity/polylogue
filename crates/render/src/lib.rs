//! Rendering (architecture 7). Every format is built from one stream of
//! rendered blocks; a view is a filter over that stream. Credentials are
//! masked on output, block by block, never by refusing.

pub mod measures;
pub mod model_pages;
pub mod results;
pub mod session_summary;

use polylogue_model::{Block, Event, Message, PromptKind, Role, Session, ToolOutcome};
use serde_json::{Value, json};
use std::io::{self, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Terminal,
    Markdown,
    Html,
    Json,
}

impl Format {
    pub fn parse(name: &str) -> Option<Format> {
        Some(match name {
            "terminal" => Format::Terminal,
            "markdown" | "md" => Format::Markdown,
            "html" => Format::Html,
            "json" => Format::Json,
            _ => return None,
        })
    }
}

/// Which blocks a view keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// Everything, tool calls and results included.
    Full,
    /// What the people and models said: prose and thinking, no tool output.
    Dialogue,
    /// The operator's own prompts only.
    Prompts,
    /// Tool calls and their outcomes.
    Actions,
}

impl View {
    pub fn parse(name: &str) -> Option<View> {
        Some(match name {
            "full" => View::Full,
            "dialogue" => View::Dialogue,
            "prompts" => View::Prompts,
            "actions" => View::Actions,
            _ => return None,
        })
    }

    /// Whether this view shows `block` of `message`.
    pub fn keeps(self, message: &Message, block: &Block) -> bool {
        match self {
            View::Full => true,
            View::Dialogue => matches!(block, Block::Text { .. } | Block::Thinking { .. }),
            View::Prompts => is_prompt(message) && matches!(block, Block::Text { .. }),
            View::Actions => matches!(block, Block::ToolCall { .. } | Block::ToolResult { .. }),
        }
    }
}

/// A prompt the operator typed: a user message neither the harness nor
/// another model wrote.
pub fn is_prompt(m: &Message) -> bool {
    m.role == Role::User && !matches!(m.prompt_kind, Some(PromptKind::Harness | PromptKind::Model))
}

/// One rendered unit of the stream every format is built from.
pub enum Rendered<'a> {
    SessionHeader(&'a Session),
    Message(&'a Message),
    Event(&'a Event),
}

/// Text with credentials masked, as one block.
pub fn mask(text: &str) -> String {
    let mut m = credential_masking::Masker::new();
    let mut out = Vec::with_capacity(text.len());
    // Writing to a Vec cannot fail.
    let _ = m.push(text, &mut out);
    let _ = m.finish(&mut out);
    String::from_utf8(out).unwrap_or_default()
}

/// One block as the stream carries it: its kind and its masked text.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub kind: &'static str,
    pub label: Option<String>,
    pub text: String,
}

/// The pieces a block renders to under a view.
pub fn pieces(message: &Message, view: View) -> Vec<Piece> {
    message
        .blocks
        .iter()
        .filter(|b| view.keeps(message, b))
        .filter_map(piece)
        .collect()
}

fn piece(block: &Block) -> Option<Piece> {
    Some(match block {
        Block::Text { text } => Piece {
            kind: "text",
            label: None,
            text: mask(text),
        },
        Block::Thinking { text, redacted, .. } => Piece {
            kind: "thinking",
            label: None,
            text: match text {
                Some(t) => mask(t),
                None if *redacted => "(redacted)".into(),
                None => "(encrypted)".into(),
            },
        },
        Block::ToolCall {
            tool,
            command,
            path,
            input,
            ..
        } => Piece {
            kind: "tool_call",
            label: Some(tool.clone().unwrap_or_else(|| "tool".into())),
            text: mask(
                &command
                    .clone()
                    .or_else(|| path.clone())
                    .unwrap_or_else(|| input.to_string()),
            ),
        },
        Block::ToolResult {
            outcome, output, ..
        } => Piece {
            kind: "tool_result",
            label: Some(outcome_label(outcome)),
            text: mask(output.as_deref().unwrap_or("")),
        },
        Block::Image(a) | Block::File(a) => Piece {
            kind: if matches!(block, Block::Image(_)) {
                "image"
            } else {
                "file"
            },
            label: a.media_type.clone(),
            text: a.name.clone().or_else(|| a.url.clone()).unwrap_or_default(),
        },
        Block::Other { type_name, .. } => Piece {
            kind: "other",
            label: Some(type_name.clone()),
            text: String::new(),
        },
    })
}

pub fn outcome_label(o: &ToolOutcome) -> String {
    match o {
        ToolOutcome::Ok => "ok".into(),
        ToolOutcome::Error { reason: None } => "error".into(),
        ToolOutcome::Error { reason: Some(r) } => match r {
            polylogue_model::ErrorReason::ExitCode(n) => format!("error (exit {n})"),
            polylogue_model::ErrorReason::Denied(why) => format!("denied ({why})"),
            polylogue_model::ErrorReason::Interrupted => "interrupted".into(),
            polylogue_model::ErrorReason::TimedOut => "timed out".into(),
        },
        ToolOutcome::NoResult => "no result".into(),
        ToolOutcome::Unknown { .. } => "unknown".into(),
    }
}

fn who(m: &Message) -> &'static str {
    match (m.role, m.prompt_kind) {
        (Role::Assistant, _) => "assistant",
        (Role::System, _) => "system",
        (Role::User, Some(PromptKind::Harness)) => "harness",
        (Role::User, Some(PromptKind::Model)) => "model",
        (Role::User, _) => "user",
    }
}

fn when(m: &Message) -> String {
    m.at.map(|t| t.to_string()).unwrap_or_default()
}

pub fn render(
    items: &[Rendered<'_>],
    format: Format,
    view: View,
    out: &mut dyn Write,
) -> io::Result<()> {
    match format {
        Format::Json => json_stream(items, view, out),
        Format::Html => html(items, view, out),
        Format::Markdown => markdown(items, view, out),
        Format::Terminal => terminal(items, view, out),
    }
}

fn title(s: &Session) -> String {
    s.title
        .as_ref()
        .map(|t| t.text.clone())
        .unwrap_or_else(|| s.id.to_string())
}

fn terminal(items: &[Rendered<'_>], view: View, out: &mut dyn Write) -> io::Result<()> {
    for item in items {
        match item {
            Rendered::SessionHeader(s) => {
                writeln!(out, "\x1b[1m{}\x1b[0m", title(s))?;
                writeln!(
                    out,
                    "\x1b[2m{}  {}\x1b[0m",
                    s.id,
                    s.cwd.last().map_or("", String::as_str)
                )?;
                writeln!(out)?;
            }
            Rendered::Message(m) => {
                let ps = pieces(m, view);
                if ps.is_empty() {
                    continue;
                }
                writeln!(out, "\x1b[2m── {} {}\x1b[0m", who(m), when(m))?;
                for p in ps {
                    match p.kind {
                        "tool_call" => writeln!(
                            out,
                            "\x1b[36m→ {}\x1b[0m {}",
                            p.label.unwrap_or_default(),
                            p.text
                        )?,
                        "tool_result" => {
                            writeln!(out, "\x1b[36m← {}\x1b[0m", p.label.unwrap_or_default())?;
                            for line in p.text.lines().take(20) {
                                writeln!(out, "  {line}")?;
                            }
                        }
                        "thinking" => writeln!(out, "\x1b[2;3m{}\x1b[0m", p.text)?,
                        "text" => writeln!(out, "{}", p.text)?,
                        kind => writeln!(
                            out,
                            "[{kind}{}] {}",
                            p.label.map(|l| format!(" {l}")).unwrap_or_default(),
                            p.text
                        )?,
                    }
                }
                writeln!(out)?;
            }
            Rendered::Event(e) if view == View::Full => {
                writeln!(
                    out,
                    "\x1b[2m· {} {}\x1b[0m",
                    e.kind,
                    e.at.map(|t| t.to_string()).unwrap_or_default()
                )?;
            }
            Rendered::Event(_) => {}
        }
    }
    Ok(())
}

fn fence(text: &str) -> String {
    let mut ticks = 3;
    while text.contains(&"`".repeat(ticks)) {
        ticks += 1;
    }
    "`".repeat(ticks)
}

fn markdown(items: &[Rendered<'_>], view: View, out: &mut dyn Write) -> io::Result<()> {
    for item in items {
        match item {
            Rendered::SessionHeader(s) => {
                writeln!(out, "# {}\n", title(s))?;
                writeln!(
                    out,
                    "`{}`{}\n",
                    s.id,
                    s.started.map(|t| format!(" · {t}")).unwrap_or_default()
                )?;
            }
            Rendered::Message(m) => {
                let ps = pieces(m, view);
                if ps.is_empty() {
                    continue;
                }
                writeln!(
                    out,
                    "## {}{}\n",
                    who(m),
                    m.at.map(|t| format!(" · {t}")).unwrap_or_default()
                )?;
                for p in ps {
                    match p.kind {
                        "text" => writeln!(out, "{}\n", p.text)?,
                        "thinking" => {
                            for line in p.text.lines() {
                                writeln!(out, "> {line}")?;
                            }
                            writeln!(out)?;
                        }
                        "tool_call" | "tool_result" => {
                            let f = fence(&p.text);
                            let arrow = if p.kind == "tool_call" { "→" } else { "←" };
                            writeln!(out, "{arrow} **{}**\n", p.label.unwrap_or_default())?;
                            if !p.text.is_empty() {
                                writeln!(out, "{f}\n{}\n{f}\n", p.text)?;
                            }
                        }
                        kind => writeln!(out, "*[{kind}]* {}\n", p.text)?,
                    }
                }
            }
            Rendered::Event(e) if view == View::Full => writeln!(out, "*· {}*\n", e.kind)?,
            Rendered::Event(_) => {}
        }
    }
    Ok(())
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn html(items: &[Rendered<'_>], view: View, out: &mut dyn Write) -> io::Result<()> {
    writeln!(
        out,
        "<!doctype html><meta charset=\"utf-8\"><article class=\"polylogue\">"
    )?;
    for item in items {
        match item {
            Rendered::SessionHeader(s) => {
                writeln!(
                    out,
                    "<h1>{}</h1><p class=\"id\"><code>{}</code></p>",
                    esc(&title(s)),
                    esc(&s.id.to_string())
                )?;
            }
            Rendered::Message(m) => {
                let ps = pieces(m, view);
                if ps.is_empty() {
                    continue;
                }
                writeln!(
                    out,
                    "<section class=\"message {}\"><h2>{} <time>{}</time></h2>",
                    who(m),
                    who(m),
                    esc(&when(m))
                )?;
                for p in ps {
                    match p.kind {
                        "text" => writeln!(
                            out,
                            "<div class=\"text\">{}</div>",
                            esc(&p.text).replace('\n', "<br>")
                        )?,
                        "thinking" => writeln!(
                            out,
                            "<details class=\"thinking\"><summary>thinking</summary><pre>{}</pre></details>",
                            esc(&p.text)
                        )?,
                        kind => writeln!(
                            out,
                            "<div class=\"{kind}\"><b>{}</b><pre>{}</pre></div>",
                            esc(&p.label.unwrap_or_default()),
                            esc(&p.text)
                        )?,
                    }
                }
                writeln!(out, "</section>")?;
            }
            Rendered::Event(e) if view == View::Full => {
                writeln!(out, "<p class=\"event\">{}</p>", esc(&e.kind))?
            }
            Rendered::Event(_) => {}
        }
    }
    writeln!(out, "</article>")
}

fn json_stream(items: &[Rendered<'_>], view: View, out: &mut dyn Write) -> io::Result<()> {
    let values: Vec<Value> = items
        .iter()
        .filter_map(|item| match item {
            Rendered::SessionHeader(s) => Some(json!({ "session": s })),
            Rendered::Message(m) => {
                let ps = pieces(m, view);
                (!ps.is_empty()).then(|| {
                    json!({
                        "message": m.id.to_string(),
                        "role": who(m),
                        "at": m.at,
                        "blocks": ps.into_iter().map(|p| json!({"kind": p.kind, "label": p.label, "text": p.text})).collect::<Vec<_>>(),
                    })
                })
            }
            Rendered::Event(e) => (view == View::Full).then(|| json!({ "event": e.kind, "at": e.at })),
        })
        .collect();
    writeln!(out, "{}", Value::Array(values))
}

#[cfg(test)]
mod tests;
