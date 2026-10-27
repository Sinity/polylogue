//! Session output: one typed view projection, prose rendered as blocks, and
//! structured JSON preserving model fields and the importable export codec.
//! Source content is masked before views or formatting; response metadata is
//! also masked at the application boundary.

mod attachments;
pub mod documents;
pub mod events;
pub use attachments::{AttachmentLabel, AttachmentRef};
pub mod html;
pub mod measures;
pub use polylogue_model::masking::{mask, mask_json};
pub mod model_pages;
pub mod results;
pub mod schema;
pub mod session_summary;
pub mod terminal;

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
    pub const NAMES: &[&str] = &["terminal", "markdown", "html", "json"];

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
    pub const NAMES: &[&str] = &["full", "dialogue", "prompts", "actions"];

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
        self.keeps_block(is_prompt(message), block)
    }

    fn keeps_block(self, prompt: bool, block: &Block) -> bool {
        match self {
            View::Full => true,
            View::Dialogue => matches!(block, Block::Text { .. } | Block::Thinking { .. }),
            View::Prompts => prompt && matches!(block, Block::Text { .. }),
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
    SessionHeader {
        session: &'a Session,
        title: Option<&'a str>,
    },
    Message(&'a Message),
    Document(&'a polylogue_model::Document),
    Event(&'a Event),
    Notice(&'a str),
}

/// A word as the shell reads it back: bare when it is plain, else quoted.
pub fn shell_quote(w: &str) -> String {
    if !w.is_empty()
        && w.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=@%+,".contains(c))
    {
        w.to_owned()
    } else {
        format!("'{}'", w.replace('\'', r"'\''"))
    }
}

/// One block as the stream carries it: its kind and its masked text.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedBlock {
    pub kind: &'static str,
    pub label: Option<String>,
    pub text: String,
}

/// The rendered blocks a block renders to under a view.
fn rendered_blocks(message: &Message, view: View) -> Vec<RenderedBlock> {
    message
        .blocks
        .iter()
        .filter(|b| view.keeps(message, b))
        .filter_map(block)
        .collect()
}

fn block(block: &Block) -> Option<RenderedBlock> {
    Some(match block {
        Block::Text { text } => RenderedBlock {
            kind: "text",
            label: None,
            text: text.clone(),
        },
        Block::Thinking { text, redacted, .. } => RenderedBlock {
            kind: "thinking",
            label: None,
            text: match text {
                Some(t) => t.clone(),
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
        } => RenderedBlock {
            kind: "tool_call",
            label: Some(tool.clone().unwrap_or_else(|| "tool".into())),
            text: command
                .clone()
                .or_else(|| path.clone())
                .unwrap_or_else(|| input.to_string()),
        },
        Block::ToolResult {
            outcome, output, ..
        } => RenderedBlock {
            kind: "tool_result",
            label: Some(outcome_label(outcome)),
            text: output.clone().unwrap_or_default(),
        },
        Block::Image(a) | Block::File(a) => RenderedBlock {
            kind: if matches!(block, Block::Image(_)) {
                "image"
            } else {
                "file"
            },
            label: a.media_type.clone(),
            text: a.name.clone().or_else(|| a.url.clone()).unwrap_or_default(),
        },
        Block::Other { type_name, .. } => RenderedBlock {
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
        (_, Some(PromptKind::Harness)) => "harness",
        (Role::Assistant, _) => "assistant",
        (Role::System, _) => "system",
        (Role::User, Some(PromptKind::Model)) => "model",
        (Role::User, _) => "user",
    }
}

fn when(m: &Message) -> String {
    m.at.map(|t| t.to_string()).unwrap_or_default()
}

/// A session read keeps typed message fields; a view only selects blocks.
/// JSON is structured data, while the other formats display the selected stream.
pub fn session(
    session: &Session,
    display_title: Option<&str>,
    messages: Vec<Message>,
    view: View,
    format: Format,
    notices: &[String],
) -> io::Result<Value> {
    session_with_attachments(
        session,
        display_title,
        messages,
        view,
        format,
        notices,
        |_| Vec::new(),
    )
}

pub fn session_with_attachments(
    session: &Session,
    display_title: Option<&str>,
    mut messages: Vec<Message>,
    view: View,
    format: Format,
    notices: &[String],
    mut labels: impl FnMut(&Message) -> Vec<AttachmentLabel>,
) -> io::Result<Value> {
    let mut masker = polylogue_model::masking::MessageMasker::default();
    for message in &mut messages {
        masker.message(message);
        let labels = labels(message);
        if format != Format::Json {
            attachments::apply(message, view, labels);
        }
    }
    let selected = project(messages, view);
    if format == Format::Json {
        return Ok(
            json!({ "session": session, "display_title": display_title, "messages": selected }),
        );
    }
    let mut items = vec![Rendered::SessionHeader {
        session,
        title: display_title,
    }];
    items.extend(notices.iter().map(|notice| Rendered::Notice(notice)));
    items.extend(selected.iter().map(Rendered::Message));
    formatted(&items, format, view)
}

/// Export JSON uses the model's importable Document codec, including fields
/// such as links that have no prose rendering. It never serializes rendered blocks.
pub fn export(mut doc: polylogue_model::export::Document, format: Format) -> io::Result<Value> {
    doc.validate().map_err(io::Error::other)?;
    for session in &mut doc.sessions {
        let mut masker = polylogue_model::masking::MessageMasker::default();
        for message in &mut session.messages {
            masker.message(message);
        }
    }
    for document in &mut doc.documents {
        documents::mask(document);
    }
    if format == Format::Json {
        for session in &mut doc.sessions {
            for event in &mut session.events {
                polylogue_model::masking::mask_json(&mut event.data);
            }
        }
        for event in &mut doc.events {
            polylogue_model::masking::mask_json(&mut event.event.data);
        }
        return serde_json::to_value(&doc).map_err(io::Error::other);
    }
    let mut items = Vec::new();
    for session in &doc.sessions {
        items.push(Rendered::SessionHeader {
            session: &session.session,
            title: session.session.title.as_ref().map(|t| t.text.as_str()),
        });
        items.extend(session.messages.iter().map(Rendered::Message));
        items.extend(session.events.iter().map(Rendered::Event));
    }
    items.extend(doc.documents.iter().map(Rendered::Document));
    items.extend(doc.events.iter().map(|e| Rendered::Event(&e.event)));
    formatted(&items, format, View::Full)
}

fn project(mut messages: Vec<Message>, view: View) -> Vec<Message> {
    if view != View::Full {
        for message in &mut messages {
            let prompt = is_prompt(message);
            message
                .blocks
                .retain(|block| view.keeps_block(prompt, block));
        }
        messages.retain(|message| !message.blocks.is_empty());
    }
    messages
}

fn formatted(items: &[Rendered<'_>], format: Format, view: View) -> io::Result<Value> {
    let mut out = Vec::new();
    match format {
        Format::Html => html::document(items, view, &mut out)?,
        Format::Markdown => markdown(items, view, &mut out)?,
        Format::Terminal => terminal(items, view, &mut out)?,
        Format::Json => unreachable!("structured JSON is encoded before prose rendering"),
    }
    let name = match format {
        Format::Html => "html",
        Format::Markdown => "markdown",
        _ => "terminal",
    };
    Ok(
        json!({ "format": name, "content": String::from_utf8(out).expect("rendered strings are UTF-8") }),
    )
}

fn title(s: &Session, selected: Option<&str>) -> String {
    mask(selected.unwrap_or(&s.id.to_string()))
}

fn terminal(items: &[Rendered<'_>], view: View, out: &mut dyn Write) -> io::Result<()> {
    use terminal::line;
    for item in items {
        match item {
            Rendered::SessionHeader {
                session: s,
                title: selected,
            } => {
                line(out, format_args!("{}", title(s, *selected)))?;
                line(
                    out,
                    format_args!("{}  {}", s.id, s.cwd.last().map_or("", String::as_str)),
                )?;
                writeln!(out)?;
            }
            Rendered::Message(m) => {
                let ps = rendered_blocks(m, view);
                if ps.is_empty() {
                    continue;
                }
                line(out, format_args!("── {} {}", who(m), when(m)))?;
                for p in ps {
                    match p.kind {
                        "tool_call" => line(
                            out,
                            format_args!("→ {} {}", p.label.unwrap_or_default(), p.text),
                        )?,
                        "tool_result" => {
                            line(out, format_args!("← {}", p.label.unwrap_or_default()))?;
                            // Escape before splitting: CR and C1 line separators must stay visible.
                            for text in terminal::escape(&p.text).split('\n') {
                                writeln!(out, "  {text}")?;
                            }
                        }
                        "thinking" => line(out, format_args!("{}", p.text))?,
                        "text" => line(out, format_args!("{}", p.text))?,
                        kind => line(
                            out,
                            format_args!(
                                "[{kind}{}] {}",
                                p.label.map(|l| format!(" {l}")).unwrap_or_default(),
                                p.text
                            ),
                        )?,
                    }
                }
                writeln!(out)?;
            }
            Rendered::Document(doc) => documents::terminal(doc, view, out)?,
            Rendered::Event(e) if view == View::Full => events::terminal(e, out)?,
            Rendered::Event(_) => {}
            Rendered::Notice(notice) => line(out, format_args!("{notice}\n"))?,
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
            Rendered::SessionHeader {
                session: s,
                title: selected,
            } => {
                writeln!(out, "# {}\n", title(s, *selected))?;
                writeln!(
                    out,
                    "`{}`{}\n",
                    s.id,
                    s.started.map(|t| format!(" · {t}")).unwrap_or_default()
                )?;
            }
            Rendered::Message(m) => {
                let ps = rendered_blocks(m, view);
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
            Rendered::Document(doc) => documents::markdown(doc, view, out)?,
            Rendered::Event(e) if view == View::Full => events::markdown(e, out)?,
            Rendered::Event(_) => {}
            Rendered::Notice(notice) => writeln!(out, "{notice}\n")?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
