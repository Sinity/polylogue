//! Maud fragments shared by the web reader and script-free HTML exports.
use crate::encode::value_text;
use maud::{Markup, PreEscaped, html};
use polylogue_model::read::*;
use std::fmt::Display;

pub fn escape(text: &str) -> String {
    maud::html! { (text) }.into_string()
}
pub fn encode_path_segment(id: &str) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for byte in id.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(byte as char);
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}
/// Only ordinary web URLs and local references may enter an href. Control
/// characters and protocol-relative URLs are refused before scheme checks.
pub fn safe_url(url: &str) -> bool {
    if url.is_empty()
        || url.chars().any(|c| c.is_control() || c.is_whitespace())
        || url.starts_with("//")
        || url.contains('\\')
    {
        return false;
    }
    match url.split_once(':') {
        Some((scheme, _)) => matches!(
            scheme.to_ascii_lowercase().as_str(),
            "http" | "https" | "mailto"
        ),
        None => true,
    }
}
pub fn value<T: Display>(state: &ValueState<T>) -> Markup {
    use web_shell::ValueState as Shared;
    let shared = match state {
        ValueState::Value { value } => Shared::Measured(value.to_string()),
        ValueState::Unknown { reason } => Shared::Unknown {
            reason: reason.clone(),
        },
        ValueState::NotRecorded => Shared::NotRecorded,
        ValueState::NotApplicable => Shared::NotApplicable,
        ValueState::Partial { value, unknown } => Shared::Partial {
            value: value.to_string(),
            unknown: *unknown,
        },
        ValueState::Inferred { value } => Shared::Inferred(value.to_string()),
        ValueState::Pending => Shared::Pending,
        ValueState::Error { reason } => Shared::Error {
            reason: reason.clone(),
        },
        ValueState::Unavailable { .. } => {
            return html! { span class="value-state" data-state="unavailable" { (value_text(state)) } };
        }
        ValueState::Stale { .. } => {
            return html! { span class="value-state" data-state="stale" { (value_text(state)) } };
        }
        ValueState::Inherited { .. } => {
            return html! { span class="value-state" data-state="inherited" { (value_text(state)) } };
        }
    };
    web_shell::value_cell(&shared)
}
pub fn header(header: &ReadHeader) -> Markup {
    html! { header class="read-header" { h1 { @if let Some(title) = header.title.value() { (title) } @else { (&header.target) } } p class="meta" { code { (&header.target) } " · " (header.view) } dl { @for figure in &header.figures { dt { (&figure.name) } dd { (value(&figure.value)) @if let Some(unit)=&figure.unit { " " (unit) } } } } }
    }
}
/// Archived Markdown is parsed with raw HTML disabled and link destinations
/// checked. Images become links so an export never fetches archive text URLs.
fn markdown(text: &str) -> Markup {
    use pulldown_cmark::{CowStr, Event, Options, Parser, Tag, TagEnd};
    let events = Parser::new_ext(
        text,
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS,
    )
    .map(|event| match event {
        Event::Html(text) | Event::InlineHtml(text) => Event::Text(text),
        Event::Start(Tag::Link {
            link_type,
            dest_url,
            title,
            id,
        }) => Event::Start(Tag::Link {
            link_type,
            dest_url: if safe_url(&dest_url) {
                dest_url
            } else {
                CowStr::Borrowed("#")
            },
            title,
            id,
        }),
        Event::Start(Tag::Image {
            link_type,
            dest_url,
            title,
            id,
        }) => Event::Start(Tag::Link {
            link_type,
            dest_url: if safe_url(&dest_url) {
                dest_url
            } else {
                CowStr::Borrowed("#")
            },
            title,
            id,
        }),
        Event::End(TagEnd::Image) => Event::End(TagEnd::Link),
        other => other,
    });
    let mut output = String::new();
    pulldown_cmark::html::push_html(&mut output, events);
    PreEscaped(output)
}
fn ready(call: &ReadyCall) -> Markup {
    html! { code class="ready-call" { (&call.cli) } }
}
fn edit(edit: &EditView) -> Markup {
    html! { section class="edit" { h3 { (&edit.path) } p { @if edit.applied { "applied" } @else { "failed, may have changed the file" } " · " (edit.failed_edits) " failed edits" } pre { (value_text(&edit.diff)) } } }
}
fn card(card: &Card) -> Markup {
    html! { section class="read-card" { @match card {
        Card::Fields { title,figures } => { h3 { (title) } dl { @for figure in figures { dt { (&figure.name) } dd { (value(&figure.value)) @if let Some(unit)=&figure.unit { " " (unit) } } } } },
        Card::Session { session,open } => { h3 { "session:" (&session.id) " · " (session.title.as_ref().map(|title| title.text.as_str()).unwrap_or("title unknown")) } details { summary { "Session fields" } pre { (serde_json::to_string_pretty(session).expect("session serializes")) } } @if let Some(call)=open { (ready(call)) } },
        Card::LinkNode { reference,title,parents,own_messages,inherited_messages,own_cost,tree_cost,seed,open } => { h3 { @if *seed { "▶ " } (reference) " · " (value(title)) } p { "parents: " (parents.join(", ")) } p { (value(own_messages)) " own + " (value(inherited_messages)) " inherited · cost " (value(own_cost)) " own / " (value(tree_cost)) " tree" } @if let Some(call)=open { (ready(call)) } },
        Card::LinkEdge { from,to,relation,how,cross_link } => { p { code { (from) } " → " code { (to) } " · " (relation) @if let Some(how)=how { " (" (how) ")" } @if *cross_link { " · cross-link" } } },
        Card::Change { edit:e } => { (edit(e)) },
        Card::Live { state,waits_on,call,task,peers,processes } => { dl { dt { "State" } dd { (value(state)) } dt { "Waits on" } dd { (value(waits_on)) } dt { "Call" } dd { (value(call)) } dt { "Task" } dd { (value(task)) } } p { "Peers: " (peers.join(", ")) } ul { @for process in processes { li { code { (&process.reference) } " (parent " (process.parent.as_deref().unwrap_or("unknown")) ") · " (value(&process.command)) " · " (value(&process.state)) } } } },
        Card::Delegation { card } => { h3 { "→ " (&card.child) } dl { dt { "Objective" } dd { (value(&card.objective)) } dt { "Result" } dd { (value(&card.result)) } dt { "Duration (ms)" } dd { (value(&card.duration_ms)) } dt { "Cost" } dd { (value(&card.cost)) } dt { "Result use" } dd { (value(&card.result_use)) } } p { "Retries: " (card.retries.join(", ")) } @if let Some(call)=&card.open { (ready(call)) } },
    } } }
}
pub fn item(item: &ReadItem) -> Markup {
    html! { section class="read-item" data-reference=(&item.source.reference) data-block=[item.source.block] { @match &item.kind {
        ItemKind::MessageHeader { header } => { h2 { time { (header.time.at.map(|at|at.to_string()).unwrap_or_else(||"--:--".into())) } " " (value(&header.author)) @if let Some(model)=header.model.value() { " · " (model) } @if header.inherited { " · inherited" } @match &header.branch { BranchState::Abandoned { n,total } => { " · abandoned branch " (n.map(|n|n.to_string()).unwrap_or_else(||"?".into())) "/" (total.map(|n|n.to_string()).unwrap_or_else(||"?".into())) }, BranchState::Unknown { versions:Some(n) } => { " · active version unknown: " (n) " versions" }, _ => {} } @if let Some(variant)=&header.variant { " · ≈ " (&variant.key) @if variant.stale { " (stale)" } @if let Some(call)=&variant.show_original { " " (ready(call)) } } } code { (&item.source.reference) } @for assertion in &header.assertions { aside { "◆ " (&assertion.kind) ": " (&assertion.value) " · " (format!("{:?}",assertion.author)) } } },
        ItemKind::Text { text } => { div class="archived-text" { (markdown(text)) } },
        ItemKind::Thinking { words,content } => { @match content {
            ThinkingContent::Text { text } => { details class="thinking" { summary { "thinking (" (value(words)) " words)" } pre { (text) } } },
            ThinkingContent::Collapsed => { p class="thinking" { "thinking (" (value(words)) " words)" } },
            ThinkingContent::Encrypted => { p class="thinking" { "thinking (encrypted)" } }, ThinkingContent::Redacted => { p class="thinking" { "thinking (redacted)" } }, ThinkingContent::Absent { blocks } => { p class="thinking" { "thinking absent · " (blocks) " blocks" } }
        } },
        ItemKind::Call { call } => { details class="tool-call" open[call.input.is_some() || call.output.is_some()] { summary { "▸ " (value(&call.tool)) " " code { (value(&call.target)) } " " (value(&call.outcome)) @if let Some(exit)=call.exit { " · exit " (exit) } " · " (value(&call.duration_ms)) " ms" } @if let Some(error)=&call.error_line { p class="error" { (error) } } @if let Some(input)=&call.input { pre class="input" { (serde_json::to_string_pretty(input).expect("JSON serializes")) } } @if let Some(output)=&call.output { pre class="output" { (output) } } @if let Some(e)=&call.edit { (edit(e)) } @if let Some(child)=&call.child { (card(&Card::Delegation { card:child.clone() })) } } },
        ItemKind::Attachment { attachment,supplied_by,availability,link } => { p class="attachment" { "▣ " @if let Some(url)=link.as_deref().filter(|url|safe_url(url)) { a href=(url) { (attachment.name.as_deref().unwrap_or("attachment")) } } @else { (attachment.name.as_deref().unwrap_or("attachment")) } " (" (attachment.media_type.as_deref().unwrap_or("media type unknown")) ", " (attachment.size.map(|n|format!("{n} bytes")).unwrap_or_else(||"size unknown".into())) ") · supplied by " (value(supplied_by)) " · " (value(availability)) } },
        ItemKind::Other { type_name,data } => { details { summary { (type_name) } pre { (serde_json::to_string_pretty(data).expect("JSON serializes")) } } },
        ItemKind::Indicator { indicator,text,expand } => { aside class="indicator" data-kind=(format!("{indicator:?}")) { (text) @if let Some(call)=expand { " " (ready(call)) } } },
        ItemKind::Omission { count,unit,why,continue_call } => { aside class="omission" { "… " (value(count)) " " (unit) ": " (why) @if let Some(call)=continue_call { " " (ready(call)) } } },
        ItemKind::Card { card:c } => { (card(c)) },
        ItemKind::Raw { record } => { p { (&record.reference) " · " (&record.input) " · " (&record.media_type) } @if let Ok(text)=std::str::from_utf8(&record.bytes) { pre { (text) } } @else { @let encoded = { use base64::Engine; base64::engine::general_purpose::STANDARD.encode(&record.bytes) }; p { "binary bytes (base64)" } pre { (encoded) } } },
        ItemKind::Fragment { encoding,text,from,until,total } => { pre data-encoding=(format!("{encoding:?}")) { (text) } p class="fragment-position" { "… characters " (from) ".." (until) " of " (total) } },
    } } }
}
/// Ordinary navigation consumes the daemon's sealed calls, never guesses URLs.
pub fn footer(footer: &ReadFooter) -> Markup {
    html! { footer class="read-footer" { @for coverage in &footer.coverage { p { (coverage.omitted) " " (format!("{:?}",coverage.target)) " omitted from " (coverage.view) } } @for reference in &footer.referenced_by { p { "Referenced by " (&reference.reference) " · " (&reference.role) @if let Some(call)=&reference.open { " " (ready(call)) } } } @if let Some(call)=&footer.previous { nav aria-label="Previous page" { (ready(call)) } } @if let Some(call)=&footer.next { nav aria-label="Next page" { (ready(call)) } } } }
}
pub fn document(document: &ReadDocument) -> Markup {
    html! { article class="read-document" { (header(&document.header)) @for i in &document.items { (item(i)) } (footer(&document.footer)) } }
}
/// Shared design tokens are embedded directly in the export. This is the same
/// stylesheet served by web-shell, obtained through its pure asset adapter.
pub(crate) fn export_css() -> String {
    let request = http::Request::get("/assets/tokens.css")
        .body(())
        .expect("constant request");
    let response = web_shell::assets::serve(&request);
    let mut css = String::from_utf8(response.into_body()).expect("shared CSS is UTF-8");
    css.push_str("body{font-family:system-ui,sans-serif;color:var(--ink);background:var(--ground);max-width:80rem;margin:auto;padding:2rem}pre,code{font-family:monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:1rem;border-top:1px solid var(--line);padding-top:1rem}.read-item{margin-block:1rem}.unknown,.omission,.meta{color:var(--muted-ink)}.value-state[data-state=unknown]{text-decoration:underline dotted}.read-card{border:1px solid var(--line);padding:1rem}dt{font-weight:bold}dd{margin-bottom:.5rem}");
    css
}
