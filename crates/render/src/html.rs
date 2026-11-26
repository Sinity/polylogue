//! Maud fragments shared by the web reader and script-free HTML exports.
use crate::encode::value_text;
use command_surface::{Failure, FailureKind, OperationDeclarations};
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
/// Only web URLs and local references may enter an href. Control
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
    let name = match state {
        ValueState::Value { .. } => "known",
        ValueState::Unknown { .. } => "unknown",
        ValueState::NotRecorded => "not-recorded",
        ValueState::NotApplicable => "not-applicable",
        ValueState::Partial { .. } => "partial",
        ValueState::Inferred { .. } => "inferred",
        ValueState::Pending => "pending",
        ValueState::Error { .. } => "error",
        ValueState::Unavailable { .. } => "unavailable",
        ValueState::Stale { .. } => "stale",
        ValueState::Inherited { .. } => "inherited",
    };
    html! { span class="value-state" data-state=(name) { (value_text(state)) } }
}

pub fn header(header: &ReadHeader) -> Markup {
    visible_controls(header_markup(header))
}
fn header_markup(header: &ReadHeader) -> Markup {
    html! { header class="read-header" { h1 { @if let Some(title) = header.title.value() { (title) } @else { (&header.target) } } p class="meta" { code { (&header.target) } " · " (header.view) } @if let Some(line) = crate::encode::logical_line(&header.kind) { p class="meta" { (line) } } dl { @for figure in &header.figures { dt { (&figure.name) } dd { (value(&figure.value)) @if let Some(unit)=&figure.unit { " " (unit) } } } } }
    }
}
/// Archived Markdown is parsed with raw HTML disabled and link destinations
/// checked. Images become links so an export never fetches archive text URLs.
fn markdown(text: &str) -> Markup {
    use pulldown_cmark::{CowStr, Event, Options, Parser, Tag, TagEnd};
    let text = &crate::terminal::escape(text);
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
fn ready(call: &ReadyCall, commands: Option<&OperationDeclarations>) -> Result<Markup, Failure> {
    let commands = commands.ok_or_else(|| {
        Failure::new(
            FailureKind::BadRequest,
            "rendering a ready call requires operation declarations",
        )
    })?;
    let cli = commands.cli_command(&call.request)?;
    Ok(
        html! { @if let Ok(url) = crate::ready::reader_url(commands, &call.request) { a class="ready-call" href=(url) { (cli) } } @else { code class="ready-call" { (cli) } } },
    )
}
/// The notes after an attachment's description, shared with the text encoders.
fn attachment_notes(
    supplied_by: &ValueState<String>,
    availability: &ValueState<String>,
) -> Vec<String> {
    let mut notes = Vec::new();
    crate::encode::attachment_notes(supplied_by, availability, |note| notes.push(note));
    notes
}
fn edit(edit: &EditView) -> Markup {
    html! { section class="edit" { h3 { (&edit.path) } p { @if edit.applied { "applied" } @else { "failed, may have changed the file" } " · " (edit.failed_edits) " failed edits" } pre { (value_text(&edit.diff)) } } }
}
fn card(
    card: &Card,
    zone: &jiff::tz::TimeZone,
    commands: Option<&OperationDeclarations>,
) -> Result<Markup, Failure> {
    Ok(html! { section class="read-card" { @match card {
        Card::SavedQuery { item } => { @for line in crate::encode::saved_query_item_lines(item, zone) { p { (line) } } @if let SavedQueryItem::Dependent { resave, .. } = item.as_ref() { @for call in resave { (ready(call, commands)?) } } },
        Card::Trace(card) => { @for line in crate::encode::trace_lines(card) {p {(line)} } },
        Card::Run { question,read_at,author,count,exact,outcome } => { @for line in crate::encode::run_lines(question.as_ref(), read_at, author, *count, *exact, outcome.as_ref()) { p { (line) } } },
        Card::Fields { title,figures } => { h3 { (title) } dl { @for figure in figures { dt { (&figure.name) } dd { (value(&figure.value)) @if let Some(unit)=&figure.unit { " " (unit) } } } } },
        Card::Session { session,open,lineage_complete,annotations } => { @if lineage_complete == &ValueState::known(false) { p class="lineage-incomplete" { "Parents incomplete" } } h3 { "session:" (&session.id) " · " (session.title.as_ref().map(|title| title.text.as_str()).unwrap_or("title unknown")) } @for line in crate::encode::session_lines(session, zone) { p { (line) } } @for annotation in annotations { aside { (crate::encode::annotation_text(annotation, crate::Format::Html)) } } @if let Some(call)=open { (ready(call, commands)?) } },
        Card::LinkNode { reference,lineage,title,parents,own_messages,inherited_messages,own_cost,tree_cost,asked,open } => { h3 { @if *asked { "▶ " } (reference) " · " (value(title)) } p { "parents: " (parents.join(", ")) } @if let Some(lineage)=lineage { p { (crate::encode::lineage_text(lineage)) } } p { (value(own_messages)) " own + " (value(inherited_messages)) " inherited · cost " (value(own_cost)) " own / " (value(tree_cost)) " tree" } @if let Some(call)=open { (ready(call, commands)?) } },
        Card::LinkEdge { from,to,relation,how,branch_point,spawning_call,copies_prefix,cross_link } => {
            // Stable session object URLs keep recorded references without wrapping
            // them in an execution call.
            @let href = |reference: &str| match polylogue_model::Reference::parse(reference) {
                Ok(polylogue_model::Reference::Session(id)) => Some(format!("/s/{}", encode_path_segment(&id.to_string()))),
                _ => None,
            };
            p {
                @if let Some(url)=href(from) { a href=(url) { code { (from) } } } @else { code { (from) } }
                " → "
                @if let Some(url)=href(to) { a href=(url) { code { (to) } } } @else { code { (to) } }
                " · " (relation) @if let Some(how)=how { " (" (how) ")" } @if *copies_prefix { " · copied prefix" } @if let Some(at)=branch_point { " · at " code { (at) } } @if let Some(call)=spawning_call { " · call " code { (call) } } @if *cross_link { " · cross-link" }
            }
        },
        Card::Change { edit:e } => { (edit(e)) },
        Card::Live { state,waits_on,call,task,peers,processes } => { dl { dt { "State" } dd { (value(state)) } dt { "Waits on" } dd { (value(waits_on)) } dt { "Call" } dd { (value(call)) } dt { "Task" } dd { (value(task)) } } p { "Peers: " (peers.join(", ")) } ul { @for process in processes { li { code { (&process.reference) } " (parent " (process.parent.as_deref().unwrap_or("unknown")) ") · " (value(&process.command)) " · " (value(&process.state)) } } } },
        Card::Delegation { card } => { h3 { "→ " (&card.child) } dl { dt { "Objective" } dd { (value(&card.objective)) } dt { "Result" } dd { (value(&card.result)) } dt { "Duration (ms)" } dd { (value(&card.duration_ms)) } dt { "Cost" } dd { (value(&card.cost)) } dt { "Result use" } dd { (value(&card.result_use)) } } p { "Retries: " (card.retries.join(", ")) } @if let Some(call)=&card.open { (ready(call, commands)?) } },
    } } })
}
pub fn item(
    item: &ReadItem,
    inline_limit: usize,
    zone: &jiff::tz::TimeZone,
    commands: Option<&OperationDeclarations>,
) -> Result<Markup, Failure> {
    Ok(visible_controls(item_markup(
        item,
        inline_limit,
        zone,
        commands,
    )?))
}
fn item_markup(
    item: &ReadItem,
    inline_limit: usize,
    zone: &jiff::tz::TimeZone,
    commands: Option<&OperationDeclarations>,
) -> Result<Markup, Failure> {
    Ok(
        html! { section class="read-item" data-row="" tabindex="-1" data-reference=(&item.source.reference) data-block=[item.source.block] { @match &item.kind {
            ItemKind::MessageHeader { header } => { h2 { time { (crate::encode::message_time(&header.time, zone, false)) } " " (value(&header.author)) @if let Some(model)=header.model.value() { " · " (model) } @if header.inherited { " · inherited" } @match &header.branch { BranchState::Abandoned { n,total } => { " · abandoned branch " (n.map(|n|n.to_string()).unwrap_or_else(||"unknown".into())) " of " (total.map(|n|n.to_string()).unwrap_or_else(||"unknown".into())) }, BranchState::Unknown { versions:Some(n) } => { " · active version unknown: " (n) " versions" }, _ => {} } @if let Some(alternate)=&header.alternate { " · ≈ " (&alternate.key) @if alternate.stale { " (stale)" } @if let Some(call)=&alternate.show_original { " " (ready(call, commands)?) } } } code { (&item.source.reference) } @for annotation in &header.annotations { aside { (crate::encode::annotation_text(annotation, crate::Format::Html)) } } },
            ItemKind::Text { text } => { div class="archived-text" { (markdown(text)) } },
            ItemKind::Thinking { words,content } => { @match content {
                ThinkingContent::Text { text } => { details class="thinking" { summary { "thinking (" (value(words)) " words)" } pre { (text) } } },
                ThinkingContent::Collapsed => { p class="thinking" { "thinking (" (value(words)) " words)" } },
                ThinkingContent::Encrypted => { p class="thinking" { "thinking (encrypted)" } }, ThinkingContent::Redacted => { p class="thinking" { "thinking (redacted)" } }, ThinkingContent::Absent { blocks } => { p class="thinking" { "thinking absent · " (blocks) " blocks" } }
            } },
            ItemKind::Call { call } => { details class="tool-call" open[call.input.is_some() || call.output.is_some()] { summary { "▸ " (crate::encode::call_label(call)) "  " code { @if let Some(target) = call.target.value() { (crate::encode::cut_target(&target.to_string())) } @else { (value(&call.target)) } } "  " (value(&call.outcome)) @for part in [crate::encode::call_exit(call), crate::encode::call_duration(call)].into_iter().flatten() { "  " (part) } } @if let Some(error)=&call.error_line { p class="error" { (error) } } @if let Some(input)=&call.input { pre class="input" { (serde_json::to_string_pretty(input).expect("JSON serializes")) } } @if let Some(output)=&call.output { pre class="output" { (output) } } @if let Some(excerpt)=&call.output_excerpt { @for part in &excerpt.parts { @match part { OutputPart::Text {text,..}=>{ pre { (text) } }, OutputPart::Omission {lines,characters,..}=>{ p class="omission" { "… " (lines) " lines / " (characters) " characters skipped" } } } } @if let Some(open)=&excerpt.open { (ready(open, commands)?) } } @if let Some(e)=&call.edit { (edit(e)) } @if let Some(child)=&call.child { (card(&Card::Delegation { card:child.clone() }, zone, commands)?) } } },
            ItemKind::Attachment { attachment,supplied_by,availability,link,inline_bytes,verified_hash,open } => { @if let Some(src)=inline_image(attachment,inline_bytes.as_deref(),verified_hash.as_ref(),inline_limit) { img src=(src) alt=(attachment.name.as_deref().unwrap_or("attachment")); } p class="attachment" { "▣ " @if let Some(url)=link.as_deref().filter(|url|safe_url(url)) { a href=(url) { (attachment.name.as_deref().unwrap_or("attachment")) } } @else { (attachment.name.as_deref().unwrap_or("attachment")) } " (" (attachment.media_type.as_deref().unwrap_or("media type unknown")) ", " (attachment.size.map(|n|format!("{n} bytes")).unwrap_or_else(||"size unknown".into())) ")" @for note in attachment_notes(supplied_by, availability) { " · " (note) } @if let Some(open)=open { " " (ready(open, commands)?) } } },
            ItemKind::TextFragment { record,origin } => { aside class="partial-source" { p { "Selected source text · " code { (&record.reference) } " · bytes " (record.fragment.coverage.start) ".." (record.fragment.coverage.end) } p { "Source classification: " (crate::encode::value_text(origin)) " · recorded source length: " (record.fragment.coverage.total_bytes.map_or_else(||"unknown".into(),|total|total.to_string())) } pre class="archived-text" { (&record.fragment.text) } } },
            ItemKind::LinkIssue { issue } => { aside class="indicator link-issue" { "Parents incomplete: " (issue) } },
            ItemKind::Other { type_name,data } => { details { summary { (type_name) } pre { (serde_json::to_string_pretty(data).expect("JSON serializes")) } } },
            ItemKind::Indicator { indicator,text,expand } => { aside class="indicator" data-kind=(indicator.as_str()) { (crate::encode::indicator_line(*indicator, text)) @if let Some(call)=expand { " " (ready(call, commands)?) } } },
            ItemKind::Omission { count,unit,why,continue_call } => { aside class="omission" { "… " (value(count)) " " (unit) ": " (why) @if let Some(call)=continue_call { " " (ready(call, commands)?) } } },
            ItemKind::Card { card:c } => { (card(c, zone, commands)?) },
            ItemKind::Raw { record } => { p { (&record.reference) " · " (&record.input) " · " (&record.media_type) } @if let Ok(text)=std::str::from_utf8(&record.bytes) { pre { (text) } } @else { @let encoded = { use base64::Engine; base64::engine::general_purpose::STANDARD.encode(&record.bytes) }; p { "binary bytes (base64)" } pre { (encoded) } } },
            ItemKind::Fragment { text,position,until,total } => { p { "Encoded ReadItem JSON fragment" } pre data-encoding="read_item_json" { (text) } p class="fragment-position" { "… encoded characters " (position.character) ".." (until) " of " (total) } },
        } } },
    )
}
/// Navigation consumes the daemon's sealed calls, never guesses URLs.
pub fn footer(
    footer: &ReadFooter,
    commands: Option<&OperationDeclarations>,
) -> Result<Markup, Failure> {
    Ok(visible_controls(footer_markup(footer, commands)?))
}
/// The block every reading ends with, under one heading in every format.
pub fn cited_and_linked(
    footer: &ReadFooter,
    commands: Option<&OperationDeclarations>,
) -> Result<Markup, Failure> {
    Ok(visible_controls(
        html! { @if !footer.referenced_by.is_empty() { h2 { (crate::encode::CITED_AND_LINKED) } @for reference in &footer.referenced_by { p { code { (&reference.reference) } " · " (&reference.role) @if let Some(call)=&reference.open { " " (ready(call, commands)?) } } } } },
    ))
}
fn footer_markup(
    footer: &ReadFooter,
    commands: Option<&OperationDeclarations>,
) -> Result<Markup, Failure> {
    Ok(
        html! { footer class="read-footer" { @for coverage in &footer.coverage { p { (coverage.omitted) " " (coverage.target.as_str()) " omitted from " (coverage.view) } } (cited_and_linked(footer, commands)?) @if let Some(call)=&footer.previous { nav aria-label="Previous page" { (ready(call, commands)?) } } @if let Some(call)=&footer.next { nav aria-label="Next page" { (ready(call, commands)?) } } } },
    )
}
pub fn document(
    document: &ReadDocument,
    commands: Option<&OperationDeclarations>,
) -> Result<Markup, Failure> {
    let zone = crate::encode::header_zone(&document.header);
    let mut days = crate::encode::Days::default();
    Ok(
        html! { article class="read-document" { (header(&document.header)) @for i in &document.items { @if let Some(day) = days.announce(i, &zone) { p class="day" { (day) } } (item(i, DEFAULT_INLINE_IMAGE_BYTES, &zone, commands)?) } (footer(&document.footer, commands)?) } },
    )
}
/// Shared design tokens are embedded directly in the export. This is the same
/// stylesheet served by web-shell, obtained through its pure asset adapter.
pub(crate) fn export_css() -> String {
    let request = http::Request::get("/assets/tokens.css")
        .body(())
        .expect("constant request");
    let response = web_shell::assets::serve(&request);
    let mut css = String::from_utf8(response.into_body()).expect("shared CSS is UTF-8");
    // Exports use system fonts; shared @font-face URLs belong to the web host.
    if let Some(start) = css.find(":root{") {
        css = css[start..].to_owned();
    }
    css.push_str("body{font-family:system-ui,sans-serif;color:var(--ink);background:var(--ground);max-width:80rem;margin:auto;padding:2rem}pre,code{font-family:monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:1rem;border-top:1px solid var(--line);padding-top:1rem}.read-item{margin-block:1rem}.unknown,.omission,.meta{color:var(--muted-ink)}.value-state[data-state=unknown]{text-decoration:underline dotted}.read-card{border:1px solid var(--line);padding:1rem}dt{font-weight:bold}dd{margin-bottom:.5rem}");
    css
}

/// The largest single image inlined as a data URI in HTML, per attachment.
pub const DEFAULT_INLINE_IMAGE_BYTES: usize = 10 * 1024 * 1024;

/// Raster types permitted in an archive HTML image data URL.
pub fn inline_media_type(mime: &str) -> bool {
    matches!(
        mime,
        "image/png" | "image/jpeg" | "image/gif" | "image/webp" | "image/avif"
    )
}

fn inline_image(
    attachment: &polylogue_model::Attachment,
    bytes: Option<&[u8]>,
    verified_hash: Option<&polylogue_model::BlobHash>,
    limit: usize,
) -> Option<String> {
    let bytes = bytes?;
    let mime = attachment.media_type.as_deref()?;
    if bytes.len() > limit
        || limit == 0
        || !inline_media_type(mime)
        || attachment
            .blob
            .as_ref()
            .zip(verified_hash)
            .is_some_and(|(source, verified)| source != verified)
        || attachment.blob.as_ref().or(verified_hash) != Some(&polylogue_model::BlobHash::of(bytes))
    {
        return None;
    }
    use base64::Engine;
    Some(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

fn visible_controls(markup: Markup) -> Markup {
    PreEscaped(crate::terminal::escape(&markup.into_string()))
}
