//! The Gemini app, from Google Takeout: `My Activity/Gemini Apps/
//! MyActivity.html`, one activity cell per prompt. A cell reads
//! `Prompted <prompt><br>[notes<br>]<date><br><response html>`; its caption
//! names the conversation (`https://gemini.google.com/app/<id>`). `Used …`
//! and `Created …` cells are other activity (a Gem, a canvas) and become
//! events.
//!
//! Messages are named by their conversation and time, which a later
//! Takeout repeats; a cell with no conversation link is a session of its
//! own. Responses are HTML in the export and are kept as text.

use crate::document::{Sink, message_id, session_id};
use crate::parse::{DocumentParser, MessageRow, Output, Parse};
use crate::{Accounting, Disposition};
use polylogue_model::{
    Attachment, AttachmentDirection, Block, Event, Json, Message, PromptKind, Role, Session,
    SessionKind, Timestamp, Usage,
};
use std::path::Path;

pub const ORIGIN: &str = "gemini-app";

pub fn recognize(path: &Path, head: &[u8]) -> bool {
    let p = path.to_string_lossy();
    p.ends_with("My Activity/Gemini Apps/MyActivity.html")
        && head.trim_ascii_start().starts_with(b"<")
}

pub fn open(_path: &Path) -> Option<Parse> {
    Some(Parse::Document(Box::new(GeminiApp)))
}

pub struct GeminiApp;

const CELL: &str = "<div class=\"outer-cell";
const BODY: &str = "mdl-typography--body-1\">";
const APP: &str = "https://gemini.google.com/app/";

struct Entry {
    verb: String,
    prompt: String,
    at: Option<Timestamp>,
    at_text: String,
    notes: Vec<String>,
    files: Vec<String>,
    response: String,
    conversation: Option<String>,
}

impl DocumentParser for GeminiApp {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let html = String::from_utf8_lossy(bytes);
        let mut entries = Vec::new();
        for cell in html.split(CELL).skip(1) {
            match entry(cell) {
                Some(e) => entries.push(e),
                None => sink.record("activity", &Disposition::refused("no content cell")),
            }
        }
        // The export lists newest first; a conversation reads oldest first.
        entries.sort_by_key(|e| e.at.map(|t| t.unix_ms()));
        let mut made = std::collections::HashSet::new();
        let mut last: std::collections::HashMap<String, polylogue_model::MessageId> =
            Default::default();
        for e in entries {
            let stamp =
                e.at.map(|t| t.unix_ms().to_string())
                    .unwrap_or_else(|| e.at_text.clone());
            let native = e
                .conversation
                .clone()
                .unwrap_or_else(|| format!("entry-{stamp}"));
            let Some(sid) = session_id(ORIGIN, &native) else {
                sink.record(&e.verb, &Disposition::refused("unusable conversation ID"));
                continue;
            };
            if made.insert(native.clone()) {
                let mut s = Session::new(sid.clone(), SessionKind::Main);
                if let Some(at) = e.at {
                    s.offer_start(at);
                }
                sink.out.sessions.push(s);
            }
            if e.verb != "Prompted" {
                sink.out.events.push(Event {
                    session: sid,
                    kind: "activity".into(),
                    at: e.at,
                    after: last.get(&native).cloned(),
                    range: None,
                    data: serde_json::json!({
                        "verb": e.verb, "text": e.prompt, "notes": e.notes, "detail": e.response
                    }),
                });
                sink.record(&e.verb, &Disposition::Materialized);
                continue;
            }
            let (Some(user_id), Some(reply_id)) = (
                message_id(ORIGIN, &format!("{native}#{stamp}-prompt")),
                message_id(ORIGIN, &format!("{native}#{stamp}-response")),
            ) else {
                sink.record(&e.verb, &Disposition::refused("unusable message ID"));
                continue;
            };
            let mut prompt_blocks = vec![Block::Text { text: e.prompt }];
            prompt_blocks.extend(e.files.iter().map(|f| {
                Block::File(Attachment {
                    name: Some(f.clone()),
                    direction: AttachmentDirection::Input,
                    ..Attachment::default()
                })
            }));
            let user = message(user_id.clone(), &sid, Role::User, e.at, prompt_blocks);
            let mut user = user;
            user.parent = last.get(&native).cloned();
            let mut reply_blocks = Vec::new();
            if !e.response.is_empty() {
                reply_blocks.push(Block::Text {
                    text: e.response.clone(),
                });
            }
            for note in &e.notes {
                reply_blocks.push(Block::Other {
                    type_name: "note".into(),
                    data: Json::String(note.clone()),
                });
            }
            for row in [
                Some(user),
                (!reply_blocks.is_empty()).then(|| {
                    let mut m =
                        message(reply_id.clone(), &sid, Role::Assistant, e.at, reply_blocks);
                    m.parent = Some(user_id.clone());
                    m
                }),
            ]
            .into_iter()
            .flatten()
            {
                let offset = sink.position();
                sink.out.messages.push(MessageRow {
                    offset,
                    end: offset + 1,
                    message: row,
                });
            }
            last.insert(native, reply_id);
            sink.record("Prompted", &Disposition::Materialized);
        }
    }
}

fn message(
    id: polylogue_model::MessageId,
    session: &polylogue_model::SessionId,
    role: Role,
    at: Option<Timestamp>,
    blocks: Vec<Block>,
) -> Message {
    Message {
        id,
        session: session.clone(),
        parent: None,
        role,
        at,
        model: None,
        prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
        blocks,
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
    }
}

/// One activity cell: the first body cell up to its closing `</div>`, and
/// the caption's conversation link.
fn entry(cell: &str) -> Option<Entry> {
    let start = cell.find(BODY)? + BODY.len();
    let body = &cell[start..];
    let body = &body[..body.find("</div>").unwrap_or(body.len())];
    let segments: Vec<&str> = body.split("<br>").collect();
    let first = text_of(segments.first()?);
    let (verb, prompt) = match first.split_once([' ', '\u{a0}']) {
        Some((v, rest)) => (v.to_owned(), rest.trim().to_owned()),
        None => (first.clone(), String::new()),
    };
    let date_at = segments
        .iter()
        .position(|s| parse_time(&text_of(s)).is_some());
    let (notes_end, at, at_text) = match date_at {
        Some(i) => {
            let t = text_of(segments[i]);
            (i, parse_time(&t), t)
        }
        None => (segments.len().min(2), None, String::new()),
    };
    let mut notes = Vec::new();
    let mut files = Vec::new();
    for s in &segments[1..notes_end.max(1)] {
        notes.push(text_of(s));
        let mut rest = *s;
        while let Some(i) = rest.find("<a href=\"") {
            rest = &rest[i + 9..];
            if let Some(end) = rest.find('"') {
                files.push(decode_entities(&rest[..end]));
                rest = &rest[end..];
            }
        }
    }
    let response = segments
        .get(notes_end + 1..)
        .map(|r| html_text(&r.join("<br>")))
        .unwrap_or_default();
    let conversation = cell.find(APP).map(|i| {
        cell[i + APP.len()..]
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect::<String>()
    });
    Some(Entry {
        verb,
        prompt,
        at,
        at_text,
        notes,
        files,
        response,
        conversation: conversation.filter(|c| !c.is_empty()),
    })
}

fn text_of(segment: &str) -> String {
    html_text(segment).trim().to_owned()
}

/// HTML to readable text: block ends become newlines, list items keep a
/// dash, entities are decoded, other tags dropped.
pub fn html_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        out.push_str(&decode_entities(&rest[..lt]));
        let Some(gt) = rest[lt..].find('>') else {
            rest = &rest[lt..];
            break;
        };
        let tag = rest[lt + 1..lt + gt].to_ascii_lowercase();
        let name = tag
            .trim_start_matches('/')
            .split([' ', '/'])
            .next()
            .unwrap_or("");
        match (tag.starts_with('/'), name) {
            (false, "br")
            | (true, "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "pre" | "tr") => {
                out.push('\n')
            }
            (false, "hr") => out.push_str("\n---\n"),
            (false, "li") => out.push_str("\n- "),
            (false, "h1" | "h2" | "h3" | "h4" | "h5" | "h6") => out.push_str("\n## "),
            (false, "td" | "th") => out.push_str(" | "),
            _ => {}
        }
        rest = &rest[lt + gt + 1..];
    }
    out.push_str(&decode_entities(rest));
    let mut cleaned = String::new();
    let mut blank = 0;
    for line in out.lines() {
        if line.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        cleaned.push_str(line.trim_end());
        cleaned.push('\n');
    }
    cleaned.trim().to_owned()
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let Some(semi) = tail.find(';').filter(|&j| j <= 10) else {
            out.push('&');
            rest = &tail[1..];
            continue;
        };
        let name = &tail[1..semi];
        let ch = match name {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some('\u{a0}'),
            "emsp" => Some('\u{2003}'),
            _ => name
                .strip_prefix("#x")
                .and_then(|h| u32::from_str_radix(h, 16).ok())
                .or_else(|| name.strip_prefix('#').and_then(|d| d.parse().ok()))
                .and_then(char::from_u32),
        };
        match ch {
            Some(c) => {
                out.push(c);
                rest = &tail[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// `Jul 27, 2026, 10:59:30 PM CEST` (a narrow no-break space before the
/// meridiem). An unknown zone abbreviation leaves the time unknown.
fn parse_time(s: &str) -> Option<Timestamp> {
    let s = s.replace(['\u{202f}', '\u{a0}'], " ");
    let mut it = s.split_whitespace();
    let month = match it.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let day: i64 = it.next()?.trim_end_matches(',').parse().ok()?;
    let year: i64 = it.next()?.trim_end_matches(',').parse().ok()?;
    let mut hms = it.next()?.split(':').map(|x| x.parse::<i64>());
    let (mut h, m, sec) = (hms.next()?.ok()?, hms.next()?.ok()?, hms.next()?.ok()?);
    match it.next()? {
        "AM" if h == 12 => h = 0,
        "PM" if h != 12 => h += 12,
        "AM" | "PM" => {}
        _ => return None,
    }
    let offset_h = match it.next()? {
        "UTC" | "GMT" => 0,
        "CET" | "BST" | "WEST" => 1,
        "CEST" | "EET" => 2,
        "EEST" | "MSK" => 3,
        "EST" | "CDT" => -5,
        "EDT" => -4,
        "CST" | "MDT" => -6,
        "MST" | "PDT" => -7,
        "PST" => -8,
        _ => return None,
    };
    if it.next().is_some() {
        return None;
    }
    // Days from the civil date (Howard Hinnant's algorithm).
    let (y, mo) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * mo + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let secs = days * 86_400 + h * 3600 + m * 60 + sec - offset_h * 3600;
    Timestamp::from_unix_ms(secs * 1000)
}

#[cfg(test)]
mod tests;
