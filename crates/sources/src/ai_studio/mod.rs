//! Google AI Studio prompts: the JSON document AI Studio saves to Drive
//! for each chat (`runSettings`, `systemInstruction`, `chunkedPrompt`).
//! They arrive two ways: files the operator brings in with `polylogue
//! import` (a Drive download or a Takeout), and the Drive collector, which
//! writes each Drive revision of a prompt to `<fileId>.json` in its spool
//! so the loop stores every revision as a revision of one input.
//!
//! A prompt is one session. Chunks are its messages in order; they carry no
//! IDs unless AI Studio wrote one, so a chunk keeps its identity by its
//! place (architecture 5.5), and an edited chunk becomes a new version of
//! the same message. `parts` split a model chunk into thoughts and answer
//! text; joined, they equal the chunk's `text`.

#[cfg(test)]
mod tests;

use crate::account::{Accounting, Disposition};
use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::fields::{Fields, Rule, unaccounted};
use crate::origin::Root;
use crate::parse::{DocumentParser, InlineBlob, MessageRow, Output, StepParser, Whole};
use base64::Engine as _;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, Event, Json, Link, LinkKind, Message,
    PromptKind, Role, Session, SessionKind, TitleSource, ToolOutcome, Usage,
};
use std::path::Path;

pub const ORIGIN: &str = "ai-studio";

/// Where the Drive collector writes prompts it fetched, one file per
/// prompt named by its Drive file ID.
pub const ROOT: Root = Root {
    env: Some("XDG_STATE_HOME"),
    default: ".local/state",
};
pub const LOCATIONS: &[&str] = &["polylogue/drive/ai-studio/*.json"];

/// The Drive collector's listing of the prompts folder, `files.json`:
/// `{"files": [{"id", "name", "createdTime", …}]}`, as Drive's
/// `files.list` returned it. It names the prompts: each entry offers its
/// name as the title of the session `ai-studio:<id>`.
pub const LISTING: &str = "files.json";

fn is_listing(path: &Path) -> bool {
    path.file_name().is_some_and(|n| n == LISTING)
}

/// A prompt document starts with one of its three top-level keys.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    if is_listing(path) {
        return first_byte(head) == Some(b'{') && first_key(head) == Some("files");
    }
    // Drive names a prompt by its title, with no extension, and titles
    // carry dots (`Notes. AI Contest`): the first key decides.
    let _ = path;
    first_byte(head) == Some(b'{')
        && matches!(
            first_key(head),
            Some("runSettings" | "systemInstruction" | "chunkedPrompt")
        )
}

pub fn open(path: &Path) -> Option<Box<dyn StepParser>> {
    if is_listing(path) {
        return Some(Box::new(Whole(Listing)));
    }
    let native = native_of(path)?;
    let title = title_of(path);
    Some(Box::new(Whole(AiStudio { native, title })))
}

/// The prompt's identity: the Drive file ID the collector names its files
/// by, otherwise the file name, which is the prompt's title in a Drive
/// download or Takeout (renaming a downloaded prompt makes a new session).
fn native_of(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    // A zip member path ends in its own name after `!/`.
    let name = name.rsplit("!/").next()?;
    let stem = name.strip_suffix(".json").unwrap_or(name);
    (!stem.is_empty()).then(|| stem.to_owned())
}

fn title_of(path: &Path) -> Option<String> {
    let in_spool = path
        .parent()
        .and_then(|p| p.file_name())
        .is_some_and(|n| n == "ai-studio");
    if in_spool {
        return None;
    }
    native_of(path)
}

pub struct AiStudio {
    native: String,
    /// The file name, used as the title when the document names none.
    title: Option<String>,
}

const DOCUMENT_FIELDS: Fields = &[
    ("runSettings", Rule::Read),
    ("systemInstruction", Rule::Read),
    ("chunkedPrompt", Rule::Read),
    ("chunkedPrompt.chunks", Rule::Read),
    (
        "chunkedPrompt.pendingInputs",
        Rule::Kept("the unsent draft in the input box"),
    ),
    ("citations", Rule::Kept("sources the replies cited")),
];

const CHUNK_FIELDS: Fields = &[
    ("role", Rule::Read),
    ("author", Rule::Read),
    ("id", Rule::Read),
    ("text", Rule::Read),
    ("parts", Rule::Read),
    ("parts[].text", Rule::Read),
    ("parts[].thought", Rule::Read),
    ("parts[].thoughtSignature", Rule::Read),
    ("parts[].executableCode", Rule::Read),
    ("parts[].codeExecutionResult", Rule::Read),
    ("parts[].inlineData", Rule::Read),
    ("parts[].fileData", Rule::Read),
    (
        "thoughtSignatures",
        Rule::Kept("the thoughts' signatures, also read from their parts"),
    ),
    ("isEdited", Rule::Kept("the operator edited the chunk")),
    (
        "safetyRatings",
        Rule::Kept("the model's safety classification of the reply"),
    ),
    (
        "videoMetadata",
        Rule::Kept("the clip range of an attached video"),
    ),
    ("isThought", Rule::Read),
    ("tokenCount", Rule::Read),
    ("finishReason", Rule::Read),
    ("createTime", Rule::Read),
    (
        "updateTime",
        Rule::Excluded("edit time; the creation time orders chunks"),
    ),
    (
        "thinkingBudget",
        Rule::Kept("the request's reasoning budget, kept with the run settings"),
    ),
    ("driveDocument", Rule::Read),
    ("driveImage", Rule::Read),
    ("driveAudio", Rule::Read),
    ("driveVideo", Rule::Read),
    ("inlineImage", Rule::Read),
    ("inlineFile", Rule::Read),
    ("youtubeVideo", Rule::Read),
    ("executableCode", Rule::Read),
    ("codeExecutionResult", Rule::Read),
    ("errorMessage", Rule::Read),
    ("grounding", Rule::Kept("search grounding sources")),
    ("branchParent", Rule::Read),
    (
        "branchChildren",
        Rule::Kept("other prompts branched from here"),
    ),
];

impl DocumentParser for AiStudio {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let doc = match serde_json::from_slice::<Json>(bytes) {
            Ok(v) if v.is_object() => v,
            Ok(_) => return sink.record("prompt", &Disposition::refused("not an object")),
            Err(e) => {
                return sink.record(
                    "prompt",
                    &Disposition::refused(format!("malformed JSON: {e}")),
                );
            }
        };
        let Some(sid) = session_id(ORIGIN, &self.native) else {
            return sink.record("prompt", &Disposition::refused("unusable file name"));
        };
        for path in unaccounted(&doc, DOCUMENT_FIELDS) {
            sink.acc.unread("prompt", &path);
        }
        let chunks = doc
            .pointer("/chunkedPrompt/chunks")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();

        let mut session = Session::new(sid.clone(), SessionKind::Main);
        if let Some(title) = &self.title {
            session.offer_title(title, TitleSource::Generated);
        }
        if let Some(at) = chunks
            .iter()
            .find_map(|c| c.get("createTime").and_then(time_of))
        {
            session.offer_start(at);
        }
        sink.out.sessions.push(session);
        sink.record("prompt", &Disposition::Materialized);

        let settings = doc.get("runSettings").cloned().unwrap_or(Json::Null);
        let model =
            str_of(&settings, "model").map(|m| m.strip_prefix("models/").unwrap_or(m).to_owned());
        sink.out.events.push(Event {
            session: sid.clone(),
            kind: "run_settings".into(),
            at: None,
            after: None,
            range: None,
            data: settings,
        });

        let mut previous = None;
        if let Some(text) = system_text(doc.get("systemInstruction")) {
            let id = message_id(ORIGIN, &format!("{}#system", self.native));
            if let Some(id) = id {
                let offset = sink.position();
                sink.out.messages.push(MessageRow {
                    offset,
                    end: offset + 1,
                    message: plain(
                        id.clone(),
                        sid.clone(),
                        Role::System,
                        vec![Block::Text { text }],
                    ),
                });
                previous = Some(id);
            }
        }

        for (i, chunk) in chunks.iter().enumerate() {
            let kind = chunk_role(chunk).unwrap_or("?").to_owned();
            for path in unaccounted(chunk, CHUNK_FIELDS) {
                sink.acc.unread(&kind, &path);
            }
            if i == 0
                && let Some(parent) = chunk
                    .pointer("/branchParent/promptId")
                    .and_then(Json::as_str)
                    .and_then(|p| session_id(ORIGIN, p))
            {
                sink.out.links.push(Link {
                    kind: LinkKind::Fork,
                    parent,
                    child: sid.clone(),
                    at: None,
                    call_id: None,
                    evidence: "branchParent.promptId".into(),
                });
            }
            match self.message(
                &sid,
                i,
                chunk,
                model.as_deref(),
                previous.clone(),
                &mut sink,
            ) {
                Ok(message) => {
                    previous = Some(message.id.clone());
                    let offset = sink.position();
                    sink.out.messages.push(MessageRow {
                        offset,
                        end: offset + 1,
                        message,
                    });
                    sink.record(&kind, &Disposition::Materialized);
                }
                Err(d) => sink.record(&kind, &d),
            }
        }
    }
}

fn chunk_role(chunk: &Json) -> Option<&str> {
    str_of(chunk, "role").or_else(|| str_of(chunk, "author"))
}

/// `systemInstruction` is `{}` when empty, `{"text": …}` or
/// `{"parts": [{"text": …}]}`.
fn system_text(value: Option<&Json>) -> Option<String> {
    let v = value?;
    if let Some(t) = str_of(v, "text") {
        return Some(t.to_owned());
    }
    let joined: String = v
        .get("parts")
        .and_then(Json::as_array)?
        .iter()
        .filter_map(|p| str_of(p, "text"))
        .collect();
    (!joined.is_empty()).then_some(joined)
}

impl AiStudio {
    fn message(
        &self,
        sid: &polylogue_model::SessionId,
        position: usize,
        chunk: &Json,
        model: Option<&str>,
        parent: Option<polylogue_model::MessageId>,
        sink: &mut Sink<'_>,
    ) -> Result<Message, Disposition> {
        let role = match chunk_role(chunk) {
            Some("user") => Role::User,
            Some("model") => Role::Assistant,
            Some(other) => return Err(Disposition::refused(format!("unknown role {other}"))),
            None => return Err(Disposition::refused("no role")),
        };
        let native = match str_of(chunk, "id") {
            Some(id) => format!("{}#{id}", self.native),
            None => format!("{}#{position}", self.native),
        };
        let id = message_id(ORIGIN, &native).ok_or_else(|| Disposition::refused("unusable id"))?;
        let thought = chunk.get("isThought").and_then(Json::as_bool) == Some(true);

        let mut blocks = text_blocks(chunk, thought);
        blocks.extend(media_blocks(chunk, role, sink));
        // Code execution and generated files sit on the chunk or in its
        // parts; each result answers the call before it.
        let parts = chunk
            .get("parts")
            .and_then(Json::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let mut calls = 0;
        for item in std::iter::once(chunk).chain(parts) {
            if let Some(code) = item.get("executableCode") {
                calls += 1;
                blocks.push(Block::ToolCall {
                    call_id: Some(format!("{native}:code{calls}")),
                    tool: Some("code_execution".into()),
                    input: code.clone(),
                    command: string_of(code, "code"),
                    path: None,
                });
            }
            if let Some(result) = item.get("codeExecutionResult") {
                let outcome = match str_of(result, "outcome") {
                    Some("OUTCOME_OK") | None => ToolOutcome::Ok,
                    Some(_) => ToolOutcome::Error { reason: None },
                };
                blocks.push(Block::ToolResult {
                    call_id: Some(format!("{native}:code{}", calls.max(1))),
                    outcome,
                    output: string_of(result, "output"),
                    data: Some(result.clone()),
                    attachments: Vec::new(),
                });
            }
            if let Some(inline) = item.get("inlineData") {
                blocks.push(inline_block(inline, role, sink));
            }
            // A file the part names by URI (`fileData`) rather than inlines.
            if let Some(file) = item.get("fileData") {
                blocks.push(Block::File(Attachment {
                    media_type: string_of(file, "mimeType"),
                    url: string_of(file, "fileUri"),
                    direction: if role == Role::User {
                        AttachmentDirection::Input
                    } else {
                        AttachmentDirection::Output
                    },
                    ..Attachment::default()
                }));
            }
        }
        if let Some(error) = str_of(chunk, "errorMessage") {
            blocks.push(Block::Other {
                type_name: "error".into(),
                data: Json::String(error.to_owned()),
            });
        }
        if blocks.is_empty() {
            return Err(Disposition::unknown("empty chunk"));
        }

        let mut message = plain(id, sid.clone(), role, blocks);
        message.parent = parent;
        message.at = chunk.get("createTime").and_then(time_of);
        if role == Role::Assistant {
            message.model = model.map(str::to_owned);
            message.stop_reason = string_of(chunk, "finishReason");
            let tokens = chunk.get("tokenCount").and_then(Json::as_u64);
            message.usage = Usage {
                output: tokens.filter(|_| !thought),
                reasoning: tokens.filter(|_| thought),
                ..Usage::default()
            };
        } else {
            message.prompt_kind = Some(PromptKind::Prose);
            message.usage = Usage {
                input: chunk.get("tokenCount").and_then(Json::as_u64),
                ..Usage::default()
            };
        }
        Ok(message)
    }
}

fn plain(
    id: polylogue_model::MessageId,
    session: polylogue_model::SessionId,
    role: Role,
    blocks: Vec<Block>,
) -> Message {
    Message {
        id,
        session,
        parent: None,
        role,
        at: None,
        model: None,
        prompt_kind: None,
        blocks,
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
    }
}

/// Thoughts and answer text, from `parts` when present (they carry the
/// split and the thought signatures), else from `text` and `isThought`.
fn text_blocks(chunk: &Json, thought: bool) -> Vec<Block> {
    let mut blocks = Vec::new();
    if let Some(parts) = chunk.get("parts").and_then(Json::as_array) {
        for part in parts {
            let Some(text) = str_of(part, "text") else {
                continue;
            };
            let is_thought = part.get("thought").and_then(Json::as_bool) == Some(true);
            let signature = string_of(part, "thoughtSignature");
            // Consecutive parts of one kind join into one block.
            match (blocks.last_mut(), is_thought) {
                (
                    Some(Block::Thinking {
                        text: Some(t),
                        signature: s,
                        ..
                    }),
                    true,
                ) => {
                    t.push_str(text);
                    if signature.is_some() {
                        *s = signature;
                    }
                }
                (Some(Block::Text { text: t }), false) => t.push_str(text),
                (_, true) => blocks.push(Block::Thinking {
                    text: Some(text.to_owned()),
                    signature,
                    redacted: false,
                }),
                (_, false) => blocks.push(Block::Text {
                    text: text.to_owned(),
                }),
            }
        }
        return blocks;
    }
    if let Some(text) = str_of(chunk, "text") {
        blocks.push(if thought {
            Block::Thinking {
                text: Some(text.to_owned()),
                signature: None,
                redacted: false,
            }
        } else {
            Block::Text {
                text: text.to_owned(),
            }
        });
    }
    blocks
}

/// Inline bytes (`{mimeType, data}`), stored as a blob: an image when the
/// type says so, else a file.
fn inline_block(inline: &Json, role: Role, sink: &mut Sink<'_>) -> Block {
    let mut a = Attachment {
        media_type: string_of(inline, "mimeType"),
        name: string_of(inline, "name"),
        direction: if role == Role::User {
            AttachmentDirection::Input
        } else {
            AttachmentDirection::Output
        },
        ..Attachment::default()
    };
    if let Some(bytes) = str_of(inline, "data")
        .and_then(|d| base64::engine::general_purpose::STANDARD.decode(d).ok())
    {
        let hash = BlobHash::of(&bytes);
        a.size = Some(bytes.len() as u64);
        a.blob = Some(hash);
        sink.out.blobs.push(InlineBlob {
            hash,
            size: bytes.len() as u64,
            bytes,
        });
    }
    let image = a
        .media_type
        .as_deref()
        .is_some_and(|m| m.starts_with("image/"));
    if image {
        Block::Image(a)
    } else {
        Block::File(a)
    }
}

/// Drive files, inline bytes and YouTube videos attached to a chunk.
fn media_blocks(chunk: &Json, role: Role, sink: &mut Sink<'_>) -> Vec<Block> {
    let direction = if role == Role::User {
        AttachmentDirection::Input
    } else {
        AttachmentDirection::Output
    };
    let mut blocks = Vec::new();
    for (key, image) in [
        ("driveImage", true),
        ("driveDocument", false),
        ("driveAudio", false),
        ("driveVideo", false),
    ] {
        if let Some(doc) = chunk.get(key) {
            let a = Attachment {
                name: string_of(doc, "name").or_else(|| string_of(doc, "title")),
                media_type: string_of(doc, "mimeType"),
                source_id: string_of(doc, "id").or_else(|| string_of(doc, "fileId")),
                url: None,
                direction,
                ..Attachment::default()
            };
            blocks.push(if image {
                Block::Image(a)
            } else {
                Block::File(a)
            });
        }
    }
    for (key, image) in [("inlineImage", true), ("inlineFile", false)] {
        if let Some(inline) = chunk.get(key) {
            let mut a = Attachment {
                media_type: string_of(inline, "mimeType"),
                name: string_of(inline, "name"),
                direction,
                ..Attachment::default()
            };
            if let Some(bytes) = str_of(inline, "data")
                .and_then(|d| base64::engine::general_purpose::STANDARD.decode(d).ok())
            {
                let hash = BlobHash::of(&bytes);
                a.size = Some(bytes.len() as u64);
                a.blob = Some(hash);
                sink.out.blobs.push(InlineBlob {
                    hash,
                    size: bytes.len() as u64,
                    bytes,
                });
            }
            blocks.push(if image {
                Block::Image(a)
            } else {
                Block::File(a)
            });
        }
    }
    if let Some(id) = chunk.pointer("/youtubeVideo/id").and_then(Json::as_str) {
        blocks.push(Block::File(Attachment {
            name: Some(format!("YouTube {id}")),
            media_type: Some("video/youtube".into()),
            url: Some(format!("https://www.youtube.com/watch?v={id}")),
            source_id: Some(id.to_owned()),
            direction,
            ..Attachment::default()
        }));
    }
    blocks
}

/// The collector's listing: session facts only, one record per prompt.
pub struct Listing;

const LISTING_FIELDS: Fields = &[
    ("id", Rule::Read),
    ("name", Rule::Read),
    ("createdTime", Rule::Read),
    (
        "modifiedTime",
        Rule::Excluded("the revisions carry when the prompt changed"),
    ),
    (
        "mimeType",
        Rule::Excluded("always the AI Studio prompt type"),
    ),
    (
        "headRevisionId",
        Rule::Excluded("the collector's own bookkeeping"),
    ),
    ("size", Rule::Excluded("each stored revision has its size")),
];

impl DocumentParser for Listing {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let files = match serde_json::from_slice::<Json>(bytes) {
            Ok(v) => v.get("files").and_then(Json::as_array).cloned(),
            Err(e) => {
                return sink.record(
                    "listing",
                    &Disposition::refused(format!("malformed JSON: {e}")),
                );
            }
        };
        let Some(files) = files else {
            return sink.record("listing", &Disposition::refused("no files array"));
        };
        for f in &files {
            for path in unaccounted(f, LISTING_FIELDS) {
                sink.acc.unread("file", &path);
            }
            let Some(sid) = str_of(f, "id").and_then(|id| session_id(ORIGIN, id)) else {
                sink.record("file", &Disposition::refused("no id"));
                continue;
            };
            let mut session = Session::new(sid, SessionKind::Main);
            if let Some(name) = str_of(f, "name") {
                session.offer_title(name, TitleSource::Generated);
            }
            if let Some(at) = f.get("createdTime").and_then(time_of) {
                session.offer_start(at);
            }
            sink.out.sessions.push(session);
            sink.record("file", &Disposition::Materialized);
        }
    }
}
