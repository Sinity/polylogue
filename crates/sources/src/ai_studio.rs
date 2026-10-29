//! Google AI Studio prompts: the JSON document AI Studio saves to Drive
//! for each chat (`runSettings`, `systemInstruction`, `chunkedPrompt`).
//! They arrive two ways: files the operator brings in with `polylogue
//! import` (a Drive download or a Takeout), and the Drive collector, which
//! writes each Drive revision of a prompt to `<fileId>.json` in its spool
//! so the loop stores every revision as a revision of one input.
//!
//! A prompt is one session. Chunks are its messages in order; they carry no
//! IDs unless AI Studio wrote one, so a chunk keeps its identity by its
//! place, and an edited chunk becomes a new version of
//! the same message. `parts` split a model chunk into thoughts and answer
//! text; joined, they equal the chunk's `text`.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::parse::{ExtractedBlob, MessageRow, Output, Parse};
use crate::{RecordOutcome, Records};
use base64::Engine as _;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, ErrorReason, Event, Json, Link, LinkKind,
    Message, PromptKind, Role, Session, SessionKind, TitleSource, ToolAction, ToolOutcome,
    UnknownReason, Usage,
};
use std::path::Path;

pub const ORIGIN: &str = "ai-studio";

/// Where the Drive collector writes prompts it fetched, one file per
/// prompt named by its Drive file ID.
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

pub fn open(path: &Path) -> Option<Parse> {
    if is_listing(path) {
        return Some(Parse::Export(Box::new(Listing::default())));
    }
    let native = native_of(path)?;
    let title = title_of(path);
    Some(Parse::Export(Box::new(AiStudio {
        native,
        title,
        state: PromptState::default(),
    })))
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
    state: PromptState,
}

#[derive(Default)]
struct PromptState {
    metadata: Option<Json>,
    model: Option<String>,
    emitted: bool,
    length: u64,
    chunks: u64,
    previous: Option<polylogue_model::MessageId>,
    pending_call: Option<String>,
    ambiguous_calls: bool,
    position: usize,
    started: bool,
    parent: Option<polylogue_model::SessionId>,
    ambiguous_parent: bool,
}

static DOCUMENT_FIELDS: Fields = Fields::new(&[
    ("runSettings", Rule::Kept("complete run settings event")),
    ("runSettings.model", Rule::Read(T::String)),
    ("systemInstruction", Rule::Read(T::StringOrObject)),
    ("systemInstruction.text", Rule::Read(T::String)),
    ("systemInstruction.parts[].text", Rule::Read(T::String)),
    ("chunkedPrompt", Rule::Read(T::Object)),
    ("chunkedPrompt.chunks", Rule::Read(T::Array)),
    (
        "chunkedPrompt.pendingInputs",
        Rule::Kept("the unsent draft in the input box"),
    ),
    ("citations", Rule::Kept("sources the replies cited")),
])
.for_kind("document");

static CHUNK_FIELDS: Fields = Fields::new(&[
    ("inlineData.mimeType", Rule::Read(T::String)),
    ("inlineData.data", Rule::Read(T::String)),
    ("fileData.mimeType", Rule::Read(T::String)),
    ("fileData.fileUri", Rule::Read(T::String)),
    ("parts[].inlineData.mimeType", Rule::Read(T::String)),
    ("parts[].inlineData.data", Rule::Read(T::String)),
    ("parts[].fileData.mimeType", Rule::Read(T::String)),
    ("parts[].fileData.fileUri", Rule::Read(T::String)),
    ("driveDocument.id", Rule::Read(T::String)),
    ("driveDocument.fileId", Rule::Read(T::String)),
    ("driveDocument.name", Rule::Read(T::String)),
    ("driveDocument.title", Rule::Read(T::String)),
    ("driveDocument.mimeType", Rule::Read(T::String)),
    ("driveImage.id", Rule::Read(T::String)),
    ("driveImage.fileId", Rule::Read(T::String)),
    ("driveImage.name", Rule::Read(T::String)),
    ("driveImage.title", Rule::Read(T::String)),
    ("driveImage.mimeType", Rule::Read(T::String)),
    ("driveAudio.id", Rule::Read(T::String)),
    ("driveAudio.fileId", Rule::Read(T::String)),
    ("driveAudio.name", Rule::Read(T::String)),
    ("driveAudio.title", Rule::Read(T::String)),
    ("driveAudio.mimeType", Rule::Read(T::String)),
    ("driveVideo.id", Rule::Read(T::String)),
    ("driveVideo.fileId", Rule::Read(T::String)),
    ("driveVideo.name", Rule::Read(T::String)),
    ("driveVideo.title", Rule::Read(T::String)),
    ("driveVideo.mimeType", Rule::Read(T::String)),
    ("inlineImage.data", Rule::Read(T::String)),
    ("inlineImage.mimeType", Rule::Read(T::String)),
    ("inlineImage.name", Rule::Read(T::String)),
    ("inlineFile.data", Rule::Read(T::String)),
    ("inlineFile.mimeType", Rule::Read(T::String)),
    ("inlineFile.name", Rule::Read(T::String)),
    ("youtubeVideo.id", Rule::Read(T::String)),
    ("role", Rule::Read(T::String)),
    ("author", Rule::Read(T::String)),
    ("id", Rule::Read(T::String)),
    ("text", Rule::Read(T::String)),
    ("parts", Rule::Read(T::Array)),
    ("parts[].text", Rule::Read(T::String)),
    ("parts[].thought", Rule::Read(T::Boolean)),
    ("parts[].thoughtSignature", Rule::Read(T::String)),
    (
        "parts[].executableCode",
        Rule::Kept("complete code input in tool block"),
    ),
    (
        "parts[].codeExecutionResult",
        Rule::Kept("complete code result in tool block"),
    ),
    ("parts[].inlineData", Rule::Read(T::Object)),
    ("parts[].fileData", Rule::Read(T::Object)),
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
    ("isThought", Rule::Read(T::Boolean)),
    ("tokenCount", Rule::Read(T::Number)),
    ("finishReason", Rule::Read(T::String)),
    ("createTime", Rule::Read(T::StringOrNumber)),
    (
        "updateTime",
        Rule::Excluded("edit time; the creation time orders chunks"),
    ),
    (
        "thinkingBudget",
        Rule::Kept("the request's reasoning budget, kept with the run settings"),
    ),
    ("driveDocument", Rule::Read(T::Object)),
    ("driveImage", Rule::Read(T::Object)),
    ("driveAudio", Rule::Read(T::Object)),
    ("driveVideo", Rule::Read(T::Object)),
    ("inlineImage", Rule::Read(T::Object)),
    ("inlineFile", Rule::Read(T::Object)),
    ("youtubeVideo", Rule::Read(T::Object)),
    (
        "executableCode",
        Rule::Kept("complete code input in tool block"),
    ),
    (
        "codeExecutionResult",
        Rule::Kept("complete code result in tool block"),
    ),
    ("errorMessage", Rule::Read(T::String)),
    ("grounding", Rule::Kept("search grounding sources")),
    (
        "branchParent",
        Rule::Kept("native branch marker in message attributes"),
    ),
    ("branchParent.promptId", Rule::Read(T::String)),
    (
        "branchChildren",
        Rule::Kept("other prompts branched from here"),
    ),
])
.for_kind("chunk");

impl ExportParser for AiStudio {
    fn message_range(&self, message: &Message) -> Option<std::ops::Range<u64>> {
        (message.role == Role::System).then_some(0..self.state.length)
    }

    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.state = PromptState::default();
        let (metadata, stats) =
            crate::export::scan_nested(reader, &["chunkedPrompt", "chunks"], |range, raw| {
                item.push(range, Vec::new(), None, raw.get().as_bytes())
            })?;
        self.state.chunks = stats.elements;
        self.state.metadata = Some(metadata);
        Ok(stats)
    }

    fn begin(
        &mut self,
        length: u64,
        _first: Option<&[u8]>,
    ) -> Result<Option<crate::parse::Step>, String> {
        let doc = self
            .state
            .metadata
            .as_ref()
            .ok_or("prompt metadata not scanned")?;
        self.state.length = length;
        let out = Output::default();
        let mut records = Records::default();
        if !doc.is_object() {
            records.record("prompt", &RecordOutcome::unreadable("not an object"));
        } else {
            records.fields("prompt", doc, &DOCUMENT_FIELDS);
            if !doc
                .pointer("/chunkedPrompt/chunks")
                .is_some_and(Json::is_array)
            {
                records.record("prompt", &RecordOutcome::unreadable("no chunks array"));
            } else if self.state.chunks == 0 {
                records.record("prompt", &RecordOutcome::skipped("empty chunks array"));
            }
            self.state.model = doc
                .get("runSettings")
                .and_then(|v| str_of(v, "model"))
                .map(|m| m.strip_prefix("models/").unwrap_or(m).to_owned());
        }
        Ok(Some(crate::parse::Step {
            output: out,
            counts: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            repeats_whole_conversations: false,
        }))
    }

    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let chunk: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let chunk = &chunk;
        let sid = session_id(ORIGIN, &self.native).ok_or("unusable file name")?;
        let i = self.state.position;
        self.state.position += 1;
        let kind = chunk_role(chunk).unwrap_or("?").to_owned();
        sink.records.fields(&kind, chunk, &CHUNK_FIELDS);
        if let Some(parent) = chunk
            .pointer("/branchParent/promptId")
            .and_then(Json::as_str)
            .and_then(|p| session_id(ORIGIN, p))
        {
            if self
                .state
                .parent
                .as_ref()
                .is_some_and(|known| known != &parent)
            {
                self.state.ambiguous_parent = true;
            }
            self.state.parent = Some(parent);
        }
        match self.message(&sid, i, chunk, sink) {
            Ok(mut message) => {
                if !self.state.emitted {
                    let doc = self
                        .state
                        .metadata
                        .take()
                        .ok_or("prompt metadata not scanned")?;
                    let mut session = Session::new(sid.clone(), SessionKind::Main);
                    if let Some(title) = &self.title {
                        session.offer_title(title, TitleSource::FileName);
                    }
                    sink.out.sessions.push(session);
                    sink.out.events.push({
                        let (at, time_evidence) = crate::time::parts(None);
                        Event {
                            session: Some(sid.clone()),
                            kind: "run_settings".into(),
                            at,
                            at_quality: time_evidence.quality,
                            at_precision: time_evidence.precision,
                            at_until: time_evidence.until,
                            after: None,
                            range: Some((0, self.state.length)),
                            data: doc.get("runSettings").cloned().unwrap_or(Json::Null),
                        }
                    });
                    let mut attrs = crate::fields::attributes(&doc, &DOCUMENT_FIELDS);
                    attrs.remove("runSettings");
                    if !attrs.is_empty() {
                        sink.out.events.push({
                            let (at, time_evidence) = crate::time::parts(None);
                            Event {
                                session: Some(sid.clone()),
                                kind: "prompt_metadata".into(),
                                at,
                                at_quality: time_evidence.quality,
                                at_precision: time_evidence.precision,
                                at_until: time_evidence.until,
                                after: None,
                                range: Some((0, self.state.length)),
                                data: Json::Object(attrs),
                            }
                        });
                    }
                    if let Some(text) = system_text(doc.get("systemInstruction")) {
                        let id = message_id(ORIGIN, &message_key(&self.native, "system", ""))
                            .ok_or("unusable system id")?;
                        sink.out.messages.push(MessageRow {
                            offset: 0,
                            end: self.state.length,
                            message: plain(
                                id.clone(),
                                sid.clone(),
                                Role::System,
                                vec![Block::Text { text }],
                            ),
                        });
                        self.state.previous = Some(id);
                    }
                    message.parent = self.state.previous.clone();
                    sink.records.record("prompt", &RecordOutcome::Used);
                    self.state.emitted = true;
                }
                if !self.state.started
                    && let Some(at) = chunk.get("createTime").and_then(time_of)
                {
                    let mut session = Session::new(sid.clone(), SessionKind::Main);
                    session
                        .offer_start(at.timestamp(), at.evidence())
                        .expect("decoded source time is valid");
                    sink.out.sessions.push(session);
                    self.state.started = true;
                }
                self.state.previous = Some(message.id.clone());
                let offset = sink.position();
                sink.out.messages.push(MessageRow {
                    offset,
                    end: offset + 1,
                    message,
                });
                sink.records.record(&kind, &RecordOutcome::Used);
            }
            Err(d) => {
                self.state.pending_call = None;
                self.state.ambiguous_calls = false;
                sink.records.record(&kind, &d);
            }
        }
        // The whole prompt must agree on its parent before a fork is emitted.
        if self.state.position as u64 == self.state.chunks
            && !self.state.ambiguous_parent
            && let Some(parent) = self.state.parent.take()
        {
            sink.out.links.push(Link {
                kind: LinkKind::Fork,
                parent,
                child: sid,
                at: None,
                call_id: None,
                evidence: "branchParent.promptId".into(),
            });
        }
        if !self.state.emitted && self.state.position as u64 == self.state.chunks {
            sink.records
                .record("prompt", &RecordOutcome::skipped("no usable chunks"));
        }
        Ok(())
    }
}

fn chunk_role(chunk: &Json) -> Option<&str> {
    str_of(chunk, "role").or_else(|| str_of(chunk, "author"))
}

/// `systemInstruction` is `{}` when empty, `{"text": …}` or
/// `{"parts": [{"text": …}]}`.
fn system_text(value: Option<&Json>) -> Option<String> {
    let v = value?;
    if let Some(text) = v.as_str() {
        return Some(text.to_owned());
    }
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
        &mut self,
        sid: &polylogue_model::SessionId,
        position: usize,
        chunk: &Json,
        sink: &mut Sink<'_>,
    ) -> Result<Message, RecordOutcome> {
        let native_role = chunk_role(chunk);
        if native_role != Some("model") {
            self.state.pending_call = None;
            self.state.ambiguous_calls = false;
        }
        let role = match native_role {
            Some("user") => Role::User,
            Some("model") => Role::Assistant,
            Some(other) => return Err(RecordOutcome::unreadable(format!("unknown role {other}"))),
            None => return Err(RecordOutcome::unreadable("no role")),
        };
        let native = match str_of(chunk, "id") {
            Some(id) => message_key(&self.native, "id", id),
            None => message_key(&self.native, "position", &position.to_string()),
        };
        let id =
            message_id(ORIGIN, &native).ok_or_else(|| RecordOutcome::unreadable("unusable id"))?;
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
                let call_id = format!("{native}:code{calls}");
                if self.state.pending_call.is_some() || self.state.ambiguous_calls {
                    self.state.pending_call = None;
                    self.state.ambiguous_calls = true;
                } else {
                    self.state.pending_call = Some(call_id.clone());
                }
                blocks.push(Block::ToolCall {
                    run_by_script: false,
                    call_id: Some(call_id),
                    tool: Some("code_execution".into()),
                    action: ToolAction::Code,
                    input: code.clone(),
                    command: string_of(code, "code"),
                    path: None,
                });
            }
            if let Some(result) = item.get("codeExecutionResult") {
                // A result answers the preceding unanswered call. Zero or
                // several unanswered calls leave its call unknown.
                let call_id = self.state.pending_call.take();
                self.state.ambiguous_calls = false;
                let outcome = match str_of(result, "outcome") {
                    Some("OUTCOME_OK") => ToolOutcome::Ok,
                    Some("OUTCOME_FAILED") => ToolOutcome::Error { reason: None },
                    Some("OUTCOME_DEADLINE_EXCEEDED") => ToolOutcome::Error {
                        reason: Some(ErrorReason::TimedOut),
                    },
                    _ => ToolOutcome::Unknown {
                        reason: UnknownReason::StatusFieldUnreadable,
                    },
                };
                blocks.push(Block::ToolResult {
                    reported_duration_ms: None,
                    call_id,
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
            return Err(RecordOutcome::skipped("empty chunk"));
        }

        let mut message = plain(id, sid.clone(), role, blocks);
        message.attrs = crate::fields::attributes(chunk, &CHUNK_FIELDS);
        for field in ["executableCode", "codeExecutionResult"] {
            message.attrs.remove(field);
        }
        message.parent = self.state.previous.clone();
        let (at, evidence) = crate::time::parts(chunk.get("createTime").and_then(time_of));
        message
            .set_time(at, evidence)
            .expect("decoded source time is valid");
        if role == Role::Assistant {
            message.model = self.state.model.clone();
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
        if role == Role::User {
            self.state.pending_call = None;
            self.state.ambiguous_calls = false;
        }
        Ok(message)
    }
}

// Lengths keep file names and native IDs containing separators unambiguous.
fn message_key(prompt: &str, kind: &str, value: &str) -> String {
    format!("{}:{prompt}#{kind}:{}:{value}", prompt.len(), value.len())
}

fn plain(
    id: polylogue_model::MessageId,
    session: polylogue_model::SessionId,
    role: Role,
    blocks: Vec<Block>,
) -> Message {
    {
        let (at, time_evidence) = crate::time::parts(None);
        Message {
            id,
            session,
            parent: None,
            role,
            at,
            at_quality: time_evidence.quality,
            at_precision: time_evidence.precision,
            at_until: time_evidence.until,
            model: None,
            prompt_kind: None,
            blocks,
            usage: Usage::default(),
            stop_reason: None,
            reported_cost_usd: None,
            duration_ms: None,
            attrs: Default::default(),
            production: Default::default(),
        }
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
        sink.out.blobs.push(ExtractedBlob {
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
                sink.out.blobs.push(ExtractedBlob {
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
#[derive(Default)]
pub struct Listing {
    files_array: bool,
}

static LISTING_FIELDS: Fields = Fields::new(&[
    ("id", Rule::Read(T::String)),
    ("name", Rule::Read(T::String)),
    ("createdTime", Rule::Read(T::String)),
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
])
.for_kind("listing");

impl ExportParser for Listing {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        let (metadata, stats) = crate::export::scan_nested(reader, &["files"], |range, raw| {
            item.push(range, Vec::new(), None, raw.get().as_bytes())
        })?;
        self.files_array = metadata.get("files").is_some_and(Json::is_array);
        Ok(stats)
    }
    fn begin(
        &mut self,
        length: u64,
        _first: Option<&[u8]>,
    ) -> Result<Option<crate::parse::Step>, String> {
        if self.files_array {
            return Ok(None);
        }
        let mut records = Records::default();
        records.record("listing", &RecordOutcome::unreadable("no files array"));
        Ok(Some(crate::parse::Step {
            output: Output::default(),
            counts: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            repeats_whole_conversations: false,
        }))
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let f: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let f = &f;
        sink.records.fields("file", f, &LISTING_FIELDS);
        let Some(sid) = str_of(f, "id").and_then(|id| session_id(ORIGIN, id)) else {
            sink.records
                .record("file", &RecordOutcome::unreadable("no id"));
            return Ok(());
        };
        let mut session = Session::new(sid, SessionKind::Main);
        if let Some(name) = str_of(f, "name") {
            session.offer_title(name, TitleSource::Generated);
        }
        if let Some(at) = f
            .get("createdTime")
            .and_then(time_of)
            .map(crate::time::SourceTime::file_mtime)
        {
            session
                .offer_start(at.timestamp(), at.evidence())
                .expect("decoded source time is valid");
        }
        sink.out.sessions.push(session);
        sink.records.record("file", &RecordOutcome::Used);
        Ok(())
    }
}

pub mod drive;

#[cfg(test)]
mod tests;
