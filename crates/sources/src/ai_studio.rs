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

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 6;

parser_modules! {}

use crate::document::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, time_of};
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::parse::{ExtractedBlob, MessageRow, Output, Parse};
use crate::{RecordOutcome, Records};
use base64::Engine as _;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, ErrorReason, Event, IdKind, Json, Link,
    LinkKind, Message, PromptKind, Role, Session, SessionKind, TitleSource, ToolAction,
    ToolOutcome, UnknownReason, Usage,
};
use std::path::Path;

pub const ORIGIN: &str = "ai-studio";

/// Where the Drive collector writes prompts it fetched, one file per
/// prompt named by its Drive file ID.
pub const LOCATIONS: &[&str] = &["polylogue/drive/ai-studio/*.json"];

/// The Drive collector's listing of the prompts folder, `files.json`:
/// `{"files": [{"id", "name", "createdTime", …}]}`, each entry the file's
/// metadata as Drive last gave it, sorted by ID. It names the prompts: each
/// entry offers its name as the title of the session `ai-studio:<id>`. An
/// entry marked `"removed": true` is a prompt no longer in the folder, whose
/// session is gone.
pub const LISTING: &str = "files.json";

fn is_listing(path: &Path) -> bool {
    path.file_name().is_some_and(|n| n == LISTING)
}

/// A prompt document starts with one of its three top-level keys.
///
/// Read whole, a prompt is the object holding a `chunkedPrompt`.
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    use crate::document::Kind;
    if head.shape.is_some()
        && !is_listing(path)
        && head.top() == Some(Kind::Object)
        && head.holds("chunkedPrompt", Kind::Object)
    {
        return true;
    }
    let head = head.bytes;
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
    let name = name_of(path)?;
    // Only the collector's spool names files by Drive ID; a download in some
    // other folder called `ai-studio` is named by its title.
    let in_spool = path
        .parent()
        .is_some_and(|p| p.ends_with("polylogue/drive/ai-studio"));
    Some(Parse::Export(Box::new(if in_spool {
        AiStudio {
            drive_id: Some(name),
            native: String::new(),
            content_key: String::new(),
            title: None,
            state: PromptState::default(),
        }
    } else {
        AiStudio {
            drive_id: None,
            native: String::new(),
            content_key: String::new(),
            title: Some(name),
            state: PromptState::default(),
        }
    })))
}

/// The file's name without `.json`: the Drive file ID in the collector's
/// spool, the prompt's title in a Drive download or Takeout.
fn name_of(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    // A zip member path ends in its own name after `!/`.
    let name = name.rsplit("!/").next()?;
    let stem = name.strip_suffix(".json").unwrap_or(name);
    (!stem.is_empty()).then(|| stem.to_owned())
}

/// The prompt's identity. The Drive collector's file is named by the Drive
/// file ID, which the listing and `branchParent` also use. A download or
/// Takeout carries no file ID and names the file by the prompt's title,
/// which two prompts can share and a rename changes, so its prompt is named
/// by its content: the first chunk (which holds its creation time), or the
/// whole document when it has no chunks. A Drive revision records the
/// content key its copies would have as an alias of its session; the index
/// shows such a copy as that session (docs/capture.md, AI Studio through
/// Drive).
fn content_key(first_chunk: Option<&Json>, metadata: &Json) -> String {
    let hash = match first_chunk {
        Some(chunk) => canonical_json::canonical_hash("polylogue-ai-studio-first-chunk", chunk),
        None => canonical_json::canonical_hash("polylogue-ai-studio-document", metadata),
    }
    .expect("a JSON value has a canonical form");
    format!("content-{}", &hash.to_hex().as_str()[..32])
}

pub struct AiStudio {
    /// The Drive file ID, for a prompt the collector fetched.
    drive_id: Option<String>,
    /// The session's native ID, set when the document is scanned.
    native: String,
    /// The content key of this reading: a copy's own name, and an alias of
    /// a Drive prompt's session.
    content_key: String,
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
    parents: std::collections::BTreeSet<polylogue_model::SessionId>,
    /// Chunks without an `id` named so far: tells their written names
    /// apart ([`crate::content_ids`]).
    named: u64,
}

static DOCUMENT_FIELDS: Fields = Fields::new(
    "document",
    &[
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
    ],
);

static CHUNK_FIELDS: Fields = Fields::new(
    "chunk",
    &[
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
            Rule::ReadWhole("complete code input in tool block"),
        ),
        (
            "parts[].codeExecutionResult",
            Rule::ReadWhole("complete code result in tool block"),
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
            Rule::Kept("AI Studio branch marker in message attributes"),
        ),
        ("branchParent.promptId", Rule::Read(T::String)),
        (
            "branchChildren",
            Rule::Kept("other prompts branched from here"),
        ),
    ],
);

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
        let mut first_chunk = None;
        let (metadata, stats) =
            crate::export::scan_nested(reader, &["chunkedPrompt", "chunks"], |range, raw| {
                if first_chunk.is_none() {
                    first_chunk = serde_json::from_str::<Json>(raw.get()).ok();
                }
                item.push(range, Vec::new(), None, raw.get().as_bytes())
            })?;
        self.content_key = content_key(first_chunk.as_ref(), &metadata);
        self.native = match &self.drive_id {
            Some(id) => id.clone(),
            None => self.content_key.clone(),
        };
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
                .and_then(|v| v.str("model").filter(|s| !s.is_empty()))
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
        self.state.position += 1;
        let kind = chunk_role(chunk).unwrap_or("?").to_owned();
        sink.records.fields(&kind, chunk, &CHUNK_FIELDS);
        if let Some(parent) = chunk
            .pointer("/branchParent/promptId")
            .and_then(Json::as_str)
            .and_then(|p| session_id(ORIGIN, p))
        {
            self.state.parents.insert(parent);
        }
        match self.message(&sid, chunk, sink) {
            Ok((mut message, id_kind)) => {
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
                    if self.drive_id.is_some() {
                        session.aliases =
                            session_id(ORIGIN, &self.content_key).into_iter().collect();
                    }
                    sink.out.sessions.push(session);
                    sink.out.events.push({
                        let (at, time_accuracy) = crate::time::parts(None);
                        Event {
                            session: Some(sid.clone()),
                            kind: "run_settings".into(),
                            at,
                            at_quality: time_accuracy.quality,
                            at_precision: time_accuracy.precision,
                            at_until: time_accuracy.until,
                            after: None,
                            range: Some((0, self.state.length)),
                            data: doc.get("runSettings").cloned().unwrap_or(Json::Null),
                        }
                    });
                    let mut attrs = crate::fields::attributes(&doc, &DOCUMENT_FIELDS);
                    attrs.remove("runSettings");
                    if !attrs.is_empty() {
                        sink.out.events.push({
                            let (at, time_accuracy) = crate::time::parts(None);
                            Event {
                                session: Some(sid.clone()),
                                kind: "prompt_metadata".into(),
                                at,
                                at_quality: time_accuracy.quality,
                                at_precision: time_accuracy.precision,
                                at_until: time_accuracy.until,
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
                            response: None,
                            page_text: false,
                            id_kind: polylogue_model::IdKind::Native,
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
                        .offer_start(at.timestamp(), at.accuracy())
                        .expect("decoded source time is valid");
                    sink.out.sessions.push(session);
                    self.state.started = true;
                }
                self.state.previous = Some(message.id.clone());
                let offset = sink.position();
                sink.out.messages.push(MessageRow {
                    response: None,
                    page_text: false,
                    id_kind,
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
        if self.state.position as u64 == self.state.chunks {
            for parent in std::mem::take(&mut self.state.parents) {
                sink.out.links.push(Link {
                    kind: LinkKind::Fork,
                    copies_prefix: false,
                    reason: None,
                    parent,
                    child: sid.clone(),
                    branch_point: None,
                    call_id: None,
                    evidence: "branchParent.promptId".into(),
                });
            }
        }
        if !self.state.emitted && self.state.position as u64 == self.state.chunks {
            sink.records
                .record("prompt", &RecordOutcome::skipped("no usable chunks"));
        }
        Ok(())
    }
}

fn chunk_role(chunk: &Json) -> Option<&str> {
    chunk
        .str("role")
        .filter(|s| !s.is_empty())
        .or_else(|| chunk.str("author").filter(|s| !s.is_empty()))
}

/// `systemInstruction` is `{}` when empty, `{"text": …}` or
/// `{"parts": [{"text": …}]}`.
fn system_text(value: Option<&Json>) -> Option<String> {
    let v = value?;
    if let Some(text) = v.as_str() {
        return Some(text.to_owned());
    }
    if let Some(t) = v.str("text").filter(|s| !s.is_empty()) {
        return Some(t.to_owned());
    }
    let joined: String = v
        .array("parts")?
        .iter()
        .filter_map(|p| p.str("text").filter(|s| !s.is_empty()))
        .collect();
    (!joined.is_empty()).then_some(joined)
}

impl AiStudio {
    fn message(
        &mut self,
        sid: &polylogue_model::SessionId,
        chunk: &Json,
        sink: &mut Sink<'_>,
    ) -> Result<(Message, IdKind), RecordOutcome> {
        let source_role = chunk_role(chunk);
        if source_role != Some("model") {
            self.state.pending_call = None;
            self.state.ambiguous_calls = false;
        }
        let role = match source_role {
            Some("user") => Role::User,
            Some("model") => Role::Assistant,
            Some(other) => return Err(RecordOutcome::unreadable(format!("unknown role {other}"))),
            None => return Err(RecordOutcome::unreadable("no role")),
        };
        // A chunk without an ID is named by its content, so deleting or
        // inserting a chunk leaves the others' IDs alone.
        let (id, id_kind) = match chunk.str("id").filter(|s| !s.is_empty()) {
            Some(id) => (
                message_id(ORIGIN, &message_key(&self.native, "id", id)),
                IdKind::Native,
            ),
            None => {
                self.state.named += 1;
                (
                    crate::content_ids::named(sid, role, chunk, self.state.named - 1),
                    IdKind::Content,
                )
            }
        };
        let id = id.ok_or_else(|| RecordOutcome::unreadable("unusable id"))?;
        let native = id.native().as_str().to_owned();
        let thought = chunk.bool("isThought") == Some(true);

        let mut blocks = text_blocks(chunk, thought);
        blocks.extend(media_blocks(chunk, role, sink));
        // Code execution and generated files sit on the chunk or in its
        // parts; each result answers the call before it.
        let parts = chunk.array("parts").map(Vec::as_slice).unwrap_or_default();
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
                    written_text: None,
                    run_by_script: false,
                    call_id: Some(call_id),
                    tool: Some("code_execution".into()),
                    action: ToolAction::Code,
                    input: code.clone(),
                    command: code
                        .str("code")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
                    path: None,
                });
            }
            if let Some(result) = item.get("codeExecutionResult") {
                // A result answers the preceding unanswered call. Zero or
                // several unanswered calls leave its call unknown.
                let call_id = self.state.pending_call.take();
                self.state.ambiguous_calls = false;
                let outcome = match result.str("outcome").filter(|s| !s.is_empty()) {
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
                    output: result
                        .str("output")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
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
                    media_type: file
                        .str("mimeType")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
                    url: file
                        .str("fileUri")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
                    direction: if role == Role::User {
                        AttachmentDirection::Input
                    } else {
                        AttachmentDirection::Output
                    },
                    ..Attachment::default()
                }));
            }
        }
        if let Some(error) = chunk.str("errorMessage").filter(|s| !s.is_empty()) {
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
        let (at, accuracy) = crate::time::parts(chunk.get("createTime").and_then(time_of));
        message
            .set_time(at, accuracy)
            .expect("decoded source time is valid");
        if role == Role::Assistant {
            message.model = self.state.model.clone();
            message.stop_reason = chunk
                .str("finishReason")
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            let tokens = chunk.u64("tokenCount");
            message.usage = Usage {
                output: tokens.filter(|_| !thought),
                reasoning: tokens.filter(|_| thought),
                ..Usage::default()
            };
        } else {
            message.prompt_kind = Some(PromptKind::Prose);
            message.usage = Usage {
                input: chunk.u64("tokenCount"),
                ..Usage::default()
            };
        }
        if role == Role::User {
            self.state.pending_call = None;
            self.state.ambiguous_calls = false;
        }
        Ok((message, id_kind))
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
        let (at, time_accuracy) = crate::time::parts(None);
        Message {
            id,
            session,
            parent: None,
            role,
            at,
            at_quality: time_accuracy.quality,
            at_precision: time_accuracy.precision,
            at_until: time_accuracy.until,
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
    if let Some(parts) = chunk.array("parts") {
        for part in parts {
            let Some(text) = part.str("text").filter(|s| !s.is_empty()) else {
                continue;
            };
            let is_thought = part.bool("thought") == Some(true);
            let signature = part
                .str("thoughtSignature")
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
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
    if let Some(text) = chunk.str("text").filter(|s| !s.is_empty()) {
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

/// Stores the base64 `data` of an inline part as a blob and names it on the
/// attachment. A value that is not base64 leaves the attachment without a blob.
fn store_inline_bytes(a: &mut Attachment, inline: &Json, sink: &mut Sink<'_>) {
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};
    let Some(data) = inline.str("data").filter(|s| !s.is_empty()) else {
        return;
    };
    // Clients wrap base64 across lines, drop the padding or use the URL-safe
    // alphabet; all of them state the same bytes.
    let data: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let Some(bytes) = STANDARD
        .decode(&data)
        .or_else(|_| STANDARD_NO_PAD.decode(&data))
        .or_else(|_| URL_SAFE.decode(&data))
        .or_else(|_| URL_SAFE_NO_PAD.decode(&data))
        .ok()
    else {
        return;
    };
    let hash = BlobHash::of(&bytes);
    a.size = Some(bytes.len() as u64);
    a.blob = Some(hash);
    sink.out.blobs.push(ExtractedBlob {
        hash,
        size: bytes.len() as u64,
        bytes,
    });
}

/// Inline bytes (`{mimeType, data}`), stored as a blob: an image when the
/// type says so, else a file.
fn inline_block(inline: &Json, role: Role, sink: &mut Sink<'_>) -> Block {
    let mut a = Attachment {
        media_type: inline
            .str("mimeType")
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        name: inline
            .str("name")
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        direction: if role == Role::User {
            AttachmentDirection::Input
        } else {
            AttachmentDirection::Output
        },
        ..Attachment::default()
    };
    store_inline_bytes(&mut a, inline, sink);
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
                name: doc
                    .str("name")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .or_else(|| {
                        doc.str("title")
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                    }),
                media_type: doc
                    .str("mimeType")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
                source_id: doc
                    .str("id")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .or_else(|| {
                        doc.str("fileId")
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                    }),
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
                media_type: inline
                    .str("mimeType")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
                name: inline
                    .str("name")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
                direction,
                ..Attachment::default()
            };
            store_inline_bytes(&mut a, inline, sink);
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

static LISTING_FIELDS: Fields = Fields::new(
    "listing",
    &[
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
        ("removed", Rule::Read(T::Boolean)),
    ],
);

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
        let Some(sid) = f
            .str("id")
            .filter(|s| !s.is_empty())
            .and_then(|id| session_id(ORIGIN, id))
        else {
            sink.records
                .record("file", &RecordOutcome::unreadable("no id"));
            return Ok(());
        };
        let mut session = Session::new(sid, SessionKind::Main);
        if let Some(name) = f.str("name").filter(|s| !s.is_empty()) {
            session.offer_title(name, TitleSource::Generated);
        }
        session.gone = f.bool("removed") == Some(true);
        if let Some(at) = f
            .get("createdTime")
            .and_then(time_of)
            .map(crate::time::SourceTime::file_mtime)
        {
            session
                .offer_start(at.timestamp(), at.accuracy())
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
