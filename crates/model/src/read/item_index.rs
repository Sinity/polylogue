//! The session item index: for each physical session, an ordered list of the
//! layout items a transcript can show, each with its token estimates in the
//! collapsed and the full form. The writer keeps it in the same transaction
//! as the rows it describes; reads page it with running sums and a binary
//! search, and read bodies only for the entries of one page.
use super::{EntryKey, Expansion, ReadOptions, View, estimate_tokens};
use crate::{Block, Message, PromptKind, ToolAction, WrittenBy};
use serde::{Deserialize, Serialize};

mod counts;
pub use counts::*;

/// What an entry stands for. One message gives a header entry and one entry
/// per stored block; an event gives one entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    Header,
    Text,
    Thinking,
    /// Reasoning kept only as a provider signature: never shown.
    ThinkingEncrypted,
    Call,
    Result,
    Attachment,
    Other,
    Event,
}
impl EntryKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Header => "header",
            Self::Text => "text",
            Self::Thinking => "thinking",
            Self::ThinkingEncrypted => "thinking_encrypted",
            Self::Call => "call",
            Self::Result => "result",
            Self::Attachment => "attachment",
            Self::Other => "other",
            Self::Event => "event",
        }
    }
    pub fn parse(name: &str) -> Option<Self> {
        [
            Self::Header,
            Self::Text,
            Self::Thinking,
            Self::ThinkingEncrypted,
            Self::Call,
            Self::Result,
            Self::Attachment,
            Self::Other,
            Self::Event,
        ]
        .into_iter()
        .find(|kind| kind.name() == name)
    }
}

/// The context facts an in-place event carries for composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    /// Everything before is replaced by the summary that follows.
    Compaction,
    /// The header of the newest compaction's harness summary message.
    CompactionSummary,
    /// Named tool results are removed from the context.
    Microcompaction,
}
impl Boundary {
    pub fn of_event(kind: &str) -> Option<Self> {
        match kind {
            "compaction" => Some(Self::Compaction),
            "microcompaction" => Some(Self::Microcompaction),
            _ => None,
        }
    }
}

/// One index entry. Everything here is known without decoding a body again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexEntry {
    pub key: EntryKey,
    pub kind: EntryKind,
    pub written_by: WrittenBy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_kind: Option<PromptKind>,
    /// On the active path: yes, no, or unknown (no marked leaf).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_path: Option<bool>,
    /// A message with more than one stored reply.
    #[serde(default)]
    pub branch_point: bool,
    /// Milliseconds since the epoch, when the source gave a time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<i64>,
    pub body_bytes: u64,
    /// Estimated tokens of the one-line form.
    pub tokens_line: u64,
    /// Estimated tokens of the full form.
    pub tokens_full: u64,
    /// Lines of a tool result's output, which the head-and-tail fold cuts.
    #[serde(default)]
    pub output_lines: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<ToolAction>,
    /// The call's target, cut to 120 characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// The provider call ID of a call or result, for pairing on a page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// An event's kind, as stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary: Option<Boundary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction: Option<crate::CompactionBoundary>,
    /// The provider-reported input of the response that produced a model
    /// message: plain input plus cache reads plus cache writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u64>,
    /// For a microcompaction: the call IDs whose results it removed from
    /// the context.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cleared: Vec<String>,
}
impl IndexEntry {
    pub fn is_header(&self) -> bool {
        self.kind == EntryKind::Header
    }
}

/// Facts about a message that come from its row rather than its body.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MessagePlace {
    pub active_path: Option<bool>,
    pub branch_point: bool,
}

const TARGET_CHARACTERS: usize = 120;

/// The entries of one stored message, header first, blocks in their original
/// order. `ordinals` are the stored blocks' original positions.
pub fn message_entries(
    message: &Message,
    ordinals: &[u64],
    place: MessagePlace,
) -> Vec<IndexEntry> {
    let written_by = WrittenBy::of(message);
    let at = message.at.map(|at| at.unix_ms());
    let base = |key: EntryKey, kind: EntryKind| IndexEntry {
        key,
        kind,
        written_by,
        prompt_kind: message.prompt_kind,
        active_path: place.active_path,
        branch_point: place.branch_point,
        at,
        body_bytes: 0,
        tokens_line: 0,
        tokens_full: 0,
        output_lines: 0,
        action: None,
        target: None,
        call_id: None,
        event_kind: None,
        boundary: None,
        compaction: None,
        context_tokens: None,
        cleared: Vec::new(),
    };
    let header_line = header_tokens(message);
    let mut header = base(EntryKey::header(&message.id), EntryKind::Header);
    header.tokens_line = header_line;
    header.tokens_full = header_line;
    if message.prompt_kind == Some(crate::PromptKind::CompactionSummary) {
        header.boundary = Some(Boundary::CompactionSummary);
    }
    if written_by == WrittenBy::Model {
        let usage = &message.usage;
        if usage.input.is_some() || usage.cached_input.is_some() || usage.cache_write.is_some() {
            header.context_tokens = Some(
                usage.input.unwrap_or(0)
                    + usage.cached_input.unwrap_or(0)
                    + usage.cache_write.unwrap_or(0),
            );
        }
    }
    let mut entries = vec![header];
    for (ordinal, block) in ordinals.iter().zip(&message.blocks) {
        let key = EntryKey::Message {
            message: message.id.clone(),
            block: Some(*ordinal),
        };
        let mut entry = base(key, EntryKind::Other);
        match block {
            Block::Text { text } => {
                entry.kind = EntryKind::Text;
                entry.body_bytes = text.len() as u64;
                entry.tokens_full = estimate(text);
                entry.tokens_line = entry.tokens_full;
            }
            Block::Thinking {
                text,
                signature,
                redacted,
            } => {
                entry.kind = if text.is_none() && signature.is_some() && !redacted {
                    EntryKind::ThinkingEncrypted
                } else {
                    EntryKind::Thinking
                };
                let text = text.as_deref().unwrap_or("");
                entry.body_bytes = text.len() as u64;
                entry.tokens_line = THINKING_LINE;
                entry.tokens_full = estimate(text).max(THINKING_LINE);
            }
            Block::ToolCall {
                call_id,
                tool,
                action,
                input,
                command,
                path,
                ..
            } => {
                entry.kind = EntryKind::Call;
                let target = command
                    .clone()
                    .or_else(|| path.clone())
                    .unwrap_or_else(|| first_string(input));
                let target: String = target.chars().take(TARGET_CHARACTERS).collect();
                let input_text = input.to_string();
                entry.body_bytes = input_text.len() as u64;
                entry.tokens_line =
                    estimate(tool.as_deref().unwrap_or("tool")) + estimate(&target) + CALL_LINE;
                entry.tokens_full = entry.tokens_line + estimate(&input_text);
                entry.action = Some(*action);
                entry.target = Some(target);
                entry.call_id.clone_from(call_id);
            }
            Block::ToolResult {
                call_id, output, ..
            } => {
                entry.kind = EntryKind::Result;
                let output = output.as_deref().unwrap_or("");
                entry.body_bytes = output.len() as u64;
                entry.output_lines = output.lines().count() as u64;
                entry.tokens_line = 0;
                entry.tokens_full = estimate(output);
                entry.call_id.clone_from(call_id);
            }
            Block::Image(attachment) | Block::File(attachment) => {
                entry.kind = EntryKind::Attachment;
                entry.body_bytes = attachment.size.unwrap_or(0);
                entry.tokens_line =
                    estimate(attachment.name.as_deref().unwrap_or("attachment")) + CALL_LINE;
                entry.tokens_full = entry.tokens_line;
            }
            Block::Other { type_name, data } => {
                let text = data.to_string();
                entry.kind = EntryKind::Other;
                entry.body_bytes = text.len() as u64;
                entry.tokens_line = estimate(type_name) + CALL_LINE;
                entry.tokens_full = entry.tokens_line + estimate(&text);
            }
        }
        entries.push(entry);
    }
    entries
}

/// The entry of one in-place event. `cleared` is what a microcompaction
/// removed; empty for any other event.
pub fn event_entry(
    id: &str,
    kind: &str,
    at: Option<i64>,
    data_bytes: u64,
    cleared: Vec<String>,
) -> IndexEntry {
    let boundary = Boundary::of_event(kind);
    // A boundary reads as one line; any other event is shown with its data.
    let tokens = if boundary.is_some() {
        EVENT_LINE
    } else {
        EVENT_LINE.max(data_bytes.div_ceil(4))
    };
    IndexEntry {
        key: EntryKey::Event { event: id.into() },
        kind: EntryKind::Event,
        written_by: WrittenBy::Harness,
        prompt_kind: None,
        active_path: None,
        branch_point: false,
        at,
        body_bytes: data_bytes,
        tokens_line: tokens,
        tokens_full: tokens,
        output_lines: 0,
        action: None,
        target: None,
        call_id: None,
        event_kind: Some(kind.into()),
        boundary,
        compaction: None,
        context_tokens: None,
        cleared,
    }
}

const THINKING_LINE: u64 = 4;
const CALL_LINE: u64 = 4;
const EVENT_LINE: u64 = 8;
const HARNESS_LINE: u64 = 24;

fn estimate(text: &str) -> u64 {
    estimate_tokens(text) as u64
}
fn header_tokens(message: &Message) -> u64 {
    // Time, author, model and the message reference.
    8 + message.model.as_deref().map_or(0, estimate) + estimate(&message.id.to_string())
}
fn first_string(input: &serde_json::Value) -> String {
    match input {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Object(fields) => fields
            .values()
            .find_map(|value| value.as_str().map(str::to_owned))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// How each entry kind is shown in one reading: derived from the view and
/// the expansion options, and the one rule the pager, the index sums and
/// layout agree on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldProfile {
    pub view: View,
    pub tools: Expansion,
    pub thinking: Expansion,
    pub harness: Expansion,
    pub branches: bool,
    pub output_lines: u64,
}
impl FoldProfile {
    pub fn of(options: &ReadOptions) -> Self {
        Self {
            view: options.view,
            tools: options.tools(),
            thinking: options.thinking(),
            harness: options.harness(),
            branches: options.branches,
            output_lines: options.output_lines,
        }
    }

    /// Whether the message an entry belongs to is shown at all.
    fn keeps_message(&self, entry: &IndexEntry) -> bool {
        let harness = entry.written_by == WrittenBy::Harness;
        let kept = match self.view {
            View::Transcript | View::Context | View::Raw | View::Changes => true,
            View::Dialogue => {
                !harness && entry.written_by != WrittenBy::Tool && entry.kind != EntryKind::Event
            }
            View::Summary | View::Links | View::Live => false,
        };
        if !kept || (entry.kind == EntryKind::Event && self.view == View::Changes) {
            return false;
        }
        if entry.active_path == Some(false) && !self.branches {
            return false;
        }
        !(harness && entry.kind != EntryKind::Event && self.harness == Expansion::None)
    }

    /// The estimated tokens of an entry in this reading; `None` when the
    /// reading hides it. A header's cost counts only when one of its blocks
    /// is shown; [`visible_entries`] applies that rule.
    pub fn cost(&self, entry: &IndexEntry) -> Option<u64> {
        if !self.keeps_message(entry) {
            return None;
        }
        let full = |level: Expansion| level == Expansion::Full;
        // The changes view reads what changed files and what ran.
        if self.view == View::Changes {
            return match entry.kind {
                EntryKind::Header => Some(entry.tokens_line),
                // Laid out as the call's line, with its edit when it has one.
                EntryKind::Call
                    if matches!(
                        entry.action,
                        Some(ToolAction::FileEdit | ToolAction::FileWrite | ToolAction::Shell)
                    ) =>
                {
                    Some(entry.tokens_line)
                }
                _ => None,
            };
        }
        // A raw reading shows each message's stored record once, at its
        // header; its blocks count toward the page by their size.
        if self.view == View::Raw {
            return Some(match entry.kind {
                EntryKind::Header | EntryKind::Event => entry.tokens_line,
                _ => entry.tokens_full,
            });
        }
        // A collapsed harness message is its header and one line of its text.
        if entry.written_by == WrittenBy::Harness
            && entry.kind != EntryKind::Event
            && self.harness == Expansion::Collapsed
        {
            return Some(match entry.kind {
                EntryKind::Header => entry.tokens_line,
                EntryKind::Text => entry.tokens_full.min(HARNESS_LINE),
                _ => 0,
            });
        }
        Some(match entry.kind {
            EntryKind::Header => entry.tokens_line,
            EntryKind::Event => entry.tokens_line,
            EntryKind::Text => entry.tokens_full,
            EntryKind::ThinkingEncrypted => return None,
            EntryKind::Thinking => {
                if self.view == View::Dialogue || self.thinking == Expansion::None {
                    return None;
                }
                if full(self.thinking) {
                    entry.tokens_full
                } else {
                    entry.tokens_line
                }
            }
            EntryKind::Call => {
                if !self.shows_calls() {
                    return None;
                }
                if full(self.tools) {
                    entry.tokens_full
                } else {
                    entry.tokens_line
                }
            }
            EntryKind::Result => {
                if !self.shows_calls() {
                    return None;
                }
                if full(self.tools) {
                    self.output_cost(entry)
                } else {
                    0
                }
            }
            EntryKind::Attachment | EntryKind::Other => {
                if self.view == View::Dialogue || self.view == View::Changes {
                    return None;
                }
                entry.tokens_line
            }
        })
    }

    fn shows_calls(&self) -> bool {
        self.tools != Expansion::None
            && self
                .view
                .block_kinds()
                .contains(&crate::BlockKind::ToolCall)
    }

    /// Output shown as its first and last `output_lines` lines.
    fn output_cost(&self, entry: &IndexEntry) -> u64 {
        let shown = self.output_lines.saturating_mul(2);
        if self.output_lines == 0 || entry.output_lines <= shown || entry.output_lines == 0 {
            return entry.tokens_full;
        }
        entry
            .tokens_full
            .saturating_mul(shown)
            .div_ceil(entry.output_lines)
            .saturating_add(CALL_LINE)
    }
}

/// One shown entry: its place in the composed sequence and its cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shown {
    pub segment: usize,
    pub entry: usize,
    pub cost: u64,
}

/// The shown entries of a composed reading, in order, with running sums.
/// `sums[i]` is the total cost of the first `i` shown entries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunningSums {
    pub shown: Vec<Shown>,
    pub sums: Vec<u64>,
}
impl RunningSums {
    pub fn total(&self) -> u64 {
        self.sums.last().copied().unwrap_or(0)
    }
    pub fn len(&self) -> usize {
        self.shown.len()
    }
    pub fn is_empty(&self) -> bool {
        self.shown.is_empty()
    }
}

/// A run of one session's entries inside a composed reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment<'a> {
    pub entries: &'a [IndexEntry],
    pub inherited: bool,
}

/// The shown entries of a composition under one profile. A header counts
/// only when the profile shows at least one of the message's blocks, or the
/// view shows headers without blocks (transcript and context).
pub fn visible_entries(segments: &[Segment<'_>], profile: &FoldProfile) -> RunningSums {
    let mut out = RunningSums {
        shown: Vec::new(),
        sums: vec![0],
    };
    let headers_alone = matches!(profile.view, View::Transcript | View::Context | View::Raw);
    for (segment_at, segment) in segments.iter().enumerate() {
        let entries = segment.entries;
        let mut at = 0;
        while at < entries.len() {
            let entry = &entries[at];
            if entry.kind != EntryKind::Header {
                if let Some(cost) = profile.cost(entry) {
                    push(&mut out, segment_at, at, cost);
                }
                at += 1;
                continue;
            }
            let end = entries[at + 1..]
                .iter()
                .position(|next| next.kind == EntryKind::Header || next.kind == EntryKind::Event)
                .map_or(entries.len(), |offset| at + 1 + offset);
            if let Some(header_cost) = profile.cost(entry) {
                let blocks: Vec<(usize, u64)> = (at + 1..end)
                    .filter_map(|index| profile.cost(&entries[index]).map(|cost| (index, cost)))
                    .collect();
                if headers_alone || !blocks.is_empty() {
                    push(&mut out, segment_at, at, header_cost);
                    for (index, cost) in blocks {
                        push(&mut out, segment_at, index, cost);
                    }
                }
            }
            at = end;
        }
    }
    out
}
fn push(out: &mut RunningSums, segment: usize, entry: usize, cost: u64) {
    let total = out.total().saturating_add(cost);
    out.shown.push(Shown {
        segment,
        entry,
        cost,
    });
    out.sums.push(total);
}

#[cfg(test)]
mod tests;
