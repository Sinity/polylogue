//! The stored form of an index entry (`session_item.entry`): a compact
//! binary record without what the row and its unit already hold. A unit's
//! first row names its message or event through `message_pk` or
//! `event_pk`, so its entry leaves out that ID. The later entries of a
//! message are its blocks; they share the header's message, author, prompt
//! kind, placement and time, which only the header's row stores. The call
//! ID has its own column. Reading puts each of these back, so a decoded
//! entry equals the one written.
//!
//! Layout: a flags number, a byte with the kind (and on a unit's first
//! entry the author), then the fields the flags name, in the order of the
//! flags. Numbers are unsigned LEB128; signed ones are zigzag-mapped first.
//! Text is a length and its UTF-8. A compaction's facts are JSON, since
//! they carry the provider's replacement history as it was recorded.
use crate::{Result, StoreError};
use polylogue_model::read::{Boundary, EntryKey, EntryKind, IndexEntry};
use polylogue_model::{MessageId, PromptKind, ToolAction, WrittenBy};

// The kind byte holds the entry kind in its low nibble and, on a unit's
// first entry, the author in its high nibble.
const KIND_BITS: u32 = 4;
const KIND_MASK: u8 = 0x0F;
const BLOCK: u32 = 1;
const OUTPUT_LINES: u32 = 1 << 1;
const ACTION: u32 = 1 << 2;
const TARGET: u32 = 1 << 3;
const EVENT_KIND: u32 = 1 << 4;
const BOUNDARY: u32 = 1 << 5;
const COMPACTION: u32 = 1 << 6;
const CONTEXT_TOKENS: u32 = 1 << 7;
const CLEARED: u32 = 1 << 8;
// Stored on a unit's first entry only.
const PROMPT_KIND: u32 = 1 << 9;
const ACTIVE_PATH_KNOWN: u32 = 1 << 10;
const ON_ACTIVE_PATH: u32 = 1 << 11;
const BRANCH_POINT: u32 = 1 << 12;
const AT: u32 = 1 << 13;

/// Whose entry a stored row holds, from the row and its unit.
pub(crate) enum EntryOf<'a> {
    /// A message's first entry: its row names the message.
    Message(MessageId),
    /// An event's entry: its row names the event.
    Event(String),
    /// A later entry of the message whose header this is.
    Header(&'a IndexEntry),
}

macro_rules! codes {
    ($encode:ident, $decode:ident, $ty:ty { $($variant:path => $code:literal),+ $(,)? }) => {
        fn $encode(value: $ty) -> u8 {
            match value {
                $($variant => $code),+
            }
        }
        fn $decode(code: u8) -> Result<$ty> {
            match code {
                $($code => Ok($variant),)+
                _ => Err(malformed()),
            }
        }
    };
}

codes!(kind_code, kind_of, EntryKind {
    EntryKind::Header => 0,
    EntryKind::Text => 1,
    EntryKind::Thinking => 2,
    EntryKind::ThinkingEncrypted => 3,
    EntryKind::Call => 4,
    EntryKind::Result => 5,
    EntryKind::Attachment => 6,
    EntryKind::Other => 7,
    EntryKind::Event => 8,
});
codes!(author_code, author_of, WrittenBy {
    WrittenBy::Human => 0,
    WrittenBy::Model => 1,
    WrittenBy::Harness => 2,
    WrittenBy::Tool => 3,
    WrittenBy::Unknown => 4,
});
codes!(prompt_code, prompt_of, PromptKind {
    PromptKind::Prose => 0,
    PromptKind::Command => 1,
    PromptKind::Harness => 2,
    PromptKind::InjectedContext => 3,
    PromptKind::CompactionSummary => 4,
    PromptKind::Model => 5,
});
codes!(action_code, action_of, ToolAction {
    ToolAction::Shell => 0,
    ToolAction::Code => 1,
    ToolAction::FileRead => 2,
    ToolAction::FileWrite => 3,
    ToolAction::FileEdit => 4,
    ToolAction::Search => 5,
    ToolAction::Web => 6,
    ToolAction::Subagent => 7,
    ToolAction::Mcp => 8,
    ToolAction::Other => 9,
});
codes!(boundary_code, boundary_of, Boundary {
    Boundary::Compaction => 0,
    Boundary::CompactionSummary => 1,
    Boundary::Microcompaction => 2,
});

fn malformed() -> StoreError {
    StoreError::stored_data("malformed session item entry")
}

/// The stored bytes of `entry`. `header` is the message header when
/// `entry` is one of the message's later entries; those must share the
/// header's message, author, prompt kind, placement and time, which only
/// the header stores.
pub(crate) fn encode(entry: &IndexEntry, header: Option<&IndexEntry>) -> Result<Vec<u8>> {
    let block = match (&entry.key, header) {
        (EntryKey::Message { block, .. }, None) => *block,
        (EntryKey::Event { .. }, None) => None,
        (EntryKey::Message { message, block }, Some(header)) => {
            let shares = header.key.message() == Some(message)
                && header.written_by == entry.written_by
                && header.prompt_kind == entry.prompt_kind
                && header.active_path == entry.active_path
                && header.branch_point == entry.branch_point
                && header.at == entry.at;
            if !shares {
                return Err(StoreError::stored_data(format!(
                    "entry {} differs from its message header in what the header stores",
                    entry.key.reference()
                )));
            }
            *block
        }
        (EntryKey::Event { .. }, Some(_)) => {
            return Err(StoreError::stored_data(
                "an event entry is never a later entry of a message",
            ));
        }
    };
    let first = header.is_none();
    let mut flags = 0;
    let mut set = |bit: u32, on: bool| {
        if on {
            flags |= bit;
        }
    };
    set(BLOCK, block.is_some());
    set(OUTPUT_LINES, entry.output_lines != 0);
    set(ACTION, entry.action.is_some());
    set(TARGET, entry.target.is_some());
    set(EVENT_KIND, entry.event_kind.is_some());
    set(BOUNDARY, entry.boundary.is_some());
    set(COMPACTION, entry.compaction.is_some());
    set(CONTEXT_TOKENS, entry.context_tokens.is_some());
    set(CLEARED, !entry.cleared.is_empty());
    if first {
        set(PROMPT_KIND, entry.prompt_kind.is_some());
        set(ACTIVE_PATH_KNOWN, entry.active_path.is_some());
        set(ON_ACTIVE_PATH, entry.active_path == Some(true));
        set(BRANCH_POINT, entry.branch_point);
        set(AT, entry.at.is_some());
    }
    let mut out = Vec::with_capacity(16 + entry.target.as_ref().map_or(0, String::len));
    number(&mut out, u64::from(flags));
    let author = if first {
        author_code(entry.written_by) << KIND_BITS
    } else {
        0
    };
    out.push(kind_code(entry.kind) | author);
    if first {
        if let Some(prompt) = entry.prompt_kind {
            out.push(prompt_code(prompt));
        }
        if let Some(at) = entry.at {
            number(&mut out, zigzag(at));
        }
    }
    if let Some(block) = block {
        number(&mut out, block);
    }
    number(&mut out, entry.body_bytes);
    number(&mut out, entry.tokens_line);
    // The full form is rarely smaller than the line; wrapping keeps every
    // pair of values exact.
    let extra = entry.tokens_full.wrapping_sub(entry.tokens_line) as i64;
    number(&mut out, zigzag(extra));
    if entry.output_lines != 0 {
        number(&mut out, entry.output_lines);
    }
    if let Some(action) = entry.action {
        out.push(action_code(action));
    }
    if let Some(target) = &entry.target {
        text(&mut out, target.as_bytes());
    }
    if let Some(kind) = &entry.event_kind {
        text(&mut out, kind.as_bytes());
    }
    if let Some(boundary) = entry.boundary {
        out.push(boundary_code(boundary));
    }
    if let Some(compaction) = &entry.compaction {
        text(&mut out, &serde_json::to_vec(compaction)?);
    }
    if let Some(tokens) = entry.context_tokens {
        number(&mut out, tokens);
    }
    if !entry.cleared.is_empty() {
        number(&mut out, entry.cleared.len() as u64);
        for call in &entry.cleared {
            text(&mut out, call.as_bytes());
        }
    }
    Ok(out)
}

/// The entry stored as `bytes` in a row of `entry_of`, with the row's call ID.
pub(crate) fn decode(
    bytes: &[u8],
    entry_of: EntryOf<'_>,
    call_id: Option<String>,
) -> Result<IndexEntry> {
    let mut input = Input { bytes, at: 0 };
    let flags = u32::try_from(input.number()?).map_err(|_| malformed())?;
    let has = |bit: u32| flags & bit != 0;
    let kind_byte = input.byte()?;
    let kind = kind_of(kind_byte & KIND_MASK)?;
    let (key_message, written_by, prompt_kind, active_path, branch_point, at) = match entry_of {
        EntryOf::Header(header) => {
            let Some(message) = header.key.message() else {
                return Err(malformed());
            };
            (
                Some(message.clone()),
                header.written_by,
                header.prompt_kind,
                header.active_path,
                header.branch_point,
                header.at,
            )
        }
        EntryOf::Message(_) | EntryOf::Event(_) => {
            let written_by = author_of(kind_byte >> KIND_BITS)?;
            let prompt_kind = if has(PROMPT_KIND) {
                Some(prompt_of(input.byte()?)?)
            } else {
                None
            };
            let at = if has(AT) {
                let value = input.number()?;
                Some(unzigzag(value))
            } else {
                None
            };
            let message = match &entry_of {
                EntryOf::Message(message) => Some(message.clone()),
                _ => None,
            };
            (
                message,
                written_by,
                prompt_kind,
                has(ACTIVE_PATH_KNOWN).then_some(has(ON_ACTIVE_PATH)),
                has(BRANCH_POINT),
                at,
            )
        }
    };
    let block = if has(BLOCK) {
        Some(input.number()?)
    } else {
        None
    };
    let key = match (entry_of, key_message) {
        (EntryOf::Event(event), _) => {
            if block.is_some() {
                return Err(malformed());
            }
            EntryKey::Event { event }
        }
        (_, Some(message)) => EntryKey::Message { message, block },
        (_, None) => return Err(malformed()),
    };
    let body_bytes = input.number()?;
    let tokens_line = input.number()?;
    let extra = input.number()?;
    let extra = unzigzag(extra);
    let tokens_full = tokens_line.wrapping_add(extra as u64);
    let output_lines = if has(OUTPUT_LINES) {
        input.number()?
    } else {
        0
    };
    let action = if has(ACTION) {
        Some(action_of(input.byte()?)?)
    } else {
        None
    };
    let target = if has(TARGET) {
        Some(input.string()?)
    } else {
        None
    };
    let event_kind = if has(EVENT_KIND) {
        Some(input.string()?)
    } else {
        None
    };
    let boundary = if has(BOUNDARY) {
        Some(boundary_of(input.byte()?)?)
    } else {
        None
    };
    let compaction = if has(COMPACTION) {
        Some(serde_json::from_slice(input.text()?)?)
    } else {
        None
    };
    let context_tokens = if has(CONTEXT_TOKENS) {
        Some(input.number()?)
    } else {
        None
    };
    let mut cleared = Vec::new();
    if has(CLEARED) {
        let count = input.number()?;
        for _ in 0..count {
            cleared.push(input.string()?);
        }
    }
    if input.at != bytes.len() {
        return Err(malformed());
    }
    Ok(IndexEntry {
        key,
        kind,
        written_by,
        prompt_kind,
        active_path,
        branch_point,
        at,
        body_bytes,
        tokens_line,
        tokens_full,
        output_lines,
        action,
        target,
        call_id,
        event_kind,
        boundary,
        compaction,
        context_tokens,
        cleared,
    })
}

/// Appends `value` as unsigned LEB128.
pub(super) fn number(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Maps a signed number to an unsigned one that is small when the signed one is.
fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

fn text(out: &mut Vec<u8>, bytes: &[u8]) {
    number(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// A cursor over stored bytes.
pub(super) struct Input<'a> {
    pub bytes: &'a [u8],
    pub at: usize,
}

impl<'a> Input<'a> {
    pub fn done(&self) -> bool {
        self.at >= self.bytes.len()
    }

    /// An unsigned LEB128 number; `None` when the bytes end or overflow.
    pub fn leb128(&mut self) -> Option<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self.bytes.get(self.at)?;
            self.at += 1;
            value |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    }

    fn number(&mut self) -> Result<u64> {
        self.leb128().ok_or_else(malformed)
    }

    fn byte(&mut self) -> Result<u8> {
        let byte = *self.bytes.get(self.at).ok_or_else(malformed)?;
        self.at += 1;
        Ok(byte)
    }

    fn text(&mut self) -> Result<&'a [u8]> {
        let length = usize::try_from(self.number()?).map_err(|_| malformed())?;
        let end = self.at.checked_add(length).ok_or_else(malformed)?;
        let text = self.bytes.get(self.at..end).ok_or_else(malformed)?;
        self.at = end;
        Ok(text)
    }

    fn string(&mut self) -> Result<String> {
        String::from_utf8(self.text()?.to_vec()).map_err(|_| malformed())
    }
}

#[cfg(test)]
mod tests;
