//! Running counts over a session's index entries, so a reading knows how
//! many entries it shows before a place without reading them. Whether a
//! reading shows an entry depends only on the entry's class: its kind, who
//! wrote its message, whether it is off the active path, and for a header
//! the kinds of block its message holds. The writer keeps, on each unit's
//! first entry, the counts per class of every entry before it and the
//! running sum of their full-form token estimates; a reading combines them
//! with [`FoldProfile::shows`].
use super::{EntryKind, FoldProfile, IndexEntry, View};
use crate::{PromptKind, ToolAction, WrittenBy};
use std::collections::BTreeMap;

/// What decides whether a reading shows an entry, packed in 15 bits: the
/// kind (4), the author (2), off the active path (1), and for a header the
/// kinds of block its message holds (8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntryClass(u16);

/// An entry's kind as its class sees it: calls split by whether they change
/// files or run commands, events by the context boundary they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ClassKind {
    Header = 0,
    Text = 1,
    Thinking = 2,
    ThinkingEncrypted = 3,
    ChangeCall = 4,
    Call = 5,
    Result = 6,
    Attachment = 7,
    Other = 8,
    Compaction = 9,
    CompactionSummary = 10,
    Microcompaction = 11,
    Event = 12,
}
const KINDS: [ClassKind; 13] = [
    ClassKind::Header,
    ClassKind::Text,
    ClassKind::Thinking,
    ClassKind::ThinkingEncrypted,
    ClassKind::ChangeCall,
    ClassKind::Call,
    ClassKind::Result,
    ClassKind::Attachment,
    ClassKind::Other,
    ClassKind::Compaction,
    ClassKind::CompactionSummary,
    ClassKind::Microcompaction,
    ClassKind::Event,
];
/// The kinds a message's blocks can have.
const BLOCK_KINDS: [ClassKind; 8] = [
    ClassKind::Text,
    ClassKind::Thinking,
    ClassKind::ThinkingEncrypted,
    ClassKind::ChangeCall,
    ClassKind::Call,
    ClassKind::Result,
    ClassKind::Attachment,
    ClassKind::Other,
];

/// Who wrote an entry's message, as far as showing it depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ClassAuthor {
    /// The model, a tool, or a person writing anything but a prompt.
    Other = 0,
    Harness = 1,
    /// A person's prompt: what opens a turn.
    Prompt = 2,
}

fn class_kind(entry: &IndexEntry) -> ClassKind {
    use super::Boundary;
    match entry.kind {
        EntryKind::Header => ClassKind::Header,
        EntryKind::Text => ClassKind::Text,
        EntryKind::Thinking => ClassKind::Thinking,
        EntryKind::ThinkingEncrypted => ClassKind::ThinkingEncrypted,
        EntryKind::Call
            if matches!(
                entry.action,
                Some(ToolAction::FileEdit | ToolAction::FileWrite | ToolAction::Shell)
            ) =>
        {
            ClassKind::ChangeCall
        }
        EntryKind::Call => ClassKind::Call,
        EntryKind::Result => ClassKind::Result,
        EntryKind::Attachment => ClassKind::Attachment,
        EntryKind::Other => ClassKind::Other,
        EntryKind::Event => match entry.boundary {
            Some(Boundary::Compaction) => ClassKind::Compaction,
            Some(Boundary::CompactionSummary) => ClassKind::CompactionSummary,
            Some(Boundary::Microcompaction) => ClassKind::Microcompaction,
            None => ClassKind::Event,
        },
    }
}

impl EntryClass {
    pub fn new(kind: ClassKind, author: ClassAuthor, off_path: bool, blocks: u8) -> Self {
        Self(kind as u16 | (author as u16) << 4 | u16::from(off_path) << 6 | u16::from(blocks) << 7)
    }
    /// The class as stored.
    pub fn bits(self) -> u16 {
        self.0
    }
    /// A stored class; `None` for bits no class has.
    pub fn from_bits(bits: u16) -> Option<Self> {
        let class = Self(bits);
        ((bits & 0xF) < KINDS.len() as u16 && (bits >> 4) & 3 < 3 && bits >> 15 == 0)
            .then_some(class)
    }
    pub fn kind(self) -> ClassKind {
        KINDS[usize::from(self.0 & 0xF)]
    }
    pub fn author(self) -> ClassAuthor {
        match (self.0 >> 4) & 3 {
            1 => ClassAuthor::Harness,
            2 => ClassAuthor::Prompt,
            _ => ClassAuthor::Other,
        }
    }
    fn off_path(self) -> bool {
        self.0 & (1 << 6) != 0
    }
    /// For a header, the kinds of block its message holds, one bit per
    /// entry of `BLOCK_KINDS`.
    fn blocks(self) -> u8 {
        (self.0 >> 7) as u8
    }
    /// A person's prompt: the header that opens a turn.
    pub fn is_prompt(self) -> bool {
        self.kind() == ClassKind::Header && self.author() == ClassAuthor::Prompt
    }
    pub fn is_call(self) -> bool {
        matches!(self.kind(), ClassKind::ChangeCall | ClassKind::Call)
    }
    pub fn is_header(self) -> bool {
        self.kind() == ClassKind::Header
    }
    /// An entry with what decides how a reading shows this class.
    fn sample(self, kind: ClassKind) -> IndexEntry {
        let (entry_kind, action, boundary) = match kind {
            ClassKind::Header => (EntryKind::Header, None, None),
            ClassKind::Text => (EntryKind::Text, None, None),
            ClassKind::Thinking => (EntryKind::Thinking, None, None),
            ClassKind::ThinkingEncrypted => (EntryKind::ThinkingEncrypted, None, None),
            ClassKind::ChangeCall => (EntryKind::Call, Some(ToolAction::Shell), None),
            ClassKind::Call => (EntryKind::Call, Some(ToolAction::Other), None),
            ClassKind::Result => (EntryKind::Result, None, None),
            ClassKind::Attachment => (EntryKind::Attachment, None, None),
            ClassKind::Other => (EntryKind::Other, None, None),
            ClassKind::Compaction => (EntryKind::Event, None, Some(super::Boundary::Compaction)),
            ClassKind::CompactionSummary => (
                EntryKind::Event,
                None,
                Some(super::Boundary::CompactionSummary),
            ),
            ClassKind::Microcompaction => (
                EntryKind::Event,
                None,
                Some(super::Boundary::Microcompaction),
            ),
            ClassKind::Event => (EntryKind::Event, None, None),
        };
        let event = entry_kind == EntryKind::Event;
        let (written_by, prompt_kind) = match self.author() {
            _ if event => (WrittenBy::Harness, None),
            ClassAuthor::Harness => (WrittenBy::Harness, Some(PromptKind::Harness)),
            ClassAuthor::Prompt => (WrittenBy::Human, Some(PromptKind::Prose)),
            ClassAuthor::Other => (WrittenBy::Model, None),
        };
        let mut entry = super::event_entry("class", "class", None, 0, Vec::new());
        entry.kind = entry_kind;
        entry.written_by = written_by;
        entry.prompt_kind = prompt_kind;
        entry.active_path = self.off_path().then_some(false);
        entry.action = action;
        entry.boundary = boundary;
        entry.event_kind = None;
        entry
    }
}

/// Each entry's class. A header's class names the kinds of block its
/// message holds: the blocks after it up to the next header or event.
pub fn entry_classes(entries: &[IndexEntry]) -> Vec<EntryClass> {
    let mut out = Vec::with_capacity(entries.len());
    let mut at = 0;
    while at < entries.len() {
        let entry = &entries[at];
        let author = if entry.kind == EntryKind::Event {
            ClassAuthor::Other
        } else if entry.written_by == WrittenBy::Harness {
            ClassAuthor::Harness
        } else if entry.written_by == WrittenBy::Human
            && entry.prompt_kind == Some(PromptKind::Prose)
        {
            ClassAuthor::Prompt
        } else {
            ClassAuthor::Other
        };
        let off_path = entry.active_path == Some(false);
        if entry.kind != EntryKind::Header {
            out.push(EntryClass::new(class_kind(entry), author, off_path, 0));
            at += 1;
            continue;
        }
        let end = entries[at + 1..]
            .iter()
            .position(|next| next.kind == EntryKind::Header || next.kind == EntryKind::Event)
            .map_or(entries.len(), |offset| at + 1 + offset);
        let mut blocks = 0u8;
        for block in &entries[at + 1..end] {
            let kind = class_kind(block);
            if let Some(bit) = BLOCK_KINDS.iter().position(|known| *known == kind) {
                blocks |= 1 << bit;
            }
        }
        out.push(EntryClass::new(ClassKind::Header, author, off_path, blocks));
        for block in &entries[at + 1..end] {
            out.push(EntryClass::new(class_kind(block), author, off_path, 0));
        }
        at = end;
    }
    out
}

impl FoldProfile {
    /// Whether this reading shows entries of `class`: the rule
    /// [`super::visible_entries`] applies, decided from the class alone.
    pub fn shows(&self, class: EntryClass) -> bool {
        if self.cost(&class.sample(class.kind())).is_none() {
            return false;
        }
        if class.kind() != ClassKind::Header
            || matches!(self.view, View::Transcript | View::Context | View::Raw)
        {
            return true;
        }
        BLOCK_KINDS.iter().enumerate().any(|(bit, kind)| {
            class.blocks() & (1 << bit) != 0 && self.cost(&class.sample(*kind)).is_some()
        })
    }
}

/// Counts per class of the entries before a place, and the running sum of
/// their full-form token estimates (what the context meter estimates from).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ItemCounts {
    pub tokens_full: u64,
    pub classes: BTreeMap<EntryClass, u64>,
}
impl ItemCounts {
    pub fn add(&mut self, entry: &IndexEntry, class: EntryClass) {
        self.tokens_full = self.tokens_full.saturating_add(entry.tokens_full);
        *self.classes.entry(class).or_default() += 1;
    }
    /// The entries counted here and not in `earlier`, a prefix of them.
    pub fn since(&self, earlier: &ItemCounts) -> ItemCounts {
        ItemCounts {
            tokens_full: self.tokens_full.saturating_sub(earlier.tokens_full),
            classes: self
                .classes
                .iter()
                .map(|(class, count)| {
                    (
                        *class,
                        count.saturating_sub(earlier.classes.get(class).copied().unwrap_or(0)),
                    )
                })
                .filter(|(_, count)| *count > 0)
                .collect(),
        }
    }
    /// The entries counted whose class satisfies `which`.
    pub fn count(&self, which: impl Fn(EntryClass) -> bool) -> u64 {
        self.classes
            .iter()
            .filter(|(class, _)| which(**class))
            .map(|(_, count)| count)
            .sum()
    }
    /// The entries counted that `profile` shows.
    pub fn shown(&self, profile: &FoldProfile) -> u64 {
        self.count(|class| profile.shows(class))
    }
}
