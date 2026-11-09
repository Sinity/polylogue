//! The page cursor of a read. It names where the next page starts by an entry
//! key or a selected row's order values, never by a row number, and it binds
//! the rewrite version of every session the reading composes. An append to a
//! session never stales a cursor; a rewrite of anything already indexed does.
use crate::{BlobHash, MessageId, SessionId};
use query_language::cursor::OrderPosition;
pub use query_language::identity::QuestionId;
use serde::{Deserialize, Serialize};

/// One place in a session's item index: a message header (no block), one of
/// its blocks by original ordinal, or an in-place event.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EntryKey {
    Message {
        message: MessageId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block: Option<u64>,
    },
    Event {
        event: String,
    },
}
impl EntryKey {
    pub fn header(message: &MessageId) -> Self {
        Self::Message {
            message: message.clone(),
            block: None,
        }
    }
    pub fn message(&self) -> Option<&MessageId> {
        match self {
            Self::Message { message, .. } => Some(message),
            Self::Event { .. } => None,
        }
    }
    /// The reference a person or agent uses for this place.
    pub fn reference(&self) -> String {
        match self {
            Self::Message {
                message,
                block: None,
            } => format!("message:{message}"),
            Self::Message {
                message,
                block: Some(n),
            } => format!("block:{message}#{n}"),
            Self::Event { event } => format!("event:{event}"),
        }
    }
}

/// The rewrite version a cursor saw for one session of its composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionVersion {
    pub session: SessionId,
    pub rewrite_version: u64,
    /// Receiving namespaces whose members determine attachment presentation.
    pub attachments: Option<BlobHash>,
}

/// Where the next page starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PagePosition {
    /// A place in a composed session reading. `inside` continues an entry
    /// too large for one page: the layout item within the entry (`item`)
    /// and the character within its encoding where the page starts.
    Entry {
        key: EntryKey,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        inside: Option<ItemPlace>,
    },
    /// A node of a links tree, by session.
    Node { session: SessionId },
    /// A place in a row reading (documents, events, authored records and
    /// query rows). `anchor_row` holds the order values of the row the place
    /// is in, so the selection resumes by key; `place.selection.row` is 0.
    Record {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        anchor_row: Option<OrderPosition>,
        place: ItemPlace,
    },
}

/// Everything a continuation needs, sealed by the daemon with the index
/// generation's key so a caller cannot forge a position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageCursor {
    /// The reference or query text as the caller gave it.
    pub target: String,
    /// Digest of the reading options, the format and the enabled variants.
    /// A continuation with other options is refused; the page budget may
    /// change for one page.
    pub options: String,
    pub versions: BlobHash,
    pub position: PagePosition,
    pub direction: ReadDirection,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CursorError {
    #[error("the read cursor is not a cursor this archive issued")]
    Invalid,
    #[error("the read cursor belongs to another reading: {0}")]
    OtherReading(&'static str),
    /// Issued by an earlier index generation: refused as stale, never
    /// restarted and never confused with a broken cursor.
    #[error("the read cursor was issued by an earlier index generation; read again")]
    Stale,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ReadRecordKind {
    Session,
    LinkIssue,
    Message,
    Event,
    Link,
    Document,
    Authored,
    Variant,
    VariantPart,
    Blob,
    TextFragment,
    Presentation,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct ReadRecordLocator {
    pub kind: ReadRecordKind,
    /// Position in the selected row's prepared metadata view, before layout.
    /// This is independent of a message's original physical-session order.
    pub ordinal: u64,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct SelectionLocator {
    pub row: u64,
    pub record: ReadRecordLocator,
}

/// A place inside a row reading: the record of the selected row, the
/// layout item within it, and the character within that item's encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ItemPlace {
    pub selection: SelectionLocator,
    pub item: u64,
    /// Unicode scalar position in the complete C1-safe encoded ReadItem JSON.
    pub character: u64,
    /// Digest of that complete encoded item, including its original Source.
    pub item_hash: BlobHash,
}

/// Fragments cite an acknowledged query or the actual exact-object run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadFraming {
    Query { question: QuestionId },
    Object { run: crate::RunId },
}
impl ReadFraming {
    pub fn source(&self) -> super::Source {
        super::Source::record(match self {
            Self::Query { question } => format!("query:{question}"),
            Self::Object { run } => format!("run:{run}"),
        })
    }
}

/// Which way a page goes from its cursor's place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadDirection {
    /// Resume at the place.
    After,
    /// Read content strictly before the place.
    Before,
    /// Read what follows the place, which ended the reading when the cursor
    /// was issued: what a live session appended since.
    Past,
}

const SEAL_CONTEXT: &[u8] = b"polylogue.page-cursor\0";

/// What a cursor is sealed with: the index generation's key and the
/// generation itself. The generation is written inside the sealed payload;
/// read without the key it only tells a cursor of an earlier generation
/// (stale) from one that was never a cursor (invalid), as query cursors do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorSeal {
    pub key: [u8; 32],
    pub generation: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sealed<C> {
    generation: String,
    cursor: C,
}

impl PageCursor {
    /// `PAYLOAD.MAC`: the hex of the generation and the cursor as JSON,
    /// and its keyed digest.
    pub fn seal(&self, seal: &CursorSeal) -> String {
        let payload = serde_json::to_vec(&Sealed {
            generation: seal.generation.clone(),
            cursor: self,
        })
        .expect("a page cursor always encodes");
        let mut digest = blake3::Hasher::new_keyed(&seal.key);
        digest.update(SEAL_CONTEXT);
        digest.update(&payload);
        let mut text = hex(&payload);
        text.push('.');
        text.push_str(&digest.finalize().to_hex());
        text
    }

    /// Authenticate before decoding; a changed byte anywhere is refused. A
    /// cursor that does not authenticate and names another generation is
    /// stale.
    pub fn open(text: &str, seal: &CursorSeal) -> Result<Self, CursorError> {
        let (payload, mac) = text.split_once('.').ok_or(CursorError::Invalid)?;
        let payload = unhex(payload).ok_or(CursorError::Invalid)?;
        let mac = blake3::Hash::from_hex(mac).map_err(|_| CursorError::Invalid)?;
        let mut digest = blake3::Hasher::new_keyed(&seal.key);
        digest.update(SEAL_CONTEXT);
        digest.update(&payload);
        if digest.finalize() != mac {
            let generation = serde_json::from_slice::<Sealed<serde::de::IgnoredAny>>(&payload)
                .map(|sealed| sealed.generation);
            return Err(match generation {
                Ok(generation) if generation != seal.generation => CursorError::Stale,
                _ => CursorError::Invalid,
            });
        }
        let sealed =
            serde_json::from_slice::<Sealed<Self>>(&payload).map_err(|_| CursorError::Invalid)?;
        // The key can outlive a generation; the generation it names decides.
        if sealed.generation != seal.generation {
            return Err(CursorError::Stale);
        }
        Ok(sealed.cursor)
    }

    /// Refuse a cursor issued for another target or other options.
    pub fn check(&self, target: &str, options: &str) -> Result<(), CursorError> {
        if self.target != target {
            return Err(CursorError::OtherReading("the target differs"));
        }
        if self.options != options {
            return Err(CursorError::OtherReading("the reading options differ"));
        }
        Ok(())
    }

    /// Bind all composed sessions without making the cursor grow with lineage.
    pub fn version(versions: &[SessionVersion]) -> BlobHash {
        BlobHash::of(&serde_json::to_vec(versions).expect("session versions encode"))
    }

    pub fn rewritten(&self, current: &BlobHash) -> bool {
        self.versions != *current
    }
}

/// The digest a cursor binds its reading options with.
pub fn options_digest(parts: &impl Serialize) -> String {
    let bytes = serde_json::to_vec(parts).expect("reading options always encode");
    blake3::hash(&bytes).to_hex().to_string()
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 15)] as char);
    }
    out
}
fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[high, low]| Some((digit(*high)? << 4) | digit(*low)?))
        .collect()
}
fn digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
