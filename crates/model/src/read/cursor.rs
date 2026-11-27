//! The page cursor of a read. It names where the next page starts by an entry
//! key or a selected row's order values, never by a row number, and it binds
//! the rewrite version of every session the reading composes. An append to a
//! session never stales a cursor; a rewrite of anything already indexed does.
use crate::{BlobHash, MessageId, SessionId, hex};
use query_language::cursor::OrderPosition;
pub use query_language::identity::QuestionId;
use serde::{Deserialize, Serialize};

/// One place in a session's transcript lines: a message header (no block), one of
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
    /// A place in the depth-first walk of a links tree.
    Links { place: LinksPlace },
    /// A place in a row reading (documents, events, authored records and
    /// query rows). `start_row` holds the order values of the row the place
    /// is in, so the selection resumes by key; `place.selection.row` is 0.
    Record {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        start_row: Option<OrderPosition>,
        place: ItemPlace,
        /// Per-reading identity for exact saved-history actions. Other row
        /// cursors omit it; saved-history cursors without it must restart.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        history_seed: Option<BlobHash>,
    },
}

/// A place in the depth-first walk of a links tree: what the walk holds
/// there, so a page resumes from it without walking the nodes before it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinksPlace {
    /// How many nodes come before the place.
    pub before: u64,
    /// The nodes being walked, outermost first, each with how many of its
    /// tree links were taken; the first has no session and holds the roots.
    pub frames: Vec<LinksFrame>,
    /// The nodes before the place that more than one link reaches, each
    /// with the link that showed it; a later link to one is a cross-link.
    pub shown: Vec<LinksShown>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinksFrame {
    pub session: Option<String>,
    pub taken: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinksShown {
    pub session: String,
    /// The node whose link showed it; none for a root.
    pub parent: Option<String>,
    /// That link's place among the parent's tree links (or the roots).
    pub slot: u32,
}

/// Everything a continuation needs, sealed by the daemon with the
/// index instance's key so a caller cannot forge a position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageCursor {
    /// The physical-session identity and accepted lower bound, even when a
    /// context continuation binds its executable target to one message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_read: Option<SessionRead>,
    /// The reference or query text as the caller gave it.
    pub target: String,
    /// Digest of the reading options, the format and the enabled alternates.
    /// A continuation with other options is refused; the page budget may
    /// change for one page.
    pub digest: String,
    /// The alternates the reading renders, resolved on its first page, so a
    /// continuation renders the same ones whatever the setting says now.
    pub alternates: Vec<String>,
    pub versions: BlobHash,
    pub position: PagePosition,
    pub direction: ReadDirection,
}

/// One session reading's accepted start. A later display acknowledgement
/// never changes the lower bound of its next or previous pages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionRead {
    pub session: SessionId,
    pub from_seen: bool,
    pub after: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub logical: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<SessionSeen>,
}

/// Accepted physical marks of a composed reading, including absent marks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionSeen {
    pub session: SessionId,
    pub after: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CursorError {
    #[error("the read cursor is not a cursor this archive issued")]
    Invalid,
    #[error("the read cursor belongs to another reading: {0}")]
    OtherReading(&'static str),
    /// Issued by a build that read its options another way: refused as
    /// stale, so the caller starts the reading again.
    #[error("the read cursor was issued by an earlier build of Polylogue; read again without it")]
    EarlierBuild,
    /// Issued before the index was rebuilt: refused as stale, never
    /// restarted and never confused with a broken cursor.
    #[error("the index was rebuilt since this cursor was issued")]
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
    Run,
    Alternate,
    AlternatePart,
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
    Query { query_id: QuestionId },
    Object { run: crate::RunId },
}
impl ReadFraming {
    pub fn source(&self) -> super::Source {
        super::Source::record(match self {
            Self::Query { query_id } => format!("query:{query_id}"),
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

/// What a cursor is sealed with: the index instance's key and the
/// index instance itself. The index instance is written inside the sealed payload;
/// read without the key it only tells a cursor of an earlier index instance
/// (stale) from one that was never a cursor (invalid), as query cursors do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorSeal {
    pub key: [u8; 32],
    pub index_instance: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sealed<C> {
    index_instance: String,
    cursor: C,
}

impl CursorSeal {
    /// `PAYLOAD.MAC`: the hex of the index instance and the cursor as JSON, and
    /// its keyed digest. `domain` names the kind of cursor, so a cursor of
    /// one kind never opens as another.
    pub fn seal(&self, domain: &[u8], cursor: &impl Serialize) -> String {
        let payload = serde_json::to_vec(&Sealed {
            index_instance: self.index_instance.clone(),
            cursor,
        })
        .expect("a cursor always encodes");
        let mut text = hex(&payload);
        text.push('.');
        text.push_str(&self.digest(domain, &payload).to_hex());
        text
    }

    /// Authenticate before decoding; a changed byte anywhere is refused. A
    /// cursor that does not authenticate and names another index instance is
    /// stale.
    pub fn open<C: serde::de::DeserializeOwned>(
        &self,
        domain: &[u8],
        text: &str,
    ) -> Result<C, CursorError> {
        let (payload, mac) = text.split_once('.').ok_or(CursorError::Invalid)?;
        let payload = unhex(payload).ok_or(CursorError::Invalid)?;
        let mac = blake3::Hash::from_hex(mac).map_err(|_| CursorError::Invalid)?;
        if self.digest(domain, &payload) != mac {
            let index_instance = serde_json::from_slice::<Sealed<serde::de::IgnoredAny>>(&payload)
                .map(|sealed| sealed.index_instance);
            return Err(match index_instance {
                Ok(index_instance) if index_instance != self.index_instance => CursorError::Stale,
                _ => CursorError::Invalid,
            });
        }
        let sealed =
            serde_json::from_slice::<Sealed<C>>(&payload).map_err(|_| CursorError::Invalid)?;
        // The key can outlive an index instance; the index instance it names decides.
        if sealed.index_instance != self.index_instance {
            return Err(CursorError::Stale);
        }
        Ok(sealed.cursor)
    }

    fn digest(&self, domain: &[u8], payload: &[u8]) -> blake3::Hash {
        let mut digest = blake3::Hasher::new_keyed(&self.key);
        digest.update(domain);
        digest.update(payload);
        digest.finalize()
    }
}

impl PageCursor {
    pub fn seal(&self, seal: &CursorSeal) -> String {
        seal.seal(SEAL_CONTEXT, self)
    }

    pub fn open(text: &str, seal: &CursorSeal) -> Result<Self, CursorError> {
        seal.open(SEAL_CONTEXT, text)
    }

    /// Refuse a cursor issued for another target or other options.
    pub fn check(&self, target: &str, digest: &str) -> Result<(), CursorError> {
        if self.target != target {
            return Err(CursorError::OtherReading("the target differs"));
        }
        if !self.digest.starts_with(OPTIONS_DOMAIN) {
            return Err(CursorError::EarlierBuild);
        }
        if self.digest != digest {
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

/// The domain of the options digest, written before it. A cursor whose
/// digest names no domain, or another, was issued by a build that read its
/// options another way.
const OPTIONS_DOMAIN: &str = "read-options-2:";

/// The digest a cursor binds its reading options with.
pub fn options_digest(parts: &impl Serialize) -> String {
    let bytes = serde_json::to_vec(parts).expect("reading options always encode");
    format!("{OPTIONS_DOMAIN}{}", blake3::hash(&bytes).to_hex())
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
