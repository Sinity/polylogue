//! Message IDs for records the source gives no ID.
//!
//! A byte offset, an ordinal or a time's text names a message only while the
//! file keeps its shape: a rewrite, an inserted line or a deleted chunk moves
//! every later position, and notes on those messages lose their target. A
//! message without a native ID is therefore named by what it says: its
//! session, a hash of its role and content, and how many earlier messages of
//! the same session in the same chain had the same role and content. An
//! edited message gets a new ID. An unchanged message keeps its ID unless an
//! earlier equal message (same session, role and content) was removed or
//! inserted: equal messages are numbered in order, so after `A X A` becomes
//! `X A` the remaining `A` has the first `A`'s ID and the second ID is gone.
//! Equal messages are interchangeable for identity; a note on the removed
//! copy names the remaining one.
//!
//! The number depends on the chain's earlier records, so a parser does not
//! write it: it names the message by its content and the position of its
//! record (`<session>~<role>-<hash>@<position>`), and the number is settled
//! when the chain's reading is formed ([`crate::chain_names`]).

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use polylogue_model::{MessageId, Role, SessionId};
use serde::Serialize;

/// The BLAKE3 of a value's canonical JSON, as the untagged hash text. Readings
/// and messages cite it as an identity, so its bytes carry no tag.
pub(crate) fn untagged_hash(value: &impl Serialize) -> String {
    polylogue_model::BlobHash::of(
        &canonical_json::canonical_json(value).expect("a serializable value has a canonical form"),
    )
    .to_string()
}

/// The content name of a message: `<session>~<role>-<hash>`, the native ID
/// without its number.
pub fn content_name(session: &SessionId, role: Role, content: &impl Serialize) -> Option<String> {
    let hash = canonical_json::canonical_hash("polylogue-message-content", &(role, content))
        .ok()?
        .to_hex();
    Some(format!(
        "{}~{}-{}",
        session.native().as_str(),
        role.as_str(),
        &hash.as_str()[..HASH_LEN]
    ))
}

const HASH_LEN: usize = 16;

/// The ID a parser gives a message named by its content: its content name
/// and the position of its record in the input. `content` is what the
/// message says, usually its blocks; it must not hold the message's position
/// or arrival time.
pub fn named(
    session: &SessionId,
    role: Role,
    content: &impl Serialize,
    position: u64,
) -> Option<MessageId> {
    let name = content_name(session, role, content)?;
    crate::document::message_id(session.origin().as_str(), &format!("{name}@{position}"))
}

/// [`named`] for a message's blocks. Normalized indexing text repeats the
/// original tool arguments and does not change message identity.
pub fn named_blocks(
    session: &SessionId,
    role: Role,
    blocks: &mut [polylogue_model::Block],
    position: u64,
) -> Option<MessageId> {
    let mut written = Vec::new();
    for (index, block) in blocks.iter_mut().enumerate() {
        if let polylogue_model::Block::ToolCall { written_text, .. } = block
            && let Some(text) = written_text.take()
        {
            written.push((index, text));
        }
    }
    let id = named(session, role, &blocks, position);
    for (index, text) in written {
        if let polylogue_model::Block::ToolCall { written_text, .. } = &mut blocks[index] {
            *written_text = Some(text);
        }
    }
    id
}

/// A message ID as a parser writes it for a content-named message: its
/// content name and its record's position. `None` for any other ID.
pub fn as_written(id: &MessageId) -> Option<(&str, u64)> {
    let (name, position) = id.native().as_str().rsplit_once('@')?;
    let position = position.parse().ok()?;
    is_content_name(name).then_some((name, position))
}

/// The content name of a settled content ID (`<name>-<number>`).
pub fn settled_name(native: &str) -> Option<&str> {
    let (name, number) = native.rsplit_once('-')?;
    (number.bytes().all(|b| b.is_ascii_digit()) && !number.is_empty() && is_content_name(name))
        .then_some(name)
}

/// The settled ID: the content name and how many earlier messages of the
/// chain have it.
pub fn numbered(origin: &str, name: &str, number: u64) -> Option<MessageId> {
    crate::document::message_id(origin, &format!("{name}-{number}"))
}

fn is_content_name(name: &str) -> bool {
    let Some((head, hash)) = name.rsplit_once('-') else {
        return false;
    };
    hash.len() == HASH_LEN && hash.bytes().all(|b| b.is_ascii_hexdigit()) && head.contains('~')
}

/// The text blocks of a message, joined: the content of a message whose
/// routes agree on its text but not on its other blocks (a page capture
/// and an activity export of one Gemini turn).
pub fn text(blocks: &[polylogue_model::Block]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            polylogue_model::Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
pub(crate) mod tests;
