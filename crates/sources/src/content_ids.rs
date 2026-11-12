//! Message IDs for records the source gives no ID.
//!
//! A byte offset, an ordinal or a time's text names a message only while the
//! file keeps its shape: a rewrite, an inserted line or a deleted chunk moves
//! every later position, and notes on those messages lose their target. A
//! message without a native ID is therefore named by what it says: its
//! session, a hash of its role and content, and how many earlier messages of
//! the same session had the same role and content. An edited message gets a
//! new ID. An unchanged message keeps its ID unless an earlier equal message
//! (same session, role and content) was removed or inserted: equal messages
//! are numbered in order, so after `A X A` becomes `X A` the remaining `A`
//! has the first `A`'s ID and the second ID is gone. Equal messages are
//! interchangeable for identity; a note on the removed copy names the
//! remaining one.

use polylogue_model::{MessageId, Role, SessionId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Counts the messages named so far, per session, role and content. A parser
/// keeps one in its carried state, so a resumed parse continues the counts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentIds {
    /// Earlier messages, by session and content hash.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    seen: BTreeMap<String, u32>,
}

impl ContentIds {
    /// Normalized indexing text repeats the original tool arguments and does not change message identity.
    pub fn next_blocks(
        &mut self,
        session: &SessionId,
        role: Role,
        blocks: &mut [polylogue_model::Block],
    ) -> Option<MessageId> {
        let mut written = Vec::new();
        for (index, block) in blocks.iter_mut().enumerate() {
            if let polylogue_model::Block::ToolCall { written_text, .. } = block
                && let Some(text) = written_text.take()
            {
                written.push((index, text));
            }
        }
        let id = self.next(session, role, &blocks);
        for (index, text) in written {
            if let polylogue_model::Block::ToolCall { written_text, .. } = &mut blocks[index] {
                *written_text = Some(text);
            }
        }
        id
    }

    /// The ID of the next message of `session` with this role and content.
    /// `content` is what the message says, usually its blocks; it must not
    /// hold the message's position or arrival time.
    pub fn next(
        &mut self,
        session: &SessionId,
        role: Role,
        content: &impl Serialize,
    ) -> Option<MessageId> {
        let hash = canonical_json::canonical_hash("polylogue-message-content", &(role, content))
            .ok()?
            .to_hex();
        let hash = &hash.as_str()[..16];
        let occurrence = self.seen.entry(format!("{session}\0{hash}")).or_insert(0);
        let native = format!(
            "{}~{}-{hash}-{occurrence}",
            session.native().as_str(),
            role.as_str()
        );
        *occurrence += 1;
        crate::document::message_id(session.origin().as_str(), &native)
    }
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
