//! Pure layout and streaming encoders over canonical projected read records.
//! The service selects records and prepares relationships; this crate never
//! reads sources, probes a terminal, or changes archived text.
//!
//! The service selects canonical `bundle::Record` values, calls
//! `model::read::project_record`, and supplies `ProjectionEvidence`.
//! `visit_record` emits `ReadItem` values to a callback; `layout_record`
//! collects the same items for one record. Selection, source access, context
//! membership, grants and tool relationship lookup belong to the service and
//! store.
//!
//! `Encoder` writes a header, items and footer incrementally to a caller-owned
//! `Write` sink. Terminal, Markdown, HTML, JSON and JSONL consume the same
//! items. There are no filesystem, environment, clock or terminal probes. A
//! full export releases each source record after writing it. JSON bundle
//! records pass directly to the bundle writer; layout substitutions never
//! become canonical source records.
//!
//! `PageBuilder` holds one bounded page. `push_at` receives the canonical
//! record reference and item ordinal separately from the item's displayed
//! source, because a paired tool call can originate outside the current record.
//! Its `ItemPlace` is the next unread item or Unicode character position. The
//! service seals that position with the selection, options and source revision,
//! and constructs every `ReadyCall` through the command table.
//!
//! An oversized text item becomes explicit `text` fragments. An oversized
//! structured item becomes `read_item_json` fragments. Concatenating their
//! `text` values reconstructs the complete original text or serialized
//! `ReadItem`; `from`, `until` and `total` count Unicode characters. Fragment
//! positions never change a canonical block ordinal. The byte budget includes
//! the page document. `finish_with` regenerates the footer from the final
//! unread position, removing trailing items or shortening the first fragment
//! when follow-up metadata needs more room. The service reserves the outer
//! transport envelope. A one-token budget still advances by a minimum source
//! fragment; accessible-empty text remains an item. Mandatory metadata that
//! cannot fit without any source content is reported explicitly.
//!
//! Expanded tool output keeps its first and last `output_lines` lines (100 by
//! default), with exact counts for the omitted middle. A line over 4,000
//! characters is bounded by retained ends. `output_lines: 0` keeps the complete
//! value. The canonical record always retains the complete result.
//! Availability, assertions, context omissions, branches and variants retain
//! typed value states instead of inventing zero or success.
//!
//! Web routes embed `html::document` in the common `web_shell::Frame`.
//! Standalone HTML exports embed the same shared visual tokens, use system
//! fonts and contain no scripts. Archived Markdown has raw HTML disabled and
//! checked link schemes. Store-prepared image bytes are verified against the
//! canonical blob hash; only supported raster MIME types within the configured
//! limit are embedded. SVG and HTML remain links. Multiple tool-result
//! attachments have separate evidence keyed by canonical source reference,
//! original block ordinal and attachment position.
mod encode;
pub mod html;
mod layout;
pub mod measures;
pub mod pager;
pub mod ready;
pub mod results;
pub mod schema;
pub mod terminal;
pub mod web;
pub use encode::{ARCHIVE_NOTICE, EncodeOptions, Encoder, encode_document, json_bytes};
pub use layout::{
    layout_record, presentation_text_visible, visit_document_root, visit_record, visit_source_block,
};
pub use polylogue_model::read;
pub use polylogue_model::read::Format;

/// One argument as a POSIX shell reads it. This only formats stored arguments.
pub fn shell_quote(word: &str) -> String {
    if !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=@%+,".contains(c))
    {
        word.to_owned()
    } else {
        format!("'{}'", word.replace('\'', r"'\''"))
    }
}
#[cfg(test)]
mod tests;
