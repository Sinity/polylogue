//! Pure layout and streaming encoders over projected read records.
//! The service selects records and prepares relationships; this crate never
//! reads sources, probes a terminal, or changes archived text.
//!
//! The service selects the shown `bundle::Record` values, calls
//! `model::read::project_record`, and supplies `LayoutFacts`.
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
//! become source records.
//!
//! `PageBuilder` holds one bounded page. `push_at` receives the
//! record reference and item ordinal separately from the item's displayed
//! source, because a paired tool call can originate outside the current record.
//! Its `ItemPlace` is the next unread item or Unicode character position. The
//! service seals that position with the selection, options and source revision,
//! and constructs every `ReadyCall` through the command table.
//!
//! An oversized item of any kind becomes `fragment` items. Concatenating
//! their `text` values reconstructs the item's complete JSON encoding;
//! `position.character`, `until` and `total` count Unicode characters. Fragment
//! positions never change a stored block ordinal. The byte budget includes
//! the page document. `finish_with` regenerates the footer from the final
//! unread position, removing trailing items or shortening the first fragment
//! when follow-up metadata needs more room. The service reserves room for the
//! transport's answer metadata. A one-token budget still advances by a minimum
//! source fragment; accessible-empty text remains an item. Mandatory metadata that
//! cannot fit without any source content is reported explicitly.
//!
//! Expanded tool output keeps its first and last N lines (`--show tools:N`,
//! 100 by default), with counts for the omitted middle. A line over 4,000
//! characters is bounded by kept ends. `--show tools:full` keeps the complete
//! value. The source record always keeps the complete result.
//! Availability, annotations, context omissions, branches and alternates keep
//! typed value states instead of inventing zero or success.
//!
//! Web routes embed `html::document` in the common `web_shell::Frame`.
//! Standalone HTML exports embed the same shared visual tokens, use system
//! fonts and contain no scripts. Archived Markdown has raw HTML disabled and
//! checked link schemes. Store-prepared image bytes are verified against the
//! stored blob hash; only supported raster MIME types within the configured
//! limit are embedded. SVG and HTML remain links. Multiple tool-result
//! attachments have separate prepared data keyed by source reference,
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
pub use polylogue_model::read::{Format, ReadError};

/// `45s`, `7m`, `3h12m`, `2d4h`.
pub fn short_duration(ms: i64) -> String {
    let s = ms.max(0) / 1000;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m", s / 60),
        3600..86400 => format!("{}h{}m", s / 3600, s % 3600 / 60),
        _ => format!("{}d{}h", s / 86400, s % 86400 / 3600),
    }
}

pub use polylogue_model::command::shell_quote;
#[cfg(test)]
mod tests;
