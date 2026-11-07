//! Pure layout and streaming encoders over canonical projected read records.
//! The service selects records and prepares relationships; this crate never
//! reads sources, probes a terminal, or changes archived text.
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
