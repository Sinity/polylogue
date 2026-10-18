//! Rendering (architecture 7). Views are filters over one stream of
//! rendered blocks, in every format. Credentials are masked on output, per
//! block, never by refusing.

pub mod cards;
pub mod model_read;

use polylogue_model::{Event, Message, Session};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Terminal,
    Markdown,
    Html,
    Json,
}

/// Which blocks a view keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// Everything, tool calls and results included.
    Full,
    /// What the people and models said: prose and thinking, no tool output.
    Dialogue,
    /// The operator's own prompts only.
    Prompts,
    /// Tool calls and their outcomes.
    Actions,
}

/// One rendered unit of the stream every format is built from.
pub enum Rendered<'a> {
    SessionHeader(&'a Session),
    Message(&'a Message),
    Event(&'a Event),
}

pub fn render(
    _items: &[Rendered<'_>],
    _format: Format,
    _view: View,
    _out: &mut dyn std::io::Write,
) -> std::io::Result<()> {
    // Architecture 7: one block stream, views as filters, masking per block.
    todo!()
}
