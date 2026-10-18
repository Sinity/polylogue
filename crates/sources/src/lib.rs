//! Provider file parsers. No IO: the caller supplies bytes and stored state.
pub mod claude_code;
pub mod codex;
mod decimal;
pub mod frame;
mod raw;
pub mod selection;
mod time;
