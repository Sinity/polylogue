//! Test worlds built from real files through the real import path, so every
//! test reads an archive the parsers and the writer built, never one
//! written row by row.
//!
//! - [`Morrow`]: the synthetic example archive in `examples/morrow/`.
//! - [`ClaudeHome`]: an empty home with one Claude Code project directory,
//!   for tests that write their own session lines
//!   ([`claude_user`], [`claude_assistant`]).

mod claude_home;
pub mod codex;
mod morrow;
mod process;

pub use claude_home::{ClaudeHome, claude_assistant, claude_user, delete_index};
pub use morrow::{Morrow, Set, morrow_dir, session, short};
#[cfg(target_os = "linux")]
pub use process::ProcessExit;
pub use process::spawn_daemon;
