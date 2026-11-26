//! Test archives built from real files through the real import path, so every
//! test reads an archive the parsers and the writer built, never one
//! written row by row, and the sandbox every spawned daemon runs in.
//!
//! - [`Morrow`]: the synthetic example archive in `examples/morrow/`.
//! - [`ClaudeHome`]: an empty home with one Claude Code project directory,
//!   for tests that write their own session lines
//!   ([`claude_user`], [`claude_assistant`]).
//! - [`Sandbox`]: the environment of a spawned daemon or CLI, under its own
//!   root, with its own web port; `cargo xtask run` uses it too.
//!
//! The archives need the engine and the store (the default `archives`
//! feature); the sandbox and the child-process support need neither.

#[cfg(feature = "archives")]
mod bundle;
#[cfg(feature = "archives")]
mod claude_home;
#[cfg(feature = "archives")]
pub mod codex;
#[cfg(feature = "archives")]
mod morrow;
mod process;
mod sandbox;

#[cfg(feature = "archives")]
pub use bundle::session_bundle;
#[cfg(feature = "archives")]
pub use claude_home::{
    ClaudeHome, claude_assistant, claude_user, delete_index, install_empty_index,
};
#[cfg(feature = "archives")]
pub use morrow::{Morrow, Set, morrow_dir, session, short};
#[cfg(target_os = "linux")]
pub use process::ProcessExit;
pub use process::{die_with_parent, spawn_daemon};
pub use sandbox::Sandbox;
