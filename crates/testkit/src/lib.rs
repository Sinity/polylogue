//! Test archives built from real files through the real import path, so every
//! test reads an archive the parsers and the writer built, never one
//! written row by row, and the sandbox every spawned daemon runs in.
//!
//! - [`Home`]: a synthetic home with an archive beside it, and
//!   [`Home::store`], which stores what is in it through the real engine.
//!   [`Home::morrow`] lays out the synthetic example archive in
//!   `examples/morrow/`; [`Home::new`] is an empty home with one Claude Code
//!   project directory for tests that write their own session lines
//!   ([`claude_user`], [`claude_assistant`]).
//! - [`Activity`]: what the harnesses did, written to the files each tool
//!   writes ([`Activity::claude_code_files`]) and, independently, to the
//!   [`View`] the docs say Polylogue shows ([`Activity::expected`]), which
//!   [`View::stored`] reads back from an archive. A [`Schedule`] says how
//!   the files arrive; [`generate`] holds proptest strategies for both.
//! - [`Sandbox`]: the environment of a spawned daemon or CLI, under its own
//!   root, with its own web port; `cargo xtask run` uses it too.
//!
//! The archives need the engine and the store (the default `archives`
//! feature); the sandbox and the child-process support need neither.

#[cfg(feature = "archives")]
pub mod activity;
#[cfg(feature = "archives")]
mod bundle;
#[cfg(feature = "archives")]
pub mod codex;
#[cfg(feature = "archives")]
pub mod generate;
#[cfg(feature = "archives")]
mod home;
#[cfg(feature = "archives")]
mod morrow;
mod process;
mod sandbox;
#[cfg(feature = "archives")]
mod schedule;
#[cfg(feature = "archives")]
pub mod view;

#[cfg(feature = "archives")]
pub use activity::{Activity, File, Transcript};
#[cfg(feature = "archives")]
pub use bundle::session_bundle;
#[cfg(feature = "archives")]
pub use home::{
    Home, MACHINE, WHOLE_FILES, claude_assistant, claude_user, delete_index, install_empty_index,
    store_with,
};
#[cfg(feature = "archives")]
pub use morrow::{Set, morrow_dir, session, short};
#[cfg(target_os = "linux")]
pub use process::ProcessExit;
pub use process::{die_with_parent, spawn_daemon};
pub use sandbox::Sandbox;
#[cfg(feature = "archives")]
pub use schedule::{Schedule, Step};
#[cfg(feature = "archives")]
pub use view::View;

/// Stable synthetic executable identity for in-process archive fixtures.
pub fn stopped_attempt() -> polylogue_model::batch::StoppedAttempt {
    polylogue_model::batch::StoppedAttempt {
        executable: polylogue_model::BlobHash::of(b"synthetic test executable"),
        parser_settings: None,
    }
}
