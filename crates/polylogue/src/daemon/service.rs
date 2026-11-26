//! Application operations and the archive state they share.
mod answers;
mod start;
mod state;
mod status;

use crate::daemon::Failure;
pub(crate) use answers::*;
pub use start::{ArchiveMode, Runtime, start};
pub use state::Daemon;
use state::ModeState;
pub(crate) use state::Running;

pub(crate) mod browser;
pub(crate) mod handlers;
pub(crate) mod line;
pub(crate) mod read;

#[cfg(test)]
mod tests;

pub(crate) mod acts;
pub(crate) mod feeds;
pub(crate) mod follow;
