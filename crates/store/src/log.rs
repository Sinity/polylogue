//! The daemon's log: each record goes to standard error as a journal line
//! with its priority (`<4>polylogue: text`), and up to [`KEPT`] event records
//! stay in memory so `status` can show what the daemon said recently.
//!
//! It lives in the store because the store is the lowest crate whose code
//! runs in the daemon and logs (the writer, the search thread); the engine and the daemon depend on it. The model
//! crate does no I/O.
//!
//! Progress records (`log!(progress, ...)`, a rate logged once a minute) go
//! to the journal at the information priority and are not kept: `status`
//! shows progress itself, and its recent records stay events.
//!
//! Messages a command prints for the person running it are not log records
//! and do not come here.
use polylogue_model::{LockExt as _, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Mutex;

/// How many records the daemon keeps for `status`.
pub const KEPT: usize = 200;

/// A record's priority, as the service journal understands it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// Something failed.
    Error,
    /// Something degraded, or will be tried again.
    Warning,
    /// Progress and lifecycle.
    Info,
}

impl Priority {
    /// The journal's syslog priority number.
    fn journal(self) -> u8 {
        match self {
            Self::Error => 3,
            Self::Warning => 4,
            Self::Info => 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Record {
    /// When it was logged, by the system clock.
    pub at: Timestamp,
    pub priority: Priority,
    pub text: String,
}

/// The records kept, oldest first.
#[derive(Default)]
pub(crate) struct Recent {
    records: VecDeque<Record>,
}

impl Recent {
    pub(crate) fn push(&mut self, record: Record) {
        if self.records.len() == KEPT {
            if let Some(at) = self
                .records
                .iter()
                .position(|record| record.priority == Priority::Info)
            {
                self.records.remove(at);
            } else if record.priority == Priority::Info {
                return;
            } else {
                self.records.pop_front();
            }
        }
        self.records.push_back(record);
    }

    pub(crate) fn last(&self, count: usize) -> Vec<Record> {
        let skip = self.records.len().saturating_sub(count);
        self.records.iter().skip(skip).cloned().collect()
    }
}

static RECENT: Mutex<Recent> = Mutex::new(Recent {
    records: VecDeque::new(),
});

/// The line the journal reads for a record.
pub(crate) fn journal_line(priority: Priority, text: &str) -> String {
    format!("<{}>polylogue: {text}", priority.journal())
}

/// Logs a progress record to the journal only; use [`log!`](crate::log!).
pub fn write_progress(text: std::fmt::Arguments<'_>) {
    eprintln!("{}", journal_line(Priority::Info, &text.to_string()));
}

/// Logs `text` with `priority`; use [`log!`](crate::log!).
pub fn write(priority: Priority, text: std::fmt::Arguments<'_>) {
    let text = text.to_string();
    eprintln!("{}", journal_line(priority, &text));
    // The journal stamps its own lines; this time is only for `status`, and
    // the log has no clock passed to it.
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|since| i64::try_from(since.as_millis()).ok())
        .and_then(Timestamp::from_unix_ms);
    if let Some(at) = at {
        RECENT.locked().push(Record { at, priority, text });
    }
}

/// The last `count` records, oldest first.
pub fn recent(count: usize) -> Vec<Record> {
    RECENT.locked().last(count)
}

/// Logs a daemon message: `log!(warning, "scope index: {error}")`.
#[macro_export]
macro_rules! log {
    (error, $($message:tt)+) => {
        $crate::log::write($crate::log::Priority::Error, format_args!($($message)+))
    };
    (warning, $($message:tt)+) => {
        $crate::log::write($crate::log::Priority::Warning, format_args!($($message)+))
    };
    (info, $($message:tt)+) => {
        $crate::log::write($crate::log::Priority::Info, format_args!($($message)+))
    };
    (progress, $($message:tt)+) => {
        $crate::log::write_progress(format_args!($($message)+))
    };
}

#[cfg(test)]
mod tests;
