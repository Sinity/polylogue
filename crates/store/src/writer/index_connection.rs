//! The index writer's connection to its index file. It is open, or the
//! writer has none: while a finished rebuild is installed, and after an
//! install or a restart of a rebuild failed. Without one every index write is
//! refused with the failure that left it so, never run against another
//! database, and the step that failed is tried again at the next commit
//! event ([`super::Inner::retry_lost_index`]) and by the next start.

use crate::{Result, StoreError};
use rusqlite::Connection;

pub(super) enum IndexConnection {
    Open(Connection),
    Lost(Lost),
}

/// Why the writer has no index connection, and what brings one back.
pub(super) struct Lost {
    pub failure: polylogue_model::Failure,
    pub retry: Retry,
}

pub(super) enum Retry {
    /// Install the finished rebuild (`Rebuilding::Ready`, or a `build.db`
    /// whose rebuild has ended), then open `index.db`.
    Install,
    /// Start the failed rebuild again, then open its new build file.
    Restart { error: String },
}

impl IndexConnection {
    pub(super) fn get(&self) -> Result<&Connection> {
        match self {
            Self::Open(conn) => Ok(conn),
            Self::Lost(lost) => Err(StoreError::Failed(lost.failure.clone())),
        }
    }

    pub(super) fn get_mut(&mut self) -> Result<&mut Connection> {
        match self {
            Self::Open(conn) => Ok(conn),
            Self::Lost(lost) => Err(StoreError::Failed(lost.failure.clone())),
        }
    }

    /// Closes the connection for `retry`, leaving the writer without one
    /// until the caller opens the next. A close that fails still gives the
    /// connection up: SQLite closes it when it is dropped. A connection
    /// already lost stays as it is.
    pub(super) fn close(&mut self, retry: Retry) -> Result<()> {
        let Self::Open(_) = self else {
            return Ok(());
        };
        let failure = polylogue_model::Failure::new(
            "closing the index connection",
            &StoreError::internal("the index connection is closed while this step runs"),
        );
        let Self::Open(conn) = std::mem::replace(self, Self::Lost(Lost { failure, retry })) else {
            unreachable!("matched as open above");
        };
        conn.flush_prepared_statement_cache();
        conn.close().map_err(|(_, error)| StoreError::from(error))
    }

    /// The step a lost connection waits for, if any.
    pub(super) fn retry(&self) -> Option<&Retry> {
        match self {
            Self::Open(_) => None,
            Self::Lost(lost) => Some(&lost.retry),
        }
    }
}
