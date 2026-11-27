//! The index writer's connection to `index.db`. It is open, absent while no
//! index this build reads is installed (a rebuild writes its own file), or
//! lost: while a ready rebuild is installed, and after an install failed.
//! Absent, every index write is refused with `index_building` and the
//! rebuild's progress; lost, with the failure that left it so, and the
//! install is tried again at the next commit event
//! ([`super::Inner::retry_lost_index`]) and by the next start. An index
//! write is never run against another database.

use crate::archive::Archive;
use crate::{Result, StoreError};
use rusqlite::Connection;

pub(super) enum IndexConnection {
    Open(Connection),
    /// No index this build reads is installed.
    Absent(Archive),
    Lost(Lost),
}

/// Why the writer has no index connection, and what brings one back.
pub(super) struct Lost {
    pub failure: polylogue_model::Failure,
    pub retry: Retry,
}

pub(super) enum Retry {
    /// Install the ready rebuild, then open `index.db`.
    Install,
}

impl IndexConnection {
    pub(super) fn get(&self) -> Result<&Connection> {
        match self {
            Self::Open(conn) => Ok(conn),
            Self::Absent(archive) => Err(absent(archive)),
            Self::Lost(lost) => Err(StoreError::Failed(lost.failure.clone())),
        }
    }

    pub(super) fn get_mut(&mut self) -> Result<&mut Connection> {
        match self {
            Self::Open(conn) => Ok(conn),
            Self::Absent(archive) => Err(absent(archive)),
            Self::Lost(lost) => Err(StoreError::Failed(lost.failure.clone())),
        }
    }

    /// Closes the connection for `retry`, leaving the writer without one
    /// until the caller opens the next. A close that fails still gives the
    /// connection up: SQLite closes it when it is dropped. A connection
    /// absent or already lost stays as it is.
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
            Self::Open(_) | Self::Absent(_) => None,
            Self::Lost(lost) => Some(&lost.retry),
        }
    }
}

/// The refusal of an index write while no index is installed: the one a
/// read gets, with the rebuild's reason and progress.
fn absent(archive: &Archive) -> StoreError {
    match archive.served_index() {
        Err(error) => error,
        Ok(_) => StoreError::internal("the index writer has no index while one is installed"),
    }
}
