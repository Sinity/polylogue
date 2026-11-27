//! Rows committed after a position, read back in commit order: what a
//! follower of the archive takes in.
use super::Position;
use crate::row::{Rows as _, row};
use crate::{Reader, Result, StoreError};
use std::ops::ControlFlow;

/// The kinds of rows [`Reader::after`] reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Followed {
    pub revisions: bool,
    pub observations: bool,
    pub alternates: bool,
}

impl Followed {
    pub const ALL: Self = Self {
        revisions: true,
        observations: true,
        alternates: true,
    };
}

/// One row of the record, at the position of the commit that wrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchiveRow {
    Revision {
        position: Position,
        input: String,
        number: i64,
        chain: i64,
        length: u64,
    },
    Observation {
        position: Position,
        id: String,
        input: String,
        class: String,
    },
    Alternate {
        position: Position,
        id: String,
    },
}

impl ArchiveRow {
    pub fn position(&self) -> Position {
        match self {
            Self::Revision { position, .. }
            | Self::Observation { position, .. }
            | Self::Alternate { position, .. } => *position,
        }
    }
}

row! {
    struct RevisionSince {
        commit_seq: i64,
        input: String,
        number: i64,
        chain: i64,
        length: i64,
    }
    statements {
        const AFTER = "SELECT commit_seq, input, number, chain, length FROM source_revision
            WHERE commit_seq > ?1 ORDER BY commit_seq, input, number";
    }
}

row! {
    struct ObservationSince {
        commit_seq: i64,
        id: String,
        input: String,
        class: String,
    }
    statements {
        const AFTER = "SELECT commit_seq, id, input, class FROM source_observation
            WHERE commit_seq > ?1 ORDER BY commit_seq, id";
    }
}

row! {
    struct AlternateSince {
        commit_seq: i64,
        id: String,
    }
    statements {
        const AFTER = "SELECT commit_seq, id FROM source_alternate
            WHERE commit_seq > ?1 ORDER BY commit_seq, id";
    }
}

fn position(value: i64) -> rusqlite::Result<Position> {
    u64::try_from(value).map(Position).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Integer,
            Box::new(error),
        )
    })
}

impl Reader<'_> {
    /// Rows committed after `after`, in commit order, of the kinds asked
    /// for. Within one commit, revisions come before observations and
    /// alternates, each in its natural key's order.
    pub fn after(&self, after: Position, kinds: Followed) -> Result<Vec<ArchiveRow>> {
        let conn = self.conn()?;
        let after = i64::try_from(after.0)
            .map_err(|_| StoreError::Invalid("position exceeds the archive's range".into()))?;
        let mut rows = Vec::new();
        if kinds.revisions {
            let _ = conn.each(&RevisionSince::AFTER, [after], |row| {
                rows.push((
                    0,
                    ArchiveRow::Revision {
                        position: position(row.commit_seq)?,
                        input: row.input,
                        number: row.number,
                        chain: row.chain,
                        length: u64::try_from(row.length).unwrap_or_default(),
                    },
                ));
                Ok(ControlFlow::Continue(()))
            })?;
        }
        if kinds.observations {
            let _ = conn.each(&ObservationSince::AFTER, [after], |row| {
                rows.push((
                    1,
                    ArchiveRow::Observation {
                        position: position(row.commit_seq)?,
                        id: row.id,
                        input: row.input,
                        class: row.class,
                    },
                ));
                Ok(ControlFlow::Continue(()))
            })?;
        }
        if kinds.alternates {
            let _ = conn.each(&AlternateSince::AFTER, [after], |row| {
                rows.push((
                    2,
                    ArchiveRow::Alternate {
                        position: position(row.commit_seq)?,
                        id: row.id,
                    },
                ));
                Ok(ControlFlow::Continue(()))
            })?;
        }
        // Each kind is read in commit order; a stable sort by position keeps
        // that order within a commit.
        rows.sort_by_key(|(kind, row)| (row.position(), *kind));
        Ok(rows.into_iter().map(|(_, row)| row).collect())
    }
}

#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<RevisionSince>();
    check.row::<ObservationSince>();
    check.row::<AlternateSince>();
}
