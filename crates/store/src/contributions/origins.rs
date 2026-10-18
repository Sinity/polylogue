//! The set of source byte ranges each message candidate was parsed from.
//! Ranges are stored as given; the source mapper decides what they mean.
use super::{
    blocks::{self, ProjectionStep},
    writer::{self, Build},
};
use crate::{Error, Result, Writer, db, parser_state};
use rusqlite::{Transaction, params};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceRange {
    pub start: u64,
    pub end: u64,
}
fn checked(range: SourceRange, parsed_upto: u64) -> Result<(i64, i64)> {
    if range.start >= range.end || range.end > parsed_upto {
        return Err(Error::Invariant(
            "message source range is empty or past the parsed end".into(),
        ));
    }
    Ok((writer::integer(range.start)?, writer::integer(range.end)?))
}
pub struct SourceRangeEntry {
    pub range: SourceRange,
    pub expected_sequence: u64,
    pub next_cursor: Vec<u8>,
}
impl SourceRangeEntry {
    pub fn retained_bytes(&self) -> Result<usize> {
        self.next_cursor
            .len()
            .checked_add(std::mem::size_of::<Self>())
            .ok_or(Error::Range)
    }
    fn step(&self) -> ProjectionStep<'_> {
        ProjectionStep {
            expected_sequence: self.expected_sequence,
            next_cursor: &self.next_cursor,
        }
    }
}
fn write_in(
    tx: &Transaction<'_>,
    path: &Path,
    build: &Build,
    range: SourceRange,
    step: &ProjectionStep<'_>,
    allowance: usize,
) -> Result<()> {
    let remaining = allowance.checked_sub(16).ok_or(Error::ParserAdmission {
        resource: "message source metadata bytes",
        required: 16,
    })?;
    if let Err(error) = blocks::admit(step, remaining) {
        return Err(match error {
            Error::ParserAdmission { required, .. } => Error::ParserAdmission {
                resource: "message source metadata bytes",
                required: required.checked_add(16).ok_or(Error::Range)?,
            },
            other => other,
        });
    }
    writer::check(tx, path, build)?;
    let head = writer::source_head(tx, path, build.predecessor())?;
    let (start, end) = checked(range, head.parsed_upto)?;
    let pending: bool = db(path, || {
        tx.query_row("SELECT active_value_pk IS NOT NULL OR active_origin_value_pk IS NOT NULL OR active_block_pk IS NOT NULL FROM message_build WHERE pk=?1",[build.pk],|r|r.get(0))
    })?;
    if pending {
        return Err(Error::MessagePreparationPending);
    }
    db(path, || {
        tx.execute("INSERT INTO message_source(candidate_pk,message_pk,session_pk,input_pk,chain,source_start,source_end) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(candidate_pk,input_pk,chain,source_start,source_end) DO NOTHING",params![build.candidate,build.message,build.session,build.source.input_pk,build.source.chain,start,end])
    })?;
    blocks::commit_cursor(tx, path, build, step)?;
    Ok(())
}
impl Writer {
    /// Add one range to this candidate's source set and advance its cursor in
    /// the same transaction. Input and chain come from the build.
    pub fn add_message_source(
        &mut self,
        build: &Build,
        range: SourceRange,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        write_in(&tx, &path, build, range, step, allowance)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    pub fn add_message_source_run(
        &mut self,
        build: &Build,
        entries: &[SourceRangeEntry],
        allowance: usize,
    ) -> Result<()> {
        if entries.is_empty() {
            return Err(Error::Invariant("empty source range run".into()));
        }
        let mut bytes = 0usize;
        for entry in entries {
            bytes = bytes
                .checked_add(entry.retained_bytes()?)
                .ok_or(Error::Range)?;
            if bytes > allowance {
                return Err(Error::ParserAdmission {
                    resource: "source range run bytes",
                    required: bytes as u64,
                });
            }
        }
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        for entry in entries {
            write_in(&tx, &path, build, entry.range, &entry.step(), allowance)?;
        }
        db(&path, || tx.commit())?;
        Ok(())
    }
}
impl parser_state::Snapshot {
    /// The build's input id, refused if longer than `allowance`. The caller
    /// compares it with the input it expects; this lookup does not validate it.
    pub fn message_source_input_id(&self, build: &Build, allowance: usize) -> Result<String> {
        if self.head.token != *build.predecessor() {
            return Err(Error::StaleMessage);
        }
        writer::check_completed(&self.db, &self.path, build)?;
        let allowance = writer::integer(u64::try_from(allowance).map_err(|_| Error::Range)?)?;
        let (required, value): (i64, Option<String>) = db(&self.path, || {
            self.db.query_row("SELECT length(CAST(id AS BLOB)),CASE WHEN length(CAST(id AS BLOB))<=?2 THEN id ELSE NULL END FROM input_ref WHERE pk=?1",params![build.source.input_pk,allowance],|r|Ok((r.get(0)?,r.get(1)?)))
        })?;
        let required = u64::try_from(required).map_err(|_| Error::Range)?;
        value.ok_or(Error::ParserAdmission {
            resource: "message source input identity bytes",
            required,
        })
    }
    /// Whether this snapshot records `range` for the build. A writer must
    /// recheck its own state before acting on the answer.
    pub fn has_message_source(&self, build: &Build, range: SourceRange) -> Result<bool> {
        if self.head.token != *build.predecessor() {
            return Err(Error::StaleMessage);
        }
        writer::check(&self.db, &self.path, build)?;
        let (start, end) = checked(range, self.head.parsed_upto)?;
        db(&self.path, || {
            self.db.query_row("SELECT EXISTS(SELECT 1 FROM message_source WHERE candidate_pk=?1 AND input_pk=?2 AND chain=?3 AND source_start=?4 AND source_end=?5)",params![build.candidate,build.source.input_pk,build.source.chain,start,end],|r|r.get(0))
        })
    }
}

#[cfg(test)]
mod tests;
