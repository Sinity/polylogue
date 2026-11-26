use super::Source;
use crate::ConnExt as _;
use crate::Result;
use crate::capture_read::{ConfirmedSource, SourceSelector, SourceStatus};
use crate::input_rows::ObservationRow;
use rusqlite::{Connection, OptionalExtension, params};

impl Source {
    /// Matching by display path diagnoses old observations without path bytes; it
    /// never selects an identity. A later observation with path bytes replaces
    /// that uncertainty.
    pub(crate) fn confirmed_source(
        conn: &Connection,
        selector: &SourceSelector,
    ) -> Result<(SourceStatus, Option<ConfirmedSource>)> {
        Self::confirmed_source_at(conn, selector, None)
    }

    pub(crate) fn confirmed_source_at(
        conn: &Connection,
        selector: &SourceSelector,
        through: Option<u64>,
    ) -> Result<(SourceStatus, Option<ConfirmedSource>)> {
        let through = through.map(i64::try_from).transpose().map_err(|_| {
            crate::StoreError::internal("source commit position exceeds archive address space")
        })?;
        let source = Self::reading(conn)?;
        let by_path_bytes: Option<(i64, String, ObservationRow)> = conn.query_row(
            &source.sql("SELECT o.rowid,o.input,o.id,o.machine,m.host,o.path,o.path_bytes,o.class,o.revision,o.chain,o.length,o.device,o.inode,o.size,o.mtime_ns,o.ctime_ns,o.polylogue_version,o.seen_at,o.details_json FROM @source.observation o JOIN @source.input i ON i.id=o.input JOIN @source.machine m ON m.id=o.machine WHERE i.format=?1 AND o.machine=?2 AND o.path_bytes=?3 AND (?4 IS NULL OR o.commit_seq<=?4) ORDER BY o.rowid DESC LIMIT 1"),
            params![selector.origin,selector.machine,selector.path_bytes,through],
            |row| Ok((row.get(0)?,row.get(1)?,ObservationRow {
                id:row.get(2)?,machine:row.get(3)?,host:row.get(4)?,path:row.get(5)?,path_bytes:row.get(6)?,class:row.get(7)?,revision:row.get(8)?,chain:row.get(9)?,length:row.get(10)?,device:row.get(11)?,inode:row.get(12)?,size:row.get(13)?,mtime_ns:row.get(14)?,ctime_ns:row.get(15)?,polylogue_version:row.get(16)?,seen_at:row.get(17)?,details_json:row.get(18)?,container:None,
            })),
        ).optional()?;
        let after = by_path_bytes.as_ref().map_or(0, |row| row.0);
        let unknown: i64 = conn.scalar(&source.sql("SELECT count(DISTINCT o.input) FROM @source.observation o JOIN @source.input i ON i.id=o.input WHERE i.format=?1 AND o.machine=?2 AND o.path_bytes IS NULL AND o.path=?3 AND o.rowid>?4 AND (?5 IS NULL OR o.commit_seq<=?5)"), params![selector.origin,selector.machine,String::from_utf8_lossy(&selector.path_bytes),after,through])?;
        if unknown > 0 {
            return Ok((
                if unknown > 1 {
                    SourceStatus::Ambiguous
                } else {
                    SourceStatus::MissingEvidence
                },
                None,
            ));
        }
        let mut confirmed =
            by_path_bytes.map(|(_, input, observation)| ConfirmedSource { input, observation });
        if let Some(confirmed) = &mut confirmed {
            confirmed.observation.container = conn.query_row(
                &source.sql("SELECT container_input,member_index,kind,path_bytes FROM @source.observation_container WHERE observation_id=?1"),
                [&confirmed.observation.id],
                |row| Ok(crate::InContainer { input:row.get(0)?,member_index:u64::try_from(row.get::<_,i64>(1)?).map_err(|error| rusqlite::Error::FromSqlConversionFailure(1,rusqlite::types::Type::Integer,Box::new(error)))?,kind:row.get(2)?,path_bytes:row.get(3)? }),
            ).optional()?;
        }
        Ok((SourceStatus::Ready, confirmed))
    }
}
