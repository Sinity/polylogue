use super::Source;
use crate::Result;
use crate::capture_read::{ConfirmedSource, SourceSelector, SourceStatus};
use crate::input_rows::{ObservationRow, Seen};
use rusqlite::{Connection, OptionalExtension, params};

/// An observation's class, revision, chain and length as stored.
type SeenColumns = (String, Option<i64>, Option<i64>, Option<i64>);

impl Source {
    /// The input last observed at the selected location, by its exact bytes.
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
        let found: Option<(String, ObservationRow, SeenColumns)> = conn.query_row(
            &source.sql("SELECT o.input,o.id,o.machine,m.host,o.path_bytes,o.class,o.revision,o.chain,o.length,o.device,o.inode,o.size,o.mtime_ns,o.ctime_ns,o.polylogue_version,o.seen_at,o.details_json FROM @source.observation o JOIN @source.input i ON i.id=o.input JOIN @source.machine m ON m.id=o.machine WHERE i.origin=?1 AND o.machine=?2 AND o.path_bytes=?3 AND (?4 IS NULL OR o.commit_seq<=?4) ORDER BY o.id DESC LIMIT 1"),
            params![selector.origin,selector.machine,selector.path_bytes,through],
            |row| Ok((row.get(0)?,ObservationRow {
                id:row.get(1)?,machine:row.get(2)?,host:row.get(3)?,path_bytes:row.get(4)?,seen:Seen::Deleted,device:row.get(9)?,inode:row.get(10)?,size:row.get(11)?,mtime_ns:row.get(12)?,ctime_ns:row.get(13)?,polylogue_version:row.get(14)?,seen_at:row.get(15)?,details_json:row.get(16)?,container:None,
            },(row.get(5)?,row.get(6)?,row.get(7)?,row.get(8)?))),
        ).optional()?;
        let by_path_bytes = found
            .map(
                |(input, mut observation, (class, revision, chain, length))| {
                    observation.seen = Seen::from_columns(&class, revision, chain, length)?;
                    Ok::<_, crate::StoreError>((input, observation))
                },
            )
            .transpose()?;
        let mut confirmed =
            by_path_bytes.map(|(input, observation)| ConfirmedSource { input, observation });
        if let Some(confirmed) = &mut confirmed {
            confirmed.observation.container = conn.query_row(
                &source.sql("SELECT container_input,member_position,kind,path_bytes FROM @source.observation_container WHERE observation_id=?1"),
                [&confirmed.observation.id],
                |row| Ok(crate::InContainer { input:row.get(0)?,member_index:u64::try_from(row.get::<_,i64>(1)?).map_err(|error| rusqlite::Error::FromSqlConversionFailure(1,rusqlite::types::Type::Integer,Box::new(error)))?,kind:row.get(2)?,path_bytes:row.get(3)? }),
            ).optional()?;
        }
        Ok((SourceStatus::Ready, confirmed))
    }
}
