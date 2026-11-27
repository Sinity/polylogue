//! Selected events keep the local input they were read from. Citations in an
//! exported file name the sender's inputs and never decide local ones.
use crate::{Reader, Result, StoreError};
use polylogue_model::bundle::{ImportedEventIdentity, InputEventRecord, StatedSource};
use polylogue_model::{Event, SessionId};

/// The columns [`shown_event`] reads from `event e LEFT JOIN session s ON
/// s.pk=e.session_pk`.
pub(crate) const SHOWN_EVENT_COLUMNS: &str =
    "e.content_group,e.content_start,e.content_length,e.data_length,s.id,e.after_message_id";

/// A shown event from the [`SHOWN_EVENT_COLUMNS`] of `row` starting at `at`:
/// its reading's stored fields and data, with its session and the message it
/// follows as the row shows them (a copy's event is shown under the names its
/// session gives).
pub(crate) fn shown_event(
    conn: &rusqlite::Connection,
    reader: &mut crate::content::Reader,
    row: &rusqlite::Row<'_>,
    at: usize,
) -> Result<Event> {
    use crate::encoding::Columns as _;
    let mut event =
        crate::writer::readings::read_event(conn, reader, crate::content::Piece::of(row, at)?)?;
    event.session = row.optional_id(at + 4)?;
    event.after = row.optional_id(at + 5)?;
    Ok(event)
}

struct StoredEvent {
    pk: i64,
    event: Event,
    input: String,
    chain: i64,
    end: Option<i64>,
}

impl Reader<'_> {
    /// The selected event, with every kept source citation and its local source.
    pub fn event_record(&self, id: &str) -> Result<Option<InputEventRecord>> {
        let conn = self.conn()?;
        let mut reader = crate::content::Reader::default();
        let stored: Option<StoredEvent> = conn
            .prepare_cached(&format!(
                "SELECT e.pk,i.id,e.source_chain,e.source_end,{SHOWN_EVENT_COLUMNS}
                 FROM event e JOIN input_ref i ON i.pk=e.input_pk LEFT JOIN session s ON s.pk=e.session_pk
                 WHERE e.id=?1"
            ))?
            .query_and_then([id], |r| {
                Ok::<_, StoreError>(StoredEvent {
                    pk: r.get(0)?,
                    input: r.get(1)?,
                    chain: r.get(2)?,
                    end: r.get(3)?,
                    event: shown_event(conn, &mut reader, r, 4)?,
                })
            })?
            .next()
            .transpose()?;
        let Some(StoredEvent {
            pk,
            mut event,
            input,
            chain,
            end,
        }) = stored
        else {
            return Ok(None);
        };
        // What the reading left in the stored bytes is shown from them; an
        // event still reads while they are not at hand (docs/archive.md,
        // content), with the value's place where the value would be.
        if let Err(error) = self.resolve_stored_value(&input, chain, &mut event.data) {
            crate::log!(
                warning,
                "event {id}: a value left in the stored bytes is not shown: {error}"
            );
        }
        let readings = crate::writer::readings::event_readings(self.conn()?, pk)?;
        let selected = readings
            .iter()
            .find(|r| r.input == input && r.chain == chain);
        let through = end
            .map(u64::try_from)
            .transpose()
            .map_err(|_| StoreError::stored(5, "negative event source coordinate"))?;
        // The commit of the extents under the event's byte range.
        let commit = self
            .source_commit(&input, chain, through)?
            .ok_or_else(|| StoreError::stored(6, "selected event has no stored commit position"))?;
        let mut sources = Vec::new();
        let mut rows: Vec<_> = readings
            .iter()
            .filter(|r| selected.is_some_and(|s| s.hash == r.hash && s.fields == r.fields))
            .map(|r| {
                Ok::<_, StoreError>((
                    r.input.clone(),
                    r.format.clone(),
                    r.chain,
                    r.start,
                    r.end,
                    r.imported.clone(),
                ))
            })
            .collect();
        rows.sort_by(|a, b| match (a, b) {
            (Ok(a), Ok(b)) => (&a.0, a.2).cmp(&(&b.0, b.2)),
            _ => std::cmp::Ordering::Equal,
        });
        for row in rows {
            let (input, source, source_chain, start, end, original) = row?;
            sources.push(StatedSource {
                input,
                source,
                source_chain,
                source_start: start,
                source_end: end,
            });
            sources.extend(serde_json::from_str::<Vec<StatedSource>>(&original)?);
        }
        let mut identity = ImportedEventIdentity {
            id: id.to_owned(),
            sources,
        };
        identity.canonicalize();
        identity
            .validate()
            .map_err(|error| StoreError::stored(2, error.to_string()))?;
        Ok(Some(InputEventRecord {
            reference: format!("event:{id}"),
            commit,
            identity,
            event,
        }))
    }

    /// Seek within the physical session's stored event order before decoding.
    pub fn visit_event_records(
        &self,
        session: &SessionId,
        ordinal: u64,
        mut visit: impl FnMut(InputEventRecord) -> std::ops::ControlFlow<()>,
    ) -> Result<()> {
        let Ok(ordinal) = i64::try_from(ordinal) else {
            return Ok(());
        };
        let mut statement=self.conn()?.prepare_cached("SELECT e.id FROM event e JOIN session s ON s.pk=e.session_pk WHERE s.id=?1 ORDER BY e.position,e.id LIMIT -1 OFFSET ?2")?;
        let mut rows = statement.query(rusqlite::params![session.to_string(), ordinal])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let record = self
                .event_record(&id)?
                .ok_or_else(|| StoreError::stored(0, "selected event disappeared"))?;
            if visit(record).is_break() {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
