//! Selected events keep the local input they were read from. Citations in an
//! exported file name the sender's inputs and never decide local ones.
use crate::{Reader, Result, StoreError};
use polylogue_model::bundle::{ImportedEventIdentity, InputEventRecord, StatedSource};
use polylogue_model::{Event, SessionId};
use rusqlite::OptionalExtension;

/// An event's metadata and its payload are stored as two separate blobs; the
/// payload is decoded into `event.data`.
pub(crate) fn decode_event(fields: &[u8], data: &[u8]) -> Result<Event> {
    let mut event: Event = crate::encoding::unpack_json(fields)?;
    event.data = crate::encoding::unpack_json(data)?;
    Ok(event)
}

struct StoredEvent {
    pk: i64,
    fields: Vec<u8>,
    data: Vec<u8>,
    input: String,
    chain: i64,
    end: Option<i64>,
    committed: Option<i64>,
}

impl Reader<'_> {
    /// The selected event, with every kept source citation and its local source.
    pub fn event_record(&self, id: &str) -> Result<Option<InputEventRecord>> {
        let stored: Option<StoredEvent> = self
            .conn()?
            .query_row("SELECT e.pk,e.fields,e.data,i.id,e.source_chain,e.source_end,NULL FROM event e JOIN input_ref i ON i.pk=e.input_pk
                WHERE e.id=?1", [id], |r| {
                Ok(StoredEvent{pk:r.get(0)?,fields:r.get(1)?,data:r.get(2)?,input:r.get(3)?,chain:r.get(4)?,end:r.get(5)?,committed:r.get(6)?})
            })
            .optional()?;
        let Some(StoredEvent {
            pk,
            fields,
            data,
            input,
            chain,
            end,
            committed,
        }) = stored
        else {
            return Ok(None);
        };
        let event = decode_event(&fields, &data)?;
        let readings = crate::writer::readings::event_readings(self.conn()?, pk)?;
        let selected = readings
            .iter()
            .find(|r| r.input == input && r.chain == chain);
        let committed = committed.or(selected.and_then(|r| r.committed));
        let through = end
            .map(u64::try_from)
            .transpose()
            .map_err(|_| StoreError::stored(5, "negative event source coordinate"))?;
        let commit = match committed {
            Some(value) => u64::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| StoreError::stored(6, "invalid event commit position"))?,
            None => self.source_commit(&input, chain, through)?.ok_or_else(|| {
                StoreError::stored(6, "selected event has no stored commit position")
            })?,
        };
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
        let mut statement=self.conn()?.prepare_cached("SELECT e.id FROM event e JOIN session s ON s.pk=e.session_pk WHERE s.id=?1 ORDER BY e.seq,e.id LIMIT -1 OFFSET ?2")?;
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
