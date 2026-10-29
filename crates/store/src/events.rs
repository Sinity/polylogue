//! Selected events keep the local input they were read from. Citations in an
//! exported file name the sender's inputs and never decide local ones.
use crate::{Reader, Result, StoreError};
use polylogue_model::bundle::{EventSourceClaim, ImportedEventIdentity, InputEventRecord};
use polylogue_model::export::EventExport;
use polylogue_model::{Event, SessionId};
use rusqlite::OptionalExtension;

pub(crate) fn canonical_event(fields: &[u8], data: &[u8]) -> Result<Event> {
    let mut event: Event = crate::encoding::unpack_json(fields)?;
    event.data = crate::encoding::unpack_json(data)?;
    Ok(event)
}

impl Reader<'_> {
    /// The canonical selected event, with all retained claims and its actual local source.
    pub fn event_record(&self, id: &str) -> Result<Option<InputEventRecord>> {
        let stored: Option<(i64, Vec<u8>, Vec<u8>)> = self
            .conn()?
            .query_row("SELECT pk,fields,data FROM event WHERE id=?1", [id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .optional()?;
        let Some((pk, fields, data)) = stored else {
            return Ok(None);
        };
        let event = canonical_event(&fields, &data)?;
        let mut sources = Vec::new();
        let mut statement = self.conn()?.prepare_cached(
            "SELECT i.id,i.format,s.chain,s.source_start,s.source_end,s.imported_provenance
             FROM event_source s JOIN input_ref i ON i.pk=s.input_pk WHERE s.event_pk=?1 ORDER BY i.id,s.chain")?;
        let rows = statement.query_map([pk], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, Option<i64>>(3)?,
                r.get::<_, Option<i64>>(4)?,
                r.get::<_, String>(5)?,
            ))
        })?;
        for row in rows {
            let (input, source, source_chain, start, end, original) = row?;
            let coordinate = |value: Option<i64>| {
                value
                    .map(|value| {
                        u64::try_from(value).map_err(|_| {
                            StoreError::Invalid("negative event source coordinate".into())
                        })
                    })
                    .transpose()
            };
            sources.push(EventSourceClaim {
                input,
                source,
                source_chain,
                source_start: coordinate(start)?,
                source_end: coordinate(end)?,
            });
            sources.extend(serde_json::from_str::<Vec<EventSourceClaim>>(&original)?);
        }
        let mut identity = ImportedEventIdentity {
            id: id.to_owned(),
            sources,
        };
        identity.canonicalize();
        identity
            .validate()
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        Ok(Some(InputEventRecord {
            reference: format!("event:{id}"),
            commit: u64::try_from(self.change_position()?.index)
                .map_err(|_| StoreError::Invalid("negative event commit".into()))?,
            identity,
            event,
        }))
    }

    /// Events in the session's reading order, preserving identities and claims.
    pub fn event_records(&self, session: &SessionId) -> Result<Vec<InputEventRecord>> {
        let ids = self.conn()?.prepare_cached("SELECT e.id FROM event e JOIN session s ON s.pk=e.session_pk WHERE s.id=?1 ORDER BY e.seq,e.id")?
            .query_map([session.to_string()],|row| row.get::<_,String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
        ids.into_iter()
            .map(|id| {
                self.event_record(&id)?
                    .ok_or_else(|| StoreError::Invalid("selected event disappeared".into()))
            })
            .collect()
    }

    pub fn event_export(&self, id: &str) -> Result<Option<EventExport>> {
        let Some(record) = self.event_record(id)? else {
            return Ok(None);
        };
        let local: (String,String,i64,Option<i64>,Option<i64>) = self.conn()?.query_row(
            "SELECT i.id,i.format,e.source_chain,e.source_start,e.source_end FROM event e JOIN input_ref i ON i.pk=e.input_pk WHERE e.id=?1",
            [id],|r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)))?;
        Ok(Some(EventExport {
            id: id.to_owned(),
            input: local.0,
            source: local.1,
            source_chain: local.2,
            source_start: local.3.map(|n| n as u64),
            source_end: local.4.map(|n| n as u64),
            event: record.event,
        }))
    }

    pub fn event_sources(&self, id: &str) -> Result<Vec<polylogue_model::Json>> {
        self.query("SELECT i.id AS input,i.format AS source,s.chain,s.source_start,s.source_end
            FROM event_source s JOIN event e ON e.pk=s.event_pk JOIN input_ref i ON i.pk=s.input_pk WHERE e.id=?1 ORDER BY i.id,s.chain",&[crate::SqlValue::Text(id.into())])
    }
}
