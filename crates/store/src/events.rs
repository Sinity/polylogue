//! Selected observations retain their actual local Input citation. Exported
//! sender citations are not used as local acquisition authority.
use crate::{Reader, Result, StoreError};
use polylogue_model::export::EventExport;
use polylogue_model::{Event, MessageId, SessionId, Timestamp};
use rusqlite::OptionalExtension;

struct Stored {
    id: String,
    input: String,
    source: String,
    chain: i64,
    start: Option<i64>,
    end: Option<i64>,
    session: Option<String>,
    kind: String,
    at: Option<i64>,
    after: Option<String>,
    data: Vec<u8>,
}
impl Reader<'_> {
    pub fn event_export(&self, id: &str) -> Result<Option<EventExport>> {
        let row=self.conn()?.query_row("SELECT e.id,i.id,i.format,e.source_chain,e.source_start,e.source_end,s.id,e.kind,e.at,m.id,e.data
            FROM current_event e JOIN input_ref i ON i.pk=e.input_pk
            LEFT JOIN session s ON s.pk=e.session_pk LEFT JOIN current_message m ON m.pk=e.after_message_pk
            WHERE e.id=?1",[id],|r|Ok(Stored {
                id:r.get(0)?,input:r.get(1)?,source:r.get(2)?,chain:r.get(3)?,start:r.get(4)?,end:r.get(5)?,
                session:r.get(6)?,kind:r.get(7)?,at:r.get(8)?,after:r.get(9)?,data:r.get(10)?,
            })).optional()?;
        row.map(|r| {
            let coordinate = |v: Option<i64>| {
                v.map(|n| {
                    u64::try_from(n)
                        .map_err(|_| StoreError::Invalid("invalid event source coordinate".into()))
                })
                .transpose()
            };
            let start = coordinate(r.start)?;
            let end = coordinate(r.end)?;
            let event = Event {
                session: r
                    .session
                    .map(|s| {
                        SessionId::parse(&s)
                            .map_err(|_| StoreError::Invalid("invalid stored event session".into()))
                    })
                    .transpose()?,
                kind: r.kind,
                at: r.at.and_then(Timestamp::from_unix_ms),
                after: r
                    .after
                    .map(|s| {
                        MessageId::parse(&s)
                            .map_err(|_| StoreError::Invalid("invalid stored event anchor".into()))
                    })
                    .transpose()?,
                range: start.zip(end),
                data: crate::encoding::unpack_json(&r.data)?,
            };
            Ok(EventExport {
                id: r.id,
                input: r.input,
                source: r.source,
                source_chain: r.chain,
                source_start: start,
                source_end: end,
                event,
            })
        })
        .transpose()
    }

    pub fn event_sources(&self, id: &str) -> Result<Vec<polylogue_model::Json>> {
        self.query("SELECT i.id AS input,i.format AS source,s.chain,s.source_start,s.source_end,s.parser_version,s.current
            FROM event_source s JOIN input_ref i ON i.pk=s.input_pk WHERE s.event_id=?1
            ORDER BY i.id,s.chain,s.source_start,s.source_snapshot",&[crate::SqlValue::Text(id.into())])
    }
}
