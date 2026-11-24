//! The message's recorded source routes, without guessing a byte range or blob.
use super::*;
use polylogue_model::{
    Reference, Route,
    read::{ChainStep, TraceCard},
};

impl Reader<'_> {
    /// The concrete message containing a traceable item. Unsupported families
    /// remain outside this API rather than masquerading as missing records.
    fn trace_message(&self, reference: &Reference) -> Result<Option<MessageId>> {
        match reference {
            Reference::Message(id) | Reference::Block { message: id, .. } => {
                let conn = self.conn()?;
                let id = crate::chain_names::resolved_message(conn, id)?;
                let pk: Option<i64> = conn
                    .query_row(
                        "SELECT pk FROM message WHERE id=?1",
                        [id.to_string()],
                        |row| row.get(0),
                    )
                    .optional()?;
                let Some(pk) = pk else { return Ok(None) };
                if let Reference::Block { n, .. } = reference {
                    let present:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM block WHERE message_pk=?1 AND coalesce(ordinal,cast(idx AS TEXT))=?2)",params![pk,n.to_string()],|row|row.get(0))?;
                    if !present {
                        return Err(StoreError::ReferenceUnavailable {
                            reference: reference.to_string(),
                            reason: "the original block is unavailable".into(),
                        });
                    }
                }
                Ok(Some(id))
            }
            Reference::Span { container, .. } => {
                let message = self.trace_message(container)?;
                if message.is_some() && self.select_span_source(reference)?.is_none() {
                    return Ok(None);
                }
                Ok(message)
            }
            Reference::ToolCall(session, call) => {
                let conn = self.conn()?;
                let mut statement=conn.prepare("SELECT m.id FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk JOIN session s ON s.pk=t.session_pk WHERE s.id=?1 AND t.call_id=?2 ORDER BY t.pk LIMIT 2")?;
                let ids = statement
                    .query_map(params![session.to_string(), call], |row| {
                        row.get::<_, String>(0)
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                if ids.len() > 1 {
                    return Err(StoreError::Ambiguous {
                        reference: reference.to_string(),
                        candidates: ids.into_iter().map(|id| format!("message:{id}")).collect(),
                    });
                }
                ids.into_iter()
                    .next()
                    .map(|id| MessageId::parse(&id).map_err(StoreError::stored_data))
                    .transpose()
            }
            _ => Err(polylogue_model::Reason::TraceTarget {}.into()),
        }
    }

    /// Source ranges and their backing extents for a concrete recorded item.
    /// Checkpoint progress is reported as observed, not invented historical parser evidence.
    pub fn record_chain(&self, reference: &Reference) -> Result<Option<TraceCard>> {
        let Some(message) = self.trace_message(reference)? else {
            return Ok(None);
        };
        let conn = self.conn()?;
        let mut card = TraceCard {
            item: reference.clone(),
            chain: vec![ChainStep::Record {
                reference: reference.clone(),
            }],
            missing: vec![
                "quote origins".into(),
                "paste origins".into(),
                "file-line lineage".into(),
            ],
        };
        let message_ref = Reference::Message(message.clone());
        if &message_ref != reference {
            card.chain.push(ChainStep::Record {
                reference: message_ref,
            });
        }
        let message = crate::chain_names::resolved_message(conn, &message)?;
        let mut statement=conn.prepare("SELECT i.id,s.chain,s.source_start,s.source_end,s.route,(s.input_pk=m.input_pk AND s.chain=m.source_chain),p.parse_version,p.status,p.parsed_upto FROM message m JOIN shown_message_parse s ON s.message_pk=m.pk JOIN input_ref i ON i.pk=s.input_pk LEFT JOIN parse_state p ON p.input_pk=s.input_pk AND p.chain=s.chain WHERE m.id=?1 ORDER BY 6 DESC,i.id,s.chain")?;
        let sources = statement
            .query_map([message.to_string()], |row| {
                <(
                    String,
                    i64,
                    i64,
                    i64,
                    String,
                    bool,
                    Option<String>,
                    Option<String>,
                    Option<u64>,
                )>::try_from(row)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if sources.is_empty() {
            card.chain.push(ChainStep::Unavailable {
                reason: "record source is unavailable".into(),
            });
        }
        for (input, chain, start, end, route, shown, version, status, parsed_upto) in sources {
            let start = u64::try_from(start).map_err(StoreError::stored_data)?;
            let end = u64::try_from(end).map_err(StoreError::stored_data)?;
            if end < start {
                return Err(StoreError::stored_data(
                    "record source byte end precedes its start",
                ));
            }
            let route = Route::from_name(&route)
                .ok_or_else(|| StoreError::stored_data("unknown source route"))?;
            card.chain.push(ChainStep::InputRecord {
                input: input.clone(),
                chain,
                start,
                end,
                route,
                shown,
            });
            if end == start {
                card.chain.push(ChainStep::Unavailable {
                    reason: format!("record byte end is not recorded for {input} chain {chain}"),
                });
            } else {
                self.check_bytes_present(&input, chain, start..end)?;
                let mut extents=conn.prepare("SELECT e.blob,b.size,e.offset,e.length,e.blob_offset FROM source_extent e JOIN source_blob b ON b.hash=e.blob WHERE e.input=?1 AND e.chain=?2 AND e.offset<?3 AND e.offset+e.length>?4 AND e.offset>=coalesce((SELECT max(prior.offset) FROM source_extent prior WHERE prior.input=?1 AND prior.chain=?2 AND prior.offset<=?4),?4) ORDER BY e.offset")?;
                let mut rows = extents.query(params![input, chain, end, start])?;
                while let Some(row) = rows.next()? {
                    card.chain.push(ChainStep::StoredBytes {
                        blob: polylogue_model::BlobHash(row.hash(0)?),
                        size: row.get(1)?,
                        offset: row.get(2)?,
                        length: row.get(3)?,
                        blob_offset: row.get(4)?,
                    });
                }
            }
            if let (Some(parse_version), Some(status), Some(parsed_upto)) =
                (version, status, parsed_upto)
                && !parse_version.is_empty()
            {
                if end > start && parsed_upto < end {
                    card.chain.push(ChainStep::Unavailable{reason:"the current reader checkpoint does not cover this record; its original reader version is unavailable".into()});
                }
                card.chain.push(ChainStep::ReaderVersion {
                    parse_version,
                    status,
                    parsed_upto,
                });
            } else {
                card.chain.push(ChainStep::Unavailable {
                    reason: format!("reader version is unavailable for {input} chain {chain}"),
                });
            }
        }
        Ok(Some(card))
    }
}

#[cfg(test)]
mod tests;
