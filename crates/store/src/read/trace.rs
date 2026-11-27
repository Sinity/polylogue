//! The message's recorded source routes, without guessing a byte range or blob.
use super::*;
use polylogue_model::{
    Reference, Route,
    bundle::fragments::seek::SourceTextPiece,
    read::{ChainStep, Source, TraceCard},
};

row! {
    /// A reading of a message: its input and byte range, whether it is the
    /// shown one, and its chain's parse standing.
    struct TraceSource {
        input: String,
        chain: i64,
        source_start: u64,
        source_end: u64,
        route: Route,
        shown: bool,
        stored: u64,
    }
    statements {
        const OF_MESSAGE = "SELECT i.id AS input, s.chain, s.source_start, s.source_start + s.source_length AS source_end, s.route,
            (s.input_pk=m.input_pk AND s.chain=m.source_chain) AS shown,
            (SELECT max(r.length) FROM source_revision r WHERE r.input=i.id AND r.chain=s.chain) AS stored
            FROM message m JOIN assembled_message_parse s ON s.message_pk=m.pk JOIN input_ref i ON i.pk=s.input_pk
            WHERE m.id=?1 ORDER BY 6 DESC, i.id, s.chain";
    }
}

row! {
    /// A stored extent under a byte range of a chain, and its blob's size.
    struct TraceExtent {
        blob: [u8; 32],
        size: u64,
        offset: u64,
        length: u64,
        blob_offset: u64,
    }
    statements {
        const COVERING = "SELECT e.blob, b.size, e.offset, e.length, e.blob_offset
            FROM source_extent e JOIN source_blob b ON b.hash=e.blob
            WHERE e.input=?1 AND e.chain=?2 AND e.offset<?3 AND e.offset+e.length>?4
              AND e.offset>=coalesce((SELECT max(prior.offset) FROM source_extent prior WHERE prior.input=?1 AND prior.chain=?2 AND prior.offset<=?4),?4)
            ORDER BY e.offset";
    }
}

row! {
    struct CallSource {
        message: MessageId,
        ordinal: String,
    }
    statements {
        const OF_CALL = "SELECT m.id AS message, coalesce(b.source_position,cast(b.position AS TEXT)) AS ordinal
            FROM tool_call t JOIN block b ON b.pk=t.call_block_pk
            JOIN message m ON m.pk=b.message_pk JOIN session s ON s.pk=t.session_pk
            WHERE s.id=?1 AND t.call_id=?2 ORDER BY t.pk LIMIT 2";
    }
}

impl Reader<'_> {
    /// A selected item and the original coordinates at which it is read.
    pub fn reading_item(&self, reference: &Reference) -> Result<Option<Source>> {
        let (message, block) = match reference {
            Reference::ToolCall(session, call) => {
                let Some(row) = self.call_source(session, call)? else {
                    return Ok(None);
                };
                let block = row
                    .ordinal
                    .parse()
                    .map_err(|_| StoreError::internal("invalid original block ordinal"))?;
                return Ok(Some(Source {
                    reference: reference.to_string(),
                    ..Source::block(&row.message, session, block, false)
                }));
            }
            Reference::Block { n, .. } => (self.trace_message(reference)?, Some(*n)),
            Reference::Span { container, .. }
                if matches!(
                    container.as_ref(),
                    Reference::Message(_) | Reference::Block { .. }
                ) =>
            {
                let Some(message) = self.trace_message(container)? else {
                    return Ok(None);
                };
                let Some(span) = self.select_span_source(reference)? else {
                    return Ok(None);
                };
                let block = match span.seek().start.piece {
                    SourceTextPiece::Body { body } => Some(body.block),
                    SourceTextPiece::Separator { before, .. } => Some(before.block),
                    SourceTextPiece::Empty => None,
                };
                (Some(message), block)
            }
            _ => return Ok(None),
        };
        let Some(message) = message else {
            return Ok(None);
        };
        let Some(session) = self.message_session(&message)? else {
            return Ok(None);
        };
        Ok(Some(Source {
            reference: reference.to_string(),
            message: Some(message),
            block,
            source_session: Some(session),
            inherited: false,
        }))
    }

    fn call_source(&self, session: &SessionId, call: &str) -> Result<Option<CallSource>> {
        let conn = self.conn()?;
        let rows = conn.all(&CallSource::OF_CALL, params![session.to_string(), call])?;
        if rows.len() > 1 {
            return Err(StoreError::Ambiguous {
                reference: Reference::ToolCall(session.clone(), call.into()).to_string(),
                candidates: rows
                    .into_iter()
                    .map(|row: CallSource| format!("block:{}#{}", row.message, row.ordinal))
                    .collect(),
            });
        }
        Ok(rows.into_iter().next())
    }

    /// The concrete message containing a traceable item. Unsupported families
    /// remain outside this API rather than masquerading as missing records.
    fn trace_message(&self, reference: &Reference) -> Result<Option<MessageId>> {
        match reference {
            Reference::Message(id) | Reference::Block { message: id, .. } => {
                let conn = self.conn()?;
                let id = crate::chain_names::resolved_message(conn, id)?;
                let pk: Option<MessagePk> =
                    conn.scalar_optional("SELECT pk FROM message WHERE id=?1", [id.to_string()])?;
                let Some(pk) = pk else { return Ok(None) };
                if let Reference::Block { n, .. } = reference {
                    let present = conn.exists("SELECT EXISTS(SELECT 1 FROM block WHERE message_pk=?1 AND coalesce(source_position,cast(position AS TEXT))=?2)", params![pk, n.to_string()])?;
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
                Ok(self.call_source(session, call)?.map(|row| row.message))
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
        let sources = conn.all(&TraceSource::OF_MESSAGE, [message.to_string()])?;
        if sources.is_empty() {
            card.chain.push(ChainStep::Unavailable {
                reason: "record source is unavailable".into(),
            });
        }
        for TraceSource {
            input,
            chain,
            source_start: start,
            source_end: end,
            route,
            shown,
            stored,
        } in sources
        {
            let state = crate::parse_state::state_of_input(conn, &input, chain)?;
            if end < start {
                return Err(StoreError::stored_data(
                    "record source byte end precedes its start",
                ));
            }
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
                for extent in conn.all(&TraceExtent::COVERING, params![input, chain, end, start])? {
                    card.chain.push(ChainStep::StoredBytes {
                        blob: polylogue_model::BlobHash(extent.blob),
                        size: extent.size,
                        offset: extent.offset,
                        length: extent.length,
                        blob_offset: extent.blob_offset,
                    });
                }
            }
            if let Some(parse_version) = state.version().map(str::to_owned) {
                let parsed_upto = state.parsed_to();
                // Shown under its own parser version: the trace names the
                // version that read the record.
                let status = polylogue_model::chain::status(
                    &state,
                    &polylogue_model::chain::ChainFacts {
                        stored,
                        repeated_by: None,
                        own_from: 0,
                        parser: Some(parse_version.clone()),
                        finished: false,
                    },
                )
                .name()
                .as_str()
                .to_owned();
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

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<TraceSource>();
    check.row::<TraceExtent>();
    check.row::<CallSource>();
}

#[cfg(test)]
mod tests;
