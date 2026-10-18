//! Source ranges cited by a message's candidates.
use super::{SharedRead, locked, message::MessageView, metadata_admission};
use crate::{Error, Result, db};
use rusqlite::{Connection, OptionalExtension, params};
use std::sync::Arc;

/// A stored source range of a message. It says nothing about whether the
/// bytes are still available or excised; the archive answers that.
pub struct SourceSpan {
    pub input: String,
    pub chain: u64,
    pub start: u64,
    pub end: u64,
    pub preferred: bool,
}
#[derive(Clone, Copy)]
struct SpanKey {
    input: i64,
    chain: i64,
    start: i64,
    end: i64,
}
#[derive(Clone)]
enum SourcePosition {
    Preferred {
        after: Option<SpanKey>,
    },
    Candidate {
        after: i64,
    },
    Spans {
        candidate: i64,
        after: Option<SpanKey>,
    },
    CheckPreferred {
        candidate: i64,
        key: SpanKey,
    },
    CheckEarlier {
        candidate: i64,
        key: SpanKey,
        after: i64,
    },
}
#[derive(Clone)]
pub struct SourceCursor {
    read: SharedRead,
    body: i64,
    position: SourcePosition,
}
pub enum SourceStep {
    Span {
        cursor: SourceCursor,
        source: SourceSpan,
    },
    Continue {
        cursor: SourceCursor,
    },
    End,
}
impl MessageView {
    /// Ranges of the selected body's preferred candidate first, then ranges
    /// only other candidates cite, each once. Each call does one step;
    /// `Continue` means call again.
    pub fn next_source(
        &self,
        after: Option<&SourceCursor>,
        allowance: usize,
    ) -> Result<SourceStep> {
        self.visible()?;
        let position = match after {
            None => SourcePosition::Preferred { after: None },
            Some(cursor) if Arc::ptr_eq(&self.read, &cursor.read) && cursor.body == self.body => {
                cursor.position.clone()
            }
            Some(_) => return Err(Error::StaleMessage),
        };
        let read = locked(&self.read)?;
        let preferred: i64 = db(&self.path, || {
            read.connection.query_row("SELECT preferred_candidate_pk FROM message_body WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND ready=1",params![self.body,self.owner.message,self.owner.session],|r|r.get(0))
        })?;
        let cursor = |position| SourceCursor {
            read: Arc::clone(&self.read),
            body: self.body,
            position,
        };
        let progress = |position| {
            Ok(SourceStep::Continue {
                cursor: cursor(position),
            })
        };
        match position {
            SourcePosition::Preferred { after } => {
                match self.source_key(&read.connection, preferred, after)? {
                    Some(key) => self.source_span(
                        &read.connection,
                        key,
                        true,
                        cursor(SourcePosition::Preferred { after: Some(key) }),
                        allowance,
                    ),
                    None => progress(SourcePosition::Candidate { after: 0 }),
                }
            }
            SourcePosition::Candidate { after } => {
                let candidate: Option<i64> = db(&self.path, || {
                    read.connection.query_row("SELECT pk FROM message_candidate WHERE message_pk=?1 AND session_pk=?2 AND pk>?3 AND pk<>?4 ORDER BY pk LIMIT 1",params![self.owner.message,self.owner.session,after,preferred],|r|r.get(0)).optional()
                })?;
                match candidate {
                    Some(candidate) => progress(SourcePosition::Spans {
                        candidate,
                        after: None,
                    }),
                    None => Ok(SourceStep::End),
                }
            }
            SourcePosition::Spans { candidate, after } => {
                match self.source_key(&read.connection, candidate, after)? {
                    Some(key) => progress(SourcePosition::CheckPreferred { candidate, key }),
                    None => progress(SourcePosition::Candidate { after: candidate }),
                }
            }
            SourcePosition::CheckPreferred { candidate, key } => {
                if source_has(&read.connection, &self.path, preferred, key)? {
                    progress(SourcePosition::Spans {
                        candidate,
                        after: Some(key),
                    })
                } else {
                    progress(SourcePosition::CheckEarlier {
                        candidate,
                        key,
                        after: 0,
                    })
                }
            }
            SourcePosition::CheckEarlier {
                candidate,
                key,
                after,
            } => {
                let earlier: Option<i64> = db(&self.path, || {
                    read.connection.query_row("SELECT pk FROM message_candidate WHERE message_pk=?1 AND session_pk=?2 AND pk>?3 AND pk<?4 AND pk<>?5 ORDER BY pk LIMIT 1",params![self.owner.message,self.owner.session,after,candidate,preferred],|r|r.get(0)).optional()
                })?;
                match earlier {
                    Some(earlier) if source_has(&read.connection, &self.path, earlier, key)? => {
                        progress(SourcePosition::Spans {
                            candidate,
                            after: Some(key),
                        })
                    }
                    Some(earlier) => progress(SourcePosition::CheckEarlier {
                        candidate,
                        key,
                        after: earlier,
                    }),
                    None => self.source_span(
                        &read.connection,
                        key,
                        false,
                        cursor(SourcePosition::Spans {
                            candidate,
                            after: Some(key),
                        }),
                        allowance,
                    ),
                }
            }
        }
    }
    fn source_key(
        &self,
        connection: &Connection,
        candidate: i64,
        after: Option<SpanKey>,
    ) -> Result<Option<SpanKey>> {
        let row = |r: &rusqlite::Row<'_>| -> rusqlite::Result<(i64, i64, SpanKey)> {
            Ok((
                r.get(0)?,
                r.get(1)?,
                SpanKey {
                    input: r.get(2)?,
                    chain: r.get(3)?,
                    start: r.get(4)?,
                    end: r.get(5)?,
                },
            ))
        };
        let found = match after {
            None => db(&self.path, || {
                connection.query_row("SELECT message_pk,session_pk,input_pk,chain,source_start,source_end FROM message_source WHERE candidate_pk=?1 ORDER BY input_pk,chain,source_start,source_end LIMIT 1",[candidate],row).optional()
            })?,
            Some(key) => db(&self.path, || {
                connection.query_row("SELECT message_pk,session_pk,input_pk,chain,source_start,source_end FROM message_source WHERE candidate_pk=?1 AND (input_pk,chain,source_start,source_end)>(?2,?3,?4,?5) ORDER BY input_pk,chain,source_start,source_end LIMIT 1",params![candidate,key.input,key.chain,key.start,key.end],row).optional()
            })?,
        };
        match found {
            Some((message, session, key))
                if message == self.owner.message
                    && session == self.owner.session
                    && key.input > 0
                    && key.chain > 0
                    && key.start >= 0
                    && key.end >= key.start =>
            {
                Ok(Some(key))
            }
            Some(_) => Err(Error::Invariant(
                "stored source range has the wrong owner or bad bounds".into(),
            )),
            None => Ok(None),
        }
    }
    fn source_span(
        &self,
        connection: &Connection,
        key: SpanKey,
        preferred: bool,
        cursor: SourceCursor,
        allowance: usize,
    ) -> Result<SourceStep> {
        let (length, input): (u64, Option<String>) = db(&self.path, || {
            connection.query_row("SELECT length(CAST(id AS BLOB)),CASE WHEN length(CAST(id AS BLOB))<=?2 THEN id END FROM input_ref WHERE pk=?1",params![key.input,i64::try_from(allowance).unwrap_or(i64::MAX)],|r|Ok((crate::parser_state::unsigned(r,0)?,r.get(1)?)))
        })?;
        let input = input.ok_or_else(|| metadata_admission("Message source input id", length))?;
        Ok(SourceStep::Span {
            cursor,
            source: SourceSpan {
                input,
                chain: u64::try_from(key.chain).map_err(|_| Error::Range)?,
                start: u64::try_from(key.start).map_err(|_| Error::Range)?,
                end: u64::try_from(key.end).map_err(|_| Error::Range)?,
                preferred,
            },
        })
    }
}
fn source_has(
    connection: &Connection,
    path: &std::path::Path,
    candidate: i64,
    key: SpanKey,
) -> Result<bool> {
    db(path, || {
        connection.query_row("SELECT EXISTS(SELECT 1 FROM message_source WHERE candidate_pk=?1 AND input_pk=?2 AND chain=?3 AND source_start=?4 AND source_end=?5)",params![candidate,key.input,key.chain,key.start,key.end],|r|r.get(0))
    })
}
