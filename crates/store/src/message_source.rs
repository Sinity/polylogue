//! One physical owner for each input chain's reading of a message. The content
//! reading lives in the message header; all other readings live in the narrow
//! extra table. Readers use the complete message_source relation.
use crate::{ConnExt as _, Result};
use polylogue_model::{Completion, IdKind, Route};
use rusqlite::{Connection, OptionalExtension as _, params};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Reading {
    pub input: i64,
    pub chain: i64,
    pub start: i64,
    pub end: i64,
    pub session: i64,
    pub main: bool,
    pub route: Route,
    pub completion: Completion,
    pub seq: i64,
    pub position: i64,
    pub order_time: Option<i64>,
    pub at: Option<i64>,
    pub content: bool,
}

impl Reading {
    pub(crate) fn id_kind(&self) -> &'static str {
        if self.content {
            IdKind::Content
        } else {
            IdKind::Native
        }
        .as_str()
    }
}

pub(crate) struct Claim {
    pub message: i64,
    pub reading: Reading,
    pub version: i64,
}

impl Claim {
    fn read(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        // Enum words in this relation are constrained by the schema.
        let word = |column: usize| -> rusqlite::Result<String> { row.get(column) };
        let route = serde_json::from_value(word(8)?.into()).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(8, rusqlite::types::Type::Text, Box::new(e))
        })?;
        let completion = serde_json::from_value(word(13)?.into()).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(13, rusqlite::types::Type::Text, Box::new(e))
        })?;
        Ok(Self {
            message: row.get(0)?,
            reading: Reading {
                input: row.get(1)?,
                chain: row.get(2)?,
                start: row.get(3)?,
                end: row.get(4)?,
                content: word(5)? == IdKind::Content.as_str(),
                session: row.get(6)?,
                main: row.get(7)?,
                route,
                seq: row.get(9)?,
                position: row.get(10)?,
                order_time: row.get(11)?,
                at: row.get(12)?,
                completion,
            },
            version: row.get(14)?,
        })
    }

    /// Store a non-selected observation, or update the current header's raw
    /// reading. The selected header itself is installed by the ordinary writer
    /// in the same statement as the composed message.
    pub(crate) fn put(&self, conn: &Connection) -> Result<()> {
        let r = &self.reading;
        let inline = conn.exists(
            "SELECT EXISTS(SELECT 1 FROM message WHERE pk=?1 AND input_pk=?2 AND source_chain=?3)",
            params![self.message, r.input, r.chain],
        )?;
        self.write(conn, inline)
    }

    fn write(&self, conn: &Connection, inline: bool) -> Result<()> {
        let r = &self.reading;
        let sql = if inline {
            "UPDATE message SET source_start=?4,source_end=?5,source_id_kind=?6,
             source_session_pk=?7,source_main=?8,source_route=?9,source_seq=?10,
             source_record_position=?11,source_order_time=?12,source_at=?13,
             source_completion=?14,source_version_pk=?15
             WHERE pk=?1 AND input_pk=?2 AND source_chain=?3"
        } else {
            "INSERT INTO message_source_extra(message_pk,input_pk,chain,source_start,source_end,
             id_kind,session_pk,main,route,seq,record_position,order_time,at,completion,version_pk)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)
             ON CONFLICT(message_pk,input_pk,chain) DO UPDATE SET
             source_start=excluded.source_start,source_end=excluded.source_end,
             id_kind=excluded.id_kind,session_pk=excluded.session_pk,main=excluded.main,
             route=excluded.route,seq=excluded.seq,record_position=excluded.record_position,
             order_time=excluded.order_time,at=excluded.at,completion=excluded.completion,
             version_pk=excluded.version_pk"
        };
        conn.prepare_cached(sql)?.execute(params![
            self.message,
            r.input,
            r.chain,
            r.start,
            r.end,
            r.id_kind(),
            r.session,
            r.main,
            r.route.as_str(),
            r.seq,
            r.position,
            r.order_time,
            r.at,
            r.completion.as_str(),
            self.version
        ])?;
        Ok(())
    }
}

/// Before replacing the header's default reading, move the displaced reading
/// to its sole extra row and remove the promoted extra row. No body is copied.
/// This runs in the writer's transaction; its caller installs the new header.
pub(crate) fn select(conn: &Connection, message: i64, input: i64, chain: i64) -> Result<()> {
    let previous = conn.prepare_cached(
        "SELECT pk,input_pk,source_chain,source_start,source_end,source_id_kind,source_session_pk,
         source_main,source_route,source_seq,source_record_position,source_order_time,source_at,
         source_completion,source_version_pk FROM message WHERE pk=?1 AND source_version_pk IS NOT NULL",
    )?.query_row([message], Claim::read).optional()?;
    if previous
        .as_ref()
        .is_some_and(|old| (old.reading.input, old.reading.chain) == (input, chain))
    {
        return Ok(());
    }
    conn.prepare_cached(
        "DELETE FROM message_source_extra WHERE message_pk=?1 AND input_pk=?2 AND chain=?3",
    )?
    .execute(params![message, input, chain])?;
    if let Some(previous) = previous {
        conn.prepare_cached("UPDATE message SET source_version_pk=NULL,source_session_pk=NULL,source_seq=NULL WHERE pk=?1")?
            .execute([message])?;
        previous.write(conn, false)?;
    }
    Ok(())
}

pub(crate) fn remove(conn: &Connection, message: i64, input: i64, chain: i64) -> Result<()> {
    conn.prepare_cached("UPDATE message SET source_version_pk=NULL,source_session_pk=NULL,source_seq=NULL WHERE pk=?1 AND input_pk=?2 AND source_chain=?3 AND source_version_pk IS NOT NULL")?
        .execute(params![message,input,chain])?;
    conn.prepare_cached(
        "DELETE FROM message_source_extra WHERE message_pk=?1 AND input_pk=?2 AND chain=?3",
    )?
    .execute(params![message, input, chain])?;
    Ok(())
}

pub(crate) fn retract(conn: &Connection, input: i64, chain: i64) -> Result<()> {
    conn.prepare_cached("UPDATE message SET source_version_pk=NULL,source_session_pk=NULL,source_seq=NULL WHERE input_pk=?1 AND source_chain=?2 AND source_version_pk IS NOT NULL")?
        .execute(params![input,chain])?;
    conn.prepare_cached("DELETE FROM message_source_extra WHERE input_pk=?1 AND chain=?2")?
        .execute(params![input, chain])?;
    Ok(())
}

pub(crate) fn order_time(
    conn: &Connection,
    message: i64,
    input: i64,
    chain: i64,
    time: Option<i64>,
) -> Result<()> {
    conn.prepare_cached("UPDATE message SET source_order_time=?4 WHERE pk=?1 AND input_pk=?2 AND source_chain=?3 AND source_version_pk IS NOT NULL")?
        .execute(params![message,input,chain,time])?;
    conn.prepare_cached("UPDATE message_source_extra SET order_time=?4 WHERE message_pk=?1 AND input_pk=?2 AND chain=?3")?
        .execute(params![message,input,chain,time])?;
    Ok(())
}

pub(crate) fn session_kind(
    conn: &Connection,
    session: i64,
    input: i64,
    chain: i64,
    main: bool,
) -> Result<()> {
    conn.prepare_cached("UPDATE message SET source_main=?4 WHERE source_session_pk=?1 AND input_pk=?2 AND source_chain=?3 AND source_main<>?4 AND source_version_pk IS NOT NULL")?
        .execute(params![session,input,chain,main])?;
    conn.prepare_cached("UPDATE message_source_extra SET main=?4 WHERE session_pk=?1 AND input_pk=?2 AND chain=?3 AND main<>?4")?
        .execute(params![session,input,chain,main])?;
    Ok(())
}

pub(crate) fn move_session(conn: &Connection, from: i64, to: i64) -> Result<()> {
    conn.prepare_cached("UPDATE message SET source_session_pk=?2 WHERE source_session_pk=?1 AND source_version_pk IS NOT NULL")?.execute(params![from,to])?;
    conn.prepare_cached("UPDATE message_source_extra SET session_pk=?2 WHERE session_pk=?1")?
        .execute(params![from, to])?;
    Ok(())
}

pub(crate) fn move_message(
    conn: &Connection,
    from: i64,
    to: i64,
    only: Option<(i64, i64, i64)>,
) -> Result<()> {
    let claims = conn
        .prepare_cached("SELECT * FROM message_source WHERE message_pk=?1")?
        .query_map([from], Claim::read)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for mut claim in claims {
        if let Some((input, chain, _)) = only
            && (claim.reading.input, claim.reading.chain) != (input, chain)
        {
            continue;
        }
        remove(conn, from, claim.reading.input, claim.reading.chain)?;
        claim.message = to;
        if let Some((_, _, session)) = only {
            claim.reading.session = session;
        }
        claim.put(conn)?;
    }
    Ok(())
}

pub(crate) fn copy_reading(conn: &Connection, input: i64, from: i64, to: i64) -> Result<()> {
    // A copy is an older chain and cannot replace the content reading.
    conn.prepare_cached(
        "INSERT INTO message_source_extra SELECT message_pk,input_pk,?3,source_start,source_end,
        id_kind,session_pk,main,route,seq,record_position,order_time,at,completion,version_pk
        FROM message_source WHERE input_pk=?1 AND chain=?2",
    )?
    .execute(params![input, from, to])?;
    Ok(())
}

#[cfg(test)]
mod tests;
