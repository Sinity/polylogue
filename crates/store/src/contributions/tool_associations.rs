//! Selected-Message operation associations share immutable directory leaves.
//! Source body certificates and raw blocks stay unchanged. A derived body
//! copies only fixed metadata and leases the existing roots plus the new path.
use super::directory as d;
use crate::{Error, Result, db};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::Path;
const BODY_COLUMNS: &[&str] = &[
    "message_pk",
    "session_pk",
    "ready",
    "basis_digest",
    "contributor_revision",
    "removal_epoch",
    "preferred_candidate_pk",
    "preferred_ready",
    "block_root_pk",
    "block_root_kind",
    "operation_root_pk",
    "operation_root_kind",
    "field_root_pk",
    "field_root_kind",
    "root_state",
    "layout",
    "role",
    "written_by",
    "prompt_kind",
    "at",
    "completed_at",
    "at_until",
    "first_seen",
    "input_pk",
    "source_chain",
    "source_start",
    "source_end",
    "id_kind",
    "version_hash",
    "in_progress",
    "on_active_path",
    "copy_origin",
    "usage_present",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "reasoning_tokens",
    "web_requests",
    "provider_cost_nusd",
    "cost_nusd",
    "duration_ms",
    "words",
    "parser_version",
    "source_sequence",
];

pub(crate) fn membership(
    c: &Connection,
    p: &Path,
    association: i64,
    message: i64,
    session: i64,
) -> Result<Option<(i64, String, String)>> {
    let body: Option<(i64, Option<i64>, String, String)> = db(p, || {
        c.query_row("SELECT b.pk,b.operation_root_pk,lower(hex(b.version_hash)),(SELECT value FROM meta WHERE key='generation') FROM message m JOIN message_body b ON b.pk=m.selected_body_pk AND b.ready=1 WHERE m.pk=?1 AND m.session_pk=?2 AND m.hidden=0 AND b.removal_epoch=m.removal_epoch",params![message,session],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()
    })?;
    let Some((body, root, hash, generation)) = body else {
        return Ok(None);
    };
    let owner = d::Owner { message, session };
    let key = u64::try_from(association).map_err(|_| Error::Range)?;
    Ok(
        match d::lookup(
            c,
            p,
            owner,
            d::Kind::Operations,
            root,
            key,
            &mut d::Work::default(),
        )? {
            Some(d::Edge {
                link: d::Link::Operation(pk),
                ..
            }) if pk == association => Some((body, hash, generation)),
            None => None,
            _ => {
                return Err(Error::Invariant(
                    "operation association directory key differs".into(),
                ));
            }
        },
    )
}
fn attach(
    tx: &Transaction<'_>,
    p: &Path,
    association: i64,
    message: i64,
    session: i64,
) -> Result<bool> {
    if membership(tx, p, association, message, session)?.is_some() {
        return Ok(false);
    }
    let (body, blocks, fields, root): (i64, Option<i64>, Option<i64>, Option<i64>) = db(p, || {
        tx.query_row("SELECT b.pk,b.block_root_pk,b.field_root_pk,b.operation_root_pk FROM message m JOIN message_body b ON b.pk=m.selected_body_pk AND b.ready=1 WHERE m.pk=?1 AND m.session_pk=?2 AND m.hidden=0 AND b.removal_epoch=m.removal_epoch",params![message,session],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))
    })?;
    let owner = d::Owner { message, session };
    let key = u64::try_from(association).map_err(|_| Error::Range)?;
    let (next, _) = d::replace(
        tx,
        p,
        owner,
        d::Kind::Operations,
        root,
        key,
        Some(d::Edge {
            link: d::Link::Operation(association),
            counts: d::Counts::default(),
        }),
        &mut d::Work::default(),
    )?;
    let selected = clone_with_root(tx, p, owner, body, blocks, fields, next)?;
    if db(p, || {
        tx.execute("UPDATE message SET selected_body_pk=?2 WHERE pk=?1 AND selected_body_pk=?3 AND session_pk=?4 AND hidden=0",params![message,selected,body,session])
    })? != 1
    {
        return Err(Error::MessagePredecessorChanged);
    }
    Ok(true)
}
// The caller has already fenced the owner and ready source body. Only a fixed
// row is copied; all three variable directories keep their original leases.
fn clone_with_root(
    tx: &Transaction<'_>,
    p: &Path,
    owner: d::Owner,
    body: i64,
    blocks: Option<i64>,
    fields: Option<i64>,
    root: i64,
) -> Result<i64> {
    for root in [blocks, fields, Some(root)].into_iter().flatten() {
        d::acquire(tx, p, owner, root)?;
    }
    // This fixed schema-owned row copy never materializes variable values or
    // modifies a ready body's roots. The source candidate keeps its own body.
    let columns = BODY_COLUMNS.join(",");
    let values = BODY_COLUMNS
        .iter()
        .map(|name| {
            if *name == "operation_root_pk" {
                "?2"
            } else {
                name
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "INSERT INTO message_body({columns}) SELECT {values} FROM message_body WHERE pk=?1 AND ready=1"
    );
    if db(p, || tx.execute(&sql, params![body, root]))? != 1 {
        return Err(Error::MessagePredecessorChanged);
    }
    Ok(tx.last_insert_rowid())
}

/// Source election must not erase independently proved operation membership.
/// Carry the prior immutable root into a fixed metadata clone of the new body.
/// A different nonempty operation root needs reconciliation, never union by ID.
pub(super) fn carry_selected(
    tx: &Transaction<'_>,
    p: &Path,
    owner: d::Owner,
    body: i64,
) -> Result<i64> {
    let old: Option<i64> = db(p, || {
        tx.query_row("SELECT CASE WHEN b.removal_epoch=m.removal_epoch THEN b.operation_root_pk END FROM message m LEFT JOIN message_body b ON b.pk=m.selected_body_pk WHERE m.pk=?1 AND m.session_pk=?2 AND m.hidden=0",params![owner.message,owner.session],|r|r.get(0))
    })?;
    let Some(old) = old else {
        return Ok(body);
    };
    let (blocks, fields, root): (Option<i64>, Option<i64>, Option<i64>) = db(p, || {
        tx.query_row("SELECT block_root_pk,field_root_pk,operation_root_pk FROM message_body WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND ready=1",params![body,owner.message,owner.session],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))
    })?;
    if root == Some(old) {
        return Ok(body);
    }
    if root.is_some() {
        return Err(Error::MessagePreparationPending);
    }
    clone_with_root(tx, p, owner, body, blocks, fields, old)
}
/// One operation admits at most16 overlaps plus its direct owner. Replays may
/// restore membership after source election selects an equivalent source body.
pub(crate) fn attach_selected(tx: &Transaction<'_>, p: &Path, item: i64) -> Result<bool> {
    let rows: Vec<(i64, i64, i64)> = db(p, || {
        let mut s=tx.prepare("SELECT a.pk,a.message_pk,a.session_pk FROM tool_item_association a JOIN tool_item t ON t.selected_version_pk=a.tool_item_version_pk WHERE t.pk=?1 ORDER BY a.pk LIMIT 18")?;
        s.query_map([item], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect()
    })?;
    if rows.len() > 17 {
        return Err(Error::ParserAdmission {
            resource: "operation association publication",
            required: rows.len() as u64,
        });
    }
    let mut changed = false;
    for (association, message, session) in rows {
        changed |= attach(tx, p, association, message, session)?;
    }
    Ok(changed)
}
