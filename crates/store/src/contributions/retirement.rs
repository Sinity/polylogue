//! Invalidated unpublished work drains in owning transactions. Each turn touches
//! one frontier/root, one <=16-edge node, one fact/source row, or fixed control
//! metadata. Durable rows remain the work description; no second queue exists.
use super::directory;
use super::writer::Build;
use crate::{Error, Result, Writer, db};
use rusqlite::{OptionalExtension, params};
use std::path::Path;

pub(super) fn release_ready_value(
    tx: &rusqlite::Transaction<'_>,
    path: &Path,
    value: i64,
) -> Result<()> {
    if db(path, || {
        tx.execute("UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND incoming>0 AND state='ready'",[value])
    })? != 1
    {
        return Err(Error::Invariant(
            "canonical value lost its ready owner lease".into(),
        ));
    }
    db(path, || {
        tx.execute(
            "UPDATE body_value SET state='retiring' WHERE pk=?1 AND incoming=0",
            [value],
        )
    })?;
    Ok(())
}
fn drop_fact(tx: &rusqlite::Transaction<'_>, path: &Path, fact: i64, value: i64) -> Result<()> {
    let (slot, origin): (i64, Option<i64>) = db(path, || {
        tx.query_row(
            "SELECT slot_pk,origin_value_pk FROM message_field_fact WHERE pk=?1",
            [fact],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
    })?;
    db(path, || {
        tx.execute("DELETE FROM message_field_fact WHERE pk=?1", [fact])
    })?;
    release_ready_value(tx, path, value)?;
    if let Some(origin) = origin {
        release_ready_value(tx, path, origin)?;
    }
    let retained: bool = db(path, || {
        tx.query_row("SELECT EXISTS(SELECT 1 FROM message_field_fact WHERE slot_pk=?1) OR EXISTS(SELECT 1 FROM message_body_edge WHERE conflict_slot_pk=?1)",[slot],|r|r.get(0))
    })?;
    if !retained {
        db(path, || {
            tx.execute("DELETE FROM message_field_slot WHERE pk=?1", [slot])
        })?;
    }
    Ok(())
}
/// A known leaf can disappear only after its final directory reference AND
/// retained source-body/current-eligibility ownership have both gone away.
pub(super) fn release_unreferenced_fact(
    tx: &rusqlite::Transaction<'_>,
    path: &Path,
    fact: i64,
) -> Result<()> {
    let row: Option<(i64, bool, Option<i64>)> = db(path, || {
        tx.query_row("SELECT f.value_pk,c.eligible,c.body_pk FROM message_field_fact f JOIN message_candidate c ON c.pk=f.candidate_pk WHERE f.pk=?1",[fact],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()
    })?;
    let Some((value, eligible, body)) = row else {
        return Ok(());
    };
    if eligible || body.is_some() {
        return Ok(());
    }
    let retained: bool = db(path, || {
        tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM message_body_edge WHERE field_fact_pk=?1)",
            [fact],
            |r| r.get(0),
        )
    })?;
    if !retained {
        drop_fact(tx, path, fact, value)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retirement {
    Frontier,
    Root,
    Node,
    Fact,
    Source,
    Complete,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueCollection {
    Chunk,
    Value,
    Idle,
}
impl Writer {
    /// Resume only an explicitly invalidated, never-published preparation.
    /// Its source may have advanced or been removed: cleanup cannot publish or
    /// clear that source's newer parse/effect work. Committed private identity
    /// and index incarnation still fence every turn.
    pub fn retire_message_turn(&mut self, build: &Build) -> Result<Retirement> {
        let path = self.root.join("index.db");
        if path != build.path {
            return Err(Error::MessagePredecessorChanged);
        }
        let tx = db(&path, || self._index.transaction())?;
        let generation: String = db(&path, || {
            tx.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
                r.get(0)
            })
        })?;
        if generation != build.source.index_generation {
            return Err(Error::MessagePredecessorChanged);
        }
        let row: Option<(
            String,
            bool,
            bool,
            bool,
            Option<i64>,
            Option<i64>,
            Option<i64>,
        )> = db(&path, || {
            tx.query_row("SELECT w.phase,b.ready,c.ready,c.eligible,w.active_value_pk,w.active_origin_value_pk,w.active_block_pk FROM message_build w JOIN message_body b ON b.pk=w.body_pk JOIN message_candidate c ON c.pk=b.preferred_candidate_pk WHERE w.pk=?1 AND w.body_pk=?2 AND c.pk=?3 AND w.basis_digest=?4 AND w.message_pk=?5 AND w.session_pk=?6",params![build.pk,build.body,build.candidate,build.digest.as_slice(),build.message,build.session],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?))).optional()
        })?;
        let Some((
            phase,
            body_ready,
            candidate_ready,
            eligible,
            active,
            active_origin,
            active_block,
        )) = row
        else {
            let retained: bool = db(&path, || {
                tx.query_row("SELECT EXISTS(SELECT 1 FROM message_body WHERE pk=?1) OR EXISTS(SELECT 1 FROM message_candidate WHERE pk=?2)",params![build.body,build.candidate],|r|r.get(0))
            })?;
            return if retained {
                Err(Error::Invariant(
                    "missing retirement control with retained preparation".into(),
                ))
            } else {
                Ok(Retirement::Complete)
            };
        };
        if phase != "retiring"
            || body_ready
            || candidate_ready
            || eligible
            || active.is_some()
            || active_block.is_some()
            || active_origin.is_some()
        {
            return Err(Error::MessagePreparationPending);
        }
        let owner = directory::Owner {
            message: build.message,
            session: build.session,
        };
        let frontier: Option<(i64, i64)> = db(&path, || {
            tx.query_row("SELECT depth,node_pk FROM message_build_frontier WHERE build_pk=?1 ORDER BY depth LIMIT 1",[build.pk],|r|Ok((r.get(0)?,r.get(1)?))).optional()
        })?;
        if let Some((depth, node)) = frontier {
            db(&path, || {
                tx.execute(
                    "DELETE FROM message_build_frontier WHERE build_pk=?1 AND depth=?2",
                    params![build.pk, depth],
                )
            })?;
            directory::release(&tx, &path, owner, node)?;
            db(&path, || tx.commit())?;
            return Ok(Retirement::Frontier);
        }
        let roots: (Option<i64>, Option<i64>, Option<i64>) = db(&path, || {
            tx.query_row("SELECT block_root_pk,field_root_pk,operation_root_pk FROM message_body WHERE pk=?1",[build.body],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))
        })?;
        for (column, root) in [
            ("block_root_pk", roots.0),
            ("field_root_pk", roots.1),
            ("operation_root_pk", roots.2),
        ] {
            if let Some(root) = root {
                db(&path, || {
                    tx.execute(
                        &format!("UPDATE message_body SET {column}=NULL WHERE pk=?1 AND ready=0"),
                        [build.body],
                    )
                })?;
                directory::release(&tx, &path, owner, root)?;
                db(&path, || tx.commit())?;
                return Ok(Retirement::Root);
            }
        }
        let node: Option<i64> = db(&path, || {
            tx.query_row("SELECT pk FROM message_body_node WHERE message_pk=?1 AND incoming=0 ORDER BY pk LIMIT 1",[build.message],|r|r.get(0)).optional()
        })?;
        if let Some(node) = node {
            if !directory::collect(&tx, &path, node)? {
                return Err(Error::Invariant("retirement node claim changed".into()));
            }
            db(&path, || tx.commit())?;
            return Ok(Retirement::Node);
        }
        let fact: Option<(i64, i64)> = db(&path, || {
            tx.query_row("SELECT pk,value_pk FROM message_field_fact WHERE candidate_pk=?1 ORDER BY slot_pk,pk LIMIT 1",[build.candidate],|r|Ok((r.get(0)?,r.get(1)?))).optional()
        })?;
        if let Some((fact, value)) = fact {
            let referenced: bool = db(&path, || {
                tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM message_body_edge WHERE field_fact_pk=?1)",
                    [fact],
                    |r| r.get(0),
                )
            })?;
            if referenced {
                return Err(Error::Invariant(
                    "unpublished candidate fact is leased outside its drained preparation".into(),
                ));
            }
            drop_fact(&tx, &path, fact, value)?;
            db(&path, || tx.commit())?;
            return Ok(Retirement::Fact);
        }
        let source: Option<(i64, i64, i64, i64)> = db(&path, || {
            tx.query_row("SELECT input_pk,chain,source_start,source_end FROM message_source WHERE candidate_pk=?1 ORDER BY input_pk,chain,source_start,source_end LIMIT 1",[build.candidate],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()
        })?;
        if let Some((input, chain, start, end)) = source {
            if db(&path, || {
                tx.execute(
                    "DELETE FROM message_source WHERE candidate_pk=?1 AND input_pk=?2 AND chain=?3 AND source_start=?4 AND source_end=?5",
                    params![build.candidate, input, chain, start, end],
                )
            })? != 1
            {
                return Err(Error::MessagePredecessorChanged);
            }
            db(&path, || tx.commit())?;
            return Ok(Retirement::Source);
        }
        // Break the unpublished body/candidate cycle without deleting a ready
        // certificate. Any unexpected foreign owner refuses and rolls back all.
        db(&path, || {
            tx.execute("UPDATE message_candidate SET body_pk=NULL,body_ready=NULL WHERE pk=?1 AND ready=0 AND eligible=0",[build.candidate])
        })?;
        db(&path, || {
            tx.execute(
                "UPDATE message_body SET preferred_candidate_pk=NULL WHERE pk=?1 AND ready=0",
                [build.body],
            )
        })?;
        super::blocks::discard_cursor(&tx, &path, build)?;
        db(&path, || {
            tx.execute("DELETE FROM message_build WHERE pk=?1", [build.pk])
        })?;
        db(&path, || {
            tx.execute(
                "DELETE FROM message_body WHERE pk=?1 AND ready=0",
                [build.body],
            )
        })?;
        db(&path, || {
            tx.execute(
                "DELETE FROM message_candidate WHERE pk=?1 AND ready=0 AND eligible=0",
                [build.candidate],
            )
        })?;
        db(&path, || tx.commit())?;
        Ok(Retirement::Complete)
    }
    /// One immutable retired value chunk or final empty metadata row. Building
    /// and ready values with incoming0 are deliberately excluded: a generic
    /// in-process handle is not an invalidation/retirement claim.
    pub fn collect_retiring_value(&mut self) -> Result<ValueCollection> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        let value: Option<i64> = db(&path, || {
            tx.query_row(
                "SELECT pk FROM body_value WHERE state='retiring' ORDER BY pk LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()
        })?;
        let Some(value) = value else {
            return Ok(ValueCollection::Idle);
        };
        let retained: bool = db(&path, || {
            tx.query_row("SELECT EXISTS(SELECT 1 FROM message_build WHERE active_value_pk=?1) OR EXISTS(SELECT 1 FROM message_build WHERE active_origin_value_pk=?1) OR EXISTS(SELECT 1 FROM message_field_fact WHERE value_pk=?1) OR EXISTS(SELECT 1 FROM message_field_fact WHERE origin_value_pk=?1) OR EXISTS(SELECT 1 FROM text WHERE value_pk=?1)",[value],|r|r.get(0))
        })?;
        if retained {
            return Err(Error::Invariant(
                "retiring value still has an owning reference".into(),
            ));
        }
        let offset: Option<i64> = db(&path, || {
            tx.query_row(
                "SELECT offset FROM body_value_chunk WHERE value_pk=?1 ORDER BY offset LIMIT 1",
                [value],
                |r| r.get(0),
            )
            .optional()
        })?;
        let step = if let Some(offset) = offset {
            db(&path, || {
                tx.execute(
                    "DELETE FROM body_value_chunk WHERE value_pk=?1 AND offset=?2",
                    params![value, offset],
                )
            })?;
            ValueCollection::Chunk
        } else {
            db(&path, || {
                tx.execute(
                    "DELETE FROM body_value WHERE pk=?1 AND incoming=0 AND state='retiring'",
                    [value],
                )
            })?;
            ValueCollection::Value
        };
        db(&path, || tx.commit())?;
        Ok(step)
    }
}
