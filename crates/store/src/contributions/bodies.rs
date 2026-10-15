//! Retiring an unselected immutable body detaches retention metadata, never its
//! rank/digest certificate. Each turn owns a fixed row transition or root lease.
use super::{directory as d, writer::Build};
use crate::{Error, Result, Writer, db};
use rusqlite::{OptionalExtension, params};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyId {
    pub(super) path: PathBuf,
    pub(super) generation: String,
    pub(super) pk: i64,
    pub(super) message: i64,
    pub(super) session: i64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateId {
    path: PathBuf,
    generation: String,
    pk: i64,
    message: i64,
    session: i64,
    basis: [u8; 32],
}
impl Build {
    pub fn body_id(&self) -> BodyId {
        BodyId {
            path: self.path.clone(),
            generation: self.source.index_generation.clone(),
            pk: self.body,
            message: self.message,
            session: self.session,
        }
    }
    pub fn candidate_id(&self) -> CandidateId {
        CandidateId {
            path: self.path.clone(),
            generation: self.source.index_generation.clone(),
            pk: self.candidate,
            message: self.message,
            session: self.session,
            basis: self.digest,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateCollection {
    Retained,
    Source,
    Candidate,
    Missing,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyRetirement {
    Selected,
    ActiveBuild,
    Eligible,
    PendingCandidate,
    DetachedCandidate,
    Body,
    Missing,
}
fn check(connection: &rusqlite::Connection, path: &Path, id: &BodyId) -> Result<()> {
    if path != id.path {
        return Err(Error::MessagePredecessorChanged);
    }
    let generation: String = db(path, || {
        connection.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get(0)
        })
    })?;
    if generation != id.generation {
        return Err(Error::MessagePredecessorChanged);
    }
    Ok(())
}
impl Writer {
    /// Certificate/provenance deletion never scans facts: known leaf collection
    /// releases the actual fact owners first, and each remaining locator is one
    /// keyset turn. A still-referenced certificate remains intact.
    pub fn collect_message_candidate(&mut self, id: &CandidateId) -> Result<CandidateCollection> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        if path != id.path {
            return Err(Error::MessagePredecessorChanged);
        }
        let generation: String = db(&path, || {
            tx.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
                r.get(0)
            })
        })?;
        if generation != id.generation {
            return Err(Error::MessagePredecessorChanged);
        }
        let row: Option<(bool, Option<i64>, Vec<u8>)> = db(&path, || {
            tx.query_row("SELECT eligible,body_pk,basis_digest FROM message_candidate WHERE pk=?1 AND message_pk=?2 AND session_pk=?3",params![id.pk,id.message,id.session],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()
        })?;
        let Some((eligible, body, basis)) = row else {
            return Ok(CandidateCollection::Missing);
        };
        if basis != id.basis {
            return Err(Error::MessagePredecessorChanged);
        }
        if eligible || body.is_some() {
            return Ok(CandidateCollection::Retained);
        }
        let retained: bool = db(&path, || {
            tx.query_row("SELECT EXISTS(SELECT 1 FROM message_body WHERE preferred_candidate_pk=?1) OR EXISTS(SELECT 1 FROM message_chain_choice WHERE candidate_pk=?1) OR EXISTS(SELECT 1 FROM message_input_choice WHERE candidate_pk=?1) OR EXISTS(SELECT 1 FROM message_field_fact WHERE candidate_pk=?1)",[id.pk],|r|r.get(0))
        })?;
        if retained {
            return Ok(CandidateCollection::Retained);
        }
        let source: Option<(i64, i64, i64, i64)> = db(&path, || {
            tx.query_row("SELECT input_pk,chain,source_start,source_end FROM message_source WHERE candidate_pk=?1 ORDER BY input_pk,chain,source_start,source_end LIMIT 1",[id.pk],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()
        })?;
        if let Some((input, chain, start, end)) = source {
            if db(&path, || {
                tx.execute(
                    "DELETE FROM message_source WHERE candidate_pk=?1 AND input_pk=?2 AND chain=?3 AND source_start=?4 AND source_end=?5",
                    params![id.pk, input, chain, start, end],
                )
            })? != 1
            {
                return Err(Error::MessagePredecessorChanged);
            }
            db(&path, || tx.commit())?;
            return Ok(CandidateCollection::Source);
        }
        if db(&path, || {
            tx.execute(
                "DELETE FROM message_candidate WHERE pk=?1 AND eligible=0 AND body_pk IS NULL",
                [id.pk],
            )
        })? != 1
        {
            return Err(Error::MessagePredecessorChanged);
        }
        db(&path, || tx.commit())?;
        Ok(CandidateCollection::Candidate)
    }
    pub fn retire_message_body(&mut self, id: &BodyId) -> Result<BodyRetirement> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        check(&tx, &path, id)?;
        let roots: Option<(Option<i64>, Option<i64>, Option<i64>, bool)> = db(&path, || {
            tx.query_row(
            "SELECT block_root_pk,field_root_pk,operation_root_pk,ready FROM message_body WHERE pk=?1 AND message_pk=?2 AND session_pk=?3",
            params![id.pk,id.message,id.session],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).optional()
        })?;
        let Some((blocks, fields, operations, ready)) = roots else {
            return Ok(BodyRetirement::Missing);
        };
        let selected: bool = db(&path, || {
            tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM message WHERE selected_body_pk=?1)",
                [id.pk],
                |r| r.get(0),
            )
        })?;
        if selected {
            return Ok(BodyRetirement::Selected);
        }
        let active: bool = db(&path, || {
            tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM message_build WHERE message_pk=?1 AND body_pk=?2)",
                params![id.message, id.pk],
                |r| r.get(0),
            )
        })?;
        if active {
            return Ok(BodyRetirement::ActiveBuild);
        }
        if !ready {
            return Err(Error::Invariant(
                "unready bodies use their owning preparation abort".into(),
            ));
        }
        let eligible: bool = db(&path, || {
            tx.query_row("SELECT EXISTS(SELECT 1 FROM message_candidate INDEXED BY message_candidate_required_body WHERE body_pk=?1 AND eligible=1)",[id.pk],|r|r.get(0))
        })?;
        if eligible {
            return Ok(BodyRetirement::Eligible);
        }
        let candidate: Option<(i64, bool)> = db(&path, || {
            tx.query_row(
                "SELECT pk,ready FROM message_candidate INDEXED BY message_candidate_body WHERE body_pk=?1 ORDER BY pk LIMIT 1",
                [id.pk],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
        })?;
        if let Some((candidate, ready)) = candidate {
            if !ready {
                return Ok(BodyRetirement::PendingCandidate);
            }
            if db(&path, || {
                tx.execute("UPDATE message_candidate SET body_pk=NULL,body_ready=NULL WHERE pk=?1 AND body_pk=?2 AND ready=1 AND eligible=0",params![candidate,id.pk])
            })? != 1
            {
                return Err(Error::MessagePredecessorChanged);
            }
            db(&path, || tx.commit())?;
            return Ok(BodyRetirement::DetachedCandidate);
        }
        if db(&path, || {
            tx.execute("DELETE FROM message_body WHERE pk=?1 AND message_pk=?2 AND session_pk=?3 AND ready=1",params![id.pk,id.message,id.session])
        })? != 1
        {
            return Err(Error::MessagePredecessorChanged);
        }
        for root in [blocks, fields, operations].into_iter().flatten() {
            d::release(
                &tx,
                &path,
                d::Owner {
                    message: id.message,
                    session: id.session,
                },
                root,
            )?;
        }
        db(&path, || tx.commit())?;
        Ok(BodyRetirement::Body)
    }
}
