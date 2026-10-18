//! Fixed-size projection metadata after the owning source worker has finished
//! projection, suppliedness and evidence verification. This does not elect a
//! winner, apply donor fills, publish search or declare parser completion.
use super::{
    blocks, directory as d,
    writer::{self, Build},
};
use crate::{Error, Result, Writer, db};
use polylogue_model::{BlobHash, merge::FinishState};
use rusqlite::params;

#[derive(Clone, Debug)]
pub struct PreparedProjection {
    pub(super) build: Build,
    pub(super) hash: BlobHash,
}
impl PreparedProjection {
    pub fn body_pk(&self) -> i64 {
        self.build.body
    }
    pub fn candidate_pk(&self) -> i64 {
        self.build.candidate
    }
    pub fn version_hash(&self) -> BlobHash {
        self.hash
    }
}
impl Writer {
    /// The source owner calls this only after its complete projection cursor
    /// and exact selected-evidence proof agree with this Build's pinned basis.
    /// No payload hashing/serialization, member scan or source IO runs here.
    /// This is an unready intermediate and remains eligible for bounded abort;
    /// canonical publication must separately establish immutable readiness.
    /// Unknown source completion remains eligible unclassified, never Finished.
    /// Source mapping/evidence gaps must remain pending instead of calling this.
    pub fn finish_message_projection(
        &mut self,
        build: &Build,
        step: &blocks::ProjectionStep<'_>,
        version_hash: BlobHash,
        finish: FinishState,
        allowance: usize,
    ) -> Result<PreparedProjection> {
        blocks::admit(step, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        let (active, active_origin, active_block, blocks, fields, operations): (
            Option<i64>,
            Option<i64>,
            Option<i64>,
            Option<i64>,
            Option<i64>,
            Option<i64>,
        ) = db(&path, || {
            tx.query_row(
            "SELECT w.active_value_pk,w.active_origin_value_pk,w.active_block_pk,b.block_root_pk,b.field_root_pk,b.operation_root_pk FROM message_build w JOIN message_body b ON b.pk=w.body_pk WHERE w.pk=?1",
            [build.pk],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?)))
        })?;
        if active.is_some() || active_origin.is_some() || active_block.is_some() {
            return Err(Error::ValueNotReady);
        }
        let owner = d::Owner {
            message: build.message,
            session: build.session,
        };
        let counts = d::summary(&tx, &path, owner, d::Kind::Groups, blocks)?;
        let facts = d::summary(&tx, &path, owner, d::Kind::Fields, fields)?;
        // Validate the fixed root even when there are no operations; operation
        // leaf/ToolItem readiness remains the owning producer's prerequisite.
        d::summary(&tx, &path, owner, d::Kind::Operations, operations)?;
        blocks::commit_cursor(&tx, &path, build, step)?;
        let (completion, progress) = match finish {
            FinishState::InProgress => (0, Some(true)),
            FinishState::Unknown => (1, None),
            FinishState::Finished => (2, Some(false)),
        };
        // Intermediate metadata remains unready and ineligible. A ready
        // body/candidate certificate is reserved for canonical publication, so
        // interrupted projection/folding still uses bounded unpublished abort.
        if db(&path, || {
            tx.execute(
                "UPDATE message_body SET in_progress=?2 WHERE pk=?1 AND ready=0",
                params![build.body, progress],
            )
        })? != 1
        {
            return Err(Error::StaleMessage);
        }
        let source_fields = facts
            .source_fields
            .checked_add(counts.source_fields)
            .ok_or(Error::Range)?;
        if db(&path, || {
            tx.execute("UPDATE message_candidate SET facts_ready=1,source_blocks=?2,source_text_blocks=?3,source_fields=?4,version_hash=?5,finish=?6 WHERE pk=?1 AND ready=0 AND eligible=0",params![build.candidate,counts.source_blocks,counts.source_text_blocks,source_fields,version_hash.bytes().as_slice(),completion])
        })? != 1
        {
            return Err(Error::StaleMessage);
        }
        db(&path, || {
            tx.execute(
                "UPDATE message_build SET phase='rank' WHERE pk=?1",
                [build.pk],
            )
        })?;
        db(&path, || tx.commit())?;
        Ok(PreparedProjection {
            build: build.clone(),
            hash: version_hash,
        })
    }
}
