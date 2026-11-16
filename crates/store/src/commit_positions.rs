//! Commit positions of the selected stored records.
use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::{Reader, Result, StoreError};
use polylogue_model::{DocumentId, Link, SessionId};
use rusqlite::{OptionalExtension, params};

fn positive(value: Option<i64>) -> Result<Option<u64>> {
    value
        .map(|value| {
            u64::try_from(value)
                .ok()
                .filter(|n| *n > 0)
                .or_invalid("invalid stored commit position")
        })
        .transpose()
}

impl Reader<'_> {
    /// Greatest commit position among the stored session facts.
    pub fn session_commit(&self, id: &SessionId) -> Result<Option<u64>> {
        positive(self.conn()?.scalar("WITH contributors(committed) AS (SELECT r.committed FROM session_record r JOIN session s ON s.pk=r.session_pk WHERE s.id=?1 UNION ALL SELECT committed FROM prefix_copy WHERE child_id=?1) SELECT CASE WHEN count(*)=count(committed) THEN max(committed) END FROM contributors", [id.to_string()])?)
    }

    /// Commit position of the selected document reading, independent of cache changes.
    pub fn document_commit(&self, id: &DocumentId) -> Result<Option<u64>> {
        let source = self.conn()?.query_row(
            "SELECT ds.committed,i.id,ds.chain,ds.source_end FROM document_source ds JOIN document d ON d.pk=ds.document_pk JOIN input_ref i ON i.pk=ds.input_pk WHERE d.id=?1 AND ds.shown=1",
            [id.to_string()], |row| <(Option<i64>, String, i64, i64)>::try_from(row))
            .optional()?;
        let Some((committed, input, chain, end)) = source else {
            return Ok(None);
        };
        match positive(committed)? {
            Some(committed) => Ok(Some(committed)),
            None => self.source_commit(
                &input,
                chain,
                Some(u64::try_from(end).map_err(|_| {
                    StoreError::Invalid("negative document source coordinate".into())
                })?),
            ),
        }
    }

    /// Greatest commit position among the stored link facts.
    pub fn link_commit(&self, link: &Link) -> Result<Option<u64>> {
        let kind = link.kind.as_str();
        let how = link.how.map(|how| how.as_str());
        positive(self.conn()?.scalar("WITH contributors(committed) AS (SELECT s.committed FROM link_source s JOIN link l ON l.pk=s.link_pk WHERE l.parent_id=?1 AND l.child_id=?2 AND l.kind=?3 AND l.how IS ?4 UNION ALL SELECT committed FROM prefix_copy WHERE parent_id=?1 AND child_id=?2) SELECT CASE WHEN count(*)=count(committed) THEN max(committed) END FROM contributors", params![link.parent.to_string(),link.child.to_string(),kind,how])?)
    }
}
