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
    /// The greatest commit position among the rows `contributors` selects
    /// (a `SELECT committed ...` over the fact's source rows and its
    /// prefix copies). One contributor without a commit position makes the
    /// whole fact's position unknown.
    fn greatest_commit(
        &self,
        contributors: &str,
        params: impl rusqlite::Params,
    ) -> Result<Option<u64>> {
        positive(self.conn()?.scalar(
            &format!("WITH contributors(committed) AS ({contributors}) SELECT CASE WHEN count(*)=count(committed) THEN max(committed) END FROM contributors"),
            params,
        )?)
    }

    /// Greatest commit position among the stored session facts.
    pub fn session_commit(&self, id: &SessionId) -> Result<Option<u64>> {
        self.greatest_commit("SELECT r.committed FROM session s JOIN session_stated x ON x.session_pk=s.pk JOIN session_record r ON r.session_id=x.stated_id WHERE s.id=?1 UNION ALL SELECT committed FROM prefix_copy WHERE child_id=?1", [id.to_string()])
    }

    /// Commit position of the selected document reading, independent of cache changes.
    pub fn document_commit(&self, id: &DocumentId) -> Result<Option<u64>> {
        let source = self.conn()?.query_row(
            "SELECT ds.committed,i.id,ds.chain,ds.source_end FROM document d JOIN document_parse ds ON ds.document_id=d.id AND ds.input_pk=d.input_pk AND ds.chain=d.source_chain JOIN input_ref i ON i.pk=ds.input_pk WHERE d.id=?1",
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
        self.greatest_commit("SELECT s.committed FROM link_parse s WHERE s.parent_id=?1 AND s.child_id=?2 AND s.kind=?3 AND s.how IS ?4 UNION ALL SELECT committed FROM prefix_copy WHERE parent_id=?1 AND child_id=?2", params![link.parent.to_string(),link.child.to_string(),kind,how])
    }
}
