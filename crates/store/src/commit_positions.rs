//! Commit positions of the selected stored records.
use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::row::{Rows as _, row};
use crate::{Reader, Result};
use polylogue_model::{DocumentId, Link, SessionId};
use rusqlite::params;

row! {
    /// Where the shown reading of a document ends in its chain.
    struct DocumentCommit {
        input: String,
        chain: i64,
        source_end: u64,
    }
    statements {
        const OF_DOCUMENT = "SELECT i.id AS input, ds.chain, ds.source_start + ds.source_length AS source_end FROM document d
            JOIN document_parse ds ON ds.document_id=d.id AND ds.input_pk=d.input_pk AND ds.chain=d.source_chain
            JOIN input_ref i ON i.pk=ds.input_pk WHERE d.id=?1";
    }
}

fn positive(value: Option<i64>) -> Result<Option<u64>> {
    value
        .map(|value| {
            u64::try_from(value)
                .ok()
                .filter(|n| *n > 0)
                .or_stored("invalid stored commit position")
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
        self.greatest_commit("SELECT r.committed FROM session s JOIN session_stated x ON x.session_pk=s.pk JOIN session_parse r ON r.session_id=x.stated_id WHERE s.id=?1 UNION ALL SELECT committed FROM shared_prefix WHERE child_id=?1", [id.to_string()])
    }

    /// Commit position of the selected document reading, independent of cache changes.
    pub fn document_commit(&self, id: &DocumentId) -> Result<Option<u64>> {
        let Some(source) = self
            .conn()?
            .maybe(&DocumentCommit::OF_DOCUMENT, [id.to_string()])?
        else {
            return Ok(None);
        };
        // A document reading has a byte range: its commit is the archive's.
        self.source_commit(&source.input, source.chain, Some(source.source_end))
    }

    /// Greatest commit position among the stored link facts.
    pub fn link_commit(&self, link: &Link) -> Result<Option<u64>> {
        let kind = link.kind.as_str();
        let how = link.how.map(|how| how.as_str());
        self.greatest_commit("SELECT s.committed FROM link_parse s WHERE s.parent_id=?1 AND s.child_id=?2 AND s.kind=?3 AND s.reason IS ?4 UNION ALL SELECT committed FROM shared_prefix WHERE parent_id=?1 AND child_id=?2", params![link.parent.to_string(),link.child.to_string(),kind,reason])
    }
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<DocumentCommit>();
}
