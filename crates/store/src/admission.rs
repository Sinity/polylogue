//! Immutable admission evidence for the selected canonical records.
use crate::{Reader, Result, StoreError};
use polylogue_model::{DocumentId, Link, SessionId};
use rusqlite::{OptionalExtension, params};

fn positive(value: Option<i64>) -> Result<Option<u64>> {
    value
        .map(|value| {
            u64::try_from(value)
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| StoreError::Invalid("invalid retained admission evidence".into()))
        })
        .transpose()
}

impl Reader<'_> {
    /// Greatest admission needed by the retained session fact fold.
    pub fn session_commit(&self, id: &SessionId) -> Result<Option<u64>> {
        positive(self.conn()?.query_row(
            "SELECT CASE WHEN count(*)=count(r.admitted) THEN max(r.admitted) END FROM session_record r JOIN session s ON s.pk=r.session_pk WHERE s.id=?1",
            [id.to_string()], |row| row.get(0))?)
    }

    /// Admission of the selected document reading, independent of cache changes.
    pub fn document_commit(&self, id: &DocumentId) -> Result<Option<u64>> {
        let source = self.conn()?.query_row(
            "SELECT ds.admitted,i.id,ds.chain,ds.source_end FROM document_source ds JOIN document d ON d.pk=ds.document_pk JOIN input_ref i ON i.pk=ds.input_pk WHERE d.id=?1 AND ds.shown=1",
            [id.to_string()], |row| Ok((row.get::<_,Option<i64>>(0)?,row.get::<_,String>(1)?,row.get::<_,i64>(2)?,row.get::<_,i64>(3)?)))
            .optional()?;
        let Some((admitted, input, chain, end)) = source else {
            return Ok(None);
        };
        match positive(admitted)? {
            Some(admitted) => Ok(Some(admitted)),
            None => self.source_commit(
                &input,
                chain,
                Some(u64::try_from(end).map_err(|_| {
                    StoreError::Invalid("negative document source coordinate".into())
                })?),
            ),
        }
    }

    /// Greatest admission needed by the retained link fact fold.
    pub fn link_commit(&self, link: &Link) -> Result<Option<u64>> {
        let (kind, how) = link.kind.columns();
        positive(self.conn()?.query_row(
            "SELECT CASE WHEN count(*)=count(s.admitted) THEN max(s.admitted) END FROM link_source s JOIN link l ON l.pk=s.link_pk WHERE l.parent_id=?1 AND l.child_id=?2 AND l.kind=?3 AND l.how=?4",
            params![link.parent.to_string(),link.child.to_string(),kind,how], |row|row.get(0))?)
    }
}
