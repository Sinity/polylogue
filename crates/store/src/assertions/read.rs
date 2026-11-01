//! Typed reads use the pinned Reader's sole currency projection. Direct
//! association is exact; container membership belongs to the query backend.
use super::*;
use crate::Reader;
use rusqlite::OptionalExtension;
use std::ops::ControlFlow;

impl Reader<'_> {
    /// An assertion revision is immutable; only its admission position and
    /// the imports that carried it change after it is written.
    pub(crate) fn fold_assertion_content_version(
        &self,
        root: &str,
        revision: &str,
        hash: &mut blake3::Hasher,
    ) -> Result<()> {
        let source = Source::reading(self.conn()?)?;
        let row: Option<(i64, String, bool)> = self
            .conn()?
            .prepare_cached(&source.sql(
                "SELECT commit_seq,imports,historical IS NOT NULL FROM @source.assertion WHERE id=?1 AND root=?2",
            ))?
            .query_row([revision, root], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .optional()?;
        let row =
            row.ok_or_else(|| StoreError::NotFound(format!("assertion:{root}@{revision}")))?;
        crate::read::fold_serialized(hash, &row)
    }

    fn resolved_assertion_revision(
        &self,
        reference: &RevisionReference,
    ) -> Result<Option<RevisionReference>> {
        if reference.revision.is_some() {
            return Ok(Some(reference.clone()));
        }
        let Some(state) = self.assertion_family_currency(reference)? else {
            return Ok(None);
        };
        if state.heads.len() > 1 {
            return Err(StoreError::Ambiguous {
                reference: Reference::Assertion(reference.clone()).to_string(),
                candidates: state.heads.iter().map(ToString::to_string).collect(),
            });
        }
        match state.heads.into_iter().next() {
            Some(Reference::Assertion(exact)) => Ok(Some(exact)),
            None => Ok(None),
            Some(_) => Err(invalid("assertion family contains another reference kind")),
        }
    }

    /// Exact revisions remain readable when superseded or withdrawn. A current
    /// kind consumer refuses a historical value outside its validated schema.
    pub fn assertion(&self, reference: &RevisionReference) -> Result<Option<Assertion>> {
        match self.assertion_evidence(reference)? {
            Some(polylogue_model::historical::AssertionEvidence::Current(row)) => Ok(Some(row)),
            Some(historical) => Err(StoreError::ReferenceUnavailable {
                reference: historical.reference().to_string(),
                reason: "the recorded historical value has no validated current-kind schema".into(),
            }),
            None => Ok(None),
        }
    }

    /// Object reads expose the original typed historical form without treating
    /// its old value schema as a current native assertion.
    pub fn assertion_evidence(
        &self,
        reference: &RevisionReference,
    ) -> Result<Option<polylogue_model::historical::AssertionEvidence>> {
        let Some(exact) = self.resolved_assertion_revision(reference)? else {
            return Ok(None);
        };
        let id = exact
            .revision
            .as_ref()
            .ok_or_else(|| invalid("assertion revision is not exact"))?;
        let row = get_evidence(self.conn()?, id)?
            .filter(|row| row.reference() == Reference::Assertion(exact.clone()));
        self.check()?;
        Ok(row)
    }

    /// Streams all current assertions with an exact `about` link, including
    /// every conflicting head. False currency never becomes a chosen winner.
    pub fn visit_assertions_about(
        &self,
        reference: &Reference,
        mut visit: impl FnMut(Assertion) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let mut statement = conn.prepare_cached(&source.sql(
            "SELECT a.root,a.id FROM temp.assertion_about l JOIN @source.assertion a ON a.id=l.assertion WHERE l.ref=?1 ORDER BY a.root,a.id",
        ))?;
        let mut rows = statement.query([reference.to_string()])?;
        while let Some(row) = rows.next()? {
            self.check()?;
            let root = row.get::<_, String>(0)?;
            let id = row.get::<_, String>(1)?;
            let revision = RevisionReference {
                root,
                revision: Some(id.clone()),
            };
            if self
                .assertion_revision_currency(&revision)?
                .is_some_and(|state| state.current)
            {
                let fact = get(conn, &id)?.ok_or_else(|| StoreError::NotFound(id))?;
                if visit(fact)?.is_break() {
                    self.check()?;
                    return Ok(ControlFlow::Break(()));
                }
            }
        }
        self.check()?;
        Ok(ControlFlow::Continue(()))
    }
}

#[cfg(test)]
mod tests;
