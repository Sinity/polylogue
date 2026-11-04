//! Typed reads of assertions over the stored currency index. Direct
//! association is exact; container membership belongs to the query backend.
use super::*;
use crate::Reader;
use std::ops::ControlFlow;

impl Reader<'_> {
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

    /// Exact revisions remain readable when superseded or withdrawn.
    pub fn assertion(&self, reference: &RevisionReference) -> Result<Option<Assertion>> {
        let Some(exact) = self.resolved_assertion_revision(reference)? else {
            return Ok(None);
        };
        let id = exact
            .revision
            .as_ref()
            .ok_or_else(|| invalid("assertion revision is not exact"))?;
        let row = get(self.conn()?, id)?.filter(|row| row.fact.root == exact.root);
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
            "SELECT a.root,a.id FROM @source.link l JOIN @source.assertion a ON a.id=l.record WHERE l.ref=?1 AND l.role='about' ORDER BY a.root,a.id",
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
