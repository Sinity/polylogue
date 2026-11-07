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

    /// Eligibility metadata without materializing an assertion's value.
    /// A missing revision is not an unexpired assertion.
    pub fn assertion_is_unexpired(
        &self,
        reference: &RevisionReference,
        at: Timestamp,
    ) -> Result<bool> {
        let Some(exact) = self.resolved_assertion_revision(reference)? else {
            return Ok(false);
        };
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let value = conn.query_row(&source.sql("SELECT EXISTS(SELECT 1 FROM @source.assertion WHERE root=?1 AND id=?2 AND (valid_until IS NULL OR valid_until>?3))"), params![exact.root, exact.revision, at.unix_ms()], |row| row.get(0))?;
        self.check()?;
        Ok(value)
    }

    /// Streams current exact assertion references with a link in `roles`, including
    /// every conflicting head, without materializing bodies. False currency
    /// never becomes a chosen winner.
    pub fn visit_assertion_references_linked(
        &self,
        reference: &Reference,
        roles: &[LinkRole],
        mut visit: impl FnMut(RevisionReference) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let mut statement = conn.prepare_cached(&source.sql(
            "SELECT DISTINCT a.root,a.id FROM @source.link l JOIN @source.assertion a ON a.id=l.record WHERE l.ref=?1 AND l.role IN (SELECT value FROM json_each(?2)) ORDER BY a.root,a.id",
        ))?;
        let roles =
            serde_json::to_string(&roles.iter().map(|role| role.as_str()).collect::<Vec<_>>())?;
        let mut rows = statement.query(params![reference.to_string(), roles])?;
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
                && visit(revision)?.is_break()
            {
                self.check()?;
                return Ok(ControlFlow::Break(()));
            }
        }
        self.check()?;
        Ok(ControlFlow::Continue(()))
    }
}

#[cfg(test)]
mod tests;
