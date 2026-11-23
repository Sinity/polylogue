//! Typed reads of assertions over the stored current-revision index. Message
//! annotations include spans through their original typed containers.
use super::*;
use crate::ConnExt as _;
use crate::Reader;
use std::ops::ControlFlow;

impl Reader<'_> {
    /// Uses the write's declared uniqueness slots, so reads and acceptance
    /// agree on a tag's value and author semantics.
    pub fn tagged(
        &self,
        targets: &[Reference],
        key: &str,
        value: Option<&serde_json::Value>,
        author: &Author,
    ) -> Result<BTreeSet<Reference>> {
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let raw = value.map(|value| serde_json::json!({"value": value}));
        let value =
            polylogue_model::assertion::values::decode(AssertionKind::Tag, Some(key), raw.as_ref())
                .map_err(|error| StoreError::invalid(error.to_string()))?;
        let mut statement = conn.prepare_cached(&source.sql(
            "SELECT EXISTS(SELECT 1 FROM @source.assertion_slot s JOIN @source.revision_state r ON r.id=s.id WHERE s.slot=?1 AND r.current=1 AND r.withdrawn=0)",
        ))?;
        let mut found = BTreeSet::new();
        for target in targets {
            self.check()?;
            let unique = polylogue_model::assertion::uniqueness::unique(
                AssertionKind::Tag,
                Some(key),
                &value,
                [(LinkRole::About, target)],
                author,
                None,
            )
            .map_err(|error| StoreError::invalid(error.to_string()))?
            .ok_or_else(|| StoreError::invalid("a tag has no uniqueness slots"))?;
            for slot in unique.slots {
                let held: bool =
                    statement.query_row([slot.digest().as_slice()], |row| row.get(0))?;
                if held {
                    found.insert(target.clone());
                    break;
                }
            }
        }
        self.check()?;
        Ok(found)
    }

    /// Current assertions About a message or a span of its original message/block text.
    /// Indexed address ranges nominate candidates; typed containers decide membership.
    pub fn visit_message_annotation_references(
        &self,
        message: &polylogue_model::MessageId,
        visit: impl FnMut(RevisionReference) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        self.visit_message_annotation_references_at(message, None, visit)
    }

    /// The same About candidates and revision graph, bounded by local commit
    /// position or known authored time. Unknown authored times stay unknown.
    pub fn visit_message_annotation_references_at(
        &self,
        message: &polylogue_model::MessageId,
        at: Option<polylogue_model::read::AnnotationAt>,
        mut visit: impl FnMut(RevisionReference) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let reference = Reference::Message(message.clone());
        let message_prefix = format!("span:{}#", reference.stored_form());
        let block_prefix = format!("span:block:{message}#");
        let message_end = format!("span:{}$", reference.stored_form());
        let block_end = format!("span:block:{message}$");
        let (through, until) = match at {
            Some(polylogue_model::read::AnnotationAt::Position(through)) => (Some(through), None),
            Some(polylogue_model::read::AnnotationAt::Time(until)) => (None, Some(until.unix_ms())),
            None => (None, None),
        };
        let sql = if at.is_some() {
            "SELECT a.root,a.id,l.ref FROM @source.link l JOIN @source.assertion a ON a.id=l.record WHERE l.role='about' AND (?6 IS NULL OR a.commit_seq<=?6) AND (?7 IS NULL OR a.created_at<=?7) AND (l.ref=?1 OR (l.ref>=?2 AND l.ref<?3) OR (l.ref>=?4 AND l.ref<?5)) ORDER BY a.root,a.id,l.ref"
        } else {
            "SELECT a.root,a.id,l.ref FROM @source.link l JOIN @source.assertion a ON a.id=l.record JOIN @source.revision_state r ON r.id=a.id WHERE l.role='about' AND r.current=1 AND r.withdrawn=0 AND ?6 IS NULL AND ?7 IS NULL AND (l.ref=?1 OR (l.ref>=?2 AND l.ref<?3) OR (l.ref>=?4 AND l.ref<?5)) ORDER BY a.root,a.id,l.ref"
        };
        let mut statement = conn.prepare_cached(&source.sql(sql))?;
        let mut links = at
            .map(|at| StoredLinks::at_annotation(conn, at))
            .transpose()?;
        let mut graph = links.as_mut().map(Currency::new);
        let mut rows = statement.query(params![
            reference.stored_form(),
            message_prefix,
            message_end,
            block_prefix,
            block_end,
            through,
            until
        ])?;
        let mut previous = None;
        while let Some(row) = rows.next()? {
            self.check()?;
            let linked = row.get::<_, String>(2)?;
            let mut target = Reference::parse_stored(&linked)
                .map_err(|error| StoreError::invalid(error.to_string()))?;
            while target != reference {
                let Some(container) = target.container() else {
                    break;
                };
                target = container;
            }
            if target != reference {
                continue;
            }
            let revision = RevisionReference::exact(row.get(0)?, row.get(1)?);
            if previous.as_ref() != Some(&revision) {
                previous = Some(revision.clone());
                if let Some(graph) = &mut graph
                    && !graph
                        .current(&authored::Revision::new(
                            &revision.root,
                            revision.revision.as_ref().unwrap(),
                        ))
                        .map_err(currency_error)?
                {
                    continue;
                }
                if visit(revision)?.is_break() {
                    self.check()?;
                    return Ok(ControlFlow::Break(()));
                }
            }
        }
        self.check()?;
        Ok(ControlFlow::Continue(()))
    }

    fn resolved_assertion_revision(
        &self,
        reference: &RevisionReference,
    ) -> Result<Option<RevisionReference>> {
        if reference.revision.is_some() {
            return Ok(Some(reference.clone()));
        }
        let Some(state) = self.assertion_family_state(reference)? else {
            return Ok(None);
        };
        if state.conflict() {
            return Err(StoreError::Ambiguous {
                reference: Reference::Assertion(reference.clone()).to_string(),
                candidates: state.heads.iter().map(ToString::to_string).collect(),
            });
        }
        match state.heads.into_iter().next() {
            Some(Reference::Assertion(exact)) => Ok(Some(exact)),
            None => Ok(None),
            Some(_) => Err(crate::StoreError::invalid(
                "assertion family contains another reference kind",
            )),
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
            .ok_or_else(|| crate::StoreError::invalid("assertion revision is not exact"))?;
        let row = get(self.conn()?, id)?.filter(|row| row.fact.root == exact.root);
        self.check()?;
        Ok(row)
    }

    /// Whether an assertion is unexpired, without loading its value.
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
        let value = conn.scalar(&source.sql("SELECT EXISTS(SELECT 1 FROM @source.assertion WHERE root=?1 AND id=?2 AND (valid_until IS NULL OR valid_until>?3))"), params![exact.root, exact.revision, at.unix_ms()])?;
        self.check()?;
        Ok(value)
    }

    /// Streams current exact assertion references with a link in `roles`, including
    /// every conflicting head, without loading bodies. A revision that is not current
    /// never becomes a chosen winner.
    pub fn visit_assertion_references_linked(
        &self,
        reference: &Reference,
        roles: &[LinkRole],
        visit: impl FnMut(RevisionReference) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        self.visit_assertion_references_linked_at(reference, roles, None, visit)
    }

    pub fn visit_assertion_references_linked_at(
        &self,
        reference: &Reference,
        roles: &[LinkRole],
        at: Option<polylogue_model::read::AnnotationAt>,
        mut visit: impl FnMut(RevisionReference) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let (through, until) = match at {
            Some(polylogue_model::read::AnnotationAt::Position(through)) => (Some(through), None),
            Some(polylogue_model::read::AnnotationAt::Time(until)) => (None, Some(until.unix_ms())),
            None => (None, None),
        };
        let mut statement = conn.prepare_cached(&source.sql(
            "SELECT DISTINCT a.root,a.id FROM @source.link l JOIN @source.assertion a ON a.id=l.record WHERE l.ref=?1 AND l.role IN (SELECT value FROM json_each(?2)) AND (?3 IS NULL OR a.commit_seq<=?3) AND (?4 IS NULL OR a.created_at<=?4) ORDER BY a.root,a.id",
        ))?;
        let roles =
            serde_json::to_string(&roles.iter().map(|role| role.as_str()).collect::<Vec<_>>())?;
        let mut rows = statement.query(params![reference.stored_form(), roles, through, until])?;
        let mut links = at
            .map(|at| StoredLinks::at_annotation(conn, at))
            .transpose()?;
        let mut graph = links.as_mut().map(Currency::new);
        while let Some(row) = rows.next()? {
            self.check()?;
            let root = row.get::<_, String>(0)?;
            let id = row.get::<_, String>(1)?;
            let revision = RevisionReference {
                root,
                revision: Some(id.clone()),
            };
            let current = if let Some(graph) = &mut graph {
                graph
                    .current(&authored::Revision::new(&revision.root, &id))
                    .map_err(currency_error)?
            } else {
                self.assertion_revision_state(&revision)?
                    .is_some_and(|state| state.current)
            };
            if current && visit(revision)?.is_break() {
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
