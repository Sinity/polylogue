//! Whole immutable authored requests, with local receipts kept distinct from native proof.
use super::Reader;
use crate::{Result, StoreError};
use polylogue_model::{
    Reference, RevisionReference, authored::PreparedRequest, bundle::AuthoredRecord,
};
use rusqlite::{Connection, OptionalExtension};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredReadIssue {
    #[error("the selected family needs an exact canonical revision")]
    UnresolvedRevision,
    #[error("the selected effect has no retained immutable request association")]
    MissingAssociation,
    #[error("the selected effect has competing immutable request associations")]
    CompetingAssociations,
    #[error("the authored request has no retained positive local admission")]
    LocalAdmissionUnavailable,
    #[error("the accepted request has incomplete retained body, outcome or effects")]
    MissingContent,
}

/// Local admission is not a Sinex event or a certificate for a native source.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthoredReadEvidence {
    Local {
        selected: Reference,
        record: Box<AuthoredRecord>,
    },
    Incomplete {
        issue: AuthoredReadIssue,
    },
}

enum Target {
    Assertion {
        id: String,
        root: String,
        selected: Reference,
    },
    Result {
        id: String,
        selected: Reference,
    },
    Delivery {
        id: String,
        selected: Reference,
    },
    Variant {
        id: String,
        part: Option<u64>,
        selected: Reference,
    },
}
impl Target {
    fn selected(&self) -> &Reference {
        match self {
            Self::Assertion { selected, .. }
            | Self::Result { selected, .. }
            | Self::Delivery { selected, .. }
            | Self::Variant { selected, .. } => selected,
        }
    }
    fn matches(&self, prepared: &PreparedRequest) -> bool {
        self.matches_effect(&prepared.effect)
    }
    fn matches_effect(&self, effect: &polylogue_model::authored::AuthoredEffect) -> bool {
        match self {
            Self::Assertion { id, root, .. } => effect
                .assertions
                .iter()
                .any(|value| value.id == *id && value.root == *root),
            Self::Result { id, .. } => effect.kept_results.iter().any(|value| value.id == *id),
            Self::Delivery { id, .. } => effect
                .deliveries
                .iter()
                .any(|value| value.delivery.id == *id),
            Self::Variant { id, part, .. } => match part {
                None => effect.variants.variants.iter().any(|value| value.id == *id),
                Some(part) => effect
                    .variants
                    .variant_parts
                    .iter()
                    .any(|value| value.variant == *id && value.n == *part),
            },
        }
    }
    fn candidate_key(&self) -> (&'static str, &str, String) {
        match self {
            Self::Assertion { id, root, .. } => ("assertion", id, root.clone()),
            Self::Result { id, .. } => ("result", id, String::new()),
            Self::Delivery { id, .. } => ("delivery", id, String::new()),
            Self::Variant { id, part: None, .. } => ("variant", id, String::new()),
            Self::Variant {
                id,
                part: Some(part),
                ..
            } => ("variant_part", id, part.to_string()),
        }
    }
}

impl Reader<'_> {
    /// Expands a selected authored effect only to its original whole request.
    /// Primary evidence stays incomplete until the source owner supplies its full proof.
    pub fn authored_record(&self, reference: &Reference) -> Result<Option<AuthoredReadEvidence>> {
        if !matches!(
            reference,
            Reference::Request(_)
                | Reference::Assertion(_)
                | Reference::Result { .. }
                | Reference::Delivery(_)
                | Reference::Variant { .. }
        ) {
            return Err(StoreError::ReferenceUnavailable {
                reference: reference.to_string(),
                reason: "reference has no authored request grain".into(),
            });
        }
        self.conn()?;
        if let Reference::Request(id) = reference {
            let result = self.exact_authored_request(id);
            self.check()?;
            return result;
        }
        let target = match self.authored_target(reference)? {
            Ok(Some(target)) => target,
            Ok(None) => return Ok(None),
            Err(issue) => return Ok(Some(AuthoredReadEvidence::Incomplete { issue })),
        };
        let conn = self.conn()?;
        let mut found = None;
        prepare_membership(conn, &mut || self.check())?;
        let (kind, id, member) = target.candidate_key();
        let mut statement = conn.prepare_cached("SELECT request FROM temp.authored_effect_request WHERE kind=?1 AND id=?2 AND member=?3 ORDER BY request")?;
        let mut rows = statement.query(rusqlite::params![kind, id, member])?;
        while let Some(row) = rows.next()? {
            self.check()?;
            let id: String = row.get(0)?;
            let Some(request) = self.user_request(&id)? else {
                continue;
            };
            let Some(prepared) = crate::primary::authored::frozen_request(conn, &request.id)?
            else {
                continue;
            };
            if target.matches(&prepared) {
                if found.is_some() {
                    return Ok(Some(AuthoredReadEvidence::Incomplete {
                        issue: AuthoredReadIssue::CompetingAssociations,
                    }));
                }
                found = Some(prepared);
            }
        }
        drop(rows);
        drop(statement);
        let Some(prepared) = found else {
            return Ok(Some(AuthoredReadEvidence::Incomplete {
                issue: AuthoredReadIssue::MissingAssociation,
            }));
        };
        let commit: i64 = conn.query_row(
            "SELECT commit_seq FROM source_client_request WHERE id=?1",
            [&prepared.request.id],
            |row| row.get(0),
        )?;
        if commit <= 0 {
            return Ok(Some(AuthoredReadEvidence::Incomplete {
                issue: AuthoredReadIssue::LocalAdmissionUnavailable,
            }));
        }
        let record = AuthoredRecord {
            reference: format!("request:{}", prepared.request.id),
            commit: commit as u64,
            prepared,
        };
        self.conn()?;
        Ok(Some(AuthoredReadEvidence::Local {
            selected: target.selected().clone(),
            record: Box::new(record),
        }))
    }

    fn exact_authored_request(&self, id: &str) -> Result<Option<AuthoredReadEvidence>> {
        let conn = self.conn()?;
        let commit: Option<i64> = conn
            .query_row(
                "SELECT commit_seq FROM source_client_request WHERE id=?1",
                [id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(commit) = commit else {
            self.check()?;
            return Ok(None);
        };
        if commit <= 0 {
            self.check()?;
            return Ok(Some(AuthoredReadEvidence::Incomplete {
                issue: AuthoredReadIssue::LocalAdmissionUnavailable,
            }));
        }
        let selected = Reference::Request(id.into());
        let reference = selected.to_string();
        let Some(prepared) = crate::primary::authored::frozen_request(conn, id)? else {
            return Ok(Some(AuthoredReadEvidence::Incomplete {
                issue: AuthoredReadIssue::MissingContent,
            }));
        };
        self.check()?;
        Ok(Some(AuthoredReadEvidence::Local {
            selected,
            record: Box::new(AuthoredRecord {
                reference,
                commit: commit as u64,
                prepared,
            }),
        }))
    }

    fn authored_target(
        &self,
        reference: &Reference,
    ) -> Result<std::result::Result<Option<Target>, AuthoredReadIssue>> {
        let conn = self.conn()?;
        Ok(Ok(match reference {
            Reference::Assertion(revision) => {
                let Some(id) = &revision.revision else {
                    return Ok(Err(AuthoredReadIssue::UnresolvedRevision));
                };
                if !conn
                    .prepare_cached("SELECT 1 FROM source_assertion WHERE id=?1 AND root=?2")?
                    .exists(rusqlite::params![id, revision.root])?
                {
                    return Ok(Ok(None));
                }
                let pinned = RevisionReference::exact(revision.root.clone(), id.clone());
                let selected = Reference::Assertion(pinned);
                Some(Target::Assertion {
                    id: id.clone(),
                    root: revision.root.clone(),
                    selected,
                })
            }

            Reference::Result { id, .. } => {
                if !conn
                    .prepare_cached("SELECT 1 FROM source_kept_result WHERE id=?1")?
                    .exists([id])?
                {
                    return Ok(Ok(None));
                }
                Some(Target::Result {
                    id: id.clone(),
                    selected: reference.clone(),
                })
            }
            Reference::Delivery(id) => {
                if !conn
                    .prepare_cached("SELECT 1 FROM source_delivery WHERE id=?1")?
                    .exists([id])?
                {
                    return Ok(Ok(None));
                }
                Some(Target::Delivery {
                    id: id.clone(),
                    selected: reference.clone(),
                })
            }
            Reference::Variant { revision, part } => {
                let Some(id) = &revision.revision else {
                    return Ok(Err(AuthoredReadIssue::UnresolvedRevision));
                };
                let Some(variant) = self
                    .variant(id)?
                    .filter(|variant| variant.root == revision.root)
                else {
                    return Ok(Ok(None));
                };
                Some(Target::Variant {
                    id: variant.id,
                    part: *part,
                    selected: reference.clone(),
                })
            }
            _ => {
                return Err(StoreError::ReferenceUnavailable {
                    reference: reference.to_string(),
                    reason: "reference has no authored request grain".into(),
                });
            }
        }))
    }
}

/// This index belongs to the Reader transaction and disappears on checkout
/// rollback. Membership comes only from retained immutable effects; counters
/// identify their local admission and cannot establish a family association.
fn prepare_membership(conn: &Connection, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    if conn.is_autocommit() {
        return Err(StoreError::Invalid(
            "authored membership requires a pinned Reader".into(),
        ));
    }
    conn.execute_batch("CREATE TEMP TABLE IF NOT EXISTS authored_effect_request(kind TEXT NOT NULL,id TEXT NOT NULL,member TEXT NOT NULL,request TEXT NOT NULL,PRIMARY KEY(kind,id,member,request)) STRICT, WITHOUT ROWID;
        CREATE TEMP TABLE IF NOT EXISTS authored_membership_ready(singleton INTEGER PRIMARY KEY CHECK(singleton=1)) STRICT;")?;
    if conn
        .prepare_cached("SELECT 1 FROM temp.authored_membership_ready")?
        .exists([])?
    {
        return check();
    }
    conn.execute("DELETE FROM temp.authored_effect_request", [])?;
    let mut insert=conn.prepare_cached("INSERT OR IGNORE INTO temp.authored_effect_request(kind,id,member,request) VALUES(?1,?2,?3,?4)")?;
    crate::primary::authored::visit(conn, &mut |prepared| {
        index_effect(&mut insert, &prepared.request.id, &prepared.effect, check)
    })?;
    check()?;
    conn.execute("INSERT INTO temp.authored_membership_ready VALUES(1)", [])?;
    Ok(())
}

fn index_effect(
    insert: &mut rusqlite::Statement<'_>,
    request: &str,
    effect: &polylogue_model::authored::AuthoredEffect,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    let mut add = |kind: &str, id: &str, member: &str| -> Result<()> {
        check()?;
        insert.execute(rusqlite::params![kind, id, member, request])?;
        Ok(())
    };
    for fact in &effect.assertions {
        add("assertion", &fact.id, &fact.root)?;
    }
    for result in &effect.kept_results {
        add("result", &result.id, "")?;
    }
    for delivery in &effect.deliveries {
        add("delivery", &delivery.delivery.id, "")?;
    }
    for variant in &effect.variants.variants {
        add("variant", &variant.id, "")?;
    }
    for part in &effect.variants.variant_parts {
        add("variant_part", &part.variant, &part.n.to_string())?;
    }
    Ok(())
}
