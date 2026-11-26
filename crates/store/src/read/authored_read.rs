//! Whole immutable authored requests. A local commit is kept distinct from an
//! event Sinex confirmed.
use super::Reader;
use crate::ConnExt as _;
use crate::row::Rows as _;
use crate::{Result, StoreError};
use polylogue_model::{
    Reference, RevisionReference, authored::PreparedRequest, bundle::AuthoredRecord,
};
use rusqlite::Connection;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredReadIssue {
    #[error("the selected family needs one current revision")]
    UnresolvedRevision,
    #[error("the selected effect has no kept immutable request association")]
    MissingAssociation,
    #[error("the selected effect has competing immutable request associations")]
    CompetingAssociations,
    #[error("the authored request has no stored positive local commit position")]
    LocalCommitUnavailable,
    #[error("the accepted request has incomplete stored body, outcome or effects")]
    MissingContent,
}

/// A local commit is not a Sinex event and does not show which source wrote it.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthoredRead {
    Local {
        selected: Reference,
        record: Box<AuthoredRecord>,
    },
    Incomplete {
        issue: AuthoredReadIssue,
    },
}

enum Target {
    Annotation {
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
    Alternate {
        id: String,
        part: Option<u64>,
        selected: Reference,
    },
}
impl Target {
    fn selected(&self) -> &Reference {
        match self {
            Self::Annotation { selected, .. }
            | Self::Result { selected, .. }
            | Self::Delivery { selected, .. }
            | Self::Alternate { selected, .. } => selected,
        }
    }
    fn matches(&self, prepared: &PreparedRequest) -> bool {
        self.matches_effect(&prepared.effect)
    }
    fn matches_effect(&self, effect: &polylogue_model::authored::AuthoredEffect) -> bool {
        match self {
            Self::Annotation { id, root, .. } => effect
                .annotations
                .iter()
                .any(|value| value.id == *id && value.root == *root),
            Self::Result { id, .. } => effect.kept_results.iter().any(|value| value.id == *id),
            Self::Delivery { id, .. } => effect
                .deliveries
                .iter()
                .any(|value| value.delivery.id == *id),
            Self::Alternate { id, part, .. } => match part {
                None => effect
                    .alternates
                    .alternates
                    .iter()
                    .any(|value| value.id == *id),
                Some(part) => effect
                    .alternates
                    .alternate_parts
                    .iter()
                    .any(|value| value.alternate == *id && value.n == *part),
            },
        }
    }
    fn candidate_key(&self) -> (&'static str, &str, String) {
        match self {
            Self::Annotation { id, root, .. } => ("annotation", id, root.clone()),
            Self::Result { id, .. } => ("result", id, String::new()),
            Self::Delivery { id, .. } => ("delivery", id, String::new()),
            Self::Alternate { id, part: None, .. } => ("alternate", id, String::new()),
            Self::Alternate {
                id,
                part: Some(part),
                ..
            } => ("alternate_part", id, part.to_string()),
        }
    }
}

impl Reader<'_> {
    /// Expands a selected authored effect only to its original whole request.
    /// In primary mode the record stays incomplete until Sinex supplies all of it.
    pub fn authored_record(&self, reference: &Reference) -> Result<Option<AuthoredRead>> {
        if !matches!(
            reference,
            Reference::Request(_)
                | Reference::Annotation(_)
                | Reference::Result { .. }
                | Reference::Delivery(_)
                | Reference::Alternate { .. }
        ) {
            return Err(StoreError::ReferenceUnavailable {
                reference: reference.to_string(),
                reason: "reference has no authored request grain".into(),
            });
        }
        self.conn()?;
        if let Reference::Request(id) = reference {
            let result = self.whole_authored_request(id);
            self.check()?;
            return result;
        }
        let target = match self.authored_target(reference)? {
            Ok(Some(target)) => target,
            Ok(None) => return Ok(None),
            Err(issue) => return Ok(Some(AuthoredRead::Incomplete { issue })),
        };
        let conn = self.conn()?;
        let mut found = None;
        prepare_membership(conn, &mut || self.check())?;
        let (kind, id, member) = target.candidate_key();
        let requests: Vec<String> = conn.values("SELECT request FROM temp.authored_effect_request WHERE kind=?1 AND id=?2 AND member=?3 ORDER BY request", rusqlite::params![kind, id, member])?;
        for id in requests {
            self.check()?;
            let Some(request) = self.user_request(&id)? else {
                continue;
            };
            let Some(prepared) = crate::authored_records::stored_request(conn, &request.id)? else {
                continue;
            };
            if target.matches(&prepared) {
                if found.is_some() {
                    return Ok(Some(AuthoredRead::Incomplete {
                        issue: AuthoredReadIssue::CompetingAssociations,
                    }));
                }
                found = Some(prepared);
            }
        }
        let Some(prepared) = found else {
            return Ok(Some(AuthoredRead::Incomplete {
                issue: AuthoredReadIssue::MissingAssociation,
            }));
        };
        let commit: i64 = conn.scalar(
            "SELECT commit_seq FROM source_client_request WHERE id=?1",
            [&prepared.request.id],
        )?;
        if commit <= 0 {
            return Ok(Some(AuthoredRead::Incomplete {
                issue: AuthoredReadIssue::LocalCommitUnavailable,
            }));
        }
        let record = AuthoredRecord {
            reference: format!("request:{}", prepared.request.id),
            commit: commit as u64,
            prepared,
        };
        self.conn()?;
        Ok(Some(AuthoredRead::Local {
            selected: target.selected().clone(),
            record: Box::new(record),
        }))
    }

    fn whole_authored_request(&self, id: &str) -> Result<Option<AuthoredRead>> {
        let conn = self.conn()?;
        let commit: Option<i64> = conn.scalar_optional(
            "SELECT commit_seq FROM source_client_request WHERE id=?1",
            [id],
        )?;
        let Some(commit) = commit else {
            self.check()?;
            return Ok(None);
        };
        if commit <= 0 {
            self.check()?;
            return Ok(Some(AuthoredRead::Incomplete {
                issue: AuthoredReadIssue::LocalCommitUnavailable,
            }));
        }
        let selected = Reference::Request(id.into());
        let reference = selected.to_string();
        let Some(prepared) = crate::authored_records::stored_request(conn, id)? else {
            return Ok(Some(AuthoredRead::Incomplete {
                issue: AuthoredReadIssue::MissingContent,
            }));
        };
        self.check()?;
        Ok(Some(AuthoredRead::Local {
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
            Reference::Annotation(revision) => {
                let Some(id) = &revision.revision else {
                    return Ok(Err(AuthoredReadIssue::UnresolvedRevision));
                };
                if !conn
                    .prepare_cached("SELECT 1 FROM source_annotation WHERE id=?1 AND root=?2")?
                    .exists(rusqlite::params![id, revision.root])?
                {
                    return Ok(Ok(None));
                }
                let pinned = RevisionReference::exact(revision.root.clone(), id.clone());
                let selected = Reference::Annotation(pinned);
                Some(Target::Annotation {
                    id: id.clone(),
                    root: revision.root.clone(),
                    selected,
                })
            }

            Reference::Result { id, .. } => {
                if !conn
                    .prepare_cached("SELECT 1 FROM source_saved_rows WHERE id=?1")?
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
            Reference::Alternate { revision, part } => {
                let Some(id) = &revision.revision else {
                    return Ok(Err(AuthoredReadIssue::UnresolvedRevision));
                };
                let Some(alternate) = self
                    .alternate(id)?
                    .filter(|alternate| alternate.root == revision.root)
                else {
                    return Ok(Ok(None));
                };
                Some(Target::Alternate {
                    id: alternate.id,
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
/// rollback. Membership comes only from kept immutable effects; counters
/// identify their local commit and cannot establish a family association.
fn prepare_membership(conn: &Connection, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    if conn.is_autocommit() {
        return Err(StoreError::internal(
            "authored membership requires a pinned Reader",
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
    crate::authored_records::visit(conn, &mut |prepared| {
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
    for fact in &effect.annotations {
        add("annotation", &fact.id, &fact.root)?;
    }
    for result in &effect.kept_results {
        add("result", &result.id, "")?;
    }
    for delivery in &effect.deliveries {
        add("delivery", &delivery.delivery.id, "")?;
    }
    for alternate in &effect.alternates.alternates {
        add("alternate", &alternate.id, "")?;
    }
    for part in &effect.alternates.alternate_parts {
        add("alternate_part", &part.alternate, &part.n.to_string())?;
    }
    Ok(())
}
