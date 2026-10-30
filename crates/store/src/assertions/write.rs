//! Resolve one accepting snapshot, then freeze facts for direct confirmed apply.
use super::*;
use polylogue_model::{
    assertion::{AssertionPatch, Change, NewAssertion, NewAssertionLink, cardinality, values},
    settings::Owner,
    user::{UserOutcome, UserRequest},
};
use std::collections::{BTreeMap, BTreeSet};

type Hash<'a> = dyn Fn(&Reference) -> Result<Option<BlobHash>> + 'a;

#[derive(Clone, Copy)]
pub(crate) struct WriteContext<'a> {
    pub request: &'a UserRequest,
    pub counter: i64,
    pub hash: &'a Hash<'a>,
    pub owner: Owner,
}

fn exact_fact(conn: &Connection, reference: &Reference) -> Result<AssertionFact> {
    let Reference::Assertion(revision) = reference else {
        return Err(invalid("expected an exact assertion revision"));
    };
    let Some(id) = &revision.revision else {
        return Err(invalid("expected an exact assertion revision"));
    };
    if matches!(get_evidence(conn, id)?, Some(polylogue_model::historical::AssertionEvidence::Historical { fact, .. }) if fact.root == revision.root)
    {
        return Err(StoreError::ReferenceUnavailable {
            reference: reference.to_string(),
            reason: "historical fields are not validated current-kind inputs".into(),
        });
    }
    get(conn, id)?
        .filter(|row| row.fact.root == revision.root)
        .map(|row| row.fact)
        .ok_or_else(|| StoreError::NotFound(reference.to_string()))
}

struct RevisionBase {
    root: String,
    statement: NewAssertion,
    unmapped_links: bool,
}
fn revision_base(conn: &Connection, reference: &Reference) -> Result<RevisionBase> {
    let Reference::Assertion(revision) = reference else {
        return Err(invalid("expected an exact assertion revision"));
    };
    let Some(id) = &revision.revision else {
        return Err(invalid("expected an exact assertion revision"));
    };
    let evidence = get_evidence(conn, id)?
        .filter(|row| row.reference() == *reference)
        .ok_or_else(|| StoreError::NotFound(reference.to_string()))?;
    let (root, kind, key, value, valid_until, links, unmapped_links) = match evidence {
        polylogue_model::historical::AssertionEvidence::Current(row) => {
            let fact = row.fact;
            (
                fact.root,
                fact.kind,
                fact.key,
                fact.value,
                fact.valid_until,
                fact.links,
                false,
            )
        }
        polylogue_model::historical::AssertionEvidence::Historical { fact, .. } => {
            let kind = AssertionKind::from_name(fact.kind.name()).ok_or_else(|| {
                StoreError::ReferenceUnavailable {
                    reference: reference.to_string(),
                    reason: "the historical kind has no current native revision schema".into(),
                }
            })?;
            let unmapped = fact.targets.iter().any(|target| {
                Reference::parse(target).ok().is_none_or(|reference| {
                    !fact
                        .links
                        .iter()
                        .any(|link| link.role == LinkRole::About && link.reference == reference)
                })
            }) || fact.relations.iter().any(|(role, target)| {
                !matches!(
                    role,
                    polylogue_model::historical::RecordedRelation::Known(
                        polylogue_model::historical::Relation::Revises
                            | polylogue_model::historical::Relation::Withdraws
                    )
                ) && Reference::parse(target).is_err()
            });
            (
                fact.root,
                kind,
                fact.key,
                fact.value.map(serde_json::Value::String),
                fact.valid_until,
                fact.links,
                unmapped,
            )
        }
    };
    Ok(RevisionBase {
        root,
        statement: NewAssertion {
            kind,
            key,
            value,
            valid_until,
            links: links
                .into_iter()
                .filter(|link| link.role != LinkRole::Revises)
                .map(|link| NewAssertionLink {
                    role: link.role,
                    reference: link.reference,
                })
                .collect(),
        },
        unmapped_links,
    })
}

fn rubric_reference(conn: &Connection, links: &[NewAssertionLink]) -> Result<Option<Reference>> {
    let mut rubrics = BTreeSet::new();
    for link in links
        .iter()
        .filter(|link| link.role == LinkRole::DerivedFrom)
    {
        if matches!(link.reference, Reference::Assertion(_))
            && exact_fact(conn, &link.reference)?.kind == AssertionKind::Rubric
        {
            rubrics.insert(link.reference.clone());
        }
    }
    if rubrics.len() > 1 {
        return Err(StoreError::Ambiguous {
            reference: "annotation rubric".into(),
            candidates: rubrics.iter().map(ToString::to_string).collect(),
        });
    }
    Ok(rubrics.pop_first())
}

fn validate_value(
    conn: &Connection,
    request: &UserRequest,
    statement: &mut NewAssertion,
    live_value: bool,
    owner: Owner,
    saved: &mut SavedResolver<'_>,
) -> Result<(values::Value, Option<Reference>)> {
    statement
        .validate_links()
        .map_err(|error| invalid(&error.to_string()))?;
    let mut value = if live_value {
        values::decode_live(
            statement.kind,
            statement.key.as_deref(),
            statement.value.as_ref(),
        )
    } else {
        values::decode(
            statement.kind,
            statement.key.as_deref(),
            statement.value.as_ref(),
        )
    }
    .map_err(|error| invalid(&error.to_string()))?;
    if let values::Value::Setting(setting) = &mut value {
        setting.fields = setting
            .fields
            .clone()
            .resolve(
                statement.key.as_deref().expect("setting key checked"),
                &request.machine,
                owner,
            )
            .map_err(|error| invalid(&error.to_string()))?;
        statement.value = Some(serde_json::to_value(setting)?);
    }
    if let values::Value::SavedQuery(value) = &value {
        saved(SavedInput::Canonical {
            value: &value.fields,
            changed: live_value,
        })?;
    }
    let rubric = if statement.kind == AssertionKind::Annotation {
        let reference = rubric_reference(conn, &statement.links)?
            .ok_or_else(|| invalid("annotation needs its exact derived_from rubric"))?;
        let values::Value::Rubric(rubric) = exact_fact(conn, &reference)?
            .validate()
            .map_err(|error| invalid(&error.to_string()))?
        else {
            return Err(invalid("annotation rubric is not a rubric"));
        };
        let values::Value::Annotation(annotation) = &value else {
            return Err(invalid("annotation value is not a field value"));
        };
        let key = statement.key.as_deref().expect("annotation key checked");
        if key == "abstain" {
            if !rubric.fields.abstain || !annotation.fields.value.is_string() {
                return Err(invalid(
                    "not_allowed_by_rubric: abstention requires an allowed text reason",
                ));
            }
        } else {
            let field = rubric
                .fields
                .fields
                .iter()
                .find(|field| field.name == key)
                .ok_or_else(|| invalid("not_allowed_by_rubric: unknown field"))?;
            if !field.allows(&annotation.fields.value) {
                return Err(invalid("not_allowed_by_rubric: field value"));
            }
        }
        if annotation.fields.confidence.is_some() && !rubric.fields.confidence {
            return Err(invalid("not_allowed_by_rubric: confidence"));
        }
        if annotation.fields.rationale.is_some() && rubric.fields.rationale == Some(false) {
            return Err(invalid("not_allowed_by_rubric: rationale"));
        }
        let mut pass_found = false;
        for origin in statement
            .links
            .iter()
            .filter(|link| link.role == LinkRole::Origin)
        {
            let pass = exact_fact(conn, &origin.reference)?;
            if pass.kind != AssertionKind::Pass || pass.author != request.authorship.author {
                return Err(invalid(
                    "annotation origin must be a pass opened by its author",
                ));
            }
            let values::Value::Pass(pass) = pass
                .validate()
                .map_err(|error| invalid(&error.to_string()))?
            else {
                return Err(invalid("annotation origin is not a pass"));
            };
            if pass.fields.rubric.as_ref() != Some(&reference) {
                return Err(invalid("annotation pass names another rubric"));
            }
            pass_found = true;
        }
        if !pass_found {
            return Err(invalid("annotation needs an exact pass origin"));
        }
        Some(reference)
    } else {
        None
    };
    Ok((value, rubric))
}

fn current_bases(conn: &Connection, statement: &NewAssertion) -> Result<Option<String>> {
    for link in statement
        .links
        .iter()
        .filter(|link| link.role == LinkRole::Withdraws)
    {
        let mut root = link.reference.clone();
        let exact = match &mut root {
            Reference::Assertion(revision)
            | Reference::Variant {
                revision,
                part: None,
            } => {
                let exact = revision.revision.is_some();
                revision.revision = None;
                exact
            }
            _ => return Err(invalid("withdrawal needs an assertion or variant identity")),
        };
        let revisions = Graph::new(conn)?.revisions(&root)?;
        if revisions.is_empty() || (exact && !revisions.contains(&link.reference)) {
            return Err(StoreError::NotFound(link.reference.to_string()));
        }
    }
    let mut root = None;
    for link in statement
        .links
        .iter()
        .filter(|link| link.role == LinkRole::Revises)
    {
        let base = revision_base(conn, &link.reference)?;
        if base.statement.kind != statement.kind {
            return Err(invalid("revision cannot change assertion kind"));
        }
        let current = heads(
            conn,
            &Reference::Assertion(RevisionReference {
                root: base.root.clone(),
                revision: None,
            }),
        )?;
        if !current.contains(&link.reference) {
            return Err(StoreError::Stale {
                reference: link.reference.to_string(),
                current: current.iter().map(ToString::to_string).collect(),
            });
        }
        root = Some(base.root);
    }
    Ok(root)
}

fn current_slots(
    conn: &Connection,
    statement: &NewAssertion,
    keys: &[cardinality::Key],
) -> Result<Vec<AssertionFact>> {
    if keys.is_empty() {
        return Ok(Vec::new());
    }
    let source = Source::reading(conn)?;
    // The kind/key index bounds candidates; all revision truth comes from the
    // same Graph/Currency kernel used by typed reads and the query backend.
    let roots = conn
        .prepare_cached(&source.sql(
            "SELECT DISTINCT root FROM @source.assertion WHERE kind=?1 AND key IS ?2 ORDER BY root",
        ))?
        .query_map(params![statement.kind.as_str(), statement.key], |row| {
            row.get::<_, String>(0)
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut matches = BTreeMap::new();
    for root in roots {
        for reference in heads(
            conn,
            &Reference::Assertion(RevisionReference {
                root,
                revision: None,
            }),
        )? {
            let fact = exact_fact(conn, &reference)?;
            let existing = NewAssertion {
                kind: fact.kind,
                key: fact.key.clone(),
                value: fact.value.clone(),
                valid_until: fact.valid_until,
                links: fact
                    .links
                    .iter()
                    .map(|link| NewAssertionLink {
                        role: link.role,
                        reference: link.reference.clone(),
                    })
                    .collect(),
            };
            let value = fact
                .validate()
                .map_err(|error| invalid(&error.to_string()))?;
            let rubric = if existing.kind == AssertionKind::Annotation {
                rubric_reference(conn, &existing.links)?
            } else {
                None
            };
            let slots = existing
                .cardinality_keys(&value, &fact.author, rubric.as_ref())
                .map_err(|error| invalid(&error.to_string()))?;
            if slots.iter().any(|slot| keys.contains(slot)) {
                matches.insert(reference, fact);
            }
        }
    }
    Ok(matches.into_values().collect())
}

#[cfg(test)]
fn unavailable_saved(_: SavedInput<'_>) -> Result<values::SavedQuery> {
    Err(StoreError::ReferenceUnavailable {
        reference: "native saved query".into(),
        reason: "trusted query preparation context is missing".into(),
    })
}

#[cfg(test)]
pub(crate) fn write(
    conn: &Connection,
    request: &UserRequest,
    statements: &[NewAssertion],
    counter: i64,
    hash: &Hash<'_>,
    owner: Owner,
) -> Result<UserOutcome> {
    write_preparing(
        conn,
        statements,
        WriteContext {
            request,
            counter,
            hash,
            owner,
        },
        &mut unavailable_saved,
    )
}

pub(crate) fn write_preparing(
    conn: &Connection,
    statements: &[NewAssertion],
    context: WriteContext<'_>,
    saved: &mut SavedResolver<'_>,
) -> Result<UserOutcome> {
    let mut expanded = Vec::new();
    for statement in statements {
        if statement.kind == AssertionKind::Tag
            && statement
                .links
                .iter()
                .filter(|link| link.role == LinkRole::About)
                .count()
                > 1
        {
            for target in statement
                .links
                .iter()
                .filter(|link| link.role == LinkRole::About)
            {
                let mut single = statement.clone();
                single.links.retain(|link| {
                    link.role != LinkRole::About || link.reference == target.reference
                });
                expanded.push(single);
            }
        } else {
            expanded.push(statement.clone());
        }
    }
    accept(conn, &expanded, true, context, saved)
}
fn accept(
    conn: &Connection,
    statements: &[NewAssertion],
    live_value: bool,
    context: WriteContext<'_>,
    saved: &mut SavedResolver<'_>,
) -> Result<UserOutcome> {
    let WriteContext {
        request,
        counter,
        hash,
        owner,
    } = context;
    if statements.is_empty() {
        return Err(invalid("assert needs at least one statement"));
    }
    let mut first = None;
    let mut inserted = false;
    for (ordinal, original) in statements.iter().enumerate() {
        let mut statement = original.clone();
        let (value, rubric) =
            validate_value(conn, request, &mut statement, live_value, owner, saved)?;
        let mut root = current_bases(conn, &statement)?;
        if root.is_none() {
            let keys = statement
                .cardinality_keys(&value, &request.authorship.author, rubric.as_ref())
                .map_err(|error| invalid(&error.to_string()))?;
            let existing = current_slots(conn, &statement, &keys)?;
            if statement.kind == AssertionKind::Tag && !existing.is_empty() {
                first.get_or_insert_with(|| existing[0].id.clone());
                continue;
            }
            if existing.len() > 1 {
                return Err(StoreError::Ambiguous {
                    reference: statement
                        .key
                        .clone()
                        .unwrap_or_else(|| statement.kind.as_str().into()),
                    candidates: existing
                        .iter()
                        .map(|fact| exact(&fact.root, &fact.id).to_string())
                        .collect(),
                });
            }
            if let Some(previous) = existing.first() {
                root = Some(previous.root.clone());
                statement.links.push(NewAssertionLink {
                    role: LinkRole::Revises,
                    reference: exact(&previous.root, &previous.id),
                });
            }
        }
        let id = if statements.len() == 1 {
            request.id.clone()
        } else {
            format!("{}.{}", request.id, ordinal)
        };
        let mut links = Vec::with_capacity(statement.links.len());
        for link in &statement.links {
            links.push(AssertionLink {
                role: link.role,
                reference: link.reference.clone(),
                seen_hash: if link.role.uses_text_hash() {
                    hash(&link.reference)?
                } else {
                    None
                },
            });
        }
        let fact = AssertionFact::from_request(
            id.clone(),
            root.unwrap_or_else(|| id.clone()),
            statement,
            links,
            request,
        );
        let family = Reference::Assertion(RevisionReference {
            root: fact.root.clone(),
            revision: None,
        });
        apply(conn, &[fact], counter)?;
        // A historical forward link must not turn a new native withdrawal
        // into a cyclic active graph. The caller owns the accepting transaction.
        heads(conn, &family)?;
        first.get_or_insert(id);
        inserted = true;
    }
    Ok(UserOutcome {
        id: first.expect("nonempty assertion batch"),
        repeat: !inserted,
        pending: false,
    })
}

#[cfg(test)]
pub(crate) fn revise(
    conn: &Connection,
    request: &UserRequest,
    bases: &[RevisionReference],
    patch: &AssertionPatch,
    counter: i64,
    hash: &Hash<'_>,
    owner: Owner,
) -> Result<UserOutcome> {
    revise_preparing(
        conn,
        bases,
        patch,
        WriteContext {
            request,
            counter,
            hash,
            owner,
        },
        &mut unavailable_saved,
    )
}

pub(crate) fn revise_preparing(
    conn: &Connection,
    bases: &[RevisionReference],
    patch: &AssertionPatch,
    context: WriteContext<'_>,
    saved: &mut SavedResolver<'_>,
) -> Result<UserOutcome> {
    let statement = revision_statement(conn, bases, patch)?;
    accept(
        conn,
        &[statement],
        !matches!(patch.value, Change::Keep),
        context,
        saved,
    )
}

/// Compose a revision once for preflight and final acceptance. Currency is
/// checked by accept after the shared exact-base lookup and patch semantics.
pub(crate) fn revision_statement(
    conn: &Connection,
    bases: &[RevisionReference],
    patch: &AssertionPatch,
) -> Result<NewAssertion> {
    if bases.is_empty() {
        return Err(invalid("revise needs at least one exact base"));
    }
    let mut facts = Vec::new();
    for base in bases {
        facts.push(revision_base(conn, &Reference::Assertion(base.clone()))?);
    }
    let first = &facts[0];
    if facts
        .iter()
        .any(|fact| fact.root != first.root || fact.statement.kind != first.statement.kind)
    {
        return Err(invalid(
            "one revision must name bases of one assertion family",
        ));
    }
    fn change<T: Clone>(old: Option<T>, change: &Change<T>) -> Option<T> {
        match change {
            Change::Keep => old,
            Change::Set(value) => Some(value.clone()),
            Change::Clear => None,
        }
    }
    if patch.links.is_none() && facts.iter().any(|base| base.unmapped_links) {
        return Err(StoreError::ReferenceUnavailable {
            reference: bases
                .iter()
                .map(|base| Reference::Assertion(base.clone()).to_string())
                .collect::<Vec<_>>()
                .join(", "),
            reason: "historical targets or relations need an explicit canonical link replacement"
                .into(),
        });
    }
    let mut statement = NewAssertion {
        kind: first.statement.kind,
        key: change(first.statement.key.clone(), &patch.key),
        value: change(first.statement.value.clone(), &patch.value),
        valid_until: change(first.statement.valid_until, &patch.valid_until),
        links: patch.links.clone().unwrap_or_else(|| {
            first
                .statement
                .links
                .iter()
                .filter(|link| link.role != LinkRole::Revises)
                .map(|link| NewAssertionLink {
                    role: link.role,
                    reference: link.reference.clone(),
                })
                .collect()
        }),
    };
    if statement
        .links
        .iter()
        .any(|link| link.role == LinkRole::Revises)
    {
        return Err(invalid(
            "revision bases belong in bases, not the link patch",
        ));
    }
    for base in bases {
        statement.links.push(NewAssertionLink {
            role: LinkRole::Revises,
            reference: Reference::Assertion(base.clone()),
        });
    }
    Ok(statement)
}

#[cfg(test)]
mod tests;
