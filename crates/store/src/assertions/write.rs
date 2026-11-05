//! The one assertion write path. Each statement is validated against its
//! kind, then the shared request rules place it: explicit bases must be
//! current, a write without bases goes where its uniqueness slots say, and
//! the writes of one request are numbered and committed together. The stored
//! currency index is recomputed for the roots the request affects.
use super::*;
use authored::{Arrival, OnStale, Placed, Plan, Refusal, Refused, Write};
use polylogue_model::{
    assertion::{AssertionPatch, Change, NewAssertion, NewAssertionLink, values},
    settings::Owner,
    user::{UserOutcome, UserRequest},
};

type Hash<'a> = dyn Fn(&Reference) -> Result<Option<BlobHash>> + 'a;
type QuoteOf<'a> = dyn Fn(&Reference) -> Result<Option<Quote>> + 'a;

#[derive(Clone, Copy)]
pub(crate) struct WriteContext<'a> {
    pub request: &'a UserRequest,
    pub counter: i64,
    pub hash: &'a Hash<'a>,
    pub quote: &'a QuoteOf<'a>,
    pub owner: Owner,
}

fn exact_fact(conn: &Connection, reference: &Reference) -> Result<AssertionFact> {
    let Reference::Assertion(RevisionReference {
        root,
        revision: Some(id),
    }) = reference
    else {
        return Err(invalid("expected an exact assertion revision"));
    };
    get(conn, id)?
        .filter(|row| row.fact.root == *root)
        .map(|row| row.fact)
        .ok_or_else(|| StoreError::NotFound(reference.to_string()))
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
        let reference = rubric_of(
            conn,
            statement
                .links
                .iter()
                .map(|link| (link.role, &link.reference)),
        )?
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

/// Withdrawals of variants are checked here; withdrawals of assertions are
/// checked by the request rules, which see this request's earlier writes.
fn check_variant_targets(conn: &Connection, statement: &NewAssertion) -> Result<()> {
    for link in &statement.links {
        let (
            LinkRole::Withdraws,
            Reference::Variant {
                revision,
                part: None,
            },
        ) = (link.role, &link.reference)
        else {
            continue;
        };
        let revisions = VariantLinks::new(conn)?.revisions(&revision.root)?;
        let found = match &revision.revision {
            None => !revisions.is_empty(),
            Some(id) => revisions.iter().any(|known| known.rev == *id),
        };
        if !found {
            return Err(StoreError::NotFound(link.reference.to_string()));
        }
    }
    Ok(())
}

/// A statement ready to place: its validated content and its slots.
struct Prepared {
    statement: NewAssertion,
    unique: Option<authored::Unique>,
}

fn refusal(kind: &NewAssertion, refused: Refused<StoreError>) -> StoreError {
    let list = |revisions: Vec<Revision>| -> Vec<String> {
        revisions.iter().map(|r| reference(r).to_string()).collect()
    };
    match refused.refusal {
        Refusal::Stale { base, heads } => StoreError::Stale {
            reference: reference(&base).to_string(),
            current: list(heads),
        },
        Refusal::UnknownBase(base) => StoreError::NotFound(reference(&base).to_string()),
        Refusal::UnknownTarget(target) => {
            StoreError::NotFound(target_reference(&target).to_string())
        }
        Refusal::BasesSpanRoots(_) => {
            invalid("one revision must name bases of one assertion family")
        }
        Refusal::Ambiguous { holders } => StoreError::Ambiguous {
            reference: kind
                .key
                .clone()
                .unwrap_or_else(|| kind.kind.as_str().into()),
            candidates: list(holders),
        },
        Refusal::SlotTaken { holders } => StoreError::Conflict(format!(
            "the revised {} would take a slot held by {}",
            kind.kind.as_str(),
            list(holders).join(", ")
        )),
        Refusal::Count { declared, written } => invalid(&format!(
            "the request declared {declared} writes and carries {written}"
        )),
        Refusal::Abandoned(write) => invalid(&format!("write {write} of this request was refused")),
        Refusal::Cycle(revision) => invalid(&format!("withdrawals of {revision} form a cycle")),
        Refusal::Links(error) => error,
    }
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
            quote: &|_| Ok(None),
            owner,
        },
        &mut unavailable_saved,
    )
}

/// Writes statements as one request. A tag about several targets is one
/// write per target, so each target's tag is its own record.
pub(crate) fn write_preparing(
    conn: &Connection,
    statements: &[NewAssertion],
    context: WriteContext<'_>,
    saved: &mut SavedResolver<'_>,
) -> Result<UserOutcome> {
    let mut expanded = Vec::new();
    for statement in statements {
        // Targets a relation names are about-targets of every copy.
        let relations: Vec<&Reference> = statement
            .links
            .iter()
            .filter(|link| link.role.is_relation())
            .map(|link| &link.reference)
            .collect();
        let about = statement
            .links
            .iter()
            .filter(|link| link.role == LinkRole::About && !relations.contains(&&link.reference))
            .count();
        if statement.kind == AssertionKind::Tag && about > 1 {
            for target in statement.links.iter().filter(|link| {
                link.role == LinkRole::About && !relations.contains(&&link.reference)
            }) {
                let mut single = statement.clone();
                single.links.retain(|link| {
                    link.role != LinkRole::About
                        || link.reference == target.reference
                        || relations.contains(&&link.reference)
                });
                expanded.push(single);
            }
        } else {
            expanded.push(statement.clone());
        }
    }
    accept(conn, &expanded, true, context, saved)
}

/// Inserts one write of the request being checked, under the ID it will
/// take, recomputes its family's currency and returns the family root. Only
/// `accept` calls this, and it removes every preview again.
fn preview(
    conn: &Connection,
    request: &UserRequest,
    id: &str,
    statement: &NewAssertion,
    counter: i64,
) -> Result<String> {
    let mut root = id.to_owned();
    for link in &statement.links {
        if let (LinkRole::Revises, Reference::Assertion(base)) = (link.role, &link.reference) {
            root = base.root.clone();
        }
    }
    let links = statement
        .links
        .iter()
        .map(|link| AssertionLink {
            role: link.role,
            reference: link.reference.clone(),
            seen_hash: None,
            quote: None,
        })
        .collect();
    let fact =
        AssertionFact::from_request(id.into(), root.clone(), statement.clone(), links, request);
    insert(conn, Source::of(conn)?, &fact, &[], counter)?;
    currency::recompute(conn, [root.clone()].into())?;
    Ok(root)
}

/// Removes previewed writes and restores their families' currency.
fn remove_previews(conn: &Connection, previews: Vec<(String, String)>) -> Result<()> {
    if previews.is_empty() {
        return Ok(());
    }
    let source = Source::of(conn)?;
    let mut roots = std::collections::BTreeSet::new();
    for (id, root) in previews {
        conn.execute(
            &source.sql("DELETE FROM @source.link WHERE record=?1"),
            [&id],
        )?;
        conn.execute(
            &source.sql("DELETE FROM @source.assertion WHERE id=?1"),
            [&id],
        )?;
        roots.insert(root);
    }
    currency::recompute(conn, roots)
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
        quote,
        owner,
    } = context;
    if statements.is_empty() {
        return Err(invalid("assert needs at least one statement"));
    }
    // Each write is checked against the archive as it stands after the
    // request's earlier writes, so a later write may name an earlier one (a
    // saved query using one defined before it, an annotation of a rubric
    // written before it). The earlier writes are previewed under their final
    // IDs and removed again before the plan below places them all against the
    // archive as it was. The removal is explicit rather than a savepoint
    // rollback: the preparation scratch runs with `journal_mode=OFF`, where
    // a rollback does not undo writes.
    let mut previews = Vec::new();
    let previewed = (|| -> Result<Vec<Prepared>> {
        let mut prepared = Vec::with_capacity(statements.len());
        for (index, original) in statements.iter().enumerate() {
            let mut statement = original.clone();
            // A relation is a statement about the revision it relates to, so it
            // also points `about` it; "everything about X" is then one lookup.
            let relations: Vec<Reference> = statement
                .links
                .iter()
                .filter(|link| link.role.is_relation())
                .map(|link| link.reference.clone())
                .collect();
            for reference in relations {
                if !statement
                    .links
                    .iter()
                    .any(|link| link.role == LinkRole::About && link.reference == reference)
                {
                    statement.links.push(NewAssertionLink {
                        role: LinkRole::About,
                        reference,
                    });
                }
            }
            let (value, rubric) =
                validate_value(conn, request, &mut statement, live_value, owner, saved)?;
            check_variant_targets(conn, &statement)?;
            for link in &statement.links {
                if link.role != LinkRole::Revises {
                    continue;
                }
                if let Reference::Assertion(RevisionReference {
                    revision: Some(id), ..
                }) = &link.reference
                    && let Some(base) = get(conn, id)?
                    && base.fact.kind != statement.kind
                {
                    return Err(invalid("revision cannot change assertion kind"));
                }
            }
            let unique = statement
                .unique(&value, &request.authorship.author, rubric.as_ref())
                .map_err(|error| invalid(&error.to_string()))?;
            // Only a write that a later one could name is previewed.
            if index + 1 < statements.len() {
                let id = authored::write_id(&request.id, index + 1, statements.len());
                let root = preview(conn, request, &id, &statement, counter)?;
                previews.push((id, root));
            }
            prepared.push(Prepared { statement, unique });
        }
        Ok(prepared)
    })();
    remove_previews(conn, previews)?;
    let prepared = previewed?;
    let arrival = Arrival {
        author: request.authorship.author.to_string(),
        via: request.authorship.via.as_str().into(),
        at: request.at.unix_ms(),
        on_stale: match request.authorship.via {
            polylogue_model::authorship::Channel::Markup
            | polylogue_model::authorship::Channel::Import => OnStale::Fork,
            _ => OnStale::Refuse,
        },
    };
    let mut store = StoredLinks::new(conn)?;
    let mut plan: Plan<'_, StoredLinks<'_>, NewAssertion> =
        Plan::new(request.id.clone(), prepared.len(), arrival, &mut store);
    let mut first = None;
    let mut written = false;
    for Prepared { statement, unique } in prepared {
        let mut revises = Vec::new();
        let mut withdraws = Vec::new();
        for link in &statement.links {
            match (link.role, &link.reference) {
                (
                    LinkRole::Revises,
                    Reference::Assertion(RevisionReference {
                        root,
                        revision: Some(id),
                    }),
                ) => revises.push(Revision::new(root, id)),
                (LinkRole::Withdraws, Reference::Assertion(revision)) => {
                    withdraws.push(target(revision))
                }
                _ => {}
            }
        }
        let write = Write {
            kind: statement.kind.as_str().into(),
            key: statement.key.clone(),
            value: statement.value.clone(),
            body: statement.clone(),
            revises,
            withdraws,
            unique,
        };
        match plan
            .add(write)
            .map_err(|refused| refusal(&statement, refused))?
        {
            Placed::Written { revision, .. } => {
                first.get_or_insert(revision.rev);
                written = true;
            }
            Placed::Already { heads } => {
                if let Some(head) = heads.first() {
                    first.get_or_insert(head.rev.clone());
                }
            }
        }
    }
    let planned = plan
        .finish()
        .map_err(|refused| refusal(statements.first().expect("a nonempty request"), refused))?;
    let source = Source::of(conn)?;
    for record in &planned.records {
        let header = &record.header;
        let statement = &record.body;
        let mut links = Vec::with_capacity(statement.links.len() + header.revises.len());
        for link in statement
            .links
            .iter()
            .filter(|link| link.role != LinkRole::Revises)
            .map(|link| (link.role, link.reference.clone()))
            .chain(
                header
                    .revises
                    .iter()
                    .map(|base| (LinkRole::Revises, reference(base))),
            )
        {
            let (role, reference) = link;
            links.push(AssertionLink {
                seen_hash: if role.uses_text_hash() {
                    hash(&reference)?
                } else {
                    None
                },
                quote: quote(&reference)?,
                role,
                reference,
            });
        }
        let fact = canonical_fact(&AssertionFact::from_request(
            header.rev.clone(),
            header.root.clone(),
            statement.clone(),
            links,
            request,
        ))?;
        insert(conn, source, &fact, &record.slots, counter)?;
    }
    currency::recompute(conn, planned.affected_roots)?;
    Ok(UserOutcome {
        id: first.expect("a nonempty request places at least one write"),
        repeat: !written,
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
            quote: &|_| Ok(None),
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

/// A revision inherits its base's kind, key, value, expiry and links unless
/// the patch replaces them; the bases become its `revises` links.
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
        facts.push(exact_fact(conn, &Reference::Assertion(base.clone()))?);
    }
    let first = &facts[0];
    if facts
        .iter()
        .any(|fact| fact.root != first.root || fact.kind != first.kind)
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
    let mut statement = NewAssertion {
        kind: first.kind,
        key: change(first.key.clone(), &patch.key),
        value: change(first.value.clone(), &patch.value),
        valid_until: change(first.valid_until, &patch.valid_until),
        links: patch.links.clone().unwrap_or_else(|| {
            first
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
