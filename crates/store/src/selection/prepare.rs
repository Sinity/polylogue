//! One binder serves pinned reads and accepting original/scratch views.
use super::*;
use polylogue_query::{Bindings, SavedQuery, SavedQueryBody};
use rusqlite::Connection;

/// Semantic objects prepared by the owner's one ordered preflight. This is
/// neither canonical nor queued custody; final acceptance never supplies it.
#[derive(Default)]
pub(crate) struct ProvisionalQueries {
    objects: BTreeMap<String, polylogue_model::query_objects::QueryObject>,
}
impl ProvisionalQueries {
    pub(crate) fn record(&mut self, plan: &Plan) -> Result<(), SelectionError> {
        let object = plan.query_object()?;
        if self
            .objects
            .get(&object.id.to_string())
            .is_some_and(|old| !crate::query_objects::equivalent(old, &object))
        {
            return Err(StoreError::Conflict(format!("query:{}", object.id)).into());
        }
        self.objects.entry(object.id.to_string()).or_insert(object);
        Ok(())
    }
    /// Prepared semantic body only. This does not establish retained custody.
    pub(crate) fn object(
        &self,
        id: QuestionId,
    ) -> Option<&polylogue_model::query_objects::QueryObject> {
        self.objects.get(&id.to_string())
    }
    pub(crate) fn objects(
        &self,
    ) -> impl Iterator<Item = &polylogue_model::query_objects::QueryObject> {
        self.objects.values()
    }
}

/// The caller owns both pinned snapshots and their source/scope guards. Native
/// references resolve only in `original`; authored metadata comes only from
/// `authored`, including pending scratch effects. No execution clock is needed.
pub(crate) fn prepare_query_plan_on(
    original: &Connection,
    authored: &Connection,
    text: &str,
    native: &Bindings,
    self_session: Option<&str>,
    pending: Option<&dyn crate::primary::Sinex>,
    provisional: Option<&ProvisionalQueries>,
) -> Result<Plan, SelectionError> {
    if original.is_autocommit() || authored.is_autocommit() {
        return Err(QueryError::Unavailable {
            what: "query preparation requires pinned original and authored views".into(),
        }
        .into());
    }
    let definitions = crate::assertions::saved_query_snapshot(authored)?;
    let mut bindings = Bindings {
        origins: native.origins.clone(),
        saved: saved_queries_from(
            original,
            authored,
            pending,
            provisional,
            definitions.definitions,
        )?,
        kept: kept_results(authored, crate::source::Source::reading(authored)?)?,
    };
    let parsed = polylogue_query::parse::with_catalog(text, &bindings.catalog())?;
    for id in polylogue_query::plan::referenced_question_ids(&parsed) {
        if bindings.saved.contains_key(&id.to_string()) {
            continue;
        }
        let frozen =
            retained_bindings(original, authored, pending, provisional, id)?.ok_or_else(|| {
                QueryError::Unavailable {
                    what: format!("retained query:{id} custody"),
                }
            })?;
        for (id, nodes) in frozen {
            bindings.saved.entry(id).or_insert(nodes);
        }
    }
    Ok(polylogue_query::plan::prepare(
        text,
        &bindings,
        |condition| resolve_reference(condition, self_session),
    )?)
}

/// Graph-selected assertions and retained objects are one definition authority
/// for both ordinary readers and native accepting transactions.
pub(crate) fn saved_queries_on(
    original: &Connection,
    conn: &Connection,
    pending: Option<&dyn crate::primary::Sinex>,
    provisional: Option<&ProvisionalQueries>,
) -> Result<BTreeMap<String, Vec<SavedQuery>>, SelectionError> {
    saved_queries_from(
        original,
        conn,
        pending,
        provisional,
        crate::assertions::saved_queries(conn)?,
    )
}

pub(super) fn saved_queries_for_family(
    conn: &Connection,
    root: &str,
) -> Result<SavedFamily, SelectionError> {
    let family = crate::assertions::saved_family(conn, root)?.definitions;
    let names = family
        .iter()
        .map(|(name, definitions)| (name.clone(), definitions.len()))
        .collect();
    Ok(SavedFamily {
        names,
        queries: saved_queries_from(conn, conn, None, None, family)?,
    })
}

fn saved_queries_from(
    original: &Connection,
    conn: &Connection,
    pending: Option<&dyn crate::primary::Sinex>,
    provisional: Option<&ProvisionalQueries>,
    definitions: BTreeMap<String, Vec<crate::assertions::CurrentSavedDefinition>>,
) -> Result<BTreeMap<String, Vec<SavedQuery>>, SelectionError> {
    let mut saved = BTreeMap::<String, Vec<SavedQuery>>::new();
    for (name, definitions) in definitions {
        for definition in definitions {
            let hash = QuestionId(definition.value.hash.0);
            let denominator = definition
                .value
                .denominator
                .as_deref()
                .map(|reference| {
                    polylogue_query::plan::question_id_from_reference(reference)
                        .map_err(SelectionError::from)
                })
                .transpose()?;
            let mut value =
                SavedQuery {
                    text: definition.value.text,
                    revision: definition.reference.revision.clone().ok_or_else(|| {
                        invalid("store", "saved definition is not an exact revision")
                    })?,
                    root: definition.reference.root,
                    hash,
                    denominator,
                    conflicts: definition.conflicts,
                    body: SavedQueryBody::Unavailable {
                        reason: format!("retained query:{hash} custody"),
                    },
                };
            if value.conflicts.is_empty() {
                match retained_bindings(original, conn, pending, provisional, hash) {
                    Ok(Some(frozen)) => {
                        value.body = frozen[&hash.to_string()][0].body.clone();
                        for (hash, nodes) in frozen {
                            saved.entry(hash).or_insert(nodes);
                        }
                    }
                    Ok(None) => {}
                    Err(SelectionError::Store(error))
                        if !matches!(
                            error,
                            StoreError::NotFound(_)
                                | StoreError::Invalid(_)
                                | StoreError::ReferenceUnavailable { .. }
                                | StoreError::Ambiguous { .. }
                        ) =>
                    {
                        return Err(error.into());
                    }
                    Err(error) => {
                        value.body = SavedQueryBody::Unavailable {
                            reason: error.to_string(),
                        }
                    }
                }
            }

            saved.entry(name.clone()).or_default().push(value);
        }
    }
    Ok(saved)
}

/// Shared custody/body resolver for named assertions and explicit typed hashes.
/// Provisional input remains an owner-private preflight facility.
fn retained_bindings(
    original: &Connection,
    authored: &Connection,
    pending: Option<&dyn crate::primary::Sinex>,
    provisional: Option<&ProvisionalQueries>,
    hash: QuestionId,
) -> Result<Option<BTreeMap<String, Vec<SavedQuery>>>, SelectionError> {
    let object = crate::query_objects::query_object_on(
        original,
        crate::source::Source::reading(original)?,
        &hash.to_string(),
        pending,
    )?
    .map(|found| found.value)
    .or_else(|| provisional.and_then(|prepared| prepared.object(hash).cloned()));
    let Some(object) = object else {
        return Ok(None);
    };
    crate::query_objects::validate_result_bindings(
        authored,
        crate::source::Source::reading(authored)?,
        &object,
    )?;
    Ok(Some(polylogue_query::plan::saved_query_bindings(&object)?))
}

pub(super) fn resolve_reference(
    condition: &mut ConditionKind,
    self_session: Option<&str>,
) -> Result<(), QueryError> {
    // `last` and `latest` depend on the query's other conditions, so the
    // compiler resolves them inside the query.
    let is_self = match condition {
        ConditionKind::Reference { reference } => reference == "self",
        ConditionKind::Text {
            text,
            mode: TextMode::Word,
        } => text == "self",
        _ => false,
    };
    if is_self {
        let id = self_session
            .map(str::to_owned)
            .ok_or_else(|| QueryError::Invalid {
                message: "self requires a calling session".into(),
                position: 0,
            })?;
        *condition = ConditionKind::Reference {
            reference: format!("session:{id}"),
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests;
