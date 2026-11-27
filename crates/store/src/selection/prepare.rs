//! One binder serves pinned reads and accepting original/scratch views.
use super::*;
use polylogue_query::{Bindings, SavedQuery, SavedQueryBody};
use rusqlite::Connection;
use std::collections::BTreeSet;

/// Query objects prepared by the writer's one ordered preflight. They are
/// neither confirmed nor queued; final acceptance never supplies them.
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
    /// The one prepared question whose id starts with `prefix`; two are
    /// ambiguous, as they are among stored questions.
    fn by_prefix(&self, prefix: &str) -> Result<Option<QuestionId>, QueryError> {
        let matching: Vec<QuestionId> = self
            .objects
            .values()
            .filter(|object| object.id.to_string().starts_with(prefix))
            .map(|object| object.id)
            .collect();
        match matching.as_slice() {
            [] => Ok(None),
            [only] => Ok(Some(*only)),
            _ => Err(QueryError::Ambiguous {
                what: format!("query:{prefix}"),
                candidates: matching.iter().map(ToString::to_string).collect(),
            }),
        }
    }
    /// The prepared body held by this preflight; reading it persists nothing.
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

/// The caller owns both pinned snapshots and their source/scope guards. Typed
/// references resolve only in `original`; authored metadata comes only from
/// `authored`, including pending scratch effects. No execution clock is needed.
pub(crate) fn prepare_query_plan_on(
    original: &Connection,
    authored: &Connection,
    text: &str,
    supplied: &Bindings,
    self_session: Option<&str>,
    provisional: Option<&ProvisionalQueries>,
) -> Result<Plan, SelectionError> {
    if original.is_autocommit() || authored.is_autocommit() {
        return Err(QueryError::Unavailable {
            reason: "query preparation requires pinned original and authored views".into(),
        }
        .into());
    }
    if let Ok(Reference::Run(id)) = Reference::parse(text) {
        let source = crate::source::Source::reading(original)?;
        let run = crate::query_objects::lookup_run(original, source, &id)?.ok_or_else(|| {
            SelectionError::invalid(
                polylogue_model::Code::NotFound,
                format!("no saved run {id}"),
            )
        })?;
        let question = run.query_id.ok_or_else(|| {
            SelectionError::invalid(
                polylogue_model::Code::BadRequest,
                "this run read an object and has no query question to run again",
            )
        })?;
        let object =
            crate::query_objects::query_object_on(original, source, &question.to_string())?
                .or_else(|| provisional.and_then(|prepared| prepared.object(question).cloned()))
                .ok_or_else(|| QueryError::Unavailable {
                    reason: format!("run:{id} question query:{question} is unavailable"),
                })?;
        // Only the immutable question is reused. Lowering resolves time, here
        // and private reading marks with this invocation's execution options.
        return Ok(polylogue_query::plan::replay(
            &object,
            polylogue_query::plan::catalog_fingerprint(),
        )?);
    }
    let bindings = named_definitions(original, authored, text, &supplied.origins, provisional)?;
    Ok(polylogue_query::plan::prepare(
        text,
        &bindings,
        |condition| resolve_reference(condition, self_session),
    )?)
}

/// The ids written as `result:<id>` in `text`, read from its characters:
/// quoted as the formatter writes them (with `\"`, `\\`, `\n`, `\r` and `\t`
/// escapes), or bare.
fn spelled_kept_results(text: &str) -> Vec<String> {
    text.match_indices("result:")
        .filter_map(|(at, prefix)| {
            let rest = &text[at + prefix.len()..];
            let id = match rest.strip_prefix('"') {
                Some(quoted) => unquote(quoted)?,
                None => {
                    let end = rest
                        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_')))
                        .unwrap_or(rest.len());
                    rest[..end].to_owned()
                }
            };
            (!id.is_empty()).then_some(id)
        })
        .collect()
}

/// The text of a quoted string whose opening quote is already consumed; None
/// when it is never closed.
fn unquote(text: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => out.push(match chars.next()? {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            }),
            c => out.push(c),
        }
    }
    None
}

/// The definitions a query names, and only those: its saved names with
/// their saved dependencies, the saved questions it names by hash or by
/// prefix, and the kept results it or those definitions read. Other saved
/// queries and kept bodies are never read, so an unrelated query costs the
/// same however many of them are stored.
pub(crate) fn named_definitions(
    original: &Connection,
    authored: &Connection,
    text: &str,
    origins: &[String],
    provisional: Option<&ProvisionalQueries>,
) -> Result<Bindings, SelectionError> {
    let mut bindings = Bindings {
        origins: origins.to_vec(),
        ..Default::default()
    };
    // Names are read from the syntax before any definition is loaded. Some
    // syntax (a join's `match.` fields) is checked against a kept result's
    // columns, so text that does not parse alone is read again with the kept
    // results it spells out. Text that still does not parse names nothing
    // more; preparing it reports why.
    let parsed = match polylogue_query::parse::with_catalog(text, &bindings.catalog()) {
        Ok(parsed) => parsed,
        Err(_) => {
            (bindings.kept, bindings.kept_names) = named_kept_bindings(
                authored,
                spelled_kept_results(text).iter().map(String::as_str),
            )?;
            match polylogue_query::parse::with_catalog(text, &bindings.catalog()) {
                Ok(parsed) => parsed,
                Err(_) => return Ok(bindings),
            }
        }
    };
    let names: BTreeSet<String> = polylogue_query::plan::referenced_saved_names(&parsed)
        .into_iter()
        .collect();
    let mut definitions =
        crate::annotations::saved_query_snapshot(authored, names.iter().map(String::as_str))?
            .definitions;
    for name in &names {
        if let Some((saved_name, revision)) = name.rsplit_once('@') {
            let revision = revision
                .parse::<u32>()
                .ok()
                .filter(|number| *number > 0)
                .ok_or_else(|| QueryError::BadValue {
                    field: "query".into(),
                    value: name.clone(),
                    position: 0,
                    message: "a saved revision is a positive integer: query:NAME@N".into(),
                })?;
            let definition = crate::annotations::named_revision(authored, saved_name, revision)?;
            definitions.insert(name.clone(), vec![definition]);
        }
    }
    bindings.saved = saved_queries_from(original, authored, provisional, definitions)?;
    let shipped = super::shipped_bindings(origins)?;
    for name in &names {
        if !bindings.saved.contains_key(name)
            && let Some(definitions) = shipped.get(name)
        {
            bindings.saved.insert(name.clone(), definitions.clone());
            for definition in definitions {
                let hash = definition.hash.to_string();
                if let Some(nodes) = shipped.get(&hash) {
                    bindings.saved.entry(hash).or_insert_with(|| nodes.clone());
                }
            }
        }
    }
    // `query:<prefix>`: a unique prefix of 8 or more hex characters names the
    // saved question it starts. Saved names are never all hex, so a prefix
    // cannot shadow one.
    for name in &names {
        let prefix = (8..64).contains(&name.len())
            && name
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
        if !prefix || bindings.saved.contains_key(name) {
            continue;
        }
        let stored = crate::query_objects::query_object_on(
            original,
            crate::source::Source::reading(original)?,
            name,
        )?
        .map(|found| found.id);
        let found = match stored {
            Some(found) => Some(found),
            None => match provisional {
                Some(prepared) => prepared.by_prefix(name)?,
                None => None,
            },
        }
        .ok_or_else(|| QueryError::Unavailable {
            reason: format!("saved query:{name} is unavailable"),
        })?;
        let mut saved_nodes =
            stored_bindings(original, authored, provisional, found)?.ok_or_else(|| {
                QueryError::Unavailable {
                    reason: format!("the stored body of saved query:{found} is unavailable"),
                }
            })?;
        let definition = saved_nodes[&found.to_string()].clone();
        saved_nodes.insert(name.clone(), definition);
        keep_missing(&mut bindings.saved, saved_nodes);
    }
    for id in polylogue_query::plan::referenced_question_ids(&parsed) {
        if bindings.saved.contains_key(&id.to_string()) {
            continue;
        }
        let saved_nodes =
            stored_bindings(original, authored, provisional, id)?.ok_or_else(|| {
                QueryError::Unavailable {
                    reason: format!("the stored body of saved query:{id} is unavailable"),
                }
            })?;
        keep_missing(&mut bindings.saved, saved_nodes);
    }
    let mut kept: BTreeSet<String> = polylogue_query::plan::referenced_kept_results(&parsed)
        .into_iter()
        .collect();
    for definition in bindings.saved.values().flatten() {
        if let SavedQueryBody::Available { syntax, .. } = &definition.body {
            match polylogue_query::parse::with_catalog(syntax, &bindings.catalog()) {
                Ok(saved) => kept.extend(polylogue_query::plan::referenced_kept_results(&saved)),
                // Text whose syntax needs a kept result's columns names them.
                Err(_) => kept.extend(spelled_kept_results(syntax)),
            }
        }
    }
    (bindings.kept, bindings.kept_names) =
        named_kept_bindings(authored, kept.iter().map(String::as_str))?;
    Ok(bindings)
}

/// Merge definitions, keeping the ones already bound.
fn keep_missing(
    into: &mut BTreeMap<String, Vec<SavedQuery>>,
    from: BTreeMap<String, Vec<SavedQuery>>,
) {
    for (id, nodes) in from {
        into.entry(id).or_insert(nodes);
    }
}

fn named_kept_bindings<'a>(
    conn: &Connection,
    ids: impl IntoIterator<Item = &'a str>,
) -> crate::Result<(
    BTreeMap<String, polylogue_query::Kept>,
    BTreeMap<String, String>,
)> {
    use rusqlite::OptionalExtension;
    let ids = ids.into_iter().collect::<BTreeSet<_>>();
    let source = crate::source::Source::reading(conn)?;
    let mut kept = kept_results_named(conn, source, ids.iter().copied())?;
    let mut names = BTreeMap::new();
    let mut lookup = conn.prepare_cached(&source.sql(
        "SELECT id FROM @source.saved_rows WHERE name=?1 ORDER BY ran_at DESC,id DESC LIMIT 1",
    ))?;
    for name in ids {
        // Client request IDs are opaque. An exact existing ID always remains
        // addressable, even when it could also be a saved name.
        if kept.contains_key(name) {
            continue;
        }
        let id = lookup
            .query_row([name], |row| row.get::<_, String>(0))
            .optional()?
            .ok_or_else(|| StoreError::Selection {
                code: polylogue_model::Code::NotFound,
                message: format!("no kept result named {name}"),
            })?;
        names.insert(name.to_owned(), id);
    }
    kept.extend(kept_results_named(
        conn,
        source,
        names.values().map(String::as_str),
    )?);
    for (name, id) in &names {
        let value = kept
            .get(id)
            .ok_or_else(|| StoreError::NotFound(format!("result:{id}")))?;
        kept.insert(name.clone(), value.clone());
    }
    Ok((kept, names))
}

pub(super) fn saved_queries_for_family(
    conn: &Connection,
    root: &str,
) -> Result<SavedFamily, SelectionError> {
    let family = crate::annotations::saved_family(conn, root)?.definitions;
    let names = family
        .iter()
        .map(|(name, definitions)| (name.clone(), definitions.len()))
        .collect();
    Ok(SavedFamily {
        names,
        queries: saved_queries_from(conn, conn, None, family)?,
    })
}

fn saved_queries_from(
    original: &Connection,
    conn: &Connection,
    provisional: Option<&ProvisionalQueries>,
    definitions: BTreeMap<String, Vec<crate::annotations::CurrentSavedDefinition>>,
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
            let mut value = SavedQuery {
                text: definition.value.text,
                revision: definition.reference.revision.clone().ok_or_else(|| {
                    SelectionError::invalid(
                        polylogue_model::Code::Store,
                        "saved definition does not name a revision",
                    )
                })?,
                root: definition.reference.root,
                hash,
                denominator,
                conflicts: definition.conflicts,
                body: SavedQueryBody::Unavailable {
                    reason: format!("saved query:{hash} stored body"),
                },
            };
            if value.conflicts.is_empty() {
                match stored_bindings(original, conn, provisional, hash) {
                    Ok(Some(saved_nodes)) => {
                        value.body = saved_nodes[&hash.to_string()][0].body.clone();
                        keep_missing(&mut saved, saved_nodes);
                    }
                    Ok(None) => {}
                    Err(SelectionError::Store(error))
                        if !matches!(
                            error,
                            StoreError::NotFound(_)
                                | StoreError::Invalid(_)
                                | StoreError::Refused(_)
                                | StoreError::Internal(_)
                                | StoreError::StoredValue { .. }
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

/// Shared stored-body resolver for named annotations and explicit typed hashes.
/// Provisional input stays private to the writer's preflight.
fn stored_bindings(
    original: &Connection,
    authored: &Connection,
    provisional: Option<&ProvisionalQueries>,
    hash: QuestionId,
) -> Result<Option<BTreeMap<String, Vec<SavedQuery>>>, SelectionError> {
    let object = crate::query_objects::query_object_on(
        original,
        crate::source::Source::reading(original)?,
        &hash.to_string(),
    )?
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
    // `latest` depends on the query's other conditions, so the
    // compiler resolves it inside the query.
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
            .ok_or_else(|| QueryError::NotFound {
                message: "`self` names the calling session, and this caller is not one".into(),
            })?;
        *condition = ConditionKind::Reference {
            reference: format!("session:{id}"),
        };
    }
    Ok(())
}

#[cfg(test)]
mod tests;
