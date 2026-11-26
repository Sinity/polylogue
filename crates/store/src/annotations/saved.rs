//! Saved definitions keep their declared identity. Query
//! preparation resolves that identity through kept immutable query objects.
use super::*;
use polylogue_model::annotation::values;
use std::collections::BTreeMap;
mod preparation;
pub(crate) use preparation::SavedPreparation;

pub(crate) enum SavedInput<'a> {
    Text {
        text: &'a str,
        denominator: Option<&'a str>,
    },
    Stored {
        value: &'a values::SavedQuery,
        changed: bool,
    },
}
pub(crate) type SavedResolver<'a> =
    dyn for<'input> FnMut(SavedInput<'input>) -> Result<values::SavedQuery> + 'a;

#[derive(Debug, Clone)]
pub(crate) struct CurrentSavedDefinition {
    pub reference: RevisionReference,
    pub value: values::SavedQuery,
    pub conflicts: Vec<Reference>,
}

#[derive(Default)]
pub(crate) struct SavedQuerySnapshot {
    pub definitions: BTreeMap<String, Vec<CurrentSavedDefinition>>,
}

pub(crate) fn saved_definition(
    fact: &AnnotationFact,
    conflicts: &[Reference],
) -> Result<Option<(String, CurrentSavedDefinition)>> {
    if fact.kind != AnnotationKind::SavedQuery {
        return Ok(None);
    }
    let name = fact
        .key
        .clone()
        .ok_or_else(|| crate::StoreError::stored_data("saved definition has no name"))?;
    let values::Value::SavedQuery(value) = fact
        .validate()
        .map_err(|error| crate::StoreError::stored_data(error.to_string()))?
    else {
        return Err(crate::StoreError::stored_data(
            "saved definition has another value shape",
        ));
    };
    Ok(Some((
        name,
        CurrentSavedDefinition {
            reference: RevisionReference::exact(fact.root.clone(), fact.id.clone()),
            value: value.fields,
            conflicts: conflicts.to_vec(),
        },
    )))
}

/// Decode one current family without dropping forks that changed its name.
/// A conflict carries every head, including those under a different key.
pub(crate) fn saved_family(conn: &Connection, root: &str) -> Result<SavedQuerySnapshot> {
    let heads = heads(
        conn,
        &Reference::Annotation(RevisionReference {
            root: root.into(),
            revision: None,
        }),
    )?;
    let conflicts = if heads.len() > 1 {
        heads.clone()
    } else {
        Vec::new()
    };
    let mut definitions = SavedQuerySnapshot::default();
    for reference in heads {
        let Reference::Annotation(revision) = reference else {
            return Err(crate::StoreError::stored_data(
                "saved definition head is not an annotation",
            ));
        };
        let id = revision
            .revision
            .as_ref()
            .ok_or_else(|| crate::StoreError::stored_data("saved definition head is not exact"))?;
        let row = get(conn, id)?.ok_or_else(|| {
            StoreError::NotFound(Reference::Annotation(revision.clone()).to_string())
        })?;
        if let Some((name, definition)) = saved_definition(&row.fact, &conflicts)? {
            definitions
                .definitions
                .entry(name)
                .or_default()
                .push(definition);
        }
    }
    Ok(definitions)
}

#[cfg(test)]
mod tests;

/// The current saved definitions under `names`, and no others. SQL
/// nominates every family any revision of which carried one of the names;
/// the stored current-revision index selects heads, and only heads still named so
/// are kept.
pub(crate) fn saved_query_snapshot<'a>(
    conn: &Connection,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<SavedQuerySnapshot> {
    let names: Vec<&str> = names.into_iter().collect();
    let mut definitions = SavedQuerySnapshot::default();
    if names.is_empty() {
        return Ok(definitions);
    }
    let source = Source::reading(conn)?;
    let roots = conn
        .prepare_cached(&source.sql(
            "SELECT DISTINCT root FROM @source.annotation WHERE kind='saved_query' AND key IN (SELECT value FROM json_each(?1)) ORDER BY root",
        ))?
        .query_map([serde_json::to_string(&names)?], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for root in roots {
        let family = saved_family(conn, &root)?;
        for (name, mut family) in family.definitions {
            if names.contains(&name.as_str()) {
                definitions
                    .definitions
                    .entry(name)
                    .or_default()
                    .append(&mut family);
            }
        }
    }
    Ok(definitions)
}

/// Current named records, including every conflicting head.
pub(crate) fn named_current(
    conn: &Connection,
    kind: AnnotationKind,
    name: &str,
) -> Result<Vec<Annotation>> {
    let source = Source::reading(conn)?;
    let ids = conn
        .prepare_cached(&source.sql(
            "SELECT a.id FROM @source.annotation a JOIN @source.revision_state r ON r.id=a.id
         WHERE a.kind=?1 AND a.key=?2 AND r.current=1 AND r.withdrawn=0 ORDER BY a.id",
        ))?
        .query_map(params![kind.as_str(), name], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    rows_by_id(conn, ids)
}

/// The stored rows of these annotation ids, in order.
fn rows_by_id(conn: &Connection, ids: Vec<String>) -> Result<Vec<Annotation>> {
    ids.into_iter()
        .map(|id| get(conn, &id)?.ok_or_else(|| StoreError::NotFound(id)))
        .collect()
}

/// One current named record; conflicting heads retain their exact candidates.
pub(crate) fn single_named_current(
    conn: &Connection,
    kind: AnnotationKind,
    name: &str,
) -> Result<Option<Annotation>> {
    let rows = named_current(conn, kind, name)?;
    if rows.len() > 1 {
        return Err(StoreError::Ambiguous {
            reference: format!("{}:{name}", kind.as_str()),
            candidates: rows
                .iter()
                .map(|row| row.fact.reference().to_string())
                .collect(),
        });
    }
    Ok(rows.into_iter().next())
}

fn revision_parents(row: &Annotation) -> Result<Vec<&RevisionReference>> {
    row.fact
        .links
        .iter()
        .filter(|link| link.role == LinkRole::Revises)
        .map(|link| match &link.reference {
            Reference::Annotation(parent) if parent.revision.is_some() => Ok(parent),
            _ => Err(StoreError::stored_data(format!(
                "saved revision {} has a non-exact parent",
                row.fact.id
            ))),
        })
        .collect()
}

/// Original revision links define ordinals. Local commit counters and clocks do not.
fn numbered_revisions(
    conn: &Connection,
    initial: Vec<Annotation>,
) -> Result<Vec<(u32, Annotation)>> {
    let mut pending = initial
        .iter()
        .map(|row| row.fact.id.clone())
        .collect::<Vec<_>>();
    let mut rows = initial
        .into_iter()
        .map(|row| (row.fact.id.clone(), row))
        .collect::<BTreeMap<_, _>>();
    while let Some(id) = pending.pop() {
        let parents = revision_parents(&rows[&id])?
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        for parent in parents {
            let parent_id = parent.revision.as_ref().expect("exact parent checked");
            if !rows.contains_key(parent_id) {
                let row =
                    get(conn, parent_id)?.ok_or_else(|| StoreError::ReferenceUnavailable {
                        reference: Reference::Annotation(parent.clone()).to_string(),
                        reason: "saved-query revision parent is not stored".into(),
                    })?;
                pending.push(parent_id.clone());
                rows.insert(parent_id.clone(), row);
            }
            let row = &rows[parent_id];
            if row.fact.root != parent.root || row.fact.kind != AnnotationKind::SavedQuery {
                return Err(StoreError::stored_data(format!(
                    "saved-query parent {} does not name its recorded saved revision",
                    Reference::Annotation(parent)
                )));
            }
        }
    }
    let mut remaining = BTreeMap::new();
    let mut children = BTreeMap::<String, Vec<String>>::new();
    let mut ready = std::collections::BTreeSet::new();
    let mut levels = BTreeMap::<String, u32>::new();
    for (id, row) in &rows {
        let parents = revision_parents(row)?;
        remaining.insert(id.clone(), parents.len());
        if parents.is_empty() {
            ready.insert(id.clone());
            levels.insert(id.clone(), 1);
        }
        for parent in parents {
            children
                .entry(parent.revision.clone().expect("exact parent checked"))
                .or_default()
                .push(id.clone());
        }
    }
    while let Some(id) = ready.pop_first() {
        let next = levels[&id]
            .checked_add(1)
            .ok_or_else(|| StoreError::stored_data("saved revision number is out of range"))?;
        for child in children.get(&id).into_iter().flatten() {
            let level = levels.entry(child.clone()).or_insert(1);
            *level = (*level).max(next);
            let parents_left = remaining.get_mut(child).expect("every child is loaded");
            *parents_left -= 1;
            if *parents_left == 0 {
                ready.insert(child.clone());
            }
        }
    }
    if remaining.values().any(|count| *count != 0) {
        let cycle = remaining
            .iter()
            .filter(|(_, count)| **count != 0)
            .map(|(id, _)| rows[id].fact.reference().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(StoreError::stored_data(format!(
            "saved-query revision links form a cycle: {cycle}"
        )));
    }
    let mut ordered = rows
        .into_iter()
        .map(|(id, row)| (levels[&id], row))
        .collect::<Vec<_>>();
    ordered.sort_by(|(an, a), (bn, b)| {
        (an, &a.fact.root, &a.fact.id).cmp(&(bn, &b.fact.root, &b.fact.id))
    });
    Ok(ordered)
}

fn numbered_named_revisions(conn: &Connection, name: &str) -> Result<Vec<(u32, Annotation)>> {
    let source = Source::reading(conn)?;
    let ids = conn
        .prepare_cached(&source.sql(
            "SELECT id FROM @source.annotation WHERE kind='saved_query' AND root IN
         (SELECT root FROM @source.annotation WHERE kind='saved_query' AND key=?1) ORDER BY id",
        ))?
        .query_map([name], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    numbered_revisions(conn, rows_by_id(conn, ids)?)
}

pub(crate) fn saved_revision_number(
    conn: &Connection,
    reference: &RevisionReference,
) -> Result<u32> {
    let id =
        reference
            .revision
            .as_ref()
            .ok_or_else(|| polylogue_model::Reason::NotAnExactRevision {
                reference: Reference::Annotation(reference.clone()).to_string(),
            })?;
    let row = get(conn, id)?.ok_or_else(|| {
        StoreError::NotFound(Reference::Annotation(reference.clone()).to_string())
    })?;
    if row.fact.root != reference.root || row.fact.kind != AnnotationKind::SavedQuery {
        return Err(polylogue_model::Reason::WrongKind {
            reference: Reference::Annotation(reference.clone()).to_string(),
            expected: "saved query revision".into(),
        }
        .into());
    }
    numbered_revisions(conn, vec![row])?
        .into_iter()
        .find(|(_, row)| row.fact.id == *id)
        .map(|(number, _)| number)
        .ok_or_else(|| StoreError::NotFound(Reference::Annotation(reference.clone()).to_string()))
}

pub(crate) fn named_revision(
    conn: &Connection,
    name: &str,
    number: u32,
) -> Result<CurrentSavedDefinition> {
    // A name with one current family binds its current heads and ancestors.
    // Withdrawn siblings do not compete with a redo; active forks and explicit
    // cross-family ancestors stay in the graph. Otherwise use all history.
    let roots = named_current(conn, AnnotationKind::SavedQuery, name)?
        .into_iter()
        .map(|row| row.fact.root)
        .collect::<std::collections::BTreeSet<_>>();
    let root = (roots.len() == 1).then(|| roots.first().expect("one current root").as_str());
    let revisions = match root {
        None => numbered_named_revisions(conn, name)?,
        Some(root) => {
            let source = Source::reading(conn)?;
            let ids = conn.prepare_cached(&source.sql(
                "SELECT a.id FROM @source.annotation a JOIN @source.revision_state r ON r.id=a.id
                 WHERE a.kind='saved_query' AND a.root=?1 AND r.current=1 AND r.withdrawn=0 ORDER BY a.id",
            ))?.query_map([root], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            numbered_revisions(conn, rows_by_id(conn, ids)?)?
        }
    };
    let mut matching = revisions
        .into_iter()
        .filter(|(n, _)| *n == number)
        .map(|(_, row)| row)
        .collect::<Vec<_>>();
    let row = match matching.len() {
        0 => return Err(StoreError::NotFound(format!("query:{name}@{number}"))),
        1 => matching.pop().expect("one matching revision"),
        _ => {
            return Err(StoreError::Ambiguous {
                reference: format!("query:{name}@{number}"),
                candidates: matching
                    .iter()
                    .map(|row| row.fact.reference().to_string())
                    .collect(),
            });
        }
    };
    saved_definition(&row.fact, &[])?
        .map(|(_, definition)| definition)
        .ok_or_else(|| StoreError::stored_data("numbered revision is not a saved query"))
}

impl crate::Reader<'_> {
    /// The current authored saved-query family, without running its question.
    pub fn saved_query_target(&self, name: &str) -> Result<Option<Reference>> {
        let target =
            single_named_current(self.conn()?, AnnotationKind::SavedQuery, name)?.map(|row| {
                Reference::Annotation(RevisionReference {
                    root: row.fact.root,
                    revision: None,
                })
            });
        self.check()?;
        Ok(target)
    }

    /// Current unambiguous names for these dependent question identities.
    /// The authored text keeps the names a re-save must resolve again.
    pub fn dependent_saved_queries(
        &self,
        queries: &[String],
    ) -> Result<Vec<polylogue_model::saved_queries::DependentSavedQuery>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let hashes = queries
            .iter()
            .map(|query| {
                query.strip_prefix("query:").ok_or_else(|| {
                    StoreError::stored_data(format!("dependent {query} is not a query identity"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let names = conn.prepare_cached(&source.sql(
            "SELECT DISTINCT a.key FROM @source.annotation a JOIN @source.revision_state r ON r.id=a.id WHERE a.kind='saved_query' AND r.current=1 AND r.withdrawn=0 AND json_extract(a.value,'$.hash') IN (SELECT value FROM json_each(?1)) ORDER BY a.key",
        ))?.query_map([serde_json::to_string(&hashes)?], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut definitions = Vec::new();
        for name in names {
            let current = named_current(conn, AnnotationKind::SavedQuery, &name)?;
            if let [row] = current.as_slice() {
                let (_, definition) = saved_definition(&row.fact, &[])?
                    .ok_or_else(|| StoreError::stored_data("dependent query has another kind"))?;
                definitions.push(polylogue_model::saved_queries::DependentSavedQuery {
                    name,
                    revision: definition.reference,
                    value: definition.value,
                });
            }
        }
        self.check()?;
        Ok(definitions)
    }

    /// Authored history retains original revisions, hashes, times and attribution.
    pub fn saved_query_record(
        &self,
        name: &str,
    ) -> Result<Option<polylogue_model::saved_queries::SavedQueryRecord>> {
        use polylogue_model::saved_queries::{KeptSummary, SavedQueryRecord, SavedRevision};
        let conn = self.conn()?;
        let rows = numbered_named_revisions(conn, name)?;
        if rows.is_empty() {
            return Ok(None);
        }
        let mut revisions = Vec::new();
        let mut hashes = std::collections::BTreeSet::new();
        for (number, row) in rows {
            let (_, definition) = saved_definition(&row.fact, &[])?
                .ok_or_else(|| StoreError::stored_data("saved history contains another kind"))?;
            let hash = format!("query:{}", definition.value.hash);
            hashes.insert(hash.clone());
            revisions.push(SavedRevision {
                revision: number,
                reference: Reference::Annotation(definition.reference).to_string(),
                text: definition.value.text,
                hash,
                saved_at: row.fact.created_at,
                author: row.fact.author.to_string(),
            });
        }
        let source = Source::reading(conn)?;
        let mut kept_statement = conn.prepare_cached(&source.sql(
            "SELECT id,json_array_length(rows_json,'$.rows'),ran_at FROM @source.saved_rows WHERE name=?1 ORDER BY ran_at DESC,id DESC",
        ))?;
        let kept = kept_statement
            .query_map([name], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<u64>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?
            .map(|row| {
                let (id, rows, at) = row?;
                Ok(KeptSummary {
                    result: format!("result:{id}"),
                    rows: rows.ok_or_else(|| StoreError::stored_data("kept result has no rows"))?,
                    at: Timestamp::from_unix_ms(at)
                        .ok_or_else(|| StoreError::stored_data("kept time is out of range"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let watches = named_current(conn, AnnotationKind::Signal, name)?;
        let watch = match watches.as_slice() {
            [] => None,
            [watch] => Some(watch.fact.reference().to_string()),
            _ => {
                return Err(StoreError::Ambiguous {
                    reference: format!("watch:{name}"),
                    candidates: watches
                        .iter()
                        .map(|row| row.fact.reference().to_string())
                        .collect(),
                });
            }
        };
        let mut dependent_statement = conn.prepare_cached(&source.sql(
            "SELECT DISTINCT query FROM @source.query_edge WHERE uses IN (SELECT value FROM json_each(?1)) ORDER BY query",
        ))?;
        let dependents = dependent_statement
            .query_map([serde_json::to_string(&hashes)?], |row| {
                row.get::<_, String>(0)
            })?
            .map(|row| row.map(|id| format!("query:{id}")))
            .collect::<rusqlite::Result<Vec<_>>>()?;
        self.check()?;
        Ok(Some(SavedQueryRecord {
            name: name.into(),
            revisions,
            kept,
            watch,
            dependents,
        }))
    }
}
