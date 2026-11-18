//! Saved definitions keep their declared identity. Query
//! preparation resolves that identity through kept immutable query objects.
use super::*;
use polylogue_model::assertion::values;
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
    fact: &AssertionFact,
    conflicts: &[Reference],
) -> Result<Option<(String, CurrentSavedDefinition)>> {
    if fact.kind != AssertionKind::SavedQuery {
        return Ok(None);
    }
    let name = fact
        .key
        .clone()
        .ok_or_else(|| crate::StoreError::invalid("saved definition has no name"))?;
    let values::Value::SavedQuery(value) = fact
        .validate()
        .map_err(|error| crate::StoreError::invalid(error.to_string()))?
    else {
        return Err(crate::StoreError::invalid(
            "saved definition has another value shape",
        ));
    };
    Ok(Some((
        name,
        CurrentSavedDefinition {
            reference: RevisionReference {
                root: fact.root.clone(),
                revision: Some(fact.id.clone()),
            },
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
        &Reference::Assertion(RevisionReference {
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
        let Reference::Assertion(revision) = reference else {
            return Err(crate::StoreError::invalid(
                "saved definition head is not an assertion",
            ));
        };
        let id = revision
            .revision
            .as_ref()
            .ok_or_else(|| crate::StoreError::invalid("saved definition head is not exact"))?;
        let row = get(conn, id)?.ok_or_else(|| {
            StoreError::NotFound(Reference::Assertion(revision.clone()).to_string())
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
            "SELECT DISTINCT root FROM @source.assertion WHERE kind='saved_query' AND key IN (SELECT value FROM json_each(?1)) ORDER BY root",
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
    kind: AssertionKind,
    name: &str,
) -> Result<Vec<Assertion>> {
    let source = Source::reading(conn)?;
    let ids = conn.prepare_cached(&source.sql(
        "SELECT a.id FROM @source.assertion a JOIN @source.revision_state r ON r.id=a.id WHERE a.kind=?1 AND a.key=?2 AND r.current=1 ORDER BY a.id",
    ))?.query_map(params![kind.as_str(), name], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    ids.into_iter()
        .map(|id| get(conn, &id)?.ok_or_else(|| StoreError::NotFound(id)))
        .collect()
}

pub(crate) fn named_revisions(conn: &Connection, name: &str) -> Result<Vec<Assertion>> {
    let source = Source::reading(conn)?;
    let ids = conn.prepare_cached(&source.sql(
        "SELECT id FROM @source.assertion WHERE kind='saved_query' AND root IN (SELECT root FROM @source.assertion WHERE kind='saved_query' AND key=?1) ORDER BY commit_seq,id",
    ))?.query_map([name], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    ids.into_iter()
        .map(|id| get(conn, &id)?.ok_or_else(|| StoreError::NotFound(id)))
        .collect()
}

impl crate::Reader<'_> {
    pub fn saved_query_record(
        &self,
        name: &str,
    ) -> Result<Option<polylogue_model::saved_queries::SavedQueryRecord>> {
        use polylogue_model::saved_queries::{KeptSummary, SavedQueryRecord, SavedRevision};
        let conn = self.conn()?;
        let rows = named_revisions(conn, name)?;
        if rows.is_empty() {
            return Ok(None);
        }
        let mut revisions = Vec::new();
        let mut hashes = Vec::new();
        for (index, row) in rows.into_iter().enumerate() {
            let value: values::SavedQuery = serde_json::from_value(
                row.fact
                    .value
                    .ok_or_else(|| StoreError::invalid("saved question has no value"))?,
            )?;
            hashes.push(value.hash.to_string());
            revisions.push(SavedRevision {
                revision: u32::try_from(index + 1)
                    .map_err(|_| StoreError::invalid("saved revision number is out of range"))?,
                reference: Reference::Assertion(RevisionReference {
                    root: row.fact.root,
                    revision: Some(row.fact.id),
                })
                .to_string(),
                text: value.text,
                hash: format!("query:{}", value.hash),
                saved_at: row.fact.created_at,
                author: row.fact.author.to_string(),
            });
        }
        let source = Source::reading(conn)?;
        let kept = conn.prepare_cached(&source.sql(
            "SELECT id,json_array_length(rows_json,'$.rows'),ran_at FROM @source.kept_result WHERE name=?1 ORDER BY id DESC",
        ))?.query_map([name], |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<u64>>(1)?, row.get::<_, i64>(2)?)))?
            .map(|row| {
                let (id, rows, at) = row?;
                Ok(KeptSummary { result: format!("result:{id}"), rows: rows.ok_or_else(|| StoreError::invalid("kept result has no rows"))?, at: Timestamp::from_unix_ms(at).ok_or_else(|| StoreError::invalid("kept time is out of range"))? })
            }).collect::<Result<Vec<_>>>()?;
        let watches = named_current(conn, AssertionKind::Signal, name)?;
        let watch = match watches.as_slice() {
            [] => None,
            [watch] => Some(
                Reference::Assertion(RevisionReference {
                    root: watch.fact.root.clone(),
                    revision: Some(watch.fact.id.clone()),
                })
                .to_string(),
            ),
            _ => {
                return Err(StoreError::Ambiguous {
                    reference: format!("watch:{name}"),
                    candidates: watches
                        .iter()
                        .map(|row| format!("assertion:{}@{}", row.fact.root, row.fact.id))
                        .collect(),
                });
            }
        };
        let dependents = conn.prepare_cached(&source.sql(
            "SELECT DISTINCT query FROM @source.query_edge WHERE uses IN (SELECT value FROM json_each(?1)) ORDER BY query",
        ))?.query_map([serde_json::to_string(&hashes)?], |row| row.get::<_, String>(0))?
            .map(|row| row.map(|id| format!("query:{id}"))).collect::<rusqlite::Result<Vec<_>>>()?;
        self.conn()?;
        Ok(Some(SavedQueryRecord {
            name: name.into(),
            revisions,
            kept,
            watch,
            dependents,
        }))
    }
}
