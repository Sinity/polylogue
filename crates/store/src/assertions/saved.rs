//! Canonical saved definitions retain their declared semantic identity. Query
//! preparation resolves that identity through retained immutable query objects.
use super::*;
use polylogue_model::assertion::values;
use std::collections::BTreeMap;
mod preparation;
pub(crate) use preparation::SavedPreparation;

pub(crate) enum SavedInput<'a> {
    Native {
        text: &'a str,
        denominator: Option<&'a str>,
    },
    Canonical {
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
        .ok_or_else(|| invalid("saved definition has no name"))?;
    let values::Value::SavedQuery(value) = fact
        .validate()
        .map_err(|error| invalid(&error.to_string()))?
    else {
        return Err(invalid("saved definition has another value shape"));
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
            return Err(invalid("saved definition head is not an assertion"));
        };
        let id = revision
            .revision
            .as_ref()
            .ok_or_else(|| invalid("saved definition head is not exact"))?;
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

/// SQL nominates saved families; the stored currency index selects heads.
/// Both pinned Reader and original/scratch preparation consume this loader.
pub(crate) fn saved_queries(
    conn: &Connection,
) -> Result<BTreeMap<String, Vec<CurrentSavedDefinition>>> {
    Ok(saved_query_snapshot(conn)?.definitions)
}

pub(crate) fn saved_query_snapshot(conn: &Connection) -> Result<SavedQuerySnapshot> {
    let source = Source::reading(conn)?;
    let roots = conn
        .prepare_cached(&source.sql(
            "SELECT DISTINCT root FROM @source.assertion WHERE kind='saved_query' ORDER BY root",
        ))?
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut definitions = SavedQuerySnapshot::default();
    for root in roots {
        let family = saved_family(conn, &root)?;
        for (name, mut family) in family.definitions {
            definitions
                .definitions
                .entry(name)
                .or_default()
                .append(&mut family);
        }
    }
    Ok(definitions)
}
