//! Session frequencies in a selection and in the same reader's whole archive.
use super::Selection;
use crate::{Reader, Result, StoreError};
use serde::Serialize;
use std::collections::BTreeMap;

type Counts = BTreeMap<String, u64>;

#[derive(Debug, Default, Serialize)]
pub struct Buckets {
    pub total_sessions: u64,
    pub total_messages: u64,
    pub families: BTreeMap<String, Counts>,
    pub omitted: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct Facets {
    pub selected: Buckets,
    pub global: Buckets,
    pub scoped_to_query: bool,
    pub idf: BTreeMap<String, BTreeMap<String, f64>>,
}

pub fn facets(reader: &Reader<'_>, selection: &Selection, options: &str) -> Result<Facets> {
    let (deep, weights) = match options {
        "" | "basic" => (false, true),
        "deep" => (true, true),
        "no-idf" => (false, false),
        "deep-no-idf" => (true, false),
        _ => {
            return Err(StoreError::Invalid(
                "facets takes basic, deep, no-idf or deep-no-idf".into(),
            ));
        }
    };
    let selected = buckets(reader, selection, deep)?;
    let global = buckets(reader, &Selection::default(), deep)?;
    let idf = if weights && global.total_sessions > 0 {
        global
            .families
            .iter()
            .map(|(family, counts)| {
                (
                    family.clone(),
                    counts
                        .iter()
                        .filter(|(_, n)| **n > 0)
                        .map(|(value, n)| {
                            (
                                value.clone(),
                                (global.total_sessions as f64 / *n as f64).ln(),
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    } else {
        BTreeMap::new()
    };
    Ok(Facets {
        selected,
        global,
        scoped_to_query: selection.sessions.is_some() || selection.origins.is_some(),
        idf,
    })
}

pub(super) const TAGS: &str = "SELECT DISTINCT s.pk, CASE WHEN n.key IS NULL THEN n.value WHEN n.value IS NULL THEN n.key ELSE n.key || ':' || n.value END AS value
 FROM session s JOIN a.note_target nt ON nt.ref='session:' || s.id JOIN a.note n ON n.id=nt.note_id
 WHERE n.kind='tag'
 AND NOT EXISTS(SELECT 1 FROM a.note newer WHERE newer.first_note=n.first_note AND newer.commit_seq>n.commit_seq)
 AND NOT EXISTS(SELECT 1 FROM a.note_relation w WHERE w.kind='withdraws' AND w.other='note:' || n.first_note)";

fn buckets(reader: &Reader<'_>, selection: &Selection, deep: bool) -> Result<Buckets> {
    let (condition, params) = selection.session_condition("s.pk")?;
    let conn = reader.conn()?;
    let (total_sessions, total_messages) = conn.query_row(&format!(
        "SELECT count(*), coalesce(sum((SELECT count(*) FROM message m WHERE m.session_pk=s.pk AND m.duplicate_of IS NULL)),0) FROM session s WHERE {condition}"),
        rusqlite::params_from_iter(params.iter()), |r| Ok((r.get(0)?, r.get(1)?)))?;
    let mut result = Buckets {
        total_sessions,
        total_messages,
        ..Buckets::default()
    };
    let mut families = vec![
        (
            "origins",
            "SELECT pk, origin AS value FROM session".to_owned(),
        ),
        ("tags", TAGS.to_owned()),
    ];
    let deeper = [
        "repositories",
        "roles",
        "written_by",
        "block_kinds",
        "actions",
        "flags",
    ];
    if deep {
        families.extend([
            ("repositories", "SELECT pk, coalesce(repo_root, record_repo, repo) AS value FROM session".into()),
            ("roles", "SELECT session_pk AS pk,role AS value FROM message WHERE duplicate_of IS NULL".into()),
            ("written_by", "SELECT session_pk AS pk,written_by AS value FROM message WHERE duplicate_of IS NULL".into()),
            ("block_kinds", "SELECT b.session_pk AS pk,b.kind AS value FROM block b JOIN message m ON m.pk=b.message_pk WHERE m.duplicate_of IS NULL".into()),
            ("actions", "SELECT t.session_pk AS pk,t.action AS value FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk WHERE m.duplicate_of IS NULL".into()),
            ("flags", "SELECT session_pk AS pk,'thinking' AS value FROM block WHERE kind='thinking' UNION ALL SELECT session_pk,'tool_call' FROM tool_call UNION ALL SELECT session_pk,'attachment' FROM block WHERE kind IN ('image','file')".into()),
        ]);
    } else {
        result.omitted = deeper.to_vec();
    }
    for (name, sql) in families {
        let mut statement = conn.prepare(&format!(
            "SELECT f.value,count(DISTINCT s.pk) FROM session s JOIN ({sql}) f ON f.pk=s.pk WHERE {condition} AND f.value IS NOT NULL GROUP BY f.value ORDER BY f.value"))?;
        let counts = statement
            .query_map(rusqlite::params_from_iter(params.iter()), |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?
            .collect::<rusqlite::Result<Counts>>()?;
        result.families.insert(name.into(), counts);
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
