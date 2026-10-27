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
    pub underived_sessions: u64,
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
    let global = buckets(
        reader,
        &Selection {
            sessions: selection.population.clone(),
            ..Selection::default()
        },
        deep,
    )?;
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

pub(super) fn tags_sql() -> String {
    format!("SELECT DISTINCT s.pk, lower(CASE WHEN n.key IS NULL THEN n.value WHEN n.value IS NULL THEN n.key ELSE n.key || ':' || n.value END) AS value
    FROM session s JOIN a.note_target nt ON nt.ref='session:' || s.id JOIN a.note n ON n.id=nt.note_id
    WHERE n.kind='tag' AND {}",polylogue_model::note_sql::live_note("n"))
}

fn buckets(reader: &Reader<'_>, selection: &Selection, deep: bool) -> Result<Buckets> {
    let (mut condition, params) = selection.session_condition("s.pk")?;
    if selection.sessions.is_none() {
        condition.push_str(&format!(" AND s.kind='main' AND s.stub=0 AND NOT EXISTS(SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id=hn.id WHERE ht.ref='session:'||s.id AND hn.kind='hidden' AND {}) AND NOT EXISTS(SELECT 1 FROM a.excision ex WHERE ex.target='session:'||s.id)",polylogue_model::note_sql::live_note("hn")));
    }
    let conn = reader.conn()?;
    let (total_sessions, total_messages, underived_sessions) = conn.query_row(&format!(
        "SELECT count(*), coalesce(sum(coalesce(p.message_count,(SELECT count(*) FROM message m WHERE m.session_pk=s.pk AND m.duplicate_of IS NULL))),0),coalesce(sum(p.session_pk IS NULL),0) FROM session s LEFT JOIN session_profile p ON p.session_pk=s.pk WHERE {condition}"),
        rusqlite::params_from_iter(params.iter()), |r| Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64, r.get::<_,i64>(2)? as u64)))?;
    let mut result = Buckets {
        total_sessions,
        total_messages,
        underived_sessions,
        ..Buckets::default()
    };
    let mut families = vec![
        (
            "origins",
            "SELECT pk, origin AS value FROM session".to_owned(),
        ),
        ("tags", tags_sql()),
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
        for family in ["roles", "written_by", "block_kinds", "actions", "flags"] {
            families.push((
                family,
                format!("SELECT session_pk AS pk,value FROM session_facet WHERE family='{family}'"),
            ));
        }
    } else {
        result.omitted = deeper.to_vec();
    }
    if deep {
        let keys: Vec<i64> = conn
            .prepare(&format!("SELECT s.pk FROM session s WHERE {condition}"))?
            .query_map(rusqlite::params_from_iter(params.iter()), |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let mut counts = Counts::new();
        for pk in keys {
            let values: std::collections::BTreeSet<_> = super::profiles::repositories(conn, pk)?
                .0
                .into_iter()
                .filter_map(|r| r.identity.or(r.label))
                .collect();
            for value in values {
                *counts.entry(value).or_default() += 1;
            }
        }
        result.families.insert("repositories".into(), counts);
    }
    for (name, sql) in families {
        let mut statement = conn.prepare(&format!(
            "SELECT f.value,count(DISTINCT s.pk) FROM session s JOIN ({sql}) f ON f.pk=s.pk WHERE {condition} AND f.value IS NOT NULL GROUP BY f.value ORDER BY f.value"))?;
        let counts = statement
            .query_map(rusqlite::params_from_iter(params.iter()), |r| {
                Ok((r.get(0)?, r.get::<_, i64>(1)? as u64))
            })?
            .collect::<rusqlite::Result<Counts>>()?;
        result.families.insert(name.into(), counts);
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
