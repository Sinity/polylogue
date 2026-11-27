//! Alternate rows, current revisions and typed reads. Confirmation applies
//! prepared records unchanged through the same codec as standalone writes.
mod rows;
mod search;
mod sql;
pub(crate) use search::project_through;
pub use search::{AlternateMatch, SCHEMA, project};
pub(crate) use sql::{current_sql, withdrawn_sql};
#[cfg(test)]
mod tests;

use crate::ResultExt as _;
use crate::source::Source;
use crate::{Reader, Result, StoreError};
use polylogue_model::user::{UserOutcome, UserRequest};
use polylogue_model::{
    Alternate, AlternateEffect, AlternateKind, AlternatePart, AlternateRelation, Author, BlobHash,
    NewAlternate, Timestamp,
};
use rusqlite::{Connection, params};

/// Derive the stable preference key used by `reading.alternates`.
pub fn key(alternate: &Alternate) -> String {
    match &alternate.to_lang {
        Some(lang) => format!("{}:{lang}", alternate.kind.as_str()),
        None => alternate.kind.as_str().into(),
    }
}

pub(crate) fn heads(conn: &Connection, root: &str) -> Result<Vec<Alternate>> {
    let root_ref = polylogue_model::Reference::Alternate {
        revision: polylogue_model::RevisionReference {
            root: root.into(),
            revision: None,
        },
        part: None,
    };
    crate::annotations::heads(conn, &root_ref)?
        .into_iter()
        .map(|reference| {
            let polylogue_model::Reference::Alternate {
                revision,
                part: None,
            } = reference
            else {
                return Err(StoreError::stored_data(
                    "alternate graph returned another reference kind",
                ));
            };
            let id = revision
                .revision
                .ok_or_else(|| StoreError::stored(0, "alternate head is not exact"))?;
            rows::get(conn, &id)?.ok_or_else(|| StoreError::NotFound(reference_string(root, &id)))
        })
        .collect()
}
pub(crate) fn reference_string(root: &str, id: &str) -> String {
    polylogue_model::Reference::Alternate {
        revision: polylogue_model::RevisionReference::exact(root.into(), id.into()),
        part: None,
    }
    .to_string()
}
pub(crate) use rows::{get as get_record, parts_while};

pub(crate) fn write(
    conn: &Connection,
    r: &UserRequest,
    new: &NewAlternate,
    counter: i64,
) -> Result<UserOutcome> {
    let source = polylogue_model::Reference::parse(&new.source).map_err(|error| {
        polylogue_model::Reason::BadReference {
            text: new.source.clone(),
            problem: error.to_string(),
        }
    })?;
    if !matches!(
        source,
        polylogue_model::Reference::Session(_)
            | polylogue_model::Reference::Message(_)
            | polylogue_model::Reference::Document(_)
            | polylogue_model::Reference::MessageRange { .. }
            | polylogue_model::Reference::Block { .. }
            | polylogue_model::Reference::Span { .. }
            | polylogue_model::Reference::Annotation(polylogue_model::RevisionReference {
                revision: Some(_),
                ..
            })
            | polylogue_model::Reference::Page(polylogue_model::RevisionReference {
                revision: Some(_),
                ..
            })
            | polylogue_model::Reference::Alternate {
                revision: polylogue_model::RevisionReference {
                    revision: Some(_),
                    ..
                },
                ..
            }
    ) {
        return Err(polylogue_model::Reason::AlternateSource {}.into());
    }
    for part in &new.parts {
        polylogue_model::Reference::parse(&part.of).map_err(|error| {
            polylogue_model::Reason::BadReference {
                text: part.of.clone(),
                problem: error.to_string(),
            }
        })?;
    }
    let root = if let Some(base) = &new.base {
        let previous = rows::get(conn, base)?
            .ok_or_else(|| StoreError::NotFound(format!("alternate {base}")))?;
        let current = heads(conn, &previous.root)?;
        if !current.iter().any(|v| v.id == *base) {
            return Err(StoreError::Stale {
                reference: reference_string(&previous.root, base),
                current: current
                    .iter()
                    .map(|v| reference_string(&v.root, &v.id))
                    .collect(),
            });
        }
        if previous.source != new.source
            || previous.source_hash != new.source_hash
            || previous.kind != new.kind
            || previous.from_lang != new.from_lang
            || previous.to_lang != new.to_lang
        {
            return Err(polylogue_model::Reason::AlternateRevisionChanges {}.into());
        }
        previous.root
    } else {
        r.id.clone()
    };
    let v = Alternate {
        id: r.id.clone(),
        root,
        base: new.base.clone(),
        source: new.source.clone(),
        source_hash: new.source_hash,
        kind: new.kind,
        from_lang: new.from_lang.clone(),
        to_lang: new.to_lang.clone(),
        author: r.authorship.author.clone(),
        via: r.authorship.via.as_str().to_owned(),
        method: new.method.clone(),
        machine: r.machine.clone(),
        created_at: r.at,
        commit_seq: counter.try_into().or_stored("negative alternate counter")?,
    };
    rows::insert(conn, &v, counter)?;
    for (n, p) in new.parts.iter().enumerate() {
        rows::insert_part(
            conn,
            &AlternatePart {
                alternate: v.id.clone(),
                n: n as u64,
                of: p.of.clone(),
                of_hash: p.of_hash,
                relation: p.relation,
                body: p.body.clone(),
            },
        )?;
    }
    Ok(UserOutcome {
        id: r.id.clone(),
        repeat: false,
        pending: false,
    })
}

pub fn capture(conn: &Connection, _request: &UserRequest, counter: i64) -> Result<AlternateEffect> {
    let mut effect = AlternateEffect::default();
    rows::visit(conn, "WHERE commit_seq=?1", Some(counter), |v| {
        rows::parts(conn, &v.id, |p| {
            effect.alternate_parts.push(p);
            Ok(())
        })?;
        effect.alternates.push(v);
        Ok(())
    })?;
    Ok(effect)
}
/// No selector or stale-base check runs during confirmed application.
pub fn apply(conn: &Connection, effect: &AlternateEffect, counter: i64) -> Result<()> {
    for v in &effect.alternates {
        rows::insert(conn, v, counter)?;
    }
    for p in &effect.alternate_parts {
        rows::insert_part(conn, p)?;
    }
    Ok(())
}
pub fn copy(from: &Connection, to: &Connection) -> Result<()> {
    rows::visit(from, "", None, |v| {
        rows::insert(to, &v, 0)?;
        rows::parts(from, &v.id, |p| rows::insert_part(to, &p))
    })
}

impl Reader<'_> {
    pub fn alternate(&self, id: &str) -> Result<Option<Alternate>> {
        rows::get(self.conn()?, id)
    }

    /// An explicit read may show an earlier source revision. Fresh matches
    /// take precedence within each key; clocks never break an ambiguity.
    pub fn selected_alternate(
        &self,
        source: &str,
        source_hash: BlobHash,
        keys: &[String],
        allow_stale: bool,
    ) -> Result<Option<Alternate>> {
        let conn = self.conn()?;
        let namespace = Source::reading(conn)?;
        let roots: Vec<String> = conn
            .prepare_cached(
                &namespace.sql(
                    "SELECT DISTINCT root FROM @source.alternate WHERE source=?1 ORDER BY root",
                ),
            )?
            .query_map([source], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let mut candidates = Vec::new();
        for root in roots {
            let mut current = heads(conn, &root)?;
            if current.len() == 1 {
                candidates.push(current.remove(0));
            }
        }
        for preference in keys {
            let mut matched: Vec<_> = candidates
                .iter()
                .filter(|v| key(v) == *preference && v.source_hash == source_hash)
                .collect();
            if matched.is_empty() && allow_stale {
                matched = candidates
                    .iter()
                    .filter(|v| key(v) == *preference)
                    .collect();
            }
            if matched.len() > 1 {
                return Err(StoreError::Ambiguous {
                    reference: preference.clone(),
                    candidates: matched
                        .iter()
                        .map(|v| reference_string(&v.root, &v.id))
                        .collect(),
                });
            }
            if let Some(v) = matched.first() {
                return Ok(Some((*v).clone()));
            }
        }
        Ok(None)
    }
}
