//! Rendition rows, current revisions and typed reads. Confirmation applies
//! prepared records unchanged through the same codec as standalone writes.
mod imports;
mod rows;
pub use imports::declare_imports;
mod search;
pub use search::{SCHEMA, VariantMatch, project};
#[cfg(test)]
mod tests;

use crate::source::Source;
use crate::{OptionExt as _, ResultExt as _};
use crate::{Reader, Result, StoreError};
use polylogue_model::user::{UserOutcome, UserRequest};
use polylogue_model::{
    Author, BlobHash, NewVariant, Timestamp, Variant, VariantEffect, VariantKind, VariantPart,
    VariantRelation,
};
use rusqlite::{Connection, params};

/// Derive the stable preference key used by `reading.variants`.
pub fn key(variant: &Variant) -> String {
    match &variant.to_lang {
        Some(lang) => format!("{}:{lang}", variant.kind.as_str()),
        None => variant.kind.as_str().into(),
    }
}

pub(crate) fn heads(conn: &Connection, root: &str) -> Result<Vec<Variant>> {
    let root_ref = polylogue_model::Reference::Variant {
        revision: polylogue_model::RevisionReference {
            root: root.into(),
            revision: None,
        },
        part: None,
    };
    crate::assertions::heads(conn, &root_ref)?
        .into_iter()
        .map(|reference| {
            let polylogue_model::Reference::Variant {
                revision,
                part: None,
            } = reference
            else {
                return Err(StoreError::Invalid(
                    "variant graph returned another reference kind".into(),
                ));
            };
            let id = revision
                .revision
                .ok_or_else(|| StoreError::stored(0, "variant head is not exact"))?;
            rows::get(conn, &id)?.ok_or_else(|| StoreError::NotFound(reference_string(root, &id)))
        })
        .collect()
}
fn reference_string(root: &str, id: &str) -> String {
    format!("variant:{root}@{id}")
}
pub(crate) use rows::{get as get_record, parts_while};

pub(crate) fn write(
    conn: &Connection,
    r: &UserRequest,
    new: &NewVariant,
    counter: i64,
) -> Result<UserOutcome> {
    let source =
        polylogue_model::Reference::parse(&new.source).map_err(crate::StoreError::invalid)?;
    if !matches!(
        source,
        polylogue_model::Reference::Session(_)
            | polylogue_model::Reference::Message(_)
            | polylogue_model::Reference::Document(_)
            | polylogue_model::Reference::MessageRange { .. }
            | polylogue_model::Reference::Block { .. }
            | polylogue_model::Reference::Span { .. }
            | polylogue_model::Reference::Assertion(polylogue_model::RevisionReference {
                revision: Some(_),
                ..
            })
            | polylogue_model::Reference::Page(polylogue_model::RevisionReference {
                revision: Some(_),
                ..
            })
            | polylogue_model::Reference::Variant {
                revision: polylogue_model::RevisionReference {
                    revision: Some(_),
                    ..
                },
                ..
            }
    ) {
        return Err(StoreError::Invalid(
            "variant source must name captured text or an exact authored revision".into(),
        ));
    }
    for part in &new.parts {
        polylogue_model::Reference::parse(&part.of).map_err(crate::StoreError::invalid)?;
    }
    let root = if let Some(base) = &new.base {
        let previous = rows::get(conn, base)?
            .ok_or_else(|| StoreError::NotFound(format!("variant {base}")))?;
        let current = heads(conn, &previous.root)?;
        if !current.iter().any(|v| v.id == *base) {
            return Err(StoreError::Stale {
                reference: format!("variant:{}@{base}", previous.root),
                current: current
                    .iter()
                    .map(|v| format!("variant:{}@{}", v.root, v.id))
                    .collect(),
            });
        }
        if previous.source != new.source
            || previous.source_hash != new.source_hash
            || previous.kind != new.kind
            || previous.from_lang != new.from_lang
            || previous.to_lang != new.to_lang
        {
            return Err(StoreError::Invalid(
                "a variant revision must keep its source revision, kind and languages".into(),
            ));
        }
        previous.root
    } else {
        r.id.clone()
    };
    let v = Variant {
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
        commit_seq: counter.try_into().or_invalid("negative variant counter")?,
    };
    rows::insert(conn, &v, counter)?;
    for (n, p) in new.parts.iter().enumerate() {
        rows::insert_part(
            conn,
            &VariantPart {
                variant: v.id.clone(),
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

pub fn capture(conn: &Connection, _request: &UserRequest, counter: i64) -> Result<VariantEffect> {
    let mut effect = VariantEffect::default();
    rows::visit(conn, "WHERE commit_seq=?1", Some(counter), |v| {
        rows::parts(conn, &v.id, |p| {
            effect.variant_parts.push(p);
            Ok(())
        })?;
        effect.variants.push(v);
        Ok(())
    })?;
    Ok(effect)
}
/// No selector or stale-base check runs during confirmed application.
pub fn apply(conn: &Connection, effect: &VariantEffect, counter: i64) -> Result<()> {
    for v in &effect.variants {
        rows::insert(conn, v, counter)?;
    }
    for p in &effect.variant_parts {
        rows::insert_part(conn, p)?;
    }
    Ok(())
}
pub fn restore(
    conn: &Connection,
    effect: &VariantEffect,
    current: &polylogue_model::authored::ImportAttribution,
    counter: i64,
) -> Result<()> {
    imports::restore(conn, effect, current, counter)
}

pub fn copy(from: &Connection, to: &Connection) -> Result<()> {
    rows::visit(from, "", None, |v| {
        rows::insert(to, &v, 0)?;
        rows::parts(from, &v.id, |p| rows::insert_part(to, &p))
    })?;
    imports::copy(from, to)
}

impl Reader<'_> {
    pub fn variant(&self, id: &str) -> Result<Option<Variant>> {
        rows::get(self.conn()?, id)
    }

    /// An explicit read may show an earlier source revision. Fresh matches
    /// take precedence within each key; clocks never break an ambiguity.
    pub fn selected_variant(
        &self,
        source: &str,
        source_hash: BlobHash,
        keys: &[String],
        allow_stale: bool,
    ) -> Result<Option<Variant>> {
        let conn = self.conn()?;
        let namespace = Source::reading(conn)?;
        let roots: Vec<String> = conn
            .prepare_cached(
                &namespace
                    .sql("SELECT DISTINCT root FROM @source.variant WHERE source=?1 ORDER BY root"),
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
                        .map(|v| format!("variant:{}@{}", v.root, v.id))
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
