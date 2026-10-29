//! Canonical rendition rows, revision currency and typed reads. Confirmation
//! applies exact prepared records through the same codec as standalone writes.
mod rows;
mod search;
pub use search::{SCHEMA, VariantMatch, project};
#[cfg(test)]
mod tests;

use crate::source::Source;
use crate::{Reader, Result, StoreError};
use polylogue_model::user::{UserOutcome, UserRequest};
use polylogue_model::{
    Author, BlobHash, NewVariant, Timestamp, Variant, VariantEffect, VariantKind, VariantPart,
    VariantRelation,
};
use rusqlite::{Connection, params};
use std::collections::{HashMap, HashSet};

/// Derive the stable preference key used by `reading.variants`.
pub fn key(variant: &Variant) -> String {
    match &variant.to_lang {
        Some(lang) => format!("{}:{lang}", variant.kind.as_str()),
        None => variant.kind.as_str().into(),
    }
}

/// Withdrawal is a graph fact. A withdrawal can itself be withdrawn; cycles
/// have no defined active state and are refused rather than choosing a winner.
struct Withdrawals {
    by_target: HashMap<String, Vec<String>>,
}
impl Withdrawals {
    fn read(conn: &Connection) -> Result<Self> {
        let source = rows::reading(conn)?;
        let mut statement=conn.prepare(&source.sql("SELECT r.note_id,r.other FROM @source.note_relation r JOIN @source.note n ON n.id=r.note_id WHERE r.kind='withdraws' AND n.kind='withdrawal'"))?;
        let mut rows = statement.query([])?;
        let mut by_target: HashMap<String, Vec<String>> = HashMap::new();
        while let Some(row) = rows.next()? {
            by_target.entry(row.get(1)?).or_default().push(row.get(0)?);
        }
        Ok(Self { by_target })
    }
    fn target(&self, target: &str, path: &mut HashSet<String>) -> Result<bool> {
        if !path.insert(target.into()) {
            return Err(StoreError::Invalid(
                "withdrawal graph contains a cycle".into(),
            ));
        }
        let mut withdrawn = false;
        if let Some(writers) = self.by_target.get(target) {
            for id in writers {
                if !self.target(&format!("note:{id}"), path)? {
                    withdrawn = true;
                    break;
                }
            }
        }
        path.remove(target);
        Ok(withdrawn)
    }
    fn variant(&self, v: &Variant) -> Result<bool> {
        Ok(
            self.target(&format!("variant:{}", v.root), &mut HashSet::new())?
                || self.target(&format!("variant:{}@{}", v.root, v.id), &mut HashSet::new())?,
        )
    }
}

fn heads(conn: &Connection, root: &str) -> Result<Vec<Variant>> {
    let withdrawals = Withdrawals::read(conn)?;
    let mut live = Vec::new();
    rows::visit(conn, "", None, |v| {
        if v.root == root && !withdrawals.variant(&v)? {
            live.push(v);
        }
        Ok(())
    })?;
    let revised: HashSet<_> = live.iter().filter_map(|v| v.base.as_deref()).collect();
    Ok(live
        .iter()
        .filter(|v| !revised.contains(v.id.as_str()))
        .cloned()
        .collect())
}

pub(crate) fn write(
    conn: &Connection,
    r: &UserRequest,
    new: &NewVariant,
    counter: i64,
) -> Result<UserOutcome> {
    if new.source.is_empty() || new.via.is_empty() {
        return Err(StoreError::Invalid(
            "variant source and channel are required".into(),
        ));
    }
    let root = if let Some(base) = &new.base {
        let previous = rows::get(conn, base)?
            .ok_or_else(|| StoreError::NotFound(format!("variant {base}")))?;
        let current = heads(conn, &previous.root)?;
        if !current.iter().any(|v| v.id == *base) {
            return Err(StoreError::Invalid(format!(
                "stale variant {base}; current revisions: {}",
                current
                    .iter()
                    .map(|v| v.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
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
        author: new.author.clone(),
        via: new.via.clone(),
        method: new.method.clone(),
        machine: r.machine.clone(),
        created_at: r.at,
        commit_seq: counter
            .try_into()
            .map_err(|_| StoreError::Invalid("negative variant counter".into()))?,
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
pub fn copy(from: &Connection, to: &Connection) -> Result<()> {
    rows::visit(from, "", None, |v| {
        rows::insert(to, &v, 0)?;
        rows::parts(from, &v.id, |p| rows::insert_part(to, &p))
    })
}

impl Reader<'_> {
    pub fn variant(&self, id: &str) -> Result<Option<Variant>> {
        rows::get(self.conn()?, id)
    }
    pub fn visit_variant_parts(
        &self,
        id: &str,
        visit: impl FnMut(VariantPart) -> Result<()>,
    ) -> Result<()> {
        rows::parts(self.conn()?, id, visit)
    }
    pub fn current_variants(&self, root: &str) -> Result<Vec<Variant>> {
        heads(self.conn()?, root)
    }
    /// Select one unambiguous current rendition, in preference order. The
    /// caller supplies the text hash of its exact source reading.
    pub fn enabled_variant(
        &self,
        source: &str,
        source_hash: BlobHash,
        enabled: &[String],
    ) -> Result<Option<Variant>> {
        let conn = self.conn()?;
        let mut roots = HashSet::new();
        rows::visit(conn, "", None, |v| {
            if v.source == source {
                roots.insert(v.root);
            }
            Ok(())
        })?;
        let mut candidates = Vec::new();
        for root in roots {
            let mut current = heads(conn, &root)?;
            if current.len() == 1 {
                let v = current.remove(0);
                if v.source_hash == source_hash {
                    candidates.push(v);
                }
            }
        }
        for preference in enabled {
            let matched: Vec<_> = candidates
                .iter()
                .filter(|v| key(v) == *preference)
                .collect();
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
