//! Settings are single-valued annotation slots. Display time and local commit
//! order never choose a current value or resolve a fork.
use super::*;
use polylogue_model::{annotation::values, settings};
use std::collections::BTreeSet;

type Candidate = (Reference, AnnotationFact, settings::Value);

fn roots(conn: &Connection, key: &str) -> Result<Vec<String>> {
    let source = Source::reading(conn)?;
    Ok(conn.prepare_cached(&source.sql("SELECT DISTINCT root FROM @source.annotation WHERE kind='setting' AND key=?1 ORDER BY root"))?
        .query_map([key], |row| row.get(0))?.collect::<rusqlite::Result<_>>()?)
}

/// SQL only nominates families. Present and as-of reads use the same slot
/// comparison over the heads the current-revision rule gives.
fn candidates(
    conn: &Connection,
    key: &str,
    target: Option<&str>,
    families: impl IntoIterator<Item = String>,
    mut heads: impl FnMut(String) -> Result<Vec<Reference>>,
) -> Result<(Vec<Candidate>, BTreeSet<Reference>)> {
    let mut candidates = Vec::new();
    let mut conflicts = BTreeSet::new();
    for root in families {
        check_read()?;
        let heads = heads(root)?;
        let mut matched = false;
        for reference in &heads {
            let Reference::Annotation(RevisionReference {
                revision: Some(id), ..
            }) = reference
            else {
                return Err(crate::StoreError::stored_data(
                    "a setting head is not an exact annotation",
                ));
            };
            let row = get(conn, id)?.ok_or_else(|| StoreError::NotFound(reference.to_string()))?;
            if row.fact.kind != AnnotationKind::Setting || row.fact.key.as_deref() != Some(key) {
                continue;
            }
            let values::Value::Setting(value) = row
                .fact
                .validate()
                .map_err(|error| crate::StoreError::stored_data(error.to_string()))?
            else {
                return Err(crate::StoreError::stored_data(
                    "a setting annotation has another value shape",
                ));
            };
            if value.fields.machine.as_deref() == target {
                matched = true;
                candidates.push((reference.clone(), row.fact, value.fields.value));
            }
        }
        if matched && heads.len() > 1 {
            conflicts.extend(heads);
        }
    }
    if candidates.len() > 1 {
        conflicts.extend(candidates.iter().map(|(reference, _, _)| reference.clone()));
    }
    Ok((candidates, conflicts))
}

fn history_boundaries(
    conn: &Connection,
    key: &str,
    until: Timestamp,
) -> Result<(BTreeSet<String>, BTreeSet<Timestamp>)> {
    let mut links = StoredLinks::as_of(conn, until)?;
    let source = Source::reading(conn)?;
    let mut statement = conn.prepare_cached(&source.sql("SELECT root,created_at FROM @source.annotation WHERE kind='setting' AND key=?1 ORDER BY root,id"))?;
    let mut rows = statement.query([key])?;
    let mut relevant = BTreeSet::new();
    while let Some(row) = rows.next()? {
        let root: String = row.get(0)?;
        if links.in_view(row.get(1)?) {
            relevant.insert(root);
        }
    }
    drop(rows);
    let mut pending = Vec::new();
    for root in &relevant {
        pending.extend(
            links
                .revisions(root)?
                .into_iter()
                .map(authored::Target::Revision),
        );
        pending.push(authored::Target::Root(root.clone()));
    }
    let mut times = BTreeSet::new();
    let mut visited = BTreeSet::new();
    while let Some(target) = pending.pop() {
        check_read()?;
        if !visited.insert(target.clone()) {
            continue;
        }
        if let authored::Target::Revision(revision) = &target {
            let row = get(conn, &revision.rev)?
                .ok_or_else(|| StoreError::NotFound(reference(revision).to_string()))?;
            times.extend(row.fact.created_at);
            pending.push(authored::Target::Root(revision.root.clone()));
        }
        // Restoration can be arbitrarily deep: walk the same withdrawal
        // links the current-revision rule reads, including withdrawals of whole families.
        pending.extend(
            links
                .withdrawers(&target)?
                .into_iter()
                .map(authored::Target::Revision),
        );
    }
    Ok((relevant, times))
}

impl crate::Reader<'_> {
    pub fn setting_current(&self, key: &str, machine: &str) -> Result<settings::Current> {
        let spec = settings::declaration(key).map_err(polylogue_model::Reason::from)?;
        let conn = self.conn()?;
        let target =
            settings::target_machine(key, None, machine).map_err(polylogue_model::Reason::from)?;
        let (mut candidates, conflicts) =
            candidates(conn, key, target.as_deref(), roots(conn, key)?, |root| {
                Ok(self
                    .annotation_family_state(&RevisionReference {
                        root,
                        revision: None,
                    })?
                    .map(|state| state.heads)
                    .unwrap_or_default())
            })?;
        let default = spec
            .default
            .map(|text| settings::parse(key, text, settings::SettingMode::Standalone))
            .transpose()
            .map_err(|error| {
                crate::StoreError::internal(format!("a declared setting default: {error}"))
            })?;
        let (value, authored) = if conflicts.is_empty() {
            if let Some((reference, fact, value)) = candidates.pop() {
                (
                    Some(value),
                    Some(settings::Authored {
                        reference,
                        machine: fact.machine,
                        author: fact.author,
                        via: fact.via,
                        at: fact.created_at,
                    }),
                )
            } else {
                (default.clone(), None)
            }
        } else {
            (None, None)
        };
        self.check()?;
        Ok(settings::Current {
            key: key.into(),
            kind: spec.kind,
            scope: spec.scope,
            supplier: "polylogue".into(),
            default,
            value,
            authored,
            conflicts: conflicts.into_iter().collect(),
        })
    }

    /// Setting history is a display order; only revision links choose the current value.
    pub fn setting_details(&self, key: &str, machine: &str) -> Result<settings::Details> {
        let spec = settings::declaration(key).map_err(polylogue_model::Reason::from)?;
        let target =
            settings::target_machine(key, None, machine).map_err(polylogue_model::Reason::from)?;
        let current = self.setting_current(key, machine)?;
        let conn = self.conn()?;
        let source = Source::reading(conn)?;
        let ids = conn.prepare_cached(&source.sql(
            "SELECT id FROM @source.annotation WHERE kind='setting' AND key=?1 ORDER BY created_at DESC,id DESC"
        ))?.query_map([key], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut history = Vec::new();
        for id in ids {
            self.check()?;
            let annotation = get(conn, &id)?.ok_or_else(|| StoreError::NotFound(id.clone()))?;
            let values::Value::Setting(value) = annotation
                .fact
                .validate()
                .map_err(|error| crate::StoreError::stored_data(error.to_string()))?
            else {
                return Err(crate::StoreError::stored_data(
                    "a setting revision has another value shape",
                ));
            };
            // A record's machine says who wrote it; the value's machine is
            // the host the setting applies to, including remote writes.
            if value.fields.machine != target {
                continue;
            }
            let revision = RevisionReference::exact(annotation.fact.root.clone(), id.clone());
            let state = self.annotation_revision_state(&revision)?.ok_or_else(|| {
                crate::StoreError::stored_data("a setting revision has no current-revision state")
            })?;
            history.push(settings::HistoryEntry {
                authored: settings::Authored {
                    reference: Reference::Annotation(revision),
                    machine: annotation.fact.machine,
                    author: annotation.fact.author,
                    via: annotation.fact.via,
                    at: annotation.fact.created_at,
                },
                value: value.fields.value,
                current: state.current,
                withdrawn: state.withdrawn,
            });
        }
        self.check()?;
        Ok(settings::Details {
            current,
            machine: target,
            allowed_values: spec
                .choices()
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
            history,
        })
    }

    /// Archive settings as they stood over time, by the writer's clock.
    pub(crate) fn archive_setting_history(
        &self,
        key: &str,
        until: Timestamp,
    ) -> Result<Vec<(Timestamp, SettingAt)>> {
        let spec = settings::declaration(key).map_err(polylogue_model::Reason::from)?;
        if spec.scope != settings::Scope::Archive {
            return Err(crate::StoreError::internal(
                "archive setting history requires an archive-scoped key",
            ));
        }
        let conn = self.conn()?;
        let history = history(conn, key, until)?;
        self.check()?;
        Ok(history)
    }
}

/// A setting over one interval of its history: its value (`None` while
/// unset), or unknown while several families held it at once.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SettingAt {
    Value(Option<settings::Value>),
    Conflict(Vec<String>),
}

fn history(conn: &Connection, key: &str, until: Timestamp) -> Result<Vec<(Timestamp, SettingAt)>> {
    if conn.is_autocommit() {
        return Err(crate::StoreError::internal(
            "setting history requires a pinned read transaction",
        ));
    }
    let (families, boundaries) = history_boundaries(conn, key, until)?;
    let mut history = Vec::new();
    let mut previous = None;
    for boundary in boundaries {
        let (mut selected, conflicts) =
            candidates(conn, key, None, families.iter().cloned(), |root| {
                Ok(Currency::new(&mut StoredLinks::as_of(conn, boundary)?)
                    .heads(&root)
                    .map_err(currency_error)?
                    .iter()
                    .map(reference)
                    .collect())
            })?;
        // A conflicting interval is unknown, not a failure of the whole
        // history; the revision that resolves it ends the interval.
        let (identity, at) = if conflicts.is_empty() {
            let selected = selected.pop();
            (
                selected
                    .as_ref()
                    .map(|(reference, _, _)| vec![reference.to_string()]),
                SettingAt::Value(selected.map(|(_, _, value)| value)),
            )
        } else {
            let conflicting: Vec<String> = conflicts.iter().map(ToString::to_string).collect();
            (Some(conflicting.clone()), SettingAt::Conflict(conflicting))
        };
        if identity != previous {
            previous = identity;
            history.push((boundary, at));
        }
    }
    Ok(history)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "settings/details_tests.rs"]
mod details_tests;
