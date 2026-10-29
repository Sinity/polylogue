//! Read-only, on-demand completion over the daemon's derived catalog.
use polylogue_store::completion::CatalogReader;
use query_language::{
    ast::ResultPart,
    catalog::{Catalog, OutputSource, SourceOutput},
    complete::{Completion, CompletionKind},
};
use std::path::{Path, PathBuf};
pub mod publisher;

pub fn path(root: &Path) -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default()
                .join(".cache")
        });
    base.join("polylogue")
        .join(crate::runtime_dir::archive_key(root))
        .join("catalog")
}
/// Completion opens only this derived file. Missing, obsolete or corrupt
/// snapshots leave the static grammar available.
pub fn complete(root: &Path, text: &str, position: usize) -> Vec<Completion> {
    complete_file(
        &path(root),
        &crate::runtime_dir::archive_key(root),
        text,
        position,
    )
}
pub(crate) fn complete_file(
    path: &Path,
    archive: &str,
    text: &str,
    position: usize,
) -> Vec<Completion> {
    if position > text.len() || !text.is_char_boundary(position) {
        return vec![];
    }
    let reader = CatalogReader::open(path, archive).ok().flatten();
    complete_snapshot(reader.as_ref(), text, position).unwrap_or_else(|_| {
        query_language::complete::complete(
            text,
            position,
            &crate::daemon::warm::Completion::default().catalog,
        )
    })
}
fn complete_snapshot(
    reader: Option<&CatalogReader>,
    text: &str,
    position: usize,
) -> polylogue_store::Result<Vec<Completion>> {
    let Some(reader) = reader else {
        return Ok(query_language::complete::complete(
            text,
            position,
            &crate::daemon::warm::Completion::default().catalog,
        ));
    };
    let mut catalog = reader.catalog()?;
    for source in query_language::complete::requirements(text, position, &catalog).sources {
        let schema = match &source {
            OutputSource::Saved { name, .. } => {
                let definitions = reader.saved(name)?;
                if definitions.len() == 1 {
                    definitions
                        .into_iter()
                        .next()
                        .and_then(|definition| definition.schema)
                } else {
                    None
                }
            }
            OutputSource::Result { reference, part } => reader
                .kept(reference.strip_prefix("result:").unwrap_or(reference))?
                .and_then(|kept| match part {
                    ResultPart::Rows => kept.rows,
                    ResultPart::Matches => kept.matches,
                }),
        };
        if let Some(schema) = schema {
            catalog.source_outputs.push(SourceOutput { source, schema });
        }
    }
    let requirements = query_language::complete::requirements(text, position, &catalog);
    let mut counts = std::collections::BTreeMap::new();
    let mut descriptions = std::collections::BTreeMap::new();
    let mut value_order = std::collections::BTreeMap::new();
    if let Some(value) = &requirements.value {
        let short = value.field.strip_prefix("session.").unwrap_or(&value.field);
        let origin = polylogue_sources::origin::ORIGINS
            .iter()
            .find(|origin| origin.name == short);
        let values = if short == "query" {
            reader
                .saved_names(&value.prefix)?
                .into_iter()
                .map(|item| {
                    descriptions.insert(item.name.clone(), item.help);
                    polylogue_store::completion::ValueCount {
                        value: item.name,
                        count: 1,
                    }
                })
                .collect()
        } else if short == "result" {
            reader
                .kept_names(&value.prefix)?
                .into_iter()
                .map(|item| {
                    descriptions.insert(
                        item.id.clone(),
                        format!(
                            "{} · {} · {} rows",
                            item.name.as_deref().unwrap_or(&item.id),
                            item.unit.as_deref().unwrap_or("unknown unit"),
                            item.row_count
                                .map_or_else(|| "unknown".into(), |count| count.to_string())
                        ),
                    );
                    polylogue_store::completion::ValueCount {
                        value: item.id,
                        count: 1,
                    }
                })
                .collect()
        } else if short == "session" || origin.is_some() {
            let prefix = origin.map_or_else(
                || value.prefix.clone(),
                |origin| format!("{}:{}", origin.name, value.prefix),
            );
            reader
                .sessions(&prefix)?
                .into_iter()
                .map(|session| {
                    let prefix = if origin.is_some() {
                        session.prefix.split_once(':').unwrap().1
                    } else {
                        &session.prefix
                    };
                    let value = if prefix.starts_with(&value.prefix) {
                        prefix.to_owned()
                    } else if origin.is_some() {
                        session.native_id.clone()
                    } else {
                        session.id.clone()
                    };
                    descriptions.insert(value.clone(), session_help(&session));
                    polylogue_store::completion::ValueCount { value, count: 1 }
                })
                .collect()
        } else {
            reader.values(short, &value.prefix)?
        };
        for (rank, item) in values.iter().enumerate() {
            counts.insert(item.value.clone(), item.count);
            value_order.entry(item.value.clone()).or_insert(rank);
        }
        fill_values(&mut catalog, &value.field, &values);
    }
    let mut candidates = query_language::complete::complete(text, position, &catalog);
    for candidate in &mut candidates {
        let value = serde_json::from_str::<String>(&candidate.value)
            .unwrap_or_else(|_| candidate.value.clone());
        if let Some(help) = descriptions.get(&value) {
            candidate.help = help.clone();
        } else if let Some(count) = counts.get(&value) {
            candidate.help = format!("{count} occurrences");
        }
    }
    if requirements.value.is_some() {
        candidates.sort_by_cached_key(|candidate| {
            value_order
                .get(
                    &serde_json::from_str::<String>(&candidate.value)
                        .unwrap_or_else(|_| candidate.value.clone()),
                )
                .copied()
                .unwrap_or(usize::MAX)
        });
    }
    if let Some(reference) = requirements.reference {
        let prefix = reference
            .prefix
            .strip_prefix("session:")
            .unwrap_or(&reference.prefix);
        let sessions = reader.sessions(prefix)?;
        let references = sessions.into_iter().map(|session| {
            let help = session_help(&session);
            Completion {
                value: session.prefix,
                replace: reference.replace,
                kind: CompletionKind::Value,
                help,
            }
        });
        candidates.extend(references);
    }
    let written_at = jiff::Timestamp::from_millisecond(reader.metadata.written_at)
        .ok()
        .map(|at| at.to_string());
    for candidate in &mut candidates {
        if candidate.kind == CompletionKind::Value {
            if let Some(written_at) = &written_at {
                let separator = if candidate.help.is_empty() {
                    ""
                } else {
                    " · "
                };
                candidate
                    .help
                    .push_str(&format!("{separator}values as of {written_at}"));
            }
        }
        candidate.help = polylogue_render::terminal::escape(&candidate.help);
    }
    candidates.retain(|candidate| {
        !candidate.value.chars().any(char::is_control)
            && polylogue_render::terminal::escape(&candidate.value) == candidate.value
    });
    Ok(candidates)
}
fn session_help(session: &polylogue_store::completion::Session) -> String {
    format!(
        "{} · {}{}{}",
        session.title,
        session.origin,
        session
            .repo
            .as_ref()
            .map(|repo| format!(" · {repo}"))
            .unwrap_or_default(),
        session
            .last_activity
            .map(|at| format!(" · {at}"))
            .unwrap_or_default()
    )
}
/// Setting consumers use their command table position, outside the query grammar.
pub fn settings(root: &Path, key: Option<&str>, prefix: &str) -> Vec<(String, String)> {
    settings_file(
        &path(root),
        &crate::runtime_dir::archive_key(root),
        key,
        prefix,
    )
}
fn settings_file(
    path: &Path,
    archive: &str,
    key: Option<&str>,
    prefix: &str,
) -> Vec<(String, String)> {
    let Ok(Some(reader)) = CatalogReader::open(path, archive) else {
        return vec![];
    };
    let Ok(settings) = reader.settings(key.unwrap_or(prefix)) else {
        return vec![];
    };
    match key {
        None => settings
            .into_iter()
            .map(|setting| {
                (
                    setting.key,
                    format!("current: {}", setting.value.as_str().unwrap_or("unset")),
                )
            })
            .collect(),
        Some(key) => settings
            .into_iter()
            .filter(|setting| setting.key == key)
            .flat_map(|setting| {
                let mut choices = setting.choices;
                if let Some(value) = setting.value.as_str() {
                    if !choices.iter().any(|choice| choice == value) {
                        choices.insert(0, value.into());
                    }
                }
                choices
                    .into_iter()
                    .filter(move |value| value.starts_with(prefix))
                    .map(|value| (value, String::new()))
            })
            .collect(),
    }
}

fn fill_values(
    catalog: &mut Catalog,
    name: &str,
    values: &[polylogue_store::completion::ValueCount],
) {
    for field in catalog
        .units
        .iter_mut()
        .flat_map(|unit| &mut unit.fields)
        .chain(
            catalog
                .source_outputs
                .iter_mut()
                .flat_map(|source| &mut source.schema.fields),
        )
    {
        if field.name == name || field.name.strip_prefix("session.") == Some(name) {
            for value in values {
                if !field.values.contains(&value.value)
                    && !value.value.chars().any(char::is_control)
                {
                    field.values.push(value.value.clone());
                }
            }
        }
    }
}
#[cfg(test)]
mod tests;
