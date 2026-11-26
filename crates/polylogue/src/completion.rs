//! Completion: the grammar from the query catalog and the CLI's own tables,
//! and the archive's values from the daemon's readers. The CLI asks the
//! daemon once per keystroke and offers the grammar alone when the daemon
//! does not answer in time.
use polylogue_store::{Reader, Result, completion as values};
use query_language::{
    ast::ResultPart,
    catalog::{Catalog, OutputSchema, OutputSource, SourceOutput},
    complete::{Completion, CompletionKind},
};

pub(crate) fn origin_names() -> Vec<String> {
    polylogue_sources::origin::ORIGINS
        .iter()
        .map(|origin| origin.name.to_owned())
        .collect()
}
fn base_catalog() -> Catalog {
    (*polylogue_query::catalog::catalog(&origin_names())).clone()
}
/// The grammar's candidates at `position`: units, fields, operators,
/// transforms, closed values and reference prefixes.
pub(crate) fn grammar(text: &str, position: usize) -> Vec<Completion> {
    if position > text.len() || !text.is_char_boundary(position) {
        return vec![];
    }
    polylogue_query::catalog::complete(text, position, &base_catalog())
}

/// A saved query name as completion offers it.
struct SavedName {
    name: String,
    help: String,
    /// The fields a query over its rows can name, when it binds alone.
    relation_scope: Option<OutputSchema>,
}

/// The grammar's candidates at `position` with the archive's values: field
/// values with their counts, sessions, saved query names and kept results,
/// and the fields of the saved queries and kept results the query reads.
pub(crate) fn complete(
    reader: &Reader<'_>,
    text: &str,
    position: usize,
) -> Result<Vec<Completion>> {
    if position > text.len() || !text.is_char_boundary(position) {
        return Ok(vec![]);
    }
    let origins = origin_names();
    let mut catalog = base_catalog();
    let mut loaded = Vec::new();
    loop {
        let before = catalog.source_outputs.len();
        for source in query_language::complete::requirements(text, position, &catalog).sources {
            if loaded.contains(&source) {
                continue;
            }
            loaded.push(source.clone());
            let schema = match &source {
                OutputSource::Saved { name, .. } => {
                    saved(reader, &origins, name)?.and_then(|saved| saved.relation_scope)
                }
                OutputSource::Result { reference, part } => {
                    kept_schema(reader, &origins, reference, *part)?
                }
            };
            if let Some(schema) = schema {
                catalog.source_outputs.push(SourceOutput { source, schema });
            }
        }
        if before == catalog.source_outputs.len() {
            break;
        }
    }
    let requirements = query_language::complete::requirements(text, position, &catalog);
    let mut counts = std::collections::BTreeMap::new();
    let mut descriptions = std::collections::BTreeMap::new();
    let mut value_order = std::collections::BTreeMap::new();
    if let Some(value) = &requirements.value {
        let bare_field = value.field.strip_prefix("session.").unwrap_or(&value.field);
        let origin = polylogue_sources::origin::ORIGINS
            .iter()
            .find(|origin| origin.name == bare_field);
        let found = if bare_field == "query" {
            saved_names(reader, &origins, &value.prefix)?
                .into_iter()
                .map(|item| {
                    descriptions.insert(item.name.clone(), item.help);
                    values::ValueCount {
                        value: item.name,
                        count: 1,
                    }
                })
                .collect()
        } else if bare_field == "result" {
            reader
                .completion_kept(&value.prefix)?
                .into_iter()
                .map(|item| {
                    descriptions.insert(item.id.clone(), kept_help(&item));
                    values::ValueCount {
                        value: item.id,
                        count: 1,
                    }
                })
                .collect()
        } else if bare_field == "session" || origin.is_some() {
            let prefix = origin.map_or_else(
                || value.prefix.clone(),
                |origin| format!("{}:{}", origin.name, value.prefix),
            );
            reader
                .completion_sessions(&prefix, &origins)?
                .into_iter()
                .map(|session| {
                    let prefix = match origin {
                        Some(origin) => session
                            .prefix
                            .strip_prefix(&format!("{}:", origin.name))
                            .unwrap_or(&session.prefix),
                        None => &session.prefix,
                    };
                    let value = if prefix.starts_with(&value.prefix) {
                        prefix.to_owned()
                    } else if origin.is_some() {
                        session.native_id.clone()
                    } else {
                        session.id.clone()
                    };
                    descriptions.insert(value.clone(), session_help(&session));
                    values::ValueCount { value, count: 1 }
                })
                .collect()
        } else {
            reader.completion_values(bare_field, &value.prefix)?
        };
        for (rank, item) in found.iter().enumerate() {
            counts.insert(item.value.clone(), item.count);
            value_order.entry(item.value.clone()).or_insert(rank);
        }
        fill_values(&mut catalog, &value.field, &found);
    }
    let mut candidates = query_language::complete::complete(text, position, &catalog);
    let unquoted =
        |value: &str| serde_json::from_str::<String>(value).unwrap_or_else(|_| value.to_owned());
    for candidate in &mut candidates {
        let value = unquoted(&candidate.value);
        if let Some(help) = descriptions.get(&value) {
            candidate.help = help.clone();
        } else if let Some(count) = counts.get(&value) {
            candidate.help = format!("{count} occurrences");
        }
    }
    if requirements.value.is_some() {
        candidates.sort_by_cached_key(|candidate| {
            value_order
                .get(&unquoted(&candidate.value))
                .copied()
                .unwrap_or(usize::MAX)
        });
    }
    if let Some(reference) = requirements.reference {
        let prefix = reference
            .prefix
            .strip_prefix("session:")
            .unwrap_or(&reference.prefix);
        let origin = polylogue_sources::origin::ORIGINS
            .iter()
            .find(|origin| prefix.starts_with(&format!("{}:", origin.name)));
        for session in reader.completion_sessions(prefix, &origins)? {
            let help = session_help(&session);
            let mut replace = reference.replace;
            let value = match origin {
                Some(origin) => {
                    replace.start += origin.name.len() + 1;
                    match session.prefix.strip_prefix(&format!("{}:", origin.name)) {
                        Some(value) => value.to_owned(),
                        None => continue,
                    }
                }
                None => session.prefix,
            };
            candidates.push(Completion {
                value,
                replace,
                kind: CompletionKind::Value,
                help,
            });
        }
        for saved in saved_names(reader, &origins, prefix)? {
            let source = query_language::ast::Source::Saved {
                name: saved.name,
                arguments: vec![],
                condition: None,
                output: query_language::ast::SourceColumns::Unknown,
            };
            if let Some(value) = source_spelling(source, &catalog) {
                candidates.push(Completion {
                    value,
                    replace: reference.replace,
                    kind: CompletionKind::Value,
                    help: saved.help,
                });
            }
        }
        for kept in reader.completion_kept(prefix)? {
            let source = query_language::ast::Source::Result {
                reference: format!("result:{}", kept.id),
                part: ResultPart::Rows,
                condition: None,
                output: query_language::ast::SourceColumns::Unknown,
            };
            if let Some(value) = source_spelling(source, &catalog) {
                candidates.push(Completion {
                    value,
                    replace: reference.replace,
                    kind: CompletionKind::Value,
                    help: kept.name.unwrap_or(kept.id),
                });
            }
        }
    }
    for candidate in &mut candidates {
        candidate.help = polylogue_render::terminal::escape(&candidate.help);
    }
    candidates.retain(|candidate| {
        !candidate.value.chars().any(char::is_control)
            && polylogue_render::terminal::escape(&candidate.value) == candidate.value
    });
    Ok(candidates)
}

/// The saved query `name` binds to, authored or shipped, described for
/// completion; `None` when no saved query has the name.
fn saved(reader: &Reader<'_>, origins: &[String], name: &str) -> Result<Option<SavedName>> {
    let bindings = reader
        .selection_definitions(&format!("query:{name}"), origins)
        .map_err(|error| error.into_store_error())?;
    let Some(definitions) = bindings.saved.get(name) else {
        return Ok(None);
    };
    let shipped = polylogue_model::saved_queries::SHIPPED_QUERIES
        .iter()
        .find(|shipped| shipped.name == name);
    let [definition] = definitions.as_slice() else {
        let text = definitions
            .first()
            .map_or("", |definition| definition.text.as_str());
        return Ok(Some(SavedName {
            name: name.into(),
            help: format!("{text} (ambiguous query name)"),
            relation_scope: None,
        }));
    };
    if !definition.conflicts.is_empty() {
        return Ok(Some(SavedName {
            name: name.into(),
            help: format!("{} (ambiguous query name)", definition.text),
            relation_scope: None,
        }));
    }
    // An authored query of the same name replaces the shipped one.
    let authored = reader
        .completion_saved_names(name)?
        .iter()
        .any(|found| found == name);
    let described = match (authored, shipped) {
        (false, Some(shipped)) => bindings
            .output_metadata(&format!("query:{name}"))
            .map(|(_, relation_scope)| (shipped.help.to_owned(), relation_scope)),
        _ => definition.executable_text().and_then(|syntax| {
            let parameters = bindings.definition_parameters(syntax)?;
            let (_, relation_scope) = bindings.output_metadata(syntax)?;
            let help = if parameters.is_empty() {
                definition.text.clone()
            } else {
                format!(
                    "{} (parameters: {})",
                    definition.text,
                    parameters.join(", ")
                )
            };
            Ok((help, relation_scope))
        }),
    };
    Ok(Some(match described {
        Ok((help, relation_scope)) => SavedName {
            name: name.into(),
            help,
            relation_scope: Some(relation_scope),
        },
        Err(error) => SavedName {
            name: name.into(),
            help: format!("{} ({error})", definition.text),
            relation_scope: None,
        },
    }))
}

/// Saved query names that start with `prefix`: the authored ones, then the
/// shipped ones no authored query replaces, each with its one line.
fn saved_names(reader: &Reader<'_>, origins: &[String], prefix: &str) -> Result<Vec<SavedName>> {
    let mut names = reader.completion_saved_names(prefix)?;
    for shipped in polylogue_model::saved_queries::SHIPPED_QUERIES {
        if shipped.name.starts_with(prefix) && !names.iter().any(|name| name == shipped.name) {
            names.push(shipped.name.into());
        }
    }
    names.sort();
    let mut described = Vec::with_capacity(names.len());
    for name in names {
        if let Some(saved) = saved(reader, origins, &name)? {
            described.push(saved);
        }
    }
    Ok(described)
}

/// The fields of a kept result's rows or match rows, read by ID or name.
fn kept_schema(
    reader: &Reader<'_>,
    origins: &[String],
    reference: &str,
    part: ResultPart,
) -> Result<Option<OutputSchema>> {
    let reference = if reference.starts_with("result:") {
        reference.to_owned()
    } else {
        format!("result:{reference}")
    };
    // A kept result's columns are read the same way a query that reads
    // it binds them; an unavailable or invalid body names no fields.
    let Ok(bindings) = reader.selection_definitions(&reference, origins) else {
        return Ok(None);
    };
    let key = reference.strip_prefix("result:").unwrap_or(&reference);
    let id = bindings.kept_names.get(key).map_or(key, String::as_str);
    Ok(bindings
        .kept
        .get(id)
        .filter(|kept| kept.validate_output_metadata().is_ok())
        .filter(|kept| matches!(part, ResultPart::Rows) || !kept.match_columns.is_empty())
        .map(|kept| kept.output_schema(part)))
}

fn kept_help(item: &values::KeptName) -> String {
    format!(
        "{} · {} · {} rows",
        item.name.as_deref().unwrap_or(&item.id),
        item.unit.as_deref().unwrap_or("unknown unit"),
        item.row_count
            .map_or_else(|| "unknown".into(), |count| count.to_string())
    )
}

fn source_spelling(source: query_language::ast::Source, catalog: &Catalog) -> Option<String> {
    query_language::format::format_query(
        &query_language::ast::Query {
            source,
            attach: vec![],
            transforms: vec![],
        },
        catalog,
    )
    .ok()
}

fn session_help(session: &values::Session) -> String {
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

/// Setting keys that start with `prefix` (`set KEY`), or with `key`, its
/// values (`set KEY VALUE`): the keys and choices the settings table
/// declares, and with `current`, each key's current value. Without
/// `current` (no daemon answered) the declarations alone are offered.
pub(crate) fn settings(
    key: Option<&str>,
    prefix: &str,
    authored_keys: &[String],
    current: impl Fn(&str) -> Option<polylogue_model::settings::Current>,
) -> Vec<Completion> {
    let replace = query_language::ast::Span {
        start: 0,
        end: prefix.len(),
    };
    let candidate = |value: String, help: String| Completion {
        value,
        replace,
        kind: CompletionKind::Value,
        help,
    };
    match key {
        None => {
            let mut keys: Vec<String> = polylogue_model::settings::CATALOG
                .iter()
                .filter(|spec| !spec.key.ends_with('*'))
                .map(|spec| spec.key.to_string())
                .chain(authored_keys.iter().cloned())
                .filter(|key| {
                    key.starts_with(prefix)
                        && polylogue_model::settings::historical_hint(key).is_none()
                })
                .collect();
            keys.sort();
            keys.dedup();
            keys.into_iter()
                .map(|key| {
                    let help = current(&key).map(|current| {
                        if !current.conflicts.is_empty() {
                            format!(
                                "current: conflict: {}",
                                current
                                    .conflicts
                                    .iter()
                                    .map(ToString::to_string)
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        } else if current.supplier == "sinex" {
                            "current: owned by Sinex".into()
                        } else {
                            let value = current
                                .value
                                .as_ref()
                                .and_then(|value| value.text().ok())
                                .unwrap_or_else(|| "unset".into());
                            format!("current: {value}")
                        }
                    });
                    candidate(key, help.unwrap_or_default())
                })
                .collect()
        }
        Some(key) => {
            let Ok(declaration) = polylogue_model::settings::declaration(key) else {
                return vec![];
            };
            let current = current(key);
            if current
                .as_ref()
                .is_some_and(|current| current.supplier == "sinex")
            {
                return vec![];
            }
            let mut choices: Vec<String> = declaration
                .choices()
                .iter()
                .map(|value| value.to_string())
                .collect();
            if let Some(current) = current
                && current.conflicts.is_empty()
                && let Some(value) = current.value
                && let Ok(value) = value.text()
                && !choices.contains(&value)
            {
                choices.insert(0, value);
            }
            choices
                .into_iter()
                .filter(|value| value.starts_with(prefix))
                .map(|value| candidate(value, String::new()))
                .collect()
        }
    }
}

fn fill_values(catalog: &mut Catalog, wanted: &str, found: &[values::ValueCount]) {
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
        if field.name == wanted || field.name.strip_prefix("session.") == Some(wanted) {
            for value in found {
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
