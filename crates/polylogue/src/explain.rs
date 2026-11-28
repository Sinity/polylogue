//! Introspection from the declarations used by commands and query execution.

use crate::commands::{self, COMMANDS};
use crate::daemon::Failure;
use polylogue_query::{ast::Unit, fields, units};
use serde::Serialize;
use serde_json::{Value, json};

const SYNTAX: &str = "FIELD:VALUE matches by the field's kind; * includes every value, including unknown; unknown matches missing values. Comparisons use = != < <= > >=. Adjacent conditions and uppercase AND mean both, OR means either, NOT or - negates; parentheses group conditions. exists UNIT(CONDITION) tests related rows; count(UNIT where CONDITION) compares their count. seq(CONDITION -> CONDITION) follows tool calls. FIELD in (QUERY) matches a grouped query's values. Pipe transforms sort by, group by, agg, count, limit, offset, distinct and matches shape rows; a unit transform selects related rows; union, intersect, except and fuse combine queries; measure ends a sessions query.";

/// The line `polylogue help WORD` prints for a word that left the surface:
/// the row of the "Removed words" table in docs/vocabulary.md naming it.
pub fn removed_word(word: &str) -> Option<String> {
    const VOCABULARY: &str = include_str!("../../../docs/vocabulary.md");
    let table = VOCABULARY.split_once("\n## Removed words\n")?.1;
    let table = table.split("\n## ").next().unwrap_or(table);
    table.lines().find_map(|line| {
        let cells: Vec<&str> = line
            .strip_prefix('|')?
            .strip_suffix('|')?
            .split(" | ")
            .map(str::trim)
            .collect();
        let [words, now] = cells.as_slice() else {
            return None;
        };
        let named = words
            .split('`')
            .skip(1)
            .step_by(2)
            .any(|named| named == word || named.split(' ').next() == Some(word));
        let plain = |text: &str| text.replace("\\|", "|").replace('`', "");
        named.then(|| {
            format!(
                "{}: no longer part of the surface; now {}\n",
                plain(words),
                plain(now)
            )
        })
    })
}

#[derive(Serialize)]
struct Page {
    subject: String,
    unit: Option<String>,
    entries: Vec<Value>,
    total: usize,
    next: Option<String>,
}

pub fn answer(
    subject: &str,
    unit: Option<&str>,
    page_size: Option<u64>,
    cursor: Option<&str>,
) -> Result<Value, Failure> {
    declarations(subject, unit, page_size, cursor).map_err(Failure::bad_request)
}

fn declarations(
    subject: &str,
    unit: Option<&str>,
    page_size: Option<u64>,
    cursor: Option<&str>,
) -> Result<Value, String> {
    if unit.is_some() && subject != "fields" {
        return Err("unit applies to fields".into());
    }
    let unit = unit
        .map(|name| Unit::from_name(name).ok_or_else(|| format!("unknown unit {name:?}")))
        .transpose()?;
    let entries = entries(subject, unit)?;
    let prefix = format!("{subject}/{}/", unit.map_or("", Unit::name));
    let offset = match cursor {
        None => 0,
        Some(text) => text
            .strip_prefix(&prefix)
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or("cursor belongs to another subject or is invalid")?,
    };
    if offset > entries.len() {
        return Err("cursor is past the declarations".into());
    }
    let page_size = usize::try_from(page_size.unwrap_or(20)).unwrap_or(usize::MAX);
    if page_size == 0 {
        return Err("page_size must be positive so the declarations can advance".into());
    }
    let total = entries.len();
    let end = offset.saturating_add(page_size).min(total);
    Ok(json!(Page {
        subject: subject.into(),
        unit: unit.map(|u| u.name().into()),
        total,
        entries: entries
            .into_iter()
            .skip(offset)
            .take(end - offset)
            .collect(),
        next: (end < total).then(|| format!("{prefix}{end}")),
    }))
}

fn entries(subject: &str, unit: Option<Unit>) -> Result<Vec<Value>, String> {
    Ok(match subject {
        "syntax" => {
            let catalog = polylogue_query::catalog::catalog(&[]);
            let keywords: Vec<_> = polylogue_query::catalog::FUNCTIONS.iter().copied()
                .chain(catalog.functions.iter().map(|function| function.name.as_str()))
                .collect();
            vec![json!({"syntax": SYNTAX, "transforms": polylogue_query::catalog::TRANSFORMS.iter().map(|s| s.completion).collect::<Vec<_>>(), "aggregates": polylogue_query::catalog::AGGREGATES, "keywords": keywords})]
        },
        "units" => units::UNITS.iter().map(|u| json!({
            "name": u.name,
            "columns": u.columns.iter().map(|c| c.name).collect::<Vec<_>>(),
            "session_fields": u.in_session,
            "relations": u.relations.iter().map(|r| json!({"prefix": r.prefix, "unit": r.target})).collect::<Vec<_>>(),
        })).collect(),
        "fields" => fields::FIELDS.iter().filter(|f| unit.is_none_or(|u| u == f.unit)).map(|f| json!({
            "unit": f.unit, "name": f.name, "kind": f.kind, "values": f.values, "help": f.help,
            "construct": fields::construct(f),
        })).collect(),
        // The measures follow the transforms: `| measure NAME` ends a query.
        "transforms" => polylogue_query::catalog::TRANSFORMS
            .iter()
            .map(|t| json!({"name": t.name, "completion": t.completion}))
            .chain(crate::daemon::measure::MEASURES.iter().map(|m| json!({
                "name": format!("measure {}", m.name), "help": m.help, "argument": m.argument,
                "metrics": m.metrics, "whole_archive": m.whole_archive,
            })))
            .collect(),
        "capabilities" => {
            // MCP tools list only their operations' common arguments and
            // carry no answer schema; both are published here in full, with
            // the arguments each command and verb leaves out of the tool list.
            let declarations = commands::declarations();
            let unlisted = |arguments: &[command_surface::ArgumentSpec]| {
                arguments
                    .iter()
                    .filter(|argument| !argument.listed && !argument.calling_process)
                    .map(|argument| argument.name.clone())
                    .collect::<Vec<_>>()
            };
            let mut entries: Vec<_> = COMMANDS
                .iter()
                .map(|c| {
                    let description = declarations
                        .descriptions()
                        .iter()
                        .find(|description| description.spec.name == c.name);
                    json!({
                        "kind": "command", "name": c.name, "help": c.help,
                        "tool": c.tool.map(commands::Tool::name), "http": c.http, "writes": c.writes,
                        "arguments": description.map(|d| d.input_schema.clone()),
                        "unlisted": description.map(|d| unlisted(&d.spec.arguments)).unwrap_or_default(),
                        "answer": description.map(|d| d.output_schema.clone()),
                    })
                })
                .collect();
            entries.extend(crate::daemon::verbs::declarations().verbs().iter().map(|verb| {
                json!({
                    "kind": "verb", "name": verb.verb(), "signature": verb.signature(),
                    "arguments": verb.input_schema(), "unlisted": unlisted(verb.arguments()),
                })
            }));
            entries.push(json!({"kind": "answer", "schema": schemars::schema_for!(command_surface::Answer<Value>).to_value()}));
            entries.extend(polylogue_model::read::View::NAMES.iter().map(|name| json!({"kind": "read_view", "name": name})));
            entries.extend(polylogue_render::Format::NAMES.iter().map(|name| json!({"kind": "format", "name": name})));
            entries
        }
        _ => return Err(format!("unknown declaration subject {subject:?}")),
    })
}

#[cfg(test)]
mod tests;
