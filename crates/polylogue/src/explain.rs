//! Introspection from the declarations used by commands and query execution.

use crate::commands::{self, COMMANDS};
use crate::daemon::query::Failure;
use polylogue_query::{ast::Unit, fields, units};
use serde::Serialize;
use serde_json::{Value, json};

pub const SUBJECTS: &[&str] = &[
    "query",
    "syntax",
    "units",
    "fields",
    "operators",
    "capabilities",
    "derived",
];

const SYNTAX: &str = "FIELD:VALUE matches by the field's kind; * includes every value, including unknown; unknown matches missing values. Comparisons use = != < <= > >=. Adjacent conditions and uppercase AND mean both, OR means either, NOT or - negates; parentheses group conditions. exists UNIT(CONDITION) tests related rows; count(UNIT where CONDITION) compares their count. seq(CONDITION -> CONDITION) follows tool calls. FIELD in (QUERY) matches a grouped query's values. Pipe stages sort by, group by, agg, count, limit, offset, distinct and matches shape rows; a unit stage selects related rows; union, intersect, except and fuse combine queries; measure ends a sessions query.";

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
    expression: Option<&str>,
    unit: Option<&str>,
    page_size: Option<u64>,
    cursor: Option<&str>,
) -> Result<Value, Failure> {
    declarations(subject, expression, unit, page_size, cursor).map_err(|message| Failure {
        kind: "bad_request".into(),
        message,
        detail: None,
    })
}

fn declarations(
    subject: &str,
    expression: Option<&str>,
    unit: Option<&str>,
    page_size: Option<u64>,
    cursor: Option<&str>,
) -> Result<Value, String> {
    if !SUBJECTS.contains(&subject) {
        return Err(format!(
            "unknown subject {subject:?}; choose {}",
            SUBJECTS.join(", ")
        ));
    }
    if expression.is_some() {
        return Err("expression applies to query".into());
    }
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
        "derived" => polylogue_model::derived::all().into_iter().map(|p|json!(p)).collect(),
        "syntax" => vec![json!({"syntax": SYNTAX, "stages": polylogue_query::syntax::STAGES, "aggregates": polylogue_query::syntax::AGGREGATES, "keywords": polylogue_query::syntax::KEYWORDS})],
        "units" => units::UNITS.iter().map(|u| json!({
            "name": u.name, "singular": u.singular,
            "columns": u.columns.iter().map(|c| c.0).collect::<Vec<_>>(),
            "session_fields": u.in_session,
            "relations": u.relations.iter().map(|r| json!({"prefix": r.prefix, "unit": r.target})).collect::<Vec<_>>(),
        })).collect(),
        "fields" => fields::FIELDS.iter().filter(|f| unit.is_none_or(|u| u == f.unit)).map(|f| json!({
            "unit": f.unit, "name": f.name, "kind": f.kind, "values": f.values, "help": f.help,
            "construct": fields::construct(f),
        })).collect(),
        "operators" => polylogue_query::ast::Op::TABLE.iter().map(|(op, _)| json!({"name": op.name(), "operation": op})).collect(),
        "capabilities" => {
            let mut entries: Vec<_> = COMMANDS.iter().map(|c| json!({
                "kind": "command", "name": c.name, "help": c.help,
                "mcp": c.mcp, "http": c.http, "writes": c.writes,
                "arguments": commands::schema(c),
            })).collect();
            entries.extend(polylogue_model::read::View::NAMES.iter().map(|name| json!({"kind": "read_view", "name": name})));
            entries.extend(polylogue_render::Format::NAMES.iter().map(|name| json!({"kind": "format", "name": name})));
            entries
        }
        _ => return Err(format!("unknown declaration subject {subject:?}")),
    })
}

#[cfg(test)]
mod tests;
