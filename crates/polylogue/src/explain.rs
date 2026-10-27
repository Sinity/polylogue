//! Introspection from the declarations used by commands and query execution.

use crate::commands::{self, COMMANDS};
use crate::daemon::{Response, service};
use polylogue_query::{Options, ast::Unit, fields, units};
use serde::Serialize;
use serde_json::{Value, json};

pub const SUBJECTS: &[&str] = &["query", "units", "fields", "operators", "capabilities"];

const SYNTAX: &str = "FIELD:VALUE matches by the field's kind; * means known and unknown means missing. Comparisons use = != < <= > >=. Adjacent conditions and uppercase AND mean both, OR means either, NOT or - negates; parentheses group conditions. exists UNIT(CONDITION) tests related rows; count(UNIT where CONDITION) compares their count. seq(CONDITION -> CONDITION) follows tool calls. FIELD in (QUERY) matches a grouped query's values. Pipe stages sort by, group by, agg, count, limit, offset, distinct and matches shape rows; a unit stage selects related rows; union, intersect, except and fuse combine queries; measure ends a sessions query.";

#[derive(Serialize)]
struct Page {
    syntax: &'static str,
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
    limit: Option<u64>,
    continuation: Option<&str>,
) -> Response {
    match explain(subject, expression, unit, limit, continuation) {
        Ok(value) if value["valid"] == false => Response::Error {
            kind: value["kind"].as_str().unwrap_or("invalid_query").into(),
            message: value["message"].as_str().unwrap_or("invalid query").into(),
            detail: Some(value),
        },
        Ok(value) => service::ok(value),
        Err(message) => service::error("bad_request", message),
    }
}

fn explain(
    subject: &str,
    expression: Option<&str>,
    unit: Option<&str>,
    limit: Option<u64>,
    continuation: Option<&str>,
) -> Result<Value, String> {
    if !SUBJECTS.contains(&subject) {
        return Err(format!(
            "unknown subject {subject:?}; choose {}",
            SUBJECTS.join(", ")
        ));
    }
    if subject == "query" {
        if unit.is_some() || limit.is_some() || continuation.is_some() {
            return Err("unit, limit and continuation apply to catalogs".into());
        }
        let expression = expression
            .filter(|s| !s.trim().is_empty())
            .ok_or("expression is required for query")?;
        let options = Options {
            clock: polylogue_query::Clock::new(
                service::now_ms(),
                polylogue_model::LogicalDay::system(),
            ),
            origins: polylogue_sources::origin::ORIGINS
                .iter()
                .map(|o| o.name.to_owned())
                .collect(),
            command_words: commands::command_words(),
            page_size: None,
            random_seed: Some(0),
            embeddings: false,
            self_session: None,
            saved: Default::default(),
            kept: Default::default(),
            fields: Vec::new(),
            chat_pages: crate::daemon::query::chat_pages(),
        };
        return match polylogue_query::explain::query(expression, &options) {
            Ok(explanation) => Ok(json!({
                "subject": subject,
                "valid": true,
                "explanation": explanation,
                "execution": "Compile to one ordered SQL statement with bound parameters; no query was executed. The clock uses the machine zone and default logical day, without archive settings or stored definitions.",
            })),
            Err(error) => Ok(json!({
                "subject": subject,
                "expression": expression,
                "valid": false,
                "kind": error.kind(),
                "suggestion": error.suggestion(),
                "error": error,
                "message": error.to_string(),
                "parsed": polylogue_query::parse(expression).ok(),
            })),
        };
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
    let entries = catalog(subject, unit);
    let prefix = format!("{subject}/{}/", unit.map_or("", Unit::name));
    let offset = match continuation {
        None => 0,
        Some(text) => text
            .strip_prefix(&prefix)
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or("continuation belongs to another catalog or is invalid")?,
    };
    if offset > entries.len() {
        return Err("continuation is past the catalog".into());
    }
    let limit = usize::try_from(limit.unwrap_or(20)).unwrap_or(usize::MAX);
    if limit == 0 {
        return Err("limit must be positive so the catalog can advance".into());
    }
    let total = entries.len();
    let end = offset.saturating_add(limit).min(total);
    Ok(json!(Page {
        syntax: SYNTAX,
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

fn catalog(subject: &str, unit: Option<Unit>) -> Vec<Value> {
    match subject {
        "units" => units::UNITS.iter().map(|u| json!({
            "name": u.name, "singular": u.singular,
            "columns": u.columns.iter().map(|c| c.0).collect::<Vec<_>>(),
            "session_fields": u.in_session,
            "relations": u.relations.iter().map(|r| json!({"prefix": r.prefix, "unit": r.target})).collect::<Vec<_>>(),
        })).collect(),
        "fields" => fields::FIELDS.iter().filter(|f| unit.is_none_or(|u| u == f.unit)).map(|f| json!({
            "unit": f.unit, "name": f.name, "kind": f.kind, "values": f.values, "help": f.help,
        })).collect(),
        "operators" => polylogue_query::ast::Op::ALL.iter().map(|op| json!({"name": op.name(), "operation": op})).collect(),
        "capabilities" => {
            let mut entries: Vec<_> = COMMANDS.iter().map(|c| json!({
                "kind": "command", "name": c.name, "help": c.help,
                "mcp": c.mcp, "http": c.http, "writes": c.writes,
                "arguments": commands::schema(c),
            })).collect();
            entries.extend(polylogue_render::View::NAMES.iter().map(|name| json!({"kind": "read_view", "name": name})));
            entries.extend(polylogue_render::Format::NAMES.iter().map(|name| json!({"kind": "format", "name": name})));
            entries
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests;
