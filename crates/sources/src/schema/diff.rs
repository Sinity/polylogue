use super::{Field, RULE_VERSION, Schema, Selection, Table};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Added,
    Removed,
    Types,
    Required,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Change {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table: Option<String>,
    pub path: String,
    pub kind: ChangeKind,
    pub before: Option<Field>,
    pub after: Option<Field>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Sample {
    pub origin: String,
    pub sample: bool,
    pub selection: Vec<Selection>,
    pub records: u64,
    pub unreadable_inputs: u64,
    pub unreadable_records: u64,
}
impl From<&Schema> for Sample {
    fn from(schema: &Schema) -> Self {
        Self {
            origin: schema.origin.clone(),
            sample: schema.sample,
            selection: schema.selection.clone(),
            records: schema.records,
            unreadable_inputs: schema.unreadable_inputs,
            unreadable_records: schema.unreadable_records,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TableChange {
    pub table: String,
    pub before: Option<Table>,
    pub after: Option<Table>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Diff {
    pub rule_version: u32,
    pub before: Sample,
    pub after: Sample,
    pub named_fields: BTreeSet<String>,
    /// Different samples describe observed schema differences, not a proven origin-wide change.
    pub comparable: bool,
    pub reasons: Vec<String>,
    pub changes: Vec<Change>,
    pub tables: Vec<TableChange>,
}

fn validate(schema: &Schema) -> Result<(), String> {
    if schema.rule_version != RULE_VERSION {
        return Err("unsupported schema rule version".into());
    }
    validate_fields(
        &schema.fields,
        if schema.fields.is_empty() && !schema.tables.is_empty() {
            0
        } else {
            schema.records
        },
    )?;
    if !schema.tables.is_empty()
        && schema
            .tables
            .values()
            .try_fold(0u64, |n, t| n.checked_add(t.records))
            != Some(schema.records)
    {
        return Err("table record counts disagree with records".into());
    }
    for table in schema.tables.values() {
        validate_fields(&table.fields, table.records)?;
        let unobserved = super::unobserved_declared_columns(&table.metadata, &table.fields);
        if table.unobserved_declared_columns != unobserved {
            return Err("declared column coverage disagrees with observed fields".into());
        }
        if table.metadata_headers > table.headers
            || table.unobserved_columns != (table.records == 0)
        {
            return Err("invalid table coverage".into());
        }
    }
    Ok(())
}

fn validate_fields(
    fields: &std::collections::BTreeMap<String, Field>,
    records: u64,
) -> Result<(), String> {
    let types = [
        "object", "array", "string", "integer", "number", "boolean", "null",
    ];
    for (path, field) in fields {
        if !path.is_empty() && !path.starts_with('/') {
            return Err("schema path must start at the root".into());
        }
        if path.is_empty() {
            if field.parents != records || field.present != records || field.occurrences != records
            {
                return Err("schema root counts disagree with records".into());
            }
        } else {
            let (parent, part) = path.rsplit_once('/').ok_or("invalid schema path")?;
            let parent = fields.get(parent).ok_or("schema path has no parent")?;
            let kind = if part == "[]" { "array" } else { "object" };
            if field.parents != parent.types.get(kind).copied().unwrap_or(0) || field.parents == 0 {
                return Err("schema field parent count is invalid".into());
            }
        }
        if field.types.is_empty()
            || field.types.keys().any(|t| !types.contains(&t.as_str()))
            || field.types.values().any(|n| *n == 0)
            || field
                .types
                .values()
                .try_fold(0u64, |n, c| n.checked_add(*c))
                != Some(field.occurrences)
            || field.present > field.parents
            || field.occurrences < field.present
            || field.missing != field.parents - field.present
            || field.required != (field.parents > 0 && field.present == field.parents)
        {
            return Err("invalid schema field counts or types".into());
        }
        if path.split('/').skip(1).any(|part| {
            let mut chars = part.chars();
            while let Some(c) = chars.next() {
                if c == '~' && !matches!(chars.next(), Some('0' | '1' | '2' | '3')) {
                    return true;
                }
            }
            false
        }) {
            return Err("schema path contains an invalid escape".into());
        }
    }
    if (records == 0) != fields.is_empty() {
        return Err("schema record count and fields disagree".into());
    }
    Ok(())
}

pub fn compare(before: &Schema, after: &Schema) -> Result<Diff, String> {
    validate(before)?;
    validate(after)?;
    if before.named_fields != after.named_fields {
        return Err(
            "schema field-name selections differ; infer both with the same --field arguments"
                .into(),
        );
    }
    let mut reasons = Vec::new();
    if before.origin != after.origin {
        reasons.push("different_origins".into());
    }
    if before.sample != after.sample || before.selection != after.selection {
        reasons.push("different_samples".into());
    }
    if before.unreadable_inputs > 0
        || after.unreadable_inputs > 0
        || before.unreadable_records > 0
        || after.unreadable_records > 0
    {
        reasons.push("unreadable_inputs_or_records".into());
    }
    if before.records == 0 || after.records == 0 {
        reasons.push("empty_sample".into());
    }
    let mut changes = field_changes(&before.fields, &after.fields, None);
    let mut tables = Vec::new();
    let names: BTreeSet<_> = before.tables.keys().chain(after.tables.keys()).collect();
    for name in names {
        let left = before.tables.get(name);
        let right = after.tables.get(name);
        let empty = std::collections::BTreeMap::new();
        changes.extend(field_changes(
            left.map_or(&empty, |t| &t.fields),
            right.map_or(&empty, |t| &t.fields),
            Some(name),
        ));
        if left.is_none()
            || right.is_none()
            || left.map(|t| &t.metadata) != right.map(|t| &t.metadata)
        {
            tables.push(TableChange {
                table: name.clone(),
                before: left.cloned(),
                after: right.cloned(),
            });
        }
    }
    Ok(Diff {
        rule_version: RULE_VERSION,
        before: before.into(),
        after: after.into(),
        named_fields: before.named_fields.clone(),
        comparable: reasons.is_empty(),
        reasons,
        changes,
        tables,
    })
}

fn field_changes(
    before: &std::collections::BTreeMap<String, Field>,
    after: &std::collections::BTreeMap<String, Field>,
    table: Option<&str>,
) -> Vec<Change> {
    let paths: BTreeSet<_> = before.keys().chain(after.keys()).collect();
    let mut changes = Vec::new();
    for path in paths {
        let left = before.get(path);
        let right = after.get(path);
        let mut add = |kind| {
            changes.push(Change {
                table: table.map(str::to_owned),
                path: path.clone(),
                kind,
                before: left.cloned(),
                after: right.cloned(),
            })
        };
        match (left, right) {
            (None, Some(_)) => add(ChangeKind::Added),
            (Some(_), None) => add(ChangeKind::Removed),
            (Some(a), Some(b)) => {
                if !a.types.keys().eq(b.types.keys()) {
                    add(ChangeKind::Types);
                }
                if a.required != b.required {
                    add(ChangeKind::Required);
                }
            }
            _ => {}
        }
    }
    changes
}
