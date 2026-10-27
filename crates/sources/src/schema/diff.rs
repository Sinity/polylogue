use super::{Field, RULE_VERSION, Schema, Selection};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Added,
    Removed,
    Types,
    Required,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Change {
    pub path: String,
    pub kind: ChangeKind,
    pub before: Option<Field>,
    pub after: Option<Field>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Population {
    pub origin: String,
    pub sample: bool,
    pub selection: Vec<Selection>,
    pub records: u64,
    pub unreadable_inputs: u64,
    pub unreadable_records: u64,
}
impl From<&Schema> for Population {
    fn from(s: &Schema) -> Self {
        Self {
            origin: s.origin.clone(),
            sample: s.sample,
            selection: s.selection.clone(),
            records: s.records,
            unreadable_inputs: s.unreadable_inputs,
            unreadable_records: s.unreadable_records,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diff {
    pub rule_version: u32,
    pub before: Population,
    pub after: Population,
    pub named_fields: BTreeSet<String>,
    /// Different samples describe observed drift, not a proven origin-wide change.
    pub comparable: bool,
    pub reasons: Vec<String>,
    pub changes: Vec<Change>,
}

fn validate(schema: &Schema) -> Result<(), String> {
    if schema.rule_version != RULE_VERSION {
        return Err("unsupported schema rule version".into());
    }
    let types = [
        "object", "array", "string", "integer", "number", "boolean", "null",
    ];
    for (path, field) in &schema.fields {
        if !path.is_empty() && !path.starts_with('/') {
            return Err("schema path must start at the root".into());
        }
        if path.is_empty() {
            if field.parents != schema.records
                || field.present != schema.records
                || field.occurrences != schema.records
            {
                return Err("schema root counts disagree with records".into());
            }
        } else {
            let (parent, part) = path.rsplit_once('/').ok_or("invalid schema path")?;
            let parent = schema
                .fields
                .get(parent)
                .ok_or("schema path has no parent")?;
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
    if (schema.records == 0) != schema.fields.is_empty() {
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
    let paths: BTreeSet<_> = before.fields.keys().chain(after.fields.keys()).collect();
    let mut changes = Vec::new();
    for path in paths {
        let left = before.fields.get(path);
        let right = after.fields.get(path);
        let mut add = |kind| {
            changes.push(Change {
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
    Ok(Diff {
        rule_version: RULE_VERSION,
        before: before.into(),
        after: after.into(),
        named_fields: before.named_fields.clone(),
        comparable: reasons.is_empty(),
        reasons,
        changes,
    })
}
