use super::*;
use query_language::catalog::{Field, FieldKind, OutputSchema};

impl Kept {
    pub fn validate_output_metadata(&self) -> Result<(), QueryError> {
        let invalid = |message: String| QueryError::Invalid {
            message,
            position: 0,
        };
        for (declared, columns) in [
            (&self.output_metadata.rows, &self.columns),
            (&self.output_metadata.matches, &self.match_columns),
        ] {
            let Some(schema) = declared else {
                continue;
            };
            let unique: std::collections::BTreeSet<_> =
                schema.fields.iter().map(|field| &field.name).collect();
            if unique.len() != schema.fields.len() || schema.fields.len() != columns.len() {
                return Err(invalid(
                    "kept logical declarations disagree with physical columns".into(),
                ));
            }
            for (field, column) in schema.fields.iter().zip(columns) {
                if field.name != column.name
                    || !compatible(&field.kind, column.kind)
                    || column.reference.is_some()
                        && (column.kind != ColKind::Text || field.kind != FieldKind::Reference)
                {
                    return Err(invalid(format!(
                        "kept logical declaration disagrees with physical column {:?}",
                        column.name
                    )));
                }
            }
            if schema.main_time.as_ref().is_some_and(|name| {
                !schema
                    .fields
                    .iter()
                    .any(|field| &field.name == name && field.kind == FieldKind::Time)
            }) {
                return Err(invalid(
                    "kept main time does not name a declared time column".into(),
                ));
            }
        }
        if matches!(
            self.shape,
            Some(Shape::Groups | Shape::Summary | Shape::Matches)
        ) && self
            .output_metadata
            .rows
            .as_ref()
            .is_some_and(|schema| schema.unit.is_some())
            || self
                .output_metadata
                .matches
                .as_ref()
                .is_some_and(|schema| schema.unit.is_some())
        {
            return Err(invalid(
                "kept grouped, summary and match outputs have no native row domain".into(),
            ));
        }
        Ok(())
    }

    pub fn output_schema(&self, part: ResultPart) -> OutputSchema {
        let (declared, columns) = match part {
            ResultPart::Rows => (&self.output_metadata.rows, &self.columns),
            ResultPart::Matches => (&self.output_metadata.matches, &self.match_columns),
        };
        let mut schema = declared
            .clone()
            .unwrap_or_else(|| physical_schema(&self.unit, columns));
        if declared.is_none()
            && (part == ResultPart::Matches
                || matches!(
                    self.shape,
                    Some(Shape::Groups | Shape::Summary | Shape::Matches)
                ))
        {
            schema.unit = None;
        }
        // Logical declarations retain no native period or pattern evidence.
        schema.main_time = None;
        schema.match_units = None;
        schema
    }
}

fn compatible(logical: &FieldKind, physical: ColKind) -> bool {
    match physical {
        ColKind::Text => matches!(
            logical,
            FieldKind::Text
                | FieldKind::Exact
                | FieldKind::Substring
                | FieldKind::Words
                | FieldKind::Subtree
                | FieldKind::Reference
        ),
        ColKind::Number => matches!(logical, FieldKind::Number | FieldKind::Integer),
        ColKind::Money => *logical == FieldKind::Money,
        ColKind::Time => *logical == FieldKind::Time,
        ColKind::Duration | ColKind::DurationSeconds => *logical == FieldKind::Duration,
        ColKind::Bool => *logical == FieldKind::Bool,
        ColKind::Ids => matches!(logical, FieldKind::List(kind) if **kind == FieldKind::Reference),
        ColKind::Json => matches!(logical, FieldKind::Json | FieldKind::List(_)),
    }
}

pub(super) fn physical_schema(unit: &str, columns: &[Column]) -> OutputSchema {
    OutputSchema {
        unit: (!unit.is_empty()).then(|| unit.to_owned()),
        fields: columns
            .iter()
            .map(|column| Field {
                name: column.name.clone(),
                kind: if column.kind == ColKind::Text && column.reference.is_some() {
                    FieldKind::Reference
                } else {
                    super::source::column_kind(column.kind)
                },
                unit: column.reference.map(|reference| reference.unit().into()),
                ..Field::default()
            })
            .collect(),
        main_time: None,
        match_units: None,
    }
}
