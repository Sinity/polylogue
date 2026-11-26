//! Saved queries (`query:NAME`) and kept results (`result:<id>`): a
//! saved query where a condition stands is the relation binding bound it
//! to, and a kept result reads back the rows it stored.

use super::*;

impl Compiler<'_> {
    /// `query:NAME` where a condition stands, bound to its saved relation:
    /// the relation binds the unit's defaults by its own top-level
    /// conditions, wherever it stands.
    pub(super) fn saved_relation(
        &mut self,
        relation: &polylogue_query::filter::SavedRelation,
        scope: &RowContext,
        cx: Cx<'_>,
    ) -> Result<String, QueryError> {
        let selection = &relation.selection;
        let defaults = selection.defaults.get(cx.measured);
        let mut parts = self.defaults(scope, defaults);
        if let Some(filter) = &selection.filter {
            parts.push(self.lower(filter, scope, cx.inner())?);
        }
        self.sessions_within_bounds(scope, defaults, &mut parts, selection.filter.is_some());
        let used = Definition {
            name: relation.name.clone(),
            revision: relation.revision.clone(),
        };
        if !self.definitions.contains(&used) {
            self.definitions.push(used);
        }
        Ok(if parts.is_empty() {
            "1".into()
        } else {
            format!("({})", parts.join(" AND "))
        })
    }
}

/// A query whose source is a kept result (`result:<id>`, `.rows` or
/// `.matches`): its stored values, read as they were kept, never re-run.
/// Only stages may follow.
impl Compiler<'_> {
    pub(super) fn kept_program(mut self, bound: &Bound) -> Result<Program, QueryError> {
        let query = &bound.query;
        let Source::Result {
            reference,
            part,
            condition,
            ..
        } = &query.source
        else {
            return Err(QueryError::Invalid {
                message: "a kept source is required".into(),
                position: 0,
            });
        };
        if condition.is_some() || !query.attach.is_empty() {
            return Err(QueryError::Invalid {
                message: "A kept result takes transforms, not conditions or attached rows".into(),
                position: 0,
            });
        }
        let id = reference.strip_prefix("result:").unwrap_or(reference);
        let opts = self.opts;
        let kept = opts.kept.get(id).ok_or_else(|| QueryError::Unavailable {
            reason: format!("kept result {id:?} is unavailable"),
        })?;
        let matched = *part == ResultPart::Matches;
        let part = if matched { "matches" } else { "rows" };
        let columns = if matched {
            &kept.match_columns
        } else {
            &kept.columns
        };
        if matched && columns.is_empty() {
            return Err(QueryError::Invalid {
                message: format!("result:{id} kept no match rows"),
                position: 0,
            });
        }
        let unit = Unit::from_name(&kept.unit).ok_or_else(|| QueryError::Unavailable {
            reason: format!("kept result {id:?} has unknown unit {:?}", kept.unit),
        })?;
        let named_unit = Some(unit);
        let shape = if matched {
            Shape::Matches
        } else {
            kept.shape.unwrap_or(Shape::Rows)
        };
        let id_param = self.text_param(id);
        let mut select = vec![
            "j.key AS _pk".into(),
            format!("json_array({id_param}, '{part}') AS _kept_source"),
        ];
        let mut slots = std::collections::BTreeMap::new();
        let mut time_text_columns = std::collections::BTreeSet::new();
        for (index, column) in columns.iter().enumerate() {
            let path = self.text_param(&format!(
                "$.{}",
                serde_json::to_string(&column.name).expect("column name is JSON text")
            ));
            let value = if matches!(column.kind, ColKind::Ids | ColKind::Json) {
                format!("j.value -> {path}")
            } else if column.kind == ColKind::Money {
                // Kept as exact dollars; read back as nano-USD from its text.
                format!("pl_kept_money(j.value -> {path})")
            } else {
                format!("json_extract(j.value, {path})")
            };
            let slot = format!("_kept_value{index}");
            select.push(format!("{value} AS {}", column_sql(&slot)));
            slots.insert(column.name.clone(), slot);
            if column.kind == ColKind::Time {
                time_text_columns.insert(column.name.clone());
            }
        }
        let native_unit =
            named_unit.filter(|_| !matched && kept.row_identity && shape == Shape::Rows);
        let row_kind = if let Some(unit) = native_unit {
            let mut identity = vec![];
            let count = units::spec(unit).identity.len();
            for index in 0..count {
                let slot = format!("_kept_identity{index}");
                select.push(format!("pl_kept_identity(k.rows_json -> ('$.identities[' || j.key || ']'), {index}, {count}) AS {slot}"));
                identity.push(slot);
            }
            RowKind::UnitRows {
                unit,
                columns: identity,
            }
        } else {
            RowKind::Distinct {
                columns: vec!["_kept_source".into(), "_pk".into()],
            }
        };
        let program = Program {
            query: self.text.into(),
            row_kind,
            unit,
            shape,
            statements: Statements::Derived(DerivedRows {
                rows: format!(
                    "SELECT {} FROM source_saved_rows k, json_each(k.rows_json, '$.{part}') j WHERE k.id={id_param} ORDER BY j.key",
                    select.join(", ")
                ),
                count: None,
                members: None,
            }),
            params: std::mem::take(&mut self.params),
            columns: columns.clone(),
            column_slots: slots,
            row_sources: Some(vec![]),
            session_rows: SessionRows::Physical,
            time_text_columns,
            ranked: false,
            snippet_terms: vec![],
            id_checks: vec![],
            change_generations: vec![],
            annotation_cutoffs: vec![],
            attach: vec![],
            notes: vec![],
            incomplete: vec![],
            why: vec![],
            why_defaults: vec![],
            why_of: format!(
                "SELECT count(*) FROM source_saved_rows k, json_each(k.rows_json, '$.{part}') j WHERE k.id={id_param}"
            ),
            page_size: opts.page_size.unwrap_or(default_page_size(unit, shape)),
            near: None,
            counts_near: false,
            definitions: vec![],
            clock: opts.clock.clone(),
            resolved_times: vec![],
            measure: None,
            order: vec![OrderKey::ascending("_pk")],
        };
        self.program_over(bound, program)
    }
}
