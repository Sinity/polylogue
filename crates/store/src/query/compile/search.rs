//! Search plans chosen by bounded samples of hits and scope blocks.
//! The bound filter keeps every other predicate and its original defaults.
use super::*;
use polylogue_query::filter::Filter;

/// A direct sort on a stored time keeps key-first selection. Other relation sorts
/// keep their existing expression and field-selection rules.
pub(super) fn recent_sort(unit: Unit, transforms: &[Transform]) -> bool {
    let [
        Transform {
            kind: TransformKind::Sort { keys },
            ..
        },
    ] = transforms
    else {
        return false;
    };
    let [key] = keys.as_slice() else { return false };
    let Expression::Field { name } = &key.expression else {
        return false;
    };
    let field = fields::lookup(unit, &name.to_ascii_lowercase());
    use polylogue_query::fields::FieldId;
    key.descending
        && matches!(
            (unit, field),
            (Unit::Sessions, Some(FieldId::SessionsLastActive))
                | (Unit::Messages, Some(FieldId::MessagesAt))
        )
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ScopeBuild {
    pub cte: usize,
    pub name: String,
    pub expression: String,
    pub spans: Vec<Span>,
}

fn without_terms(filter: &Filter, spans: &[Span]) -> Option<Filter> {
    match filter {
        Filter::All(all) => {
            let mut rest = all
                .iter()
                .filter_map(|part| without_terms(part, spans))
                .collect::<Vec<_>>();
            match rest.len() {
                0 => None,
                1 => rest.pop(),
                _ => Some(Filter::All(rest)),
            }
        }
        Filter::Word(word) if spans.contains(&word.span) => None,
        Filter::Field(field) if spans.contains(&field.span) => None,
        _ => Some(filter.clone()),
    }
}

impl Compiler<'_> {
    pub(super) fn search_scope(
        &mut self,
        selection: &Selection,
        scope: &RowContext,
        defaults: &[String],
        recent_order: Option<&[String]>,
    ) -> Result<(), QueryError> {
        let Some(build) = self
            .text_search
            .as_mut()
            .and_then(|search| search.scope_build.take())
        else {
            return Ok(());
        };
        let filter = selection
            .filter
            .as_ref()
            .filter(|_| selection.unit == scope.unit)
            .and_then(|filter| without_terms(filter, &build.spans));
        if filter.is_none() && recent_order.is_none() {
            // With only the unit's defaults there is no narrower scope.
            self.ctes.push(format!(
                "{}_plan AS (SELECT 'ranked_prefix' AS plan)",
                build.name
            ));
            self.ctes.push(format!(
                "{}_scope_sample AS (SELECT NULL AS block,NULL AS text_id WHERE 0)",
                build.name
            ));
            return Ok(());
        }
        let mut conditions = defaults.to_vec();
        if let Some(filter) = filter {
            conditions.push(self.lower(&filter, scope, false)?);
        }
        let hit_budget = self.params.len() + 1;
        let hits = self.bind(Param::Int(crate::search::HIT_BUDGET));
        let scope_budget = self.params.len() + 1;
        let scope_limit = self.bind(Param::Int(crate::search::SCOPE_BUDGET));
        let recent_batch = recent_order.map(|_| {
            let parameter = self.params.len() + 1;
            self.bind(Param::Text("[]".into()));
            parameter
        });
        let search = self.text_search.as_ref().expect("selected search");
        let index = search.index;
        let limit = search.limit;
        let (table, kinds, writer) = crate::search::index_parts(index);
        let name = &build.name;
        let batch_parameter = recent_batch.map(|parameter| format!("?{parameter}"));
        let batch = batch_parameter.as_deref().unwrap_or("'[]'");
        let scope_rows = format!("{name}_scope_rows");
        let scope_blocks = format!("{name}_scope_blocks");
        let pk = scope.a.expand(units::spec(scope.unit).key);
        let from = self.from(scope)?;
        let condition = and_all(&conditions);
        let scope_row_sql = match recent_batch {
            Some(parameter) => format!(
                "SELECT {pk} AS pk FROM {from} WHERE ?{parameter}='[]' AND {condition} UNION ALL \
                 SELECT {pk} FROM json_each(?{parameter}) chosen CROSS JOIN {from} \
                 WHERE ?{parameter}<>'[]' AND {pk}=chosen.value AND {condition}"
            ),
            None => format!("SELECT {pk} AS pk FROM {from} WHERE {condition}"),
        };
        self.ctes.push(format!(
            "{scope_rows} AS NOT MATERIALIZED ({scope_row_sql})"
        ));
        let eligible = match scope.unit {
            Unit::Sessions => format!(
                "WITH RECURSIVE tree(pk) AS (\
                 SELECT pk FROM {scope_rows} WHERE {batch}<>'[]' UNION \
                 SELECT min(pk) FROM {scope_rows} WHERE {batch}='[]' UNION \
                 SELECT l.child_pk FROM tree t CROSS JOIN session p ON p.pk=t.pk \
                 CROSS JOIN session_link l ON l.parent_id=p.id AND l.kind='spawn' \
                 WHERE l.child_pk IS NOT NULL UNION \
                 SELECT (SELECT min(pk) FROM {scope_rows} WHERE pk>t.pk) \
                 FROM tree t WHERE {batch}='[]' AND t.pk IS NOT NULL) \
                 SELECT b.pk AS block,b.text_id FROM tree t \
                 CROSS JOIN message m ON m.session_pk=t.pk \
                 CROSS JOIN block b ON b.message_pk=m.pk \
                 WHERE b.text_id IS NOT NULL AND b.kind IN ({kinds}){writer} AND {}",
                fields::own_messages("m", None)
            ),
            Unit::Messages => format!(
                "SELECT b.pk AS block,b.text_id FROM {scope_rows} o \
                 CROSS JOIN message m ON m.pk=o.pk \
                 CROSS JOIN block b ON b.message_pk=m.pk \
                 WHERE b.text_id IS NOT NULL AND b.kind IN ({kinds}){writer}"
            ),
            _ => unreachable!("scope-driven source units"),
        };
        // Outside a recency batch, scope rows enter the tree one at a time
        // so the sample can stop before reading the whole scope.
        self.ctes
            .push(format!("{scope_blocks} AS NOT MATERIALIZED ({eligible})"));
        self.ctes.extend(crate::search::plan_ctes(
            name,
            index,
            &build.expression,
            &scope_blocks,
            crate::search::Budgets {
                hits: &hits,
                scope: &scope_limit,
            },
            batch_parameter.as_deref(),
        ));
        self.ctes[build.cte] = format!(
            "{name} AS MATERIALIZED (WITH index_prefix AS MATERIALIZED (\
               SELECT rowid AS text_id,bm25({table}) AS rank FROM {table} \
               WHERE (SELECT plan FROM {name}_plan)<>'scope' AND {table} MATCH {expression} \
               ORDER BY {table}.rank LIMIT ?{limit}), \
             scoped_texts AS MATERIALIZED (SELECT DISTINCT text_id FROM {name}_scope_sample \
               WHERE (SELECT plan FROM {name}_plan)='scope') \
             SELECT text_id,rank FROM index_prefix UNION ALL \
             SELECT t.text_id,bm25({table}) AS rank FROM scoped_texts t \
             CROSS JOIN {table} ON {table}.rowid=t.text_id \
             WHERE {table} MATCH {expression})",
            expression = build.expression,
        );
        let choice = format!(
            "WITH RECURSIVE {} SELECT plan FROM {name}_plan",
            self.ctes.join(", ")
        );
        let recent = recent_order.zip(recent_batch).map(|(order, batch)| {
            let (hidden, _) = hidden_order(order, "_o");
            RecentSearch {
                keys: format!(
                    "WITH RECURSIVE {} SELECT {pk} AS _pk,{} FROM {from} WHERE {condition}",
                    self.ctes.join(", "),
                    hidden.join(", ")
                ),
                batch,
            }
        });
        self.text_search.as_mut().expect("selected search").plan = Some(SearchPlan {
            choice,
            hit_budget,
            scope_budget,
            recent,
        });
        Ok(())
    }
}
