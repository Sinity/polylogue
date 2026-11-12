//! A physical scope-driven alternative to the selected full-text leg.
//! The bound filter keeps every other predicate and its original defaults.
use super::*;
use polylogue_query::filter::Filter;

/// A direct native-time sort keeps key-first selection. Other relation sorts
/// retain their existing expression and projection rules.
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
    pub mode: usize,
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
            // With only the native defaults there is no narrower scope to
            // drive this search. Keep the ranked index-prefix path.
            self.ctes.push(format!(
                "{}_scope_candidates AS (SELECT NULL AS block WHERE 0)",
                build.name
            ));
            return Ok(());
        }
        let mut conditions = defaults.to_vec();
        if let Some(filter) = filter {
            conditions.push(self.lower(&filter, scope, false)?);
        }
        let recent_owners = recent_order.map(|_| {
            let parameter = self.params.len() + 1;
            self.bind(Param::Text("[]".into()));
            parameter
        });
        let search = self.text_search.as_ref().expect("selected search");
        let (table, index) = match search.index {
            polylogue_query::search::SnippetIndex::Prose => ("text_fts", Index::Prose),
            polylogue_query::search::SnippetIndex::Output => ("output_fts", Index::Output),
        };
        let name = &build.name;
        let mode = format!("?{}", build.mode);
        let owners = format!("{name}_scope_owners");
        let blocks = format!("{name}_scope_blocks");
        let candidates = format!("{name}_scope_candidates");
        let hits = units::Hits {
            cte: name,
            index,
            source: false,
            scope: None,
        };
        let kind = hits.block_kind("b");
        let writer = hits.writer("m");
        let pk = scope.a.expand(units::spec(scope.unit).key);
        let from = self.from(scope)?;
        let condition = and_all(&conditions);
        let owner_rows = match recent_owners {
            Some(parameter) => format!(
                "SELECT {pk} AS pk FROM {from} WHERE {mode}<>2 AND {condition} UNION ALL \
                 SELECT {pk} FROM json_each(?{parameter}) chosen CROSS JOIN {from} \
                 WHERE {mode}=2 AND {pk}=chosen.value AND {condition}"
            ),
            None => format!("SELECT {pk} AS pk FROM {from} WHERE {condition}"),
        };
        self.ctes
            .push(format!("{owners} AS NOT MATERIALIZED ({owner_rows})"));
        let eligible = match scope.unit {
            Unit::Sessions => format!(
                "WITH RECURSIVE tree(pk) AS (\
                 SELECT pk FROM {owners} WHERE {mode}<>0 UNION \
                 SELECT min(pk) FROM {owners} WHERE {mode}=0 UNION \
                 SELECT l.child_pk FROM tree t CROSS JOIN session p ON p.pk=t.pk \
                 CROSS JOIN session_link l ON l.parent_id=p.id AND l.kind='spawn' \
                 WHERE l.child_pk IS NOT NULL UNION \
                 SELECT (SELECT min(pk) FROM {owners} WHERE pk>t.pk) \
                 FROM tree t WHERE {mode}=0 AND t.pk IS NOT NULL) \
                 SELECT b.pk AS block,b.text_id FROM tree t \
                 CROSS JOIN message m ON m.session_pk=t.pk \
                 CROSS JOIN block b ON b.message_pk=m.pk \
                 WHERE b.text_id IS NOT NULL AND {kind}{writer} AND {}",
                fields::own_messages("m", None)
            ),
            Unit::Messages => format!(
                "SELECT b.pk AS block,b.text_id FROM {owners} o \
                 CROSS JOIN message m ON m.pk=o.pk \
                 CROSS JOIN block b ON b.message_pk=m.pk \
                 WHERE b.text_id IS NOT NULL AND {kind}{writer}"
            ),
            _ => unreachable!("scope-driven source units"),
        };
        // The recursive probe admits owners one at a time: seeding every
        // owner would read the full scope before its first block. Execution
        // seeds all owners directly. Only execution materializes blocks, and it
        // shares that materialization with text probes and owner selection.
        self.ctes
            .push(format!("{blocks} AS NOT MATERIALIZED ({eligible})"));
        self.ctes.push(format!(
            "{candidates} AS MATERIALIZED (SELECT * FROM {blocks} WHERE {mode}<>0)"
        ));
        self.ctes[build.cte] = format!(
            "{name} AS MATERIALIZED (WITH index_prefix AS MATERIALIZED (\
               SELECT rowid AS text_id,bm25({table}) AS rank FROM {table} \
               WHERE {mode}=0 AND {table} MATCH {expression} \
               ORDER BY {table}.rank LIMIT ?{limit}), \
             scoped_texts AS MATERIALIZED (SELECT DISTINCT text_id FROM {candidates}) \
             SELECT text_id,rank FROM index_prefix UNION ALL \
             SELECT t.text_id,bm25({table}) AS rank FROM scoped_texts t \
             CROSS JOIN {table} ON {table}.rowid=t.text_id \
             WHERE {mode}<>0 AND {table} MATCH {expression})",
            expression = build.expression,
            limit = search.limit,
        );
        let blocks = format!("WITH {} SELECT block FROM {blocks}", self.ctes.join(", "));
        let recent = recent_order.zip(recent_owners).map(|(order, owners)| {
            let (hidden, _) = hidden_order(order, "_o");
            RecentSearch {
                keys: format!(
                    "WITH {} SELECT {pk} AS _pk,{} FROM {from} WHERE {condition}",
                    self.ctes.join(", "),
                    hidden.join(", ")
                ),
                owners,
            }
        });
        self.text_search.as_mut().expect("selected search").scope = Some(ScopedSearch {
            blocks,
            mode: build.mode,
            recent,
        });
        Ok(())
    }
}
