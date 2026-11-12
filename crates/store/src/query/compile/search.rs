//! A physical scope-driven alternative to the selected full-text leg.
//! The bound filter keeps every other predicate and its original defaults.
use super::*;
use polylogue_query::filter::Filter;

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
    ) -> Result<(), QueryError> {
        let Some(build) = self
            .ranked_search
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
        let Some(filter) = filter else {
            // With only the native defaults there is no narrower scope to
            // drive this search. Keep the ranked index-prefix path.
            self.ctes.push(format!(
                "{}_scope_candidates AS (SELECT NULL AS block WHERE 0)",
                build.name
            ));
            return Ok(());
        };
        let mut conditions = defaults.to_vec();
        conditions.push(self.lower(&filter, scope, false)?);
        let search = self.ranked_search.as_ref().expect("selected search");
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
        self.ctes.push(format!(
            "{owners} AS NOT MATERIALIZED (SELECT {} AS pk FROM {} WHERE {})",
            scope.a.expand(units::spec(scope.unit).key),
            self.from(scope)?,
            and_all(&conditions)
        ));
        let eligible = match scope.unit {
            Unit::Sessions => format!(
                "WITH RECURSIVE tree(pk) AS (\
                 SELECT pk FROM {owners} WHERE {mode}=1 UNION \
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
            "{candidates} AS MATERIALIZED (SELECT * FROM {blocks} WHERE {mode}=1)"
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
             WHERE {mode}=1 AND {table} MATCH {expression})",
            expression = build.expression,
            limit = search.limit,
        );
        let blocks = format!("WITH {} SELECT block FROM {blocks}", self.ctes.join(", "));
        self.ranked_search.as_mut().expect("selected search").scope = Some(ScopedSearch {
            blocks,
            mode: build.mode,
        });
        Ok(())
    }
}
