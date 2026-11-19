//! Compile separate full-text reads from supplied owners or final time order.
//! Ordinary filters never trigger a speculative scope walk.
use super::*;
use crate::query::fields::STRUCTURAL_KINDS;
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
pub(super) struct SearchBuild {
    pub cte: usize,
    pub name: String,
    pub expression: String,
    pub through: usize,
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

/// The selected owner statement's CTEs and final batch membership.
pub(super) struct OwnerRows {
    pub ctes: Vec<String>,
    pub restriction: Option<String>,
}

fn supplied(filter: &Filter) -> bool {
    use polylogue_query::filter::Test;
    match filter {
        Filter::All(parts) => parts.iter().any(supplied),
        Filter::Any(parts) => !parts.is_empty() && parts.iter().all(supplied),
        Filter::Session { .. } | Filter::Rows { .. } => true,
        Filter::Field(field) if field.field.declaration().name == "id" => {
            matches!(&field.test, Test::Prefix(_))
        }
        _ => false,
    }
}

impl Compiler<'_> {
    pub(super) fn search_scope(
        &mut self,
        selection: &Selection,
        scope: &RowContext,
        defaults: &[String],
        recent_order: Option<&[String]>,
    ) -> Result<Option<OwnerRows>, QueryError> {
        let Some(build) = self
            .text_search
            .as_mut()
            .and_then(|search| search.build.take())
        else {
            return Ok(None);
        };
        let filter = selection
            .filter
            .as_ref()
            .filter(|_| selection.unit == scope.unit)
            .and_then(|filter| without_terms(filter, &build.spans));
        let explicit = filter.as_ref().is_some_and(supplied);
        if !explicit && recent_order.is_none() {
            return Ok(None);
        }
        let mut conditions = defaults.to_vec();
        if let Some(filter) = filter {
            conditions.push(self.lower(&filter, scope, false)?);
        }
        let recent = !explicit && recent_order.is_some();
        let batch = recent.then(|| {
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
        let owners = format!("{name}_owners");
        let blocks = format!("{name}_owner_blocks");
        let hits = units::Hits {
            cte: name,
            index,
            source: true,
            scope: Some(&blocks),
        };
        let kind = hits.block_kind("b");
        let writer = hits.writer("m");
        let pk = scope.a.expand(units::spec(scope.unit).key);
        let from = self.from(scope)?;
        let condition = and_all(&conditions);
        let owner_sql = match batch {
            Some(parameter) => format!(
                "SELECT {pk} AS pk FROM json_each(?{parameter}) chosen CROSS JOIN {from} \
                 WHERE {pk}=chosen.value AND {condition}"
            ),
            None => format!("SELECT {pk} AS pk FROM {from} WHERE {condition}"),
        };
        let eligible = match scope.unit {
            Unit::Sessions => format!(
                "WITH RECURSIVE tree(owner_pk,pk) AS (SELECT pk,pk FROM {owners} UNION \
                 SELECT t.owner_pk,l.child_pk FROM tree t CROSS JOIN session p ON p.pk=t.pk \
                 CROSS JOIN session_link l ON l.parent_id=p.id AND l.kind IN ({STRUCTURAL_KINDS}) \
                 WHERE l.child_pk IS NOT NULL) \
                 SELECT t.owner_pk,b.pk AS block,b.text_id FROM tree t \
                 CROSS JOIN message m ON m.session_pk=t.pk CROSS JOIN block b ON b.message_pk=m.pk \
                 WHERE b.text_id IS NOT NULL AND {kind}{writer} AND {}",
                fields::own_messages("m", None)
            ),
            Unit::Messages => format!(
                "SELECT o.pk AS owner_pk,b.pk AS block,b.text_id FROM {owners} o \
                 CROSS JOIN message m ON m.pk=o.pk CROSS JOIN block b ON b.message_pk=m.pk \
                 WHERE b.text_id IS NOT NULL AND {kind}{writer}"
            ),
            _ => unreachable!("owner-driven source units"),
        };
        let Search::Hits { rows } = units::spec(scope.unit).search else {
            unreachable!("selected full-text search")
        };
        let found = rows(&scope.a, &hits).expect("session or message full-text sources");
        let replacement = [
            format!("{owners} AS MATERIALIZED ({owner_sql})"),
            format!("{blocks} AS MATERIALIZED ({eligible})"),
            format!(
                "{name} AS MATERIALIZED (WITH texts AS MATERIALIZED (SELECT DISTINCT text_id FROM {blocks}) \
                 SELECT t.text_id,bm25({table}) AS rank FROM texts t CROSS JOIN {table} \
                 ON {table}.rowid=t.text_id WHERE {table} MATCH {})", build.expression),
        ].into_iter().chain(found.ctes).collect::<Vec<_>>();
        let mut ctes = self.ctes.clone();
        ctes.splice(build.cte..build.through, replacement);
        let restriction =
            batch.map(|parameter| format!("{pk} IN (SELECT value FROM json_each(?{parameter}))"));
        self.text_search.as_mut().expect("selected search").page = match batch {
            Some(batch) => {
                let (hidden, _) = hidden_order(recent_order.expect("time order"), "_o");
                SearchPage::Recent {
                    candidates: format!(
                        "WITH {} SELECT {pk} AS _pk,{} FROM {from} WHERE {condition}",
                        self.ctes.join(", "),
                        hidden.join(", ")
                    ),
                    keys: String::new(),
                    batch,
                }
            }
            None => SearchPage::Owners {
                keys: String::new(),
            },
        };
        Ok(Some(OwnerRows { ctes, restriction }))
    }
}
