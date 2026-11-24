//! The plans of a ranked text search (docs/search.md: The search leg):
//! the index drives, the non-text scope drives, or recency drives. Each is
//! its own statement over the same rows; the scope keeps every other
//! predicate of the bound filter and its original defaults.
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

/// The filter with the word and field terms at `spans` removed, keeping
/// every other predicate; `None` when no predicate remains.
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

/// The rows a ranked text search's non-text conditions select, from which
/// the scope and recency plans start.
pub(super) struct SearchScope {
    /// `{from} WHERE {condition}` selects the scope's rows, keyed by `key`.
    from: String,
    condition: String,
    key: String,
    /// The recency order's hidden order columns, when recency can drive.
    recent: Option<Vec<String>>,
}

/// How the scope plans read the scope's rows.
#[derive(Clone, Copy, PartialEq)]
enum ScopeRows<'a> {
    /// The planning probe: sessions one at a time, so deciding the plan
    /// never reads the whole scope before its first block.
    Probe,
    /// Every scope row.
    All,
    /// Only the batch of row keys bound as a JSON array at this parameter.
    Batch(&'a str),
}

impl Compiler<'_> {
    /// The scope that can drive a ranked text search: the selection's
    /// conditions without the ranked words, and the recency order. `None`
    /// when the search cannot run scoped, or only the unit's defaults
    /// remain, which never narrow it.
    pub(super) fn search_scope(
        &mut self,
        selection: &Selection,
        scope: &RowContext,
        ranking: Option<&Ranking>,
        defaults: &[String],
        recent_order: Option<&[String]>,
        measured: bool,
    ) -> Result<Option<SearchScope>, QueryError> {
        let Some(leg) = ranking
            .and_then(Ranking::leg)
            .filter(|leg| leg.scoped.is_some())
        else {
            return Ok(None);
        };
        let filter = selection
            .filter
            .as_ref()
            .filter(|_| selection.unit == scope.unit)
            .and_then(|filter| without_terms(filter, &leg.spans));
        if filter.is_none() && recent_order.is_none() {
            return Ok(None);
        }
        let mut conditions = defaults.to_vec();
        if let Some(filter) = filter {
            conditions.push(self.lower(&filter, scope, Cx::plain(measured))?);
        }
        Ok(Some(SearchScope {
            from: self.from(scope, ranking),
            condition: and_all(&conditions),
            key: scope.a.expand(units::spec(scope.unit).key),
            recent: recent_order.map(|order| hidden_order(order, "_o").0),
        }))
    }

    /// The statements that can read a ranked text search besides the whole
    /// ranking: `keys` renders one keys statement over a CTE list. Inputs
    /// a plan reads at execution bind at the parameter after this query's.
    pub(super) fn search_plans(
        &self,
        scope: &RowContext,
        leg: &ranking::SearchLeg,
        search_scope: Option<&SearchScope>,
        keys: impl Fn(&str, Option<&str>) -> String,
    ) -> TextSearch {
        let input = format!("?{}", self.params.len() + 1);
        let mut indexed = self.ctes.clone();
        indexed[leg.ctes.start] = leg.indexed(Some(&input));
        TextSearch {
            index: leg.index,
            expression: leg.expression.clone(),
            by_index: keys(&with(&indexed), None),
            scope: search_scope.map(|search_scope| {
                let scoped = |rows| self.scoped_ctes(scope, leg, search_scope, rows);
                ScopedSearch {
                    blocks: format!(
                        "{}SELECT block FROM {}_scope_blocks",
                        with(&scoped(ScopeRows::Probe)),
                        leg.name
                    ),
                    by_scope: keys(&with(&scoped(ScopeRows::All)), None),
                    recent: search_scope.recent.as_ref().map(|hidden| RecentSearch {
                        candidates: format!(
                            "{}SELECT {} AS _pk,{} FROM {} WHERE {}",
                            with(&self.ctes),
                            search_scope.key,
                            hidden.join(", "),
                            search_scope.from,
                            search_scope.condition
                        ),
                        batch: keys(
                            &with(&scoped(ScopeRows::Batch(&input))),
                            Some(&format!(
                                "{} IN (SELECT value FROM json_each({input}))",
                                search_scope.key
                            )),
                        ),
                    }),
                }
            }),
        }
    }

    /// The CTE list with the search reading the scope's blocks: the ranked
    /// texts are only those the scope's blocks hold.
    fn scoped_ctes(
        &self,
        scope: &RowContext,
        leg: &ranking::SearchLeg,
        search_scope: &SearchScope,
        rows: ScopeRows<'_>,
    ) -> Vec<String> {
        let SearchScope {
            from,
            condition,
            key,
            ..
        } = search_scope;
        let name = &leg.name;
        let scope_rows = format!("{name}_scope_rows");
        let blocks = format!("{name}_scope_blocks");
        let candidates = leg.candidates();
        let hits = units::Hits {
            cte: name,
            index: match leg.index {
                polylogue_query::search::SnippetIndex::Prose => Index::Prose,
                polylogue_query::search::SnippetIndex::Output => Index::Output,
            },
            source: false,
            scope: None,
        };
        let kind = hits.block_kind("b");
        let writer = hits.writer("m");
        let scope_row_sql = match rows {
            ScopeRows::Batch(batch) => format!(
                "SELECT {key} AS pk FROM json_each({batch}) chosen CROSS JOIN {from} \
                 WHERE {key}=chosen.value AND {condition}"
            ),
            ScopeRows::Probe | ScopeRows::All => {
                format!("SELECT {key} AS pk FROM {from} WHERE {condition}")
            }
        };
        let eligible = match scope.unit {
            Unit::Sessions => {
                // The probe adds scope rows one at a time: seeding every row
                // would read the full scope before its first block.
                let seeds = if rows == ScopeRows::Probe {
                    format!(
                        "SELECT min(pk) FROM {scope_rows} UNION \
                         SELECT c.pk FROM tree t CROSS JOIN session p ON p.pk=t.pk \
                         CROSS JOIN session_link l ON l.parent_id=p.id AND l.kind IN ({STRUCTURAL_KINDS}) \
                         CROSS JOIN session c ON c.id=l.child_id UNION \
                         SELECT (SELECT min(pk) FROM {scope_rows} WHERE pk>t.pk) \
                         FROM tree t WHERE t.pk IS NOT NULL"
                    )
                } else {
                    format!(
                        "SELECT pk FROM {scope_rows} UNION \
                         SELECT c.pk FROM tree t CROSS JOIN session p ON p.pk=t.pk \
                         CROSS JOIN session_link l ON l.parent_id=p.id AND l.kind IN ({STRUCTURAL_KINDS}) \
                         CROSS JOIN session c ON c.id=l.child_id"
                    )
                };
                format!(
                    "WITH RECURSIVE tree(pk) AS ({seeds}) \
                     SELECT b.pk AS block,b.text_id FROM tree t \
                     CROSS JOIN message m ON m.session_pk=t.pk \
                     CROSS JOIN block b ON b.message_pk=m.pk \
                     WHERE b.text_id IS NOT NULL AND {kind}{writer} AND {}",
                    fields::own_messages("m", None)
                )
            }
            Unit::Messages => format!(
                "SELECT b.pk AS block,b.text_id FROM {scope_rows} o \
                 CROSS JOIN message m ON m.pk=o.pk \
                 CROSS JOIN block b ON b.message_pk=m.pk \
                 WHERE b.text_id IS NOT NULL AND {kind}{writer}"
            ),
            _ => unreachable!("scope-driven source units"),
        };
        let mut ctes = self.ctes[..leg.ctes.start].to_vec();
        ctes.push(format!(
            "{name} AS MATERIALIZED (WITH scoped_texts AS MATERIALIZED (\
               SELECT DISTINCT text_id FROM {candidates}) \
             SELECT text_id,rank FROM ({}))",
            crate::search::ranked_texts(leg.index, &leg.parameter, Some("scoped_texts"), None)
        ));
        ctes.extend(leg.scoped.iter().flatten().cloned());
        ctes.extend(self.ctes[leg.ctes.end..].iter().cloned());
        ctes.push(format!(
            "{scope_rows} AS NOT MATERIALIZED ({scope_row_sql})"
        ));
        ctes.push(format!("{blocks} AS NOT MATERIALIZED ({eligible})"));
        // Execution stores the blocks once and shares them with the text
        // probes and row selection.
        ctes.push(format!(
            "{candidates} AS MATERIALIZED (SELECT * FROM {blocks})"
        ));
        ctes
    }
}

/// A CTE list as the WITH clause that starts a statement.
pub(super) fn with(ctes: &[String]) -> String {
    if ctes.is_empty() {
        String::new()
    } else {
        format!("WITH {} ", ctes.join(", "))
    }
}
