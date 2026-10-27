//! Queries inside a query: combining stages (`| union (…)`, `| intersect
//! (…)`, `| except (…)`, `| fuse (…)`) and `FIELD in (…)`. The other query
//! is compiled on its own, over this query's parameters, and read as a
//! subquery.

use super::*;

/// Consecutive `| union` and `| fuse` stages, fused together once: every
/// list adds `1 / (offset + place)` to a row's score.
#[derive(Default)]
pub(super) struct PendingFusion {
    /// The rows of each `| union` query.
    pub(super) unions: Vec<Program>,
    /// The rankings of each `| fuse` query.
    pub(super) fuses: Vec<Program>,
}

impl PendingFusion {
    pub(super) fn is_empty(&self) -> bool {
        self.unions.is_empty() && self.fuses.is_empty()
    }
}

impl<'a> Compiler<'a> {
    /// Another query in parentheses, compiled over this query's parameters.
    /// Conditions without a unit are on `unit`. Its notes, terms and checks
    /// join this query's.
    pub(super) fn other_query(
        &mut self,
        query: &Query,
        unit: Unit,
        at: usize,
    ) -> Result<Program, QueryError> {
        let mut query = query.clone();
        if let Source::Sessions { cond } = &query.source {
            query.source = Source::Unit {
                unit,
                cond: Some(cond.clone()),
            };
        }
        if let Some(child) = query.attach.first() {
            return Err(QueryError::Invalid {
                message: "a query in parentheses gives rows to compare, not attached rows".into(),
                position: child.span.start,
            });
        }
        let mut inner = Compiler::new(self.text, self.opts);
        inner.params = std::mem::take(&mut self.params);
        inner.saved_stack = self.saved_stack.clone();
        let mut program = inner.program(&query)?;
        self.params = std::mem::take(&mut program.params);
        // A condition is compiled again for `--why`: what it adds is kept once.
        join(&mut self.notes, &mut program.notes);
        join(&mut self.incomplete, &mut program.incomplete);
        join(&mut self.text_terms, &mut program.text_terms);
        join(&mut self.id_checks, &mut program.id_checks);
        join(
            &mut self.change_generations,
            &mut program.change_generations,
        );
        join(&mut self.definitions, &mut program.definitions);
        match (
            self.near_request
                .as_ref()
                .map(|(near, _)| near)
                .or(self.near.as_ref()),
            &program.near,
        ) {
            (Some(mine), Some(theirs)) if mine != theirs => {
                return Err(QueryError::Invalid {
                    message: "one query ranks by one near: text; this one has two".into(),
                    position: at,
                });
            }
            (None, Some(theirs)) => self.near = Some(theirs.clone()),
            _ => {}
        }
        Ok(program)
    }

    /// The other query of a combining stage: rows of the same unit.
    pub(super) fn combined_rows(
        &mut self,
        query: &Query,
        unit: Unit,
        at: usize,
    ) -> Result<Program, QueryError> {
        if let Some(stage) = query.stages.iter().find(|s| {
            matches!(
                s.kind,
                StageKind::Distinct | StageKind::Measure { .. } | StageKind::Matches
            )
        }) {
            return Err(QueryError::Invalid {
                message: "a combined query gives rows; distinct, matches and measure do not".into(),
                position: stage.span.start,
            });
        }
        let program = self.other_query(query, unit, at)?;
        if program.shape != Shape::Rows {
            return Err(QueryError::Invalid {
                message: "a combined query gives rows, not groups or a summary".into(),
                position: at,
            });
        }
        if program.unit != unit {
            return Err(QueryError::Invalid {
                message: format!(
                    "rows of {} combine only with rows of {}; this query gives {}",
                    unit.name(),
                    unit.name(),
                    program.unit.name()
                ),
                position: at,
            });
        }
        Ok(program)
    }

    /// `FIELD in (QUERY)`: the field's value is one of the values the
    /// other query groups by. An unknown value is in no set, and its `NOT`
    /// is unknown too.
    pub(super) fn member_of(
        &mut self,
        field: &str,
        query: &Query,
        scope: &RowContext,
        span: Span,
    ) -> Result<String, QueryError> {
        let grouped_by_one = query.stages.iter().rev().find_map(|s| match &s.kind {
            StageKind::GroupBy { keys } => Some(keys.len() == 1),
            _ => None,
        });
        if grouped_by_one != Some(true) {
            return Err(QueryError::Invalid {
                message: format!(
                    "`{field} in (…)` compares {field} with the values a query groups by: \
                     end it with `| group by FIELD` (one field)"
                ),
                position: span.start,
            });
        }
        let program = self.other_query(query, Unit::Sessions, span.start)?;
        let (expr, kind) = self.expr(field, scope, span.start)?;
        scalar(kind, field, span.start)?;
        let grouped = &program.columns[0];
        if grouped.kind != kind {
            return Err(QueryError::Invalid {
                message: format!(
                    "`{field} in (…)` compares fields of the same type; {} has a different type",
                    grouped.name
                ),
                position: span.start,
            });
        }
        let expr = match kind {
            ColKind::Money => format!("({expr}) / 1e9"),
            _ => expr,
        };
        let key = &program.columns[0].name;
        Ok(format!(
            "CASE WHEN ({expr}) IS NULL THEN NULL ELSE ({expr}) IN (SELECT \"{key}\" FROM ({}) WHERE \"{key}\" IS NOT NULL) END",
            program.sql
        ))
    }

    /// `| intersect (…)` and `| except (…)`: membership in the other
    /// query's rows; this query's order stays.
    pub(super) fn membership(
        &mut self,
        how: Combination,
        query: &Query,
        scope: &RowContext,
        where_sql: &mut Vec<String>,
        at: usize,
    ) -> Result<(), QueryError> {
        let other = self.combined_rows(query, scope.unit, at)?;
        let (pk, _) = row_columns(scope.unit, &scope.a, self.opts.clock.now);
        let not = if how == Combination::Except {
            "NOT "
        } else {
            ""
        };
        where_sql.push(format!("{pk} {not}IN (SELECT _pk FROM ({}))", other.sql));
        Ok(())
    }

    /// Ends a run of `| union` and `| fuse` stages. Unions of unranked rows
    /// are plain membership in this query's order; otherwise every list
    /// (this query's rows, each union's and each fuse's, the last over
    /// the rows some list ranks) ranks the rows by reciprocal rank fusion,
    /// and `found_in` names the lists that rank each row.
    pub(super) fn fuse_pending(
        &mut self,
        pending: PendingFusion,
        scope: &RowContext,
        where_sql: &mut Vec<String>,
        order: &mut Option<Vec<String>>,
    ) -> Result<(), QueryError> {
        if pending.is_empty() {
            return Ok(());
        }
        self.setup_near();
        let (pk, _) = row_columns(scope.unit, &scope.a, self.opts.clock.now);
        let any_ranked = self.ranking().is_some() || pending.unions.iter().any(|p| p.ranked);
        if pending.fuses.is_empty() && !any_ranked {
            let mut any = vec![format!("({})", and_all(where_sql))];
            for other in &pending.unions {
                any.push(format!("{pk} IN (SELECT _pk FROM ({}))", other.sql));
            }
            *where_sql = vec![format!("({})", any.join(" OR "))];
            return Ok(());
        }
        self.fuse_text_and_near(scope, where_sql)?;
        let from = self.from(scope)?;
        let here = and_all(where_sql);
        let own_order = order
            .take()
            .unwrap_or_else(|| self.default_order(scope))
            .join(", ");
        let own_counts = match self.ranking() {
            Some(rank) => format!("({rank}) IS NOT NULL"),
            None => "1".into(),
        };
        let near = self
            .near_rank
            .clone()
            .or_else(|| self.fusion.as_ref().and_then(|f| f.near.clone()));
        let has_near = near.is_some()
            || pending
                .unions
                .iter()
                .chain(&pending.fuses)
                .any(|p| p.near.is_some());
        let near = near.as_deref().unwrap_or("NULL");
        // Each list is `SELECT k, place, counts, list, n`: the row, its place
        // in the list's order, whether that place is a rank (a ranked list
        // holds rows without a rank, such as messages with no vector, after
        // its ranked ones) and which list it is (0 for a fuse list).
        let mut lists = vec![format!(
            "SELECT {pk} AS k, row_number() OVER (ORDER BY {own_order}) AS place, {own_counts} AS counts, 1 AS list, {near} AS n FROM {from} WHERE {here}"
        )];
        for (i, other) in pending.unions.iter().enumerate() {
            lists.push(other_list(other, None, i + 2));
        }
        let rows = format!("c{}", self.ctes.len() + 1);
        self.ctes.push(format!(
            "{rows} AS MATERIALIZED ({})",
            lists.join(" UNION ALL ")
        ));
        // A fuse list ranks only rows some list ranks: it orders what was
        // found, it finds nothing.
        let found = format!("SELECT k FROM {rows} WHERE counts");
        let mut fuses = vec![format!("SELECT * FROM {rows}")];
        for other in &pending.fuses {
            fuses.push(other_list(other, Some(&found), 0));
        }
        let found_in: Vec<String> = (1..=pending.unions.len() + 1)
            .map(|n| format!("max(list = {n} AND counts) AS in{n}"))
            .collect();
        let fused = format!("c{}", self.ctes.len() + 1);
        self.ctes.push(format!(
            "{fused} AS MATERIALIZED (SELECT k, \
             -sum(CASE WHEN counts THEN 1.0 / ({RANK_FUSION_OFFSET} + place) ELSE 0.0 END) AS r, max(n) AS n, {} \
             FROM ({}) GROUP BY k)",
            found_in.join(", "),
            fuses.join(" UNION ALL ")
        ));
        let alias = format!("{fused}_rank");
        self.fusion = Some(Fusion {
            join: format!("JOIN {fused} {alias} ON {alias}.k = {pk}"),
            rank: format!("{alias}.r"),
            near: has_near.then(|| format!("{alias}.n")),
        });
        let numbered: Vec<String> = (1..=pending.unions.len() + 1)
            .map(|n| format!("{alias}.in{n} * {n}"))
            .collect();
        self.shown.retain(|(name, ..)| name != "found_in");
        self.shown.push((
            "found_in".into(),
            format!(
                "(SELECT json_group_array(value) FROM json_each(json_array({})) WHERE value > 0)",
                numbered.join(", ")
            ),
            ColKind::Json,
            None,
        ));
        *where_sql = vec![format!("{pk} IN (SELECT k FROM {fused})")];
        Ok(())
    }
}

/// Moves the items of `more` not already in `list` to its end.
fn join<T: PartialEq>(list: &mut Vec<T>, more: &mut Vec<T>) {
    for item in more.drain(..) {
        if !list.contains(&item) {
            list.push(item);
        }
    }
}

/// The ranked list of another query's rows: each row's place in its
/// order, counted when the query ranks it. With `within`, only the rows
/// that query selects take places. `list` names it in `found_in` (0:
/// none).
fn other_list(other: &Program, within: Option<&str>, list: usize) -> String {
    let order = if other.order.is_empty() {
        "\"_pk\"".to_owned()
    } else {
        order_by(&other.order)
    };
    let counts = if other.ranked {
        "rank IS NOT NULL"
    } else {
        "1"
    };
    let near = if other.near.is_some() {
        "_near"
    } else {
        "NULL"
    };
    let within = within
        .map(|rows| format!(" WHERE _pk IN ({rows})"))
        .unwrap_or_default();
    format!(
        "SELECT _pk AS k, row_number() OVER (ORDER BY {order}) AS place, {counts} AS counts, {list} AS list, {near} AS n FROM ({}){within}",
        other.sql
    )
}
