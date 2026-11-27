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
    /// Another query in parentheses, bound on its own and compiled over
    /// this query's parameters. A nested query's bound unit, notes, terms
    /// and checks come from its tree. `similar` is the `similar:` text the
    /// enclosing selection ranks by: a query ranks by one.
    pub(super) fn other_query(
        &mut self,
        bound: &Bound,
        at: usize,
        similar: Option<&str>,
    ) -> Result<Program, QueryError> {
        let mut program = self.lower_other(bound)?;
        self.take_in(&mut program, at, similar)?;
        Ok(program)
    }

    /// Another query's program, its parameters continuing this query's.
    fn lower_other(&mut self, bound: &Bound) -> Result<Program, QueryError> {
        if let Some(child) = bound.query.attach.first() {
            return Err(QueryError::Invalid {
                message: "a query in parentheses gives rows to compare, not attached rows".into(),
                position: child.span.start,
            });
        }
        let mut inner = self.nested(self.text);
        inner.params = std::mem::take(&mut self.params);
        let mut program = inner.program(bound, Reading::whole())?;
        self.params = std::mem::take(&mut program.params);
        Ok(program)
    }

    /// Takes over what another query's program recorded: its resolved
    /// times, notes, terms, checks and definitions, and its `similar:` text,
    /// which must be the enclosing selection's own.
    fn take_in(
        &mut self,
        program: &mut Program,
        at: usize,
        similar: Option<&str>,
    ) -> Result<(), QueryError> {
        self.resolved_times
            .borrow_mut()
            .extend(program.resolved_times.iter().cloned());
        // A condition is compiled again for `--why`: what it adds is kept once.
        join(&mut self.notes, &mut program.notes);
        join(&mut self.incomplete, &mut program.incomplete);
        join(&mut self.snippet_terms, &mut program.snippet_terms);
        join(&mut self.id_checks, &mut program.id_checks);
        join(
            &mut self.annotation_cutoffs,
            &mut program.annotation_cutoffs,
        );
        join(
            &mut self.change_generations,
            &mut program.change_generations,
        );
        join(&mut self.definitions, &mut program.definitions);
        match (
            similar.or(self.similar_text.as_ref().map(|near| near.text.as_str())),
            &program.near,
        ) {
            (Some(mine), Some(theirs)) if mine != theirs.text => {
                return Err(QueryError::Invalid {
                    message: "one query ranks by one similar: text; this one has two".into(),
                    position: at,
                });
            }
            (None, Some(theirs)) => self.similar_text = Some(theirs.clone()),
            _ => {}
        }
        Ok(())
    }

    /// The other query of a combining stage: rows of the same unit.
    pub(super) fn combined_rows(
        &mut self,
        query: &Query,
        bound: &Result<Box<Bound>, QueryError>,
        unit: Unit,
        at: usize,
    ) -> Result<Program, QueryError> {
        if let Some(stage) = query.transforms.iter().find(|s| {
            matches!(
                s.kind,
                TransformKind::Matches
            ) || matches!(&s.kind, TransformKind::Extension { name, .. } if name == polylogue_query::catalog::MEASURE
            )
        }) {
            return Err(QueryError::Invalid {
                message: "a combined query gives rows; matches and measure do not".into(),
                position: stage.span.start,
            });
        }
        let program = self.other_query(bound.as_ref().map_err(Clone::clone)?, at, None)?;
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
        if !matches!(program.row_kind, RowKind::UnitRows { .. })
            || !relations::Relation::new(program.clone())
                .sources
                .iter()
                .any(|source| source.prefix.is_empty())
        {
            return Err(not_built("combining rows without source coordinates"));
        }
        Ok(program)
    }

    /// `FIELD in (QUERY)`: the field's value is one of the values the
    /// other query groups by. An unknown value is in no set, and its `NOT`
    /// is unknown too.
    pub(super) fn member_of(
        &mut self,
        field: &str,
        query: &Bound,
        scope: &RowContext,
        span: Span,
        cx: Cx<'_>,
    ) -> Result<String, QueryError> {
        // The filter and its `--why` clause hold the same query: it is
        // lowered once, over the parameters it was given then, and its
        // statement serves both.
        let key = std::ptr::from_ref(query);
        let program = match self.lowered_queries.get(&key) {
            Some(program) => {
                let program = program.clone();
                self.take_in(&mut program.clone(), span.start, cx.similar)?;
                program
            }
            None => {
                let mut program = self.lower_other(query)?;
                self.lowered_queries.insert(key, program.clone());
                self.take_in(&mut program, span.start, cx.similar)?;
                program
            }
        };
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
        let key = column_sql(program.column_slot(&grouped.name));
        Ok(format!(
            "CASE WHEN ({expr}) IS NULL THEN NULL ELSE ({expr}) IN (SELECT {key} FROM ({}) WHERE {key} IS NOT NULL) END",
            program.rows()
        ))
    }

    /// Combine the original operand rows once. Values come from the first
    /// membership operand; every ranking keeps its captured, full input order.
    pub(super) fn fuse_pending(
        &mut self,
        pending: PendingFusion,
        relation: &mut relations::Relation,
    ) -> Result<(), QueryError> {
        if pending.is_empty() {
            return Ok(());
        }
        if !matches!(relation.program.row_kind, RowKind::UnitRows { .. })
            || !relation
                .sources
                .iter()
                .any(|source| source.prefix.is_empty())
        {
            return Err(not_built("combining rows after a change of row grain"));
        }
        let ranked = !pending.fuses.is_empty()
            || relation.program.ranked
            || pending.unions.iter().any(|program| program.ranked);
        let mut left = relation.program.clone();
        // The active relation owns coordinates added by matching row stages; its
        // Program snapshot is otherwise finalized only after all stages.
        left.row_sources = Some(relation.sources.clone());
        let mut members = vec![left];
        members.extend(pending.unions);
        let mut fuses = pending.fuses;
        relation.program.session_rows = if members
            .iter()
            .all(|program| program.session_rows == SessionRows::Logical)
        {
            SessionRows::Logical
        } else if members
            .iter()
            .all(|program| program.session_rows == SessionRows::Physical)
        {
            SessionRows::Physical
        } else {
            SessionRows::PerRow(self.slot())
        };
        for program in members.iter_mut().chain(&mut fuses) {
            let name = self.slot();
            self.ctes
                .push(format!("{name} AS MATERIALIZED ({})", program.rows()));
            program.derived_mut().rows = format!("SELECT * FROM {name}");
        }
        let source_list = self.slot();
        let mut union_branches = Vec::new();
        let mut slots = Vec::new();
        for (index, program) in members.iter().enumerate() {
            let columns = aligned_columns(self, relation, program, ranked)?;
            slots = columns.keys().cloned().collect();
            union_branches.push(format!(
                "SELECT {}, {index} AS {source_list} FROM ({}) r",
                columns
                    .iter()
                    .map(|(slot, expression)| format!("{expression} AS {}", column_sql(slot)))
                    .collect::<Vec<_>>()
                    .join(", "),
                program.rows()
            ));
        }
        let values = self.slot();
        let precedence = self.slot();
        self.ctes.push(format!(
            "{values} AS MATERIALIZED (SELECT *, row_number() OVER (PARTITION BY _pk ORDER BY {source_list}) AS {precedence} FROM ({}))",
            union_branches.join(" UNION ALL ")
        ));
        // Counting needs selected identities, not value precedence or fusion
        // scores. Fall back to the captured operand only when its stages made
        // membership depend on ranking or another relational selection.
        let mut membership_inputs = Vec::new();
        for program in &members {
            let name = self.slot();
            let input = program.statements.members(program.counts_near);
            let near = if relation.program.counts_near && program.counts_near {
                "_near"
            } else {
                "NULL"
            };
            self.ctes.push(format!(
                "{name} AS MATERIALIZED (SELECT _pk,{near} AS _near FROM ({input}))"
            ));
            membership_inputs.push(name);
        }
        let keys = membership_inputs
            .iter()
            .map(|name| format!("SELECT _pk FROM {name}"))
            .collect::<Vec<_>>()
            .join(" UNION ALL ");
        let near = if relation.program.counts_near {
            // Choose the first supplying membership operand even when its
            // similarity is NULL. RHS rankings cannot fill an earlier unknown.
            format!(
                "CASE {} END",
                membership_inputs.iter().map(|name| format!(
                    "WHEN EXISTS (SELECT 1 FROM {name} WHERE _pk=k._pk) THEN (SELECT _near FROM {name} WHERE _pk=k._pk)"
                )).collect::<Vec<_>>().join(" ")
            )
        } else {
            "1".into()
        };
        // One unordered identity set avoids UNION's merge-sorted inputs;
        // similarity matching rows never participate in duplicate elimination.
        let membership =
            format!("SELECT k._pk,{near} AS _near FROM (SELECT DISTINCT _pk FROM ({keys})) k");
        let count = format!(
            "SELECT count(*),{} FROM ({membership})",
            if relation.program.counts_near {
                "count(*) FILTER (WHERE _near IS NULL)"
            } else {
                "0"
            }
        );
        let mut select_list = slots
            .iter()
            .map(|slot| format!("d.{0} AS {0}", column_sql(slot)))
            .collect::<Vec<_>>();
        let mut join_rank = String::new();
        let mut order = relation.identity_order();
        if ranked {
            let rows = self.slot();
            self.ctes.push(format!(
                "{rows} AS MATERIALIZED ({})",
                members
                    .iter()
                    .enumerate()
                    .map(|(index, program)| other_list(program, None, index + 1))
                    .collect::<Vec<_>>()
                    .join(" UNION ALL ")
            ));
            let found = format!("SELECT k FROM {rows}");
            let mut lists = vec![format!("SELECT * FROM {rows}")];
            lists.extend(
                fuses
                    .iter()
                    .map(|program| other_list(program, Some(&found), 0)),
            );
            let found_in = (1..=members.len())
                .map(|index| format!("max(list={index} AND counts) AS in{index}"))
                .collect::<Vec<_>>();
            let scores = self.slot();
            self.ctes.push(format!(
                "{scores} AS MATERIALIZED (SELECT k,CASE WHEN max(counts) THEN -sum(CASE WHEN counts THEN 1.0/({RANK_FUSION_OFFSET}+place) ELSE 0.0 END ORDER BY place) END AS r,max(recency) AS recency,{} FROM ({}) GROUP BY k)",
                found_in.join(", "), lists.join(" UNION ALL ")
            ));
            let rank = self.slot();
            let missing = self.slot();
            let contributors = self.slot();
            let recency = self.slot();
            let undated = self.slot();
            select_list.push("f.r AS _rank".into());
            select_list.push(format!("f.r AS {}", column_sql(&rank)));
            select_list.push(format!("f.r IS NULL AS {}", column_sql(&missing)));
            select_list.push(format!("f.recency AS {}", column_sql(&recency)));
            select_list.push(format!("f.recency IS NULL AS {}", column_sql(&undated)));
            select_list.push(format!(
                "(SELECT json_group_array(value) FROM json_each(json_array({})) WHERE value>0) AS {}",
                (1..=members.len()).map(|index| format!("f.in{index}*{index}")).collect::<Vec<_>>().join(", "),
                column_sql(&contributors)
            ));
            join_rank = format!(" JOIN {scores} f ON f.k=d._pk");
            if let Some(index) = slots.iter().position(|slot| slot == "_recency") {
                select_list[index] = "f.recency AS _recency".into();
            }
            relation
                .program
                .columns
                .retain(|column| !matches!(column.name.as_str(), "rank" | "found_in"));
            relation
                .program
                .column_slots
                .retain(|name, _| !matches!(name.as_str(), "rank" | "found_in"));
            for (name, slot, kind) in [
                ("rank", rank.clone(), ColKind::Number),
                ("found_in", contributors, ColKind::Json),
            ] {
                relation.program.columns.push(Column {
                    name: name.into(),
                    kind,
                    reference: None,
                });
                relation.program.column_slots.insert(name.into(), slot);
            }
            order.splice(
                0..0,
                [
                    OrderKey::ascending(&missing),
                    OrderKey::ascending(&rank),
                    OrderKey::ascending(&undated),
                    OrderKey {
                        column: recency,
                        descending: true,
                        ..Default::default()
                    },
                ],
            );
        }
        self.set_rows(
            relation,
            format!(
                "SELECT {} FROM {values} d{join_rank} WHERE d.{precedence}=1 ORDER BY {}",
                select_list.join(", "),
                order_by(&order)
            ),
        );
        relation.program.order = order;
        relation.program.ranked = ranked;
        relation.program.derived_mut().count = Some(count);
        relation.program.derived_mut().members = Some(membership);
        relation.program.time_text_columns.clear();
        Ok(())
    }
}

/// Moves the items of `more` not already in `list` to its end.
pub(super) fn join<T: PartialEq>(list: &mut Vec<T>, more: &mut Vec<T>) {
    for item in more.drain(..) {
        if !list.contains(&item) {
            list.push(item);
        }
    }
}

/// The ranked list of another query's rows: each row's place in its
/// order, counted when the query ranks it. `within` restricts emitted
/// members after their full operand ranks have been captured. `list` names
/// it in `found_in` (0: none).
fn other_list(other: &Program, within: Option<&str>, list: usize) -> String {
    let order = if other.order.is_empty() {
        "\"_pk\"".to_owned()
    } else {
        order_by(&other.order)
    };
    let counts: String = if other.ranked {
        "_rank IS NOT NULL".into()
    } else {
        "1".into()
    };
    let ranked = format!(
        "SELECT _pk AS k,row_number() OVER (ORDER BY {order}) AS place,{counts} AS counts,{list} AS list,_recency AS recency FROM ({})",
        other.rows()
    );
    match within {
        Some(rows) => format!("SELECT * FROM ({ranked}) WHERE k IN ({rows})"),
        None => ranked,
    }
}

/// Align each operand's own values and private row coordinates, never aliases
/// fetched afresh from the current unit tables.
fn aligned_columns(
    compiler: &mut Compiler<'_>,
    relation: &relations::Relation,
    input: &Program,
    ranked: bool,
) -> Result<std::collections::BTreeMap<String, String>, QueryError> {
    let mut columns = std::collections::BTreeMap::from([("_pk".into(), "r._pk".into())]);
    if let SessionRows::PerRow(column) = &relation.program.session_rows {
        columns.insert(column.clone(), input.session_rows.sql("r"));
    }
    if let (
        RowKind::UnitRows {
            columns: output, ..
        },
        RowKind::UnitRows {
            columns: source, ..
        },
    ) = (&relation.program.row_kind, &input.row_kind)
    {
        if output.len() != source.len() {
            return Err(not_built("combining different row identities"));
        }
        for (output, source) in output.iter().zip(source) {
            columns.insert(output.clone(), format!("r.{}", column_sql(source)));
        }
    }
    let input_relation = relations::Relation::new(input.clone());
    for source in &relation.sources {
        let input_source = input_relation
            .sources
            .iter()
            .find(|candidate| candidate.prefix == source.prefix && candidate.unit == source.unit);
        let value = match input_source {
            Some(input_source) => format!("r.{}", column_sql(&input_source.key)),
            // An operand without this optional matching row has no matching row source.
            None if source.prefix.starts_with("match.") => "NULL".into(),
            None => return Err(not_built("combining different source coordinates")),
        };
        columns.insert(source.key.clone(), value);
    }
    columns.insert(
        "_near".into(),
        if relation.program.counts_near && !input.counts_near {
            "NULL".into()
        } else {
            "r._near".into()
        },
    );
    columns.insert("_recency".into(), "r._recency".into());
    for column in &relation.program.columns {
        if ranked && matches!(column.name.as_str(), "rank" | "found_in") {
            continue;
        }
        let input_column = input
            .columns
            .iter()
            .find(|candidate| candidate.name == column.name);
        let Some(input_column) = input_column else {
            if relation.open_fields
                && input_relation.open_fields
                && column.name.starts_with("match.")
            {
                // The shared scope permits this matching row namespace on unit
                // rows. RHS-only members without a matching row keep unknowns.
                columns.insert(
                    relation.program.column_slot(&column.name).to_owned(),
                    "NULL".into(),
                );
                continue;
            }
            return Err(QueryError::Invalid {
                message: format!("combined output has no column {:?}", column.name),
                position: 0,
            });
        };
        let value =
            input_relation.display_column_on(compiler, input_column, column.kind, 0, "r")?;
        columns.insert(relation.program.column_slot(&column.name).to_owned(), value);
    }
    Ok(columns)
}
