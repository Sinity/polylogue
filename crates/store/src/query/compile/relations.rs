//! Relational stage boundaries. Logical output names are independent of SQL slots.
use super::*;
use polylogue_query::stages::{Stage, StageKind};

mod order;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NativeSource {
    pub prefix: String,
    pub unit: Unit,
    pub key: String,
}

#[derive(Clone)]
pub(super) struct Relation {
    pub program: Program,
    pub sources: Vec<NativeSource>,
    pub open_fields: bool,
    /// The step holding the rows, once a stage has read them.
    step: Option<String>,
    patterns: Vec<Condition>,
}

struct ConditionScope<'a> {
    fields: Option<&'a query_language::scope::Scope>,
    path: query_language::scope::NodePath,
}

impl Relation {
    pub fn new(program: Program) -> Self {
        let sources = program.native_sources.clone().unwrap_or_else(|| {
            if matches!(program.row_basis, RowBasis::Native { .. }) {
                vec![NativeSource {
                    prefix: String::new(),
                    unit: program.unit,
                    key: "_pk".into(),
                }]
            } else {
                vec![]
            }
        });
        let open_fields = !sources.is_empty();
        Self {
            program,
            sources,
            open_fields,
            step: None,
            patterns: vec![],
        }
    }
    pub fn field(
        &self,
        compiler: &mut Compiler<'_>,
        name: &str,
        at: usize,
    ) -> Result<(String, ColKind), QueryError> {
        self.field_on(compiler, name, at, "r")
    }
    pub fn field_on(
        &self,
        compiler: &mut Compiler<'_>,
        name: &str,
        at: usize,
        alias: &str,
    ) -> Result<(String, ColKind), QueryError> {
        if let Some(column) = self.program.columns.iter().find(|c| c.name == name) {
            let sql = format!("{alias}.{}", column_sql(self.program.column_slot(name)));
            return Ok((
                match column.kind {
                    ColKind::DurationSeconds => format!("({sql}) * 1000.0"),
                    ColKind::Time if self.program.time_text_columns.contains(name) => {
                        format!("pl_kept_time({sql})")
                    }
                    _ => sql,
                },
                column.kind,
            ));
        }
        {
            for source in self
                .sources
                .iter()
                .rev()
                .filter(|source| self.open_fields || source.prefix.starts_with("match."))
            {
                if let Some(field) = name.strip_prefix(&source.prefix) {
                    let context = RowContext {
                        unit: source.unit,
                        a: compiler.fresh(),
                    };
                    let (value, kind) = compiler.expr(field, &context, at)?;
                    let key = context.a.expand(units::spec(source.unit).key);
                    return Ok((
                        format!(
                            "(SELECT {value} FROM {} WHERE {key} = {alias}.{})",
                            base_from(source.unit, &context.a),
                            column_sql(&source.key)
                        ),
                        kind,
                    ));
                }
            }
        }
        Err(QueryError::Invalid {
            message: format!("unknown relation column {name:?}"),
            position: at,
        })
    }
    pub(super) fn display_column_on(
        &self,
        compiler: &mut Compiler<'_>,
        column: &Column,
        output_kind: ColKind,
        at: usize,
        alias: &str,
    ) -> Result<String, QueryError> {
        if column.kind == output_kind && !self.program.time_text_columns.contains(&column.name) {
            return Ok(format!(
                "{alias}.{}",
                column_sql(self.program.column_slot(&column.name))
            ));
        }
        // The bound declarations already agree. Convert the captured physical
        // representation through the existing expression and display units.
        let (value, _) = self.field_on(compiler, &column.name, at, alias)?;
        Ok(Compiler::display_value(&value, output_kind))
    }

    pub fn reference(&self, value: &Expression) -> Option<ColumnReference> {
        match value {
            Expression::Field { name } => {
                if let Some(column) = self
                    .program
                    .columns
                    .iter()
                    .find(|column| &column.name == name)
                {
                    return column.reference;
                }
                self.sources
                    .iter()
                    .rev()
                    .filter(|source| self.open_fields || source.prefix.starts_with("match."))
                    .find_map(|source| {
                        name.strip_prefix(&source.prefix).and_then(|name| {
                            Compiler::field_reference(name, &RowContext::plain(source.unit))
                        })
                    })
            }
            Expression::Function { name, arguments } if name == "case_eq" => {
                arguments.get(2).and_then(|arg| self.reference(arg))
            }
            _ => None,
        }
    }
}

struct RelationOutput {
    columns: Vec<Column>,
    slots: std::collections::BTreeMap<String, String>,
    sources: Vec<NativeSource>,
    time_text_columns: std::collections::BTreeSet<String>,
}

impl RelationOutput {
    fn capture(relation: &Relation) -> Self {
        Self {
            columns: relation.program.columns.clone(),
            slots: relation.program.column_slots.clone(),
            sources: relation.sources.clone(),
            time_text_columns: relation.program.time_text_columns.clone(),
        }
    }
    fn install(self, relation: &mut Relation) {
        relation.program.columns = self.columns;
        relation.program.column_slots = self.slots;
        relation.sources = self.sources;
        relation.program.time_text_columns = self.time_text_columns;
    }
}

/// A group key: its column, its slot and its expression over the rows
/// entering the grouping.
type GroupKey = (Column, String, String);

pub(super) fn as_of_unavailable() -> QueryError {
    QueryError::Unavailable {
        what: "Polylogue does not retain historical query membership at earlier read positions; use evidence_until:TIME to select evidence received by then, read with current parsers and prices".into(),
    }
}

impl<'a> Compiler<'a> {
    pub(super) fn program_with(
        mut self,
        query: &Query,
        preset: Option<MatchRel>,
    ) -> Result<Program, QueryError> {
        let Some(selection) = Selection::of(query, &self.binder(), &self.opts.fields)? else {
            return Err(not_built("this shared query source"));
        };
        let Some(first) = selection.relations_from else {
            let stages = &query.transforms[selection.stages_from..];
            return self.native_program_with(&selection, stages, preset);
        };
        let source_order = match &query.source {
            Source::Unit {
                condition: Some(c), ..
            } if contains_order(c) => Some(c),
            _ => None,
        };
        let mut compiler = self.nested(self.text);
        compiler.saved_stack = self.saved_stack.clone();
        compiler.query_scope = self.query_scope.clone();
        compiler.typing = self.typing.clone();
        compiler.source_near_allowed = self.source_near_allowed;
        // The stages read the selection's rows whole.
        self.answer = false;
        let native = &query.transforms[selection.stages_from.min(first)..first];
        let mut program = self.native_program_with(&selection, native, preset)?;
        compiler.params = std::mem::take(&mut program.params);
        compiler.near = program.near.clone();
        let relation = Relation::new(program);
        let after = selection
            .stages
            .iter()
            .position(|stage| stage.transforms.start >= first)
            .unwrap_or(selection.stages.len());
        compiler.finish_relations(query, &selection.stages[after..], relation, source_order)
    }

    /// Lowers `stages` over `relation`, each to one step reading the step
    /// before it by name.
    pub(super) fn finish_relations(
        mut self,
        query: &Query,
        stages: &[Stage],
        mut relation: Relation,
        source_order: Option<&Condition>,
    ) -> Result<Program, QueryError> {
        if let Source::Unit {
            condition: Some(condition),
            ..
        } = &query.source
        {
            relation.patterns = match_patterns(condition);
        }
        if let Some(condition) = source_order {
            let scope = self.query_scope.as_ref().map(|query| query.source.clone());
            self.relation_where(
                &mut relation,
                condition,
                scope.as_ref(),
                query_language::scope::NodePath::default()
                    .field("source")
                    .field("condition"),
            )?;
        }
        for stage in stages {
            let transforms = &query.transforms[stage.transforms.clone()];
            match (stage.kind, transforms) {
                (StageKind::Fusion, _) => {
                    let mut pending = combine::PendingFusion::default();
                    for transform in transforms {
                        let TransformKind::Combine { how, query: other } = &transform.kind else {
                            unreachable!("a fusion holds unions and fuses");
                        };
                        let operand =
                            self.combined_rows(other, relation.program.unit, transform.span.start)?;
                        match how {
                            Combination::Union => pending.unions.push(operand),
                            _ => pending.fuses.push(operand),
                        }
                    }
                    self.fuse_pending(pending, &mut relation)?;
                }
                (StageKind::PerGroup, [group, then]) => {
                    let TransformKind::Group { keys } = &group.kind else {
                        unreachable!("a per-group stage starts with its grouping");
                    };
                    match &then.kind {
                        TransformKind::First { count } => self.relation_group_first(
                            &mut relation,
                            keys,
                            *count,
                            group.span.start,
                        )?,
                        TransformKind::Aggregate { aggregates } => self.relation_aggregate(
                            &mut relation,
                            Some(keys),
                            aggregates,
                            then.span.start,
                        )?,
                        _ => unreachable!("a per-group stage ends with first or agg"),
                    }
                    relation.program.derived_mut().count = None;
                    relation.program.derived_mut().members = None;
                }
                (_, [transform]) => {
                    self.relation_stage(&mut relation, transform, query, stage.transforms.start)?
                }
                _ => unreachable!("a stage of one transform"),
            }
        }
        if let Some(step) = &relation.step {
            let rows = match relation.program.order.as_slice() {
                [] => format!("SELECT * FROM {step}"),
                order => format!("SELECT * FROM {step} ORDER BY {}", order_by(order)),
            };
            relation.program.derived_mut().rows = rows;
        }
        if !self.ctes.is_empty() {
            let prefix = format!("WITH RECURSIVE {}", self.ctes.join(", "));
            let scoped = |sql: &str| format!("{prefix} SELECT * FROM ({sql})");
            let derived = relation.program.derived_mut();
            derived.rows = scoped(&derived.rows);
            derived.count = derived.count.as_deref().map(scoped);
            derived.members = derived.members.as_deref().map(scoped);
            if let Some(measure) = &mut relation.program.measure {
                measure.selection_sql = scoped(&measure.selection_sql);
                measure.population_sql = scoped(&measure.population_sql);
            }
        }
        if !relation.program.attach.is_empty() && relation.program.shape != Shape::Rows {
            return Err(QueryError::Invalid {
                message: "attached rows apply to rows, not to groups or summaries".into(),
                position: query
                    .attach
                    .first()
                    .map_or(0, |attachment| attachment.span.start),
            });
        }
        relation.program.params = self.params;
        join(&mut relation.program.notes, &mut self.notes);
        join(&mut relation.program.snippet_terms, &mut self.snippet_terms);
        join(&mut relation.program.incomplete, &mut self.incomplete);
        join(&mut relation.program.id_checks, &mut self.id_checks);
        join(&mut relation.program.definitions, &mut self.definitions);
        join(
            &mut relation.program.change_generations,
            &mut self.change_generations,
        );
        join(
            &mut relation.program.assertion_cutoffs,
            &mut self.assertion_cutoffs,
        );
        relation
            .program
            .resolved_times
            .extend(self.resolved_times.into_inner());
        relation.program.near = self.near.or(relation.program.near);
        relation.program.native_sources = Some(relation.sources.clone());
        Ok(relation.program)
    }

    /// The step holding `relation`'s rows, naming its statement as one
    /// when no stage has read it yet.
    pub(super) fn input(&mut self, relation: &mut Relation) -> String {
        if let Some(step) = &relation.step {
            return step.clone();
        }
        let rows = relation.program.rows().to_owned();
        self.set_rows(relation, rows)
    }

    /// `relation`'s rows become `sql`, a statement over earlier steps,
    /// named as the next step; returns its name. A step is inlined where
    /// it is read, as the subquery it replaces would be.
    pub(super) fn set_rows(&mut self, relation: &mut Relation, sql: String) -> String {
        let step = format!("_step{}", self.next_number());
        self.ctes
            .push(format!("{step} AS NOT MATERIALIZED ({sql})"));
        relation.program.derived_mut().rows = format!("SELECT * FROM {step}");
        relation.step = Some(step.clone());
        step
    }

    fn relation_filter(&mut self, relation: &mut Relation, predicate: &str) {
        let input = self.input(relation);
        let sql = format!(
            "SELECT * FROM {input} r WHERE ({predicate}) ORDER BY {}",
            relation_order(&relation.program.order)
        );
        self.set_rows(relation, sql);
        relation.program.derived_mut().count = None;
        relation.program.derived_mut().members = None;
    }

    fn relation_stage(
        &mut self,
        relation: &mut Relation,
        stage: &Transform,
        query: &Query,
        index: usize,
    ) -> Result<(), QueryError> {
        let at = stage.span.start;
        match &stage.kind {
            TransformKind::AsOf { .. } => return Err(as_of_unavailable()),
            TransformKind::Fields { columns } => {
                let input = self.input(relation);
                let mut select = relation.private_columns();
                let mut output = vec![];
                let mut slots = std::collections::BTreeMap::new();
                for column in columns {
                    let (sql, kind) = self.sql_expression_with(
                        &column.expression,
                        at,
                        &mut |compiler, name, at| relation.field(compiler, name, at),
                    )?;
                    let slot = self.slot();
                    select.push(format!(
                        "{} AS {}",
                        Self::display_value(&sql, kind),
                        column_sql(&slot)
                    ));
                    output.push(Column {
                        name: column.column(),
                        kind,
                        reference: relation.reference(&column.expression),
                    });
                    slots.insert(column.column(), slot);
                }
                let sql = format!(
                    "SELECT {} FROM {input} r ORDER BY {}",
                    select.join(", "),
                    relation_order(&relation.program.order)
                );
                self.set_rows(relation, sql);
                relation.program.columns = output;
                relation.program.column_slots = slots;
                relation.program.time_text_columns.clear();
                relation.open_fields = false;
            }
            TransformKind::Where { condition } => {
                let scope = self
                    .query_scope
                    .as_ref()
                    .and_then(|query| query.transforms.get(index))
                    .map(|transform| transform.before.clone());
                self.relation_where(
                    relation,
                    condition,
                    scope.as_ref(),
                    query_language::scope::NodePath::default()
                        .field("transforms")
                        .index(index)
                        .field("kind")
                        .field("condition"),
                )?;
                relation.patterns.extend(match_patterns(condition));
            }
            TransformKind::Sort { keys } => {
                let input = self.input(relation);
                let mut select = vec!["r.*".into()];
                let mut order = vec![];
                for key in keys {
                    let (sql, kind) = if matches!(&key.expression, Expression::Function { name, .. } if name == "random")
                    {
                        let sql =
                            match parse_random_seed(&key.expression)?.or(self.opts.random_seed) {
                                Some(seed) => format!(
                                    "pl_random(json_array({}), '{seed}')",
                                    relation
                                        .identity_order()
                                        .iter()
                                        .map(|key| format!("r.{}", column_sql(&key.column)))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                ),
                                None => "random()".into(),
                            };
                        (sql, ColKind::Number)
                    } else {
                        self.sql_expression_with(&key.expression, at, &mut |compiler, name, at| {
                            relation.field(compiler, name, at)
                        })?
                    };
                    scalar(kind, &key.expression.column(), at)?;
                    let null = self.slot();
                    let value = self.slot();
                    select.push(format!("({sql}) IS NULL AS {}", column_sql(&null)));
                    select.push(format!("{sql} AS {}", column_sql(&value)));
                    order.push(OrderKey::ascending(&null));
                    order.push(OrderKey {
                        column: value,
                        descending: key.descending,
                    });
                }
                // An explicit sort is tied by semantic identity, never by an earlier time order.
                order.extend(relation.identity_order());
                let sql = format!(
                    "SELECT {} FROM {input} r ORDER BY {}",
                    select.join(", "),
                    relation_order(&order)
                );
                self.set_rows(relation, sql);
                relation.program.order = order;
            }
            TransformKind::Limit { count } | TransformKind::First { count } => {
                self.relation_window(
                    relation,
                    &format!("LIMIT {}", (*count).min(i64::MAX as u64)),
                );
            }
            TransformKind::Offset { count } => {
                self.relation_window(
                    relation,
                    &format!("LIMIT -1 OFFSET {}", (*count).min(i64::MAX as u64)),
                );
            }
            TransformKind::Distinct => self.relation_distinct(relation),
            TransformKind::Group { keys } => self.relation_group(relation, keys, at)?,
            TransformKind::Aggregate { aggregates } => {
                self.relation_aggregate(relation, None, aggregates, at)?
            }
            TransformKind::Join { query, on } => {
                let inner = Relation::new(self.other_query(query, relation.program.unit, at)?);
                self.rel_join_pair(relation, inner, on, at)?;
            }
            TransformKind::Diff { query } => self.relation_diff(relation, query, at)?,
            TransformKind::Scope { unit, condition } => {
                self.relation_scope(relation, unit, condition.as_ref(), at)?
            }
            _ => self.relation_native_stage(relation, stage, query)?,
        }
        if !matches!(
            stage.kind,
            TransformKind::Fields { .. } | TransformKind::Sort { .. }
        ) {
            relation.program.derived_mut().count = None;
            relation.program.derived_mut().members = None;
        }
        Ok(())
    }

    fn relation_where(
        &mut self,
        relation: &mut Relation,
        condition: &Condition,
        scope: Option<&query_language::scope::Scope>,
        path: query_language::scope::NodePath,
    ) -> Result<(), QueryError> {
        let input = relation.clone();
        let mut exposed = None;
        let predicate = self.relation_condition(
            relation,
            &input,
            condition,
            &ConditionScope {
                fields: scope,
                path,
            },
            condition.exposed_order(),
            &mut exposed,
        )?;
        self.relation_filter(relation, &predicate);
        exposed
            .unwrap_or_else(|| RelationOutput::capture(&input))
            .install(relation);
        Ok(())
    }

    fn relation_condition(
        &mut self,
        relation: &mut Relation,
        input: &Relation,
        condition: &Condition,
        scope: &ConditionScope<'_>,
        selected: Option<&Condition>,
        exposed: &mut Option<RelationOutput>,
    ) -> Result<String, QueryError> {
        query_language::depth::deeper(|| {
            if input.open_fields
                && !contains_order(condition)
                && !materialized_condition(condition, &input.program.column_slots)
                && input
                    .sources
                    .iter()
                    .filter(|source| source.prefix.is_empty())
                    .count()
                    == 1
                && let Some(source) = input.sources.iter().find(|source| source.prefix.is_empty())
            {
                let context = RowContext {
                    unit: source.unit,
                    a: self.fresh(),
                };
                let predicate = self.cond(condition, &context, false)?;
                let key = context.a.expand(units::spec(context.unit).key);
                return Ok(format!(
                    "(SELECT ({predicate}) FROM {} WHERE {key} = r.{})",
                    base_from(context.unit, &context.a),
                    column_sql(&source.key)
                ));
            }
            Ok(match &condition.kind {
                ConditionKind::Order {
                    relation: order,
                    query,
                    within,
                    at,
                    scope,
                    pick,
                } => {
                    let inner = Relation::new(self.other_query(
                        query,
                        input.program.unit,
                        condition.span.start,
                    )?);
                    let predicate = self.rel_order_pair(
                        relation,
                        inner,
                        *order,
                        within.as_ref(),
                        *at,
                        scope.as_deref(),
                        pick,
                        condition.span.start,
                    )?;
                    if selected.is_some_and(|selected| std::ptr::eq(selected, condition)) {
                        *exposed = Some(RelationOutput::capture(relation));
                    }
                    RelationOutput::capture(input).install(relation);
                    predicate
                }
                ConditionKind::And { conditions } | ConditionKind::Or { conditions } => {
                    let op = if matches!(condition.kind, ConditionKind::And { .. }) {
                        " AND "
                    } else {
                        " OR "
                    };
                    let values = conditions
                        .iter()
                        .enumerate()
                        .map(|(index, c)| {
                            self.relation_condition(
                                relation,
                                input,
                                c,
                                &ConditionScope {
                                    fields: scope.fields,
                                    path: scope.path.field("kind").field("conditions").index(index),
                                },
                                selected,
                                exposed,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    format!("({})", values.join(op))
                }
                ConditionKind::Not { condition } => {
                    format!(
                        "NOT ({})",
                        self.relation_condition(
                            relation,
                            input,
                            condition,
                            &ConditionScope {
                                fields: scope.fields,
                                path: scope.path.field("kind").field("condition")
                            },
                            selected,
                            exposed
                        )?
                    )
                }
                ConditionKind::Compare { left, op, right } => {
                    let info = self
                        .typing
                        .as_ref()
                        .and_then(|typing| typing.operand(&scope.path.field("kind").field("left")))
                        .map(|operand| operand.result.clone());
                    let path = *op == Op::Prefix
                        && info.as_ref().is_some_and(|info| {
                            info.field.natural_match()
                                == query_language::catalog::NaturalMatch::Subtree
                        });
                    if path {
                        self.natural_comparison(
                            left,
                            right,
                            query_language::catalog::NaturalMatch::Subtree,
                            condition.span.start,
                            &mut |compiler, name, at| input.field(compiler, name, at),
                        )?
                    } else {
                        self.expression_comparison(
                            left,
                            *op,
                            right,
                            condition.span.start,
                            &mut |compiler, name, at| input.field(compiler, name, at),
                        )?
                    }
                }
                ConditionKind::Between { left, from, to } => {
                    let lo = self.expression_comparison(
                        left,
                        Op::Ge,
                        &Expression::value(from.clone()),
                        condition.span.start,
                        &mut |compiler, name, at| input.field(compiler, name, at),
                    )?;
                    let hi = self.expression_comparison(
                        left,
                        Op::Le,
                        &Expression::value(to.clone()),
                        condition.span.start,
                        &mut |compiler, name, at| input.field(compiler, name, at),
                    )?;
                    format!("({lo}) AND ({hi})")
                }
                ConditionKind::Match {
                    field,
                    value: Value::Unknown | Value::Known,
                } => {
                    let (sql, _) = input.field(self, field, condition.span.start)?;
                    format!(
                        "({sql}) IS {}NULL",
                        if matches!(
                            condition.kind,
                            ConditionKind::Match {
                                value: Value::Known,
                                ..
                            }
                        ) {
                            "NOT "
                        } else {
                            ""
                        }
                    )
                }
                ConditionKind::Match { field, value } => {
                    let info = scope.fields.and_then(|scope| scope.field(field));
                    if !matches!(value, Value::Any)
                        && info.is_some_and(|info| {
                            matches!(info.field.kind, query_language::catalog::FieldKind::List(_))
                        })
                    {
                        return Err(not_built("natural matching on list-valued derived columns"));
                    }
                    let natural = info
                        .map(|info| info.field.natural_match())
                        .unwrap_or(query_language::catalog::NaturalMatch::Equal);
                    if matches!(value, Value::Any) {
                        "1".into()
                    } else {
                        self.natural_comparison(
                            &Expression::field(field),
                            &Expression::value(value.clone()),
                            natural,
                            condition.span.start,
                            &mut |compiler, name, at| input.field(compiler, name, at),
                        )?
                    }
                }
                _ => return Err(not_built("this condition on a derived relation")),
            })
        })
    }
}

impl Compiler<'_> {
    /// Bound the selected keys before projecting rows. A LIMIT over answer
    /// expressions would decode every candidate body while finding a page.
    fn relation_window(&mut self, relation: &mut Relation, window: &str) {
        let input = self.input(relation);
        if relation.program.order.is_empty() {
            self.set_rows(relation, format!("SELECT * FROM {input} {window}"));
            return;
        }
        let identity = relation.identity_order();
        let mut keys = std::collections::BTreeSet::new();
        keys.extend(identity.iter().map(|key| key.column.clone()));
        keys.extend(relation.program.order.iter().map(|key| key.column.clone()));
        let selected = format!(
            "SELECT {} FROM {input} r ORDER BY {} {window}",
            keys.iter()
                .map(|key| column_sql(key))
                .collect::<Vec<_>>()
                .join(", "),
            relation_order(&relation.program.order)
        );
        let same = identity
            .iter()
            .map(|key| {
                let key = column_sql(&key.column);
                format!("r.{key} IS selected.{key}")
            })
            .collect::<Vec<_>>()
            .join(" AND ");
        let mut columns = relation
            .private_columns()
            .into_iter()
            .map(|column| {
                column
                    .strip_prefix("r.")
                    .expect("qualified private column")
                    .to_owned()
            })
            .collect::<std::collections::BTreeSet<_>>();
        columns.extend(
            relation
                .program
                .columns
                .iter()
                .map(|column| column_sql(relation.program.column_slot(&column.name))),
        );
        let order_columns = relation
            .program
            .order
            .iter()
            .map(|key| column_sql(&key.column))
            .collect::<std::collections::BTreeSet<_>>();
        let projection = columns
            .iter()
            .map(|column| {
                let source = if order_columns.contains(column) {
                    "selected"
                } else {
                    "r"
                };
                format!("{source}.{column} AS {column}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT {projection} FROM ({selected}) selected JOIN {input} r ON {same} ORDER BY {}",
            relation
                .program
                .order
                .iter()
                .map(|key| {
                    format!(
                        "selected.{}{}",
                        column_sql(&key.column),
                        if key.descending { " DESC" } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.set_rows(relation, sql);
    }
}

impl Relation {
    fn private_columns(&self) -> Vec<String> {
        let mut names = std::collections::BTreeSet::new();
        names.insert("_pk".to_owned());
        if self.program.counts_near {
            names.insert("_near".into());
        }
        if self.program.ranked {
            names.insert("_rank".into());
        }
        if self.sources.iter().any(|source| source.prefix.is_empty())
            && matches!(self.program.row_basis, RowBasis::Native { .. })
        {
            names.insert("_near".into());
            names.insert("_recency".into());
        }
        names.extend(self.program.order.iter().map(|k| k.column.clone()));
        names.extend(self.sources.iter().map(|s| s.key.clone()));
        match &self.program.row_basis {
            RowBasis::Native { columns, .. }
            | RowBasis::Group { columns }
            | RowBasis::Distinct { columns } => names.extend(columns.iter().cloned()),
            _ => {}
        }
        names
            .into_iter()
            .map(|name| format!("r.{}", column_sql(&name)))
            .collect()
    }
    pub(super) fn identity_order(&self) -> Vec<OrderKey> {
        match &self.program.row_basis {
            RowBasis::Native { columns, .. }
            | RowBasis::Group { columns }
            | RowBasis::Distinct { columns } => columns
                .iter()
                .map(|name| OrderKey::ascending(name))
                .collect(),
            _ => self.program.order.clone(),
        }
    }
}

fn relation_order(order: &[OrderKey]) -> String {
    if order.is_empty() {
        "1".into()
    } else {
        order_by(order)
    }
}

impl Compiler<'_> {
    /// The keys of a grouping of `input`'s rows.
    fn group_keys(
        &mut self,
        input: &Relation,
        keys: &[query_language::ast::Column],
        at: usize,
    ) -> Result<Vec<GroupKey>, QueryError> {
        keys.iter()
            .map(|key| {
                let (expression, kind) =
                    self.sql_expression_with(&key.expression, at, &mut |compiler, name, at| {
                        input.field(compiler, name, at)
                    })?;
                scalar(kind, &key.column(), at)?;
                let column = Column {
                    name: key.column(),
                    kind,
                    reference: input.reference(&key.expression),
                };
                Ok((column, self.slot(), Self::display_value(&expression, kind)))
            })
            .collect()
    }

    /// `| group by K`: one row per group with its count.
    fn relation_group(
        &mut self,
        relation: &mut Relation,
        keys: &[query_language::ast::Column],
        at: usize,
    ) -> Result<(), QueryError> {
        let input = self.input(relation);
        let group_keys = self.group_keys(relation, keys, at)?;
        let mut columns = vec![];
        let mut select = vec![];
        let mut slots = std::collections::BTreeMap::new();
        for (column, slot, expression) in &group_keys {
            select.push(format!("{expression} AS {}", column_sql(slot)));
            slots.insert(column.name.clone(), slot.clone());
            columns.push(column.clone());
        }
        let count_slot = self.slot();
        select.push(format!("count(*) AS {}", column_sql(&count_slot)));
        columns.push(Column {
            name: "count".into(),
            kind: ColKind::Number,
            reference: None,
        });
        slots.insert("count".into(), count_slot.clone());
        let expressions = group_keys
            .iter()
            .map(|(_, _, expression)| expression.clone())
            .collect::<Vec<_>>()
            .join(", ");
        select.push(format!("json_array({expressions}) AS _pk"));
        let key_slots = group_keys
            .iter()
            .map(|(_, slot, _)| slot.clone())
            .collect::<Vec<_>>();
        let mut order = vec![OrderKey {
            column: count_slot,
            descending: true,
        }];
        order.extend(key_slots.iter().map(|slot| OrderKey::ascending(slot)));
        let sql = format!(
            "SELECT {} FROM {input} r GROUP BY {expressions} ORDER BY {}",
            select.join(", "),
            relation_order(&order)
        );
        self.set_rows(relation, sql);
        relation.program.columns = columns;
        relation.program.column_slots = slots;
        relation.program.time_text_columns.clear();
        relation.program.order = order;
        relation.program.row_basis = RowBasis::Group { columns: key_slots };
        relation.program.shape = Shape::Groups;
        relation.program.counts_near = false;
        relation.program.ranked = false;
        relation.sources.clear();
        relation.open_fields = false;
        Ok(())
    }

    /// `| group by K | first N`: the first `count` rows of each group, in
    /// the order the rows entered the grouping.
    fn relation_group_first(
        &mut self,
        relation: &mut Relation,
        keys: &[query_language::ast::Column],
        count: u64,
        at: usize,
    ) -> Result<(), QueryError> {
        let input = self.input(relation);
        let partitions = self
            .group_keys(relation, keys, at)?
            .into_iter()
            .map(|(_, _, expression)| expression)
            .collect::<Vec<_>>()
            .join(", ");
        let ordinal = self.slot();
        let order = relation_order(&relation.program.order);
        let sql = format!(
            "SELECT * FROM (SELECT r.*, row_number() OVER (PARTITION BY {partitions} ORDER BY {order}) AS {ordinal} FROM {input} r) WHERE {ordinal} <= {count} ORDER BY {order}"
        );
        self.set_rows(relation, sql);
        Ok(())
    }

    /// `| agg …` over the rows entering it, or with `group` (`| group by K
    /// | agg …`) one row per group of them.
    fn relation_aggregate(
        &mut self,
        relation: &mut Relation,
        group: Option<&[query_language::ast::Column]>,
        aggregates: &[Aggregate],
        at: usize,
    ) -> Result<(), QueryError> {
        let step = self.input(relation);
        let input = relation.clone();
        let native = if input.open_fields
            && input.sources.len() == 1
            && input.sources[0].prefix.is_empty()
        {
            Some(RowContext {
                unit: input.sources[0].unit,
                a: self.fresh(),
            })
        } else {
            None
        };
        let input_from = if let Some(context) = &native {
            let key = context.a.expand(units::spec(context.unit).key);
            format!(
                "{} JOIN {step} r ON {key}=r.{}",
                base_from(context.unit, &context.a),
                column_sql(&input.sources[0].key)
            )
        } else {
            format!("{step} r")
        };
        let mut columns = vec![];
        let mut slots = std::collections::BTreeMap::new();
        let mut select = vec![];
        let mut keys = vec![];
        let mut expressions = vec![];
        if let Some(group) = group {
            for (column, slot, expression) in self.group_keys(&input, group, at)? {
                select.push(format!("{expression} AS {}", column_sql(&slot)));
                slots.insert(column.name.clone(), slot.clone());
                columns.push(column);
                keys.push(slot);
                expressions.push(expression);
            }
        }
        let mut values = vec![];
        if group.is_some()
            && !aggregates
                .iter()
                .any(|a| matches!(a.function, AggregateFunction::Count))
        {
            values.push(("count".into(), "count(*)".into(), ColKind::Number, None));
        }
        for aggregate in aggregates {
            let lineage = if matches!(
                aggregate.function,
                AggregateFunction::Min | AggregateFunction::Max
            ) {
                aggregate
                    .arguments
                    .first()
                    .and_then(|arg| input.reference(arg))
            } else {
                None
            };
            let values_for_aggregate = if let Some(context) = &native
                && !aggregate
                    .arguments
                    .iter()
                    .any(|argument| materialized_expression(argument, &input.program.column_slots))
            {
                self.agg(aggregate, context, at)?
            } else {
                self.agg_with(aggregate, at, &mut |compiler, name, at| {
                    input.field(compiler, name, at)
                })?
            };
            for (name, expression, kind) in values_for_aggregate {
                values.push((name, expression, kind, lineage));
            }
        }
        for (name, expression, kind, reference) in values {
            let slot = self.slot();
            select.push(format!("{expression} AS {}", column_sql(&slot)));
            slots.insert(name.clone(), slot);
            columns.push(Column {
                name,
                kind,
                reference,
            });
        }
        let group_by = if expressions.is_empty() {
            String::new()
        } else {
            format!(" GROUP BY {}", expressions.join(", "))
        };
        select.push(if expressions.is_empty() {
            "1 AS _pk".into()
        } else {
            format!("json_array({}) AS _pk", expressions.join(", "))
        });
        let mut order = vec![];
        if !keys.is_empty() {
            if let Some(count) = slots.get("count") {
                order.push(OrderKey {
                    column: count.clone(),
                    descending: true,
                });
            }
            order.extend(keys.iter().map(|key| OrderKey::ascending(key)));
        }
        let sql = format!(
            "SELECT {} FROM {input_from}{group_by} ORDER BY {}",
            select.join(", "),
            relation_order(&order)
        );
        self.set_rows(relation, sql);
        relation.program.columns = columns;
        relation.program.column_slots = slots;
        relation.program.time_text_columns.clear();
        relation.program.order = order;
        relation.program.shape = if group.is_some() {
            Shape::Groups
        } else {
            Shape::Summary
        };
        relation.program.row_basis = if group.is_some() {
            RowBasis::Group { columns: keys }
        } else {
            RowBasis::Summary
        };
        relation.program.counts_near = false;
        relation.program.ranked = false;
        relation.sources.clear();
        relation.open_fields = false;
        Ok(())
    }

    fn relation_distinct(&mut self, relation: &mut Relation) {
        let input = self.input(relation);
        let values = relation
            .identity_order()
            .iter()
            .map(|key| column_sql(&key.column))
            .collect::<Vec<_>>();
        let ordinal = self.slot();
        let order = relation_order(&relation.program.order);
        let sql = format!(
            "SELECT * FROM (SELECT r.*, row_number() OVER (PARTITION BY {} ORDER BY {order}) AS {} FROM {input} r) WHERE {} = 1 ORDER BY {order}",
            if values.is_empty() {
                "_pk".into()
            } else {
                values.join(", ")
            },
            column_sql(&ordinal),
            column_sql(&ordinal),
        );
        self.set_rows(relation, sql);
    }
}

impl Compiler<'_> {
    fn relation_native_stage(
        &mut self,
        relation: &mut Relation,
        stage: &Transform,
        _original: &Query,
    ) -> Result<(), QueryError> {
        if !matches!(relation.program.row_basis, RowBasis::Native { .. })
            || !relation
                .sources
                .iter()
                .any(|source| source.prefix.is_empty())
        {
            return Err(not_built("this native stage after a change of row grain"));
        }
        if let TransformKind::Combine {
            how: Combination::Intersect | Combination::Except,
            query,
        } = &stage.kind
        {
            let other = self.other_query(query, relation.program.unit, stage.span.start)?;
            let yes = matches!(
                stage.kind,
                TransformKind::Combine {
                    how: Combination::Intersect,
                    ..
                }
            );
            let predicate = format!(
                "r._pk {} IN (SELECT _pk FROM ({}))",
                if yes { "" } else { "NOT" },
                other.rows()
            );
            self.relation_filter(relation, &predicate);
            return Ok(());
        }
        debug_assert!(!matches!(stage.kind, TransformKind::Combine { .. }));
        let condition =
            if matches!(stage.kind, TransformKind::Matches) && !relation.patterns.is_empty() {
                Some(Condition {
                    kind: ConditionKind::And {
                        conditions: relation.patterns.clone(),
                    },
                    span: stage.span,
                })
            } else {
                None
            };
        let pattern_unit = relation
            .patterns
            .iter()
            .find_map(|condition| match &condition.kind {
                ConditionKind::Exists { unit, .. } | ConditionKind::Sequence { unit, .. } => {
                    bind_unit(unit).ok()
                }
                _ => None,
            });
        let parent_unit = relation.program.unit;
        let native = Query {
            source: Source::Unit {
                unit: relation.program.unit.name().into(),
                implicit: false,
                condition,
            },
            transforms: vec![stage.clone()],
            attach: vec![],
        };
        let input = self.input(relation);
        let mut parent_input = relation.program.clone();
        if matches!(stage.kind, TransformKind::Matches) {
            let name = self.slot();
            self.ctes
                .push(format!("{name} AS MATERIALIZED (SELECT * FROM {input})"));
            parent_input.derived_mut().rows = format!("SELECT * FROM {name}");
        }
        let mut compiler = self.nested(self.text);
        compiler.params = std::mem::take(&mut self.params);
        compiler.input_rows = Some(parent_input.rows().to_owned());
        compiler.near = self.near.clone().or(relation.program.near.clone());
        let Some(selection) = Selection::of(&native, &compiler.binder(), &self.opts.fields)? else {
            unreachable!("a unit source");
        };
        let mut result = compiler.native_program_with(
            &selection,
            &native.transforms[selection.stages_from..],
            None,
        )?;
        if matches!(stage.kind, TransformKind::Matches) {
            sequences::project_match_parent(&mut result, &parent_input);
        }
        self.near = result.near.clone();
        self.params = std::mem::take(&mut result.params);
        if let Some(mut measure) = result.measure.take() {
            // The preceding relation owns membership and order. The native
            // descriptor contributes only the default comparison population.
            measure.selection_sql = relation.program.rows().to_owned();
            measure.everything = false;
            relation.program.measure = Some(measure);
            return Ok(());
        }
        join(&mut result.notes, &mut relation.program.notes);
        join(&mut result.incomplete, &mut relation.program.incomplete);
        join(&mut result.id_checks, &mut relation.program.id_checks);
        join(&mut result.definitions, &mut relation.program.definitions);
        join(
            &mut result.change_generations,
            &mut relation.program.change_generations,
        );
        join(
            &mut result.assertion_cutoffs,
            &mut relation.program.assertion_cutoffs,
        );
        result
            .resolved_times
            .extend(relation.program.resolved_times.iter().cloned());
        *relation = Relation::new(result);
        if matches!(stage.kind, TransformKind::Matches) {
            if let Some(unit) = pattern_unit {
                relation.sources.push(NativeSource {
                    prefix: "step.".into(),
                    unit,
                    key: "_pk".into(),
                });
                relation.sources.push(NativeSource {
                    prefix: "parent.".into(),
                    unit: parent_unit,
                    key: "_match_parent_pk".into(),
                });
            }
            relation.program.row_basis = RowBasis::Group {
                columns: vec!["parent".into(), "match".into(), "position".into()],
            };
            relation.open_fields = true;
        }
        Ok(())
    }

    fn relation_diff(
        &mut self,
        relation: &mut Relation,
        query: &Query,
        at: usize,
    ) -> Result<(), QueryError> {
        let mut before = Relation::new(self.other_query(query, relation.program.unit, at)?);
        let after_input = self.input(relation);
        let before_input = self.input(&mut before);
        let left_keys = relation.identity_order();
        let right_keys = before.identity_order();
        if left_keys.len() != right_keys.len() || left_keys.is_empty() {
            return Err(QueryError::Invalid {
                message: "diff requires the same semantic row identity on both sides".into(),
                position: at,
            });
        }
        let join = left_keys
            .iter()
            .zip(&right_keys)
            .map(|(left, right)| {
                format!(
                    "a.{} IS b.{}",
                    column_sql(&left.column),
                    column_sql(&right.column)
                )
            })
            .collect::<Vec<_>>()
            .join(" AND ");
        let ordinal = self.slot();
        let previous = self.slot();
        let change = self.slot();
        let after_rows = format!(
            "SELECT r.*, row_number() OVER (ORDER BY {}) AS {ordinal} FROM {after_input} r",
            relation_order(&relation.program.order),
        );
        let before_rows = format!(
            "SELECT r.*, row_number() OVER (ORDER BY {}) AS {previous} FROM {before_input} r",
            relation_order(&before.program.order),
        );
        // The public relation carries rows that entered, left, or changed rank.
        let mut values = vec![];
        for column in &relation.program.columns {
            let left = column_sql(relation.program.column_slot(&column.name));
            let right = before
                .program
                .columns
                .iter()
                .find(|candidate| candidate.name == column.name);
            let left_value = relation.display_column_on(self, column, column.kind, at, "a")?;
            let value = match right {
                Some(right) => {
                    let right_value =
                        before.display_column_on(self, right, column.kind, at, "b")?;
                    format!(
                        "CASE WHEN a.{ordinal} IS NULL THEN {right_value} ELSE {left_value} END"
                    )
                }
                None => left_value,
            };
            values.push(format!("{value} AS {left}"));
        }
        values.push(format!(
            "CASE WHEN a.{ordinal} IS NULL THEN b._pk ELSE a._pk END AS _pk"
        ));
        for (left, right) in left_keys.iter().zip(&right_keys) {
            let left = column_sql(&left.column);
            let right = column_sql(&right.column);
            values.push(format!(
                "CASE WHEN a.{ordinal} IS NULL THEN b.{right} ELSE a.{left} END AS {left}"
            ));
        }
        values.push(format!("CASE WHEN a.{ordinal} IS NULL THEN 'left' WHEN b.{previous} IS NULL THEN 'joined' ELSE 'moved' END AS {change}"));
        values.push(format!("a.{ordinal} AS {ordinal}"));
        values.push(format!("b.{previous} AS {previous}"));
        let changes = format!(
            "WITH after_rows AS ({after_rows}), before_rows AS ({before_rows}) SELECT {} FROM after_rows a FULL OUTER JOIN before_rows b ON {join} WHERE a.{ordinal} IS NULL OR b.{previous} IS NULL OR a.{ordinal} != b.{previous} ORDER BY coalesce(a.{ordinal}, b.{previous})",
            values.join(", ")
        );
        let changes = self.set_rows(relation, changes);
        for (name, slot, kind) in [
            ("change", change, ColKind::Text),
            ("new_rank", ordinal.clone(), ColKind::Number),
            ("old_rank", previous, ColKind::Number),
        ] {
            relation.program.columns.push(Column {
                name: name.into(),
                kind,
                reference: None,
            });
            relation.program.column_slots.insert(name.into(), slot);
        }
        let final_order = self.slot();
        let sql = format!(
            "SELECT d.*, row_number() OVER (ORDER BY {ordinal} IS NULL, coalesce({ordinal}, {})) AS {final_order} FROM {changes} d",
            relation.program.column_slot("old_rank"),
        );
        self.set_rows(relation, sql);
        relation.program.order = vec![OrderKey::ascending(&final_order)];
        relation.program.ranked = false;
        relation.program.counts_near = false;
        relation.program.time_text_columns.clear();
        relation.sources.clear();
        relation.open_fields = false;
        Ok(())
    }
}

impl Compiler<'_> {
    fn relation_scope(
        &mut self,
        relation: &mut Relation,
        unit: &str,
        condition: Option<&Condition>,
        at: usize,
    ) -> Result<(), QueryError> {
        let target = bind_unit(unit)?;
        let Some(source) = relation
            .sources
            .iter()
            .find(|s| s.prefix.is_empty())
            .cloned()
        else {
            return Err(QueryError::Invalid {
                message: "a related unit requires native source row identity".into(),
                position: at,
            });
        };
        let source_context = RowContext {
            unit: source.unit,
            a: self.fresh(),
        };
        let target_context = RowContext {
            unit: target,
            a: self.fresh(),
        };
        let source_key = source_context.a.expand(units::spec(source.unit).key);
        let target_key = target_context.a.expand(units::spec(target).key);
        let link = if source.unit == target {
            format!("{source_key}={target_key}")
        } else if target == Unit::Sessions && fields::has_session(source.unit) {
            format!(
                "{}={target_key}",
                child_session_column(source.unit, &source_context.a)
            )
        } else if source.unit == Unit::Sessions && fields::has_session(target) {
            format!(
                "{}={source_key}",
                child_session_column(target, &target_context.a)
            )
        } else if let Some((_, parent)) = units::spec(target)
            .parents
            .iter()
            .find(|(parent, _)| *parent == source.unit)
        {
            format!("{}={source_key}", target_context.a.expand(parent))
        } else if let Some((_, parent)) = units::spec(source.unit)
            .parents
            .iter()
            .find(|(parent, _)| *parent == target)
        {
            format!("{}={target_key}", source_context.a.expand(parent))
        } else if fields::has_session(source.unit) && fields::has_session(target) {
            format!(
                "{}={}",
                child_session_column(source.unit, &source_context.a),
                child_session_column(target, &target_context.a)
            )
        } else {
            return Err(QueryError::Invalid {
                message: format!(
                    "{} has no native relation to {}; there is no implicit session association",
                    source.unit.name(),
                    target.name()
                ),
                position: at,
            });
        };
        let input = self.input(relation);
        let keys = format!(
            "SELECT DISTINCT {target_key} AS _pk FROM {}, {} WHERE {link} AND {source_key} IN (SELECT {} FROM {input})",
            base_from(source.unit, &source_context.a),
            base_from(target, &target_context.a),
            column_sql(&source.key),
        );
        let query = Query {
            source: Source::Unit {
                unit: target.name().into(),
                implicit: false,
                condition: condition.cloned(),
            },
            transforms: vec![],
            attach: vec![],
        };
        let mut compiler = self.nested(self.text);
        compiler.params = std::mem::take(&mut self.params);
        compiler.input_rows = Some(keys);
        compiler.source_near_allowed = false;
        compiler.near = self.near.clone().or(relation.program.near.clone());
        let mut program = compiler.program(&query)?;
        self.near = program.near.clone();
        self.params = std::mem::take(&mut program.params);
        join(&mut program.notes, &mut relation.program.notes);
        join(&mut program.incomplete, &mut relation.program.incomplete);
        join(&mut program.id_checks, &mut relation.program.id_checks);
        join(&mut program.definitions, &mut relation.program.definitions);
        join(
            &mut program.change_generations,
            &mut relation.program.change_generations,
        );
        join(
            &mut program.assertion_cutoffs,
            &mut relation.program.assertion_cutoffs,
        );
        program
            .resolved_times
            .extend(relation.program.resolved_times.iter().cloned());
        *relation = Relation::new(program);
        if let Some(condition) = condition {
            relation.patterns = match_patterns(condition);
        }
        Ok(())
    }
}

fn match_patterns(condition: &Condition) -> Vec<Condition> {
    top_level(condition)
        .into_iter()
        .filter(|condition| {
            matches!(
                condition.kind,
                ConditionKind::Exists { .. } | ConditionKind::Sequence { .. }
            )
        })
        .collect()
}

fn materialized_expression(
    value: &Expression,
    columns: &std::collections::BTreeMap<String, String>,
) -> bool {
    query_language::depth::deeper(|| match value {
        Expression::Field { name } => columns.contains_key(name),
        Expression::Function { arguments, .. } => arguments
            .iter()
            .any(|value| materialized_expression(value, columns)),
        Expression::Arithmetic { left, right, .. } => {
            materialized_expression(left, columns) || materialized_expression(right, columns)
        }
        Expression::Value { .. } => false,
    })
}
fn materialized_condition(
    condition: &Condition,
    columns: &std::collections::BTreeMap<String, String>,
) -> bool {
    query_language::depth::deeper(|| match &condition.kind {
        ConditionKind::And { conditions } | ConditionKind::Or { conditions } => conditions
            .iter()
            .any(|condition| materialized_condition(condition, columns)),
        ConditionKind::Not { condition } => materialized_condition(condition, columns),
        ConditionKind::Match { field, .. }
        | ConditionKind::Since { field, .. }
        | ConditionKind::Until { field, .. } => columns.contains_key(field),
        ConditionKind::Compare { left, right, .. } => {
            materialized_expression(left, columns) || materialized_expression(right, columns)
        }
        ConditionKind::Between { left, .. }
        | ConditionKind::TimeRange { left, .. }
        | ConditionKind::In { left, .. } => materialized_expression(left, columns),
        _ => false,
    })
}
