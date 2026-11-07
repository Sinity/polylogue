//! Order relations retain native evidence and full-scope positions across stages.
use super::*;

struct PairOrder {
    ctes: Vec<String>,
    predicate: String,
    distance: String,
    distance_key: String,
    coarse: String,
    outer_unknown: String,
}

impl Compiler<'_> {
    #[allow(
        clippy::too_many_arguments,
        reason = "Mirrors the fixed order-expression fields."
    )]
    pub(super) fn rel_order_pair(
        &mut self,
        outer: &mut Relation,
        inner: Relation,
        relation: OrderRelation,
        within: Option<&Distance>,
        point: TimePoint,
        scope: Option<&str>,
        pick: &WitnessChoice,
        at: usize,
    ) -> Result<String, QueryError> {
        outer.reserve(&inner);
        let pair = self.pair_order(outer, &inner, relation, within, point, scope, at)?;
        let index = outer.slot();
        let chosen = outer.slot();
        let unknown = outer.slot();
        let truth = outer.slot();
        let inner_name = outer.slot();
        let outer_name = outer.slot();
        let ctes = if pair.ctes.is_empty() {
            String::new()
        } else {
            format!("{}, ", pair.ctes.join(", "))
        };
        let witness_order = match pick {
            WitnessChoice::Nearest => format!(
                "{}, {}, {}",
                pair.coarse,
                pair.distance_key,
                identity_sql(&inner, "i")
            ),
            WitnessChoice::FirstBy { keys } => {
                if keys.is_empty() {
                    return Err(order_invalid("pick first by requires a sort key", at));
                }
                let mut terms = vec![];
                for key in keys {
                    let exact_distance = matches!(&key.expression,
                        Expression::Function { name, arguments } if name == "abs"
                            && matches!(arguments.as_slice(), [Expression::Field { name }] if name == "match.distance"));
                    let (sql, kind) = if exact_distance
                        && !matches!(within, Some(Distance::Steps { .. }))
                    {
                        (pair.distance_key.clone(), ColKind::Text)
                    } else {
                        self.sql_expression_with(&key.expression, at, &mut |compiler, name, at| {
                            match name {
                                "match.distance" => {
                                    Ok(if matches!(within, Some(Distance::Steps { .. })) {
                                        (pair.distance.clone(), ColKind::Number)
                                    } else {
                                        (
                                            format!("({}) * 1000.0", pair.distance),
                                            ColKind::DurationSeconds,
                                        )
                                    })
                                }
                                "match.coarse" => Ok((pair.coarse.clone(), ColKind::Bool)),
                                "match.id" => {
                                    inner.field_on(compiler, "id", at, "i").or_else(|_| {
                                        let columns = identity_columns(&inner);
                                        let identity = if matches!(
                                            inner.program.row_basis,
                                            RowBasis::Native { .. }
                                        ) && columns.len() == 1
                                        {
                                            format!("i.{}", column_sql(&columns[0]))
                                        } else {
                                            format!("json_array({})", identity_sql(&inner, "i"))
                                        };
                                        Ok((identity, ColKind::Text))
                                    })
                                }
                                _ => inner.field_on(compiler, name, at, "i"),
                            }
                        })?
                    };
                    scalar(kind, &key.expression.column(), at)?;
                    terms.push(format!("({sql}) IS NULL"));
                    terms.push(format!(
                        "({sql}){}",
                        if key.descending { " DESC" } else { "" }
                    ));
                }
                terms.push(identity_sql(&inner, "i"));
                terms.join(", ")
            }
        };
        let mut projections = vec!["r.*".into()];
        self.project_match(
            outer,
            &inner,
            &mut projections,
            within.map(|_| {
                (
                    pair.distance.as_str(),
                    if matches!(within, Some(Distance::Steps { .. })) {
                        ColKind::Number
                    } else {
                        ColKind::DurationSeconds
                    },
                )
            }),
        );
        projections.push(format!(
            "CASE WHEN i.{} IS NOT NULL THEN 1 WHEN r.{} THEN NULL ELSE 0 END AS {}",
            column_sql(&index),
            column_sql(&unknown),
            column_sql(&truth)
        ));
        outer.program.sql = format!(
            "WITH {ctes}{inner_name} AS MATERIALIZED (SELECT i.*, row_number() OVER (ORDER BY {}) AS {} FROM ({}) i), \
             {outer_name} AS MATERIALIZED (SELECT r.*, \
             (SELECT i.{} FROM {inner_name} i WHERE ({}) ORDER BY {witness_order} LIMIT 1) AS {}, \
             (({}) OR EXISTS (SELECT 1 FROM {inner_name} i WHERE ({}) IS NULL)) AS {} FROM ({}) r) \
             SELECT {} FROM {outer_name} r LEFT JOIN {inner_name} i ON i.{} = r.{} ORDER BY {}",
            qualified_order(&inner.program.order, "i"),
            column_sql(&index),
            inner.program.sql,
            column_sql(&index),
            pair.predicate,
            column_sql(&chosen),
            pair.outer_unknown,
            pair.predicate,
            column_sql(&unknown),
            outer.program.sql,
            projections.join(", "),
            column_sql(&index),
            column_sql(&chosen),
            qualified_order(&outer.program.order, "r")
        );
        outer.program.row_count_sql = None;
        outer.program.row_membership_sql = None;
        Ok(format!("r.{}", column_sql(&truth)))
    }

    pub(super) fn rel_join_pair(
        &mut self,
        outer: &mut Relation,
        inner: Relation,
        on: &JoinOn,
        at: usize,
    ) -> Result<(), QueryError> {
        outer.reserve(&inner);
        let pair = match on {
            JoinOn::Order {
                relation,
                within,
                at: point,
                scope,
            } => self.pair_order(
                outer,
                &inner,
                *relation,
                within.as_ref(),
                *point,
                scope.as_deref(),
                at,
            )?,
            JoinOn::Key { left, right } => {
                let (left, _) = outer.field_on(self, left, at, "r")?;
                let (right, _) = inner.field_on(self, right, at, "i")?;
                PairOrder {
                    ctes: vec![],
                    predicate: format!("({left}) = ({right})"),
                    distance: "NULL".into(),
                    distance_key: "NULL".into(),
                    coarse: "(0 + 0)".into(),
                    outer_unknown: "0".into(),
                }
            }
        };
        let mut projections = vec!["r.*".into()];
        self.project_match(
            outer,
            &inner,
            &mut projections,
            matches!(
                on,
                JoinOn::Order {
                    within: Some(_),
                    ..
                }
            )
            .then_some((
                pair.distance.as_str(),
                if matches!(
                    on,
                    JoinOn::Order {
                        within: Some(Distance::Steps { .. }),
                        ..
                    }
                ) {
                    ColKind::Number
                } else {
                    ColKind::DurationSeconds
                },
            )),
        );
        let mut identity = identity_columns(outer);
        for key in identity_columns(&inner) {
            let slot = outer.slot();
            projections.push(format!("i.{} AS {}", column_sql(&key), column_sql(&slot)));
            identity.push(slot);
        }
        if matches!(on, JoinOn::Order { .. }) {
            for value in [&pair.coarse, &pair.distance_key] {
                let slot = outer.slot();
                projections.push(format!("{value} AS {}", column_sql(&slot)));
                outer.program.order.push(OrderKey::ascending(&slot));
            }
        }
        for key in identity_columns(&inner) {
            let slot = outer.slot();
            projections.push(format!("i.{} AS {}", column_sql(&key), column_sql(&slot)));
            outer.program.order.push(OrderKey::ascending(&slot));
        }
        let ctes = if pair.ctes.is_empty() {
            String::new()
        } else {
            format!("WITH {} ", pair.ctes.join(", "))
        };
        outer.program.sql = format!(
            "{ctes}SELECT {} FROM ({}) r JOIN ({}) i ON ({}) ORDER BY {}",
            projections.join(", "),
            outer.program.sql,
            inner.program.sql,
            pair.predicate,
            relation_order(&outer.program.order)
        );
        outer.program.row_basis = RowBasis::Distinct { columns: identity };
        outer.program.row_count_sql = None;
        outer.program.row_membership_sql = None;
        Ok(())
    }

    fn project_match(
        &mut self,
        outer: &mut Relation,
        inner: &Relation,
        projections: &mut Vec<String>,
        distance: Option<(&str, ColKind)>,
    ) {
        outer
            .program
            .columns
            .retain(|column| !column.name.starts_with("match."));
        outer
            .program
            .column_slots
            .retain(|name, _| !name.starts_with("match."));
        outer
            .program
            .time_text_columns
            .retain(|name| !name.starts_with("match."));
        for column in &inner.program.columns {
            let slot = outer.slot();
            projections.push(format!(
                "i.{} AS {}",
                column_sql(inner.program.column_slot(&column.name)),
                column_sql(&slot)
            ));
            let name = format!("match.{}", column.name);
            if inner.program.time_text_columns.contains(&column.name) {
                outer.program.time_text_columns.insert(name.clone());
            }
            outer.program.columns.retain(|prior| prior.name != name);
            outer.program.columns.push(Column {
                name: name.clone(),
                kind: column.kind,
                reference: column.reference,
            });
            outer.program.column_slots.insert(name, slot);
        }
        outer
            .sources
            .retain(|source| !source.prefix.starts_with("match."));
        for source in &inner.sources {
            let slot = outer.slot();
            projections.push(format!(
                "i.{} AS {}",
                column_sql(&source.key),
                column_sql(&slot)
            ));
            outer.sources.push(NativeSource {
                prefix: format!("match.{}", source.prefix),
                unit: source.unit,
                key: slot,
            });
        }
        if let Some((distance, kind)) = distance {
            let slot = outer.slot();
            projections.push(format!("{distance} AS {}", column_sql(&slot)));
            let name = "match.distance".to_owned();
            outer.program.columns.retain(|prior| prior.name != name);
            outer.program.columns.push(Column {
                name: name.clone(),
                kind,
                reference: None,
            });
            outer.program.column_slots.insert(name, slot);
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "Mirrors order inputs shared by predicates and joins."
    )]
    fn pair_order(
        &mut self,
        outer: &Relation,
        inner: &Relation,
        relation: OrderRelation,
        within: Option<&Distance>,
        point: TimePoint,
        scope: Option<&str>,
        at: usize,
    ) -> Result<PairOrder, QueryError> {
        if let Some(Distance::Steps { count }) = within {
            let timeline = self.fresh().row;
            let mut ctes = vec![];
            let left = self.order_position(outer, "r", scope, at, &timeline, &mut ctes)?;
            let right = self.order_position(inner, "i", scope, at, &timeline, &mut ctes)?;
            let distance = format!("(({}) - ({}))", right.1, left.1);
            let direction = match relation {
                OrderRelation::Before => format!("{distance} > 0"),
                OrderRelation::After => format!("{distance} < 0"),
                _ => "1".into(),
            };
            let limit = self.bind(Param::Text(count.to_string()));
            return Ok(PairOrder {
                ctes,
                predicate: format!(
                    "CASE WHEN ({}) IS NULL THEN NULL WHEN ({}) IS NULL THEN NULL ELSE (({}) = ({}) AND ({distance}) != 0 AND abs({distance}) <= CAST({limit} AS REAL) AND ({direction})) END",
                    left.1, right.1, left.0, right.0
                ),
                distance_key: format!("abs({distance})"),
                distance,
                coarse: "(0 + 0)".into(),
                outer_unknown: format!("({}) IS NULL", left.1),
            });
        }
        if outer.sources.is_empty() || inner.sources.is_empty() {
            return Err(QueryError::Unavailable {
                what: "native time evidence was not retained for this relation".into(),
            });
        }
        let (left, left_storage) = self.order_time(outer, "r");
        let (right, right_storage) = self.order_time(inner, "i");
        let within = match within {
            Some(Distance::Time { duration }) => {
                let text = duration
                    .text()
                    .ok_or_else(|| order_invalid("order window requires an exact duration", at))?;
                let nanos = text
                    .strip_suffix("ns")
                    .and_then(|n| n.parse::<u128>().ok())
                    .or_else(|| {
                        query_language::time::duration(text)
                            .and_then(|v| v.as_nanos().try_into().ok())
                    })
                    .ok_or_else(|| order_invalid("order window requires an exact duration", at))?;
                self.bind(Param::Text(nanos.to_string()))
            }
            _ => "NULL".into(),
        };
        let point = match point {
            TimePoint::When => "when",
            TimePoint::Start => "start",
            TimePoint::End => "end",
            TimePoint::MatchWhen
                if !matches!(relation, OrderRelation::During | OrderRelation::Containing) =>
            {
                "match.when"
            }
            TimePoint::MatchWhen | TimePoint::MatchStart | TimePoint::MatchEnd => {
                return Err(not_built(
                    "matched span endpoints require a source with a declared actual span; Polylogue rows expose occurrence evidence",
                ));
            }
        };
        Ok(PairOrder {
            ctes: vec![],
            predicate: format!(
                "pl_time_order({left}, '{left_storage}', {right}, '{right_storage}', '{}', {within}, '{point}')",
                relation.name()
            ),
            distance: format!(
                "pl_time_distance({left}, '{left_storage}', {right}, '{right_storage}', '{point}')"
            ),
            distance_key: format!(
                "pl_time_distance_key({left}, '{left_storage}', {right}, '{right_storage}', '{point}')"
            ),
            coarse: format!(
                "NOT (pl_time_is_point({left}, '{left_storage}') AND pl_time_is_point({right}, '{right_storage}'))"
            ),
            outer_unknown: format!(
                "pl_time_order({left}, '{left_storage}', {left}, '{left_storage}', 'near', NULL, 'when') IS NULL"
            ),
        })
    }

    fn order_time(&mut self, relation: &Relation, alias: &str) -> (String, &'static str) {
        let Some(source) = relation
            .sources
            .iter()
            .find(|source| source.prefix.is_empty())
        else {
            return ("NULL".into(), "tuple");
        };
        let context = RowContext {
            unit: source.unit,
            a: self.fresh(),
        };
        let Some(field) = polylogue_query::units::spec(source.unit).main_time else {
            return ("NULL".into(), "tuple");
        };
        let Some(time) = fields::native_time(source.unit, field, &context.a) else {
            return ("NULL".into(), "tuple");
        };
        let key = context.a.expand(units::spec(source.unit).key);
        (
            format!(
                "(SELECT {} FROM {} WHERE {key} = {alias}.{})",
                time.bytes,
                base_from(source.unit, &context.a),
                column_sql(&source.key)
            ),
            time.storage.code(),
        )
    }

    fn order_position(
        &mut self,
        relation: &Relation,
        alias: &str,
        scope: Option<&str>,
        at: usize,
        timeline: &str,
        ctes: &mut Vec<String>,
    ) -> Result<(String, String), QueryError> {
        let Some(source) = relation
            .sources
            .iter()
            .find(|source| source.prefix.is_empty())
        else {
            return Err(QueryError::Unavailable {
                what: "native full-scope step positions were not retained for this relation".into(),
            });
        };
        if scope.is_some_and(|name| name != "session_tool_calls") {
            return Err(order_invalid("tool calls step in session_tool_calls", at));
        }
        let a = self.fresh();
        let spec = units::spec(source.unit);
        let (terms, admission) = match source.unit {
            Unit::ToolCalls => (
                format!(
                    "{}.order_input_id, {}.seq, {}.record_position, {}.id, {}.idx",
                    a.m, a.m, a.m, a.m, a.cb
                ),
                format!(
                    "{} AND {}",
                    fields::own_messages(&a.m, None),
                    fields::counts_as_action(&a.t)
                ),
            ),
            _ => return Err(order_invalid("unit has no native step order", at)),
        };
        let session = child_session_column(source.unit, &a);
        let key = a.expand(spec.key);
        let rows = format!(
            "SELECT {key} AS native_key, {session} AS parent, row_number() OVER (PARTITION BY {session} ORDER BY {terms}) AS position FROM {} WHERE {admission}",
            base_from(source.unit, &a)
        );
        if ctes.is_empty() {
            ctes.push(format!("{timeline} AS MATERIALIZED ({rows})"));
        }
        let selected = format!(
            "FROM {timeline} timeline WHERE timeline.native_key = {alias}.{}",
            column_sql(&source.key)
        );
        Ok((
            format!("(SELECT parent {selected})"),
            format!("(SELECT position {selected})"),
        ))
    }
}

fn order_invalid(message: &str, position: usize) -> QueryError {
    QueryError::Invalid {
        message: message.into(),
        position,
    }
}
fn identity_columns(relation: &Relation) -> Vec<String> {
    match &relation.program.row_basis {
        RowBasis::Native { columns, .. }
        | RowBasis::Group { columns }
        | RowBasis::Distinct { columns }
            if !columns.is_empty() =>
        {
            columns.clone()
        }
        _ => vec!["_pk".into()],
    }
}
fn identity_sql(relation: &Relation, alias: &str) -> String {
    identity_columns(relation)
        .iter()
        .map(|key| format!("{alias}.{}", column_sql(key)))
        .collect::<Vec<_>>()
        .join(", ")
}
fn qualified_order(order: &[OrderKey], alias: &str) -> String {
    if order.is_empty() {
        "1".into()
    } else {
        order
            .iter()
            .map(|key| {
                format!(
                    "{alias}.{}{}",
                    column_sql(&key.column),
                    if key.descending { " DESC" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}
