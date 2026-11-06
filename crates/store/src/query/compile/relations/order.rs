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
    pub(super) fn rel_order_pair(
        &mut self,
        outer: &mut Relation,
        inner: Relation,
        relation: OrderRelation,
        within: Option<&Distance>,
        point: TimePoint,
        scope: Option<&str>,
        at: usize,
    ) -> Result<String, QueryError> {
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
        let nearest = format!(
            "{}, {}, {}",
            pair.coarse,
            pair.distance_key,
            identity_sql(&inner, "i")
        );
        let mut projections = vec!["r.*".into()];
        self.project_match(
            outer,
            &inner,
            &mut projections,
            within.map(|_| pair.distance.as_str()),
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
             (SELECT i.{} FROM {inner_name} i WHERE ({}) ORDER BY {nearest} LIMIT 1) AS {}, \
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
                    coarse: "0".into(),
                    outer_unknown: "0".into(),
                }
            }
        };
        let mut projections = vec!["r.*".into()];
        self.project_match(
            outer,
            &inner,
            &mut projections,
            matches!(on, JoinOn::Order { .. }).then_some(pair.distance.as_str()),
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
        distance: Option<&str>,
    ) {
        for column in &inner.program.columns {
            let slot = outer.slot();
            projections.push(format!(
                "i.{} AS {}",
                column_sql(inner.program.column_slot(&column.name)),
                column_sql(&slot)
            ));
            let name = format!("match.{}", column.name);
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
        if let Some(distance) = distance {
            let slot = outer.slot();
            projections.push(format!("{distance} AS {}", column_sql(&slot)));
            let name = "match.distance".to_owned();
            outer.program.columns.retain(|prior| prior.name != name);
            outer.program.columns.push(Column {
                name: name.clone(),
                kind: ColKind::Number,
                reference: None,
            });
            outer.program.column_slots.insert(name, slot);
        }
    }

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
                coarse: "0".into(),
                outer_unknown: format!("({}) IS NULL", left.1),
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
            return Ok(("NULL".into(), "NULL".into()));
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
