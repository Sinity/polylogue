//! Prepared questions and streaming rows bound to one Reader and access scope.
pub use crate::query::{Param, Program, RowBasis};
use crate::{ReadScopeToken, Reader, SqlValue, StoreError};
use polylogue_model::{DocumentId, MessageId, Reference, SessionId};
use polylogue_query::catalog::reference_unit;
pub use polylogue_query::plan::{Plan, ResolvedDefinition};
use polylogue_query::{
    Options, QueryError, Shape,
    ast::{ConditionKind, TextMode, Unit},
};
use query_language::{
    bind::BoundQuery,
    cursor::{CursorContext, OrderPosition},
    identity::QuestionId,
};
use serde_json::Value;
mod content;
mod execution;
mod seek;
use std::{collections::BTreeMap, ops::ControlFlow, sync::Arc};

#[derive(Debug, thiserror::Error)]
pub enum SelectionError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error(transparent)]
    Language(#[from] query_language::Error),
    #[error("{message}")]
    Invalid {
        kind: &'static str,
        message: String,
        detail: Option<Value>,
    },
}
impl SelectionError {
    pub fn kind(&self) -> &str {
        match self {
            Self::Store(StoreError::ReadDeadlineExceeded) => "deadline_exceeded",
            Self::Store(StoreError::ReadCancelled) => "cancelled",
            Self::Store(StoreError::ScopeChanged) => "scope_changed",
            Self::Store(StoreError::NotFound(_)) => "not_found",
            Self::Store(_) => "store",
            Self::Query(error) => error.kind(),
            Self::Language(error) => error.kind.code(),
            Self::Invalid { kind, .. } => kind,
        }
    }
    pub fn detail(&self) -> Option<Value> {
        match self {
            Self::Query(error) => error.detail(),
            Self::Language(error) => serde_json::to_value(error).ok(),
            Self::Invalid { detail, .. } => detail.clone(),
            Self::Store(_) => None,
        }
    }
}
fn invalid(kind: &'static str, message: impl Into<String>) -> SelectionError {
    SelectionError::Invalid {
        kind,
        message: message.into(),
        detail: None,
    }
}

/// Provider work is injected; preparing or visiting never selects a provider.
pub type NearFn<'a> = dyn Fn(&str) -> Result<Arc<crate::vectors::Similarity>, String> + 'a;

pub struct PreparedSelection<'read> {
    reader: usize,
    visiting: std::cell::Cell<bool>,
    pinned: std::marker::PhantomData<&'read ()>,
    plan: Plan,
    program: Program,
    scope: ReadScopeToken,
    context: CursorContext,
    direct: Option<Reference>,
    execution_options: polylogue_model::query_objects::QueryExecutionOptions,
}
impl PreparedSelection<'_> {
    pub fn query_object(
        &self,
    ) -> Result<polylogue_model::query_objects::QueryObject, SelectionError> {
        Ok(self.plan.query_object()?)
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn direct_reference(&self) -> Option<&Reference> {
        self.direct.as_ref()
    }
    pub fn question_id(&self) -> QuestionId {
        self.plan.question_id()
    }
    pub fn bound_query(&self) -> &BoundQuery {
        self.plan.bound_query()
    }
    pub fn program(&self) -> &Program {
        &self.program
    }
    pub fn definitions(&self) -> &[ResolvedDefinition] {
        self.plan.definitions()
    }
    pub fn cursor_context(&self) -> &CursorContext {
        &self.context
    }
    pub fn scope_token(&self) -> ReadScopeToken {
        self.scope.clone()
    }
}
/// Natural source coordinates emitted by the compiler's own selected relation.
/// These are not Sinex parents; lifecycle evidence resolves their exact native facts.
#[derive(Debug, Clone)]
pub struct SelectionFactKey {
    pub unit: Unit,
    pub values: Vec<Value>,
}

impl SelectionFactKey {
    pub fn reference(&self) -> Option<Reference> {
        identity_reference(self.unit, &self.values)
    }
}
#[derive(Debug, Clone)]
pub struct SelectionRow {
    pub reference: Option<Reference>,
    pub values: Value,
    pub position: OrderPosition,
    pub identity: Option<RowIdentity>,
    /// The stored observation basis, never the current clock or event time.
    pub first_seen: Option<i64>,
    pub aggregates: BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RowIdentity {
    Entity { unit: Unit, key: Vec<Value> },
    Group { keys: BTreeMap<String, Value> },
    Summary,
    Distinct { values: BTreeMap<String, Value> },
}
impl RowIdentity {
    fn from_row(program: &Program, row: &Value) -> Option<Self> {
        Some(match &program.row_basis {
            RowBasis::Native { unit, columns } => Self::Entity {
                unit: *unit,
                key: columns.iter().map(|column| row[column].clone()).collect(),
            },
            RowBasis::Group { columns } => Self::Group {
                keys: columns
                    .iter()
                    .map(|column| (column.clone(), row[column].clone()))
                    .collect(),
            },
            RowBasis::Summary => Self::Summary,
            RowBasis::Distinct { columns } if !program.ranked => Self::Distinct {
                values: columns
                    .iter()
                    .map(|column| (column.clone(), row[column].clone()))
                    .collect(),
            },
            _ => return None,
        })
    }
}

impl Reader<'_> {
    /// Bind a query at its declared grain using this Reader's immutable definitions.
    /// Execution options and the captured clock remain outside semantic identity.
    pub fn prepare_query_plan(
        &self,
        text: &str,
        options: &Options,
    ) -> Result<Plan, SelectionError> {
        self.check()?;
        let mut options = options.clone();
        options.saved = self.selection_saved_queries()?;
        options.kept = self.selection_kept_results()?;
        Ok(polylogue_query::plan::prepare(
            text,
            &options.bindings(),
            |condition| self.resolve_question_reference(condition, &options),
        )?)
    }
    /// Evaluate the full semantic row selection, including an atomic native
    /// range. Read-target source grouping is a separate presentation input.
    pub fn prepare_query_selection<'read>(
        &'read self,
        text: &str,
        options: &Options,
    ) -> Result<PreparedSelection<'read>, SelectionError> {
        self.with_clock(&options.clock, || {
            let plan = self.prepare_query_plan(text, options)?;
            let program = crate::query::lower(&plan, options)?;
            self.validate_selection(&program)?;
            self.finish_selection(plan, program, None, options)
        })
    }

    /// Resolve names and references once against this pinned read. Incoming
    /// saved/kept maps are replaced with the authoritative Reader snapshot.
    pub fn prepare_selection<'read>(
        &'read self,
        target: &str,
        options: &Options,
    ) -> Result<PreparedSelection<'read>, SelectionError> {
        self.with_clock(&options.clock, || {
            self.prepare_selection_scoped(target, options)
        })
    }
    fn prepare_selection_scoped<'read>(
        &'read self,
        target: &str,
        options: &Options,
    ) -> Result<PreparedSelection<'read>, SelectionError> {
        self.check()?;
        let scope = self.scope_token();
        scope.check()?;
        let mut options = options.clone();
        options.saved = self.selection_saved_queries()?;
        options.kept = self.selection_kept_results()?;
        let parsed = polylogue_query::parse::with_catalog(target, &options.catalog())?;
        let direct = direct_reference(&parsed)?;
        let direct = direct
            .map(|reference| self.bind_direct_reference(reference))
            .transpose()?;
        if let Some(reference) = &direct
            && !matches!(reference, Reference::Span { .. })
        {
            let unit = reference_unit(reference).expect("readable native reference");
            let mut params = Vec::new();
            let members = membership(unit, reference, &mut params)?
                .ok_or_else(|| invalid("not_built", "reference has no retained membership"))?;
            if self
                .query(&format!("SELECT 1 FROM ({members}) LIMIT 1"), &params)?
                .is_empty()
            {
                return Err(invalid("not_found", "referenced source is unavailable"));
            }
        }
        let text = direct
            .as_ref()
            .map(reference_query)
            .transpose()?
            .unwrap_or_else(|| target.to_owned());
        let mut program = crate::query::compile(&text, &options)?;
        self.validate_selection(&program)?;
        let identity_text = if direct.is_some() {
            format!(
                "{} where {}",
                program.unit.name(),
                reference_atom(direct.as_ref().expect("direct reference"))
            )
        } else {
            target.to_owned()
        };
        let plan =
            polylogue_query::plan::prepare(&identity_text, &options.bindings(), |condition| {
                self.resolve_question_reference(condition, &options)
            })?;
        if direct.is_none() {
            program = crate::query::lower(&plan, &options)?;
        }
        program.query = target.to_owned();
        self.finish_selection(plan, program, direct, &options)
    }

    /// Replay an acknowledged canonical object using its authenticated catalog
    /// version and the caller's already captured operation clock.
    pub fn prepare_query_object_selection<'read>(
        &'read self,
        object: &polylogue_model::query_objects::QueryObject,
        options: &Options,
        expected_catalog: &str,
    ) -> Result<PreparedSelection<'read>, SelectionError> {
        self.with_clock(&options.clock, || {
            self.check()?;
            crate::query_objects::validate_result_bindings(
                self.conn()?,
                crate::source::Source::Cache,
                object,
            )?;
            let plan = polylogue_query::plan::replay(object, expected_catalog)?;
            let program = crate::query::lower(&plan, options)?;
            self.validate_selection(&program)?;
            self.finish_selection(plan, program, None, options)
        })
    }

    fn bind_direct_reference(&self, reference: Reference) -> Result<Reference, SelectionError> {
        if let Reference::Assertion(revision) = &reference {
            if revision.revision.is_none() {
                let rows = self.query(
                    "SELECT id FROM source_note WHERE first_note=?1 ORDER BY id LIMIT 2",
                    &[SqlValue::Text(revision.root.clone())],
                )?;
                match rows.as_slice() {
                    [row] => {
                        return Ok(Reference::Assertion(
                            polylogue_model::RevisionReference::exact(
                                revision.root.clone(),
                                row["id"]
                                    .as_str()
                                    .ok_or_else(|| {
                                        invalid("store", "assertion revision has no identity")
                                    })?
                                    .into(),
                            ),
                        ));
                    }
                    [] => return Err(invalid("not_found", "assertion family is unavailable")),
                    _ => {
                        return Err(invalid(
                            "ambiguous",
                            "assertion family has multiple revisions; name an exact root@revision",
                        ));
                    }
                }
            }
        }
        Ok(reference)
    }

    fn finish_selection<'read>(
        &'read self,
        plan: Plan,
        program: Program,
        direct: Option<Reference>,
        options: &Options,
    ) -> Result<PreparedSelection<'read>, SelectionError> {
        let scope = self.scope_token();
        scope.check()?;
        let position = self.change_position()?;
        let index_file = self.index_identity()?;
        let mut versions = BTreeMap::from([
            ("compiler".into(), "shared-query-v1".into()),
            (
                "catalog".into(),
                polylogue_query::plan::catalog_fingerprint(),
            ),
            ("definitions".into(), plan.question_id().to_string()),
        ]);
        if let Some(scope) = self.query_scope_fingerprint()? {
            versions.insert("scope".into(), scope);
        }
        if program.near.is_some() {
            versions.insert("ranking".into(), crate::vectors::writes().to_string());
        }
        let mut context = CursorContext {
            question: plan.question_id(),
            clock: options.clock.clone(),
            resolved_times: program.resolved_times.clone(),
            random_seed: options.random_seed.unwrap_or(0),
            read_positions: BTreeMap::from([
                ("generation".into(), position.generation),
                ("index".into(), position.index.to_string()),
                ("archive".into(), position.archive.to_string()),
                (
                    "index_file".into(),
                    format!("{}:{}", index_file.0, index_file.1),
                ),
            ]),
            versions,
        };
        crate::query::continuation::compact_resolved_times(&mut context)?;
        scope.check()?;
        Ok(PreparedSelection {
            reader: std::ptr::from_ref(self).cast::<()>() as usize,
            visiting: std::cell::Cell::new(false),
            pinned: std::marker::PhantomData,
            plan,
            program,
            scope,
            context,
            direct,
            execution_options: crate::query::execute::options_snapshot(options, false, None)?,
        })
    }

    /// Decode one row at a time. A callback break stops SQLite immediately;
    /// no previous body or complete answer is materialized to seek a page.
    pub fn visit_selection<E>(
        &self,
        selection: &PreparedSelection<'_>,
        after: Option<&OrderPosition>,
        near: Option<&NearFn<'_>>,
        mut visit: impl FnMut(SelectionRow) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        self.in_selection_context(selection, near, || {
            self.visit_selection_scoped(selection, after, &mut visit)
        })
    }

    fn visit_selection_scoped<E>(
        &self,
        selection: &PreparedSelection<'_>,
        after: Option<&OrderPosition>,
        visit: &mut impl FnMut(SelectionRow) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        let _visit = VisitGuard {
            previous: selection.visiting.replace(true),
            cell: &selection.visiting,
        };
        selection
            .scope
            .check()
            .map_err(SelectionError::from)
            .map_err(E::from)?;
        let current = self.scope_token();
        if !selection.scope.same(&current)
            || selection.reader != std::ptr::from_ref(self).cast::<()>() as usize
        {
            return Err(E::from(invalid(
                "scope_changed",
                "selection belongs to another access scope",
            )));
        }
        self.validate_selection(&selection.program)
            .map_err(E::from)?;
        let program = &selection.program;
        if after.is_some_and(|after| after.values.len() != program.order.len()) {
            return Err(E::from(invalid(
                "invalid_cursor",
                "selection order position has the wrong arity",
            )));
        }
        let mut params = values(&program.params);
        if let Some(after) = after {
            params.extend(after.values.iter().map(sql_value));
        }
        params.push(SqlValue::Int(-1));
        let sql = program.page_sql(after.is_some());
        let result = self.query_each(&sql, &params, |mut row| {
            selection
                .scope
                .check()
                .map_err(SelectionError::from)
                .map_err(VisitError::Selection)?;
            let order = program
                .order
                .iter()
                .map(|key| row[&key.column].clone())
                .collect::<Vec<_>>();
            let reference = selection
                .direct
                .clone()
                .or_else(|| row_reference(program, &row));
            let identity = RowIdentity::from_row(program, &row);
            let first_seen = row["_first_seen"].as_i64();
            let aggregates = aggregate_values(program, &row);
            let row_identity = reference
                .as_ref()
                .map(ToString::to_string)
                .or_else(|| {
                    identity
                        .as_ref()
                        .map(|key| serde_json::to_string(key).expect("row identity is JSON"))
                })
                .unwrap_or_else(|| serde_json::to_string(&order).expect("order position is JSON"));
            crate::query::execute::present(program, &mut row).map_err(VisitError::Selection)?;
            match visit(SelectionRow {
                reference,
                values: row,
                identity,
                first_seen,
                aggregates,
                position: OrderPosition {
                    values: order,
                    row_identity,
                },
            })
            .map_err(VisitError::Consumer)?
            {
                ControlFlow::Break(()) => Err(VisitError::Stop),
                ControlFlow::Continue(()) if selection.direct.is_some() => Err(VisitError::Stop),
                ControlFlow::Continue(()) => Ok(()),
            }
        });
        match result {
            Ok(()) | Err(VisitError::Stop) => Ok(()),
            Err(VisitError::Consumer(e)) => Err(e),
            Err(VisitError::Selection(e)) => Err(E::from(e)),
        }
    }

    fn in_selection_context<T, E>(
        &self,
        selection: &PreparedSelection<'_>,
        near: Option<&NearFn<'_>>,
        run: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<SelectionError>,
    {
        selection
            .scope
            .check()
            .map_err(SelectionError::from)
            .map_err(E::from)?;
        if selection.reader != std::ptr::from_ref(self).cast::<()>() as usize
            || !selection.scope.same(&self.scope_token())
        {
            return Err(E::from(invalid(
                "scope_changed",
                "selection belongs to another access scope",
            )));
        }
        self.validate_selection(&selection.program)
            .map_err(E::from)?;
        self.with_query_context(&selection.program, near, || {
            let result = run();
            selection
                .scope
                .check()
                .map_err(SelectionError::from)
                .map_err(E::from)?;
            result
        })
    }

    /// Stream the complete selected native population without answer projection
    /// or body decoding. Aggregated populations need a retained compiler basis.
    pub fn visit_selection_facts<E>(
        &self,
        selection: &PreparedSelection<'_>,
        near: Option<&NearFn<'_>>,
        mut visit: impl FnMut(SelectionFactKey) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        self.in_selection_context(selection, near, || {
            self.selection_facts_scoped(selection, &mut visit)
        })
    }
    fn selection_facts_scoped<E>(
        &self,
        selection: &PreparedSelection<'_>,
        visit: &mut impl FnMut(SelectionFactKey) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        let RowBasis::Native { unit, columns } = &selection.program.row_basis else {
            return Err(E::from(invalid(
                "not_yet_available",
                "the query does not retain its criterion population",
            )));
        };
        let projection = columns
            .iter()
            .map(|c| format!("q.\"{c}\""))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT {projection} FROM ({}) q ORDER BY q._pk",
            selection.program.sql
        );
        let result = self.query_each(&sql, &values(&selection.program.params), |row| {
            selection
                .scope
                .check()
                .map_err(SelectionError::from)
                .map_err(VisitError::Selection)?;
            let fact = SelectionFactKey {
                unit: *unit,
                values: columns.iter().map(|c| row[c].clone()).collect(),
            };
            match visit(fact).map_err(VisitError::Consumer)? {
                ControlFlow::Continue(()) => Ok(()),
                ControlFlow::Break(()) => Err(VisitError::Stop),
            }
        });
        match result {
            Ok(()) | Err(VisitError::Stop) => Ok(()),
            Err(VisitError::Consumer(e)) => Err(e),
            Err(VisitError::Selection(e)) => Err(E::from(e)),
        }
    }
    /// Membership uses the already-prepared selection, including its stages.
    /// Ranked membership is evaluated inside the visit callback's shared ranking scope.
    pub fn selection_contains(
        &self,
        selection: &PreparedSelection<'_>,
        reference: &Reference,
    ) -> Result<bool, SelectionError> {
        self.with_clock(&selection.program.clock, || {
            self.selection_contains_scoped(selection, reference)
        })
    }
    fn selection_contains_scoped(
        &self,
        selection: &PreparedSelection<'_>,
        reference: &Reference,
    ) -> Result<bool, SelectionError> {
        selection.scope.check()?;
        if selection.reader != std::ptr::from_ref(self).cast::<()>() as usize {
            return Err(invalid(
                "scope_changed",
                "selection belongs to another reader",
            ));
        }
        if selection.program.near.is_some() && !selection.visiting.get() {
            return Err(invalid(
                "embed",
                "ranked membership needs the active selection visit",
            ));
        }
        if selection.program.shape != Shape::Rows {
            return Ok(false);
        }
        if let Some(direct) = &selection.direct {
            if direct == reference {
                return Ok(true);
            }
            if let Reference::Span { container, .. } = direct {
                return self.selection_reference_contains(container, reference);
            }
            if matches!(direct, Reference::MessageRange { .. }) {
                return self.selection_reference_contains(direct, reference);
            }
        }
        let mut params = values(&selection.program.params);
        let Some(predicate) = membership(selection.program.unit, reference, &mut params)? else {
            return Ok(false);
        };
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM ({}) selected WHERE selected._pk IN ({predicate})) AS present",
            selection.program.sql
        );
        Ok(self
            .query(&sql, &params)?
            .first()
            .and_then(|row| row["present"].as_i64())
            == Some(1))
    }
    fn selection_reference_contains(
        &self,
        outer: &Reference,
        inner: &Reference,
    ) -> Result<bool, SelectionError> {
        let mut params = vec![];
        let Some(outer) = membership(Unit::Messages, outer, &mut params)? else {
            return Ok(false);
        };
        let Some(inner) = membership(Unit::Messages, inner, &mut params)? else {
            return Ok(false);
        };
        Ok(self.query(&format!("SELECT EXISTS(SELECT 1 FROM ({outer}) a JOIN ({inner}) b ON a.pk=b.pk) AS present"),&params)?.first().and_then(|row|row["present"].as_i64())==Some(1))
    }
    pub fn validate_selection(&self, program: &Program) -> Result<(), SelectionError> {
        let position = self.change_position()?;
        if program
            .change_generations
            .iter()
            .any(|generation| generation != &position.generation)
        {
            return Err(SelectionError::Invalid {
                kind: "resync_required",
                message: "the index generation changed; read the current state again".into(),
                detail: Some(
                    serde_json::json!({"generation":position.generation,"reason":"generation_changed"}),
                ),
            });
        }
        for check in &program.id_checks {
            let found = self.query(&check.sql, &values(&check.params))?;
            let ids = found
                .iter()
                .filter_map(|row| row["id"].as_str().map(str::to_owned))
                .collect::<Vec<_>>();
            match ids.len() {
                1 => {}
                0 => {
                    return Err(SelectionError::Invalid {
                        kind: "not_found",
                        message: format!(
                            "no session or message has an ID starting {:?}",
                            check.word
                        ),
                        detail: Some(serde_json::json!({"word":check.word})),
                    });
                }
                _ => {
                    return Err(SelectionError::Invalid {
                        kind: "ambiguous",
                        message: format!("{:?} starts several IDs: {}", check.word, ids.join(", ")),
                        detail: Some(serde_json::json!({"word":check.word,"ids":ids})),
                    });
                }
            }
        }
        Ok(())
    }
    fn resolve_question_reference(
        &self,
        condition: &mut ConditionKind,
        options: &Options,
    ) -> Result<(), QueryError> {
        let word = match condition {
            ConditionKind::Reference { reference }
                if matches!(reference.as_str(), "last" | "latest" | "self") =>
            {
                Some(reference.clone())
            }
            ConditionKind::Text {
                text,
                mode: TextMode::Word,
            } if matches!(text.as_str(), "last" | "latest" | "self") => Some(text.clone()),
            _ => None,
        };
        if let Some(word) = word {
            let fail = |message: String| QueryError::Invalid {
                message,
                position: 0,
            };
            let id = if word == "self" {
                options
                    .self_session
                    .clone()
                    .ok_or_else(|| fail("self requires a calling session".into()))?
            } else {
                let rows=self.query("SELECT id FROM session WHERE kind='main' AND stub=0 ORDER BY last_activity_at IS NULL,last_activity_at DESC,pk DESC LIMIT 1",&[]).map_err(|e|fail(e.to_string()))?;
                rows.first()
                    .and_then(|row| row["id"].as_str())
                    .map(str::to_owned)
                    .ok_or_else(|| fail("there is no latest session".into()))?
            };
            *condition = ConditionKind::Reference {
                reference: format!("session:{id}"),
            };
        }
        Ok(())
    }
    pub fn selection_saved_queries(
        &self,
    ) -> Result<BTreeMap<String, Vec<polylogue_query::SavedQuery>>, SelectionError> {
        let sql = format!(
            "SELECT n.key AS name,n.value AS text,n.id,n.first_note FROM source_note n WHERE n.kind='saved_query' AND n.key IS NOT NULL AND n.value IS NOT NULL AND {} ORDER BY n.first_note",
            crate::note_sql::live_note("n")
        );
        let mut saved = BTreeMap::<String, Vec<_>>::new();
        for row in self.query(&sql, &[])? {
            if let (Some(name), Some(text), Some(id), Some(root)) = (
                row["name"].as_str(),
                row["text"].as_str(),
                row["id"].as_str(),
                row["first_note"].as_str(),
            ) {
                saved
                    .entry(name.into())
                    .or_default()
                    .push(polylogue_query::SavedQuery {
                        text: text.into(),
                        revision: id.into(),
                        first_note: root.into(),
                    });
            }
        }
        Ok(saved)
    }
    pub fn selection_kept_results(
        &self,
    ) -> Result<BTreeMap<String, polylogue_query::Kept>, SelectionError> {
        let conn = self.conn()?;
        let values = kept_results(conn, crate::source::Source::Cache)?;
        self.conn()?;
        Ok(values)
    }
}
/// One canonical kept-column decoder serves pinned readers and writer admission.
pub(crate) fn kept_results(
    conn: &rusqlite::Connection,
    source: crate::source::Source,
) -> crate::Result<BTreeMap<String, polylogue_query::Kept>> {
    let sql = source.sql("SELECT id,json_extract(rows_json,'$.unit'),json_extract(rows_json,'$.columns'),json_extract(rows_json,'$.match_columns'),json_type(rows_json,'$.match_columns') FROM @source.kept_result");
    let mut statement = conn.prepare_cached(&sql).map_err(StoreError::from)?;
    let mut rows = statement.query([]).map_err(StoreError::from)?;
    let mut values = BTreeMap::new();
    while let Some(row) = rows.next().map_err(StoreError::from)? {
        let id: String = row.get(0).map_err(StoreError::from)?;
        let unit: String = row.get(1).map_err(StoreError::from)?;
        let columns: String = row.get(2).map_err(StoreError::from)?;
        let columns = serde_json::from_str(&columns).map_err(StoreError::from)?;
        let kind: Option<String> = row.get(4).map_err(StoreError::from)?;
        let match_columns = if kind.is_none() {
            Vec::new()
        } else {
            let text: String = row.get(3).map_err(StoreError::from)?;
            serde_json::from_str(&text).map_err(StoreError::from)?
        };
        values.insert(
            id,
            polylogue_query::Kept {
                unit,
                columns,
                match_columns,
            },
        );
    }
    Ok(values)
}
struct VisitGuard<'a> {
    previous: bool,
    cell: &'a std::cell::Cell<bool>,
}
impl Drop for VisitGuard<'_> {
    fn drop(&mut self) {
        self.cell.set(self.previous);
    }
}
enum VisitError<E> {
    Consumer(E),
    Selection(SelectionError),
    Stop,
}
impl<E> From<StoreError> for VisitError<E> {
    fn from(e: StoreError) -> Self {
        Self::Selection(e.into())
    }
}
pub fn values(params: &[Param]) -> Vec<SqlValue> {
    params
        .iter()
        .map(|p| match p {
            Param::Int(n) => SqlValue::Int(*n),
            Param::Real(n) => SqlValue::Real(*n),
            Param::Text(s) => SqlValue::Text(s.clone()),
        })
        .collect()
}
pub fn sql_value(value: &Value) -> SqlValue {
    match value {
        Value::Null => SqlValue::Null,
        Value::Bool(value) => SqlValue::Int(i64::from(*value)),
        Value::Number(n) => n
            .as_i64()
            .map(SqlValue::Int)
            .unwrap_or_else(|| SqlValue::Real(n.as_f64().unwrap_or_default())),
        Value::String(s) => SqlValue::Text(s.clone()),
        other => SqlValue::Text(other.to_string()),
    }
}

fn reference_atom(reference: &Reference) -> String {
    let text = reference.to_string();
    let (kind, value) = text
        .split_once(':')
        .expect("native reference has a namespace");
    format!("{kind}:{}", polylogue_query::quoted(value))
}
fn direct_reference(
    query: &polylogue_query::ast::Query,
) -> Result<Option<Reference>, SelectionError> {
    use polylogue_query::ast::{Source, Value as Scalar};
    if !query.stages.is_empty() || !query.attach.is_empty() {
        return Ok(None);
    }
    let Source::Unit {
        implicit,
        condition: Some(condition),
        ..
    } = &query.source
    else {
        return Ok(None);
    };
    let reference = match &condition.kind {
        ConditionKind::Reference { reference } => match Reference::parse(reference) {
            Ok(reference) => reference,
            Err(
                polylogue_model::ReferenceError::NoKind(_)
                | polylogue_model::ReferenceError::UnknownKind(_),
            ) => return Ok(None),
            Err(error) => return Err(invalid("invalid_query", error.to_string())),
        },
        ConditionKind::Match {
            field,
            value: Scalar::Scalar { text, .. },
        } if matches!(
            field.as_str(),
            "session"
                | "message"
                | "block"
                | "document"
                | "event"
                | "span"
                | "assertion"
                | "delivery"
        ) =>
        {
            Reference::parse(&format!("{field}:{text}"))
                .map_err(|error| invalid("invalid_query", error.to_string()))?
        }
        _ => return Ok(None),
    };
    if reference_unit(&reference).is_none() {
        return Ok(None);
    }
    if *implicit {
        Ok(Some(reference))
    } else {
        Ok(None)
    }
}
fn reference_query(reference: &Reference) -> Result<String, SelectionError> {
    let unit = reference_unit(reference)
        .ok_or_else(|| invalid("invalid_query", "reference has no readable row"))?;
    Ok(format!(
        "{} where {}",
        unit.name(),
        reference_atom(reference)
    ))
}

fn identity_reference(unit: Unit, values: &[Value]) -> Option<Reference> {
    let text = |at: usize| values.get(at)?.as_str();
    match unit {
        Unit::Sessions => SessionId::parse(text(0)?).ok().map(Reference::Session),
        Unit::Messages => MessageId::parse(text(0)?).ok().map(Reference::Message),
        Unit::Documents => DocumentId::parse(text(0)?).ok().map(Reference::Document),
        Unit::Events => Some(Reference::Event(text(0)?.to_owned())),
        Unit::Notes => Some(Reference::Assertion(
            polylogue_model::RevisionReference::exact(text(1)?.to_owned(), text(0)?.to_owned()),
        )),
        Unit::Deliveries => Some(Reference::Delivery(text(0)?.to_owned())),
        Unit::Blocks | Unit::ToolCalls => Some(Reference::Block {
            message: MessageId::parse(text(0)?).ok()?,
            n: values.get(1)?.as_u64()?,
        }),
        _ => None,
    }
}
fn row_reference(program: &Program, row: &Value) -> Option<Reference> {
    if program.shape != Shape::Rows {
        return None;
    }
    let RowBasis::Native { unit, columns } = &program.row_basis else {
        return None;
    };
    identity_reference(
        *unit,
        &columns
            .iter()
            .map(|name| row[name].clone())
            .collect::<Vec<_>>(),
    )
}

fn aggregate_values(program: &Program, row: &Value) -> BTreeMap<String, Option<f64>> {
    if !matches!(program.shape, Shape::Groups | Shape::Summary) {
        return BTreeMap::new();
    }
    let keys = match &program.row_basis {
        RowBasis::Group { columns } => columns.as_slice(),
        _ => &[],
    };
    program
        .columns
        .iter()
        .filter(|column| {
            matches!(
                column.kind,
                polylogue_query::ColKind::Number
                    | polylogue_query::ColKind::Money
                    | polylogue_query::ColKind::Duration
            ) && !keys.contains(&column.name)
        })
        .map(|column| (column.name.clone(), row[&column.name].as_f64()))
        .collect()
}

fn membership(
    unit: Unit,
    reference: &Reference,
    params: &mut Vec<SqlValue>,
) -> Result<Option<String>, SelectionError> {
    Ok(crate::query::references::membership(
        unit,
        reference,
        &mut |value| {
            params.push(SqlValue::Text(value));
            format!("?{}", params.len())
        },
    ))
}

#[cfg(test)]
mod tests;
