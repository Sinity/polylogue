//! Prepared questions and streaming rows bound to one Reader and access scope.
pub use crate::query::{Param, Program, RowBasis};
use crate::{ReadScopeToken, Reader, SqlValue, StoreError};
use polylogue_model::read::ReadTarget;
use polylogue_model::{DocumentId, MessageId, Reference, RevisionReference, SessionId};
use polylogue_query::catalog::{reference_unit, target_unit};
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

mod execution;
mod prepare;
pub(crate) use prepare::{ProvisionalQueries, prepare_query_plan_on, saved_queries_on};

mod keyed;
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
    Invalid { kind: &'static str, message: String },
}
impl SelectionError {
    /// Preserve typed query diagnostics and Store control failures at accepting
    /// and metadata boundaries. No string code is reclassified here.
    pub fn into_store_error(self) -> StoreError {
        match self {
            Self::Store(error) => error,
            Self::Query(error) => StoreError::Query(error),
            Self::Language(error) => StoreError::Query(QueryError::Language {
                diagnostic: error.to_string(),
                error: Box::new(error),
            }),
            Self::Invalid { kind, message } => StoreError::Selection {
                code: kind,
                message,
            },
        }
    }
    /// The product failure code: the store's one mapping for a store error,
    /// and the diagnostic's own code otherwise.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Store(error) => error.code(),
            Self::Query(error) => error.kind(),
            Self::Language(error) => error.kind.code(),
            Self::Invalid { kind, .. } => kind,
        }
    }
}
fn invalid(kind: &'static str, message: impl Into<String>) -> SelectionError {
    SelectionError::Invalid {
        kind,
        message: message.into(),
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
    direct: Option<ReadTarget>,
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
    pub fn direct_target(&self) -> Option<&ReadTarget> {
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
    pub fn target(&self) -> Option<ReadTarget> {
        identity_target(self.unit, &self.values)
    }
}
#[derive(Debug, Clone)]
pub struct SelectionRow {
    pub target: Option<ReadTarget>,
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
                    .map(|column| {
                        (
                            program
                                .column_slots
                                .iter()
                                .find_map(|(name, slot)| (slot == column).then(|| name.clone()))
                                .unwrap_or_else(|| column.clone()),
                            row[column].clone(),
                        )
                    })
                    .collect(),
            },
            RowBasis::Summary => Self::Summary,
            RowBasis::Distinct { columns } if !program.ranked => Self::Distinct {
                values: columns
                    .iter()
                    .map(|column| {
                        (
                            program
                                .column_slots
                                .iter()
                                .find_map(|(name, slot)| (slot == column).then(|| name.clone()))
                                .unwrap_or_else(|| column.clone()),
                            row[column].clone(),
                        )
                    })
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
        let conn = self.conn()?;
        let plan = prepare_query_plan_on(
            conn,
            conn,
            text,
            &options.bindings(),
            options.self_session.as_deref(),
            None,
            None,
        );
        self.conn()?;
        plan
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
        let direct = match Reference::parse(target)
            .ok()
            .filter(|reference| reference_unit(reference).is_some())
        {
            Some(reference) => Some(reference),
            None => {
                let parsed = polylogue_query::parse::with_catalog(target, &options.catalog())?;
                direct_reference(&parsed)?
            }
        };
        let direct = direct
            .map(|reference| self.bind_direct_reference(reference))
            .transpose()?;
        if let Some(target) = &direct {
            let mut params = Vec::new();
            // A reference outside its unit's membership names no stored row.
            let present = match membership(target_unit(target), &target.reference(), &mut params)? {
                Some(members) => !self
                    .query(&format!("SELECT 1 FROM ({members}) LIMIT 1"), &params)?
                    .is_empty(),
                None => false,
            };
            if !present {
                return Err(invalid("not_found", "referenced source is unavailable"));
            }
        }
        let text = direct
            .as_ref()
            .map(reference_query)
            .unwrap_or_else(|| target.to_owned());
        let mut program = crate::query::compile(&text, &options)?;
        self.validate_selection(&program)?;
        let identity_text = if let Some(reference) = &direct {
            format!(
                "{} where {}",
                program.unit.name(),
                reference_atom(reference)
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

    /// The one point where a selected reference becomes a read target: an
    /// assertion family binds its current revision, other kinds are refused.
    fn bind_direct_reference(&self, reference: Reference) -> Result<ReadTarget, SelectionError> {
        let reference = match reference {
            Reference::Assertion(revision) if revision.revision.is_none() => {
                let row = self.assertion(&revision)?.ok_or_else(|| {
                    invalid("not_found", "assertion family has no current revision")
                })?;
                Reference::Assertion(RevisionReference {
                    root: row.fact.root,
                    revision: Some(row.fact.id),
                })
            }
            reference => reference,
        };
        ReadTarget::try_from(reference).map_err(|error| invalid("invalid_query", error.to_string()))
    }

    fn finish_selection<'read>(
        &'read self,
        plan: Plan,
        program: Program,
        direct: Option<ReadTarget>,
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
                polylogue_query::plan::catalog_fingerprint().to_owned(),
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
        self.visit_selection_scoped_in_direction(selection, after, false, visit)
    }

    fn visit_selection_scoped_in_direction<E>(
        &self,
        selection: &PreparedSelection<'_>,
        after: Option<&OrderPosition>,
        reverse: bool,
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
        // The enclosing selection context has validated the program once.
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
        let sql = if reverse {
            program.metadata_page_sql(after.is_some(), true)
        } else {
            program.page_sql(after.is_some())
        };
        let result = self.query_each(&sql, &params, |mut row| {
            selection
                .scope
                .check()
                .map_err(SelectionError::from)
                .map_err(VisitError::Selection)?;
            if reverse {
                let mut exact_params = values(&program.params);
                exact_params.extend(program.order.iter().map(|key| sql_value(&row[&key.column])));
                let mut projected = self
                    .query(&program.row_at_sql(), &exact_params)
                    .map_err(SelectionError::from)
                    .map_err(VisitError::Selection)?;
                row = projected.pop().ok_or_else(|| {
                    VisitError::Selection(invalid(
                        "stale_cursor",
                        "the selected row position is unavailable",
                    ))
                })?;
            }
            let order = program
                .order
                .iter()
                .map(|key| row[&key.column].clone())
                .collect::<Vec<_>>();
            let target = selection
                .direct
                .clone()
                .or_else(|| row_target(program, &row));
            let identity = RowIdentity::from_row(program, &row);
            let first_seen = row["_first_seen"].as_i64();
            let aggregates = aggregate_values(program, &row);
            let row_identity = target
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
                target,
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
    /// Visit all recorded native conversation memberships without hydrating
    /// message bodies. The shown message's chosen session is not uniqueness proof.
    pub fn visit_message_sessions(
        &self,
        message: &MessageId,
        mut visit: impl FnMut(SessionId) -> crate::Result<ControlFlow<()>>,
    ) -> crate::Result<()> {
        let conn = self.conn()?;
        let message = message.to_string();
        let (sql, id) = crate::query::references::message_sessions(&message);
        let mut statement = conn.prepare_cached(sql)?;
        let mut rows = statement.query([id])?;
        while let Some(row) = rows.next()? {
            self.conn()?;
            let id: String = row.get(0)?;
            let session = SessionId::parse(&id).map_err(|_| {
                StoreError::Invalid("a message membership has an invalid session identity".into())
            })?;
            if visit(session)?.is_break() {
                break;
            }
        }
        self.conn()?;
        Ok(())
    }

    /// Membership uses the already-prepared selection, including its stages.
    /// Ranked membership is evaluated inside the visit callback's shared ranking scope.
    pub fn selection_contains(
        &self,
        selection: &PreparedSelection<'_>,
        reference: &Reference,
        near: Option<&NearFn<'_>>,
    ) -> Result<bool, SelectionError> {
        if selection.visiting.get() {
            return self.selection_contains_scoped(selection, reference);
        }
        self.in_selection_context(selection, near, || {
            let _visit = VisitGuard {
                previous: selection.visiting.replace(true),
                cell: &selection.visiting,
            };
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
            return Err(invalid(
                "invalid_query",
                "grouped or summarized rows have no member records",
            ));
        }
        if let Some(direct) = &selection.direct {
            let direct = direct.reference();
            if &direct == reference {
                return Ok(true);
            }
            if matches!(direct, Reference::MessageRange { .. }) {
                return self.selection_reference_contains(&direct, reference);
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
        for cutoff in &program.assertion_cutoffs {
            self.prepare_assertion_history(*cutoff)?;
        }
        let position = self.change_position()?;
        if program
            .change_generations
            .iter()
            .any(|generation| generation != &position.generation)
        {
            return Err(SelectionError::Invalid {
                kind: "resync_required",
                message: format!(
                    "the index generation changed to {}; read the current state again",
                    position.generation
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
                    });
                }
                _ => {
                    return Err(SelectionError::Store(StoreError::Ambiguous {
                        reference: check.word.clone(),
                        candidates: ids,
                    }));
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
        prepare::resolve_reference(
            self.conn().map_err(|error| QueryError::Unavailable {
                what: error.to_string(),
            })?,
            condition,
            options.self_session.as_deref(),
        )
    }

    pub(crate) fn selection_saved_queries_for_family(
        &self,
        root: &str,
    ) -> Result<
        (
            BTreeMap<String, usize>,
            BTreeMap<String, Vec<polylogue_query::SavedQuery>>,
        ),
        SelectionError,
    > {
        let values = prepare::saved_queries_for_family(self.conn()?, root)?;
        self.conn()?;
        Ok(values)
    }

    pub fn selection_saved_queries(
        &self,
    ) -> Result<BTreeMap<String, Vec<polylogue_query::SavedQuery>>, SelectionError> {
        let conn = self.conn()?;
        let values = saved_queries_on(conn, conn, None, None);
        self.conn()?;
        values
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
    let sql = source.sql(&kept_sql("SELECT id,rows_json FROM @source.kept_result"));
    let mut statement = conn.prepare_cached(&sql)?;
    decode_kept(statement.query([])?)
}

/// Refuses a kept body that [`kept_results`] cannot decode, so a stored kept
/// result can always be read back by every later preparation.
pub(crate) fn check_kept_body(
    conn: &rusqlite::Connection,
    id: &str,
    rows_json: &str,
) -> crate::Result<()> {
    let mut statement = conn.prepare_cached(&kept_sql("SELECT ?1 AS id,?2 AS rows_json"))?;
    match decode_kept(statement.query(rusqlite::params![id, rows_json])?) {
        Ok(_) => Ok(()),
        Err(StoreError::Json(error)) => Err(StoreError::Invalid(error.to_string())),
        Err(error) => Err(error),
    }
}

/// The kept-column query over `stored`, which yields `id` and `rows_json`.
fn kept_sql(stored: &str) -> String {
    format!(
        "WITH stored AS ({stored}), kept AS (SELECT id,json_valid(rows_json) AS valid,CASE WHEN json_valid(rows_json) THEN rows_json ELSE '{{}}' END AS body FROM stored) SELECT id,json_extract(body,'$.unit'),json_extract(body,'$.columns'),json_extract(body,'$.match_columns'),json_type(body,'$.match_columns'),valid,json_extract(body,'$.shape'),json_type(body,'$.identities'),json_extract(body,'$.output_metadata') FROM kept"
    )
}

fn decode_kept(
    mut rows: rusqlite::Rows<'_>,
) -> crate::Result<BTreeMap<String, polylogue_query::Kept>> {
    let mut values = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let damaged = |field: &str, message: &dyn std::fmt::Display| {
            StoreError::Json(<serde_json::Error as serde::de::Error>::custom(format!(
                "kept result {id:?} field {field}: {message}"
            )))
        };
        if !row.get::<_, bool>(5)? {
            return Err(damaged("rows_json", &"invalid JSON"));
        }
        let unit: String = row.get(1).map_err(|error| damaged("unit", &error))?;
        let columns: String = row.get(2).map_err(|error| damaged("columns", &error))?;
        let columns = serde_json::from_str(&columns).map_err(|error| damaged("columns", &error))?;
        let kind: Option<String> = row
            .get(4)
            .map_err(|error| damaged("match_columns", &error))?;
        let match_columns = if kind.is_none() {
            Vec::new()
        } else {
            let text: String = row
                .get(3)
                .map_err(|error| damaged("match_columns", &error))?;
            serde_json::from_str(&text).map_err(|error| damaged("match_columns", &error))?
        };
        let shape: Option<String> = row.get(6).map_err(|error| damaged("shape", &error))?;
        let shape = shape
            .map(|value| serde_json::from_value(serde_json::Value::String(value)))
            .transpose()
            .map_err(|error| damaged("shape", &error))?;
        let identity_kind: Option<String> =
            row.get(7).map_err(|error| damaged("identities", &error))?;
        let native_identity = matches!(shape, Some(polylogue_query::Shape::Rows))
            && identity_kind.as_deref() == Some("array");
        let metadata: String = row
            .get(8)
            .map_err(|error| damaged("output_metadata", &error))?;
        let output_metadata =
            serde_json::from_str(&metadata).map_err(|error| damaged("output_metadata", &error))?;
        let kept = polylogue_query::Kept {
            output_metadata,
            shape,
            native_identity,
            unit,
            columns,
            match_columns,
        };
        kept.validate_output_metadata()
            .map_err(|error| damaged("output_metadata", &error))?;
        values.insert(id, kept);
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

fn reference_atom(target: &ReadTarget) -> String {
    let text = target.to_string();
    let (kind, value) = text
        .split_once(':')
        .expect("native reference has a namespace");
    format!("{kind}:{}", polylogue_query::quoted(value))
}
fn direct_reference(
    query: &polylogue_query::ast::Query,
) -> Result<Option<Reference>, SelectionError> {
    use polylogue_query::ast::{Source, Value as Scalar};
    if !query.transforms.is_empty() || !query.attach.is_empty() {
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
fn reference_query(target: &ReadTarget) -> String {
    format!(
        "{} where {}",
        target_unit(target).name(),
        reference_atom(target)
    )
}

fn identity_target(unit: Unit, values: &[Value]) -> Option<ReadTarget> {
    let text = |at: usize| values.get(at)?.as_str();
    match unit {
        Unit::Sessions => SessionId::parse(text(0)?).ok().map(ReadTarget::Session),
        Unit::Messages => MessageId::parse(text(0)?).ok().map(ReadTarget::Message),
        Unit::Documents => DocumentId::parse(text(0)?).ok().map(ReadTarget::Document),
        Unit::Events => Some(ReadTarget::Event(text(0)?.to_owned())),
        Unit::Assertions => Some(ReadTarget::Assertion {
            root: text(1)?.to_owned(),
            revision: text(0)?.to_owned(),
        }),
        Unit::Deliveries => Some(ReadTarget::Delivery(text(0)?.to_owned())),
        Unit::Blocks | Unit::ToolCalls => Some(ReadTarget::Block {
            message: MessageId::parse(text(0)?).ok()?,
            n: values.get(1)?.as_u64()?,
        }),
        _ => None,
    }
}
fn row_target(program: &Program, row: &Value) -> Option<ReadTarget> {
    if program.shape != Shape::Rows {
        return None;
    }
    let RowBasis::Native { unit, columns } = &program.row_basis else {
        return None;
    };
    identity_target(
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
                    | polylogue_query::ColKind::DurationSeconds
            ) && !keys
                .iter()
                .any(|key| key == program.column_slot(&column.name))
        })
        .map(|column| {
            (
                column.name.clone(),
                row[program.column_slot(&column.name)].as_f64(),
            )
        })
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
