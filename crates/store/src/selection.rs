//! Prepared questions and streaming rows bound to one Reader and access scope.
pub use crate::query::{Param, Program, RowKind};
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
pub(crate) use prepare::{ProvisionalQueries, prepare_query_plan_on};
mod shipped;
pub use shipped::shipped_bindings;

mod keyed;
use std::{collections::BTreeMap, ops::ControlFlow, sync::Arc};

/// A current annotation family's names and resolved query definitions.
pub(crate) struct SavedFamily {
    pub names: BTreeMap<String, usize>,
    pub queries: BTreeMap<String, Vec<polylogue_query::SavedQuery>>,
}

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
        kind: polylogue_model::Code,
        message: String,
    },
}
impl SelectionError {
    pub(crate) fn invalid(kind: polylogue_model::Code, message: impl Into<String>) -> Self {
        Self::Invalid {
            kind,
            message: message.into(),
        }
    }

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
    pub fn code(&self) -> polylogue_model::Code {
        match self {
            Self::Store(error) => error.code(),
            Self::Query(error) => error.kind(),
            Self::Language(error) => error.kind.into(),
            Self::Invalid { kind, .. } => *kind,
        }
    }
}

/// Provider work is injected; preparing or visiting never selects a provider.
pub type NearFn<'a> = dyn Fn(&str) -> Result<Arc<crate::vectors::Similarity>, StoreError> + 'a;

pub struct PreparedSelection<'read> {
    reader: usize,
    visiting: std::cell::Cell<bool>,
    pinned: std::marker::PhantomData<&'read ()>,
    plan: Plan,
    program: Program,
    scope: ReadScopeToken,
    context: CursorContext,
    direct: Option<ReadTarget>,
    fixed: Option<Vec<ReadTarget>>,
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
    /// Resolve the ranking before reading metadata and records.
    pub fn ranking(
        &mut self,
        reader: &Reader<'_>,
        near: Option<&NearFn<'_>>,
    ) -> Result<Option<Arc<crate::vectors::Similarity>>, SelectionError> {
        self.program.ranking(reader, near)
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
#[derive(Debug, Clone)]
pub struct SelectionRow {
    pub target: Option<ReadTarget>,
    pub values: Value,
    pub position: OrderPosition,
    pub identity: Option<RowIdentity>,
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
        Some(match &program.row_kind {
            RowKind::UnitRows { unit, columns } => Self::Entity {
                unit: *unit,
                key: columns.iter().map(|column| row[column].clone()).collect(),
            },
            RowKind::Group { columns } => Self::Group {
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
            RowKind::Summary => Self::Summary,
            RowKind::Distinct { columns } if !program.ranked => Self::Distinct {
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
    /// Execution options and the captured clock stay outside the query's identity.
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
        );
        self.conn()?;
        plan
    }
    /// Evaluate the full row selection, including an atomic unit-row
    /// range. Read-target source grouping is a separate presentation input.
    pub fn prepare_query_selection<'read>(
        &'read self,
        text: &str,
        options: &Options,
    ) -> Result<PreparedSelection<'read>, SelectionError> {
        self.with_clock(&options.clock, || {
            let plan = self.prepare_query_plan(text, options)?;
            let program = self.lower_plan(&plan, options)?;
            self.validate_selection(&program)?;
            self.finish_selection(plan, program, None, options)
        })
    }

    /// Resolve names and references once against this pinned read. Incoming
    /// saved/kept maps are replaced with the Reader snapshot.
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
        if matches!(Reference::parse(target), Ok(Reference::Run(_))) {
            return self.prepare_query_selection(target, options);
        }
        self.check()?;
        let scope = self.scope_token();
        scope.check()?;
        let mut options = options.clone();
        let named = self.selection_definitions(target, &options.origins)?;
        options.saved = named.saved;
        options.kept = named.kept;
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
                return Err(SelectionError::invalid(
                    polylogue_model::Code::NotFound,
                    "referenced source is unavailable",
                ));
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

    /// A line's already accepted population, in its original order and grain.
    pub fn prepare_fixed_selection<'read>(
        &'read self,
        text: &str,
        references: &[Reference],
        options: &Options,
    ) -> Result<PreparedSelection<'read>, SelectionError> {
        let mut selected = match references {
            [reference] => self.prepare_selection(&reference.to_string(), options)?,
            _ => self.prepare_query_selection(text, options)?,
        };
        selected.fixed = Some(
            references
                .iter()
                .cloned()
                .map(|reference| self.bind_direct_reference(reference))
                .collect::<Result<_, _>>()?,
        );
        Ok(selected)
    }

    /// The one point where a selected reference becomes a read target: an
    /// annotation family binds its current revision, other kinds are refused.
    fn bind_direct_reference(&self, reference: Reference) -> Result<ReadTarget, SelectionError> {
        let reference = match reference {
            Reference::Annotation(revision) if revision.revision.is_none() => {
                let row = self.annotation(&revision)?.ok_or_else(|| {
                    SelectionError::invalid(
                        polylogue_model::Code::NotFound,
                        "annotation family has no current revision",
                    )
                })?;
                Reference::Annotation(RevisionReference {
                    root: row.fact.root,
                    revision: Some(row.fact.id),
                })
            }
            reference => reference,
        };
        ReadTarget::try_from(reference).map_err(|error| {
            SelectionError::invalid(polylogue_model::Code::InvalidQuery, error.to_string())
        })
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
            fixed: None,
            execution_options: crate::query::execute::options_snapshot(options, false, None)?,
        })
    }

    /// Decode one row at a time. A callback break stops SQLite immediately;
    /// no previous body or complete answer is built to seek a page.
    pub fn visit_selection<E>(
        &self,
        selection: &PreparedSelection<'_>,
        from: crate::read::VisitFrom<&OrderPosition>,
        near: Option<&NearFn<'_>>,
        mut visit: impl FnMut(SelectionRow) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        if let Some(fixed) = &selection.fixed {
            selection
                .scope
                .check()
                .map_err(SelectionError::from)
                .map_err(E::from)?;
            if selection.reader != std::ptr::from_ref(self).cast::<()>() as usize {
                return Err(E::from(SelectionError::invalid(
                    polylogue_model::Code::ScopeChanged,
                    "selection belongs to another reader",
                )));
            }
            use crate::read::VisitFrom;
            let position = match &from {
                VisitFrom::At(at) | VisitFrom::After(at) | VisitFrom::Before(at) => {
                    let index = at
                        .values
                        .first()
                        .and_then(Value::as_u64)
                        .and_then(|index| usize::try_from(index).ok())
                        .ok_or_else(|| {
                            E::from(SelectionError::invalid(
                                polylogue_model::Code::InvalidQuery,
                                "fixed population cursor has no row position",
                            ))
                        })?;
                    if fixed
                        .get(index)
                        .is_none_or(|target| target.reference().to_string() != at.row_identity)
                    {
                        return Err(E::from(SelectionError::invalid(
                            polylogue_model::Code::StaleCursor,
                            "fixed population cursor no longer names its row",
                        )));
                    }
                    Some(index)
                }
                _ => None,
            };
            let reverse = matches!(&from, VisitFrom::Before(_) | VisitFrom::Last);
            let rows = fixed.iter().enumerate().filter(|(index, _)| match &from {
                VisitFrom::At(_) => *index >= position.unwrap(),
                VisitFrom::After(_) => *index > position.unwrap(),
                VisitFrom::Before(_) => *index < position.unwrap(),
                _ => true,
            });
            let mut rows: Box<dyn Iterator<Item = (usize, &ReadTarget)> + '_> = if reverse {
                Box::new(rows.rev())
            } else {
                Box::new(rows)
            };
            for (index, target) in &mut rows {
                self.check()
                    .map_err(SelectionError::from)
                    .map_err(E::from)?;
                selection
                    .scope
                    .check()
                    .map_err(SelectionError::from)
                    .map_err(E::from)?;
                if visit(SelectionRow {
                    target: Some(target.clone()),
                    values: Value::Null,
                    identity: None,
                    aggregates: BTreeMap::new(),
                    position: OrderPosition {
                        values: vec![serde_json::json!(index)],
                        row_identity: target.reference().to_string(),
                    },
                })?
                .is_break()
                {
                    break;
                }
            }
            return Ok(());
        }
        self.in_selection_context(selection, near, || {
            use crate::read::VisitFrom;
            let (after, reverse) = match from {
                VisitFrom::First => (None, false),
                VisitFrom::At(at) if selection.direct.is_none() => {
                    (self.row_before(selection, at).map_err(E::from)?, false)
                }
                VisitFrom::At(_) => (None, false),
                VisitFrom::After(after) => (Some(after.values.clone()), false),
                VisitFrom::Before(_) if selection.direct.is_some() => return Ok(()),
                VisitFrom::Before(before) => (Some(before.values.clone()), true),
                VisitFrom::Last => (None, true),
            };
            self.visit_selection_scoped_in_direction(
                selection,
                after.as_deref(),
                reverse,
                &mut visit,
            )
        })
    }

    fn visit_selection_scoped_in_direction<E>(
        &self,
        selection: &PreparedSelection<'_>,
        after: Option<&[serde_json::Value]>,
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
            return Err(E::from(SelectionError::invalid(
                polylogue_model::Code::ScopeChanged,
                "selection belongs to another access scope",
            )));
        }
        // The enclosing selection context has validated the program once.
        let program = &selection.program;
        self.visit_page(program, after, reverse, -1, |mut row| {
            selection
                .scope
                .check()
                .map_err(SelectionError::from)
                .map_err(E::from)?;
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
            let aggregates = aggregate_values(program, &row);
            let row_identity = target
                .as_ref()
                .map(|target| target.reference().to_string())
                .or_else(|| {
                    identity
                        .as_ref()
                        .map(|key| serde_json::to_string(key).expect("row identity is JSON"))
                })
                .unwrap_or_else(|| serde_json::to_string(&order).expect("order position is JSON"));
            crate::query::execute::present(program, &mut row).map_err(E::from)?;
            let flow = visit(SelectionRow {
                target,
                values: row,
                identity,
                aggregates,
                position: OrderPosition {
                    values: order,
                    row_identity,
                },
            })?;
            // A direct reference selects its one row.
            Ok(if selection.direct.is_some() {
                ControlFlow::Break(())
            } else {
                flow
            })
        })
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
            return Err(E::from(SelectionError::invalid(
                polylogue_model::Code::ScopeChanged,
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

    /// Visit all recorded conversation memberships without hydrating
    /// message bodies. The shown message's chosen session does not show it is the only one.
    pub fn visit_message_sessions(
        &self,
        message: &MessageId,
        mut visit: impl FnMut(SessionId) -> crate::Result<ControlFlow<()>>,
    ) -> crate::Result<()> {
        let conn = self.conn()?;
        let message = crate::chain_names::resolved_message(conn, message)?.to_string();
        let (sql, id) = crate::query::references::message_sessions(&message);
        let mut statement = conn.prepare_cached(sql)?;
        let mut rows = statement.query([id])?;
        while let Some(row) = rows.next()? {
            self.conn()?;
            let id: String = row.get(0)?;
            let session = SessionId::parse(&id).map_err(|_| {
                StoreError::internal("a message membership has an invalid session identity")
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
        if selection.visiting.get() || selection.fixed.is_some() {
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
            return Err(SelectionError::invalid(
                polylogue_model::Code::ScopeChanged,
                "selection belongs to another reader",
            ));
        }
        if let Some(fixed) = &selection.fixed {
            for target in fixed {
                let outer = target.reference();
                if &outer == reference || self.selection_reference_contains(&outer, reference)? {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        if selection.program.near.is_some() && !selection.visiting.get() {
            return Err(SelectionError::invalid(
                polylogue_model::Code::Embed,
                "ranked membership needs the active selection visit",
            ));
        }
        if selection.program.shape != Shape::Rows {
            return Err(SelectionError::invalid(
                polylogue_model::Code::InvalidQuery,
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
            selection
                .program
                .statements
                .members(selection.program.counts_near)
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
        for cutoff in &program.annotation_cutoffs {
            self.prepare_annotation_history(*cutoff)?;
        }
        let position = self.change_position()?;
        if program
            .change_generations
            .iter()
            .any(|generation| generation != &position.generation)
        {
            return Err(SelectionError::Invalid {
                kind: polylogue_model::Code::ResyncRequired,
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
                // A field condition on an ID no session has is a condition no
                // row meets: the query answers empty, and a waiting listing
                // keeps waiting for the session.
                0 if !check.reference => {}
                0 => {
                    return Err(SelectionError::Invalid {
                        kind: polylogue_model::Code::NotFound,
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
        prepare::resolve_reference(condition, options.self_session.as_deref())
    }

    pub(crate) fn selection_saved_queries_for_family(
        &self,
        root: &str,
    ) -> Result<SavedFamily, SelectionError> {
        let values = prepare::saved_queries_for_family(self.conn()?, root)?;
        self.conn()?;
        Ok(values)
    }

    /// The saved queries and kept results `text` names, read from this
    /// pinned snapshot, and nothing else (see [`prepare::named_definitions`]).
    pub fn selection_definitions(
        &self,
        text: &str,
        origins: &[String],
    ) -> Result<polylogue_query::Bindings, SelectionError> {
        let conn = self.conn()?;
        let values = prepare::named_definitions(conn, conn, text, origins, None);
        self.conn()?;
        values
    }

    /// Lower a plan against the kept results it reads as this snapshot holds
    /// them: lowering checks each against the plan's bound columns.
    pub fn lower_plan(&self, plan: &Plan, options: &Options) -> Result<Program, SelectionError> {
        let conn = self.conn()?;
        let mut options = options.clone();
        options.kept = kept_results_named(
            conn,
            crate::source::Source::Cache,
            plan.bindings().kept.keys().map(String::as_str),
        )?;
        self.conn()?;
        Ok(crate::query::lower(plan, &options)?)
    }
}
/// The kept results with these ids, through the one kept-column decoder that
/// pinned readers and the writer share; an id with no stored result is
/// absent from the map.
pub(crate) fn kept_results_named<'a>(
    conn: &rusqlite::Connection,
    source: crate::source::Source,
    ids: impl IntoIterator<Item = &'a str>,
) -> crate::Result<BTreeMap<String, polylogue_query::Kept>> {
    let ids: Vec<&str> = ids.into_iter().collect();
    if ids.is_empty() {
        return Ok(BTreeMap::new());
    }
    let sql = source.sql(&kept_sql(
        "SELECT id,rows_json FROM @source.kept_result WHERE id IN (SELECT value FROM json_each(?1))",
    ));
    let mut statement = conn.prepare_cached(&sql)?;
    decode_kept(statement.query([serde_json::to_string(&ids)?])?).map_err(|error| match error {
        StoreError::Refused(reason) => StoreError::stored_data(reason),
        error => error,
    })
}

/// Refuses a kept body that [`kept_results_named`] cannot decode, so a stored kept
/// result can always be read back by every later preparation.
pub(crate) fn check_kept_body(
    conn: &rusqlite::Connection,
    id: &str,
    rows_json: &str,
) -> crate::Result<()> {
    let mut statement = conn.prepare_cached(&kept_sql("SELECT ?1 AS id,?2 AS rows_json"))?;
    decode_kept(statement.query(rusqlite::params![id, rows_json])?).map(drop)
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
        // A body that does not decode is refused at write; read back, the
        // reader reports it as stored data.
        let damaged = |field: &str, message: &dyn std::fmt::Display| {
            StoreError::from(polylogue_model::Reason::BadKeptResult {
                id: id.clone(),
                field: field.into(),
                problem: message.to_string(),
            })
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
        let row_identity = matches!(shape, Some(polylogue_query::Shape::Rows))
            && identity_kind.as_deref() == Some("array");
        let metadata: Option<String> = row
            .get(8)
            .map_err(|error| damaged("output_metadata", &error))?;
        let output_metadata = metadata
            .map(|metadata| serde_json::from_str(&metadata))
            .transpose()
            .map_err(|error| damaged("output_metadata", &error))?
            .unwrap_or_default();
        let kept = polylogue_query::Kept {
            output_metadata,
            shape,
            row_identity,
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
        .expect("a typed reference has a namespace");
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
            Err(error) => {
                return Err(SelectionError::invalid(
                    polylogue_model::Code::InvalidQuery,
                    error.to_string(),
                ));
            }
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
                | "annotation"
                | "delivery"
        ) =>
        {
            Reference::parse(&format!("{field}:{text}")).map_err(|error| {
                SelectionError::invalid(polylogue_model::Code::InvalidQuery, error.to_string())
            })?
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
        Unit::Annotations => Some(ReadTarget::Annotation {
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
/// The native identity carried by a raw query row before presentation.
pub fn row_reference(program: &Program, row: &Value) -> Option<Reference> {
    row_target(program, row).map(|target| target.reference())
}

fn row_target(program: &Program, row: &Value) -> Option<ReadTarget> {
    if program.shape != Shape::Rows {
        return None;
    }
    let RowKind::UnitRows { unit, columns } = &program.row_kind else {
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
    let keys = match &program.row_kind {
        RowKind::Group { columns } => columns.as_slice(),
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
