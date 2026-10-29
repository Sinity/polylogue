//! Prepared questions and streaming rows bound to one Reader and access scope.
use crate::{ReadScopeToken, Reader, SqlValue, StoreError};
use polylogue_model::{DocumentId, MessageId, Reference, SessionId};
use polylogue_query::{
    Options, Param, QueryError, Shape,
    ast::{ConditionKind, TextMode, Unit},
};
pub use polylogue_query::{
    Program,
    plan::{Plan, ResolvedDefinition},
};
use query_language::{
    bind::BoundQuery,
    cursor::{CursorContext, OrderPosition},
    identity::QuestionId,
};
use serde_json::Value;
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
            Self::Store(_) => "store",
            Self::Query(error) => error.kind(),
            Self::Language(error) => error.kind.code(),
            Self::Invalid { kind, .. } => kind,
        }
    }
    pub fn detail(&self) -> Option<Value> {
        match self {
            Self::Query(error) => serde_json::to_value(error).ok(),
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
    pinned: std::marker::PhantomData<&'read ()>,
    plan: Plan,
    program: Program,
    scope: ReadScopeToken,
    context: CursorContext,
    direct: Option<Reference>,
}
impl PreparedSelection<'_> {
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
#[derive(Debug, Clone)]
pub struct SelectionRow {
    pub reference: Option<Reference>,
    pub values: Value,
    pub position: OrderPosition,
}

impl Reader<'_> {
    /// Resolve names and references once against this pinned read. Incoming
    /// saved/kept maps are replaced with the authoritative Reader snapshot.
    pub fn prepare_selection<'read>(
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
        let direct_syntax = polylogue_query::parse::with_catalog(target, &options.catalog())
            .is_ok_and(|query| {
                query.stages.is_empty()
                    && query.attach.is_empty()
                    && matches!(
                        query.source,
                        polylogue_query::ast::Source::Unit {
                            condition: Some(polylogue_query::ast::Condition {
                                kind: ConditionKind::Match { .. } | ConditionKind::Reference { .. },
                                ..
                            }),
                            ..
                        }
                    )
            });
        let direct = direct_syntax
            .then(|| Reference::parse(target))
            .transpose()
            .ok()
            .flatten()
            .filter(|reference| {
                matches!(
                    reference,
                    Reference::Session(_)
                        | Reference::Message(_)
                        | Reference::MessageRange { .. }
                        | Reference::Block { .. }
                        | Reference::Document(_)
                        | Reference::Event(_)
                        | Reference::Span { .. }
                )
            });
        let text = direct
            .as_ref()
            .map(reference_query)
            .transpose()?
            .unwrap_or_else(|| target.to_owned());
        let mut program = polylogue_query::compile(&text, &options)?;
        self.validate_selection(&program)?;
        let identity_text = if direct.is_some() {
            format!("{} {}", program.unit.name(), target)
        } else {
            target.to_owned()
        };
        let plan = polylogue_query::plan::prepare(&identity_text, &options, |condition| {
            self.resolve_question_reference(condition, &options)
        })?;
        program.query = target.to_owned();
        let position = self.change_position()?;
        let index_file = self.index_identity()?;
        let mut versions = BTreeMap::from([
            ("compiler".into(), "shared-query-v1".into()),
            ("definitions".into(), digest(&plan.definitions())?),
        ]);
        if program.near.is_some() {
            versions.insert("ranking".into(), crate::vectors::writes().to_string());
        }
        let context = CursorContext {
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
        scope.check()?;
        Ok(PreparedSelection {
            reader: std::ptr::from_ref(self).cast::<()>() as usize,
            pinned: std::marker::PhantomData,
            plan,
            program,
            scope,
            context,
            direct,
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
        let mut run = || {
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
                let row_identity = reference
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| serde_json::to_string(&order).unwrap_or_default());
                if let Some(object) = row.as_object_mut() {
                    object.retain(|key, _| !key.starts_with('_'));
                }
                match visit(SelectionRow {
                    reference,
                    values: row,
                    position: OrderPosition {
                        values: order,
                        row_identity,
                    },
                })
                .map_err(VisitError::Consumer)?
                {
                    ControlFlow::Break(()) => Err(VisitError::Stop),
                    ControlFlow::Continue(()) if selection.direct.is_some() => {
                        Err(VisitError::Stop)
                    }
                    ControlFlow::Continue(()) => Ok(()),
                }
            });
            match result {
                Ok(()) | Err(VisitError::Stop) => Ok(()),
                Err(VisitError::Consumer(e)) => Err(e),
                Err(VisitError::Selection(e)) => Err(E::from(e)),
            }
        };
        if let Some(request) = &selection.program.near {
            let near =
                near.ok_or_else(|| E::from(invalid("embed", "semantic search unavailable")))?;
            let similarity = near(&request.text).map_err(|e| E::from(invalid("embed", e)))?;
            self.with_similarity(similarity, run)
        } else {
            run()
        }
    }
    pub fn validate_selection(&self, program: &Program) -> Result<(), SelectionError> {
        let position = self.change_position()?;
        if program
            .change_generations
            .iter()
            .any(|generation| generation != &position.generation)
        {
            return Err(invalid("resync_required", "the index generation changed"));
        }
        for check in &program.id_checks {
            let found = self.query(&check.sql, &values(&check.params))?;
            match found.len() {
                1 => {}
                0 => {
                    return Err(invalid(
                        "not_found",
                        format!("no session or message starts {:?}", check.word),
                    ));
                }
                _ => {
                    return Err(SelectionError::Invalid {
                        kind: "ambiguous",
                        message: format!("several sessions or messages start {:?}", check.word),
                        detail: Some(serde_json::json!({"matches":found})),
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
            polylogue_model::note_sql::live_note("n")
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
        self.query("SELECT id,rows_json FROM source_kept_result", &[])?
            .into_iter()
            .map(|row| {
                let id = row["id"]
                    .as_str()
                    .ok_or_else(|| invalid("store", "kept result has no ID"))?
                    .to_owned();
                let text = row["rows_json"]
                    .as_str()
                    .ok_or_else(|| invalid("store", "kept result has no rows"))?;
                let kept = serde_json::from_str(text).map_err(StoreError::from)?;
                Ok((id, kept))
            })
            .collect()
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
fn digest(value: &impl serde::Serialize) -> Result<String, SelectionError> {
    let bytes = serde_json::to_vec(value).map_err(StoreError::from)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
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
fn sql_value(value: &Value) -> SqlValue {
    match value {
        Value::Null => SqlValue::Null,
        Value::Number(n) => n
            .as_i64()
            .map(SqlValue::Int)
            .unwrap_or_else(|| SqlValue::Real(n.as_f64().unwrap_or_default())),
        Value::String(s) => SqlValue::Text(s.clone()),
        other => SqlValue::Text(other.to_string()),
    }
}
fn reference_query(reference: &Reference) -> Result<String, SelectionError> {
    Ok(match reference {
        Reference::Session(id) => format!(
            "sessions session:{}",
            polylogue_query::quoted(&id.to_string())
        ),
        Reference::Message(id)
        | Reference::Block { message: id, .. }
        | Reference::MessageRange { first: id, .. } => format!(
            "messages message:{}",
            polylogue_query::quoted(&id.to_string())
        ),
        Reference::Document(id) => {
            format!("documents id:{}", polylogue_query::quoted(&id.to_string()))
        }
        Reference::Event(id) => format!("events id:{}", polylogue_query::quoted(id)),
        Reference::Span { container, .. } => reference_query(container)?,
        _ => return Err(invalid("invalid_query", "reference has no readable row")),
    })
}
fn row_reference(program: &Program, row: &Value) -> Option<Reference> {
    if program.shape != Shape::Rows {
        return None;
    }
    let string = |key: &str| row[key].as_str();
    match program.unit {
        Unit::Sessions => SessionId::parse(string("id")?).ok().map(Reference::Session),
        Unit::Messages => MessageId::parse(string("id")?).ok().map(Reference::Message),
        Unit::Documents => DocumentId::parse(string("id")?)
            .ok()
            .map(Reference::Document),
        Unit::Events => Some(Reference::Event(string("id")?.into())),
        Unit::Blocks => Some(Reference::Block {
            message: MessageId::parse(string("message")?).ok()?,
            n: row["idx"].as_u64()?,
        }),
        _ => None,
    }
}
#[cfg(test)]
mod tests;
