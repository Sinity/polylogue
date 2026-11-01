//! Native saved values are checked against the one original/scratch binder.
//! Provisional objects provide syntax only; final acceptance requires custody.
use super::*;
use crate::{
    primary::Sinex,
    selection::{Plan, ProvisionalQueries, SelectionError},
    user_writes::{Preparation, PreparationContext},
};
use polylogue_model::{query_objects::QueryObject, user::UserRequest};
use polylogue_query::QueryError;
use query_language::identity::QuestionId;

pub(crate) struct SavedPreparation<'a> {
    original: &'a Connection,
    authored: &'a Connection,
    request: &'a UserRequest,
    context: Option<&'a PreparationContext>,
    pending: Option<&'a dyn Sinex>,
    provisional: Option<&'a mut ProvisionalQueries>,
}
impl<'a> SavedPreparation<'a> {
    pub(crate) fn new(
        original: &'a Connection,
        authored: &'a Connection,
        request: &'a UserRequest,
        preparation: Preparation<'a>,
    ) -> Self {
        Self {
            original,
            authored,
            request,
            context: preparation.context,
            pending: preparation.pending,
            provisional: preparation.provisional,
        }
    }
    fn unavailable(&self, reference: impl ToString, reason: impl Into<String>) -> StoreError {
        StoreError::ReferenceUnavailable {
            reference: reference.to_string(),
            reason: reason.into(),
        }
    }
    fn context(&self) -> Result<&PreparationContext> {
        self.context.ok_or_else(|| {
            self.unavailable(
                format!("request:{}", self.request.id),
                "trusted saved-query preparation context is missing",
            )
        })
    }
    fn prepare(&mut self, text: &str) -> Result<Plan> {
        let context = self.context()?;
        let session = match &self.request.authorship.author {
            Author::Agent(Some(session)) => Some(session.to_string()),
            _ => None,
        };
        let plan = crate::selection::prepare_query_plan_on(
            self.original,
            self.authored,
            text,
            context.bindings(),
            session.as_deref(),
            self.pending,
            self.provisional.as_deref(),
        )
        .map_err(SelectionError::into_store_error)?;
        self.admit_plan(&plan)?;
        Ok(plan)
    }
    fn object(&self, hash: QuestionId) -> Result<QueryObject> {
        if let Some(found) = crate::query_objects::query_object_on(
            self.original,
            Source::reading(self.original)?,
            &hash.to_string(),
            self.pending,
        )? {
            return Ok(found.value);
        }
        if let Some(object) = self
            .provisional
            .as_deref()
            .and_then(|queries| queries.object(hash))
        {
            return Ok(object.clone());
        }
        Err(self.unavailable(
            format!("query:{hash}"),
            "retained query object custody is missing",
        ))
    }
    fn admit_plan(&mut self, plan: &Plan) -> Result<()> {
        if let Some(provisional) = self.provisional.as_deref_mut() {
            provisional
                .record(plan)
                .map_err(SelectionError::into_store_error)?;
            return Ok(());
        }
        let object = self.object(plan.question_id())?;
        crate::query_objects::validate_result_bindings(
            self.authored,
            Source::reading(self.authored)?,
            &object,
        )?;
        let prepared = plan.query_object()?;
        if !crate::query_objects::equivalent(&object, &prepared) {
            return Err(StoreError::Conflict(format!(
                "query:{}",
                plan.question_id()
            )));
        }
        Ok(())
    }
    fn retained_plan(&mut self, hash: QuestionId) -> Result<Plan> {
        self.context()?;
        let object = self.object(hash)?;
        let plan =
            polylogue_query::plan::replay(&object, polylogue_query::plan::catalog_fingerprint())?;
        self.admit_plan(&plan)?;
        Ok(plan)
    }
    fn denominator(&mut self, reference: Option<&str>) -> Result<Option<QuestionId>> {
        reference
            .map(|reference| {
                let hash = polylogue_query::plan::question_id_from_reference(reference)?;
                self.retained_plan(hash)?;
                Ok(hash)
            })
            .transpose()
    }
    fn figure_denominator(plan: &Plan, denominator: Option<QuestionId>) -> Result<()> {
        if plan.requires_denominator() && denominator.is_none() {
            return Err(QueryError::Invalid {
                message: "a saved measure or aggregate figure requires a denominator query".into(),
                position: 0,
            }
            .into());
        }
        Ok(())
    }
    pub(crate) fn resolve(&mut self, input: SavedInput<'_>) -> Result<values::SavedQuery> {
        match input {
            SavedInput::Native { text, denominator } => {
                let plan = self.prepare(text)?;
                let denominator = denominator
                    .map(|text| self.prepare(text).map(|plan| plan.question_id()))
                    .transpose()?;
                Self::figure_denominator(&plan, denominator)?;
                Ok(values::SavedQuery {
                    text: text.into(),
                    hash: BlobHash(plan.question_id().0),
                    denominator: denominator.map(|hash| format!("query:{hash}")),
                })
            }
            SavedInput::Canonical { value, changed } => {
                let expected = QuestionId(value.hash.0);
                let plan = if changed {
                    self.prepare(&value.text)?
                } else {
                    self.retained_plan(expected)?
                };
                if plan.question_id() != expected {
                    return Err(invalid(
                        "saved_query hash does not identify its prepared text",
                    ));
                }
                let denominator = self.denominator(value.denominator.as_deref())?;
                Self::figure_denominator(&plan, denominator)?;
                Ok(value.clone())
            }
        }
    }
}
