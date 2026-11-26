//! A saved query's objects are stored before the assertion that names them.
//! The private preflight orders the proposals and writes nothing else.
use super::*;
use polylogue_model::{
    annotation::AnnotationKind,
    user::{UserRequest, UserWrite},
};

impl Inner {
    pub(super) fn preflight_saved(
        &mut self,
        request: &UserRequest,
        preparation: Option<&crate::user_writes::PreparationContext>,
        scope: Option<&crate::ReadScopeToken>,
    ) -> Result<Option<crate::ReadScopeToken>> {
        let needed = match &request.write {
            UserWrite::ReviseAnnotations { .. } => true,
            UserWrite::Save(save) => save.name.is_some(),
            UserWrite::Assert { annotations } => annotations
                .iter()
                .any(|statement| statement.kind == AnnotationKind::SavedQuery),
            UserWrite::AssertQuery { annotation, .. } => {
                annotation.kind == AnnotationKind::SavedQuery
            }
            _ => false,
        };
        if !needed {
            return Ok(None);
        }
        if crate::authored_records::known_request(&self.index, request)?.is_some() {
            return Ok(None);
        }
        let selected = match scope {
            Some(scope) => scope.clone(),
            None => self.location.scope_token()?,
        };
        scoped::require_current(&self.location, Some(&selected))?;
        let mut provisional = crate::selection::ProvisionalQueries::default();
        let prepared = crate::authored_records::with_overlay(&self.index, |authored| {
            crate::user_writes::apply_resolving(
                authored,
                &self.index,
                request,
                // Copied and pending baseline records use zero. Keep the
                // proposal separate when capturing its private effects.
                1,
                crate::user_writes::Preparation::preflight(preparation, &mut provisional),
            )?;
            super::query_objects::PreparedObjects::validate(authored, selected, provisional)
        })?;
        // No SQLite statement, transaction or borrowed source body survives
        // this boundary. Only validated owned question requirements remain.
        self.write_prepared_objects(prepared).map(Some)
    }
}
