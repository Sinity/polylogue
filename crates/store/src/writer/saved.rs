//! A query object is stored before Sinex accepts the saved assertion. The
//! private preflight orders the proposals but never sends their records.
use super::*;
use polylogue_model::{
    assertion::AssertionKind,
    settings::SettingMode,
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
            UserWrite::ReviseAssertions { .. } => true,
            UserWrite::Save(save) => save.name.is_some(),
            UserWrite::Assert { assertions } => assertions
                .iter()
                .any(|statement| statement.kind == AssertionKind::SavedQuery),
            UserWrite::AssertQuery { assertion, .. } => assertion.kind == AssertionKind::SavedQuery,
            _ => false,
        };
        if !needed {
            return Ok(None);
        }
        if crate::primary::authored::known_request(&self.index, request)?.is_some() {
            return Ok(None);
        }
        let sinex = self.sinex.clone();
        if let Some(sinex) = &sinex {
            if crate::primary::authored::pending_outcome(sinex.as_ref(), request)?.is_some() {
                return Ok(None);
            }
            self.primary_cache_ready()?;
        }
        let selected = match scope {
            Some(scope) => scope.clone(),
            None => self.location.scope_token()?,
        };
        scoped::require_current(&self.location, Some(&selected))?;
        let mut provisional = crate::selection::ProvisionalQueries::default();
        let mode = if sinex.is_some() {
            SettingMode::Primary
        } else {
            SettingMode::Standalone
        };
        let prepared = crate::primary::authored::with_overlay(
            &self.index,
            |visit| match &sinex {
                Some(sinex) => sinex.visit_pending_requests(visit),
                None => Ok(()),
            },
            |authored| {
                crate::user_writes::apply_resolving(
                    authored,
                    &self.index,
                    request,
                    // Copied and pending baseline records use zero. Keep the
                    // proposal separate when capturing its private effects.
                    1,
                    crate::user_writes::Preparation::preflight(
                        preparation,
                        sinex.as_deref(),
                        mode,
                        &mut provisional,
                    ),
                )?;
                super::query_objects::PreparedObjects::validate(authored, selected, provisional)
            },
        )?;
        // No SQLite statement, transaction or borrowed source body survives
        // this boundary. Only validated owned question requirements remain.
        self.write_prepared_objects(prepared).map(Some)
    }
}
