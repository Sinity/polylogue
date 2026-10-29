//! Replay authority comes from this index's existing seal and durable run custody.
use super::*;
use polylogue_model::{RunId, query_objects::ExecutionBinding};
use query_language::{
    cursor::{CursorSeal, SealedCursor},
    time::Clock,
};

/// Inputs captured once by the accepting operation. No default context is implied.
pub struct RunReplayContext<'a> {
    pub current_clock: &'a Clock,
    pub current_versions: &'a BTreeMap<String, String>,
    pub ranking_identity: Option<&'a str>,
    pub page_size: Option<u64>,
    pub near: Option<&'a NearFn<'a>>,
}

/// No deserializer or public constructor can certify a foreign run body.
pub(super) struct ReplayAuthority {
    root: RunId,
    binding: ExecutionBinding,
}

impl PreparedSelection<'_> {
    /// The execution was opened through this Reader's authenticated replay path.
    /// This establishes replay custody only, never native fact or projection proof.
    pub fn authenticated_execution(
        &self,
        reader: &Reader<'_>,
    ) -> Result<(&RunId, &ExecutionBinding), SelectionError> {
        reader.check()?;
        self.scope.check()?;
        if self.reader != std::ptr::from_ref(reader).cast::<()>() as usize {
            return Err(invalid("scope_changed", "replay belongs to another Reader"));
        }
        let authority = self.replay_authority.as_ref().ok_or_else(|| {
            invalid(
                "not_yet_available",
                "selection has no authenticated run replay authority",
            )
        })?;
        Ok((&authority.root, &authority.binding))
    }
}

impl Reader<'_> {
    /// Authenticate the existing cursor, obtain the actual acknowledged root run,
    /// then rebind and validate its content in this same pinned Reader. Merely
    /// supplying a serialized run to semantic preparation cannot enter this path.
    pub fn prepare_authenticated_run_selection<'read>(
        &'read self,
        cursor: &SealedCursor,
        question: QuestionId,
        pending: Option<&dyn crate::primary::Sinex>,
        context: RunReplayContext<'_>,
    ) -> Result<(PreparedSelection<'read>, OrderPosition), SelectionError> {
        self.check()?;
        let opened = CursorSeal::new(self.cursor_key()?).open(
            cursor,
            question,
            |name| {
                if name == context.current_clock.day.zone_name() {
                    Ok(context.current_clock.day.zone().clone())
                } else {
                    Err(query_language::Error::new(
                        query_language::ErrorKind::StaleCursor,
                        "the operation's configured day zone changed",
                        0,
                    ))
                }
            },
            |captured| {
                context
                    .current_versions
                    .iter()
                    .all(|(key, value)| captured.versions.get(key) == Some(value))
            },
        )?;
        let captured = &opened.context;
        if captured.clock.day.wire() != context.current_clock.day.wire() {
            return Err(invalid(
                "stale_cursor",
                "the operation's configured day changed",
            ));
        }
        let required = |key: &str| {
            captured.versions.get(key).ok_or_else(|| {
                invalid("invalid_cursor", format!("the cursor has no {key} binding"))
            })
        };
        let root: RunId = required("read_root_run")?
            .parse()
            .map_err(|_| invalid("invalid_cursor", "the cursor root run identity is invalid"))?;
        let expected_hash = polylogue_model::BlobHash::parse(required("read_binding")?)
            .map_err(|_| invalid("invalid_cursor", "the cursor binding hash is invalid"))?;
        let run = self
            .query_run(&root, pending)?
            .ok_or_else(|| {
                invalid(
                    "not_yet_available",
                    "the acknowledged root execution is unavailable",
                )
            })?
            .value;
        crate::query_objects::validate_run(&run)?;
        if run.id != root
            || run.question != question
            || run.binding.hash().map_err(StoreError::from)? != expected_hash
        {
            return Err(invalid(
                "stale_cursor",
                "the acknowledged execution binding changed",
            ));
        }
        let replay = &run.binding.replay.context;
        if replay.clock != captured.clock.wire()
            || replay.random_seed != captured.random_seed
            || replay.read_positions != captured.read_positions
            || replay
                .versions
                .iter()
                .any(|(key, value)| captured.versions.get(key) != Some(value))
        {
            return Err(invalid(
                "stale_cursor",
                "the cursor names another execution context",
            ));
        }
        let object = self
            .query_object(&question.to_string(), pending)?
            .ok_or_else(|| {
                invalid(
                    "not_yet_available",
                    "the acknowledged semantic query is unavailable",
                )
            })?
            .value;
        let mut selection = self.prepare_query_run_selection(
            &object,
            &run,
            &captured.clock,
            context.ranking_identity,
            context.page_size,
        )?;
        let inherited = match &run.binding.presentation {
            polylogue_model::query_objects::ExecutionPresentation::Read { options, .. } => {
                options.inherited
            }
            polylogue_model::query_objects::ExecutionPresentation::Query => false,
        };
        let actual_content =
            self.selection_content_fingerprint(&selection, inherited, context.near)?;
        if required("read_content")? != &actual_content {
            return Err(invalid("stale_cursor", "the selected content changed"));
        }
        selection.scope.check()?;
        selection.replay_authority = Some(ReplayAuthority {
            root,
            binding: run.binding,
        });
        Ok((selection, opened.after))
    }
}

#[cfg(test)]
mod tests;
