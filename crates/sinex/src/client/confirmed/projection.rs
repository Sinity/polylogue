//! Bounded owner-certified snapshot pages share the ordinary pending-fact writer.
use super::*;
use futures_util::stream;
use sinex_protocol::{InputReplayReply, ReplayContinuation, ReplayQuery, ReplayReply};

fn start(
    scope: String,
    origin: i64,
    stream_instance: String,
    reason: Option<String>,
) -> Vec<Confirmed> {
    vec![control_fact(ReplayControl::ProjectionStart {
        scope,
        origin,
        stream_instance,
        reason,
    })]
}

fn matches(
    reference: &sinex_protocol::InputReference,
    event: &sinex_protocol::ConfirmedEvent,
) -> bool {
    event.id == reference.event
        && event.commit_seq == reference.commit_seq
        && event.event_type == reference.kind.name
        && event.kind_version == reference.kind.version
        && event.kind_definition_hash == reference.kind_definition_hash
        && event.projection == reference.projection
        && event.source_ref == reference.source
        && event.source_fingerprint == reference.source_fingerprint
        && event.projection.verifies(&event.payload)
}

impl Client {
    pub(super) async fn projection_page(
        &mut self,
        state: &ReplayState,
        cache: &str,
        origin: i64,
        head: i64,
        scope: String,
        stream_instance: String,
    ) -> Result<Vec<Confirmed>> {
        if origin < 0 || head < origin || scope.is_empty() || stream_instance.is_empty() {
            return Err(Error::Invalid(
                "the projection origin has inconsistent native identity".into(),
            ));
        }
        self.consumer = None;
        self.consumer_cache = None;
        self.consumer_instance = None;
        let Some(progress) = state.projection.as_ref().filter(|progress| {
            state.scope.as_deref() == Some(scope.as_str())
                && progress.origin == origin
                && progress.stream_instance == stream_instance
        }) else {
            return Ok(start(scope, origin, stream_instance, None));
        };
        let reply: ReplayReply = self
            .request(
                self.subjects.replay_scope(&self.producer),
                ReplayQuery::Inputs {
                    cache: cache.into(),
                    cursor: progress.cursor.clone(),
                },
            )
            .await?;
        let ReplayReply::Inputs { result } = reply else {
            return Err(Error::Invalid(
                "projection replay returned another mode".into(),
            ));
        };
        let (position, returned_scope, inputs, continuation) = match result {
            InputReplayReply::Page {
                position,
                scope,
                inputs,
                continuation,
            } => (position, scope, inputs, continuation),
            InputReplayReply::Incomplete {
                reason,
                basis: sinex_protocol::ReplayBasisStatus::Invalidated { .. },
            } => {
                return Ok(start(scope, origin, stream_instance, Some(reason)));
            }
            InputReplayReply::Incomplete {
                reason,
                basis: sinex_protocol::ReplayBasisStatus::Unverified,
            } => {
                return Err(Error::Refused {
                    name: "projection replay".into(),
                    reason,
                });
            }
        };
        if position < origin
            || returned_scope != scope
            || progress.position.is_some_and(|old| old != position)
        {
            return Err(Error::Invalid(
                "a projection page changed its cut or scope".into(),
            ));
        }
        // Bound requests in flight without serializing every network round trip.
        let client = &*self;
        let mut reads = stream::iter(inputs.iter().map(|reference| async move {
            if reference.commit_seq <= 0
                || reference.commit_seq > position
                || reference
                    .lifecycle_position
                    .is_some_and(|changed| changed < reference.commit_seq || changed > position)
                || reference.archived && reference.lifecycle_position.is_none()
            {
                return Err(Error::Invalid(
                    "a projection reference has inconsistent positions".into(),
                ));
            }
            let Some(event) = client
                .fetch_event(reference.event, reference.commit_seq)
                .await?
            else {
                return Ok(None);
            };
            if !matches(reference, &event) {
                return Err(Error::Invalid(
                    "a projection fetch differs from its certified reference".into(),
                ));
            }
            let live = decode_event(&event)?;
            let archived = reference.archived.then(|| Confirmed {
                position: reference
                    .lifecycle_position
                    .expect("validated lifecycle position"),
                event: reference.event.to_string(),
                output_index: 0,
                native: None,
                change: ConfirmedChange::Archived,
            });
            Ok(Some((live, archived)))
        }))
        .buffered(16);
        let mut page = Vec::new();
        while let Some(read) = reads.next().await {
            let Some((live, archived)) = read? else {
                // Erasure after the owner's cut invalidated this fetch. Restart
                // the snapshot; its missing historical body is not fabricated.
                return Ok(start(
                    scope,
                    origin,
                    stream_instance,
                    Some("a snapshot event was erased before its payload could be read".into()),
                ));
            };
            page.push(live);
            page.extend(archived);
        }
        drop(reads);
        page.push(control_fact(match continuation {
            ReplayContinuation::Next { cursor } => {
                if cursor.is_empty() || progress.cursor.as_ref() == Some(&cursor) {
                    return Err(Error::Invalid(
                        "the projection cursor did not advance".into(),
                    ));
                }
                ReplayControl::ProjectionPage { position, cursor }
            }
            ReplayContinuation::Complete { projection, .. } => {
                if projection.position != position || projection.scope != scope {
                    return Err(Error::Invalid(
                        "the projection certificate differs from its pages".into(),
                    ));
                }
                ReplayControl::ProjectionComplete {
                    basis: local::ProjectionBasis {
                        stamp: projection.stamp,
                        position,
                        scope,
                        catalog: projection.catalog,
                    },
                }
            }
        }));
        Ok(page)
    }
}

#[cfg(test)]
mod tests;
