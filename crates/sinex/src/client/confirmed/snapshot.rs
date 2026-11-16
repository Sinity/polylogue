//! Snapshot pages of Sinex's current state share the pending-event writer.
use super::*;
use futures_util::stream;
use sinex_protocol::{
    ReplayQuery, ReplayReply, SnapshotCursor, SnapshotNext, SnapshotReading, SnapshotReply,
};

fn start(
    scope: String,
    origin: i64,
    stream_instance: String,
    reason: Option<String>,
) -> Vec<Confirmed> {
    vec![control_event(ReplayControl::SnapshotStart {
        scope,
        origin,
        stream_instance,
        reason,
    })]
}

impl Client {
    pub(super) async fn snapshot_page(
        &mut self,
        state: &ReplayState,
        origin: i64,
        latest_position: i64,
        scope: String,
        stream_instance: String,
    ) -> Result<Vec<Confirmed>> {
        if origin < 0 || latest_position < origin || scope.is_empty() || stream_instance.is_empty()
        {
            return Err(Error::Invalid(
                "the snapshot origin has an inconsistent stream identity".into(),
            ));
        }
        self.consumer = None;
        self.consumer_cache = None;
        self.consumer_instance = None;
        let Some(progress) = state.snapshot_progress.as_ref().filter(|progress| {
            state.scope.as_deref() == Some(scope.as_str())
                && progress.origin == origin
                && progress.stream_instance == stream_instance
        }) else {
            return Ok(start(scope, origin, stream_instance, None));
        };
        let cursor = progress
            .cursor
            .as_deref()
            .map(serde_json::from_str::<SnapshotCursor>)
            .transpose()?;
        let reply: ReplayReply = self
            .request(
                self.subjects.replay_scope(&self.producer),
                ReplayQuery::Snapshot {
                    cursor: cursor.clone(),
                },
            )
            .await?;
        let ReplayReply::Snapshot { reply } = reply else {
            return Err(Error::Invalid(
                "a snapshot request returned another mode".into(),
            ));
        };
        let (events, next) = match reply {
            SnapshotReply::Page { events, next } => (events, next),
            SnapshotReply::Restart { reason } => {
                return Ok(start(scope, origin, stream_instance, Some(reason)));
            }
            SnapshotReply::Unavailable { reason } => {
                return Err(Error::Refused {
                    name: "snapshot".into(),
                    reason,
                });
            }
        };
        let position = match &next {
            SnapshotNext::More { cursor } => cursor.position,
            SnapshotNext::Complete { complete } => complete.position,
        };
        let (next_scope, next_instance) = match &next {
            SnapshotNext::More { cursor } => (&cursor.scope, &cursor.stream_instance),
            SnapshotNext::Complete { complete } => (&complete.scope, &complete.stream_instance),
        };
        if position < origin
            || *next_scope != scope
            || *next_instance != stream_instance
            || progress.position.is_some_and(|old| old != position)
        {
            return Err(Error::Invalid(
                "a snapshot page changed its position, scope or stream".into(),
            ));
        }
        // Withheld readings are fetched with a bound on requests in flight.
        let client = &*self;
        let mut reads = stream::iter(events.into_iter().map(|entry| async move {
            let (commit_seq, id) = match &entry.event {
                SnapshotReading::Inline(event) => (event.commit_seq, event.id),
                SnapshotReading::Withheld(event) => (event.commit_seq, event.event),
            };
            if commit_seq <= 0
                || commit_seq > position
                || entry
                    .lifecycle_position
                    .is_some_and(|changed| changed < commit_seq || changed > position)
                || entry.archived && entry.lifecycle_position.is_none()
            {
                return Err(Error::Invalid(
                    "a snapshot reading has inconsistent positions".into(),
                ));
            }
            let event = match entry.event {
                SnapshotReading::Inline(event) => *event,
                SnapshotReading::Withheld(_) => {
                    let Some(event) = client.fetch_event(id, commit_seq).await? else {
                        return Ok(None);
                    };
                    event
                }
            };
            if event.id != id
                || event.commit_seq != commit_seq
                || !event.projection.verifies(&event.payload)
            {
                return Err(Error::Invalid(
                    "a snapshot reading differs from its position or payload hash".into(),
                ));
            }
            let live = confirmed_event(&event)?;
            let archived = entry.archived.then(|| Confirmed {
                position: entry
                    .lifecycle_position
                    .expect("validated lifecycle position"),
                event: id.to_string(),
                output_index: 0,
                sinex_context: None,
                change: ConfirmedChange::Archived,
            });
            Ok(Some((live, archived)))
        }))
        .buffered(FETCH_CONCURRENCY);
        let mut page = Vec::new();
        while let Some(read) = reads.next().await {
            let Some((live, archived)) = read? else {
                // An erasure after the snapshot position: restart the snapshot
                // rather than make up the missing body.
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
        page.push(control_event(match next {
            SnapshotNext::More { cursor: next } => {
                if cursor.as_ref() == Some(&next) {
                    return Err(Error::Invalid("the snapshot cursor did not advance".into()));
                }
                ReplayControl::SnapshotPage {
                    position,
                    cursor: serde_json::to_string(&next)?,
                }
            }
            SnapshotNext::Complete { complete } => ReplayControl::SnapshotComplete {
                snapshot: local::AppliedSnapshot {
                    position,
                    scope,
                    stream_instance,
                    last: complete.last.map(|last| local::LastStreamMessage {
                        sequence: last.sequence,
                        fingerprint: last.fingerprint,
                    }),
                },
            },
        }));
        Ok(page)
    }
}

#[cfg(test)]
mod tests;
