//! Primary writes prepare producer operations in the spool. Only confirmed
//! events write the cache.
use super::*;
mod reading;
use crate::primary::{Product, ProductWrite, SourceLocation};
use polylogue_model::authored::PreparedRequest;

/// What a page of confirmed events changes. A lifecycle change, a refusal or
/// a control may concern any record, so it counts as every kind.
fn confirmed_kinds(events: &[crate::primary::Confirmed]) -> crate::completion::Committed {
    use crate::completion::Committed;
    let mut kinds = Committed::default();
    for event in events {
        let kind = match &event.change {
            crate::primary::ConfirmedChange::Live(product) => match product.as_ref() {
                Product::Query { .. }
                | Product::QueryRun { .. }
                | Product::QueryRunOutcome { .. } => Committed::QUERY_RECORDS,
                Product::Request(_)
                | Product::RestoredRequest { .. }
                | Product::RestoredAnnotation { .. } => Committed::AUTHORED,
                _ => Committed::CONTENT,
            },
            _ => Committed::EVERY,
        };
        kinds.merge(kind);
    }
    kinds
}

impl Inner {
    pub(super) fn primary_cache_ready(&self) -> Result<()> {
        self.location.read_scope.token()?;
        crate::primary::scope::require_readable(&self.index)
    }

    pub(super) fn stage_confirmed(
        &mut self,
        events: &[crate::primary::Confirmed],
    ) -> Result<crate::primary::confirmed::Step> {
        if !self.location.is_primary() {
            return Err(StoreError::Invalid(
                "confirmed events require a primary cache".into(),
            ));
        }
        if let Some((scope, snapshot_start, reason)) =
            events.iter().find_map(|event| match &event.change {
                crate::primary::ConfirmedChange::Control(control) => match control.as_ref() {
                    crate::primary::ReplayControl::Reset { scope } => Some((scope, None, None)),
                    crate::primary::ReplayControl::SnapshotStart {
                        scope,
                        origin,
                        stream_instance,
                        reason,
                    } => Some((scope, Some((*origin, stream_instance)), reason.as_deref())),
                    _ => None,
                },
                _ => None,
            })
        {
            if events.len() != 1 {
                return Err(StoreError::Invalid(
                    "a scope reset must be a separate control page".into(),
                ));
            }
            if snapshot_start.is_some_and(|(origin, instance)| origin < 0 || instance.is_empty()) {
                return Err(StoreError::Invalid(
                    "a snapshot needs its Sinex origin and stream instance".into(),
                ));
            }
            let replacing = snapshot_start.is_some()
                || crate::primary::confirmed::replay_state(&self.index)?
                    .snapshot_progress
                    .is_some();
            crate::archive::index_build::primary::withdraw(
                &self.location,
                Some(scope),
                snapshot_start.map(|(_, instance)| instance.as_str()),
                reason.is_some(),
            )?;
            self.texts.discard_cached();
            let changed = match crate::primary::scope::reset(
                &mut self.index,
                &self.location,
                &self.resolver,
                scope,
                replacing,
            ) {
                Ok(changed) => changed,
                Err(error) => {
                    // Reads stay fenced after a failed reset. Record its
                    // failure before replying, so a blocked turn can observe it.
                    self.stream_followed(Some(error.to_string()));
                    return Err(error);
                }
            };
            if let Some((origin, stream_instance)) = snapshot_start {
                let mut state = crate::primary::confirmed::replay_state(&self.index)?;
                state.snapshot_progress = Some(crate::primary::SnapshotProgress {
                    origin,
                    stream_instance: stream_instance.clone(),
                    position: None,
                    cursor: None,
                });
                crate::primary::confirmed::save_replay(&self.index, &state)?;
            }
            if changed {
                self.commits.committed(crate::completion::Committed::EVERY);
                self.report.transactions += 1;
            }
            if let Some(reason) = reason {
                let error = StoreError::Sinex(reason.into());
                self.stream_followed(Some(error.to_string()));
                return Err(error);
            }
            return Ok(crate::primary::confirmed::Step::Complete(0));
        }
        let result = crate::primary::confirmed::stage_next(&mut self.index, events)?;
        Ok(result)
    }

    pub(super) fn apply_confirmed(
        &mut self,
        prepared: crate::primary::confirmed::Preparation,
        kinds: crate::completion::Committed,
    ) -> Result<crate::primary::confirmed::Step> {
        let before = crate::primary::confirmed::checkpoint(&self.index)?;
        let result =
            crate::primary::confirmed::apply_prepared(&self.texts, &mut self.index, prepared);
        // An earlier atomic application of this staged page may already have
        // committed. A later failure must remain visible even if this step
        // itself made no progress.
        if let Err(error) = &result {
            self.stream_followed(Some(error.to_string()));
        }
        if crate::primary::confirmed::checkpoint(&self.index)? != before {
            self.commits.committed(kinds);
            self.search.wake();
            self.report.transactions += 1;
        }
        result
    }

    pub(super) fn product_writes(&self, products: Vec<Product>) -> Result<Vec<ProductWrite>> {
        self.primary_cache_ready()?;
        product_writes(&self.index, products)
    }

    pub(super) fn validate_primary_reading(&self, batch: &Batch) -> Result<i64> {
        self.primary_cache_ready()?;
        validate_primary_reading(&self.index, batch)
    }

    pub(super) fn primary_request(
        &mut self,
        request: &polylogue_model::user::UserRequest,
        scope: Option<&crate::ReadScopeToken>,
        preparation: Option<&crate::user_writes::PreparationContext>,
    ) -> Result<crate::user_writes::WriteResult> {
        scoped::require_current(&self.location, scope)?;
        self.primary_cache_ready()?;
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        if let Some(known) = crate::primary::authored::known_request(&self.index, request)? {
            let mut outcome = known.outcome.clone();
            outcome.repeat = true;
            return crate::user_writes::WriteResult::recorded(&known, outcome);
        }
        if let Some(known) = crate::primary::authored::pending_prepared(sinex.as_ref(), request)? {
            let outcome = polylogue_model::user::UserOutcome {
                repeat: true,
                pending: true,
                ..known.outcome.clone()
            };
            return crate::user_writes::WriteResult::recorded(&known, outcome);
        }
        // The local overlay stays available while the stream is behind or offline.
        let prepared: PreparedRequest = crate::primary::authored::prepare(
            &self.index,
            request,
            preparation,
            Some(sinex.as_ref()),
            |apply| sinex.visit_pending_requests(apply),
            |apply| sinex.visit_pending_blobs(apply),
        )?;
        // Scope controls use this same serialized writer. Check after effect
        // preparation, immediately before the request is handed to the spool.
        scoped::require_current(&self.location, scope)?;
        if matches!(
            request.write,
            polylogue_model::user::UserWrite::MarkSeen { .. }
        ) && prepared.outcome.repeat
            && prepared.effect.annotations.is_empty()
        {
            return crate::user_writes::WriteResult::recorded(&prepared, prepared.outcome.clone());
        }
        let outcome = sinex.prepare_request(&prepared)?;
        crate::user_writes::WriteResult::recorded(&prepared, outcome)
    }

    pub(super) fn primary_blob(
        &mut self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
        self.primary_cache_ready()?;
        let known: Option<i64> = self.index.scalar_optional(
            "SELECT size FROM source_blob WHERE hash=?1",
            [&new_blob.hash.0[..]],
        )?;
        if let Some(size) = known {
            if u64::try_from(size).ok() != Some(new_blob.size) {
                return Err(StoreError::BlobCorrupt(new_blob.hash));
            }
            return Ok(new_blob.hash);
        }
        let prepared = file.or_invalid("new primary blob has no prepared bytes")?;
        if prepared.owner != self.location.root
            || prepared.hash() != new_blob.hash
            || prepared.len() != new_blob.size
        {
            return Err(StoreError::Invalid(
                "primary blob bytes differ from their declaration".into(),
            ));
        }
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        let prepared = sinex.prepare_blob(new_blob.hash, new_blob.size, prepared.body()?)?;
        // Captured bytes are saved before their temporary preparation is released.
        if let Err(error) = self.drain_primary_captures() {
            return Err(match error {
                StoreError::Sinex(_) => StoreError::PrimaryPending {
                    identity: prepared.identity,
                },
                error => error,
            });
        }
        if self.index.scalar::<bool>(
            "SELECT EXISTS(SELECT 1 FROM source_blob WHERE hash=?1)",
            [&new_blob.hash.0[..]],
        )? {
            Ok(new_blob.hash)
        } else {
            Err(StoreError::PrimaryPending {
                identity: prepared.identity,
            })
        }
    }
}

pub(super) fn product_writes(
    index: &Connection,
    products: Vec<Product>,
) -> Result<Vec<ProductWrite>> {
    let mut writes = Vec::with_capacity(products.len());
    for (ordinal, product) in products.into_iter().enumerate() {
        let (location, machine) = if let Some(reading) = product.reading() {
            let start = match &product {
                Product::Message { facts, .. } => facts.offset,
                Product::ToolCall { facts, .. } => facts.source_start,
                Product::Document { facts, .. } => facts.offset,
                Product::Fragment { facts, .. } => facts.offset,
                Product::Event { facts, .. } => facts.range.map_or(reading.start, |range| range.0),
                _ => reading.start,
            };
            let location = crate::primary::confirmed::material_at(
                index,
                &reading.input.id,
                reading.chain,
                start,
            )?;
            let machine=index.scalar_optional::<String>("SELECT machine FROM source_observation WHERE input=?1 ORDER BY commit_seq DESC,rowid DESC LIMIT 1",[&reading.input.id])?;
            (location, machine)
        } else {
            match &product {
                Product::Coverage(coverage) => (None, Some(coverage.machine.clone())),
                Product::Input { rows, material } => {
                    let location = material.as_ref().map(|material| SourceLocation {
                        material: material.id.clone(),
                        anchor: 0,
                    });
                    (
                        location,
                        rows.observation.as_ref().map(|row| row.machine.clone()),
                    )
                }
                Product::Blob(material) => (
                    Some(SourceLocation {
                        material: material.id.clone(),
                        anchor: 0,
                    }),
                    None,
                ),
                Product::Request(prepared) => (None, Some(prepared.request.machine.clone())),
                Product::RestoredRequest { prepared } => {
                    (None, Some(prepared.request.machine.clone()))
                }
                Product::RestoredAnnotation { fact, .. } => (None, Some(fact.machine.clone())),
                _ => unreachable!(),
            }
        };
        writes.push(ProductWrite {
            product,
            location,
            parents: Vec::new(),
            machine,
            output_index: ordinal as u64,
        });
    }
    Ok(writes)
}

pub(super) fn validate_primary_reading(index: &Connection, batch: &Batch) -> Result<i64> {
    let Change::Reparse { chain } = batch.change else {
        return Err(StoreError::Invalid(
            "primary parsed products need their confirmed source chain".into(),
        ));
    };
    if batch.capture_read.is_some()
        || batch.observation.is_some()
        || batch.new_blob.is_some()
        || batch.base.is_some()
        || batch.required_outcome.is_some()
        || batch.continues
    {
        return Err(StoreError::Invalid(
            "a stored reading cannot capture or rebind its source".into(),
        ));
    }
    let known: bool = index.exists(
        "SELECT EXISTS(SELECT 1 FROM source_input WHERE id=?1 AND format=?2)",
        params![batch.input.id, batch.input.format],
    )?;
    if !known {
        return Err(StoreError::Invalid(
            "stored reading input identity differs".into(),
        ));
    }
    crate::staging::require_range(index, "source_", &batch.input.id, chain, 0..batch.length)?;
    if crate::source::Source::committed_through(index, &batch.input.id, chain, batch.length)?
        .is_none()
    {
        return Err(StoreError::Invalid(
            "stored reading source extents do not cover its range".into(),
        ));
    }
    for step in &batch.parse {
        step.counts.check()?;
        if step.start > step.parsed_upto || step.parsed_upto > batch.length {
            return Err(StoreError::Invalid(
                "stored reading exceeds its source range".into(),
            ));
        }
    }
    Ok(chain)
}

impl WriterSender {
    /// Loads one capture read from the spool with its complete metadata.
    pub fn capture(&self, identity: String) -> Result<crate::primary::Capture> {
        self.0.ask(|reply| Msg::Capture(identity, reply))
    }

    /// Pins one pending source body held in the spool, without network I/O.
    pub fn capture_body(&self, identity: String) -> Result<Vec<crate::primary::PreparedBody>> {
        self.0.ask(|reply| Msg::CaptureBody(identity, reply))
    }

    /// Reads a spool byte cursor for one observed provider state.
    pub fn byte_cursor(
        &self,
        query: crate::primary::ByteCursorQuery,
    ) -> Result<Option<crate::primary::ByteCursor>> {
        self.0.ask(|reply| Msg::ByteCursor(Box::new(query), reply))
    }

    /// Pages original group membership without loading parsed events.
    pub fn pending_group_headers(
        &self,
        group: String,
        after_ordinal: Option<u64>,
        limit: usize,
    ) -> Result<crate::primary::PendingGroupPage> {
        self.0
            .ask(|reply| Msg::PendingGroupHeaders(group, after_ordinal, limit, reply))
    }

    /// Reads the confirmed identity and the pending source held in the spool,
    /// in writer order.
    pub fn source_snapshot(
        &self,
        source: crate::primary::SourceSelector,
    ) -> Result<crate::primary::SourceSnapshot> {
        self.0.ask(|reply| Msg::SourceSnapshot(source, reply))
    }

    pub fn apply_confirmed(&self, events: Vec<crate::primary::Confirmed>) -> Result<i64> {
        let result: Result<i64> = (|| {
            let kinds = confirmed_kinds(&events);
            let mut step = self.0.ask(|reply| Msg::StageConfirmed(events, reply))?;
            loop {
                match step {
                    crate::primary::confirmed::Step::Complete(position) => return Ok(position),
                    crate::primary::confirmed::Step::Prepare(mut prepared) => {
                        // Queued groups can hold these key locks. Waiting here
                        // leaves the serial writer free to commit or reject them.
                        prepare_source_retractions(&self.0.resolver, &mut prepared.retractions)?;
                        step = self
                            .0
                            .ask(|reply| Msg::ApplyConfirmed(prepared, kinds, reply))?;
                    }
                }
            }
        })();
        if let Err(error) = &result {
            // Staging or off-writer resolution can also fail after a preceding
            // position committed. Report it through the same serialized stream error.
            let _ = self.0.send(Msg::StreamFollowed(Some(error.to_string())));
        }
        result
    }
}
