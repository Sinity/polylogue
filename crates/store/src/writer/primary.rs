//! Primary writes prepare durable producer intents. Only confirmed facts write the cache.
use super::*;
mod reading;
use crate::primary::{Product, ProductWrite, SourceLocation};
use polylogue_model::authored::PreparedRequest;

/// What a page of confirmed facts changes. A lifecycle change, a refusal or
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
                | Product::RestoredAssertion { .. } => Committed::AUTHORED,
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
    ) -> Result<crate::primary::projection::Step> {
        if !self.location.is_primary() {
            return Err(StoreError::Invalid(
                "confirmed facts require a primary cache".into(),
            ));
        }
        if let Some((scope, projection, reason)) =
            events.iter().find_map(|event| match &event.change {
                crate::primary::ConfirmedChange::Control(control) => match control.as_ref() {
                    crate::primary::ReplayControl::Reset { scope } => Some((scope, None, None)),
                    crate::primary::ReplayControl::ProjectionStart {
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
            if projection.is_some_and(|(origin, instance)| origin < 0 || instance.is_empty()) {
                return Err(StoreError::Invalid(
                    "a projection needs its native origin and incarnation".into(),
                ));
            }
            let replacing = projection.is_some()
                || crate::primary::projection::replay_state(&self.index)?
                    .projection
                    .is_some();
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
                    // Reads stay fenced after a failed reset. Publish its real
                    // failure before replying, so a blocked turn can observe it.
                    self.stream_followed(Some(error.to_string()));
                    return Err(error);
                }
            };
            if let Some((origin, stream_instance)) = projection {
                let mut state = crate::primary::projection::replay_state(&self.index)?;
                state.projection = Some(crate::primary::ProjectionReplay {
                    origin,
                    stream_instance: stream_instance.clone(),
                    position: None,
                    cursor: None,
                });
                crate::primary::projection::save_replay(&self.index, &state)?;
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
            return Ok(crate::primary::projection::Step::Complete(0));
        }
        let result = crate::primary::projection::stage_next(&mut self.index, events)?;
        Ok(result)
    }

    pub(super) fn apply_confirmed(
        &mut self,
        prepared: crate::primary::projection::Preparation,
        kinds: crate::completion::Committed,
    ) -> Result<crate::primary::projection::Step> {
        let before = crate::primary::projection::checkpoint(&self.index)?;
        let result =
            crate::primary::projection::apply_prepared(&self.texts, &mut self.index, prepared);
        // An earlier atomic application of this staged page may already have
        // committed. A later failure must remain visible even if this step
        // itself made no progress.
        if let Err(error) = &result {
            self.stream_followed(Some(error.to_string()));
        }
        if crate::primary::projection::checkpoint(&self.index)? != before {
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
    ) -> Result<polylogue_model::user::UserOutcome> {
        scoped::require_current(&self.location, scope)?;
        self.primary_cache_ready()?;
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        if let Some(known) = crate::primary::authored::known_request(&self.index, request)? {
            let mut outcome = known.outcome;
            outcome.repeat = true;
            return Ok(outcome);
        }
        if let Some(outcome) = crate::primary::authored::pending_outcome(sinex.as_ref(), request)? {
            return Ok(outcome);
        }
        // The durable local overlay remains available while the stream is behind or offline.
        let prepared: PreparedRequest = crate::primary::authored::prepare(
            &self.index,
            request,
            preparation,
            Some(sinex.as_ref()),
            |apply| sinex.visit_pending_requests(apply),
            |apply| sinex.visit_pending_blobs(apply),
        )?;
        // Scope controls use this same serialized writer. Check after effect
        // preparation, immediately before transferring durable outbox custody.
        scoped::require_current(&self.location, scope)?;
        sinex.prepare_request(&prepared)
    }

    pub(super) fn primary_blob(
        &mut self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
        self.primary_cache_ready()?;
        let known: Option<i64> = self
            .index
            .query_row(
                "SELECT size FROM source_blob WHERE hash=?1",
                [&new_blob.hash.0[..]],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(size) = known {
            if u64::try_from(size).ok() != Some(new_blob.size) {
                return Err(StoreError::BlobCorrupt(new_blob.hash));
            }
            return Ok(new_blob.hash);
        }
        let prepared = file
            .ok_or_else(|| StoreError::Invalid("new primary blob has no prepared bytes".into()))?;
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
        // Captured bytes are durable before their temporary preparation is released.
        if let Err(error) = self.drain_primary_captures() {
            return Err(match error {
                StoreError::Sinex(_) => StoreError::PrimaryPending {
                    identity: prepared.identity,
                },
                error => error,
            });
        }
        if self.index.query_row(
            "SELECT EXISTS(SELECT 1 FROM source_blob WHERE hash=?1)",
            [&new_blob.hash.0[..]],
            |row| row.get::<_, bool>(0),
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
            let location = crate::primary::projection::material_at(
                index,
                &reading.input.id,
                reading.chain,
                start,
            )?;
            let machine=index.query_row("SELECT machine FROM source_observation WHERE input=?1 ORDER BY commit_seq DESC,rowid DESC LIMIT 1",[&reading.input.id],|row|row.get::<_,String>(0)).optional()?;
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
                Product::RestoredAssertion { fact, .. } => (None, Some(fact.machine.clone())),
                _ => unreachable!(),
            }
        };
        writes.push(ProductWrite {
            product,
            location,
            parents: Vec::new(),
            machine,
            output_index: ordinal as u64,
            publish: true,
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
    if batch.acquisition.is_some()
        || batch.observation.is_some()
        || batch.new_blob.is_some()
        || batch.basis.is_some()
        || batch.required_disposition.is_some()
        || batch.continues
    {
        return Err(StoreError::Invalid(
            "a stored reading cannot acquire or rebind its source".into(),
        ));
    }
    let exact: bool = index.query_row(
        "SELECT EXISTS(SELECT 1 FROM source_input WHERE id=?1 AND format=?2)",
        params![batch.input.id, batch.input.format],
        |row| row.get(0),
    )?;
    if !exact {
        return Err(StoreError::Invalid(
            "stored reading input identity differs".into(),
        ));
    }
    crate::staging::require_range(index, "source_", &batch.input.id, chain, 0..batch.length)?;
    if crate::source::Source::admitted_through(index, &batch.input.id, chain, batch.length)?
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

impl WriterQueue {
    pub(super) fn apply_confirmed(&self, events: Vec<crate::primary::Confirmed>) -> Result<i64> {
        let result: Result<i64> = (|| {
            let kinds = confirmed_kinds(&events);
            let (reply, wait) = sync_channel(1);
            self.send(Msg::StageConfirmed(events, reply))
                .map_err(|_| StoreError::WriterGone)?;
            let mut step = wait.recv().map_err(|_| StoreError::WriterGone)??;
            loop {
                match step {
                    crate::primary::projection::Step::Complete(position) => return Ok(position),
                    crate::primary::projection::Step::Prepare(mut prepared) => {
                        // Ordinary queued groups can hold these leases. Waiting here
                        // leaves the serial writer free to commit or reject them.
                        prepare_source_retractions(&self.resolver, &mut prepared.retractions)?;
                        let (reply, wait) = sync_channel(1);
                        self.send(Msg::ApplyConfirmed(prepared, kinds, reply))
                            .map_err(|_| StoreError::WriterGone)?;
                        step = wait.recv().map_err(|_| StoreError::WriterGone)??;
                    }
                }
            }
        })();
        if let Err(error) = &result {
            // Staging or off-writer resolution can also fail after a preceding
            // position committed. Use the same serialized stream error owner.
            let _ = self.send(Msg::StreamFollowed(Some(error.to_string())));
        }
        result
    }
}
