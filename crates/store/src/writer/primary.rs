//! Primary writes prepare durable producer intents. Only confirmed facts write the cache.
use super::*;
mod export;
use crate::primary::{Product, ProductWrite, SourceLocation};
use polylogue_model::authored::PreparedRequest;

impl Inner {
    pub(super) fn primary_cache_ready(&self) -> Result<()> {
        self.location.read_scope.token()?;
        crate::primary::scope::require_readable(&self.index)
    }

    pub(super) fn apply_confirmed(&mut self, events: &[crate::primary::Confirmed]) -> Result<i64> {
        if !self.location.is_primary() {
            return Err(StoreError::Invalid(
                "confirmed facts require a primary cache".into(),
            ));
        }
        if let Some(scope) = events.iter().find_map(|event| match &event.change {
            crate::primary::ConfirmedChange::Control(control) => match control.as_ref() {
                crate::primary::ReplayControl::Reset { scope } => Some(scope),
                _ => None,
            },
            _ => None,
        }) {
            if events.len() != 1 {
                return Err(StoreError::Invalid(
                    "a scope reset must be a separate control page".into(),
                ));
            }
            if crate::primary::scope::reset(&mut self.index, &self.location, &self.resolver, scope)?
            {
                self.commits.committed();
                self.derive_due = false;
                self.derived_version = None;
                self.report.transactions += 1;
            }
            return Ok(0);
        }
        let before = crate::primary::projection::checkpoint(&self.index)?;
        let result = crate::primary::projection::apply(&mut self.index, &self.resolver, events);
        if crate::primary::projection::checkpoint(&self.index)? != before {
            // A later complete position can fail after earlier positions committed.
            // Make that failure observable before notifying their real progress.
            if let Err(error) = &result {
                self.stream_followed(Some(error.to_string()));
            }
            self.commits.committed();
            self.derive_due = true;
            self.search.wake();
            self.report.transactions += 1;
        }
        result
    }

    pub(super) fn product_writes(&self, products: Vec<Product>) -> Result<Vec<ProductWrite>> {
        self.primary_cache_ready()?;
        let mut writes = Vec::with_capacity(products.len());
        for (index, product) in products.into_iter().enumerate() {
            let (location, machine) = if let Some(reading) = product.reading() {
                let start = match &product {
                    Product::Message { facts, .. } => facts.offset,
                    Product::Document { facts, .. } => facts.offset,
                    Product::Fragment { facts, .. } => facts.offset,
                    Product::Event { facts, .. } => {
                        facts.range.map_or(reading.start, |range| range.0)
                    }
                    _ => reading.start,
                };
                let location = crate::primary::projection::material_at(
                    &self.index,
                    &reading.input.id,
                    reading.chain,
                    start,
                )?;
                let machine=self.index.query_row("SELECT machine FROM source_observation WHERE input=?1 ORDER BY commit_seq DESC,rowid DESC LIMIT 1",[&reading.input.id],|row|row.get::<_,String>(0)).optional()?;
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
                    Product::RestoredAssertion { fact } => (None, Some(fact.machine.clone())),
                    _ => unreachable!(),
                }
            };
            writes.push(ProductWrite {
                product,
                location,
                parents: Vec::new(),
                machine,
                output_index: index as u64,
            });
        }
        Ok(writes)
    }

    pub(super) fn validate_primary_reading(&self, batch: &Batch) -> Result<i64> {
        self.primary_cache_ready()?;
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
        let exact: bool = self.index.query_row(
            "SELECT EXISTS(SELECT 1 FROM source_input WHERE id=?1 AND format=?2)",
            params![batch.input.id, batch.input.format],
            |row| row.get(0),
        )?;
        if !exact {
            return Err(StoreError::Invalid(
                "stored reading input identity differs".into(),
            ));
        }
        crate::staging::require_range(
            &self.index,
            "source_",
            &batch.input.id,
            chain,
            0..batch.length,
        )?;
        if crate::source::Source::admitted_through(
            &self.index,
            &batch.input.id,
            chain,
            batch.length,
        )?
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

    pub(super) fn primary_products<'a>(
        &mut self,
        batches: impl Iterator<Item = &'a Batch>,
    ) -> Result<()> {
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        for batch in batches {
            let chain = self.validate_primary_reading(batch)?;
            for step in &batch.parse {
                let key = step.delivery.map(|_| crate::primary::ExportReadingKey {
                    input: batch.input.clone(),
                    chain,
                    reading_version: step.reading_version.clone(),
                    length: batch.length,
                });
                if let Some(key) = &key
                    && !self.admit_primary_export(key, step)?
                {
                    continue;
                }
                let writes = self.product_writes(crate::primary::products::parse(
                    &batch.input,
                    chain,
                    Some(batch.length),
                    step,
                )?)?;
                if writes
                    .iter()
                    .any(|write| write.product.reading().is_some() && write.location.is_none())
                {
                    return Err(StoreError::Invalid(
                        "stored reading has no confirmed source material anchor".into(),
                    ));
                }
                if let Some(key) = &key {
                    sinex.prepare_export_products(key, &writes)?;
                } else {
                    sinex.prepare_products(&writes)?;
                }
            }
        }
        // This fire-and-forget batch acknowledges durable preparation. The
        // follower independently makes its records visible after confirmation.
        Ok(())
    }

    pub(super) fn primary_reading(
        &mut self,
        input: InputKey,
        chain: i64,
        length: u64,
        mut source: ChainReading,
    ) -> Result<u64> {
        let mut version = None;
        let mut complete = false;
        let mut refusal = false;
        let mut has_export = false;
        source.visit_source(|step| {
            if version
                .as_ref()
                .is_some_and(|old| old != &step.reading_version)
            {
                return Err(StoreError::Invalid(
                    "one replacement has several reading versions".into(),
                ));
            }
            version.get_or_insert_with(|| step.reading_version.clone());
            complete = step.progress.reached_end() && step.parsed_upto <= length;
            refusal = crate::wire_shapes::is_refusal(step);
            has_export |= step.delivery.is_some();
            Ok(())
        })?;
        if !complete && !refusal {
            return Err(StoreError::Invalid(
                "a replacement reading is not complete".into(),
            ));
        }
        let version =
            version.ok_or_else(|| StoreError::Invalid("a replacement has no reading".into()))?;
        let key = crate::primary::ExportReadingKey {
            input: input.clone(),
            chain,
            reading_version: version,
            length,
        };
        let mut has_apply = false;
        let mut has_replay = false;
        if has_export {
            let mut prefix = self.pending_export_prefix(&key)?;
            source.visit_source(|step| {
                let apply = crate::wire_shapes::advance_prefix(&mut prefix, length, step)?;
                has_apply |= apply;
                has_replay |= !apply;
                Ok(())
            })?;
            if !has_apply {
                return Ok(0);
            }
        }
        self.primary_cache_ready()?;
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        if has_export && (has_replay || refusal) {
            let mut prefix = self.pending_export_prefix(&key)?;
            let mut messages = 0;
            let mut pending = None;
            source.visit_source(|step| {
                if crate::wire_shapes::advance_prefix(&mut prefix, length, step)? {
                    let writes = self.product_writes(crate::primary::products::parse(
                        &input,
                        chain,
                        Some(length),
                        step,
                    )?)?;
                    pending = Some(sinex.prepare_export_products(&key, &writes)?.identity);
                    messages += step.output.messages.len() as u64;
                }
                Ok(())
            })?;
            return match pending {
                Some(identity) => Err(StoreError::PrimaryPending { identity }),
                None => Ok(messages),
            };
        }
        let root:Option<String>=self.index.query_row("SELECT id FROM source_material WHERE input=?1 AND chain=?2 AND offset=0 ORDER BY rowid LIMIT 1",params![input.id,chain],|row|row.get(0)).optional()?;
        let root = root.ok_or_else(|| {
            StoreError::Invalid("the confirmed source chain has no root material".into())
        })?;
        let prepared = sinex.prepare_reading(&root, &key, &mut |emit| {
            source.visit_source(|step| {
                for product in self.product_writes(crate::primary::products::parse(
                    &input,
                    chain,
                    Some(length),
                    step,
                )?)? {
                    emit(product)?;
                }
                Ok(())
            })
        })?;
        Err(StoreError::PrimaryPending {
            identity: prepared.identity,
        })
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
