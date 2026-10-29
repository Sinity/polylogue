//! Primary writes prepare durable producer intents. Only confirmed facts write the cache.
use super::*;
use crate::primary::{Product, ProductWrite, SourceLocation};
use polylogue_model::authored::PreparedRequest;

impl Inner {
    pub(super) fn apply_confirmed(&mut self, events: &[crate::primary::Confirmed]) -> Result<i64> {
        if !self.location.is_primary() {
            return Err(StoreError::Invalid(
                "confirmed facts require a primary cache".into(),
            ));
        }
        let before = crate::primary::projection::position(&self.index)?;
        let position = crate::primary::projection::apply(&mut self.index, &self.resolver, events)?;
        if position != before {
            self.derive_due = true;
            self.search.wake();
            self.report.transactions += 1;
        }
        Ok(position)
    }

    pub(super) fn poll_primary(&mut self) -> Result<i64> {
        let sinex = Arc::clone(
            self.sinex
                .as_ref()
                .ok_or_else(|| StoreError::Invalid("this writer is standalone".into()))?,
        );
        let after = crate::primary::projection::position(&self.index)?;
        let identity = crate::primary::cache_identity(&self.index)?;
        let events = sinex.confirmed(after, &identity)?;
        let applied = self.apply_confirmed(&events)?;
        // Staging is durable even when a large position has not reached its barrier.
        sinex.acknowledge(applied)?;
        Ok(applied)
    }

    pub(super) fn synchronize_primary(&mut self) -> Result<i64> {
        self.poll_primary()?;
        self.drain_primary_captures()?;
        self.poll_primary()
    }

    pub(super) fn product_writes(&self, products: Vec<Product>) -> Result<Vec<ProductWrite>> {
        let mut writes = Vec::with_capacity(products.len());
        for (index, product) in products.into_iter().enumerate() {
            let (location, machine) = if let Some(reading) = product.reading() {
                let start = match &product {
                    Product::Message { facts, .. } => facts.offset,
                    Product::Document { facts, .. } => facts.offset,
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
                    Product::RestoredRequest { current, .. } => {
                        (None, Some(current.machine.clone()))
                    }
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

    pub(super) fn primary_products<'a>(
        &mut self,
        batches: impl Iterator<Item = &'a Batch>,
    ) -> Result<()> {
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        let mut pending = None;
        for batch in batches {
            let Change::Reparse { chain } = batch.change else {
                return Err(StoreError::Invalid(
                    "primary parsed products need their confirmed source chain".into(),
                ));
            };
            for step in &batch.parse {
                step.counts.check()?;
                let products = crate::primary::products::parse(&batch.input, chain, step)?;
                let writes = self.product_writes(products)?;
                pending = Some(sinex.prepare_products(&writes)?.identity);
            }
        }
        // A pending producer write is already durable; visibility comes through this poll.
        if let Err(error) = self.poll_primary() {
            return match (error, pending) {
                (StoreError::Sinex(_), Some(identity)) => {
                    Err(StoreError::PrimaryPending { identity })
                }
                (error, _) => Err(error),
            };
        }
        Ok(())
    }

    pub(super) fn primary_reading(
        &mut self,
        input: InputKey,
        chain: i64,
        length: u64,
        mut source: ChainReading,
    ) -> Result<u64> {
        let root:Option<String>=self.index.query_row("SELECT id FROM source_material WHERE input=?1 AND chain=?2 AND offset=0 ORDER BY rowid LIMIT 1",params![input.id,chain],|row|row.get(0)).optional()?;
        let root = root.ok_or_else(|| {
            StoreError::Invalid("the confirmed source chain has no root material".into())
        })?;
        let mut version = None;
        let mut complete = false;
        let mut messages = 0;
        source.visit_source(|step| {
            if version
                .as_ref()
                .is_some_and(|previous| previous != &step.reading_version)
            {
                return Err(StoreError::Invalid(
                    "one replacement has several reading versions".into(),
                ));
            }
            version.get_or_insert_with(|| step.reading_version.clone());
            complete = step.at_end && step.parsed_upto <= length;
            messages += step.output.messages.len() as u64;
            Ok(())
        })?;
        if !complete {
            return Err(StoreError::Invalid(
                "a replacement reading is not complete".into(),
            ));
        }
        let version =
            version.ok_or_else(|| StoreError::Invalid("a replacement has no reading".into()))?;
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        let prepared = sinex.prepare_reading(&root, &version, &mut |emit| {
            source.visit_source(|step| {
                for product in
                    self.product_writes(crate::primary::products::parse(&input, chain, step)?)?
                {
                    emit(product)?;
                }
                Ok(())
            })
        })?;
        if let Err(error) = self.poll_primary() {
            return Err(match error {
                StoreError::Sinex(_) => StoreError::PrimaryPending {
                    identity: prepared.identity,
                },
                error => error,
            });
        }
        Ok(messages)
    }

    pub(super) fn primary_request(
        &mut self,
        request: &polylogue_model::user::UserRequest,
    ) -> Result<polylogue_model::user::UserOutcome> {
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        // The durable local overlay remains available while the stream is behind or offline.
        let prepared: PreparedRequest = crate::primary::authored::prepare(
            &self.index,
            request,
            |apply| sinex.visit_pending_requests(apply),
            |apply| sinex.visit_pending_blobs(apply),
        )?;
        sinex.prepare_request(&prepared)
    }

    pub(super) fn primary_blob(
        &mut self,
        new_blob: crate::NewBlob,
        file: Option<crate::staging::PreparedBytes>,
    ) -> Result<BlobHash> {
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
        let crate::staging::Prepared::Part { file, .. } = prepared.file else {
            return Err(StoreError::Invalid(
                "local segment preparation reached the primary producer".into(),
            ));
        };
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        let prepared = sinex.prepare_blob(new_blob.hash, new_blob.size, &mut file.reopen()?)?;
        // Captured bytes are durable before their temporary preparation is released.
        if let Err(error) = self
            .drain_primary_captures()
            .and_then(|_| self.poll_primary())
        {
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
