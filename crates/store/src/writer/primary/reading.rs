//! Local rows in primary mode come from parsing confirmed original bytes. The
//! parsed events of each step are prepared in the producer spool before the
//! rows of the same parse commit; nothing here waits for Sinex to acknowledge
//! them or for them to come back on the confirmed stream.
use super::*;
use crate::primary::ExportReadingKey;
use std::time::Duration;

impl WriterSender {
    /// Decoded attachment bytes are materials. The rows that cite them are
    /// written once the confirmed stream has delivered those materials.
    pub(in crate::writer) fn wait_primary_blobs(&self, bodies: &[(BlobHash, u64)]) -> Result<()> {
        if bodies.is_empty() {
            return Ok(());
        }
        let location = self.0.resolver.location();
        let token = location.read_scope.token()?;
        let commits = self.completion_commits();
        loop {
            if self.0.stop.is_cancelled() {
                return Err(StoreError::Stopped);
            }
            scoped::require_current(location, Some(&token))?;
            let index = crate::archive::open_read(&location.index_db())?;
            if confirmed_blobs(&index, bodies)? {
                return Ok(());
            }
            drop(index);
            let _ = commits.recv_timeout(Duration::from_millis(250));
        }
    }

    /// Prepares the parsed events of every step of a batch in the spool.
    pub(in crate::writer) fn spool_primary_batch(&self, batch: &mut Batch) -> Result<()> {
        if batch.parse.is_empty() {
            return Ok(());
        }
        let location = self.0.resolver.location();
        let token = location.read_scope.token()?;
        let mut index = self.open_current(&token)?;
        let chain = validate_primary_reading(&index, batch)?;
        for blob in &batch.blobs {
            if !confirmed_blobs(&index, &[(blob.hash, blob.size)])? {
                return Err(StoreError::Invalid(
                    "decoded bytes must be confirmed by Sinex before they are parsed".into(),
                ));
            }
        }
        let root = root_material(&index, &batch.input.id, chain)?;
        let sinex = self.0.sinex.as_ref().expect("primary sender");
        for step in &mut batch.parse {
            step.prepare();
            let source = crate::primary::ParsedSource {
                root_material: root.clone(),
                reading: crate::primary::Reading::from_step(
                    &batch.input,
                    chain,
                    Some(batch.length),
                    step,
                ),
            };
            // One read snapshot decides the anchors and what Sinex already holds.
            let snapshot = index.transaction()?;
            let mut writes = Vec::new();
            for write in product_writes(
                &snapshot,
                crate::primary::products::parse(&batch.input, chain, Some(batch.length), step)?,
            )? {
                let record =
                    crate::primary::SourceRecord::of(&write.product)?.expect("source identity");
                if crate::primary::products::needs_sending(&snapshot, &source.reading, &record)? {
                    writes.push(write);
                }
            }
            drop(snapshot);
            if !writes.is_empty() {
                sinex.prepare_parsed(&source, &writes)?;
            }
        }
        Ok(())
    }

    /// Prepares a complete re-parse of one chain for Sinex. When Sinex holds
    /// parsed events of an earlier reading version, a producer replay replaces
    /// them; otherwise the steps are sent as parsed frames.
    pub(in crate::writer) fn spool_primary_reading(
        &self,
        input: &InputKey,
        chain: i64,
        length: u64,
        source: &mut ChainParse,
    ) -> Result<()> {
        let location = self.0.resolver.location();
        let token = location.read_scope.token()?;
        let commits = self.completion_commits();
        let mut version = None;
        let mut complete = false;
        let mut diagnosed = false;
        let mut partial = false;
        let mut parsed_through = 0;
        source.visit_source(|step| {
            step.counts.check()?;
            if step.start > step.parsed_upto || step.parsed_upto > length {
                return Err(StoreError::Invalid(
                    "reading exceeds its original bytes".into(),
                ));
            }
            if version
                .as_ref()
                .is_some_and(|old| old != &step.parse_version)
            {
                return Err(StoreError::Invalid(
                    "one replacement has several parse versions".into(),
                ));
            }
            version.get_or_insert_with(|| step.parse_version.clone());
            parsed_through = parsed_through.max(step.parsed_upto);
            complete = step.progress.reached_end() && step.parsed_upto == length;
            partial = matches!(step.progress, polylogue_model::batch::StepProgress::AtEnd)
                && step.parsed_upto < length;
            diagnosed = crate::wire_shapes::is_refusal(step);
            Ok(())
        })?;
        if !complete && !diagnosed && !partial {
            return Err(StoreError::Invalid(
                "a replacement reading is not complete".into(),
            ));
        }
        let key = ExportReadingKey {
            input: input.clone(),
            chain,
            length,
            parse_version: version.or_invalid("a replacement has no reading")?,
        };
        key.validate()?;

        let decoded = source.take_decoded();
        let blobs: Vec<_> = decoded
            .iter()
            .flat_map(|item| item.batch.new_blob.iter().chain(&item.batch.blobs))
            .map(|blob| (blob.hash, blob.size))
            .collect();
        if !decoded.is_empty() {
            for result in self.store_group(decoded, Arc::new(location.blobs()?))? {
                match result {
                    Ok(_) | Err(StoreError::PrimaryPending { .. }) => {}
                    Err(error) => return Err(error),
                }
            }
        }
        self.wait_primary_blobs(&blobs)?;
        let root = {
            let index = self.open_current(&token)?;
            let known: bool = index.exists(
                "SELECT EXISTS(SELECT 1 FROM source_input WHERE id=?1 AND format=?2)",
                params![input.id, input.format],
            )?;
            if !known {
                return Err(StoreError::Invalid("reading input identity differs".into()));
            }
            crate::staging::require_range(&index, "source_", &input.id, chain, 0..length)?;
            root_material(&index, &input.id, chain)?
        };
        let previous = crate::primary::products::previous_reading_bound(
            &self.open_current(&token)?,
            &input.id,
            chain,
            &key.parse_version,
        )?;
        if let Some(required) = previous.filter(|required| *required > parsed_through) {
            return Err(StoreError::Invalid(format!(
                "the replacement parsed through {parsed_through}, but Sinex's earlier events need source bytes through {required}",
            )));
        }
        if previous.is_none() {
            source.visit_source(|step| {
                let mut batch = Batch {
                    input: input.clone(),
                    change: Change::Reparse { chain },
                    observation: None,
                    continues: false,
                    new_blob: None,
                    blobs: Vec::new(),
                    length,
                    parse: vec![step.clone()],
                    release: Default::default(),
                    capture_read: None,
                    base: None,
                    required_outcome: None,
                };
                self.spool_primary_batch(&mut batch)
            })?;
            source.mark_spooled();
            return Ok(());
        }
        let sinex = self.0.sinex.as_ref().expect("primary sender");
        // The replay names the live set Sinex holds for the chain, so it is
        // prepared once Sinex answers and no earlier frame of the chain is
        // unacknowledged. The local rows do not wait for the replay itself.
        loop {
            self.open_current(&token)?;
            let prepared = sinex.prepare_reading(&root, &key, &mut |emit| {
                let mut index = self.open_current(&token)?;
                let snapshot = index.transaction()?;
                source.visit_source(|step| {
                    for write in product_writes(
                        &snapshot,
                        crate::primary::products::parse(input, chain, Some(length), step)?,
                    )? {
                        emit(write)?;
                    }
                    Ok(())
                })
            });
            match prepared {
                Ok(Some(_)) => break,
                Ok(None) | Err(StoreError::Sinex(_)) => {
                    match commits.recv_timeout(Duration::from_millis(250)) {
                        Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            return Err(StoreError::WriterGone);
                        }
                    }
                }
                Err(error) => return Err(error),
            }
        }
        source.mark_spooled();
        Ok(())
    }

    fn open_current(&self, token: &crate::ReadScopeToken) -> Result<Connection> {
        if self.0.stop.is_cancelled() {
            return Err(StoreError::Stopped);
        }
        let location = self.0.resolver.location();
        scoped::require_current(location, Some(token))?;
        let index = crate::archive::open_read(&location.index_db())?;
        crate::primary::scope::require_readable(&index)?;
        Ok(index)
    }
}

fn confirmed_blobs(index: &Connection, bodies: &[(BlobHash, u64)]) -> Result<bool> {
    for (hash, size) in bodies {
        if !index.scalar::<bool>(
            "SELECT EXISTS(SELECT 1 FROM source_blob WHERE hash=?1 AND size=?2)",
            params![hash.0.as_slice(), size],
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn root_material(index: &Connection, input: &str, chain: i64) -> Result<String> {
    index.scalar(
        "SELECT id FROM source_material WHERE input=?1 AND chain=?2 AND offset=0 ORDER BY rowid LIMIT 1",
        params![input, chain])
}
