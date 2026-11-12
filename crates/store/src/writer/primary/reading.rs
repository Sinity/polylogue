//! Original parser scratch waits for native admission before local resolution.
use super::*;
use crate::primary::{ExportReadingKey, ParsedPublication, ReadingScope};
use std::time::Duration;

impl WriterSender {
    pub(in crate::writer) fn wait_primary_blobs(&self, bodies: &[(BlobHash, u64)]) -> Result<()> {
        if bodies.is_empty() {
            return Ok(());
        }
        let location = self.0.resolver.location();
        let token = location.read_scope.token()?;
        let commits = self.completion_commits();
        loop {
            if self.0.stop.requested() {
                return Err(StoreError::WriteCancelled);
            }
            scoped::require_current(location, Some(&token))?;
            let index = crate::archive::open_read(&location.index_db())?;
            let mut complete = true;
            for (hash, size) in bodies {
                complete &= index.query_row(
                    "SELECT EXISTS(SELECT 1 FROM source_blob WHERE hash=?1 AND size=?2)",
                    params![hash.0.as_slice(), size],
                    |row| row.get::<_, bool>(0),
                )?;
            }
            if complete {
                return Ok(());
            }
            drop(index);
            let _ = commits.recv_timeout(Duration::from_millis(250));
        }
    }

    pub(in crate::writer) fn prepare_primary_batch(&self, batch: &mut Batch) -> Result<()> {
        if batch.parse.is_empty() {
            return Ok(());
        }
        let location = self.0.resolver.location();
        let token = location.read_scope.token()?;
        let commits = self.completion_commits();
        let open = || -> Result<Connection> {
            if self.0.stop.requested() {
                return Err(StoreError::WriteCancelled);
            }
            scoped::require_current(location, Some(&token))?;
            let index = crate::archive::open_read(&location.index_db())?;
            crate::primary::scope::require_readable(&index)?;
            Ok(index)
        };
        let index = open()?;
        let chain = validate_primary_reading(&index, batch)?;
        for blob in &batch.blobs {
            if !index.query_row(
                "SELECT EXISTS(SELECT 1 FROM source_blob WHERE hash=?1 AND size=?2)",
                params![blob.hash.0.as_slice(), blob.size],
                |row| row.get::<_, bool>(0),
            )? {
                return Err(StoreError::Invalid(
                    "decoded bytes need confirmed native custody before interpretation".into(),
                ));
            }
        }
        let root: String = index.query_row(
            "SELECT id FROM source_material WHERE input=?1 AND chain=?2 AND offset=0 ORDER BY rowid LIMIT 1",
            params![batch.input.id,chain], |row| row.get(0),
        )?;
        drop(index);
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
            let writes = {
                let mut index = open()?;
                let snapshot = index.transaction()?;
                let mut writes = product_writes(
                    &snapshot,
                    crate::primary::products::parse(&batch.input, chain, Some(batch.length), step)?,
                )?;
                for write in &mut writes {
                    let record =
                        crate::primary::SourceRecord::of(&write.product)?.expect("source identity");
                    write.publish =
                        crate::primary::products::allowed(&snapshot, &source.reading, &record)?;
                }
                writes
            };
            let receipt = loop {
                open()?;
                match sinex.prepare_parsed(&source, &writes) {
                    Ok(ParsedPublication::Admitted(receipt)) => break receipt,
                    Ok(ParsedPublication::Pending { .. }) | Err(StoreError::Sinex(_)) => {
                        let _ = commits.recv_timeout(Duration::from_millis(250));
                    }
                    Ok(ParsedPublication::Following(_)) => {
                        return Err(StoreError::Invalid(
                            "an append returned a replacement selection".into(),
                        ));
                    }
                    Err(error) => return Err(error),
                }
            };
            loop {
                let mut index = open()?;
                let snapshot = index.transaction()?;
                let applied = crate::primary::projection::checkpoint(&snapshot)?;
                if receipt
                    .commit_seq
                    .is_some_and(|position| applied.is_none_or(|local| local < position))
                {
                    drop(snapshot);
                    drop(index);
                    let _ = commits.recv_timeout(Duration::from_millis(250));
                    continue;
                }
                super::super::source_eligibility::retain_source_records(
                    &snapshot,
                    &batch.input,
                    chain,
                    batch.length,
                    step,
                )?;
                step.publication = Some(receipt);
                break;
            }
        }
        Ok(())
    }

    pub(in crate::writer) fn prepare_primary_reading(
        &self,
        input: &InputKey,
        chain: i64,
        length: u64,
        source: &mut ChainReading,
    ) -> Result<()> {
        let location = self.0.resolver.location();
        let token = location.read_scope.token()?;
        // Subscribe before the first snapshot: native completion can race admission.
        let commits = self.completion_commits();
        let mut version = None;
        let mut complete = false;
        let mut diagnosed = false;
        source.visit_source(|step| {
            step.counts.check()?;
            if step.start > step.parsed_upto || step.parsed_upto > length {
                return Err(StoreError::Invalid(
                    "reading exceeds its original bytes".into(),
                ));
            }
            if version
                .as_ref()
                .is_some_and(|old| old != &step.reading_version)
            {
                return Err(StoreError::Invalid(
                    "one replacement has several reading versions".into(),
                ));
            }
            version.get_or_insert_with(|| step.reading_version.clone());
            complete = step.progress.reached_end() && step.parsed_upto == length;
            diagnosed = crate::wire_shapes::is_refusal(step);
            Ok(())
        })?;
        if !complete && !diagnosed {
            return Err(StoreError::Invalid(
                "a replacement reading is not complete".into(),
            ));
        }
        let key = ExportReadingKey {
            input: input.clone(),
            chain,
            length,
            reading_version: version
                .ok_or_else(|| StoreError::Invalid("a replacement has no reading".into()))?,
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
        let open = || -> Result<Connection> {
            if self.0.stop.requested() {
                return Err(StoreError::WriteCancelled);
            }
            scoped::require_current(location, Some(&token))?;
            let index = crate::archive::open_read(&location.index_db())?;
            crate::primary::scope::require_readable(&index)?;
            Ok(index)
        };
        let pause = || -> Result<()> {
            match commits.recv_timeout(Duration::from_millis(250)) {
                Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(StoreError::WriterGone);
                }
            }
            if self.0.stop.requested() {
                return Err(StoreError::WriteCancelled);
            }
            scoped::require_current(location, Some(&token))
        };
        loop {
            let index = open()?;
            let mut available = true;
            for (hash, size) in &blobs {
                available &= index.query_row(
                    "SELECT EXISTS(SELECT 1 FROM source_blob WHERE hash=?1 AND size=?2)",
                    params![hash.0.as_slice(), size],
                    |row| row.get::<_, bool>(0),
                )?;
            }
            if available {
                break;
            }
            drop(index);
            pause()?;
        }
        let root: String = {
            let index = open()?;
            let exact: bool = index.query_row(
                "SELECT EXISTS(SELECT 1 FROM source_input WHERE id=?1 AND format=?2)",
                params![input.id, input.format],
                |row| row.get(0),
            )?;
            if !exact {
                return Err(StoreError::Invalid("reading input identity differs".into()));
            }
            crate::staging::require_range(&index, "source_", &input.id, chain, 0..length)?;
            index.query_row(
                "SELECT id FROM source_material WHERE input=?1 AND chain=?2 AND offset=0 ORDER BY rowid LIMIT 1",
                params![input.id,chain], |row| row.get(0),
            )?
        };
        let sinex = self.0.sinex.as_ref().expect("primary sender");
        let mut following: Option<ReadingScope> = None;
        let publication = loop {
            if let Some(scope) = &following {
                let index = open()?;
                if crate::primary::products::scope_hash(&index, scope)? != scope.expected_live_hash
                {
                    drop(index);
                    pause()?;
                    continue;
                }
                following = None;
            }
            open()?;
            let result = sinex.prepare_reading(&root, &key, &mut |scope, emit| {
                let mut index = open()?;
                let snapshot = index.transaction()?;
                let live_hash = crate::primary::products::scope_hash(&snapshot, scope)?;
                if live_hash != scope.expected_live_hash {
                    return Ok(live_hash);
                }
                source.visit_source(|step| {
                    for mut write in product_writes(
                        &snapshot,
                        crate::primary::products::parse(input, chain, Some(length), step)?,
                    )? {
                        let reading = write.product.reading().expect("source interpretation");
                        let record = crate::primary::SourceRecord::of(&write.product)?
                            .expect("source identity");
                        write.publish =
                            crate::primary::products::allowed(&snapshot, reading, &record)?;
                        emit(write)?;
                    }
                    Ok(())
                })?;
                Ok(live_hash)
            });
            match result {
                Ok(ParsedPublication::Admitted(receipt)) => break receipt,
                Ok(ParsedPublication::Following(scope)) => following = Some(scope),
                Ok(ParsedPublication::Pending { .. }) | Err(StoreError::Sinex(_)) => pause()?,
                Err(error) => return Err(error),
            }
        };
        loop {
            let mut index = open()?;
            let snapshot = index.transaction()?;
            let applied = crate::primary::projection::checkpoint(&snapshot)?;
            if publication
                .commit_seq
                .is_some_and(|position| applied.is_none_or(|local| local < position))
            {
                drop(snapshot);
                drop(index);
                pause()?;
                continue;
            }
            source.select_source(&snapshot, input, chain, length, publication)?;
            return Ok(());
        }
    }
}
