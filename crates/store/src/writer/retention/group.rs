//! Reserve a bounded worker group before any of its private files are installed.
use super::*;
use crate::retention::PreparedBatch;
use std::collections::{BTreeMap, BTreeSet};

struct Reservation {
    retained: Vec<Retained>,
    ready: BTreeSet<BlobHash>,
    checked: BTreeSet<crate::retention::extracted::SourceKey>,
}

impl Inner {
    pub(in crate::writer) fn retain_group(
        &mut self,
        mut group: Vec<PreparedBatch>,
        store: std::sync::Arc<hashed_blobs::Store>,
    ) -> Vec<Result<Retained>> {
        if group.is_empty() {
            return Vec::new();
        }
        let publication = match crate::publication::write(&self.archive) {
            Ok(publication) => publication,
            Err(error) => return errors(&group, error),
        };
        let previous: i64 = match self
            .archive
            .pragma_query_value(None, "synchronous", |r| r.get(0))
        {
            Ok(value) => value,
            Err(error) => return errors(&group, error.into()),
        };
        if let Err(error) = self.archive.pragma_update(None, "synchronous", "full") {
            return errors(&group, error.into());
        }
        let reserved = self.reserve_group(&mut group, &store);
        let Reservation {
            retained,
            ready,
            mut checked,
        } = match reserved {
            Ok(value) => value,
            Err(error) => {
                let _ = self.archive.pragma_update(None, "synchronous", previous);
                drop(publication);
                // A rejected reservation installed nothing. Retry independent
                // inputs separately, as ordinary Writer groups already do.
                if group.len() > 1 {
                    return group
                        .into_iter()
                        .flat_map(|batch| {
                            self.retain_group(vec![batch], std::sync::Arc::clone(&store))
                        })
                        .collect();
                }
                return errors(&group, error);
            }
        };
        let raw: BTreeSet<_> = group
            .iter()
            .flat_map(|g| {
                g.files
                    .iter()
                    .filter(|file| {
                        g.batch
                            .piece
                            .as_ref()
                            .is_some_and(|p| p.hash == file.hash())
                            || !g.extracted.iter().any(|x| x.blob == file.hash())
                    })
                    .map(|file| file.hash())
            })
            .collect();
        let mut outcomes: Vec<Option<String>> = group
            .iter()
            .map(|item| {
                crate::fault::check("import.reserved", &item.batch.input.id)
                    .err()
                    .map(|e| e.to_string())
            })
            .collect();
        let permitted_raw: BTreeSet<_> = group
            .iter()
            .zip(&outcomes)
            .filter(|(_, error)| error.is_none())
            .flat_map(|(item, _)| {
                item.files
                    .iter()
                    .filter(|file| {
                        item.batch
                            .piece
                            .as_ref()
                            .is_some_and(|p| p.hash == file.hash())
                            || !item.extracted.iter().any(|x| x.blob == file.hash())
                    })
                    .map(|file| file.hash())
            })
            .collect();
        let mut rejected = BTreeSet::new();
        let mut files = BTreeMap::new();
        for item in &mut group {
            for file in item.files.drain(..) {
                files.entry(file.hash()).or_insert(file);
            }
        }
        let mut installed = ready;
        let mut failed = BTreeMap::<BlobHash, String>::new();
        for hash in &raw {
            if permitted_raw.contains(hash)
                && let Some(file) = files.remove(hash)
            {
                install(file, &mut installed, &mut failed, &group);
            }
        }
        for (index, (item, retained)) in group.iter().zip(&retained).enumerate() {
            if outcomes[index].is_some() {
                continue;
            }
            let result = (|| {
                if let Some(piece) = &item.batch.piece
                    && let Some(error) = failed.get(&piece.hash)
                {
                    return Err(StoreError::Invalid(error.clone()));
                }
                for extraction in &item.extracted {
                    crate::retention::extracted::check_once(
                        &self.archive,
                        &store,
                        &item.batch.input.id,
                        retained.chain,
                        extraction,
                        &mut checked,
                    )?;
                }
                Ok(())
            })();
            if let Err(error) = result {
                if matches!(error, StoreError::Conflict(_)) {
                    rejected.insert(index);
                }
                outcomes[index] = Some(error.to_string());
            }
        }
        // Source authority is checked before decoded installation. A stale
        // in-flight parser cannot republish removed decoded content.
        let permitted: BTreeSet<_> = group
            .iter()
            .zip(&outcomes)
            .filter(|(_, error)| error.is_none())
            .flat_map(|(item, _)| item.extracted.iter().map(|x| x.blob))
            .collect();
        for (hash, file) in files {
            if permitted.contains(&hash) {
                install(file, &mut installed, &mut failed, &group);
            }
        }
        for (index, item) in group.iter().enumerate() {
            if outcomes[index].is_none() {
                outcomes[index] = item
                    .extracted
                    .iter()
                    .find_map(|x| failed.get(&x.blob).cloned());
            }
        }
        let finalized: Result<()> = (|| {
            let tx = self.archive.transaction()?;
            for hash in &installed {
                let place: String =
                    tx.query_row("SELECT place FROM blob WHERE hash=?1", [&hash.0[..]], |r| {
                        r.get(0)
                    })?;
                let size = if place == "file" {
                    Some(
                        i64::try_from(store.stored_size(&hashed_blobs::Hash(hash.0))?).map_err(
                            |_| {
                                StoreError::Invalid(
                                    "encoded blob exceeds archive address space".into(),
                                )
                            },
                        )?,
                    )
                } else {
                    None
                };
                tx.execute("UPDATE blob SET state='present',stored_size=COALESCE(?2,stored_size) WHERE hash=?1",params![&hash.0[..],size])?;
            }
            for (index, item) in group.iter().enumerate() {
                if outcomes[index].is_none() {
                    crate::fault::check("import.installed", &item.batch.input.id)?;
                    for extraction in &item.extracted {
                        tx.execute("UPDATE extracted_blob SET state='present' WHERE input=?1 AND chain=?2 AND source_start=?3 AND source_end=?4 AND source_hash=?5 AND parser_version=?6 AND blob=?7",params![item.batch.input.id,retained[index].chain,extraction.range.start as i64,extraction.range.end as i64,&extraction.source_hash.0[..],extraction.parser_version,&extraction.blob.0[..]])?;
                    }
                    crate::fault::check("import.admit_commit", &item.batch.input.id)?;
                } else if rejected.contains(&index) {
                    for extraction in &item.extracted {
                        tx.execute("DELETE FROM extracted_blob WHERE input=?1 AND chain=?2 AND source_start=?3 AND source_end=?4 AND source_hash=?5 AND parser_version=?6 AND blob=?7 AND state='writing'",params![item.batch.input.id,retained[index].chain,extraction.range.start as i64,extraction.range.end as i64,&extraction.source_hash.0[..],extraction.parser_version,&extraction.blob.0[..]])?;
                    }
                }
            }
            // Canonical raw reservations survive. A refused decoded stage
            // never became public; remove only its uninstalled row.
            for (index, item) in group.iter().enumerate() {
                if rejected.contains(&index) {
                    for extraction in &item.extracted {
                        if !store.contains(&hashed_blobs::Hash(extraction.blob.0)) {
                            tx.execute("DELETE FROM blob WHERE hash=?1 AND state='writing' AND NOT EXISTS(SELECT 1 FROM extent WHERE extent.blob=blob.hash) AND NOT EXISTS(SELECT 1 FROM extracted_blob x WHERE x.blob=blob.hash)",[&extraction.blob.0[..]])?;
                        }
                    }
                }
            }
            tx.commit()?;
            self.report.transactions += 1;
            Ok(())
        })();
        if let Err(error) = finalized {
            for outcome in &mut outcomes {
                if outcome.is_none() {
                    *outcome = Some(error.to_string());
                }
            }
        }
        let restored = self.archive.pragma_update(None, "synchronous", previous);
        if let Err(error) = restored {
            for outcome in &mut outcomes {
                if outcome.is_none() {
                    *outcome = Some(error.to_string());
                }
            }
        }
        drop(publication);
        let mut parsed = Vec::new();
        for ((mut item, retained), outcome) in group.into_iter().zip(&retained).zip(&outcomes) {
            if outcome.is_none() && !item.batch.parse.is_empty() {
                item.batch.change = Change::Reparse {
                    chain: retained.chain,
                };
                item.batch.continues = false;
                item.batch.piece = None;
                item.batch.observation = None;
                item.batch.blobs.clear();
                parsed.push(Box::new(item.batch));
            }
        }
        if !parsed.is_empty() {
            self.write_group(&parsed);
        }
        drop(parsed);
        retained
            .into_iter()
            .zip(outcomes)
            .map(|(retained, error)| {
                if let Some(reason) = error {
                    Err(StoreError::InputUnfinished {
                        input: retained.input,
                        reason,
                    })
                } else if self.report.failed_inputs.contains(&retained.input) {
                    Err(StoreError::Invalid(format!(
                        "input {}: parsed rows were refused",
                        retained.input
                    )))
                } else {
                    Ok(retained)
                }
            })
            .collect()
    }

    fn reserve_group(
        &mut self,
        group: &mut [PreparedBatch],
        store: &hashed_blobs::Store,
    ) -> Result<Reservation> {
        let mut checked = BTreeSet::new();
        let mut inputs = BTreeSet::new();
        let mut ready = BTreeSet::new();
        for item in group.iter_mut() {
            let b = &mut item.batch;
            check_import_format(
                &self.archive,
                &b.input.id,
                b.observation.as_ref().and_then(|o| o.import.as_ref()),
            )?;
            if !inputs.insert(b.input.id.clone()) {
                return Err(StoreError::Invalid(
                    "one retention group repeats an input".into(),
                ));
            }
            i64::try_from(b.length).map_err(|_| {
                StoreError::Invalid("retained input exceeds archive address space".into())
            })?;
            if b.input.id.is_empty() {
                return Err(StoreError::Invalid(
                    "retention needs an input identity".into(),
                ));
            }
            for piece in b.piece.iter().chain(&b.blobs) {
                let size = usize::try_from(piece.size).map_err(|_| {
                    StoreError::Invalid("retained piece exceeds address space".into())
                })?;
                i64::try_from(piece.size).map_err(|_| {
                    StoreError::Invalid("retained piece exceeds archive address space".into())
                })?;
                match &piece.stored {
                    Stored::Inline(encoded) => {
                        let bytes = zstd::bulk::decompress(encoded, size)?;
                        if bytes.len() != size || BlobHash::of(&bytes) != piece.hash {
                            return Err(StoreError::BlobCorrupt(piece.hash));
                        }
                    }
                    Stored::File { .. } => {
                        if !item
                            .files
                            .iter()
                            .any(|file| file.hash() == piece.hash && file.len() == piece.size)
                        {
                            return Err(StoreError::Invalid(
                                "file piece has no matching private stage".into(),
                            ));
                        }
                    }
                }
            }
            for piece in &b.blobs {
                if !item.extracted.iter().any(|x| x.blob == piece.hash) {
                    return Err(StoreError::Invalid(
                        "decoded piece has no source association".into(),
                    ));
                }
            }
            for extraction in &item.extracted {
                if !b.blobs.iter().any(|p| p.hash == extraction.blob) {
                    return Err(StoreError::Invalid(
                        "decoded association has no matching piece".into(),
                    ));
                }
            }
            for file in &item.files {
                if !b
                    .piece
                    .iter()
                    .chain(&b.blobs)
                    .any(|piece| piece.hash == file.hash() && piece.size == file.len())
                {
                    return Err(StoreError::Invalid(
                        "private stage has no source owner".into(),
                    ));
                }
                if file.owner != self.location.root {
                    return Err(StoreError::Invalid(
                        "private stage belongs to another archive".into(),
                    ));
                }
            }
            if let Some(observation) = &b.observation {
                for piece in b.piece.iter().chain(&b.blobs) {
                    if self
                        .refused
                        .raw_import(&b.input.id, &piece.hash, observation)?
                    {
                        return Err(StoreError::InputRefused {
                            input: b.input.id.clone(),
                        });
                    }
                }
            }
            let known: Option<(i64, i64)> = self
                .archive
                .query_row(
                    "SELECT chain,length FROM revision WHERE input=?1 ORDER BY number DESC LIMIT 1",
                    [&b.input.id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            if b.input.route_rank == 2 && b.piece.is_some() {
                let p = b.piece.as_ref().unwrap();
                if b.input.id != p.hash.to_string() || p.size != b.length || b.continues {
                    return Err(StoreError::Invalid(
                        "immutable imports must name the complete captured bytes".into(),
                    ));
                }
                if let Some((chain, length)) = known {
                    let same:bool=self.archive.query_row("SELECT ?3=?5 AND count(*)=1 AND min(offset)=0 AND min(length)=?3 AND min(blob_offset)=0 AND min(blob)=?4 FROM extent WHERE input=?1 AND chain=?2",params![b.input.id,chain,length,&p.hash.0[..],b.length as i64],|r|r.get(0))?;
                    if !same {
                        return Err(StoreError::Conflict(
                            "immutable input byte basis differs from acquisition".into(),
                        ));
                    }
                    b.change = Change::Unchanged;
                    b.piece = None;
                }
            }
            // Already retained sources can be rejected before reserving anything.
            if matches!(b.change, Change::Reparse { .. } | Change::Unchanged) && b.piece.is_none() {
                let chain = match b.change {
                    Change::Reparse { chain } => chain,
                    _ => {
                        known
                            .ok_or_else(|| {
                                StoreError::Invalid("unchanged input has no retained chain".into())
                            })?
                            .0
                    }
                };
                for extraction in &item.extracted {
                    crate::retention::extracted::check_once(
                        &self.archive,
                        store,
                        &b.input.id,
                        chain,
                        extraction,
                        &mut checked,
                    )?;
                }
            }
            for piece in b.piece.iter().chain(&b.blobs) {
                let old: Option<(i64, String, String)> = self
                    .archive
                    .query_row(
                        "SELECT size,place,state FROM blob WHERE hash=?1",
                        [&piece.hash.0[..]],
                        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                    )
                    .optional()?;
                if let Some((size, place, state)) = old {
                    if size != piece.size as i64 {
                        return Err(StoreError::Conflict(
                            "retained blob length differs from its hash".into(),
                        ));
                    }
                    if state == "present"
                        || place == "inline"
                        || matches!(piece.stored, Stored::Inline(_))
                    {
                        let valid = if place == "inline" {
                            let bytes = crate::blobs::read(&self.archive, store, &piece.hash)?;
                            BlobHash::of(&bytes) == piece.hash && bytes.len() as u64 == piece.size
                        } else {
                            match store.verify_plain::<StoreError>(
                                &hashed_blobs::Hash(piece.hash.0),
                                piece.size,
                                |_| Ok(()),
                            ) {
                                Ok(_) => true,
                                Err(StoreError::Blobs(hashed_blobs::Error::Missing(_))) => false,
                                Err(error) => return Err(error),
                            }
                        };
                        if valid {
                            ready.insert(piece.hash);
                        }
                    }
                }
            }
        }
        // A duplicate immutable acquisition may have cleared its piece. Its
        // stage still names the exact retained extent and can complete a retry.
        for item in group.iter() {
            for file in &item.files {
                let old: Option<(String, String)> = self
                    .archive
                    .query_row(
                        "SELECT place,state FROM blob WHERE hash=?1",
                        [&file.hash().0[..]],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()?;
                if let Some((place, _state)) = old {
                    if place == "inline" {
                        let bytes = crate::blobs::read(&self.archive, store, &file.hash())?;
                        if BlobHash::of(&bytes) != file.hash() || bytes.len() as u64 != file.len() {
                            return Err(StoreError::BlobCorrupt(file.hash()));
                        }
                        ready.insert(file.hash());
                    } else {
                        match store.verify_plain::<StoreError>(
                            &hashed_blobs::Hash(file.hash().0),
                            file.len(),
                            |_| Ok(()),
                        ) {
                            Ok(_) => {
                                ready.insert(file.hash());
                            }
                            Err(StoreError::Blobs(hashed_blobs::Error::Missing(_))) => {}
                            Err(error) => return Err(error),
                        }
                    }
                }
            }
        }
        let tx = self.archive.transaction()?;
        let seq = bump_counter(&tx)?;
        for item in group.iter() {
            for file in &item.files {
                tx.execute("INSERT OR IGNORE INTO blob(hash,size,stored_size,place,encoding,state) VALUES (?1,?2,0,'file','plain','writing')",params![&file.hash().0[..],file.len() as i64])?;
                if !ready.contains(&file.hash()) {
                    tx.execute(
                        "UPDATE blob SET state='writing' WHERE hash=?1",
                        [&file.hash().0[..]],
                    )?;
                }
            }
        }
        let mut retained = Vec::new();
        for item in group.iter() {
            let b = &item.batch;
            for piece in b.piece.iter().chain(&b.blobs) {
                if let Stored::Inline(bytes) = &piece.stored
                    && !ready.contains(&piece.hash)
                {
                    tx.execute("UPDATE blob SET place='inline',encoding='plain',base=NULL,data=?2,stored_size=?3,state='present' WHERE hash=?1",params![&piece.hash.0[..],bytes,bytes.len() as i64])?;
                }
            }
            let chain = write_bytes(&tx, b, seq)?
                .ok_or_else(|| StoreError::Invalid("retention produced no source chain".into()))?;
            for file in &item.files {
                if !item.extracted.iter().any(|x| x.blob == file.hash()) {
                    let offset = b.length.checked_sub(file.len()).ok_or_else(|| {
                        StoreError::Invalid("captured bytes exceed input length".into())
                    })?;
                    let owned:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM extent WHERE input=?1 AND chain=?2 AND offset=?3 AND length=?4 AND blob_offset=0 AND blob=?5)",params![b.input.id,chain,offset as i64,file.len() as i64,&file.hash().0[..]],|r|r.get(0))?;
                    if !owned {
                        return Err(StoreError::Invalid(
                            "private stage is not retained by its source extent".into(),
                        ));
                    }
                }
            }
            for extraction in &item.extracted {
                if extraction.range.start >= extraction.range.end || extraction.range.end > b.length
                {
                    return Err(StoreError::Invalid(
                        "decoded bytes have an invalid source envelope".into(),
                    ));
                }
                let owned:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM revision WHERE input=?1 AND chain=?2 AND length>=?3)",params![b.input.id,chain,extraction.range.end as i64],|r|r.get(0))?;
                if !owned {
                    return Err(StoreError::Invalid(
                        "decoded source range is not retained".into(),
                    ));
                }
                tx.execute("INSERT OR IGNORE INTO extracted_blob VALUES (?1,?2,?3,?4,?5,?6,?7,'writing',?8)",params![b.input.id,chain,extraction.range.start as i64,extraction.range.end as i64,&extraction.source_hash.0[..],extraction.parser_version,&extraction.blob.0[..],seq])?;
            }
            crate::fault::check("import.reserve_commit", &b.input.id)?;
            retained.push(Retained {
                input: b.input.id.clone(),
                chain,
                length: b.length,
                already_stored: matches!(b.change, Change::Unchanged)
                    && match crate::retention::require_present(&tx, "", &b.input.id, Some(chain)) {
                        Ok(()) => true,
                        Err(StoreError::InputUnfinished { .. }) => false,
                        Err(error) => return Err(error),
                    },
            });
        }
        tx.commit()?;
        self.report.transactions += 1;
        self.report.batches += group.iter().filter(|g| g.batch.parse.is_empty()).count() as u64;
        Ok(Reservation {
            retained,
            ready,
            checked,
        })
    }
}

fn errors(group: &[PreparedBatch], error: StoreError) -> Vec<Result<Retained>> {
    if group.len() == 1 {
        return vec![Err(error)];
    }
    let reason = error.to_string();
    group
        .iter()
        .map(|_| Err(StoreError::Invalid(reason.clone())))
        .collect()
}
fn install(
    file: PreparedBytes,
    installed: &mut BTreeSet<BlobHash>,
    failed: &mut BTreeMap<BlobHash, String>,
    group: &[PreparedBatch],
) {
    let hash = file.hash();
    if installed.contains(&hash) {
        return;
    }
    let subject = group
        .iter()
        .find(|g| {
            g.batch
                .piece
                .iter()
                .chain(&g.batch.blobs)
                .any(|p| p.hash == hash)
        })
        .map(|g| g.batch.input.id.as_str())
        .unwrap_or("");
    match file
        .prepared
        .install_checked::<StoreError>(|| crate::fault::check("import.install", subject))
    {
        Ok(actual) if actual.0 == hash.0 => {
            installed.insert(hash);
        }
        Ok(_) => {
            failed.insert(hash, "installed bytes differ from reserved identity".into());
        }
        Err(error) => {
            failed.insert(hash, error.to_string());
        }
    }
}
