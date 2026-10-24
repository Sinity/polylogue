//! One serialized owner for refusal, durable reservation, installation and admission.
use super::*;
use crate::retention::{PreparedBytes, Reconciled, Retained};
use polylogue_model::BlobHash;

impl Inner {
    pub(super) fn retain_bytes(
        &mut self,
        capture: PreparedBytes,
        mut batch: Batch,
    ) -> Result<Retained> {
        if capture.owner != self.location.root
            || batch.piece.is_some()
            || !batch.blobs.is_empty()
            || !batch.parse.is_empty()
        {
            return Err(StoreError::Invalid(
                "byte retention requires one owned private stage and an acquisition-only batch"
                    .into(),
            ));
        }
        let hash = capture.hash();
        let length = capture.len();
        let size = i64::try_from(length).map_err(|_| {
            StoreError::Invalid("captured extent exceeds the archive address space".into())
        })?;
        let offset = batch.length.checked_sub(length).ok_or_else(|| {
            StoreError::Invalid("captured bytes exceed the retained input length".into())
        })?;
        let offset = i64::try_from(offset).map_err(|_| {
            StoreError::Invalid("captured offset exceeds the archive address space".into())
        })?;
        i64::try_from(batch.length).map_err(|_| {
            StoreError::Invalid("retained input exceeds the archive address space".into())
        })?;
        let input = batch.input.id.clone();
        if batch.input.route_rank == 2
            && (input != hash.to_string() || batch.length != length || batch.continues)
        {
            return Err(StoreError::Invalid(
                "immutable imports must name the complete captured bytes".into(),
            ));
        }
        let observation = batch.observation.as_ref().ok_or_else(|| {
            StoreError::Invalid("byte retention needs its acquisition observation".into())
        })?;
        let _publication = crate::publication::write(&self.archive)?;
        if self.refused.raw_import(&input, &hash, observation)? {
            return Err(StoreError::InputRefused { input });
        }
        let previous: String = self.archive.pragma_query_value(None, "synchronous", |r| {
            r.get::<_, i64>(0).map(|n| n.to_string())
        })?;
        self.archive.pragma_update(None, "synchronous", "full")?;
        let result = (|| {
            let known:Option<(i64,i64,i64)>=self.archive.query_row("SELECT number,chain,length FROM revision WHERE input=?1 ORDER BY number DESC LIMIT 1",[&input],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
            let already_stored = if batch.input.route_rank == 2 {
                if let Some((_, chain, stored)) = known {
                    if stored != size {
                        return Err(StoreError::Conflict(
                            "immutable input length differs from the captured bytes".into(),
                        ));
                    }
                    let same:bool=self.archive.query_row("SELECT count(*)=1 AND min(offset)=0 AND min(length)=?3 AND min(blob_offset)=0 AND min(blob)=?4 FROM extent WHERE input=?1 AND chain=?2",params![input,chain,size,&hash.0[..]],|r|r.get(0))?;
                    if !same {
                        return Err(StoreError::Conflict(
                            "immutable input byte basis differs from the acquisition".into(),
                        ));
                    }
                    batch.change = Change::Unchanged;
                    match crate::retention::require_present(&self.archive, "", &input, Some(chain))
                    {
                        Ok(()) => true,
                        Err(StoreError::InputUnfinished { .. }) => false,
                        Err(error) => return Err(error),
                    }
                } else {
                    batch.change = Change::First;
                    false
                }
            } else {
                false
            };
            let stored: Option<(String, String, i64)> = self
                .archive
                .query_row(
                    "SELECT place,state,size FROM blob WHERE hash=?1",
                    [&hash.0[..]],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .optional()?;
            if stored.as_ref().is_some_and(|(_, _, old)| *old != size) {
                return Err(StoreError::Conflict(
                    "blob length differs from its captured identity".into(),
                ));
            }
            let inline = stored
                .as_ref()
                .is_some_and(|(place, state, _)| place == "inline" && state == "present");
            let ready = if inline {
                let bytes = crate::blobs::read(&self.archive, &capture.store, &hash)?;
                if bytes.len() as u64 != length || BlobHash::of(&bytes) != hash {
                    return Err(StoreError::BlobCorrupt(hash));
                }
                true
            } else if stored
                .as_ref()
                .is_some_and(|(_, state, _)| state == "present")
            {
                match capture.store.verify_plain::<StoreError>(
                    &hashed_blobs::Hash(hash.0),
                    length,
                    |_| Ok(()),
                ) {
                    Ok(_) => true,
                    Err(StoreError::Blobs(hashed_blobs::Error::Missing(_))) => false,
                    Err(error) => return Err(error),
                }
            } else {
                false
            };
            let tx = self.archive.transaction()?;
            let seq = bump_counter(&tx)?;
            tx.execute("INSERT OR IGNORE INTO blob(hash,size,stored_size,place,encoding,state) VALUES (?1,?2,0,'file','plain','writing')",params![&hash.0[..],size])?;
            if !ready {
                tx.execute(
                    "UPDATE blob SET state='writing' WHERE hash=?1",
                    [&hash.0[..]],
                )?;
            }
            if !matches!(batch.change, Change::Unchanged) || known.is_none() {
                batch.piece = Some(Piece {
                    hash,
                    size: length,
                    stored: Stored::File { stored_size: 0 },
                });
            }
            let chain = write_bytes(&tx, &batch, seq)?.ok_or_else(|| {
                StoreError::Invalid("byte retention produced no revision chain".into())
            })?;
            let owned:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM extent WHERE input=?1 AND chain=?2 AND blob=?3 AND blob_offset=0 AND length=?4 AND offset=?5)",params![input,chain,&hash.0[..],size,offset],|row|row.get(0))?;
            if !owned {
                return Err(StoreError::Invalid(
                    "acquisition does not retain its prepared bytes".into(),
                ));
            }
            crate::fault::check("import.reserve_commit", &input)?;
            tx.commit()?;
            self.report.batches += 1;
            self.report.transactions += 1;
            let finish = (|| {
                crate::fault::check("import.reserved", &input)?;
                if ready {
                    drop(capture.prepared);
                } else {
                    let installed = capture.prepared.install_checked::<StoreError>(|| {
                        crate::fault::check("import.install", &input)
                    })?;
                    if installed.0 != hash.0 {
                        return Err(StoreError::BlobCorrupt(hash));
                    }
                }
                crate::fault::check("import.installed", &input)?;
                let stored_size = if inline {
                    None
                } else {
                    Some(
                        i64::try_from(capture.store.stored_size(&hashed_blobs::Hash(hash.0))?)
                            .map_err(|_| {
                                StoreError::Invalid(
                                    "encoded blob exceeds archive address space".into(),
                                )
                            })?,
                    )
                };
                let tx = self.archive.transaction()?;
                tx.execute("UPDATE blob SET state='present',stored_size=COALESCE(?2,stored_size) WHERE hash=?1",params![&hash.0[..],stored_size])?;
                crate::fault::check("import.admit_commit", &input)?;
                tx.commit()?;
                self.report.transactions += 1;
                Ok(Retained {
                    input: input.clone(),
                    chain,
                    length,
                    already_stored,
                })
            })();
            finish.map_err(|error: StoreError| StoreError::InputUnfinished {
                input: input.clone(),
                reason: error.to_string(),
            })
        })();
        let restored = self.archive.pragma_update(None, "synchronous", previous);
        match (result, restored) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), _) => Err(error),
            (_, Err(error)) => Err(error.into()),
        }
    }

    pub(super) fn reconcile_retained_bytes(
        &mut self,
        store: &hashed_blobs::Store,
    ) -> Result<Reconciled> {
        let _publication = crate::publication::write(&self.archive)?;
        let pending = crate::retention::unfinished(&self.archive, "")?;
        let mut report = Reconciled::default();
        for mut pending in pending {
            let hash = hashed_blobs::Hash(pending.blob.0);
            let writing: bool = self.archive.query_row(
                "SELECT state='writing' FROM blob WHERE hash=?1",
                [&hash.0[..]],
                |row| row.get(0),
            )?;
            if !writing {
                continue;
            }
            match store.verify_plain::<StoreError>(&hash, pending.length, |_| Ok(())) {
                Ok(_) => {}
                Err(StoreError::Blobs(hashed_blobs::Error::Missing(_))) => {
                    pending.state = crate::retention::UnfinishedState::Missing;
                    report.missing.push(pending);
                    continue;
                }
                Err(error) => return Err(error),
            }
            let size = i64::try_from(store.stored_size(&hash)?).map_err(|_| {
                StoreError::Invalid("encoded blob exceeds archive address space".into())
            })?;
            let previous: i64 = self
                .archive
                .pragma_query_value(None, "synchronous", |r| r.get(0))?;
            self.archive.pragma_update(None, "synchronous", "full")?;
            let result: Result<()> = (|| {
                let tx = self.archive.transaction()?;
                tx.execute("UPDATE blob SET state='present',stored_size=?2 WHERE hash=?1 AND state='writing'",params![&hash.0[..],size])?;
                tx.commit()?;
                Ok(())
            })();
            self.archive.pragma_update(None, "synchronous", previous)?;
            result?;
            self.report.transactions += 1;
            report.completed += 1;
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests;
