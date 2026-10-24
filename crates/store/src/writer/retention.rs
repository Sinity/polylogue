//! One serialized owner for refusal, durable reservation, installation and admission.
use super::*;
use crate::retention::{PreparedBytes, Reconciled, Retained};
use polylogue_model::BlobHash;

mod group;

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
            || batch.observation.is_none()
        {
            return Err(StoreError::Invalid(
                "byte retention requires one owned private stage and an acquisition-only batch"
                    .into(),
            ));
        }
        batch.piece = Some(Piece {
            hash: capture.hash(),
            size: capture.len(),
            stored: Stored::File { stored_size: 0 },
        });
        let store = std::sync::Arc::clone(&capture.store);
        self.retain_group(
            vec![crate::retention::PreparedBatch {
                batch,
                files: vec![capture],
                extracted: Vec::new(),
            }],
            store,
        )
        .pop()
        .expect("one retention result")
    }

    pub(super) fn reconcile_retained_bytes(
        &mut self,
        store: &hashed_blobs::Store,
    ) -> Result<Reconciled> {
        let _publication = crate::publication::write(&self.archive)?;
        let pending = crate::retention::unfinished(&self.archive, "")?;
        let mut report = Reconciled::default();
        let mut verified = std::collections::BTreeMap::new();
        let mut missing = std::collections::BTreeSet::new();
        for item in &pending {
            let state: String = self.archive.query_row(
                "SELECT state FROM blob WHERE hash=?1",
                [&item.blob.0[..]],
                |r| r.get(0),
            )?;
            if state == "present"
                || verified.contains_key(&item.blob)
                || missing.contains(&item.blob)
            {
                continue;
            }
            let hash = hashed_blobs::Hash(item.blob.0);
            match store.verify_plain::<StoreError>(&hash, item.length, |_| Ok(())) {
                Ok(_) => {
                    verified.insert(
                        item.blob,
                        i64::try_from(store.stored_size(&hash)?).map_err(|_| {
                            StoreError::Invalid("encoded blob exceeds archive address space".into())
                        })?,
                    );
                }
                Err(StoreError::Blobs(hashed_blobs::Error::Missing(_))) => {
                    missing.insert(item.blob);
                }
                Err(error) => return Err(error),
            }
        }
        let previous: i64 = self
            .archive
            .pragma_query_value(None, "synchronous", |r| r.get(0))?;
        self.archive.pragma_update(None, "synchronous", "full")?;
        let result: Result<()> = (|| {
            let tx = self.archive.transaction()?;
            for (hash, size) in &verified {
                tx.execute(
                    "UPDATE blob SET state='present',stored_size=?2 WHERE hash=?1",
                    params![&hash.0[..], size],
                )?;
                report.completed += 1;
            }
            let mut stmt=tx.prepare(&format!("SELECT {} FROM extracted_blob x WHERE x.state='writing' ORDER BY input,chain,source_start,source_end",crate::retention::extracted::key_sql("x")))?;
            let keys: Vec<String> = stmt
                .query_map([], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            drop(stmt);
            for value in keys {
                let key = crate::retention::extracted::Key::parse(&value)?;
                if matches!(
                    crate::retention::require_range(
                        &tx,
                        "",
                        &key.input,
                        key.chain,
                        key.start as u64..key.end as u64
                    ),
                    Err(StoreError::InputUnfinished { .. })
                ) {
                    continue;
                }
                crate::retention::extracted::check_source(
                    &tx,
                    store,
                    &key.input,
                    key.chain,
                    &key.extraction(),
                )?;
                let present: bool = tx.query_row(
                    "SELECT state='present' FROM blob WHERE hash=?1",
                    [&key.blob.0[..]],
                    |r| r.get(0),
                )?;
                if present {
                    tx.execute(
                        &format!(
                            "UPDATE extracted_blob SET state='present' WHERE {}",
                            crate::retention::extracted::MATCH
                        ),
                        rusqlite::params_from_iter(key.args().iter()),
                    )?;
                    report.completed += 1;
                }
            }
            tx.commit()?;
            if report.completed > 0 {
                self.report.transactions += 1;
            }
            Ok(())
        })();
        let restored = self.archive.pragma_update(None, "synchronous", previous);
        result?;
        restored?;
        report.unfinished = crate::retention::unfinished(&self.archive, "")?;
        for item in &mut report.unfinished {
            if missing.contains(&item.blob) {
                item.state = crate::retention::UnfinishedState::Missing;
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests;
