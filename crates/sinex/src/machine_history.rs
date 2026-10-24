//! A cached historical comparison, never a current material-availability claim.
use super::*;

pub(super) struct Memo {
    pub(super) identity: [u8; 32],
    acknowledgments: [u8; 32],
    evidence: [u8; 32],
}

pub(super) fn record(
    id: &str,
    catalog: Option<&sinex_rows::Catalog>,
    digest: &mut blake3::Hasher,
    candidates: Vec<(u64, Unit)>,
    known: &HashMap<String, Acknowledgment>,
) -> Result<u64> {
    let mut count = 0;
    for (position, unit) in candidates {
        let frame = format!("rows:{id}:{position}");
        if let Some(ack) = known.get(&frame) {
            if let Some(reason) = &ack.refusal {
                return Err(Error::Refused {
                    frame,
                    reason: reason.clone(),
                });
            }
            if let Some(catalog) = catalog {
                let events = ack
                    .admitted
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                catalog.expect_machine_frame(position, &unit, &events)?;
            }
            serde_json::to_writer(&mut *digest, &(position, &unit, ack))?;
            count += 1;
        }
    }
    Ok(count)
}

impl Sinex {
    pub(super) fn machine_identity(&self, id: &str, state: [u8; 32]) -> Result<[u8; 32]> {
        let mut digest = blake3::Hasher::new();
        serde_json::to_writer(
            &mut digest,
            &(id, &self.producer, self.subjects.root(), state),
        )?;
        Ok(*digest.finalize().as_bytes())
    }

    pub(super) async fn check_machine_history(
        &mut self,
        archive: &ArchiveRows,
        id: &str,
        identity: [u8; 32],
        mut acknowledgments: [u8; 32],
        catalog: Option<sinex_rows::Catalog>,
        inventory: &sinex_rows::Inventory<'_>,
    ) -> Result<[u8; 32]> {
        if let Some(memo) = self
            .machine_memo
            .as_ref()
            .filter(|memo| memo.identity == identity && memo.acknowledgments == acknowledgments)
        {
            return Ok(memo.evidence);
        }
        let catalog = if let Some(catalog) = catalog {
            catalog
        } else {
            // A changed candidate acknowledgment invalidated a possible cache
            // hit. Re-read only candidate frames; stable polls staged no history.
            let catalog = sinex_rows::Catalog::new()?;
            let mut digest = blake3::Hasher::new();
            let mut after = 0;
            while after < inventory.len() {
                let end = (after + 500).min(inventory.len());
                let candidates = inventory.machine_candidates(after, end)?;
                if !candidates.is_empty() {
                    let frames = candidates
                        .iter()
                        .map(|(position, _)| format!("rows:{id}:{position}"))
                        .collect::<Vec<_>>();
                    let known = self.have(&frames).await?;
                    record(id, Some(&catalog), &mut digest, candidates, &known)?;
                }
                after = end;
            }
            acknowledgments = *digest.finalize().as_bytes();
            catalog
        };
        let selected = self.confirmed(id, catalog).await?;
        let reader = hashed_blobs::Reader::new(self.store_root.clone(), vec![]);
        let mut evidence = blake3::Hasher::new();
        selected.each_machine_frame(|position,unit| {
            let name=unit.batch.as_ref().ok_or_else(||polylogue_store::StoreError::Invalid(format!("held machine frame rows:{id}:{position} is retired in the confirmed view")))?;
            let hash=parse_hash(name).map_err(|e|polylogue_store::StoreError::Invalid(e.to_string()))?;
            let payload=reader.read_plain(&hash,unit.bytes,||Ok(())).map_err(|e|polylogue_store::StoreError::Invalid(format!("held machine frame rows:{id}:{position} evidence {name} is unavailable: {e}")))?;
            archive.verify_machine_frame(&selected,unit,payload).map_err(|e|polylogue_store::StoreError::Invalid(format!("held machine frame rows:{id}:{position}, verified batch {name}: {e}")))?;
            serde_json::to_writer(&mut evidence,&(position,unit))?;
            Ok(())
        })?;
        let evidence = *evidence.finalize().as_bytes();
        self.machine_memo = Some(Memo {
            identity,
            acknowledgments,
            evidence,
        });
        Ok(evidence)
    }
}
