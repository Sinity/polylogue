//! A cached historical comparison, never a current material-availability claim.
use super::*;

pub(super) struct Memo {
    key: [u8; 32],
    evidence: [u8; 32],
}

impl Sinex {
    pub(super) async fn check_machine_history(
        &mut self,
        archive: &ArchiveRows,
        id: &str,
        state: [u8; 32],
        acknowledgments: [u8; 32],
        catalog: sinex_rows::Catalog,
    ) -> Result<[u8; 32]> {
        let mut key = blake3::Hasher::new();
        serde_json::to_writer(
            &mut key,
            &(
                id,
                &self.producer,
                self.subjects.root(),
                state,
                acknowledgments,
            ),
        )?;
        let key = *key.finalize().as_bytes();
        if let Some(memo) = self.machine_memo.as_ref().filter(|memo| memo.key == key) {
            return Ok(memo.evidence);
        }
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
        self.machine_memo = Some(Memo { key, evidence });
        Ok(evidence)
    }
}
