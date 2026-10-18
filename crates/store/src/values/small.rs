//! A single-chunk JSON value hashed by the worker before the write
//! transaction; `install_in` stores and seals it with the usual dedup rules.
use super::*;
pub(crate) struct Prepared {
    bytes: Vec<u8>,
    digest: [u8; 32],
    hasher: blake3::Hasher,
}
impl Prepared {
    pub(crate) fn json(bytes: Vec<u8>, allowance: usize) -> Result<Self> {
        if bytes.len() > MAX_CHUNK || bytes.len() > allowance {
            return Err(admission("small field chunk bytes", bytes.len()));
        }
        let mut utf8 = Utf8::default();
        utf8.update(&bytes)?;
        if utf8.len != 0 {
            return Err(invalid("payload ends in an incomplete UTF-8 codepoint"));
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(&bytes);
        let digest = *hasher.finalize().as_bytes();
        Ok(Self {
            bytes,
            digest,
            hasher,
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }
    pub(crate) fn hash(&self) -> [u8; 32] {
        self.digest
    }
    pub(crate) fn install_in(&self, tx: &Transaction<'_>, path: &Path) -> Result<ReadyValue> {
        let h = begin_in(tx, path, Codec::Json, self.bytes.len() as u64, 1)?;
        if !self.bytes.is_empty() {
            crate::db(path, || {
                tx.execute(
                    "INSERT INTO body_value_chunk(value_pk,offset,bytes) VALUES(?1,0,?2)",
                    params![h.id.pk, &self.bytes],
                )
            })?;
        }
        let proof = Validation {
            id: h.id,
            codec: Codec::Json,
            length: self.bytes.len() as u64,
            offset: self.bytes.len() as u64,
            chunk: None,
            hasher: self.hasher.clone(),
            digest: Some(self.digest),
            utf8: Utf8::default(),
            failed: false,
        };
        Ok(seal_in(tx, path, &proof, |tx, ready| {
            if ready.id != proof.id {
                crate::db(path, || {
                    tx.execute(
                        "UPDATE body_value SET incoming=incoming+1 WHERE pk=?1 AND state='ready'",
                        [ready.id.pk],
                    )
                })?;
                if crate::db(path, || {
                    tx.execute(
                        "UPDATE body_value SET incoming=incoming-1 WHERE pk=?1 AND incoming=1",
                        [proof.id.pk],
                    )
                })? != 1
                {
                    return Err(Error::StaleValue);
                }
            }
            Ok(())
        })?
        .ready)
    }
}
