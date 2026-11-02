//! Exact object preparation uses pinned retained evidence.
use super::Reader;
use crate::{ReadScopeToken, Result, StoreError};
use polylogue_model::{BlobHash, Reference, bundle::Record};
use std::{io::Write, marker::PhantomData};

/// Store-derived object content and provenance, bound to the preparing Reader.
/// No wire decoder or public constructor can certify a caller's digest.
pub struct ObjectRead<'read> {
    record: Record,
    digest: BlobHash,
    scope: ReadScopeToken,
    _reader: PhantomData<&'read ()>,
}
impl ObjectRead<'_> {
    pub fn record(&self) -> Result<&Record> {
        self.check()?;
        Ok(&self.record)
    }
    pub fn content_hash(&self) -> BlobHash {
        self.digest
    }
    pub fn into_record(self) -> Result<Record> {
        self.check()?;
        Ok(self.record)
    }
    pub fn check(&self) -> Result<()> {
        self.scope.check()?;
        if let Some(limit) = super::limit::ReadLimit::current() {
            limit.check()?;
        }
        Ok(())
    }
}

struct Digest<'a> {
    hash: blake3::Hasher,
    check: &'a dyn Fn() -> Result<()>,
    error: Option<StoreError>,
}
impl Write for Digest<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Err(error) = (self.check)() {
            self.error = Some(error);
            return Err(std::io::Error::other("object preparation interrupted"));
        }
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Reader<'_> {
    pub fn prepare_object_read<'read>(
        &'read self,
        reference: &Reference,
    ) -> Result<ObjectRead<'read>> {
        let record = match reference {
            Reference::Request(id) if !id.is_empty() => match self.authored_record(reference)? {
                Some(super::authored_evidence::AuthoredReadEvidence::Local { record, .. }) => {
                    Record::Authored(record)
                }
                Some(super::authored_evidence::AuthoredReadEvidence::Incomplete { issue }) => {
                    return Err(StoreError::ReferenceUnavailable {
                        reference: reference.to_string(),
                        reason: issue.to_string(),
                    });
                }
                None => return Err(StoreError::NotFound(reference.to_string())),
            },
            _ => {
                return Err(StoreError::Invalid(
                    "exact object preparation requires a request reference".into(),
                ));
            }
        };
        record
            .validate()
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        let check = || self.check();
        let mut digest = Digest {
            hash: blake3::Hasher::new(),
            check: &check,
            error: None,
        };
        digest.hash.update(b"polylogue-exact-request-read-v1\0");
        let encoded = serde_json::to_writer(&mut digest, &record);
        if let Some(error) = digest.error {
            return Err(error);
        }
        encoded?;
        self.check()?;
        Ok(ObjectRead {
            record,
            digest: BlobHash(*digest.hash.finalize().as_bytes()),
            scope: self.scope_token(),
            _reader: PhantomData,
        })
    }
}
