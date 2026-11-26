//! Object preparation reads pinned stored rows.
use super::Reader;
use crate::{ReadScopeToken, Result, StoreError};
use polylogue_model::{BlobHash, Reference, bundle::Record};
use std::{io::Write, marker::PhantomData};

/// Accepted objects have their own stored identity; runs are not bundle inputs.
pub enum ObjectRecord {
    Request(Box<Record>),
    Run(Box<polylogue_model::query_objects::RecordedRun>),
}

/// Store-derived object content and its source, bound to the preparing Reader.
/// No wire decoder or public constructor can vouch for a caller's digest.
pub struct ObjectRead<'read> {
    record: ObjectRecord,
    digest: BlobHash,
    scope: ReadScopeToken,
    _reader: PhantomData<&'read ()>,
}
impl ObjectRead<'_> {
    pub fn record(&self) -> Result<&ObjectRecord> {
        self.check()?;
        Ok(&self.record)
    }
    pub fn content_hash(&self) -> BlobHash {
        self.digest
    }
    pub fn into_record(self) -> Result<ObjectRecord> {
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
                Some(super::authored_read::AuthoredRead::Local { record, .. }) => {
                    ObjectRecord::Request(Box::new(Record::Authored(record)))
                }
                Some(super::authored_read::AuthoredRead::Incomplete { issue }) => {
                    return Err(StoreError::ReferenceUnavailable {
                        reference: reference.to_string(),
                        reason: issue.to_string(),
                    });
                }
                None => return Err(StoreError::NotFound(reference.to_string())),
            },
            Reference::Run(id) => {
                let run = self.query_run(id)?;
                let outcome = self.query_run_outcome(id)?;
                if outcome.as_ref().is_some_and(|outcome| &outcome.run != id) {
                    return Err(StoreError::stored_data(
                        "stored run completion names another run",
                    ));
                }
                let Some(run) = run else {
                    return Err(if outcome.is_some() {
                        StoreError::ReferenceUnavailable {
                            reference: reference.to_string(),
                            reason:
                                "the run completion is recorded but its beginning is unavailable"
                                    .into(),
                        }
                    } else {
                        StoreError::NotFound(reference.to_string())
                    });
                };
                if &run.id != id {
                    return Err(StoreError::stored_data(
                        "stored run beginning names another run",
                    ));
                }
                crate::query_objects::validate_run(&run)?;
                if let Some(outcome) = &outcome {
                    crate::query_objects::validate_run_outcome(outcome)?;
                    if outcome.binding_hash != run.binding_hash()?
                        || outcome.run_hash != run.record_hash()?
                    {
                        return Err(StoreError::stored_data(
                            "stored run completion belongs to another execution",
                        ));
                    }
                }
                ObjectRecord::Run(Box::new(polylogue_model::query_objects::RecordedRun {
                    run,
                    outcome,
                }))
            }
            _ => {
                return Err(StoreError::internal(
                    "object preparation requires a request or run reference",
                ));
            }
        };
        let check = || self.check();
        let mut digest = Digest {
            hash: blake3::Hasher::new(),
            check: &check,
            error: None,
        };
        let encoded = match &record {
            ObjectRecord::Request(record) => {
                record.validate().map_err(crate::StoreError::stored_data)?;
                digest.hash.update(b"polylogue-exact-request-read-v1\0");
                serde_json::to_writer(&mut digest, record)
            }
            ObjectRecord::Run(record) => {
                digest.hash.update(b"polylogue-exact-run-read-v1\0");
                serde_json::to_writer(&mut digest, record)
            }
        };
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
