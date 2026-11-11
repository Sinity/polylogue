//! The register rows one stored step of an input writes: its revision, the
//! extents its new bytes fill and the observation that found them. The writer
//! builds one [`InputRows`] per step and inserts it; in primary mode Sinex
//! keeps it with the bytes it describes (or alone, when the step found no new
//! bytes), and a rebuild from Sinex inserts the same rows again.
use crate::Result;
use polylogue_model::BlobHash;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

/// A field this shape does not name is refused, so rows written in an
/// earlier shape never read as rows missing that field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InputRows {
    /// Receiver-owned evidence for the acquired body, independent of overlap trimming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acquisition: Option<crate::primary::AcquisitionReceipt>,
    pub input: String,
    /// The origin that parses it (`claude-code`).
    pub format: String,
    /// A new revision, or the current one grown by a later chunk of the
    /// same read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<Revision>,
    /// Where the step's new bytes sit in the revision chain, in order and
    /// contiguous: one extent, or one per content-defined chunk of bytes
    /// stored as chunks.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extents: Vec<Extent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation: Option<ObservationRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Revision {
    pub number: i64,
    pub produced: String,
    pub chain: i64,
    pub length: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Extent {
    pub chain: i64,
    pub offset: i64,
    pub length: i64,
    pub blob: BlobHash,
    /// Offset into the original captured material after an exact prefix overlap.
    pub blob_offset: i64,
}

/// Drops the first `length` bytes of contiguous `extents`: the extents they
/// cover go, and the one they end in starts later in its blob.
pub(crate) fn drop_front(extents: &mut Vec<Extent>, mut length: u64) {
    extents.retain_mut(|extent| {
        let size = extent.length as u64;
        if length >= size && size > 0 {
            length -= size;
            return false;
        }
        extent.offset += length as i64;
        extent.blob_offset += length as i64;
        extent.length -= length as i64;
        length = 0;
        true
    });
}

/// One look at a path that found something.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ObservationRow {
    pub id: String,
    pub machine: String,
    pub host: String,
    pub path: String,
    /// Historical observations may not have retained the original path bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_bytes: Option<Vec<u8>>,
    pub class: String,
    pub revision: Option<i64>,
    /// A byte-proven chain prefix that is not a recorded revision endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain: Option<i64>,
    pub length: Option<i64>,
    pub device: Option<i64>,
    pub inode: Option<i64>,
    pub size: Option<i64>,
    pub mtime_ns: Option<i64>,
    pub ctime_ns: Option<i64>,
    pub polylogue_version: String,
    pub seen_at: i64,
    pub details_json: Option<String>,
    /// The exact container membership established by this observation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<crate::InContainer>,
}

impl InputRows {
    /// Inserts the rows at write counter `seq`. A revision or extent already
    /// present takes the recorded one: a later chunk grew the revision, or a
    /// write that stopped before its commit sent the bytes at that place
    /// again, longer. Everything else present is left as it is, so inserting
    /// the rows twice changes nothing.
    pub fn insert(&self, tx: &Connection, seq: i64) -> Result<()> {
        self.insert_rows(tx, seq, true)
    }

    /// Confirmed historical observations retain their original admission proof
    /// when a source dependency is temporarily absent from the live projection.
    pub(crate) fn insert_confirmed(&self, tx: &Connection, seq: i64) -> Result<()> {
        self.insert_rows(tx, seq, false)
    }

    fn insert_rows(&self, tx: &Connection, seq: i64, prove_prefix: bool) -> Result<()> {
        self.validate_acquisition()?;
        let source = crate::source::Source::of(tx)?;
        tx.prepare_cached(
            &source.sql("INSERT OR IGNORE INTO @source.input(id, format) VALUES (?1, ?2)"),
        )?
        .execute(params![self.input, self.format])?;
        if let Some(r) = &self.revision {
            tx.prepare_cached(&source.sql(
                "INSERT INTO @source.revision(input, number, produced, chain, length, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(input, number) DO UPDATE SET length = excluded.length",
            ))?
            .execute(params![
                self.input, r.number, r.produced, r.chain, r.length, seq
            ])?;
        }
        for e in &self.extents {
            tx.prepare_cached(
                &source.sql("INSERT INTO @source.extent(input, chain, offset, length, blob, blob_offset, commit_seq, acquisition)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(input,chain,offset) DO UPDATE SET
                     commit_seq=CASE WHEN length=excluded.length AND blob=excluded.blob AND blob_offset=excluded.blob_offset THEN min(commit_seq,excluded.commit_seq) ELSE excluded.commit_seq END,
                     acquisition=CASE WHEN length=excluded.length AND blob=excluded.blob AND blob_offset=excluded.blob_offset THEN coalesce(acquisition,excluded.acquisition) ELSE excluded.acquisition END,
                     length=excluded.length,blob=excluded.blob,blob_offset=excluded.blob_offset"),
            )?
            .execute(params![
                self.input,
                e.chain,
                e.offset,
                e.length,
                &e.blob.0[..],e.blob_offset,seq,
                self.acquisition.as_ref().map(crate::encoding::pack_json).transpose()?
            ])?;
        }
        if let Some(o) = &self.observation {
            let details = self.observation_details(o)?;
            if let Some(chain) = o.chain {
                let length = o
                    .length
                    .and_then(|length| u64::try_from(length).ok())
                    .ok_or_else(|| {
                        crate::StoreError::Invalid(
                            "a prefix observation needs its observed byte length".into(),
                        )
                    })?;
                if o.revision.is_some()
                    || chain < 1
                    || prove_prefix
                        && crate::source::Source::admitted_through(tx, &self.input, chain, length)?
                            .is_none()
                {
                    return Err(crate::StoreError::Invalid(
                        "a prefix observation needs a committed chain prefix and no revision claim"
                            .into(),
                    ));
                }
            }
            tx.prepare_cached(&source.sql(
                "INSERT OR IGNORE INTO @source.machine(id, host, first_seen) VALUES (?1, ?2, ?3)",
            ))?
            .execute(params![o.machine, o.host, o.seen_at])?;
            tx.prepare_cached(
                &source.sql("INSERT OR IGNORE INTO @source.observation(id, input, machine, path, class, revision, length,
                     device, inode, size, mtime_ns, ctime_ns, polylogue_version, seen_at, commit_seq,
                     details_json, chain, path_bytes)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)"),
            )?
            .execute(params![
                o.id,
                self.input,
                o.machine,
                o.path,
                o.class,
                o.revision,
                o.length,
                o.device,
                o.inode,
                o.size,
                o.mtime_ns,
                o.ctime_ns,
                o.polylogue_version,
                o.seen_at,
                seq,
                details,
                o.chain,
                o.path_bytes,
            ])?;
            if let Some(c) = &o.container {
                tx.prepare_cached(
                    &source.sql("INSERT OR IGNORE INTO @source.observation_container VALUES (?1, ?2, ?3, ?4, ?5)"),
                )?
                .execute(params![
                    o.id,
                    c.input,
                    c.member_index as i64,
                    c.kind,
                    c.path_bytes
                ])?;
            }
        }
        Ok(())
    }
    fn validate_acquisition(&self) -> Result<()> {
        let Some(receipt) = &self.acquisition else {
            return Ok(());
        };
        receipt.validate_for(&crate::InputKey {
            format: self.format.clone(),
            id: self.input.clone(),
            route: polylogue_model::Route::Live,
        })?;
        let length = receipt.range.length()?;
        let invalid = || {
            crate::StoreError::Invalid(
                "acquisition evidence does not describe these source rows".into(),
            )
        };
        if receipt.version != crate::primary::AcquisitionReceipt::VERSION
            || receipt.source.origin != self.format
            || receipt.source.machine.is_empty()
        {
            return Err(invalid());
        }
        if let Some(observation) = &self.observation
            && (observation.machine != receipt.source.machine
                || observation.path_bytes.as_ref() != Some(&receipt.source.path_bytes))
        {
            return Err(invalid());
        }
        if self.extents.is_empty() && self.observation.is_none() {
            return Err(invalid());
        }
        // Each extent lies inside one blob of the acquired body, at its
        // place: the body itself, or the chunk the body was stored as there.
        let whole = [crate::Chunk {
            hash: receipt.hash,
            size: length,
        }];
        let blobs = if receipt.chunks.is_empty() {
            &whole[..]
        } else {
            &receipt.chunks[..]
        };
        if blobs.iter().map(|chunk| chunk.size).sum::<u64>() != length {
            return Err(invalid());
        }
        let mut next = None;
        for extent in &self.extents {
            let start = u64::try_from(extent.offset).map_err(|_| invalid())?;
            let size = u64::try_from(extent.length).map_err(|_| invalid())?;
            let offset = u64::try_from(extent.blob_offset).map_err(|_| invalid())?;
            let at = start.checked_sub(receipt.range.start).ok_or_else(invalid)?;
            let end = at.checked_add(size).ok_or_else(invalid)?;
            let mut blob_start = 0u64;
            let placed = blobs.iter().any(|blob| {
                let here = blob_start;
                blob_start += blob.size;
                blob.hash == extent.blob && here <= at && end <= blob_start && offset == at - here
            });
            if !placed || next.is_some_and(|next| next != start) {
                return Err(invalid());
            }
            next = Some(start + size);
        }
        Ok(())
    }

    fn observation_details(&self, observation: &ObservationRow) -> Result<Option<String>> {
        let Some(receipt) = self
            .acquisition
            .as_ref()
            .filter(|_| self.extents.is_empty())
        else {
            return Ok(observation.details_json.clone());
        };
        let mut details = observation
            .details_json
            .as_deref()
            .map(serde_json::from_str::<serde_json::Value>)
            .transpose()?
            .unwrap_or_else(|| serde_json::json!({}));
        let object = details.as_object_mut().ok_or_else(|| {
            crate::StoreError::Invalid("observation details must be an object".into())
        })?;
        let receipt = serde_json::to_value(receipt)?;
        if object
            .get("acquisition")
            .is_some_and(|prior| prior != &receipt)
        {
            return Err(crate::StoreError::Invalid(
                "observation acquisition evidence conflicts".into(),
            ));
        }
        object.insert("acquisition".into(), receipt);
        Ok(Some(serde_json::to_string(&details)?))
    }
}
