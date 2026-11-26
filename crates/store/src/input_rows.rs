//! The archive rows one stored step of an input writes: its revision, the
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
    pub capture_read: Option<crate::primary::CaptureRead>,
    pub input: String,
    /// The origin that parses it (`claude-code`).
    pub origin: String,
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
    /// Inserts the rows of one step at write counter `seq`. A revision already
    /// present takes the incoming length: a later chunk of the same read grew
    /// it. A stored extent never changes: the writer places a step's bytes at
    /// its chain's stored end, so an extent at a stored place is either the
    /// one stored there, which keeps its first commit, or is refused.
    /// Inserting the rows twice changes nothing.
    pub fn insert(&self, tx: &Connection, seq: i64) -> Result<()> {
        self.insert_rows(tx, seq, false)
    }

    /// Inserts rows confirmed by Sinex, in their commit order. A prefix
    /// observation's chain prefix may be missing from the index for a while,
    /// so it is not checked. An extent already present takes the incoming
    /// length, blob and blob offset: a write that stopped before its local
    /// commit sent the bytes at that place to Sinex again, longer. Everything
    /// else present is left as it is, so inserting the rows twice changes
    /// nothing.
    pub(crate) fn insert_confirmed(&self, tx: &Connection, seq: i64) -> Result<()> {
        self.insert_rows(tx, seq, true)
    }

    fn insert_rows(&self, tx: &Connection, seq: i64, confirmed: bool) -> Result<()> {
        self.validate_source_read()?;
        let source = crate::source::Source::of(tx)?;
        tx.prepare_cached(
            &source.sql("INSERT OR IGNORE INTO @source.input(id, format) VALUES (?1, ?2)"),
        )?
        .execute(params![self.input, self.origin])?;
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
        let replace = if confirmed {
            "ON CONFLICT(input,chain,offset) DO UPDATE SET
                 commit_seq=CASE WHEN length=excluded.length AND blob=excluded.blob AND blob_offset=excluded.blob_offset THEN min(commit_seq,excluded.commit_seq) ELSE excluded.commit_seq END,
                 capture_read=CASE WHEN length=excluded.length AND blob=excluded.blob AND blob_offset=excluded.blob_offset THEN coalesce(capture_read,excluded.capture_read) ELSE excluded.capture_read END,
                 length=excluded.length,blob=excluded.blob,blob_offset=excluded.blob_offset"
        } else {
            // Only the extent already stored there matches; any other
            // changes no row, which is refused below.
            "ON CONFLICT(input,chain,offset) DO UPDATE SET
                 commit_seq=min(commit_seq,excluded.commit_seq),
                 capture_read=coalesce(capture_read,excluded.capture_read)
             WHERE length=excluded.length AND blob=excluded.blob AND blob_offset=excluded.blob_offset"
        };
        for e in &self.extents {
            let written = tx.prepare_cached(
                &source.sql(&format!("INSERT INTO @source.extent(input, chain, offset, length, blob, blob_offset, commit_seq, capture_read)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) {replace}")),
            )?
            .execute(params![
                self.input,
                e.chain,
                e.offset,
                e.length,
                &e.blob.0[..],e.blob_offset,seq,
                self.capture_read.as_ref().map(crate::encoding::pack_json).transpose()?
            ])?;
            if written == 0 {
                return Err(crate::StoreError::Invalid(
                    "a stored extent cannot be replaced".into(),
                ));
            }
        }
        if let Some(o) = &self.observation {
            let details = self.observation_details(o)?;
            if let Some(chain) = o.chain {
                let length = o
                    .length
                    .and_then(|length| u64::try_from(length).ok())
                    .ok_or_else(|| {
                        crate::StoreError::internal(
                            "a prefix observation needs its observed byte length",
                        )
                    })?;
                if o.revision.is_some()
                    || chain < 1
                    || !confirmed
                        && crate::source::Source::committed_through(tx, &self.input, chain, length)?
                            .is_none()
                {
                    return Err(crate::StoreError::internal(
                        "a prefix observation needs a committed chain prefix and no revision number",
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
    fn validate_source_read(&self) -> Result<()> {
        let Some(source_read) = &self.capture_read else {
            return Ok(());
        };
        source_read.validate_for(&crate::InputKey {
            format: self.origin.clone(),
            id: self.input.clone(),
            route: polylogue_model::Route::Live,
        })?;
        let length = source_read.range.length()?;
        let inconsistent =
            || crate::StoreError::internal("the capture read does not describe these source rows");
        if source_read.version != crate::primary::CaptureRead::VERSION
            || source_read.source.origin != self.origin
            || source_read.source.machine.is_empty()
        {
            return Err(inconsistent());
        }
        if let Some(observation) = &self.observation
            && (observation.machine != source_read.source.machine
                || observation.path_bytes.as_ref() != Some(&source_read.source.path_bytes))
        {
            return Err(inconsistent());
        }
        if self.extents.is_empty() && self.observation.is_none() {
            return Err(inconsistent());
        }
        // Each extent lies inside one blob of the acquired body, at its
        // place: the body itself, or the chunk the body was stored as there.
        let whole = [crate::Chunk {
            hash: source_read.hash,
            size: length,
        }];
        let blobs = if source_read.chunks.is_empty() {
            &whole[..]
        } else {
            &source_read.chunks[..]
        };
        if blobs.iter().map(|chunk| chunk.size).sum::<u64>() != length {
            return Err(inconsistent());
        }
        let mut next = None;
        for extent in &self.extents {
            let start = u64::try_from(extent.offset).map_err(|_| inconsistent())?;
            let size = u64::try_from(extent.length).map_err(|_| inconsistent())?;
            let offset = u64::try_from(extent.blob_offset).map_err(|_| inconsistent())?;
            let at = start
                .checked_sub(source_read.range.start)
                .ok_or_else(inconsistent)?;
            let end = at.checked_add(size).ok_or_else(inconsistent)?;
            let mut blob_start = 0u64;
            let placed = blobs.iter().any(|blob| {
                let here = blob_start;
                blob_start += blob.size;
                blob.hash == extent.blob && here <= at && end <= blob_start && offset == at - here
            });
            if !placed || next.is_some_and(|next| next != start) {
                return Err(inconsistent());
            }
            next = Some(start + size);
        }
        Ok(())
    }

    fn observation_details(&self, observation: &ObservationRow) -> Result<Option<String>> {
        let Some(source_read) = self
            .capture_read
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
        let object = details
            .as_object_mut()
            .ok_or_else(|| crate::StoreError::internal("observation details must be an object"))?;
        let source_read = serde_json::to_value(source_read)?;
        if object
            .get("capture_read")
            .is_some_and(|prior| prior != &source_read)
        {
            return Err(crate::StoreError::internal(
                "the observation's capture read conflicts",
            ));
        }
        object.insert("capture_read".into(), source_read);
        Ok(Some(serde_json::to_string(&details)?))
    }
}
