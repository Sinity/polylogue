//! The archive rows one stored step of an input writes: its revision, the
//! extents its new bytes fill and the observation that found them. The writer
//! builds one [`InputRows`] per step and inserts it.
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
    pub capture_read: Option<crate::capture_read::CaptureRead>,
    pub input: String,
    /// The origin that parses it (`claude-code`).
    pub origin: String,
    /// A new revision, or the current one grown by a later chunk of the
    /// same read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<Revision>,
    /// With the first revision of a chain: the earlier chain of the input
    /// its bytes begin like, and through which byte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain: Option<crate::archive::writer::agreement::Agreement>,
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
    pub produced: Produced,
    pub chain: i64,
    pub length: i64,
    /// When its bytes were read, by Polylogue's clock, and the file's
    /// modification and change times then: what the next look at an
    /// unchanged location compares with, since an append there writes no
    /// observation.
    pub seen_at: Option<i64>,
    pub mtime_ns: Option<i64>,
    pub ctime_ns: Option<i64>,
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

/// One look at a location that found something.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ObservationRow {
    pub id: String,
    pub machine: String,
    pub host: String,
    /// The location's bytes as read; shown as text where it is shown.
    pub path_bytes: Vec<u8>,
    pub seen: Seen,
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

polylogue_model::stored_names! {
    /// What made a revision: a location's first bytes, an imported file,
    /// bytes appended to its chain, or a rewrite that starts a chain.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Produced {
        First => "first",
        Imported => "imported",
        Appended => "appended",
        Rewritten => "rewritten",
    }
}

impl Produced {
    /// Whether a revision of this kind starts its own chain.
    pub fn starts_chain(self) -> bool {
        !matches!(self, Self::Appended)
    }
}

polylogue_model::stored_names! {
    /// What an observation found at a location.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum ObservationClass {
        First => "first",
        Appended => "appended",
        Rewritten => "rewritten",
        Deleted => "deleted",
        Back => "back",
        Imported => "imported",
    }
}

/// What one observation saw, with the revision, chain and length it names:
/// the pairs a stored row can hold, and no others. `length` is the length of
/// the revision or prefix it names, which for a file stored in several
/// steps of one read is the revision's length after the step that made the
/// observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "class", rename_all = "snake_case", deny_unknown_fields)]
pub enum Seen {
    /// A revision this observation stored: a location's first, an import, a
    /// rewrite, or an append found at another location than the last.
    Stored {
        produced: Produced,
        revision: i64,
        length: i64,
    },
    /// Seen again holding exactly a stored revision.
    Back { revision: i64, length: i64 },
    /// Seen holding a prefix of a stored chain: no byte the chain lacks,
    /// `length` of its bytes.
    Prefix { chain: i64, length: i64 },
    /// Gone from its location.
    Deleted,
}

impl Seen {
    /// The stored columns: class, revision, chain, length. The one encoder.
    pub fn columns(self) -> (ObservationClass, Option<i64>, Option<i64>, Option<i64>) {
        match self {
            Self::Stored {
                produced,
                revision,
                length,
            } => (
                match produced {
                    Produced::First => ObservationClass::First,
                    Produced::Imported => ObservationClass::Imported,
                    Produced::Appended => ObservationClass::Appended,
                    Produced::Rewritten => ObservationClass::Rewritten,
                },
                Some(revision),
                None,
                Some(length),
            ),
            Self::Back { revision, length } => {
                (ObservationClass::Back, Some(revision), None, Some(length))
            }
            Self::Prefix { chain, length } => {
                (ObservationClass::Back, None, Some(chain), Some(length))
            }
            Self::Deleted => (ObservationClass::Deleted, None, None, None),
        }
    }

    /// Reads stored columns back; a pair the writer cannot make is refused.
    /// The one decoder.
    pub fn from_columns(
        class: &str,
        revision: Option<i64>,
        chain: Option<i64>,
        length: Option<i64>,
    ) -> crate::Result<Self> {
        let refused = || {
            crate::StoreError::stored_data(format!(
                "observation {class} with revision {revision:?}, chain {chain:?}, length {length:?}"
            ))
        };
        let class = ObservationClass::from_name(class).ok_or_else(refused)?;
        Ok(match (class, revision, chain, length) {
            (ObservationClass::Deleted, None, None, None) => Self::Deleted,
            (ObservationClass::Back, Some(revision), None, Some(length)) => {
                Self::Back { revision, length }
            }
            (ObservationClass::Back, None, Some(chain), Some(length)) => {
                Self::Prefix { chain, length }
            }
            (class, Some(revision), None, Some(length)) => Self::Stored {
                produced: match class {
                    ObservationClass::First => Produced::First,
                    ObservationClass::Imported => Produced::Imported,
                    ObservationClass::Appended => Produced::Appended,
                    ObservationClass::Rewritten => Produced::Rewritten,
                    ObservationClass::Back | ObservationClass::Deleted => return Err(refused()),
                },
                revision,
                length,
            },
            _ => return Err(refused()),
        })
    }
}

impl Seen {
    /// The revision it names, if it names one.
    pub fn revision(self) -> Option<i64> {
        self.columns().1
    }

    /// The length of what it names.
    pub fn length(self) -> Option<i64> {
        self.columns().3
    }
}

impl InputRows {
    /// Inserts the rows of one step at write counter `seq`. A revision already
    /// present takes the incoming length: a later chunk of the same read grew
    /// it. A stored extent never changes: the writer places a step's bytes at
    /// its chain's stored end, so an extent at a stored place is either the
    /// one stored there, which keeps its first commit, or is refused.
    /// Inserting the rows twice changes nothing.
    pub fn insert(&self, tx: &Connection, seq: i64) -> Result<()> {
        self.validate_source_read()?;
        let source = crate::source::Source::of(tx)?;
        tx.prepare_cached(
            &source.sql("INSERT OR IGNORE INTO @source.input(id, origin) VALUES (?1, ?2)"),
        )?
        .execute(params![self.input, self.origin])?;
        if let Some(r) = &self.revision {
            tx.prepare_cached(&source.sql(
                "INSERT INTO @source.revision(input, number, produced, chain, length, seen_at,
                     mtime_ns, ctime_ns, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(input, number) DO UPDATE SET length = excluded.length,
                     mtime_ns = coalesce(excluded.mtime_ns, mtime_ns),
                     ctime_ns = coalesce(excluded.ctime_ns, ctime_ns)",
            ))?
            .execute(params![
                self.input,
                r.number,
                r.produced.as_str(),
                r.chain,
                r.length,
                r.seen_at,
                r.mtime_ns,
                r.ctime_ns,
                seq
            ])?;
        }
        if let (Some(r), Some(agreement)) = (&self.revision, &self.chain) {
            if r.chain != r.number || !r.produced.starts_chain() {
                return Err(crate::StoreError::internal(
                    "a chain's agreement is written with its first revision",
                ));
            }
            tx.prepare_cached(&source.sql(
                "INSERT INTO @source.chain(input, number, begins_like, agrees_to, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(input, number) DO NOTHING",
            ))?
            .execute(params![
                self.input,
                r.number,
                agreement.begins_like,
                crate::to_i64(
                    agreement.agrees_to,
                    "agreement exceeds archive address space"
                )?,
                seq
            ])?;
        }
        // Only the extent already stored there matches; any other
        // changes no row, which is refused below.
        let replace = "ON CONFLICT(input,chain,offset) DO UPDATE SET
             commit_seq=min(commit_seq,excluded.commit_seq)
         WHERE length=excluded.length AND blob=excluded.blob AND blob_offset=excluded.blob_offset";
        for e in &self.extents {
            let written = tx.prepare_cached(
                &source.sql(&format!("INSERT INTO @source.extent(input, chain, offset, length, blob, blob_offset, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) {replace}")),
            )?
            .execute(params![
                self.input,
                e.chain,
                e.offset,
                e.length,
                &e.blob.0[..],
                e.blob_offset,
                seq
            ])?;
            if written == 0 {
                return Err(crate::StoreError::Invalid(
                    "a stored extent cannot be replaced".into(),
                ));
            }
        }
        if let Some(o) = &self.observation {
            let details = self.observation_details(o)?;
            if let Seen::Prefix { chain, length } = o.seen {
                let length = u64::try_from(length).map_err(|_| {
                    crate::StoreError::internal(
                        "a prefix observation needs its observed byte length",
                    )
                })?;
                if chain < 1
                    || crate::source::Source::committed_through(tx, &self.input, chain, length)?
                        .is_none()
                {
                    return Err(crate::StoreError::internal(
                        "a prefix observation needs a committed chain prefix",
                    ));
                }
            }
            let (class, revision, chain, length) = o.seen.columns();
            tx.prepare_cached(&source.sql(
                "INSERT OR IGNORE INTO @source.machine(id, host, first_seen_at) VALUES (?1, ?2, ?3)",
            ))?
            .execute(params![o.machine, o.host, o.seen_at])?;
            tx.prepare_cached(
                &source.sql("INSERT OR IGNORE INTO @source.observation(id, input, machine, class, revision, length,
                     device, inode, size, mtime_ns, ctime_ns, polylogue_version, seen_at, commit_seq,
                     details_json, chain, path_bytes)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)"),
            )?
            .execute(params![
                o.id,
                self.input,
                o.machine,
                class.as_str(),
                revision,
                length,
                o.device,
                o.inode,
                o.size,
                o.mtime_ns,
                o.ctime_ns,
                o.polylogue_version,
                o.seen_at,
                seq,
                details,
                chain,
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
        if source_read.version != crate::capture_read::CaptureRead::VERSION
            || source_read.source.origin != self.origin
            || source_read.source.machine.is_empty()
        {
            return Err(inconsistent());
        }
        if let Some(observation) = &self.observation
            && (observation.machine != source_read.source.machine
                || observation.path_bytes != source_read.source.path_bytes)
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
