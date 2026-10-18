//! Hashes the Codex version evidence of one Message over a pinned snapshot.
//! Each advance reads at most one cell and hashes at most its byte allowance.
//! Hasher state is not stored; a restart begins at unit zero.
use crate::state::codex::Attempt;
use polylogue_model::{
    BlobHash,
    version_evidence::{Error as FramingError, Framer},
};
use polylogue_sources::codex::keyed::{
    Cell, EvidenceCursor, EvidenceHead, EvidenceUnit, Key, ReadView,
};
use polylogue_store::{
    Error as StoreError,
    parser_state::{Budget, Snapshot, Token},
};
#[derive(Debug)]
pub enum Error {
    Store(StoreError),
    Source(String),
    Evidence(String),
    Revision { expected: u64, actual: u64 },
    Framing(FramingError),
    Finished,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => std::fmt::Display::fmt(e, f),
            Self::Source(e) | Self::Evidence(e) => f.write_str(e),
            Self::Revision { expected, actual } => write!(
                f,
                "Codex evidence revision changed: expected {expected}, got {actual}"
            ),
            Self::Framing(e) => std::fmt::Display::fmt(e, f),
            Self::Finished => f.write_str("version digest driver already finished"),
        }
    }
}
impl std::error::Error for Error {}
impl From<FramingError> for Error {
    fn from(e: FramingError) -> Self {
        Self::Framing(e)
    }
}
type Result<T> = std::result::Result<T, Error>;
/// The writer must compare this token and recheck the evidence head before
/// accepting the digest for its row.
#[derive(Debug)]
pub struct Digest {
    token: Token,
    message: usize,
    head: EvidenceHead,
    digest: BlobHash,
}
impl Digest {
    pub fn token(&self) -> &Token {
        &self.token
    }
    pub fn message(&self) -> usize {
        self.message
    }
    pub fn head(&self) -> &EvidenceHead {
        &self.head
    }
    pub fn digest(&self) -> BlobHash {
        self.digest
    }
}
pub enum Advance {
    Progress {
        read_cells: usize,
        hashed_bytes: u64,
        minimum_bytes: Option<u64>,
    },
    Complete(Digest),
}
enum Stage {
    Begin,
    Tag,
    Payload(usize),
}
struct Current {
    unit: EvidenceUnit,
    stage: Stage,
}
pub struct Driver<'a> {
    snapshot: &'a Snapshot,
    message: usize,
    head: EvidenceHead,
    cursor: EvidenceCursor,
    seen: u64,
    framer: Option<Framer>,
    credit: u64,
    current: Option<Current>,
    single: bool,
    failed: bool,
    finished: bool,
}
fn read(snapshot: &Snapshot, key: &Key, budget: Budget) -> Result<Option<Cell>> {
    let attempt = Attempt::new(snapshot, budget);
    let result = attempt.read(key);
    if let Some(e) = attempt.take_failure() {
        return Err(Error::Store(e));
    }
    result.map_err(Error::Source)
}
impl<'a> Driver<'a> {
    /// Reads one head cell. Later reads use a fresh Attempt per advance, so
    /// read limits apply per turn.
    pub fn start(
        snapshot: &'a Snapshot,
        message: usize,
        expected_revision: u64,
        budget: Budget,
    ) -> Result<Self> {
        if snapshot.head.parser_version != polylogue_sources::codex::PARSER_VERSION {
            return Err(Error::Evidence(
                "snapshot does not use the current Codex parser".into(),
            ));
        }
        let head = match read(snapshot, &Key::EvidenceHead(message), budget)? {
            Some(Cell::EvidenceHead(h)) => h,
            _ => return Err(Error::Evidence("missing Codex evidence head".into())),
        };
        head.validate().map_err(Error::Evidence)?;
        if head.revision != expected_revision {
            return Err(Error::Revision {
                expected: expected_revision,
                actual: head.revision,
            });
        }
        Ok(Self {
            snapshot,
            message,
            head,
            cursor: EvidenceCursor::default(),
            seen: 0,
            framer: None,
            credit: 0,
            current: None,
            single: false,
            failed: false,
            finished: false,
        })
    }
    pub fn hashed_bytes(&self) -> u64 {
        self.framer.as_ref().map_or(0, Framer::used_bytes)
    }
    /// Fixed framing steps may need a larger allowance; that minimum (from the
    /// Framer) is reported without advancing the hash or dropping the loaded
    /// unit. Large JSON payloads are borrowed from the current cell.
    pub fn advance(&mut self, read_budget: Budget, byte_allowance: u64) -> Result<Advance> {
        if self.failed || self.finished {
            return Err(Error::Finished);
        }
        let result = self.advance_inner(read_budget, byte_allowance);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn advance_inner(&mut self, read_budget: Budget, allowance: u64) -> Result<Advance> {
        let before = self.hashed_bytes();
        let mut left = allowance;
        let mut reads = 0;
        loop {
            if self.current.is_none() {
                if reads != 0 {
                    return Ok(progress(reads, self.hashed_bytes() - before, None));
                }
                let Some(key) = self.head.next_slot(self.message, &mut self.cursor) else {
                    if self.seen != self.head.units {
                        return Err(Error::Evidence(
                            "Codex selected evidence count differs from head".into(),
                        ));
                    }
                    let digest = self
                        .framer
                        .as_ref()
                        .ok_or_else(|| Error::Evidence("no selected Codex evidence".into()))?
                        .finish()?;
                    self.finished = true;
                    return Ok(Advance::Complete(Digest {
                        token: self.snapshot.head.token.clone(),
                        message: self.message,
                        head: self.head.clone(),
                        digest,
                    }));
                };
                reads = 1;
                match read(self.snapshot, &key, read_budget)? {
                    Some(Cell::Evidence(unit)) => {
                        unit.validate_at(&key).map_err(Error::Evidence)?;
                        if self.seen >= self.head.units {
                            return Err(Error::Evidence(
                                "extra Codex evidence unit beyond declared count".into(),
                            ));
                        }
                        self.current = Some(Current {
                            unit,
                            stage: Stage::Begin,
                        });
                    }
                    None if !self.head.required_slot(&key) => {
                        return Ok(progress(reads, self.hashed_bytes() - before, None));
                    }
                    _ => {
                        return Err(Error::Evidence(
                            "missing required Codex evidence unit".into(),
                        ));
                    }
                }
            }
            let current = self.current.as_mut().expect("loaded current evidence");
            if self.framer.is_none() {
                self.single = self.head.units == 1 && current.unit.whole_record;
                if self.single {
                    self.framer = Some(Framer::single_whole_record(
                        current.unit.canonical.len() as u64,
                        0,
                    ));
                    current.stage = Stage::Payload(0);
                } else {
                    match Framer::compound(self.head.units, left) {
                        Ok(f) => {
                            let used = f.used_bytes();
                            left -= used;
                            self.credit = allowance.saturating_sub(used);
                            self.framer = Some(f);
                        }
                        Err(FramingError::Admission) => {
                            return Ok(progress(
                                reads,
                                self.hashed_bytes() - before,
                                Some(Framer::compound_minimum_bytes()),
                            ));
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            }
            let framer = self.framer.as_mut().expect("initialized evidence framer");
            let cost = match current.stage {
                Stage::Begin => 9,
                Stage::Tag => (current.unit.tag.len() as u64)
                    .checked_add(8)
                    .ok_or(FramingError::Overflow)?,
                Stage::Payload(offset) => (current.unit.canonical.len() - offset) as u64,
            };
            let work = if matches!(current.stage, Stage::Payload(_)) {
                cost.min(left)
            } else {
                cost
            };
            if work > left || (work == 0 && cost > 0) {
                return Ok(progress(
                    reads,
                    framer.used_bytes() - before,
                    Some(if matches!(current.stage, Stage::Payload(_)) {
                        1
                    } else {
                        cost
                    }),
                ));
            }
            if self.credit < work {
                framer.grant(work - self.credit)?;
                self.credit = work;
            }
            match current.stage {
                Stage::Begin => {
                    framer.begin_json(
                        current.unit.tag.len() as u64,
                        current.unit.canonical.len() as u64,
                    )?;
                    current.stage = Stage::Tag;
                }
                Stage::Tag => {
                    framer.tag(&current.unit.tag)?;
                    current.stage = Stage::Payload(0);
                }
                Stage::Payload(offset) => {
                    // Valid JSON has at least one byte.
                    let n = usize::try_from(work).map_err(|_| FramingError::Overflow)?;
                    framer.json(&current.unit.canonical[offset..offset + n])?;
                    if offset + n == current.unit.canonical.len() {
                        self.current = None;
                        self.seen += 1;
                    } else {
                        current.stage = Stage::Payload(offset + n);
                    }
                }
            }
            self.credit -= work;
            left -= work;
            // Finishing a unit ends the turn even if no cell was read, so the
            // next slot's read is charged to the next turn.
            if self.current.is_none() {
                return Ok(progress(reads, self.hashed_bytes() - before, None));
            }
        }
    }
}
fn progress(read_cells: usize, hashed_bytes: u64, minimum_bytes: Option<u64>) -> Advance {
    Advance::Progress {
        read_cells,
        hashed_bytes,
        minimum_bytes,
    }
}
