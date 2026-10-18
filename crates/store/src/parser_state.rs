//! Per-input parser state in index.db: a `parse_state` head row plus
//! key/value cells in `parser_cell`, scoped to a state generation. Workers read
//! a pinned read-only snapshot; each step writes its cell changes and progress
//! in one transaction and fails with `StaleParserState` if the head moved.
use crate::{Error as StoreError, Result};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::Path;

// Runs a SQLite call so its errors carry the database path.
macro_rules! sql {
    ($path:expr, $operation:expr) => {
        crate::db($path, || $operation)
    };
}

pub mod prepared;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Token {
    pub index_generation: String,
    pub input_pk: i64,
    pub chain: i64,
    pub generation: i64,
    pub step: i64,
    pub basis: [u8; 32],
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Accounting {
    pub framed: u64,
    pub used: u64,
    pub ignored: u64,
    pub unrecognized: u64,
    pub undecodable: u64,
}
impl Accounting {
    fn checked(self) -> Result<Self> {
        let total = self
            .used
            .checked_add(self.ignored)
            .and_then(|n| n.checked_add(self.unrecognized))
            .and_then(|n| n.checked_add(self.undecodable));
        if total != Some(self.framed) {
            return Err(StoreError::Invariant(
                "parser record counts do not add up".into(),
            ));
        }
        for n in [
            self.framed,
            self.used,
            self.ignored,
            self.unrecognized,
            self.undecodable,
        ] {
            integer(n)?;
        }
        Ok(self)
    }
    fn add(self, other: Self) -> Result<Self> {
        fn sum(a: u64, b: u64) -> Result<u64> {
            a.checked_add(b)
                .ok_or_else(|| StoreError::Invariant("parser record count overflow".into()))
        }
        Self {
            framed: sum(self.framed, other.framed)?,
            used: sum(self.used, other.used)?,
            ignored: sum(self.ignored, other.ignored)?,
            unrecognized: sum(self.unrecognized, other.unrecognized)?,
            undecodable: sum(self.undecodable, other.undecodable)?,
        }
        .checked()
    }
}
#[derive(Clone, Debug)]
pub struct Head {
    pub token: Token,
    pub parser_version: u32,
    pub acquired_upto: u64,
    pub inspected_upto: u64,
    pub parsed_upto: u64,
    pub status: Phase,
    pub finalization_pending: bool,
    pub counts: Accounting,
}
/// What a parse is computed from: the input's acquired bytes (by digest and
/// length), the parser version, companion files and excisions seen.
pub struct Basis<'a> {
    pub input_pk: i64,
    pub chain: i64,
    pub parser_version: u32,
    pub digest: [u8; 32],
    pub acquired_upto: u64,
    pub companions_json: Option<&'a str>,
    pub excisions_seen: u64,
}
/// The `parse_state.status` values. A step may set any phase except `Retired`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Parsing,
    WaitingTail,
    Finalizing,
    Done,
    Retired,
}
impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Parsing => "parsing",
            Self::WaitingTail => "waiting_tail",
            Self::Finalizing => "finalizing",
            Self::Done => "done",
            Self::Retired => "retired",
        }
    }
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "parsing" => Self::Parsing,
            "waiting_tail" => Self::WaitingTail,
            "finalizing" => Self::Finalizing,
            "done" => Self::Done,
            "retired" => Self::Retired,
            _ => {
                return Err(StoreError::Invariant("unknown parser state status".into()));
            }
        })
    }
}
pub struct Progress {
    pub inspected_upto: u64,
    pub parsed_upto: u64,
    /// Set by the finalizer. Reaching the end of the bytes alone does not make
    /// a state `Done`.
    pub phase: Phase,
    pub finalization_pending: bool,
    pub counts: Accounting,
}
pub struct CellChange {
    pub namespace: String,
    pub key: Vec<u8>,
    pub value: Option<Vec<u8>>,
}
#[derive(Clone, Copy)]
pub struct Budget {
    pub cells: usize,
    pub encoded_bytes: usize,
}

pub(super) fn unsigned(row: &rusqlite::Row<'_>, index: usize) -> rusqlite::Result<u64> {
    let value: i64 = row.get(index)?;
    u64::try_from(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            index,
            rusqlite::types::Type::Integer,
            Box::new(error),
        )
    })
}
fn integer(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| StoreError::Invariant("parser position exceeds SQLite integer range".into()))
}
fn index_generation(db: &Connection, path: &Path) -> Result<String> {
    Ok(sql!(
        path,
        db.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get(0)
        })
    )?)
}
// The counter lives in meta and survives head and input_ref deletion, so a
// reused key never recreates an old token.
fn allocate_generation(tx: &Transaction<'_>, path: &Path) -> Result<i64> {
    let stored: String = sql!(
        path,
        tx.query_row(
            "SELECT value FROM meta WHERE key='parser_generation'",
            [],
            |r| r.get(0),
        )
    )?;
    let generation = stored
        .parse::<i64>()
        .ok()
        .filter(|n| *n >= 0)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| {
            StoreError::Invariant("parser generation counter is invalid or exhausted".into())
        })?;
    sql!(
        path,
        tx.execute(
            "UPDATE meta SET value=?1 WHERE key='parser_generation'",
            [generation.to_string()],
        )
    )?;
    Ok(generation)
}
fn carry(generation: i64, step: i64) -> Vec<u8> {
    [generation.to_be_bytes(), step.to_be_bytes()].concat()
}
pub(crate) fn head(
    db: &Connection,
    path: &Path,
    input_pk: i64,
    chain: i64,
) -> Result<Option<Head>> {
    let index_generation = index_generation(db, path)?;
    let row = sql!(path, db
        .query_row(
            "SELECT state_generation, state_step, basis_digest, parser_version, acquired_upto,
                inspected_upto, parsed_upto, status, finalization_pending,framed,used,ignored,unrecognized_n,undecodable,carry
         FROM parse_state WHERE input_pk=?1 AND chain=?2 AND state_generation>0",
            params![input_pk, chain],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, Vec<u8>>(2)?,
                    r.get::<_, u32>(3)?,
                    unsigned(r, 4)?,
                    unsigned(r, 5)?,
                    unsigned(r, 6)?,
                    r.get::<_, String>(7)?,
                    r.get::<_, bool>(8)?,
                    Accounting {framed:unsigned(r,9)?,used:unsigned(r,10)?,ignored:unsigned(r,11)?,unrecognized:unsigned(r,12)?,undecodable:unsigned(r,13)?},
                    r.get::<_,Vec<u8>>(14)?,
                ))
            },
        )
        .optional())?;
    row.map(
        |(
            generation,
            step,
            digest,
            parser_version,
            acquired_upto,
            inspected_upto,
            parsed_upto,
            status,
            finalization_pending,
            counts,
            carry,
        )| {
            if carry != self::carry(generation, step) {
                return Err(StoreError::Invariant(
                    "parser state carry does not match its generation and step".into(),
                ));
            }
            let basis = digest
                .try_into()
                .map_err(|_| StoreError::Invariant("parser input digest is not 32 bytes".into()))?;
            Ok(Head {
                token: Token {
                    index_generation,
                    input_pk,
                    chain,
                    generation,
                    step,
                    basis,
                },
                parser_version,
                acquired_upto,
                inspected_upto,
                parsed_upto,
                status: Phase::parse(&status)?,
                finalization_pending,
                counts: counts.checked()?,
            })
        },
    )
    .transpose()
}
fn current(db: &Connection, path: &Path, token: &Token) -> Result<Head> {
    match head(db, path, token.input_pk, token.chain)? {
        Some(h) if h.token == *token => Ok(h),
        _ => Err(StoreError::StaleParserState),
    }
}
/// Starts or extends the parser state for one input chain. The same digest with
/// a longer acquired length keeps the generation (the caller guarantees the new
/// bytes extend the old ones); a different digest starts a new generation.
pub(crate) fn begin(db: &mut Connection, path: &Path, basis: &Basis<'_>) -> Result<Head> {
    let acquired = integer(basis.acquired_upto)?;
    let excisions = integer(basis.excisions_seen)?;
    let tx = sql!(path, db.transaction())?;
    let old = head(&tx, path, basis.input_pk, basis.chain)?;
    if let Some(old) = old
        .as_ref()
        .filter(|h| h.token.basis == basis.digest && h.status != Phase::Retired)
    {
        let identity: (Option<String>, u64) = sql!(path, tx.query_row(
            "SELECT companions_json, excisions_seen FROM parse_state WHERE input_pk=?1 AND chain=?2",
            params![basis.input_pk, basis.chain], |r| Ok((r.get(0)?, unsigned(r,1)?))))?;
        if old.parser_version != basis.parser_version
            || identity.0.as_deref() != basis.companions_json
            || identity.1 != basis.excisions_seen
        {
            return Err(StoreError::Invariant(
                "same input digest arrived with a different parser version, companions or excisions".into(),
            ));
        }
        if basis.acquired_upto < old.acquired_upto {
            return Err(StoreError::Invariant(
                "acquired length shrank for the same input digest".into(),
            ));
        }
        if basis.acquired_upto > old.acquired_upto {
            sql!(path, tx.execute("DELETE FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4", params![basis.input_pk,basis.chain,old.token.generation,SOURCE_ROWS_NAMESPACE]))?;
        }
        sql!(path, tx.execute(
            "UPDATE parse_state SET acquired_upto=?3,
             status=CASE WHEN acquired_upto<?3 AND status IN ('done','waiting_tail') THEN 'parsing' ELSE status END
             WHERE input_pk=?1 AND chain=?2", params![basis.input_pk, basis.chain, acquired]))?;
    } else {
        let generation = allocate_generation(&tx, path)?;
        prepared::invalidate(&tx, path, basis.input_pk, basis.chain)?;
        sql!(path, tx.execute(
            "INSERT INTO parse_state(input_pk,chain,parser_version,status,acquired_upto,inspected_upto,parsed_upto,
                basis_digest,state_generation,state_step,carry,companions_json,excisions_seen)
             VALUES(?1,?2,?3,'parsing',?4,0,0,?5,?6,0,?7,?8,?9)
             ON CONFLICT(input_pk,chain) DO UPDATE SET parser_version=excluded.parser_version,status='parsing',
                acquired_upto=excluded.acquired_upto,inspected_upto=0,parsed_upto=0,basis_digest=excluded.basis_digest,
                state_generation=excluded.state_generation,state_step=0,carry=excluded.carry,companions_json=excluded.companions_json,
                excisions_seen=excluded.excisions_seen,finalization_pending=0,detail=NULL,
                framed=0,used=0,ignored=0,unrecognized_n=0,undecodable=0",
            params![basis.input_pk,basis.chain,basis.parser_version,acquired,basis.digest.as_slice(),generation,
                carry(generation,0),basis.companions_json,excisions]))?;
    }
    let result =
        head(&tx, path, basis.input_pk, basis.chain)?.ok_or(StoreError::StaleParserState)?;
    sql!(path, tx.commit())?;
    Ok(result)
}

/// A read-only transaction pinned at one parser state. It holds WAL pages while
/// open, so release it before writing the step's result. A stale snapshot stays
/// readable; the writer rejects results built from it.
pub struct Snapshot {
    // Other store readers share this pinned view.
    pub(crate) db: Connection,
    pub(crate) path: std::path::PathBuf,
    pub head: Head,
    reads: std::cell::Cell<Option<ReadBudget>>,
    // Declared after `db` so the connection closes before the archive lock.
    lock: std::sync::Arc<std::fs::File>,
}
/// An idle connection reused across snapshots. It keeps SQLite caches but no
/// read transaction; each snapshot checks its token again.
pub struct ParserReader {
    db: Connection,
    path: std::path::PathBuf,
    lock: std::sync::Arc<std::fs::File>,
}
impl ParserReader {
    fn open(path: &Path, lock: std::sync::Arc<std::fs::File>) -> Result<Self> {
        Ok(Self {
            db: crate::open(path, true, false)?,
            path: path.to_owned(),
            lock,
        })
    }
    pub fn snapshot(self, expected: &Token) -> Result<Snapshot> {
        let Self { db, path, lock } = self;
        sql!(&path, db.execute_batch("BEGIN DEFERRED"))?;
        let head = current(&db, &path, expected)?;
        prepared::ensure_idle(&db, &path, expected)?;
        if head.status == Phase::Retired {
            return Err(StoreError::StaleParserState);
        }
        Ok(Snapshot {
            db,
            path,
            head,
            lock,
            reads: std::cell::Cell::new(None),
        })
    }
}
/// Read budget shared by every reader of one snapshot while a `ReadScope` is
/// open.
#[derive(Clone, Copy)]
struct ReadBudget {
    remaining: Budget,
    exceeded: bool,
    enforced: bool,
}
/// Meters snapshot reads against a budget until dropped. Overruns are only
/// recorded until `enforce` is called; after that they fail.
pub struct ReadScope<'a>(&'a Snapshot);
impl ReadScope<'_> {
    /// Makes further overruns fail. Returns false, and changes nothing, when
    /// the budget was already exceeded.
    pub fn enforce(&self) -> bool {
        let mut work = self.0.reads.get().expect("open read scope");
        if work.exceeded {
            return false;
        }
        work.enforced = true;
        self.0.reads.set(Some(work));
        true
    }
}
impl Drop for ReadScope<'_> {
    fn drop(&mut self) {
        self.0.reads.set(None);
    }
}
impl Snapshot {
    pub fn bounded_reads(&self, budget: Budget) -> Result<ReadScope<'_>> {
        if self.reads.get().is_some() {
            return Err(StoreError::Invariant(
                "snapshot read budget is already open".into(),
            ));
        }
        self.reads.set(Some(ReadBudget {
            remaining: budget,
            exceeded: false,
            enforced: false,
        }));
        Ok(ReadScope(self))
    }
    /// Charges bytes read outside this snapshot (before allocating them) to the
    /// open read budget. Does nothing without an open `ReadScope`.
    pub fn charge_read_bytes(&self, bytes: usize) -> Result<()> {
        if let Some(mut left) = self.reads.get() {
            if bytes > left.remaining.encoded_bytes {
                left.exceeded = true;
                self.reads.set(Some(left));
                if left.enforced {
                    return Err(StoreError::ParserAdmission {
                        resource: "worker read bytes",
                        required: bytes as u64,
                    });
                }
            }
            left.remaining.encoded_bytes = left.remaining.encoded_bytes.saturating_sub(bytes);
            self.reads.set(Some(left));
        }
        Ok(())
    }
    pub fn cumulative_reads_enforced(&self) -> bool {
        self.reads.get().is_some_and(|work| work.enforced)
    }
    pub fn charge_selector_record(&self, bytes: usize) -> Result<()> {
        self.charge_read("", &[], usize::MAX)?;
        self.charge_read_bytes(bytes)
    }
    /// Charges one cell lookup to the open budget and returns the byte
    /// allowance left for its result.
    fn charge_read(&self, namespace: &str, key: &[u8], allowance: usize) -> Result<usize> {
        if let Some(mut left) = self.reads.get() {
            if left.remaining.cells == 0 {
                left.exceeded = true;
                self.reads.set(Some(left));
                if left.enforced {
                    return Err(StoreError::ParserAdmission {
                        resource: "worker read cells",
                        required: 1,
                    });
                }
            }
            left.remaining.cells = left.remaining.cells.saturating_sub(1);
            self.reads.set(Some(left));
            self.charge_read_bytes(
                namespace
                    .len()
                    .checked_add(key.len())
                    .ok_or(StoreError::Range)?,
            )?;
            let work = self.reads.get().unwrap();
            return Ok(if work.enforced {
                allowance.min(work.remaining.encoded_bytes)
            } else {
                allowance
            });
        }
        Ok(allowance)
    }
    fn open(path: &Path, expected: &Token, lock: std::sync::Arc<std::fs::File>) -> Result<Self> {
        ParserReader::open(path, lock)?.snapshot(expected)
    }
    /// Ends the read transaction and returns the connection for reuse.
    pub fn into_reader(self) -> Result<ParserReader> {
        sql!(&self.path, self.db.execute_batch("ROLLBACK"))?;
        Ok(ParserReader {
            db: self.db,
            path: self.path,
            lock: self.lock,
        })
    }
    pub fn cell(&self, namespace: &str, key: &[u8], allowance: usize) -> Result<Option<Vec<u8>>> {
        let path = &self.path;
        let allowance = integer(self.charge_read(namespace, key, allowance)? as u64)?;
        let row = sql!(path, self.db.query_row(
            "SELECT length(value),CASE WHEN length(value)<=?6 THEN value ELSE NULL END FROM parser_cell
             WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",
            params![self.head.token.input_pk,self.head.token.chain,self.head.token.generation,namespace,key,allowance],
            |r| Ok((unsigned(r, 0)?,r.get::<_, Option<Vec<u8>>>(1)?)),
        ).optional())?;
        match row {
            None => Ok(None),
            Some((required, value)) => {
                self.charge_read_bytes(usize::try_from(required).map_err(|_| StoreError::Range)?)?;
                value.map(Some).ok_or(StoreError::ParserAdmission {
                    resource: "parser cell bytes",
                    required,
                })
            }
        }
    }
    /// The next (`after`) or previous key in one namespace, in byte order.
    /// Providers must encode keys so byte order matches their logical order.
    pub fn neighbor(
        &self,
        namespace: &str,
        key: &[u8],
        after: bool,
        allowance: usize,
    ) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let path = &self.path;
        let allowance = integer(self.charge_read(namespace, key, allowance)? as u64)?;
        let sql = if after {
            "SELECT length(key)+length(value),CASE WHEN length(key)+length(value)<=?6 THEN key ELSE NULL END,
                    CASE WHEN length(key)+length(value)<=?6 THEN value ELSE NULL END FROM parser_cell
             WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key>?5 ORDER BY key LIMIT 1"
        } else {
            "SELECT length(key)+length(value),CASE WHEN length(key)+length(value)<=?6 THEN key ELSE NULL END,
                    CASE WHEN length(key)+length(value)<=?6 THEN value ELSE NULL END FROM parser_cell
             WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key<?5 ORDER BY key DESC LIMIT 1"
        };
        let row = sql!(
            path,
            self.db
                .query_row(
                    sql,
                    params![
                        self.head.token.input_pk,
                        self.head.token.chain,
                        self.head.token.generation,
                        namespace,
                        key,
                        allowance
                    ],
                    |r| {
                        Ok((
                            unsigned(r, 0)?,
                            r.get::<_, Option<Vec<u8>>>(1)?,
                            r.get::<_, Option<Vec<u8>>>(2)?,
                        ))
                    },
                )
                .optional()
        )?;
        match row {
            None => Ok(None),
            Some((required, Some(key), Some(value))) => {
                self.charge_read_bytes(usize::try_from(required).map_err(|_| StoreError::Range)?)?;
                Ok(Some((key, value)))
            }
            Some((required, _, _)) => Err(StoreError::ParserAdmission {
                resource: "parser neighbor bytes",
                required,
            }),
        }
    }
}
fn check_changes(changes: &[CellChange], budget: Budget) -> Result<()> {
    if changes.len() > budget.cells {
        return Err(StoreError::ParserAdmission {
            resource: "parser delta cells",
            required: changes.len() as u64,
        });
    }
    let mut bytes = 0usize;
    for c in changes {
        if c.namespace.is_empty() || c.namespace.starts_with("@work:") {
            return Err(StoreError::Invariant(
                "empty or reserved parser cell namespace".into(),
            ));
        }
        bytes = bytes
            .saturating_add(c.namespace.len())
            .saturating_add(c.key.len())
            .saturating_add(c.value.as_ref().map_or(0, Vec::len));
        if bytes > budget.encoded_bytes {
            return Err(StoreError::ParserAdmission {
                resource: "parser delta encoded bytes",
                required: bytes as u64,
            });
        }
    }
    Ok(())
}
fn check_unchanged(expected: &Head, now: &Head) -> Result<()> {
    if expected.parsed_upto != now.parsed_upto
        || expected.inspected_upto != now.inspected_upto
        || expected.counts != now.counts
        || expected.parser_version != now.parser_version
        || expected.finalization_pending != now.finalization_pending
        || expected.acquired_upto > now.acquired_upto
    {
        return Err(StoreError::StaleParserState);
    }
    Ok(())
}
fn check_progress(expected: &Head, progress: &Progress) -> Result<()> {
    progress.counts.checked()?;
    if progress.inspected_upto < expected.inspected_upto
        || progress.parsed_upto < expected.parsed_upto
        || progress.parsed_upto > progress.inspected_upto
        || progress.inspected_upto > expected.acquired_upto
    {
        return Err(StoreError::Invariant(
            "parser progress moved backwards or past the acquired length".into(),
        ));
    }
    if progress.phase == Phase::Retired {
        return Err(StoreError::Invariant(
            "a parser step cannot retire its state".into(),
        ));
    }
    if matches!(progress.phase, Phase::Done)
        && (progress.parsed_upto != expected.acquired_upto || progress.finalization_pending)
    {
        return Err(StoreError::Invariant(
            "done requires everything parsed and nothing left to finalize".into(),
        ));
    }
    if matches!(progress.phase, Phase::WaitingTail)
        && (progress.inspected_upto != expected.acquired_upto
            || progress.parsed_upto == progress.inspected_upto)
    {
        return Err(StoreError::Invariant(
            "waiting_tail requires all bytes inspected and an incomplete record".into(),
        ));
    }
    if matches!(progress.phase, Phase::Finalizing) && !progress.finalization_pending {
        return Err(StoreError::Invariant(
            "finalizing requires pending finalization".into(),
        ));
    }
    Ok(())
}
/// Applies one parser step in a single transaction: the cell changes, `output`
/// (the caller's row writes, no parsing or file IO), then the new progress.
/// Production steps go through `prepared`; tests use this shortcut.
#[cfg(test)]
pub(crate) fn write_step<F>(
    db: &mut Connection,
    path: &Path,
    expected: &Head,
    progress: Progress,
    changes: &[CellChange],
    budget: Budget,
    output: F,
) -> Result<Head>
where
    F: FnOnce(&Transaction<'_>) -> Result<()>,
{
    step(db, path, expected, progress, changes, budget, false, output)
}
/// Fails with `StaleParserState` when the head moved since `expected` was read.
/// `from_prepared` skips the pending-work check for `prepared::publish`.
#[allow(clippy::too_many_arguments)]
fn step<F>(
    db: &mut Connection,
    path: &Path,
    expected: &Head,
    progress: Progress,
    changes: &[CellChange],
    budget: Budget,
    from_prepared: bool,
    output: F,
) -> Result<Head>
where
    F: FnOnce(&Transaction<'_>) -> Result<()>,
{
    check_changes(changes, budget)?;
    check_progress(expected, &progress)?;
    let next = expected
        .token
        .step
        .checked_add(1)
        .ok_or_else(|| StoreError::Invariant("parser state step exhausted".into()))?;
    let tx = sql!(path, db.transaction())?;
    let now = current(&tx, path, &expected.token)?;
    check_unchanged(expected, &now)?;
    if !from_prepared {
        prepared::ensure_idle(&tx, path, &expected.token)?;
    }
    if now.status == Phase::Retired {
        return Err(StoreError::StaleParserState);
    }
    let counts = now.counts.add(progress.counts)?;
    for c in changes {
        let args = params![
            expected.token.input_pk,
            expected.token.chain,
            expected.token.generation,
            &c.namespace,
            &c.key
        ];
        if let Some(value) = &c.value {
            sql!(path, tx.execute("INSERT INTO parser_cell(input_pk,chain,state_generation,namespace,key,value) VALUES(?1,?2,?3,?4,?5,?6)
                        ON CONFLICT(input_pk,chain,state_generation,namespace,key) DO UPDATE SET value=excluded.value",
                params![expected.token.input_pk,expected.token.chain,expected.token.generation,&c.namespace,&c.key,value]))?;
        } else {
            sql!(path, tx.execute("DELETE FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",args))?;
        }
    }
    output(&tx)?;
    let status = if now.acquired_upto > expected.acquired_upto
        && matches!(progress.phase, Phase::Done | Phase::WaitingTail)
    {
        "parsing"
    } else {
        progress.phase.as_str()
    };
    let inspected = integer(progress.inspected_upto)?;
    let parsed = integer(progress.parsed_upto)?;
    let record_counts = [
        counts.framed,
        counts.used,
        counts.ignored,
        counts.unrecognized,
        counts.undecodable,
    ]
    .map(integer)
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    sql!(path, tx.execute("UPDATE parse_state SET inspected_upto=?3,parsed_upto=?4,status=?5,finalization_pending=?6,state_step=?7,carry=?8,
                    framed=?9,used=?10,ignored=?11,unrecognized_n=?12,undecodable=?13
                WHERE input_pk=?1 AND chain=?2",
        params![expected.token.input_pk,expected.token.chain,inspected,parsed,status,
            progress.finalization_pending,next,carry(expected.token.generation,next),record_counts[0],record_counts[1],record_counts[2],record_counts[3],record_counts[4]]))?;
    let result = head(&tx, path, expected.token.input_pk, expected.token.chain)?
        .ok_or(StoreError::StaleParserState)?;
    sql!(path, tx.commit())?;
    Ok(result)
}

/// Moves the state to a new empty generation with status `retired`, so every
/// in-flight result becomes stale. Message rows are left to their owner.
pub(crate) fn retire(db: &mut Connection, path: &Path, expected: &Token) -> Result<Head> {
    let tx = sql!(path, db.transaction())?;
    current(&tx, path, expected)?;
    prepared::invalidate(&tx, path, expected.input_pk, expected.chain)?;
    let generation = allocate_generation(&tx, path)?;
    sql!(
        path,
        tx.execute(
            "UPDATE parse_state SET state_generation=?3,state_step=0,status='retired',carry=?4,
                acquired_upto=0,inspected_upto=0,parsed_upto=0,finalization_pending=0,
                framed=0,used=0,ignored=0,unrecognized_n=0,undecodable=0
                WHERE input_pk=?1 AND chain=?2",
            params![
                expected.input_pk,
                expected.chain,
                generation,
                carry(generation, 0)
            ],
        )
    )?;
    let result =
        head(&tx, path, expected.input_pk, expected.chain)?.ok_or(StoreError::StaleParserState)?;
    sql!(path, tx.commit())?;
    Ok(result)
}
#[derive(Debug)]
pub struct Collection {
    pub deleted: usize,
    pub encoded_bytes: usize,
    pub remaining: bool,
}
/// Deletes cells of generations older than `expected`, within the budget and
/// without loading their values. A first cell larger than the byte budget fails
/// with `ParserAdmission`.
pub(crate) fn collect_retired(
    db: &mut Connection,
    path: &Path,
    expected: &Token,
    budget: Budget,
) -> Result<Collection> {
    if budget.cells == 0 {
        return Err(StoreError::ParserAdmission {
            resource: "retired parser cells",
            required: 1,
        });
    }
    let encoded_allowance = integer(budget.encoded_bytes as u64)?;
    let cell_allowance = integer(budget.cells as u64)?;
    let tx = sql!(path, db.transaction())?;
    current(&tx, path, expected)?;
    let mut keys = Vec::new();
    let mut bytes = 0usize;
    {
        let mut st = sql!(path, tx.prepare_cached(
            "SELECT state_generation,length(CAST(namespace AS BLOB))+length(key)+length(value),
             CASE WHEN length(CAST(namespace AS BLOB))+length(key)+length(value)<=?4 THEN namespace ELSE NULL END,
             CASE WHEN length(CAST(namespace AS BLOB))+length(key)+length(value)<=?4 THEN key ELSE NULL END
             FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation<?3
             ORDER BY state_generation,namespace,key LIMIT ?5",
        ))?;
        let mut rows = sql!(
            path,
            st.query(params![
                expected.input_pk,
                expected.chain,
                expected.generation,
                encoded_allowance,
                cell_allowance
            ])
        )?;
        while let Some(row) = sql!(path, rows.next())? {
            let needed = usize::try_from(sql!(path, unsigned(row, 1))?).map_err(|_| {
                StoreError::Invariant("parser cell size exceeds addressable memory".into())
            })?;
            if needed > budget.encoded_bytes.saturating_sub(bytes) {
                if keys.is_empty() {
                    return Err(StoreError::ParserAdmission {
                        resource: "retired parser cell encoded bytes",
                        required: needed as u64,
                    });
                }
                break;
            }
            let namespace = sql!(path, row.get::<_, Option<String>>(2))?
                .ok_or_else(|| StoreError::Invariant("parser cell namespace missing".into()))?;
            let key = sql!(path, row.get::<_, Option<Vec<u8>>>(3))?
                .ok_or_else(|| StoreError::Invariant("parser cell key missing".into()))?;
            keys.push((sql!(path, row.get::<_, i64>(0))?, namespace, key));
            bytes += needed;
        }
    }
    let mut deleted = 0;
    for (generation, namespace, key) in keys {
        deleted+=sql!(path, tx.execute("DELETE FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",
            params![expected.input_pk,expected.chain,generation,namespace,key]))?;
    }
    let remaining=sql!(path, tx.query_row("SELECT EXISTS(SELECT 1 FROM parser_cell WHERE input_pk=?1 AND chain=?2 AND state_generation<?3 LIMIT 1)",
        params![expected.input_pk,expected.chain,expected.generation],|r|r.get(0)))?;
    sql!(path, tx.commit())?;
    Ok(Collection {
        deleted,
        encoded_bytes: bytes,
        remaining,
    })
}
/// Deletes a retired head once all its cells are gone. The input reference and
/// message rows are removed separately by their owners.
pub(crate) fn remove_retired(db: &mut Connection, path: &Path, expected: &Token) -> Result<()> {
    let tx = sql!(path, db.transaction())?;
    let head = current(&tx, path, expected)?;
    if head.status != Phase::Retired {
        return Err(StoreError::Invariant(
            "parser state must be retired before removal".into(),
        ));
    }
    let remains: bool = sql!(
        path,
        tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM parser_cell WHERE input_pk=?1 AND chain=?2 LIMIT 1)",
            params![expected.input_pk, expected.chain],
            |r| r.get(0),
        )
    )?;
    if remains {
        return Err(StoreError::Invariant(
            "parser cells must be deleted before their parser state".into(),
        ));
    }
    sql!(
        path,
        tx.execute(
            "DELETE FROM parse_state WHERE input_pk=?1 AND chain=?2",
            params![expected.input_pk, expected.chain],
        )
    )?;
    sql!(path, tx.commit())?;
    Ok(())
}

impl crate::Writer {
    pub fn parser_head(&self, input_pk: i64, chain: i64) -> Result<Option<Head>> {
        head(&self._index, &self.root.join("index.db"), input_pk, chain)
    }
    /// The failure detail recorded for this state, if any (at most 4096 bytes).
    pub fn parser_failure(&self, token: &Token) -> Result<Option<String>> {
        let path = self.root.join("index.db");
        current(&self._index, &path, token)?;
        let (length,detail):(Option<i64>,Option<String>)=sql!(&path,self._index.query_row(
            "SELECT length(CAST(detail AS BLOB)), CASE WHEN length(CAST(detail AS BLOB))<=4096 THEN detail ELSE NULL END FROM parse_state WHERE input_pk=?1 AND chain=?2",
            params![token.input_pk,token.chain],|r|Ok((r.get(0)?,r.get(1)?))))?;
        if let Some(length) = length.filter(|n| *n > 4096) {
            return Err(StoreError::ParserAdmission {
                resource: "parser diagnostic bytes",
                required: length as u64,
            });
        }
        Ok(detail)
    }
    /// Records a failure detail, or clears it with `None` to allow a retry.
    /// Progress and pending finalization are unchanged.
    pub fn record_parser_failure(
        &mut self,
        token: &Token,
        detail: Option<&str>,
        allowance: usize,
    ) -> Result<()> {
        let required = detail.map_or(0, str::len);
        if required > allowance.min(4096) {
            return Err(StoreError::ParserAdmission {
                resource: "parser diagnostic bytes",
                required: required as u64,
            });
        }
        let path = self.root.join("index.db");
        let tx = sql!(&path, self._index.transaction())?;
        current(&tx, &path, token)?;
        sql!(
            &path,
            tx.execute(
                "UPDATE parse_state SET detail=?3 WHERE input_pk=?1 AND chain=?2",
                params![token.input_pk, token.chain, detail]
            )
        )?;
        sql!(&path, tx.commit())?;
        Ok(())
    }
    /// See [`begin`].
    pub fn begin_parser(&mut self, basis: &Basis<'_>) -> Result<Head> {
        begin(&mut self._index, &self.root.join("index.db"), basis)
    }
    pub fn parser_snapshot(&self, token: &Token) -> Result<Snapshot> {
        Snapshot::open(
            &self.root.join("index.db"),
            token,
            std::sync::Arc::clone(&self._lock),
        )
    }
    pub fn parser_reader(&self) -> Result<ParserReader> {
        ParserReader::open(
            &self.root.join("index.db"),
            std::sync::Arc::clone(&self._lock),
        )
    }
}
impl crate::ReadHandle {
    pub fn parser_snapshot(&self, token: &Token) -> Result<Snapshot> {
        Snapshot::open(
            &self.root.join("index.db"),
            token,
            std::sync::Arc::clone(&self._lock),
        )
    }
}

/// Cell namespace holding the source-row receipt of the current generation.
pub const SOURCE_ROWS_NAMESPACE: &str = "source.rows.v1";
impl crate::Writer {
    /// Stores the source-row receipt for a finalizing state that has parsed all
    /// acquired bytes. With `complete`, also marks the state done, which fails
    /// if any records were unrecognized or undecodable.
    pub fn publish_source_rows(
        &mut self,
        expected: &Head,
        receipt: &[u8],
        complete: bool,
    ) -> Result<Head> {
        if receipt.is_empty() || receipt.len() > 16384 {
            return Err(StoreError::ParserAdmission {
                resource: "source row receipt",
                required: receipt.len() as u64,
            });
        }
        let path = self.root.join("index.db");
        let tx = sql!(&path, self._index.transaction())?;
        let now = current(&tx, &path, &expected.token)?;
        check_unchanged(expected, &now)?;
        prepared::ensure_idle(&tx, &path, &expected.token)?;
        if now.status != Phase::Finalizing
            || !now.finalization_pending
            || now.parsed_upto != now.acquired_upto
        {
            return Err(StoreError::StaleParserState);
        }
        if complete && (now.counts.unrecognized > 0 || now.counts.undecodable > 0) {
            return Err(StoreError::Invariant(
                "source rows cannot complete with unrecognized or undecodable records".into(),
            ));
        }
        let excised: bool = sql!(
            &path,
            tx.query_row(
                "SELECT excised FROM input_ref WHERE pk=?1",
                [now.token.input_pk],
                |r| r.get(0)
            )
        )?;
        if excised {
            return Err(StoreError::StaleParserState);
        }
        sql!(&path,tx.execute("INSERT INTO parser_cell(input_pk,chain,state_generation,namespace,key,value) VALUES(?1,?2,?3,?4,x'',?5) ON CONFLICT DO NOTHING",rusqlite::params![now.token.input_pk,now.token.chain,now.token.generation,SOURCE_ROWS_NAMESPACE,receipt]))?;
        if tx.changes() != 1 {
            return Err(StoreError::StaleParserState);
        }
        if complete {
            sql!(&path,tx.execute("UPDATE parse_state SET status='done',finalization_pending=0 WHERE input_pk=?1 AND chain=?2",rusqlite::params![now.token.input_pk,now.token.chain]))?;
        }
        let result = head(&tx, &path, now.token.input_pk, now.token.chain)?
            .ok_or(StoreError::StaleParserState)?;
        sql!(&path, tx.commit())?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
