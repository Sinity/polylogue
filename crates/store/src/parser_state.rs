//! Addressable disposable parser state, owned by index.db. Interpretation uses a
//! real read-only SQLite snapshot; only its bounded touched delta enters the
//! output transaction. This module checks predecessor identity, not whether the
//! canonical message/index finalizer has actually earned completion.
use crate::{Error as StoreError, Result};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::path::Path;

// Every actual SQLite operation uses the owning path/checksum error boundary.
macro_rules! sql {
    ($path:expr, $operation:expr) => {
        crate::db($path, || $operation)
    };
}

pub mod prepared;

#[derive(Clone, Debug, PartialEq, Eq)]
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
                "unbalanced parser record accounting".into(),
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
                .ok_or_else(|| StoreError::Invariant("parser accounting overflow".into()))
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
    pub status: String,
    pub finalization_pending: bool,
    pub counts: Accounting,
}
pub struct Basis<'a> {
    pub input_pk: i64,
    pub chain: i64,
    pub parser_version: u32,
    pub digest: [u8; 32],
    pub acquired_upto: u64,
    pub companions_json: Option<&'a str>,
    pub excisions_seen: u64,
}
#[derive(Clone, Copy, Debug)]
pub enum Phase {
    Parsing,
    WaitingTail,
    Finalizing,
    Done,
}
impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Parsing => "parsing",
            Self::WaitingTail => "waiting_tail",
            Self::Finalizing => "finalizing",
            Self::Done => "done",
        }
    }
}
pub struct Progress {
    pub inspected_upto: u64,
    pub parsed_upto: u64,
    /// Supplied by the owning finalizer after checking its actual obligations.
    /// A provider reaching byte EOF is not evidence for Done.
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
// Called only within the owning write transaction. The allocator survives head
// and input_ref removal, so reusing either key cannot recreate an old token.
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
            StoreError::Invariant("parser generation allocator invalid or exhausted".into())
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
fn reference(generation: i64, step: i64) -> Vec<u8> {
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
            if carry != reference(generation, step) {
                return Err(StoreError::Invariant(
                    "parser state resume reference differs from its head".into(),
                ));
            }
            let basis = digest
                .try_into()
                .map_err(|_| StoreError::Invariant("parser basis digest is not 32 bytes".into()))?;
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
                status,
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
        _ => Err(StoreError::ParserPredecessorChanged),
    }
}
/// Admission/namespace changes are metadata transactions, not provider work.
/// The caller proves same-digest growth is a verified contiguous acquisition.
pub(crate) fn begin(db: &mut Connection, path: &Path, basis: &Basis<'_>) -> Result<Head> {
    let acquired = integer(basis.acquired_upto)?;
    let excisions = integer(basis.excisions_seen)?;
    let tx = sql!(path, db.transaction())?;
    let old = head(&tx, path, basis.input_pk, basis.chain)?;
    if let Some(old) = old
        .as_ref()
        .filter(|h| h.token.basis == basis.digest && h.status != "retired")
    {
        let identity: (Option<String>, u64) = sql!(path, tx.query_row(
            "SELECT companions_json, excisions_seen FROM parse_state WHERE input_pk=?1 AND chain=?2",
            params![basis.input_pk, basis.chain], |r| Ok((r.get(0)?, unsigned(r,1)?))))?;
        if old.parser_version != basis.parser_version
            || identity.0.as_deref() != basis.companions_json
            || identity.1 != basis.excisions_seen
        {
            return Err(StoreError::Invariant(
                "unchanged parser basis changed its interpretation identity".into(),
            ));
        }
        if basis.acquired_upto < old.acquired_upto {
            return Err(StoreError::Invariant(
                "same parser basis cannot shrink its acquired prefix".into(),
            ));
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
                reference(generation,0),basis.companions_json,excisions]))?;
    }
    let result = head(&tx, path, basis.input_pk, basis.chain)?
        .ok_or(StoreError::ParserPredecessorChanged)?;
    sql!(path, tx.commit())?;
    Ok(result)
}

/// A live read-only snapshot, not merely a mutable generation/key lookup.
/// Its lifetime pins WAL pages and must stay charged/admitted by the owner.
/// Drop it before applying the result. An old snapshot stays internally coherent;
/// the output writer subsequently rejects its stale predecessor.
pub struct Snapshot {
    db: Connection,
    path: std::path::PathBuf,
    pub head: Head,
    // Drop the SQLite snapshot before releasing its retained archive authority.
    _lock: std::sync::Arc<std::fs::File>,
}
impl Snapshot {
    fn open(path: &Path, expected: &Token, lock: std::sync::Arc<std::fs::File>) -> Result<Self> {
        let db = crate::open(path, true, false)?;
        sql!(path, db.execute_batch("BEGIN DEFERRED"))?;
        let head = current(&db, path, expected)?; // Establishes the actual read snapshot now.
        prepared::ensure_idle(&db, path, expected)?;
        if head.status == "retired" {
            return Err(StoreError::ParserPredecessorChanged);
        }
        Ok(Self {
            db,
            path: path.to_owned(),
            head,
            _lock: lock,
        })
    }
    pub fn cell(&self, namespace: &str, key: &[u8], allowance: usize) -> Result<Option<Vec<u8>>> {
        let path = &self.path;
        let allowance = integer(allowance as u64)?;
        let row = sql!(path, self.db.query_row(
            "SELECT length(value),CASE WHEN length(value)<=?6 THEN value ELSE NULL END FROM parser_cell
             WHERE input_pk=?1 AND chain=?2 AND state_generation=?3 AND namespace=?4 AND key=?5",
            params![self.head.token.input_pk,self.head.token.chain,self.head.token.generation,namespace,key,allowance],
            |r| Ok((unsigned(r, 0)?,r.get::<_, Option<Vec<u8>>>(1)?)),
        ).optional())?;
        match row {
            None => Ok(None),
            Some((required, value)) => value.map(Some).ok_or(StoreError::ParserAdmission {
                resource: "parser cell bytes",
                required,
            }),
        }
    }
    /// Keyset neighbor within one provider-owned family/namespace. The key codec
    /// must preserve that provider's order; JSON lexical order is not a substitute.
    pub fn neighbor(
        &self,
        namespace: &str,
        key: &[u8],
        after: bool,
        allowance: usize,
    ) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let path = &self.path;
        let allowance = integer(allowance as u64)?;
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
            Some((_, Some(key), Some(value))) => Ok(Some((key, value))),
            Some((required, _, _)) => Err(StoreError::ParserAdmission {
                resource: "parser neighbor bytes",
                required,
            }),
        }
    }
}
fn validate(changes: &[CellChange], budget: Budget) -> Result<()> {
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
fn validate_expected_head(expected: &Head, now: &Head) -> Result<()> {
    if expected.parsed_upto != now.parsed_upto
        || expected.inspected_upto != now.inspected_upto
        || expected.counts != now.counts
        || expected.parser_version != now.parser_version
        || expected.finalization_pending != now.finalization_pending
        || expected.acquired_upto > now.acquired_upto
    {
        return Err(StoreError::ParserPredecessorChanged);
    }
    Ok(())
}
fn validate_progress(expected: &Head, progress: &Progress) -> Result<()> {
    progress.counts.checked()?;
    if progress.inspected_upto < expected.inspected_upto
        || progress.parsed_upto < expected.parsed_upto
        || progress.parsed_upto > progress.inspected_upto
        || progress.inspected_upto > expected.acquired_upto
    {
        return Err(StoreError::Invariant(
            "parser step exceeds or reverses its acquired basis".into(),
        ));
    }
    if matches!(progress.phase, Phase::Done)
        && (progress.parsed_upto != expected.acquired_upto || progress.finalization_pending)
    {
        return Err(StoreError::Invariant(
            "byte EOF or pending finalization cannot certify done".into(),
        ));
    }
    if matches!(progress.phase, Phase::WaitingTail)
        && (progress.inspected_upto != expected.acquired_upto
            || progress.parsed_upto == progress.inspected_upto)
    {
        return Err(StoreError::Invariant(
            "waiting tail requires a fully inspected incomplete record".into(),
        ));
    }
    if matches!(progress.phase, Phase::Finalizing) && !progress.finalization_pending {
        return Err(StoreError::Invariant(
            "finalizing requires an actual remaining obligation".into(),
        ));
    }
    Ok(())
}
/// Cells must already be encoded/admitted outside the write transaction. The
/// output closure is the existing writer's bounded row step, never interpretation
/// or file IO. A successful receipt is returned only after the transaction commits.
/// This primitive does not certify the semantic completeness of the closure.
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
    write_step_inner(db, path, expected, progress, changes, budget, false, output)
}
fn write_step_inner<F>(
    db: &mut Connection,
    path: &Path,
    expected: &Head,
    progress: Progress,
    changes: &[CellChange],
    budget: Budget,
    prepared: bool,
    output: F,
) -> Result<Head>
where
    F: FnOnce(&Transaction<'_>) -> Result<()>,
{
    validate(changes, budget)?;
    validate_progress(expected, &progress)?;
    let next = expected
        .token
        .step
        .checked_add(1)
        .ok_or_else(|| StoreError::Invariant("parser state step exhausted".into()))?;
    let tx = sql!(path, db.transaction())?;
    let now = current(&tx, path, &expected.token)?;
    validate_expected_head(expected, &now)?;
    if !prepared {
        prepared::ensure_idle(&tx, path, &expected.token)?;
    }
    if now.status == "retired" {
        return Err(StoreError::ParserPredecessorChanged);
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
        progress.phase.name()
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
            progress.finalization_pending,next,reference(expected.token.generation,next),record_counts[0],record_counts[1],record_counts[2],record_counts[3],record_counts[4]]))?;
    let result = head(&tx, path, expected.token.input_pk, expected.token.chain)?
        .ok_or(StoreError::ParserPredecessorChanged)?;
    sql!(path, tx.commit())?;
    Ok(result)
}

/// Invalidate the current working state before draining it. The new empty
/// generation rejects every in-flight result; canonical rows are not removed by
/// this operation and remain the excision/reconciliation owner's responsibility.
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
                reference(generation, 0)
            ],
        )
    )?;
    let result = head(&tx, path, expected.input_pk, expected.chain)?
        .ok_or(StoreError::ParserPredecessorChanged)?;
    sql!(path, tx.commit())?;
    Ok(result)
}
#[derive(Debug)]
pub struct Collection {
    pub deleted: usize,
    pub encoded_bytes: usize,
    pub remaining: bool,
}
/// Drain retired generations by indexed keysets, without reading their values
/// into memory. A cell exceeding the admitted work allowance remains owed and
/// requires a larger explicit allowance. This is not a bound on SQLite page IO.
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
                .ok_or_else(|| StoreError::Invariant("missing admitted parser namespace".into()))?;
            let key = sql!(path, row.get::<_, Option<Vec<u8>>>(3))?
                .ok_or_else(|| StoreError::Invariant("missing admitted parser key".into()))?;
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
/// Remove only an explicitly retired, fully drained head. Input-reference and
/// canonical-contribution removal is the caller's separate existing lifecycle.
pub(crate) fn remove_retired(db: &mut Connection, path: &Path, expected: &Token) -> Result<()> {
    let tx = sql!(path, db.transaction())?;
    let head = current(&tx, path, expected)?;
    if head.status != "retired" {
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
            "drain parser cells before removing their head/input reference".into(),
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
    /// Inspect disposable state through the private index connection.
    pub fn parser_head(&self, input_pk: i64, chain: i64) -> Result<Option<Head>> {
        head(&self._index, &self.root.join("index.db"), input_pk, chain)
    }
    /// Admit a verified contiguous retained-source basis. This does not publish
    /// messages and never certifies row finalization merely from byte EOF.
    pub fn begin_parser(&mut self, basis: &Basis<'_>) -> Result<Head> {
        begin(&mut self._index, &self.root.join("index.db"), basis)
    }
    pub fn parser_snapshot(&self, predecessor: &Token) -> Result<Snapshot> {
        Snapshot::open(
            &self.root.join("index.db"),
            predecessor,
            std::sync::Arc::clone(&self._lock),
        )
    }
}
impl crate::ReadHandle {
    pub fn parser_snapshot(&self, predecessor: &Token) -> Result<Snapshot> {
        Snapshot::open(
            &self.root.join("index.db"),
            predecessor,
            std::sync::Arc::clone(&self._lock),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Writer;
    fn fixture() -> (tempfile::TempDir, std::path::PathBuf, Writer, i64) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let writer = Writer::open(&root).unwrap();
        writer
            ._index
            .execute("INSERT INTO input_ref(id) VALUES('synthetic-state')", [])
            .unwrap();
        let input = writer
            ._index
            .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
            .unwrap();
        (dir, root, writer, input)
    }
    fn basis(input_pk: i64, end: u64) -> Basis<'static> {
        Basis {
            input_pk,
            chain: 1,
            parser_version: 1,
            digest: [31; 32],
            acquired_upto: end,
            companions_json: None,
            excisions_seen: 0,
        }
    }
    fn budget() -> Budget {
        Budget {
            cells: 16,
            encoded_bytes: 4096,
        }
    }
    fn change(key: &str, value: &str) -> CellChange {
        CellChange {
            namespace: "test".into(),
            key: key.as_bytes().to_vec(),
            value: Some(value.as_bytes().to_vec()),
        }
    }
    #[test]
    fn parser_snapshot_retains_archive_authority_after_its_factory_closes() {
        let (_dir, root, mut writer, input) = fixture();
        let head = writer.begin_parser(&basis(input, 100)).unwrap();
        let reader = writer.read_handle().unwrap();
        let snapshot = reader.parser_snapshot(&head.token).unwrap();
        drop(reader);
        drop(writer);
        assert!(matches!(
            Writer::open(&root),
            Err(StoreError::WriterHeld { .. })
        ));
        assert!(matches!(
            crate::ReadHandle::open(&root),
            Err(StoreError::WriterHeld { .. })
        ));
        assert_eq!(snapshot.cell("test", b"absent", 100).unwrap(), None);
        drop(snapshot);
        assert!(Writer::open(&root).is_ok());
    }
    fn progress(parsed: u64, inspected: u64, phase: Phase, pending: bool) -> Progress {
        Progress {
            parsed_upto: parsed,
            inspected_upto: inspected,
            phase,
            finalization_pending: pending,
            counts: Accounting::default(),
        }
    }
    #[test]
    fn target_private_writer_snapshot_and_stale_step_preserve_newer_acquisition() {
        let (_dir, root, mut writer, input) = fixture();
        let file = root.join("index.db");
        let head = writer.begin_parser(&basis(input, 10)).unwrap();
        let head = write_step(
            &mut writer._index,
            &file,
            &head,
            progress(5, 5, Phase::Parsing, true),
            &[change("a", "old")],
            budget(),
            |_| Ok(()),
        )
        .unwrap();
        let old = writer
            .read_handle()
            .unwrap()
            .parser_snapshot(&head.token)
            .unwrap();
        let extended = writer.begin_parser(&basis(input, 20)).unwrap();
        assert_eq!(extended.token, head.token);
        let next = write_step(
            &mut writer._index,
            &file,
            &head,
            progress(10, 10, Phase::Done, false),
            &[change("a", "new")],
            budget(),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(next.acquired_upto, 20);
        assert_eq!(next.status, "parsing");
        assert_eq!(old.cell("test", b"a", 10).unwrap().unwrap(), b"old");
        assert!(matches!(
            writer.parser_snapshot(&head.token),
            Err(StoreError::ParserPredecessorChanged)
        ));
        assert!(matches!(
            write_step(
                &mut writer._index,
                &file,
                &head,
                progress(10, 10, Phase::Done, false),
                &[],
                budget(),
                |_| panic!("stale output")
            ),
            Err(StoreError::ParserPredecessorChanged)
        ));
        let fresh = writer.parser_snapshot(&next.token).unwrap();
        assert_eq!(fresh.cell("test", b"a", 10).unwrap().unwrap(), b"new");
    }
    #[test]
    fn removed_head_and_reused_input_key_never_recreate_a_target_token() {
        let (_dir, root, mut writer, input) = fixture();
        let file = root.join("index.db");
        let old = writer.begin_parser(&basis(input, 10)).unwrap();
        let retired = retire(&mut writer._index, &file, &old.token).unwrap();
        remove_retired(&mut writer._index, &file, &retired.token).unwrap();
        writer
            ._index
            .execute("DELETE FROM input_ref WHERE pk=?1", [input])
            .unwrap();
        writer
            ._index
            .execute("INSERT INTO input_ref(id) VALUES('replacement')", [])
            .unwrap();
        let reused: i64 = writer
            ._index
            .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
            .unwrap();
        assert_eq!(input, reused);
        let next = writer.begin_parser(&basis(reused, 10)).unwrap();
        assert_ne!(next.token, old.token);
        assert!(matches!(
            writer.parser_snapshot(&old.token),
            Err(StoreError::ParserPredecessorChanged)
        ));
    }
    #[test]
    fn partial_tail_and_retirement_are_independent_admitted_work() {
        let (_dir, root, mut writer, input) = fixture();
        let file = root.join("index.db");
        let head = writer.begin_parser(&basis(input, 10)).unwrap();
        let head = write_step(
            &mut writer._index,
            &file,
            &head,
            progress(5, 10, Phase::WaitingTail, true),
            &[change("a", &"x".repeat(100)), change("b", "second")],
            budget(),
            |_| Ok(()),
        )
        .unwrap();
        let view = writer.parser_snapshot(&head.token).unwrap();
        assert!(matches!(
            view.cell("test", b"a", 10),
            Err(StoreError::ParserAdmission { required: 100, .. })
        ));
        assert!(head.finalization_pending);
        assert_eq!(head.status, "waiting_tail");
        assert_eq!(
            writer.begin_parser(&basis(input, 10)).unwrap().status,
            "waiting_tail"
        );
        drop(view);
        let retired = retire(&mut writer._index, &file, &head.token).unwrap();
        assert!(remove_retired(&mut writer._index, &file, &retired.token).is_err());
        let first = collect_retired(
            &mut writer._index,
            &file,
            &retired.token,
            Budget {
                cells: 1,
                encoded_bytes: 200,
            },
        )
        .unwrap();
        assert_eq!(first.deleted, 1);
        assert!(first.remaining);
        let second = collect_retired(
            &mut writer._index,
            &file,
            &retired.token,
            Budget {
                cells: 1,
                encoded_bytes: 200,
            },
        )
        .unwrap();
        assert_eq!(second.deleted, 1);
        assert!(!second.remaining);
        remove_retired(&mut writer._index, &file, &retired.token).unwrap();
    }
    #[test]
    fn child_process_exit_releases_real_writer_lock_and_rolls_back_step() {
        let (_dir, root, mut writer, input) = fixture();
        let file = root.join("index.db");
        let head = writer.begin_parser(&basis(input, 10)).unwrap();
        drop(writer);
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "parser_state::tests::crash_child", "--nocapture"])
            .env("PARSER_TARGET_CRASH_ROOT", &root)
            .output()
            .unwrap();
        assert_eq!(child.status.code(), Some(71), "{child:?}");
        let mut writer = Writer::open(&root).unwrap();
        let restored = writer.parser_head(input, 1).unwrap().unwrap();
        assert_eq!(restored.token, head.token);
        assert!(
            writer
                .parser_snapshot(&head.token)
                .unwrap()
                .cell("test", b"a", 10)
                .unwrap()
                .is_none()
        );
        let count: i64 = writer
            ._index
            .query_row("SELECT count(*) FROM unrecognized", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
        let mut p = progress(10, 10, Phase::Finalizing, true);
        p.counts = Accounting {
            framed: 1,
            unrecognized: 1,
            ..Accounting::default()
        };
        let next=write_step(&mut writer._index,&file,&head,p,&[change("a","new")],budget(), |tx| {
            crate::db(&file,||tx.execute("INSERT INTO unrecognized(input_pk,chain,format,what,count) VALUES(?1,1,'synthetic','record:test',1)",[input]))?; Ok(())
        }).unwrap();
        assert_eq!(next.token.step, 1);
        assert_eq!(next.counts.framed, 1);
        assert!(next.finalization_pending);
        let count: i64 = writer
            ._index
            .query_row("SELECT count(*) FROM unrecognized", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }
    #[test]
    fn crash_child() {
        let Some(root) = std::env::var_os("PARSER_TARGET_CRASH_ROOT") else {
            return;
        };
        let root = std::path::PathBuf::from(root);
        let file = root.join("index.db");
        let mut writer = Writer::open(&root).unwrap();
        let input = writer
            ._index
            .query_row("SELECT pk FROM input_ref", [], |r| r.get(0))
            .unwrap();
        let head = writer.parser_head(input, 1).unwrap().unwrap();
        let _ = write_step(
            &mut writer._index,
            &file,
            &head,
            progress(10, 10, Phase::Finalizing, true),
            &[change("a", "new")],
            budget(),
            |tx| {
                crate::db(&file, || {
                    tx.execute("INSERT INTO unrecognized(input_pk,chain,format,what,count) VALUES(?1,1,'synthetic','record:test',1)",[input])
                })?;
                std::process::exit(71)
            },
        );
        panic!("crash callback must exit before commit");
    }
}
