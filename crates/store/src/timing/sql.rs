//! Temporary writer SQL attribution, using the existing worker-timing switch.
//! SQL is unexpanded; parameter values never enter the diagnostic. Statement
//! lifetimes include time between cursor steps and must not be summed as CPU.
use crate::{Result, StoreError};
use rusqlite::{Connection, Transaction, ffi};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::os::raw::{c_int, c_uint, c_void};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

const STATEMENTS: usize = 128;
const ACTIVE: usize = 64;
const SQL_BYTES: usize = 4096;
const SNAPSHOTS: usize = 12;

#[derive(Clone, Default, Serialize)]
struct StatementWork {
    phase: &'static str,
    sql_hash: String,
    verb: String,
    calls: u64,
    lifetime_ns: u64,
    vm_steps: u64,
    fullscan_steps: u64,
    sorts: u64,
    invalid_counters: u64,
}

#[derive(Clone, Copy, Default, Serialize)]
struct Io {
    rchar: u64,
    wchar: u64,
    syscr: u64,
    syscw: u64,
    read_bytes: u64,
    write_bytes: u64,
    probe_read_bytes: u64,
}

fn io() -> Option<Io> {
    // One read gives an exact observer-read correction, unlike read_to_string.
    let mut file = std::fs::File::open("/proc/thread-self/io").ok()?;
    let mut bytes = [0; 4096];
    let size = file.read(&mut bytes).ok()?;
    if size == bytes.len() {
        return None;
    }
    let rows: BTreeMap<_, _> = std::str::from_utf8(&bytes[..size])
        .ok()?
        .lines()
        .filter_map(|line| line.split_once(':'))
        .collect();
    let get = |key| rows.get(key)?.trim().parse::<u64>().ok();
    Some(Io {
        rchar: get("rchar")?,
        wchar: get("wchar")?,
        syscr: get("syscr")?,
        syscw: get("syscw")?,
        read_bytes: get("read_bytes")?,
        write_bytes: get("write_bytes")?,
        probe_read_bytes: size as u64,
    })
}

#[allow(unsafe_code)]
fn cpu() -> Option<u64> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: the thread clock writes one live timespec and owns no state.
    let code = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut value) };
    (code == 0)
        .then(|| {
            u64::try_from(value.tv_sec)
                .ok()?
                .checked_mul(1_000_000_000)?
                .checked_add(u64::try_from(value.tv_nsec).ok()?)
        })
        .flatten()
}

#[allow(unsafe_code)]
unsafe fn cache(db: *mut ffi::sqlite3) -> [Option<u64>; 3] {
    [
        ffi::SQLITE_DBSTATUS_CACHE_HIT,
        ffi::SQLITE_DBSTATUS_CACHE_MISS,
        ffi::SQLITE_DBSTATUS_CACHE_WRITE,
    ]
    .map(|kind| {
        let (mut current, mut high) = (0, 0);
        // SAFETY: callers keep this connection live for the complete call.
        let code = unsafe { ffi::sqlite3_db_status(db, kind, &mut current, &mut high, 0) };
        (code == ffi::SQLITE_OK)
            .then(|| u64::try_from(current).ok())
            .flatten()
    })
}

#[derive(Clone, Default, Serialize)]
struct PhaseWork {
    calls: u64,
    wall_ns: u64,
    cpu_ns: u64,
    cache_hits: u64,
    cache_misses: u64,
    cache_writes: u64,
    rchar: u64,
    syscr: u64,
    read_bytes: u64,
    write_bytes: u64,
    observer_output_bytes: u64,
    unavailable: u64,
}

struct Before {
    wall: Instant,
    cpu: Option<u64>,
    cache: [Option<u64>; 3],
    io: Option<Io>,
    output_bytes: u64,
    previous: &'static str,
}
struct State {
    phase: &'static str,
    statements: BTreeMap<String, StatementWork>,
    active: BTreeMap<usize, String>,
    phases: BTreeMap<&'static str, PhaseWork>,
    dropped_statements: u64,
    dropped_active: u64,
    oversized_sql: u64,
    attach_failures: u64,
    callback_errors: u64,
    output_bytes: u64,
    output_errors: u64,
    snapshots: usize,
    last_snapshot: Instant,
}
impl Default for State {
    fn default() -> Self {
        Self {
            phase: "writer_other",
            statements: BTreeMap::new(),
            active: BTreeMap::new(),
            phases: BTreeMap::new(),
            dropped_statements: 0,
            dropped_active: 0,
            oversized_sql: 0,
            attach_failures: 0,
            callback_errors: 0,
            output_bytes: 0,
            output_errors: 0,
            snapshots: 0,
            last_snapshot: Instant::now(),
        }
    }
}
thread_local! { static STATE: RefCell<Option<State>> = const { RefCell::new(None) }; }

impl State {
    fn emit(&mut self, value: serde_json::Value) {
        if let Ok(mut line) = serde_json::to_vec(&value) {
            line.push(b'\n');
            self.output_bytes += line.len() as u64;
            if std::io::stderr().lock().write_all(&line).is_err() {
                self.output_errors += 1;
            }
        }
    }
    fn flush(&mut self, final_record: bool) {
        if !final_record
            && (self.snapshots >= SNAPSHOTS
                || self.last_snapshot.elapsed() < Duration::from_secs(5))
        {
            return;
        }
        self.last_snapshot = Instant::now();
        self.snapshots += usize::from(!final_record);
        self.emit(serde_json::json!({"writer_sql_profile":{
            "final":final_record,"phase":self.phase,
            "statements":self.statements.values().collect::<Vec<_>>(),
            "phases":self.phases,"active":self.active.values().collect::<Vec<_>>(),
            "dropped_statements":self.dropped_statements,"dropped_active":self.dropped_active,
            "oversized_sql":self.oversized_sql,"attach_failures":self.attach_failures,"callback_errors":self.callback_errors,"observer_output_bytes":self.output_bytes,
            "output_errors":self.output_errors,"statement_time_is_exclusive":false,
            "cache_pages_are_physical_io":false}}));
    }
}

pub(crate) struct Worker(bool);
pub(crate) fn worker(conn: &Connection) -> Worker {
    let enabled = super::WORKERS_ENABLED.load(Ordering::Relaxed);
    if enabled {
        STATE.with(|state| *state.borrow_mut() = Some(State::default()));
        attach(conn);
    }
    Worker(enabled)
}
impl Drop for Worker {
    fn drop(&mut self) {
        if self.0 {
            STATE.with(|state| {
                if let Some(mut state) = state.borrow_mut().take() {
                    state.flush(true);
                }
            });
        }
    }
}

#[allow(unsafe_code)]
pub(crate) fn attach(conn: &Connection) {
    if STATE.with(|state| state.borrow().is_none()) {
        return;
    }
    // SAFETY: SQLite owns the callback's statement and connection arguments.
    // No Rust allocation is stored in its context; closing/replacing a traced
    // connection cannot leave a dangling collector or require a cleanup query.
    let result = unsafe {
        ffi::sqlite3_trace_v2(
            conn.handle(),
            ffi::SQLITE_TRACE_STMT | ffi::SQLITE_TRACE_PROFILE,
            Some(observe),
            std::ptr::null_mut(),
        )
    };
    if result != ffi::SQLITE_OK {
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                state.attach_failures += 1;
            }
        });
    }
}

#[allow(unsafe_code)]
unsafe extern "C" fn observe(
    kind: c_uint,
    _: *mut c_void,
    statement: *mut c_void,
    detail: *mut c_void,
) -> c_int {
    let observed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            let Some(state) = state.as_mut() else {
                return;
            };
            let stmt = statement.cast::<ffi::sqlite3_stmt>();
            if kind == ffi::SQLITE_TRACE_STMT {
                if state.active.contains_key(&(stmt as usize)) {
                    return;
                }
                // SAFETY: this is the unexpanded SQL of SQLite's live VM.
                let ptr = unsafe { ffi::sqlite3_sql(stmt) };
                if ptr.is_null() {
                    return;
                }
                let sql = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_bytes();
                if sql.len() > SQL_BYTES {
                    state.oversized_sql += 1;
                    return;
                }
                let hash = blake3::hash(sql).to_hex().to_string();
                let key = format!("{}:{hash}", state.phase);
                if !state.statements.contains_key(&key) {
                    if state.statements.len() == STATEMENTS {
                        state.dropped_statements += 1;
                        return;
                    }
                    state.statements.insert(
                        key.clone(),
                        StatementWork {
                            phase: state.phase,
                            sql_hash: hash.clone(),
                            verb: String::from_utf8_lossy(sql)
                                .split_whitespace()
                                .next()
                                .unwrap_or("")
                                .to_uppercase(),
                            ..Default::default()
                        },
                    );
                    state.emit(serde_json::json!({"writer_sql_statement":{
                        "phase":state.phase,"hash":hash,"sql":String::from_utf8_lossy(sql),
                        "expanded":false}}));
                }
                if state.active.len() < ACTIVE {
                    // SAFETY: reset only this live VM's diagnostic counters at
                    // execution entry; a cached VM may predate trace attachment.
                    for counter in [
                        ffi::SQLITE_STMTSTATUS_VM_STEP,
                        ffi::SQLITE_STMTSTATUS_FULLSCAN_STEP,
                        ffi::SQLITE_STMTSTATUS_SORT,
                    ] {
                        unsafe { ffi::sqlite3_stmt_status(stmt, counter, 1) };
                    }
                    state.active.insert(stmt as usize, key);
                } else {
                    state.dropped_active += 1;
                }
                state.flush(false);
            } else if kind == ffi::SQLITE_TRACE_PROFILE {
                let Some(key) = state.active.remove(&(stmt as usize)) else {
                    return;
                };
                let Some(work) = state.statements.get_mut(&key) else {
                    return;
                };
                work.calls += 1;
                // SAFETY: SQLite supplies a live VM and unsigned nanosecond duration.
                work.lifetime_ns += unsafe { *detail.cast::<u64>() };
                for (kind, sum) in [
                    (ffi::SQLITE_STMTSTATUS_VM_STEP, &mut work.vm_steps),
                    (
                        ffi::SQLITE_STMTSTATUS_FULLSCAN_STEP,
                        &mut work.fullscan_steps,
                    ),
                    (ffi::SQLITE_STMTSTATUS_SORT, &mut work.sorts),
                ] {
                    // Reset only diagnostic counters, preserving the statement cache.
                    match u64::try_from(unsafe { ffi::sqlite3_stmt_status(stmt, kind, 1) }) {
                        Ok(value) => *sum += value,
                        Err(_) => work.invalid_counters += 1,
                    }
                }
                state.flush(false);
            }
        });
    }));
    if observed.is_err() {
        let _ = STATE.try_with(|state| {
            if let Ok(mut state) = state.try_borrow_mut()
                && let Some(state) = state.as_mut()
            {
                state.callback_errors += 1;
            }
        });
    }
    0
}

#[allow(unsafe_code)]
fn begin(conn: &Connection, name: &'static str) -> Option<Before> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = state.as_mut()?;
        let previous = std::mem::replace(&mut state.phase, name);
        if !state.phases.contains_key(name) {
            state.phases.insert(name, PhaseWork::default());
            state.emit(serde_json::json!({"writer_sql_phase_begin":name}));
        }
        Some(Before {
            wall: Instant::now(),
            cpu: cpu(),
            cache: unsafe { cache(conn.handle()) },
            io: io(),
            output_bytes: state.output_bytes,
            previous,
        })
    })
}

#[allow(unsafe_code)]
unsafe fn end(db: *mut ffi::sqlite3, name: &'static str, before: Before) {
    // SAFETY: the phase or owning commit keeps the parent connection live.
    let after_cache = unsafe { cache(db) };
    let after_io = io();
    let after_cpu = cpu();
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else {
            return;
        };
        state.phase = before.previous;
        let work = state.phases.entry(name).or_default();
        work.calls += 1;
        work.wall_ns += before.wall.elapsed().as_nanos() as u64;
        work.observer_output_bytes += state.output_bytes - before.output_bytes;
        let delta =
            |old: Option<u64>, new: Option<u64>| old.zip(new).and_then(|(a, b)| b.checked_sub(a));
        if let Some(value) = delta(before.cpu, after_cpu) {
            work.cpu_ns += value;
        } else {
            work.unavailable += 1;
        }
        for (n, sum) in [
            &mut work.cache_hits,
            &mut work.cache_misses,
            &mut work.cache_writes,
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(value) = delta(before.cache[n], after_cache[n]) {
                *sum += value;
            } else {
                work.unavailable += 1;
            }
        }
        if let Some((old, new)) = before.io.zip(after_io) {
            for (a, b, sum, correction) in [
                (old.rchar, new.rchar, &mut work.rchar, old.probe_read_bytes),
                (old.syscr, new.syscr, &mut work.syscr, 1),
                (old.read_bytes, new.read_bytes, &mut work.read_bytes, 0),
                (old.write_bytes, new.write_bytes, &mut work.write_bytes, 0),
            ] {
                if let Some(value) = b.checked_sub(a).and_then(|n| n.checked_sub(correction)) {
                    *sum += value;
                } else {
                    work.unavailable += 1;
                }
            }
        } else {
            work.unavailable += 1;
        }
        state.flush(false);
    });
}

pub(crate) struct Phase<'a> {
    conn: &'a Connection,
    name: &'static str,
    before: Option<Before>,
}
pub(crate) fn phase<'a>(conn: &'a Connection, name: &'static str) -> Phase<'a> {
    Phase {
        conn,
        name,
        before: begin(conn, name),
    }
}
impl Drop for Phase<'_> {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if let Some(before) = self.before.take() {
            // SAFETY: this phase borrows the connection until after this call.
            unsafe { end(self.conn.handle(), self.name, before) };
        }
    }
}

#[allow(unsafe_code)]
pub(crate) fn commit(tx: Transaction<'_>, name: &'static str) -> Result<()> {
    let before = begin(&tx, name);
    // SAFETY: Transaction borrows its parent connection; consuming only the
    // transaction does not destroy that parent during this function call.
    let db = unsafe { tx.handle() };
    let result = tx.commit().map_err(StoreError::from);
    if let Some(before) = before {
        unsafe { end(db, name, before) };
    }
    result
}

#[cfg(test)]
mod tests;
