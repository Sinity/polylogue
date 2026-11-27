//! What each long-lived thread is doing now and which SQL statements it is
//! running, held in memory so `status` can show it without a reader or a
//! lock on anything busy.
//!
//! Each thread has one record, made on its first use and registered in a
//! process-wide list; `status` reads every record ([`threads`]). A thread
//! says what it is doing with [`doing`] (or [`doing_on`] with a subject such
//! as an operation or a build step); the returned guard takes the entry away
//! when dropped. Entries nest: a writer building the index shows the build
//! and the step inside it. [`progressed`] marks progress inside long work; a
//! new entry counts as progress too.
//!
//! The statements come from SQLite's own trace hook, installed on the
//! connections the store opens ([`trace_statements`]): a statement's start
//! adds it to the record of the thread running it and its end removes it.
//! A statement that ran longer than [`SLOW_STATEMENT`] is logged with its
//! counters. The hook is a plain function, so it finds the record through a
//! thread-local; each connection runs on one thread at a time.
//!
//! It lives in the store beside the log, for the same reason: the store is
//! the lowest crate whose threads run in the daemon.
use polylogue_model::LockExt as _;
use rusqlite::trace::{TraceEvent, TraceEventCodes};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

/// A statement running at least this long is logged when it ends.
pub const SLOW_STATEMENT: Duration = Duration::from_secs(1);

/// Completed slow statements, grouped by connection and SQL text.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SlowStatement {
    pub connection: String,
    pub sql: String,
    pub count: u64,
    pub at: polylogue_model::Timestamp,
    pub seconds: f64,
    pub longest_seconds: f64,
    pub thread: String,
    pub operation: Option<String>,
}

static SLOW: Mutex<std::collections::VecDeque<SlowStatement>> =
    Mutex::new(std::collections::VecDeque::new());

/// The latest hundred distinct slow statements, oldest first.
pub fn slow_statements() -> Vec<SlowStatement> {
    SLOW.locked().iter().cloned().collect()
}

fn finished_slow_statement(connection: &str, sql: &str, measured: Duration) {
    let operation = RECORD
        .try_with(|record| {
            record
                .state
                .locked()
                .doing
                .last()
                .map(|work| match &work.subject {
                    Some(subject) => format!("{}: {subject}", work.what),
                    None => work.what.to_owned(),
                })
        })
        .ok()
        .flatten();
    // Like the journal log, completed statements carry the time they ended.
    let Some(at) = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|since| i64::try_from(since.as_millis()).ok())
        .and_then(polylogue_model::Timestamp::from_unix_ms)
    else {
        return;
    };
    let mut statements = SLOW.locked();
    let previous = statements
        .iter()
        .position(|statement| statement.connection == connection && statement.sql == sql)
        .and_then(|at| statements.remove(at));
    let seconds = measured.as_secs_f64();
    let count = previous.as_ref().map_or(1, |statement| statement.count + 1);
    let longest_seconds = previous
        .as_ref()
        .map_or(seconds, |statement| statement.longest_seconds.max(seconds));
    if statements.len() == 100 {
        statements.pop_front();
    }
    statements.push_back(SlowStatement {
        connection: connection.to_owned(),
        sql: sql.to_owned(),
        count,
        at,
        seconds,
        longest_seconds,
        thread: std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_owned(),
        operation,
    });
}

/// One thread's activity as `status` shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Thread {
    /// The thread's name.
    pub name: String,
    /// What it is doing, outermost first; empty while it is idle.
    pub doing: Vec<Work>,
    /// Seconds since it last made progress; none while it is idle.
    pub seconds_since_progress: Option<f64>,
    /// The SQL statements it is running, longest first.
    pub statements: Vec<Statement>,
    /// State of commit-following work, including failures while idle.
    pub worker: Option<crate::follow::Worker>,
}

/// One thing a thread is doing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Work {
    pub what: String,
    /// What it is done to or for: an operation, a build step, an input.
    pub subject: Option<String>,
    /// Seconds since it began.
    pub seconds: f64,
}

/// A SQL statement a thread is running.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Statement {
    /// The connection it runs on: `writer`, `archive writer` or `reader`.
    pub connection: String,
    /// Its SQL text, whitespace runs as single spaces.
    pub sql: String,
    /// Seconds since it started.
    pub seconds: f64,
}

/// The connections whose statements are traced.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Connection {
    /// A write connection to `index.db` or for setup.
    Writer,
    /// The writer's connection to `archive.db`.
    ArchiveWriter,
    /// A read-only connection.
    Reader,
}

impl Connection {
    const NAMES: [&'static str; 3] = ["writer", "archive writer", "reader"];
}

struct Record {
    name: String,
    state: Mutex<State>,
}

struct State {
    worker: Option<crate::follow::Worker>,
    doing: Vec<Entry>,
    next_entry: u64,
    last_progress: Instant,
    statements: Vec<Running>,
    /// Text buffers of ended statements, reused so a statement's start
    /// copies its SQL without allocating.
    spare: Vec<String>,
}

struct Entry {
    id: u64,
    what: &'static str,
    subject: Option<Cow<'static, str>>,
    since: Instant,
}

struct Running {
    /// The address of the statement's SQL text, which SQLite keeps for the
    /// statement's life: it tells two statements with the same text apart.
    key: usize,
    connection: &'static str,
    sql: String,
    since: Instant,
}

static THREADS: Mutex<Vec<Weak<Record>>> = Mutex::new(Vec::new());

thread_local! {
    static RECORD: Arc<Record> = register();
}

fn register() -> Arc<Record> {
    let current = std::thread::current();
    let name = current
        .name()
        .map_or_else(|| format!("{:?}", current.id()), str::to_owned);
    let record = Arc::new(Record {
        name,
        state: Mutex::new(State {
            worker: None,
            doing: Vec::new(),
            next_entry: 0,
            last_progress: Instant::now(),
            statements: Vec::new(),
            spare: Vec::new(),
        }),
    });
    let mut threads = THREADS.locked();
    threads.retain(|thread| thread.strong_count() > 0);
    threads.push(Arc::downgrade(&record));
    record
}

/// Something the current thread is doing, until dropped.
#[must_use = "the entry ends when the guard is dropped"]
pub struct Doing {
    record: Arc<Record>,
    id: u64,
}

impl Drop for Doing {
    fn drop(&mut self) {
        self.record
            .state
            .locked()
            .doing
            .retain(|entry| entry.id != self.id);
    }
}

/// Records that the current thread is doing `what`.
pub fn doing(what: &'static str) -> Doing {
    begin(what, None, Instant::now())
}

/// Records that the current thread is doing `what` to or for `subject`.
pub fn doing_on(what: &'static str, subject: impl Into<Cow<'static, str>>) -> Doing {
    begin(what, Some(subject.into()), Instant::now())
}

/// Records that the current thread is doing `what`, which began at `since`
/// (a step that continues across several of the thread's turns).
pub fn doing_since(what: &'static str, since: Instant) -> Doing {
    begin(what, None, since)
}

fn begin(what: &'static str, subject: Option<Cow<'static, str>>, since: Instant) -> Doing {
    let record = RECORD.with(Arc::clone);
    let id = {
        let mut state = record.state.locked();
        let id = state.next_entry;
        state.next_entry += 1;
        state.last_progress = Instant::now();
        state.doing.push(Entry {
            id,
            what,
            subject,
            since,
        });
        id
    };
    Doing { record, id }
}

/// Records that the current thread made progress in what it is doing.
pub fn progressed() {
    let _ = RECORD.try_with(|record| record.state.locked().last_progress = Instant::now());
}

/// Every other thread that is doing something or running a statement for
/// at least `statements_over`, by name, with those statements. The calling
/// thread is reading this, so it is left out.
pub fn threads(statements_over: Duration) -> Vec<Thread> {
    let caller = RECORD.try_with(Arc::as_ptr).ok();
    let records: Vec<Arc<Record>> = THREADS
        .locked()
        .iter()
        .filter_map(Weak::upgrade)
        .filter(|record| Some(Arc::as_ptr(record)) != caller)
        .collect();
    let now = Instant::now();
    let seconds = |since: Instant| now.saturating_duration_since(since).as_secs_f64();
    let mut threads: Vec<Thread> = records
        .iter()
        .filter_map(|record| {
            let state = record.state.locked();
            let mut statements: Vec<(Instant, Statement)> = state
                .statements
                .iter()
                .filter(|running| now.saturating_duration_since(running.since) >= statements_over)
                .map(|running| {
                    (
                        running.since,
                        Statement {
                            connection: running.connection.to_owned(),
                            sql: running.sql.clone(),
                            seconds: seconds(running.since),
                        },
                    )
                })
                .collect();
            if state.doing.is_empty()
                && statements.is_empty()
                && state
                    .worker
                    .as_ref()
                    .is_none_or(|worker| matches!(worker, crate::follow::Worker::Running))
            {
                return None;
            }
            let doing = state
                .doing
                .iter()
                .map(|entry| Work {
                    what: entry.what.to_owned(),
                    subject: entry.subject.as_ref().map(|subject| subject.to_string()),
                    seconds: seconds(entry.since),
                })
                .collect();
            let seconds_since_progress =
                (!state.doing.is_empty()).then(|| seconds(state.last_progress));
            let worker = state.worker.clone();
            drop(state);
            statements.sort_by_key(|(since, _)| *since);
            for (_, statement) in &mut statements {
                statement.sql = single_spaced(&statement.sql);
            }
            Some(Thread {
                name: record.name.clone(),
                worker,
                doing,
                seconds_since_progress,
                statements: statements
                    .into_iter()
                    .map(|(_, statement)| statement)
                    .collect(),
            })
        })
        .collect();
    threads.sort_by(|a, b| a.name.cmp(&b.name));
    threads
}

/// Seconds the threads named `name` have spent on what they are doing now,
/// each from the start of its outermost entry.
pub(crate) fn busy_seconds(name: &str) -> f64 {
    let records: Vec<Arc<Record>> = THREADS
        .locked()
        .iter()
        .filter_map(Weak::upgrade)
        .filter(|record| record.name == name)
        .collect();
    records
        .iter()
        .filter_map(|record| {
            let state = record.state.locked();
            state
                .doing
                .first()
                .map(|entry| entry.since.elapsed().as_secs_f64())
        })
        .sum()
}

fn single_spaced(sql: &str) -> String {
    sql.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Traces the statements `conn` runs into the running thread's record.
pub(crate) fn trace_statements(conn: &rusqlite::Connection, connection: Connection) {
    let hook: fn(TraceEvent<'_>) = match connection {
        Connection::Writer => traced::<0>,
        Connection::ArchiveWriter => traced::<1>,
        Connection::Reader => traced::<2>,
    };
    conn.trace_v2(
        TraceEventCodes::SQLITE_TRACE_STMT | TraceEventCodes::SQLITE_TRACE_PROFILE,
        Some(hook),
    );
}

fn traced<const CONNECTION: usize>(event: TraceEvent<'_>) {
    match event {
        // The event's text is a copy for a nested statement and a marker
        // for a trigger's subprogram; the statement's own text, which
        // SQLite keeps for its life, names it. Rust's SQL is UTF-8, so the
        // text is borrowed, not converted.
        TraceEvent::Stmt(statement, _) => {
            let sql = statement.sql();
            if matches!(sql, Cow::Owned(_)) || sql.is_empty() {
                return;
            }
            let key = sql.as_ptr() as usize;
            let _ = RECORD.try_with(|record| {
                let mut state = record.state.locked();
                // A trigger's subprogram reports its statement again.
                if state.statements.iter().any(|running| running.key == key) {
                    return;
                }
                let mut text = state.spare.pop().unwrap_or_default();
                text.clear();
                text.push_str(&sql);
                state.statements.push(Running {
                    key,
                    connection: Connection::NAMES[CONNECTION],
                    sql: text,
                    since: Instant::now(),
                });
            });
        }
        TraceEvent::Profile(statement, measured) => {
            let sql = statement.sql();
            let key = sql.as_ptr() as usize;
            let _ = RECORD.try_with(|record| {
                let mut state = record.state.locked();
                let at = state
                    .statements
                    .iter()
                    .position(|running| running.key == key)?;
                let running = state.statements.swap_remove(at);
                state.spare.push(running.sql);
                Some(())
            });
            // SQLite's own measure, to the millisecond, saves reading the
            // clock again.
            if measured >= SLOW_STATEMENT {
                use rusqlite::StatementStatus as Counter;
                finished_slow_statement(
                    Connection::NAMES[CONNECTION],
                    &single_spaced(&sql),
                    measured,
                );
                crate::log!(
                    progress,
                    "slow statement on a {} connection: {:.1} s, {} full-scan steps, {} sorts, \
                     {} automatic-index rows, {} virtual machine steps: {}",
                    Connection::NAMES[CONNECTION],
                    measured.as_secs_f64(),
                    statement.get_status(Counter::FullscanStep),
                    statement.get_status(Counter::Sort),
                    statement.get_status(Counter::AutoIndex),
                    statement.get_status(Counter::VmStep),
                    single_spaced(&sql),
                );
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;

/// Sets the commit-following worker state on the current thread.
pub(crate) fn worker(worker: crate::follow::Worker) {
    RECORD.with(|record| record.state.locked().worker = Some(worker));
}
