//! Temporary, test-only discriminator. Nothing in this module ships.
use super::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const EXHAUSTED: &str = "diagnostic SQLite work budget exhausted";
const SLOW: &str = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000) SELECT count(*) AS n FROM n";

#[derive(Default)]
pub(crate) struct WorkBudget {
    active: AtomicBool,
    ceiling: AtomicU64,
    steps: AtomicU64,
    exhausted: AtomicBool,
    action: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl WorkBudget {
    pub(crate) fn before_checks(&self) {
        if self.active.load(Ordering::Relaxed) {
            let action = self.action.lock().unwrap().take();
            if let Some(action) = action {
                action();
            }
        }
    }
    pub(crate) fn tick(&self) -> Result<()> {
        if self.active.load(Ordering::Relaxed)
            && self.steps.fetch_add(1000, Ordering::Relaxed) + 1000
                >= self.ceiling.load(Ordering::Relaxed)
        {
            self.exhausted.store(true, Ordering::Relaxed);
            return Err(StoreError::Invalid(EXHAUSTED.into()));
        }
        Ok(())
    }
    fn arm(&self, steps: u64, action: Option<Box<dyn FnOnce() + Send>>) {
        self.ceiling.store(steps, Ordering::Relaxed);
        self.steps.store(0, Ordering::Relaxed);
        self.exhausted.store(false, Ordering::Relaxed);
        *self.action.lock().unwrap() = action;
        self.active.store(true, Ordering::Relaxed);
    }
    fn clear(&self) {
        self.active.store(false, Ordering::Relaxed);
        drop(self.action.lock().unwrap().take());
    }
}

struct Clear(Arc<WorkBudget>);
impl Drop for Clear {
    fn drop(&mut self) {
        self.0.clear();
    }
}

pub(crate) fn attempt<T>(
    reader: &Reader<'_>,
    steps: u64,
    action: Option<Box<dyn FnOnce() + Send>>,
    work: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    reader.check()?;
    reader.diagnostic_budget.arm(steps, action);
    let clear = Clear(reader.diagnostic_budget.clone());
    let result = work();
    drop(clear);
    // A real stop arriving together with the private interrupt always wins.
    reader.check()?;
    match result {
        Err(StoreError::Invalid(message))
            if message == EXHAUSTED
                && reader.diagnostic_budget.exhausted.load(Ordering::Relaxed) =>
        {
            Ok(None)
        }
        result => result.map(Some),
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ExactWork {
    pub(crate) steps: u64,
}
thread_local! { static EXACT: Cell<u64> = const { Cell::new(0) }; }

/// Called only by the existing cancellation progress owner at interval one.
pub(crate) fn instruction() {
    EXACT.with(|steps| steps.set(steps.get() + 1));
}

pub(crate) fn measured<T>(reader: &Reader<'_>, work: impl FnOnce() -> T) -> (T, ExactWork) {
    let _ = reader.conn().unwrap();
    let before = EXACT.with(Cell::get);
    let result = work();
    let steps = EXACT.with(Cell::get) - before;
    (result, ExactWork { steps })
}

#[test]
fn search_budget_diagnostic_owner_matches_the_existing_per_instruction_oracle() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = crate::fixture::Fixture::new(dir.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 2, 8 << 20).unwrap();
    let reader = pool.reader().unwrap();
    let calibration = pool.reader().unwrap();
    let old = Arc::new(AtomicU64::new(0));
    let counter = old.clone();
    // Independent sacrificial connection retains the unchanged old oracle.
    calibration
        .conn()
        .unwrap()
        .progress_handler(
            1,
            Some(move || {
                counter.fetch_add(1, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
    let sql = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<100) SELECT sum(x) AS n FROM n";
    reader.query(sql, &[]).unwrap();
    calibration.query(sql, &[]).unwrap();
    for _ in 0..3 {
        let (rows, exact) = measured(&reader, || reader.query(sql, &[]).unwrap());
        old.store(0, Ordering::Relaxed);
        assert_eq!(rows, calibration.query(sql, &[]).unwrap());
        assert_eq!(rows[0]["n"], 5050);
        assert_eq!(exact.steps, old.load(Ordering::Relaxed));
        assert!(exact.steps > 100);
        eprintln!("CALIBRATION steps={}", exact.steps);
    }
}

#[test]
fn search_budget_diagnostic_interrupt_preserves_the_same_snapshot_and_next_statement() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = crate::fixture::Fixture::new(dir.path()).unwrap();
    fixture
        .session(crate::fixture::SessionFacts {
            id: "test:original",
            title: Some("before"),
            ..Default::default()
        })
        .unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 2, 8 << 20).unwrap();
    let reader = pool.reader().unwrap();
    let sql = "SELECT title FROM session WHERE id='test:original'";
    assert_eq!(reader.query(sql, &[]).unwrap()[0]["title"], "before");
    let identity = reader.index_identity().unwrap();
    let position = reader.change_position().unwrap();
    let (exhausted, exact) = measured(&reader, || {
        attempt(&reader, 2000, None, || reader.query(SLOW, &[]))
    });
    assert!(exhausted.unwrap().is_none());
    assert!(
        exact.steps >= 1000 && exact.steps <= 4000,
        "intentional interrupt work {exact:?}"
    );
    assert!(!reader.conn().unwrap().is_autocommit());
    let write = crate::archive::open_write(&fixture.archive.index_db(), 1024).unwrap();
    write
        .execute(
            "UPDATE session SET title='after' WHERE id='test:original'",
            [],
        )
        .unwrap();
    assert_eq!(reader.query(sql, &[]).unwrap()[0]["title"], "before");
    assert_eq!(reader.index_identity().unwrap(), identity);
    assert_eq!(reader.change_position().unwrap(), position);
    assert!(
        attempt(&reader, 100_000, None, || reader
            .query("SELECT 7 AS n", &[]))
        .unwrap()
        .is_some()
    );
    let fresh = pool.reader().unwrap();
    assert_eq!(fresh.query(sql, &[]).unwrap()[0]["title"], "after");
    eprintln!("SNAPSHOT preserved; interrupted_steps={}", exact.steps);
}

#[test]
fn search_budget_diagnostic_real_cancellation_precedes_budget_exhaustion() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = crate::fixture::Fixture::new(dir.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let limit = limit::ReadLimit::without_deadline(Default::default());
    limit.scope(|| {
        let reader = pool.reader().unwrap();
        reader.query("SELECT 1", &[]).unwrap();
        let cancel = limit.cancellation.clone();
        let result = attempt(
            &reader,
            1000,
            Some(Box::new(move || cancel.cancel())),
            || reader.query(SLOW, &[]),
        );
        assert!(matches!(result, Err(StoreError::ReadCancelled)));
        assert!(!reader.diagnostic_budget.exhausted.load(Ordering::Relaxed));
        assert!(matches!(
            reader.query("SELECT 1", &[]),
            Err(StoreError::ReadCancelled)
        ));
    });
}

#[test]
fn search_budget_diagnostic_real_stop_precedes_budget_exhaustion() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = crate::fixture::Fixture::new(dir.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let reader = pool.reader().unwrap();
    reader.query("SELECT 1", &[]).unwrap();
    let stop = fixture.archive.stop().clone();
    let result = attempt(&reader, 1000, Some(Box::new(move || stop.cancel())), || {
        reader.query(SLOW, &[])
    });
    assert!(matches!(result, Err(StoreError::Stopped)));
    assert!(!reader.diagnostic_budget.exhausted.load(Ordering::Relaxed));
}

#[test]
fn search_budget_diagnostic_real_scope_reset_precedes_budget_exhaustion() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = crate::fixture::Fixture::new(dir.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    let reader = pool.reader().unwrap();
    reader.query("SELECT 1", &[]).unwrap();
    let state = fixture.archive.read_scope.clone();
    let result = attempt(
        &reader,
        1000,
        Some(Box::new(move || {
            drop(state.begin_reset().unwrap());
            state.finish_reset();
        })),
        || reader.query(SLOW, &[]),
    );
    assert!(matches!(result, Err(StoreError::ScopeChanged)));
    assert!(!reader.diagnostic_budget.exhausted.load(Ordering::Relaxed));
    drop(reader);
    assert!(pool.reader().unwrap().query("SELECT 1", &[]).is_ok());
}

#[test]
fn search_budget_diagnostic_real_deadline_precedes_budget_exhaustion() {
    use std::time::Duration;
    let dir = tempfile::tempdir().unwrap();
    let fixture = crate::fixture::Fixture::new(dir.path()).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 1, 8 << 20).unwrap();
    drop(pool.reader().unwrap());
    let limit = limit::ReadLimit::new(Duration::from_secs(1), Default::default());
    limit.scope(|| {
        let reader = pool.reader().unwrap();
        reader.query("SELECT 1", &[]).unwrap();
        let action_ran = Arc::new(AtomicBool::new(false));
        let ran = action_ran.clone();
        let result = attempt(
            &reader,
            1000,
            Some(Box::new(move || {
                ran.store(true, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(1100));
            })),
            || reader.query(SLOW, &[]),
        );
        assert!(action_ran.load(Ordering::Relaxed));
        assert!(matches!(result, Err(StoreError::ReadDeadlineExceeded)));
        assert!(!reader.diagnostic_budget.exhausted.load(Ordering::Relaxed));
    });
}
