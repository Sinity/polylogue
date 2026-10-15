//! Storage errors fuse the existing per-record route; EOF cannot turn them into success.
#![allow(clippy::unwrap_used)]
use polylogue_sources::frame::{
    FailurePhase, IgnoredKind, JsonlDriver, Record, RecordOutcome, RecordParser, Row, StepBudget,
    drive_json_array, drive_rows,
};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Copy, Debug)]
enum Never {}
impl IgnoredKind for Never {
    fn name(self) -> &'static str {
        match self {}
    }
    fn reason(self) -> &'static str {
        match self {}
    }
}
struct StateReader {
    calls: Arc<AtomicUsize>,
    finishes: Arc<AtomicUsize>,
    fail_record: Option<usize>,
    fail_finish: bool,
    touched: Vec<u64>,
}
impl RecordParser for StateReader {
    type Ignored = Never;
    type Output = Vec<u64>;
    type Error = &'static str;
    fn record(&mut self, rec: &Record<'_>) -> Result<RecordOutcome<Never>, Self::Error> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        // Deliberately mutate the ephemeral attempt before the injected lookup
        // error. Reusing this object after failure would duplicate the key.
        self.touched.push(rec.value["key"].as_u64().unwrap());
        if self.fail_record == Some(n) {
            return Err("state read failed");
        }
        Ok(RecordOutcome::Used)
    }
    fn finish(self) -> Result<Vec<u64>, Self::Error> {
        self.finishes.fetch_add(1, Ordering::SeqCst);
        if self.fail_finish {
            Err("state finalization failed")
        } else {
            Ok(self.touched)
        }
    }
}
fn parser(
    fail_record: Option<usize>,
    fail_finish: bool,
) -> (StateReader, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let finishes = Arc::new(AtomicUsize::new(0));
    (
        StateReader {
            calls: Arc::clone(&calls),
            finishes: Arc::clone(&finishes),
            fail_record,
            fail_finish,
            touched: Vec::new(),
        },
        calls,
        finishes,
    )
}
fn budget() -> StepBudget {
    StepBudget {
        bytes: 1 << 20,
        records: 100,
        record_bytes: 1 << 20,
    }
}

#[test]
fn mutate_then_fail_cannot_resume_or_finalize_and_restart_replays_uncommitted_attempt() {
    let input = b"{\"key\":1}\n{\"key\":2}\n{\"key\":3}\n";
    let (reader, calls, finishes) = parser(Some(2), false);
    let mut driver = JsonlDriver::new(reader, 40);
    assert!(driver.advance(input, budget()).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(driver.failure().unwrap().offset, 50);
    assert_eq!(driver.failure().unwrap().error, "state read failed");
    assert_eq!(driver.parsed_upto(), 50);
    assert_eq!(driver.counts().framed, 1);
    assert_eq!(
        driver.counts().unrecognized.len(),
        0,
        "lookup error is not an unknown record"
    );
    assert!(driver.counts().balanced());
    assert!(driver.advance(&input[10..], budget()).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 2, "failed attempt is fused");
    let failed = driver.finish();
    assert_eq!(failed.output.unwrap_err().phase, FailurePhase::Record);
    assert_eq!(
        finishes.load(Ordering::SeqCst),
        0,
        "failed state must never reach provider finalization"
    );

    // The external committed checkpoint was40, not the failed attempt's50.
    // Reconstruct from that pinned state and replay the complete uncommitted run.
    let (reader, calls, finishes) = parser(None, false);
    let mut restarted = JsonlDriver::new(reader, 40);
    restarted.advance(input, budget()).unwrap();
    let recovered = restarted.finish();
    assert_eq!(recovered.output.unwrap(), [1, 2, 3]);
    assert_eq!(recovered.parsed_upto, 40 + input.len() as u64);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(finishes.load(Ordering::SeqCst), 1);
}

#[test]
fn byte_eof_does_not_mask_finalization_failure() {
    let (reader, _, finishes) = parser(None, true);
    let mut driver = JsonlDriver::new(reader, 17);
    let input = b"{\"key\":1}\n";
    driver.advance(input, budget()).unwrap();
    assert_eq!(driver.parsed_upto(), 27);
    let failed = driver.finish();
    let error = failed.output.unwrap_err();
    assert_eq!(error.phase, FailurePhase::Finalization);
    assert_eq!(error.offset, 27);
    assert_eq!(error.error, "state finalization failed");
    assert_eq!(finishes.load(Ordering::SeqCst), 1);
    assert_eq!(failed.counts.used, 1);
}

#[test]
fn array_and_row_routes_stop_on_the_same_provider_error() {
    let (reader, calls, finishes) = parser(Some(2), false);
    let failed = drive_json_array(reader, b"[{\"key\":1},{\"key\":2},{\"key\":3}]");
    assert_eq!(failed.output.unwrap_err().phase, FailurePhase::Record);
    assert_eq!(failed.counts.framed, 1);
    assert!(failed.counts.balanced());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(finishes.load(Ordering::SeqCst), 0);

    let (reader, calls, finishes) = parser(Some(2), false);
    let rows = (1..=3).map(|n| Row {
        table: "synthetic",
        rowid: n,
        value: json!({"key":n}),
    });
    let failed = drive_rows(reader, rows);
    assert_eq!(failed.output.unwrap_err().offset, 2);
    assert_eq!(failed.parsed_upto, 1);
    assert_eq!(failed.counts.framed, 1);
    assert!(failed.counts.balanced());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(finishes.load(Ordering::SeqCst), 0);
}
