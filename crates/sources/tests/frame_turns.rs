//! Parsing in budgeted steps gives the same result as one whole parse.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use polylogue_sources::frame::{
    Counts, IgnoredKind, JsonlDriver, Record, RecordOutcome, RecordParser, StepBudget, StepState,
    drive_jsonl,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Copy, Debug)]
enum Ignored {
    Metadata,
}
impl IgnoredKind for Ignored {
    fn name(self) -> &'static str {
        "metadata"
    }
}
struct Recorder {
    rows: Vec<(u64, u64, String)>,
    finished: Arc<AtomicUsize>,
}
impl RecordParser for Recorder {
    type Error = std::convert::Infallible;
    type Ignored = Ignored;
    type Output = Vec<(u64, u64, String)>;
    fn record(
        &mut self,
        record: &Record<'_>,
    ) -> std::result::Result<RecordOutcome<Self::Ignored>, Self::Error> {
        Ok({
            self.rows.push((
                record.start,
                record.end,
                String::from_utf8(record.bytes.to_vec()).unwrap(),
            ));
            match record.value["kind"].as_str() {
                Some("ignored") => RecordOutcome::Ignored(Ignored::Metadata),
                Some("unknown") => RecordOutcome::Unrecognized("synthetic"),
                _ => RecordOutcome::Used,
            }
        })
    }
    fn finish(self) -> std::result::Result<Self::Output, Self::Error> {
        Ok({
            self.finished.fetch_add(1, Ordering::SeqCst);
            self.rows
        })
    }
}
fn recorder(finished: &Arc<AtomicUsize>) -> Recorder {
    Recorder {
        rows: Vec::new(),
        finished: Arc::clone(finished),
    }
}
fn budget(bytes: usize, records: u64, record_bytes: usize) -> StepBudget {
    StepBudget {
        bytes,
        records,
        record_bytes,
    }
}

#[test]
fn every_cut_preserves_offsets_accounting_and_one_finalization() {
    let input = b" \t\n{\"kind\":\"used\",\"text\":\"snow \xe9\x9b\xaa\"}\nnot json\n{\"kind\":\"ignored\"}\n{\"kind\":\"unknown\"}\n{\"partial\":";
    let base = 175;
    let finished = Arc::new(AtomicUsize::new(0));
    let whole = drive_jsonl(recorder(&finished), input, base);
    assert_eq!(
        whole.counts.framed, 4,
        "baseline whole-run route cannot yield after one record"
    );
    for window in 1..=input.len() {
        let done = Arc::new(AtomicUsize::new(0));
        let mut driver = JsonlDriver::new(recorder(&done), base);
        let mut pos = 0;
        while pos < input.len() {
            let end = input.len().min(pos + window);
            let step = driver
                .advance(&input[pos..end], budget(7, 1, input.len()))
                .unwrap();
            assert!(step.consumed > 0);
            assert!(step.consumed <= 7 && step.records <= 1);
            pos += step.consumed;
            assert_eq!(driver.scanned_upto(), base + pos as u64);
            assert_eq!(
                done.load(Ordering::SeqCst),
                0,
                "a window boundary must not finish provider state"
            );
            assert!(driver.counts().balanced());
        }
        assert!(
            driver.parsed_upto() < driver.scanned_upto(),
            "partial tail is scanned, not certified parsed"
        );
        let result = driver.finish();
        assert_eq!(done.load(Ordering::SeqCst), 1);
        assert_eq!(
            result.output.as_ref().unwrap(),
            whole.output.as_ref().unwrap()
        );
        assert_eq!(result.counts, whole.counts);
        assert_eq!(result.parsed_upto, whole.parsed_upto);
    }
}

#[test]
fn oversized_record_waits_for_capacity_and_is_never_truncated() {
    let done = Arc::new(AtomicUsize::new(0));
    let mut driver = JsonlDriver::new(recorder(&done), 19);
    let input = b"{\"text\":\"a record spanning several turns\"}\n";
    let first = driver.advance(&input[..10], budget(10, 1, 10)).unwrap();
    assert_eq!(first.consumed, 10);
    assert_eq!(driver.parsed_upto(), 19);
    let denied = driver
        .advance(&input[10..], budget(input.len(), 1, 10))
        .unwrap();
    assert_eq!(
        denied,
        polylogue_sources::frame::Step {
            consumed: 0,
            records: 0,
            state: StepState::NeedsCapacity {
                required: input.len() - 1
            }
        }
    );
    assert_eq!(driver.scanned_upto(), 29);
    assert_eq!(driver.counts(), &Counts::default());
    let resumed = driver
        .advance(&input[10..], budget(input.len(), 1, input.len()))
        .unwrap();
    assert_eq!(resumed.records, 1);
    assert_eq!(driver.parsed_upto(), 19 + input.len() as u64);
    assert!(driver.buffer_capacity() >= input.len() - 1);
    let output = driver.finish();
    assert_eq!(
        output.output.unwrap()[0],
        (
            19,
            19 + input.len() as u64 - 1,
            String::from_utf8(input[..input.len() - 1].to_vec()).unwrap()
        )
    );
}

#[test]
fn yielded_record_budget_does_not_scan_the_next_record() {
    let done = Arc::new(AtomicUsize::new(0));
    let mut driver = JsonlDriver::new(recorder(&done), 0);
    let input = b"{}\n{}\n";
    let first = driver.advance(input, budget(input.len(), 1, 2)).unwrap();
    assert_eq!(first.state, StepState::Yielded);
    assert_eq!(first.consumed, 3);
    assert_eq!(driver.scanned_upto(), 3);
    let second = driver
        .advance(&input[3..], budget(input.len(), 1, 2))
        .unwrap();
    assert_eq!(second.state, StepState::Exhausted);
    assert_eq!(driver.finish().counts.framed, 2);
}

#[test]
fn real_provider_interpretation_is_identical_across_framing_turns() {
    use polylogue_sources::codex::keyed::{KeyedRecords, MemoryReadView, Resume};
    fn records() -> KeyedRecords<MemoryReadView> {
        KeyedRecords::new(MemoryReadView::default(), Resume::default())
    }
    let input = include_bytes!("data/codex-stateful.jsonl");
    let whole = drive_jsonl(records(), input, 0);
    let mut driver = JsonlDriver::new(records(), 0);
    let mut pos = 0;
    while pos < input.len() {
        let step = driver
            .advance(&input[pos..], budget(41, 1, input.len()))
            .unwrap();
        pos += step.consumed;
    }
    let result = driver.finish();
    assert_eq!(
        serde_json::to_value((
            &result.output.as_ref().unwrap().resume,
            result
                .output
                .as_ref()
                .unwrap()
                .cells
                .iter()
                .collect::<Vec<_>>(),
            &result.output.as_ref().unwrap().events,
            &result.output.as_ref().unwrap().links
        ))
        .unwrap(),
        serde_json::to_value((
            &whole.output.as_ref().unwrap().resume,
            whole
                .output
                .as_ref()
                .unwrap()
                .cells
                .iter()
                .collect::<Vec<_>>(),
            &whole.output.as_ref().unwrap().events,
            &whole.output.as_ref().unwrap().links
        ))
        .unwrap()
    );
    assert_eq!(result.counts, whole.counts);
}
