use super::*;

#[derive(Clone, Copy, Debug)]
enum Never {}
impl IgnoredKind for Never {
    fn name(self) -> &'static str {
        match self {}
    }
}
/// Counts records; can fail on every record or at finish.
struct Counter {
    n: usize,
    fail_record: bool,
    fail_finish: bool,
}
impl Counter {
    fn new(n: usize) -> Self {
        Self {
            n,
            fail_record: false,
            fail_finish: false,
        }
    }
}
impl RecordParser for Counter {
    type Error = &'static str;
    type Ignored = Never;
    type Output = usize;
    fn record(&mut self, _: &Record<'_>) -> Result<RecordOutcome<Never>, Self::Error> {
        self.n += 1;
        if self.fail_record {
            Err("record failed")
        } else {
            Ok(RecordOutcome::Used)
        }
    }
    fn finish(self) -> Result<usize, Self::Error> {
        if self.fail_finish {
            Err("finish failed")
        } else {
            Ok(self.n)
        }
    }
}
fn budget(bytes: usize) -> StepBudget {
    StepBudget {
        bytes,
        records: 100,
        record_bytes: 1 << 20,
    }
}

#[test]
fn whole_window_does_not_copy_a_large_unfinished_tail() {
    let mut input = b"{}\n".to_vec();
    input.extend(std::iter::repeat_n(b'x', 1 << 20));
    let driver = whole_jsonl_driver(Counter::new(0), &input, 10);
    assert_eq!(
        driver.buffer_capacity(),
        0,
        "the unfinished tail is not copied"
    );
    let result = driver.finish();
    assert_eq!(result.output.unwrap(), 1);
    assert_eq!(result.parsed_upto, 13);
}

#[test]
fn incremental_growth_does_not_hide_geometric_spare_capacity() {
    let mut driver = JsonlDriver::new(Counter::new(0), 0);
    for admitted in 10..=30 {
        let bytes = if admitted == 10 {
            &b"1234567890"[..]
        } else {
            &b"x"[..]
        };
        let step = driver
            .advance(
                bytes,
                StepBudget {
                    bytes: bytes.len(),
                    records: 1,
                    record_bytes: admitted,
                },
            )
            .unwrap();
        assert_eq!(step.state, StepState::Exhausted);
        assert!(
            driver.buffer_capacity() <= admitted,
            "capacity {} over allowance {admitted}",
            driver.buffer_capacity()
        );
    }
    let step = driver
        .advance(
            b"\n",
            StepBudget {
                bytes: 1,
                records: 1,
                record_bytes: 1,
            },
        )
        .unwrap();
    assert_eq!(
        step.state,
        StepState::NeedsCapacity {
            required: driver.buffer_capacity()
        }
    );
    assert_eq!(
        step.consumed, 0,
        "the buffer still counts against a smaller allowance"
    );
}

#[test]
fn committed_checkpoint_moves_partial_buffer_and_starts_fresh_accounting() {
    let mut driver = JsonlDriver::new(Counter::new(0), 100);
    driver.advance(b"{}\n{\"text\":\"", budget(100)).unwrap();
    let address = driver.partial.as_ptr();
    let pending = match driver.checkpoint() {
        Ok(p) => p,
        Err(_) => panic!("valid step"),
    };
    assert_eq!(pending.parsed_upto(), 103);
    assert!(pending.scanned_upto() > pending.parsed_upto());
    assert_eq!(pending.counts().used, 1);
    let (committed, continuation) = pending
        .commit_with(|output, counts, parsed, scanned| {
            assert_eq!(counts.used, 1);
            assert_eq!(parsed, 103);
            assert!(scanned > parsed);
            Ok::<_, ()>(output)
        })
        .unwrap();
    assert_eq!(
        continuation.partial.as_ptr(),
        address,
        "ownership transfer must not copy a partial line"
    );
    let mut driver = JsonlDriver::resume(Counter::new(committed), continuation);
    assert_eq!(driver.partial.as_ptr(), address);
    assert_eq!(
        driver.counts().framed,
        0,
        "committed prefix is not counted twice"
    );
    driver.advance(b"x\"}\n", budget(100)).unwrap();
    let ready = match driver.checkpoint() {
        Ok(p) => p,
        Err(_) => panic!("valid next step"),
    };
    let (committed, continuation) = ready
        .commit_with(|output, counts, _, _| {
            assert_eq!(counts.framed, 1);
            Ok::<_, ()>(output)
        })
        .unwrap();
    assert_eq!(committed, 2);
    assert_eq!(continuation.parsed_upto(), continuation.scanned_upto());
}
#[test]
fn failed_writer_discards_continuation_and_replays_only_committed_state() {
    let mut driver = JsonlDriver::new(Counter::new(7), 0);
    driver.advance(b"{}\n{", budget(100)).unwrap();
    let ready = match driver.checkpoint() {
        Ok(p) => p,
        Err(_) => panic!("valid step"),
    };
    let result = ready.commit_with(|output, _, _, _| {
        assert_eq!(output, 8);
        Err::<(), _>("CAS rejected")
    });
    assert!(
        matches!(result, Err("CAS rejected")),
        "failed writer must not release a continuation"
    );
    let mut replay = JsonlDriver::new(Counter::new(7), 0);
    replay.advance(b"{}\n{}\n", budget(100)).unwrap();
    assert_eq!(
        replay.finish().output.unwrap(),
        9,
        "discarded uncommitted effects are not replayed twice"
    );
}
#[test]
fn callback_and_finalization_failure_cannot_produce_checkpoint_continuations() {
    for fail_record in [false, true] {
        let mut driver = JsonlDriver::new(
            Counter {
                n: 0,
                fail_record,
                fail_finish: !fail_record,
            },
            50,
        );
        let _ = driver.advance(b"{}\n{", budget(100));
        match driver.checkpoint() {
            Ok(_) => panic!("failed attempt must not expose resumable state"),
            Err(framed) => {
                let error = framed.output.unwrap_err();
                assert_eq!(
                    error.phase,
                    if fail_record {
                        FailurePhase::Record
                    } else {
                        FailurePhase::Finalization
                    }
                );
                assert_eq!(error.offset, if fail_record { 50 } else { 53 });
            }
        }
    }
}
