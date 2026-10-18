use super::*;
fn walk(view: &MemoryReadView, legacy: bool, block_only: bool) -> (Vec<String>, usize, usize) {
    let meter = Meter {
        view,
        reads: Counter::new(0),
        fail: false,
    };
    let mut cursor = if block_only {
        Cursor::block(0, 0, Mode::Annotated)
    } else {
        Cursor::new(0, Mode::Annotated)
    };
    let mut units = Vec::new();
    let mut steps = 0;
    loop {
        let mut old = serde_json::to_value(&cursor).unwrap();
        steps += 1;
        let step = if legacy && old["phase"] == "AdvanceBlock" {
            if block_only {
                old["phase"] = serde_json::json!("Done");
            } else {
                old["block"] = serde_json::json!(old["block"].as_u64().unwrap() + 1);
                old["phase"] = serde_json::json!("Block");
            }
            cursor = serde_json::from_value(old).unwrap();
            Step::Continue
        } else {
            cursor = serde_json::from_slice(&serde_json::to_vec(&cursor).unwrap()).unwrap();
            let before = meter.reads.get();
            let step = cursor.advance(&meter).unwrap();
            assert!(meter.reads.get() - before <= 1);
            step
        };
        match step {
            Step::Done => break,
            Step::Unit(unit) => units.push(format!("{unit:?}")),
            Step::Absent => panic!("present"),
            Step::Continue => {}
        }
    }
    (units, steps, meter.reads.get())
}
#[test]
fn old_advance_block_cursors_resume_with_the_same_units_and_no_extra_reads() {
    let view = fixture(3);
    for block_only in [false, true] {
        let (units, steps, reads) = walk(&view, false, block_only);
        let (old, old_steps, old_reads) = walk(&view, true, block_only);
        assert_eq!(units, old);
        assert_eq!(reads, old_reads);
        assert_eq!(old_steps - steps, if block_only { 1 } else { 2 });
    }
}
#[test]
fn resumed_bookkeeping_does_not_bypass_next_read_failure_or_fused_cursor() {
    let view = fixture(0);
    let mut cursor = Cursor::new(0, Mode::Message);
    loop {
        if serde_json::to_value(&cursor).unwrap()["phase"] == "AdvanceBlock" {
            break;
        }
        cursor.advance(&view).unwrap();
    }
    let meter = Meter {
        view: &view,
        reads: Counter::new(0),
        fail: true,
    };
    assert!(
        cursor
            .advance(&meter)
            .unwrap_err()
            .contains("injected projection read")
    );
    assert_eq!(meter.reads.get(), 1);
    assert!(
        cursor
            .advance(&meter)
            .unwrap_err()
            .contains("already failed")
    );
    assert_eq!(meter.reads.get(), 1);
}
