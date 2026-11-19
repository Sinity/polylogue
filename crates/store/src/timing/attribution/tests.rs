use super::{Current, Part, Sample};
use std::time::{Duration, Instant};

#[test]
fn nested_parts_charge_each_interval_once_and_restore_the_parent() {
    let start = Instant::now();
    let at = |wall, cpu| Sample {
        wall: start + Duration::from_nanos(wall),
        cpu: Some(cpu),
    };
    let mut current = Current::default();
    assert!(current.switch(Some(Part::WriterOther), at(0, 0)).is_none());
    assert_eq!(
        current.switch(Some(Part::RowOther), at(20, 10)),
        Some((Part::WriterOther, 20, Some(10)))
    );
    assert_eq!(
        current.switch(Some(Part::MessageClaim), at(50, 30)),
        Some((Part::RowOther, 30, Some(20)))
    );
    assert_eq!(
        current.switch(Some(Part::RowOther), at(90, 45)),
        Some((Part::MessageClaim, 40, Some(15)))
    );
    assert_eq!(
        current.switch(Some(Part::WriterOther), at(110, 60)),
        Some((Part::RowOther, 20, Some(15)))
    );
    assert_eq!(
        current.switch(None, at(130, 70)),
        Some((Part::WriterOther, 20, Some(10)))
    );
    // Exclusive totals: writer 40/20, row 50/35, claim 40/15.
    // The parent does not include its child's 40 wall / 15 CPU nanoseconds.
    assert!(current.switch(None, at(150, 80)).is_none());
}

#[test]
fn an_unavailable_cpu_clock_remains_unknown() {
    let start = Instant::now();
    let mut current = Current::default();
    current.switch(
        Some(Part::RowOther),
        Sample {
            wall: start,
            cpu: None,
        },
    );
    assert_eq!(
        current.switch(
            None,
            Sample {
                wall: start + Duration::from_nanos(17),
                cpu: Some(42),
            }
        ),
        Some((Part::RowOther, 17, None))
    );
}
