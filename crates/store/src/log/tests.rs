use super::*;

fn record(n: usize, priority: Priority) -> Record {
    Record {
        at: Timestamp::from_unix_ms(1_700_000_000_000 + n as i64).unwrap(),
        priority,
        text: format!("record {n}"),
    }
}

#[test]
fn the_journal_line_names_the_priority_and_the_daemon() {
    assert_eq!(
        journal_line(Priority::Error, "writer: checkpoint: disk full"),
        "<3>polylogue: writer: checkpoint: disk full"
    );
    assert_eq!(
        journal_line(Priority::Warning, "scope index: busy"),
        "<4>polylogue: scope index: busy"
    );
    assert_eq!(
        journal_line(Priority::Info, "stopping on SIGTERM"),
        "<6>polylogue: stopping on SIGTERM"
    );
}

#[test]
fn the_recent_records_keep_the_last_two_hundred_in_order() {
    let mut recent = Recent::default();
    for n in 0..KEPT + 25 {
        let priority = if n % 2 == 0 {
            Priority::Info
        } else {
            Priority::Warning
        };
        recent.push(record(n, priority));
    }
    let kept = recent.last(usize::MAX);
    assert_eq!(kept.len(), KEPT);
    assert_eq!(kept.first(), Some(&record(25, Priority::Warning)));
    assert_eq!(kept.last(), Some(&record(KEPT + 24, Priority::Info)));
    assert_eq!(
        recent.last(3),
        [
            record(KEPT + 22, Priority::Info),
            record(KEPT + 23, Priority::Warning),
            record(KEPT + 24, Priority::Info),
        ]
    );
}

#[test]
fn a_logged_message_is_kept_with_its_priority() {
    crate::log!(warning, "synthetic warning {}", 7);
    crate::log!(error, "synthetic failure");
    let kept = recent(2);
    assert_eq!(
        kept.iter()
            .map(|record| (record.priority, record.text.as_str()))
            .collect::<Vec<_>>(),
        [
            (Priority::Warning, "synthetic warning 7"),
            (Priority::Error, "synthetic failure")
        ]
    );
}
