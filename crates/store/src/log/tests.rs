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
        recent.push(record(n, Priority::Info));
    }
    let kept = recent.last(usize::MAX);
    assert_eq!(kept.len(), KEPT);
    assert_eq!(kept.first(), Some(&record(25, Priority::Info)));
    assert_eq!(kept.last(), Some(&record(KEPT + 24, Priority::Info)));
    assert_eq!(
        recent.last(3),
        [
            record(KEPT + 22, Priority::Info),
            record(KEPT + 23, Priority::Info),
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

#[test]
fn information_records_do_not_displace_warnings_or_errors() {
    let mut recent = Recent::default();
    recent.push(record(0, Priority::Warning));
    recent.push(record(1, Priority::Error));
    for n in 2..302 {
        recent.push(record(n, Priority::Info));
    }
    let kept = recent.last(KEPT);
    assert_eq!(kept.len(), KEPT);
    assert_eq!(
        &kept[..2],
        &[record(0, Priority::Warning), record(1, Priority::Error)]
    );
    assert_eq!(kept[2], record(104, Priority::Info));
    assert_eq!(kept[199], record(301, Priority::Info));
}
