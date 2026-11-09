use super::*;

fn facts(inode: u64, size: u64, mtime: i128) -> Option<FileFacts> {
    Some(FileFacts {
        device: 1,
        inode,
        size,
        mtime_ns: mtime,
        ctime_ns: mtime,
    })
}

#[test]
fn classifies_every_kind_of_change() {
    assert_eq!(classify(None, facts(1, 10, 1)), Change::First);
    assert_eq!(
        classify(facts(1, 10, 1), facts(1, 10, 1)),
        Change::Unchanged
    );
    assert_eq!(
        classify(facts(1, 10, 1), facts(1, 25, 2)),
        Change::Appended { from: 10 }
    );
    assert_eq!(classify(facts(1, 10, 1), facts(1, 5, 2)), Change::Rewritten);
    assert_eq!(
        classify(facts(1, 10, 1), facts(2, 30, 2)),
        Change::Rewritten
    );
    assert_eq!(classify(facts(1, 10, 1), None), Change::Disappeared);
    assert_eq!(classify(None, None), Change::Unchanged);
    // A remount renumbers the device; the same inode, size and times are
    // the same file. A changed change time with a new device is not.
    let remounted = facts(1, 10, 1).map(|f| FileFacts { device: 7, ..f });
    assert_eq!(classify(facts(1, 10, 1), remounted), Change::Unchanged);
    let touched = remounted.map(|f| FileFacts { ctime_ns: 2, ..f });
    assert_eq!(classify(facts(1, 10, 1), touched), Change::Rewritten);
}
