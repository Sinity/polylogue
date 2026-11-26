use super::*;
use polylogue_store::log::{Priority, recent};
use std::os::unix::fs::PermissionsExt;

/// The log records naming `path`, with their priorities.
fn records_naming(path: &Path) -> Vec<Priority> {
    let path = path.display().to_string();
    recent(usize::MAX)
        .into_iter()
        .filter(|record| record.text.contains(&path))
        .map(|record| record.priority)
        .collect()
}

/// A tracked file that cannot be read is one warning naming it, however
/// many passes fail to read it, and one record more once it is read
/// (docs/operations.md, the log).
#[test]
fn an_unreadable_file_is_logged_once_and_its_recovery_once() {
    let home = Home::new();
    home.append("other.jsonl", &claude_user("o1", "other", "readable"));
    home.append("locked.jsonl", &claude_user("l1", "locked", "kept away"));
    let locked = home.file("locked.jsonl");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o200)).unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    let first = engine.run_pass().unwrap();
    assert!(
        first
            .errors
            .iter()
            .any(|failure| failure.subject == locked.display().to_string()),
        "{:?}",
        first.errors
    );
    for n in 2..5 {
        home.append(
            "other.jsonl",
            &claude_user(&format!("o{n}"), "other", "more"),
        );
        engine.run_pass().unwrap();
    }
    assert_eq!(records_naming(&locked), [Priority::Warning]);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o600)).unwrap();
    let read = engine.run_pass().unwrap();
    assert!(read.errors.is_empty(), "{:?}", read.errors);
    assert_eq!(records_naming(&locked), [Priority::Warning, Priority::Info]);
    assert_eq!(home.texts("locked"), ["kept away"]);
}
