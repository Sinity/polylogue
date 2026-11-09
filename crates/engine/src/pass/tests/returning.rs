//! A file whose identity changed continues its chain only when it is shown
//! to begin with every stored byte (docs/archive.md, stored inputs).
use super::*;
use std::os::unix::fs::PermissionsExt;

/// A stored file that disappeared, with the engine's state after the pass
/// that saw it gone, then written back as `bytes`.
fn gone_and_back(bytes: &str) -> (ClaudeHome, Engine, polylogue_store::ArchiveLock, PathBuf) {
    let home = ClaudeHome::new();
    home.append("back.jsonl", &claude_user("u1", "back", "stored body"));
    let (mut engine, lock) = crate::import::tests::engine(&home, 64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let path = home.file("back.jsonl");
    std::fs::remove_file(&path).unwrap();
    assert!(engine.run_pass().unwrap().errors.is_empty());
    std::fs::write(&path, bytes).unwrap();
    (home, engine, lock, path)
}

fn facts(path: &Path) -> FileFacts {
    FileFacts::of(&std::fs::metadata(path).unwrap())
}

/// An unreadable file is not shown to hold the stored bytes: it plans a new
/// chain, and planning goes on for the other files instead of failing.
#[test]
fn an_unreadable_returning_file_plans_a_new_chain() {
    let stored = claude_user("u1", "back", "stored body");
    let (_home, mut engine, _lock, path) = gone_and_back(&stored);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::File::open(&path).is_ok() {
        // Permissions do not apply to this user (root); nothing to show.
        return;
    }
    let plan = engine.plan(&path, facts(&path)).unwrap().unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    // The file was gone, so the new chain is planned as a first reading.
    assert_eq!(plan.change, Change::First);
}

/// The index can trail writes the engine has queued; a continuation from
/// that older length would start before them, so the file starts a new
/// chain instead of an append the store refuses.
#[test]
fn a_returning_file_does_not_continue_from_an_index_behind_the_engine() {
    let stored = claude_user("u1", "back", "stored body");
    let grown = stored.clone() + &claude_user("u2", "back", "grown body");
    let (_home, mut engine, _lock, path) = gone_and_back(&grown);
    engine.inputs.get_mut(&path).unwrap().stored = stored.len() as u64 + 1;
    let plan = engine.plan(&path, facts(&path)).unwrap().unwrap();
    // The file was gone, so the new chain is planned as a first reading.
    assert_eq!(plan.change, Change::First);
}

/// Rejoining the input's latest chain from another chain takes that
/// chain's parser position and admitted state, not the path's own.
#[test]
fn a_file_rejoining_another_chain_takes_that_chains_parser_position() {
    let stored = claude_user("u1", "back", "stored body");
    let grown = stored.clone() + &claude_user("u2", "back", "grown body");
    let (_home, mut engine, _lock, path) = gone_and_back(&grown);
    let st = engine.inputs.get_mut(&path).unwrap();
    st.chain = Some(99);
    st.parsed_upto = 1;
    st.parser_state = Some(serde_json::json!("a state of chain 99"));
    let plan = engine.plan(&path, facts(&path)).unwrap().unwrap();
    assert_eq!(
        plan.change,
        Change::Appended {
            from: stored.len() as u64
        }
    );
    let st = &engine.inputs[&path];
    assert_eq!(st.chain, Some(1));
    assert_ne!(
        plan.parser_state,
        Some(serde_json::json!("a state of chain 99"))
    );
    assert_eq!(plan.parse_pos, stored.len() as u64);
}
