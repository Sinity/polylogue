use super::*;
use polylogue_testkit::{ClaudeHome, claude_user, delete_index};

fn basis(w: &ClaudeHome) -> Vec<polylogue_model::Json> {
    let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
    pool.reader().unwrap().query("SELECT m.id,r.id AS input,s.chain,s.source_start,s.source_end,s.parser_version,hex(s.version_hash) AS version FROM message_source s JOIN message m ON m.pk=s.message_pk JOIN input_ref r ON r.pk=s.input_pk ORDER BY m.id,r.id,s.chain,s.source_start",&[]).unwrap()
}

#[test]
fn immutable_imports_sharing_a_path_rebuild_every_source_and_never_become_watchers() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("same.jsonl");
    let (mut e, lock) = engine(&w, 64);
    for (id, body) in [
        ("one", "first retained import"),
        ("two", "second retained import"),
    ] {
        std::fs::write(&path, claude_user(id, "s", body)).unwrap();
        assert!(
            e.import(std::slice::from_ref(&path))
                .unwrap()
                .pass
                .errors
                .is_empty()
        );
    }
    let before = basis(&w);
    assert_eq!(before.len(), 2);
    assert_eq!(w.texts("s").len(), 2);
    drop(e);
    drop(lock);
    std::fs::remove_file(&path).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 64);
    let first = e.converge().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    assert_eq!(basis(&w), before);
    assert_eq!(w.texts("s").len(), 2);
    assert!(e.inputs.is_empty());
    let quiet = e.converge().unwrap();
    assert!(quiet.errors.is_empty());
    assert_eq!(quiet.bytes_read, 0);
    assert_eq!(quiet.inputs_changed, 0);
    let pool = ReadPool::open(&e.archive).unwrap();
    let r = pool.reader().unwrap();
    assert_eq!(r.retained_imports().unwrap().len(), 2);
    assert!(
        r.known_watched_inputs(&e.config.machine)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        r.query(
            "SELECT count(*) AS n FROM a.observation WHERE class='deleted'",
            &[]
        )
        .unwrap()[0]["n"],
        0
    );
}

#[test]
fn an_import_observation_does_not_replace_mutable_source_discovery_state() {
    let w = ClaudeHome::new();
    w.append(
        "live.jsonl",
        &claude_user("one", "s", "original native text"),
    );
    let path = w.home.join(".claude/projects/-proj/live.jsonl");
    let (mut e, lock) = engine(&w, 64);
    assert!(e.converge().unwrap().errors.is_empty());
    assert!(
        e.import(std::slice::from_ref(&path))
            .unwrap()
            .pass
            .errors
            .is_empty()
    );
    drop(e);
    drop(lock);
    std::fs::write(
        &path,
        claude_user("one", "s", "changed native text with new bytes"),
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(w.texts("s"), ["changed native text with new bytes"]);
    let pool = ReadPool::open(&e.archive).unwrap();
    let r = pool.reader().unwrap();
    let watched = r.known_watched_inputs(&e.config.machine).unwrap();
    assert_eq!(watched.len(), 1);
    assert_ne!(watched[0].input_id, r.retained_imports().unwrap()[0].id);
}
