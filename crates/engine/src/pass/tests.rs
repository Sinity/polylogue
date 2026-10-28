use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};
use std::io::Read;
use std::os::unix::fs::MetadataExt;

#[test]
fn extent_checkpoint_survives_a_rewrite_read_failure_before_same_inode_growth() {
    let home = ClaudeHome::new();
    let old = "original body ".repeat(40);
    home.append("checkpoint.jsonl", &claude_user("one", "session", &old));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 256);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    home.append(
        "checkpoint.jsonl",
        &claude_user("two", "session", "second body"),
    );
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let path = home.file("checkpoint.jsonl");
    let inode = std::fs::metadata(&path).unwrap().ino();
    // Touch the same file so planning selects a rewrite, then make only that
    // planned read fail. Moving it aside preserves its inode and bytes.
    std::fs::File::open(&path)
        .unwrap()
        .set_modified(std::time::UNIX_EPOCH)
        .unwrap();
    let now = FileFacts::of(&std::fs::metadata(&path).unwrap());
    let plan = engine.plan(&path, now, false).unwrap().unwrap();
    assert_eq!(plan.change, Change::Rewritten);
    let aside = home.dir.path().join("held-source");
    std::fs::rename(&path, &aside).unwrap();
    let mut failed = Pass::default();
    engine.execute_pool(vec![plan], &mut failed).unwrap();
    assert_eq!(failed.errors.len(), 1, "{failed:?}");
    assert!(failed.errors[0].starts_with("read "), "{failed:?}");
    std::fs::rename(&aside, &path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    home.append(
        "checkpoint.jsonl",
        &claude_user("three", "session", "third body"),
    );
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        home.texts("session"),
        [old, "second body".into(), "third body".into()]
    );
    let reader = engine.reads.reader().unwrap();
    let known = reader
        .known_input_at("test-machine", &path.to_string_lossy())
        .unwrap()
        .unwrap();
    let chain = reader
        .stored_chain(&known.input_id, known.chain.unwrap(), known.stored)
        .expect("a failed read must not let the next append overwrite the stored prefix");
    let mut actual = Vec::new();
    chain
        .reader(Arc::clone(&engine.blobs))
        .read_to_end(&mut actual)
        .unwrap();
    assert_eq!(actual, std::fs::read(&path).unwrap());
    assert_eq!(known.parsed_upto, known.stored);
}
