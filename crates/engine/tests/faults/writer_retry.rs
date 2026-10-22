use super::*;
use polylogue_testkit::{ClaudeHome, claude_user as user};
use std::io::Read;
use std::sync::Arc;

fn exact_retained(w: &ClaudeHome, file: &str) {
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let known = r
        .known_input_at("test-machine", &w.file(file).to_string_lossy())
        .unwrap()
        .unwrap();
    let expected = fs::read(w.file(file)).unwrap();
    assert_eq!(known.stored, expected.len() as u64);
    assert_eq!(known.parsed_upto, known.stored);
    let chain = r
        .stored_chain(&known.input_id, known.chain.unwrap(), known.stored)
        .unwrap();
    let store = Arc::new(hashed_blobs::Store::open(w.root.clone(), vec![], 0o600).unwrap());
    let mut actual = Vec::new();
    chain.reader(store).read_to_end(&mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn a_new_input_retries_without_a_source_change_or_restart() {
    for point in [
        "write.archive_commit",
        "write.between_commits",
        "write.index_commit",
    ] {
        fault::clear();
        let w = ClaudeHome::new();
        w.append("new.jsonl", &user("one", "new", "new retained body"));
        let (mut engine, _lock) = w.engine(256);
        fault::arm(point, "", Action::Fail, 0, 64);
        let failed = engine.converge().unwrap();
        assert!(!failed.errors.is_empty());
        assert!(w.texts("new").is_empty());
        fault::clear();
        let retry = engine.converge().unwrap();
        assert!(retry.errors.is_empty(), "{retry:?}");
        assert_eq!(w.texts("new"), ["new retained body"]);
        assert!(engine.converge().unwrap().errors.is_empty());
        assert_eq!(w.texts("new"), ["new retained body"]);
        exact_retained(&w, "new.jsonl");
    }
}

#[test]
fn append_and_rewrite_failures_reload_only_the_rejected_inputs_checkpoints() {
    for rewrite in [false, true] {
        fault::clear();
        let w = ClaudeHome::new();
        w.append("target.jsonl", &user("old", "target", "old body"));
        let (mut engine, _lock) = w.engine(256);
        assert!(engine.converge().unwrap().errors.is_empty());
        let pool = w.reads();
        let r = pool.reader().unwrap();
        let known = r
            .known_input_at("test-machine", &w.file("target.jsonl").to_string_lossy())
            .unwrap()
            .unwrap();
        drop(r);
        drop(pool);
        let replacement = (0..12)
            .map(|n| {
                user(
                    &if rewrite && n == 0 {
                        "old".into()
                    } else {
                        format!("new-{n}")
                    },
                    "target",
                    &format!("new body {n}"),
                )
            })
            .collect::<String>();
        if rewrite {
            fs::write(w.file("target.jsonl"), &replacement).unwrap();
        } else {
            w.append("target.jsonl", &replacement);
        }
        w.append("unrelated.jsonl", &user("other", "other", "other admitted"));
        fault::arm("write.input", &known.input_id, Action::Fail, 0, 64);
        let failed = engine.converge().unwrap();
        assert!(!failed.errors.is_empty());
        assert_eq!(w.texts("target"), ["old body"]);
        assert_eq!(w.texts("other"), ["other admitted"]);
        fault::clear();
        let retry = engine.converge().unwrap();
        assert!(retry.errors.is_empty(), "rewrite={rewrite}: {retry:?}");
        let texts = w.texts("target");
        for n in 0..12 {
            assert!(
                texts.contains(&format!("new body {n}")),
                "rewrite={rewrite}: {texts:?}"
            );
        }
        assert_eq!(texts.len(), 12 + usize::from(!rewrite));
        exact_retained(&w, "target.jsonl");
        assert_eq!(w.texts("other"), ["other admitted"]);
    }
    fault::clear();
}

#[test]
fn a_growing_rewrite_after_restart_checks_the_retained_prefix_before_appending() {
    let w = ClaudeHome::new();
    w.append("rewrite.jsonl", &user("same", "target", "before"));
    {
        let (mut engine, _lock) = w.engine(256);
        assert!(engine.converge().unwrap().errors.is_empty());
    }
    let body = "complete replacement ".repeat(50);
    fs::write(w.file("rewrite.jsonl"), user("same", "target", &body)).unwrap();
    let (mut engine, _lock) = w.engine(256);
    assert!(engine.converge().unwrap().errors.is_empty());
    assert_eq!(w.texts("target"), [body]);
    exact_retained(&w, "rewrite.jsonl");
    assert_eq!(
        w.count("SELECT count(DISTINCT chain) AS n FROM a.revision"),
        2
    );
}
