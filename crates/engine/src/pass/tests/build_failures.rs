use super::*;
use polylogue_store::fault::{self, Action};

/// The parse state a build left on the one Claude Code chain.
fn standing(engine: &Engine) -> Vec<(String, Option<String>)> {
    engine
        .reads
        .reader()
        .unwrap()
        .query(
            "SELECT CASE WHEN p.state='failed' AND p.failure_kind='refused' THEN 'refused' WHEN p.state='failed' THEN 'failed' WHEN p.waiting>0 THEN 'waiting' ELSE 'done' END AS status,p.failure_detail AS reason FROM parse_state p
             JOIN input_ref i ON i.pk=p.input_pk WHERE i.format='claude-code'",
            &[],
        )
        .unwrap()
        .into_iter()
        .map(|row| {
            (
                row["status"].as_str().unwrap().to_owned(),
                row["reason"].as_str().map(str::to_owned),
            )
        })
        .collect()
}

/// An archive whose stored file `name` is due a reading by the next build.
fn stored_and_unbuilt(name: &str) -> ClaudeHome {
    let home = ClaudeHome::new();
    home.append(
        &format!("{name}.jsonl"),
        &claude_user("first", name, "first body"),
    );
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 1 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    polylogue_testkit::delete_index(&home.root);
    home
}

/// A failure the chain's bytes did not cause (an injected I/O failure
/// while the build prepares the chain, which passes by itself) is never the
/// chain's failed parse state: the build leaves the chain out and reports
/// the failure, and the chain is read after the install
/// (docs/archive.md, field accounting and outcomes: parse state).
#[test]
fn a_store_failure_while_the_build_reads_a_chain_is_not_its_parse_failure() {
    let home = stored_and_unbuilt("busy-build");
    let path = home.file("busy-build.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1 << 20);
    fault::clear();
    fault::arm("reparse.prepare", "", Action::Fail, 0, 1);
    let built = engine.run_pass().unwrap();
    fault::clear();
    assert_eq!(built.errors.len(), 1, "{:?}", built.errors);
    assert_eq!(
        built.errors[0].key(),
        (
            path.display().to_string().as_str(),
            polylogue_model::Cause::Transient,
            polylogue_model::Code::Io
        )
    );
    assert!(
        built.errors[0]
            .message
            .contains("injected fault at reparse.prepare"),
        "{:?}",
        built.errors
    );
    assert!(
        standing(&engine)
            .iter()
            .all(|(status, _)| status != "failed"),
        "{:?}",
        standing(&engine)
    );
    let retried = engine.run_pass().unwrap();
    assert!(retried.errors.is_empty(), "{:?}", retried.errors);
    assert_eq!(standing(&engine), [("done".to_owned(), None)]);
    assert_eq!(home.texts("busy-build"), ["first body"]);
}

/// A parser that panics while the build reads its chain fails that chain,
/// with the panic's message as the recorded reason.
#[test]
fn a_parser_panic_in_the_build_fails_the_chain_with_its_message() {
    let home = stored_and_unbuilt("panicking-build");
    let path = home.file("panicking-build.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1 << 20);
    fault::clear();
    fault::arm("parse", "panicking-build", Action::Panic, 0, 1);
    let pass = engine.run_pass().unwrap();
    fault::clear();
    let reason = format!("panicked: injected panic at parse ({})", path.display());
    assert_eq!(
        standing(&engine),
        [("failed".to_owned(), Some(reason.clone()))]
    );
    assert_eq!(pass.errors.len(), 1, "{:?}", pass.errors);
    assert_eq!(
        (pass.errors[0].cause, pass.errors[0].code),
        (polylogue_model::Cause::Bug, polylogue_model::Code::Internal)
    );
    assert_eq!(pass.errors[0].message, reason);
}
