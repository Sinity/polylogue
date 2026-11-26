use super::*;
use polylogue_store::fault::{self, Action};

/// The parse state a build left on the one Claude Code chain.
fn standing(engine: &Engine) -> Vec<(String, Option<String>)> {
    engine
        .reads
        .reader()
        .unwrap()
        .query(
            "SELECT p.status,p.reason FROM parse_state p
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
        let (mut engine, _lock) = home.engine(1 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    polylogue_testkit::delete_index(&home.root);
    home
}

/// A failure the chain's bytes did not cause (an injected write failure
/// while the build prepares the chain) is never the chain's failed parse
/// state: the build fails with it, and the next pass reads the chain
/// (docs/archive.md, field accounting and outcomes: parse state).
#[test]
fn a_store_failure_while_the_build_reads_a_chain_is_not_its_parse_failure() {
    let home = stored_and_unbuilt("busy-build");
    let (mut engine, _lock) = home.engine(1 << 20);
    fault::clear();
    fault::arm("reparse.prepare", "", Action::Fail, 0, 1);
    let failed = engine.run_pass();
    fault::clear();
    let error = failed.expect_err("the machine's failure ends the build");
    assert!(
        error
            .to_string()
            .contains("injected fault at reparse.prepare"),
        "{error}"
    );
    assert_eq!(
        polylogue_model::Caused::cause(&error),
        polylogue_model::Cause::Environment
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
    let (mut engine, _lock) = home.engine(1 << 20);
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
    assert_eq!(pass.errors[0].cause, polylogue_model::Cause::Content);
    assert_eq!(pass.errors[0].message, reason);
}
