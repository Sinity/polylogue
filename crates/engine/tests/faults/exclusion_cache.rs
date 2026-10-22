use super::*;
use polylogue_engine::{ExciseRequest, ExciseScope};
use polylogue_model::SessionId;
use polylogue_testkit::{ClaudeHome, claude_user as user};

fn remove(id: &str) -> ExciseRequest {
    ExciseRequest {
        id: format!("remove-{id}"),
        scope: ExciseScope::Session(SessionId::parse(&format!("claude-code:{id}")).unwrap()),
        reason: None,
        dry_run: false,
    }
}

#[test]
fn failed_replacement_cache_rolls_back_excision_and_keeps_previous_exclusions() {
    fault::clear();
    let w = ClaudeHome::new();
    w.append("first.jsonl", &user("a1", "first", "first synthetic body"));
    w.append(
        "second.jsonl",
        &user("b1", "second", "second synthetic body"),
    );
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    engine.excise(remove("first")).unwrap();
    fault::arm(
        "exclusion_cache.insert",
        "claude-code:second",
        Action::Fail,
        0,
        1,
    );
    let error = engine.excise(remove("second")).unwrap_err();
    assert!(
        error.to_string().contains("exclusion_cache.insert"),
        "{error}"
    );
    assert_eq!(w.texts("second"), ["second synthetic body"]);
    w.append(
        "different.jsonl",
        &user("a2", "first", "still excluded after failed replacement"),
    );
    assert!(engine.converge().unwrap().errors.is_empty());
    assert!(w.texts("first").is_empty());
    assert_eq!(w.texts("second"), ["second synthetic body"]);
    engine.excise(remove("second")).unwrap();
    assert!(w.texts("second").is_empty());
    fault::clear();
}

#[test]
fn failed_lookup_writes_nothing_and_restart_rebuilds_the_lookup() {
    fault::clear();
    let w = ClaudeHome::new();
    w.append("original.jsonl", &user("a1", "removed", "removed body"));
    let (mut engine, lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    engine.excise(remove("removed")).unwrap();
    w.append(
        "mixed.jsonl",
        &(user("a2", "removed", "must remain excluded")
            + &user("b1", "retained", "admitted after retry")),
    );
    fault::arm("exclusion_cache.lookup", "messages", Action::Fail, 0, 1);
    let failed = engine.converge().unwrap();
    assert!(
        failed
            .errors
            .iter()
            .any(|error| error.contains("exclusion_cache.lookup")),
        "{failed:?}"
    );
    assert!(w.texts("removed").is_empty());
    assert!(w.texts("retained").is_empty());
    drop(engine);
    drop(lock);
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    assert!(w.texts("removed").is_empty());
    assert_eq!(w.texts("retained"), ["admitted after retry"]);
    fault::clear();
}

#[test]
fn failed_lookup_retries_in_the_same_engine_from_committed_progress() {
    fault::clear();
    let w = ClaudeHome::new();
    w.append("original.jsonl", &user("a1", "removed", "removed body"));
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    engine.excise(remove("removed")).unwrap();
    w.append(
        "mixed.jsonl",
        &(user("a2", "removed", "must remain excluded")
            + &user("b1", "retained", "admitted after retry")),
    );
    fault::arm("exclusion_cache.lookup", "messages", Action::Fail, 0, 1);
    let failed = engine.converge().unwrap();
    assert!(
        failed
            .errors
            .iter()
            .any(|error| error.contains("exclusion_cache.lookup")),
        "{failed:?}"
    );
    assert!(w.texts("removed").is_empty());
    assert!(w.texts("retained").is_empty());
    assert!(engine.converge().unwrap().errors.is_empty());
    assert!(w.texts("removed").is_empty());
    assert_eq!(w.texts("retained"), ["admitted after retry"]);
    fault::clear();
}
