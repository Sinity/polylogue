use super::*;
use serde_json::json;
use sinex_model::MaterialId;

fn source(anchor: u64) -> RecordRef {
    RecordRef {
        material: MaterialId("00000000-0000-0000-0000-000000000001".parse().unwrap()),
        anchor,
    }
}

#[test]
fn native_references_preserve_competing_objectives_and_every_source() {
    let mut native = Native::new(Some("https://example.test/project.git".into()));
    let body = json!({"repository":"https://example.test/project.git",
        "objectives":[{"kind":"task","reference":"task-1","repository":"https://example.test/project.git"}],
        "git_identity":{"common_dir":"/synthetic/.git","root_commits":["root"],"head":"revision","branch":"topic"},
        "terminal_identity":{"host":"synthetic","adapter":"kitty","process_id":22,"window_id":"4","parent_process_id":1},
        "boot_id":"boot-one"});
    native.observe(&body, source(19));
    let mut competing = body.clone();
    competing["objectives"][0]["reference"] = json!("task-2");
    native.observe(&competing, source(200));
    native.observe(&competing, source(200));
    let facts = native.finish();
    assert_eq!(
        facts.repo.as_deref(),
        Some("https://example.test/project.git")
    );
    assert_eq!(
        facts
            .objectives
            .iter()
            .map(|o| o.reference.as_str())
            .collect::<Vec<_>>(),
        ["task-1", "task-2"]
    );
    assert_eq!(facts.sources, vec![source(19), source(200)]);
    assert_eq!(facts.terminal.unwrap().boot_id.as_deref(), Some("boot-one"));
    assert_eq!(
        facts.git_identity.unwrap().head.as_deref(),
        Some("revision")
    );
    assert!(facts.unavailable.is_empty());
}

#[test]
fn conflicting_native_identity_is_unknown_with_both_records_preserved() {
    let mut native = Native::new(None);
    for (index, boot) in ["boot-one", "boot-two"].into_iter().enumerate() {
        native.observe(&json!({"repository":format!("https://example.test/{index}.git"),"boot_id":boot,
            "git_identity":{"common_dir":format!("/synthetic/{index}/.git"),"root_commits":["root"],"head":"revision","branch":null},
            "terminal_identity":{"host":"synthetic","adapter":"kitty","process_id":22,"window_id":"4","parent_process_id":1}}),source(index as u64));
    }
    let facts = native.finish();
    assert!(facts.repo.is_none());
    assert!(facts.git_identity.is_none());
    assert!(facts.terminal.is_none());
    assert!(facts.boot_id.is_none());
    assert_eq!(facts.sources.len(), 2);
    assert_eq!(
        facts.unavailable,
        vec![
            "conflicting native boot observations",
            "conflicting native git observations",
            "conflicting native repository observations",
            "conflicting native terminal observations"
        ]
    );
}

#[test]
fn prose_cwd_and_invalid_native_values_cannot_supply_attribution() {
    let mut native = Native::new(None);
    native.observe(
        &json!({"cwd":"/synthetic/project","objective":"finish task-1", "command":"gh pr view 12"}),
        source(1),
    );
    native.observe(&json!({"objectives":[{"kind":"task","reference":"", "repository":"repo"}],"git_identity":{},"terminal_identity":null,"native_unavailable":["git query timed out"]}),source(2));
    let facts = native.finish();
    assert!(facts.repo.is_none());
    assert!(facts.objectives.is_empty());
    assert!(facts.git_identity.is_none());
    assert!(facts.terminal.is_none());
    assert_eq!(facts.sources, vec![source(2)]);
    assert_eq!(
        facts.unavailable,
        vec![
            "git query timed out",
            "invalid native git_identity",
            "invalid native objective reference"
        ]
    );
}
