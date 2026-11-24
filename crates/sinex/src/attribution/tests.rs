use super::*;
use serde_json::json;
use sinex_model::MaterialId;

fn source(offset: u64) -> RecordRef {
    RecordRef {
        material: MaterialId("00000000-0000-0000-0000-000000000001".parse().unwrap()),
        anchor: offset,
    }
}

#[test]
fn stated_references_keep_objectives_and_all_sources() {
    let mut stated = SessionAttribution::new(Some("https://example.test/project.git".into()));
    let body = json!({"repository":"https://example.test/project.git",
        "objectives":[{"kind":"task","reference":"task-1","repository":"https://example.test/project.git"}],
        "git_identity":{"common_dir":"/synthetic/.git","root_commits":["root"],"head":"revision","branch":"topic"},
        "terminal_identity":{"host":"synthetic","adapter":"kitty","process_id":22,"window_id":"4","parent_process_id":1},
        "boot_id":"boot-one"});
    stated.observe(&body, source(19));
    let mut competing = body.clone();
    competing["objectives"][0]["reference"] = json!("task-2");
    stated.observe(&competing, source(200));
    stated.observe(&competing, source(200));
    let facts = stated.finish();
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
fn conflicting_stated_identity_is_unknown() {
    let mut stated = SessionAttribution::new(None);
    for (index, boot) in ["boot-one", "boot-two"].into_iter().enumerate() {
        stated.observe(&json!({"repository":format!("https://example.test/{index}.git"),"boot_id":boot,
            "git_identity":{"common_dir":format!("/synthetic/{index}/.git"),"root_commits":["root"],"head":"revision","branch":null},
            "terminal_identity":{"host":"synthetic","adapter":"kitty","process_id":22,"window_id":"4","parent_process_id":1}}),source(index as u64));
    }
    let facts = stated.finish();
    assert!(facts.repo.is_none());
    assert!(facts.git_identity.is_none());
    assert!(facts.terminal.is_none());
    assert!(facts.boot_id.is_none());
    assert_eq!(facts.sources.len(), 2);
    assert_eq!(
        facts.unavailable,
        vec![
            "conflicting stated boot observations",
            "conflicting stated git observations",
            "conflicting stated repository observations",
            "conflicting stated terminal observations"
        ]
    );
}

#[test]
fn prose_cwd_and_bad_stated_values_give_no_attribution() {
    let mut stated = SessionAttribution::new(None);
    stated.observe(
        &json!({"cwd":"/synthetic/project","objective":"finish task-1", "command":"gh pr view 12"}),
        source(1),
    );
    stated.observe(&json!({"objectives":[{"kind":"task","reference":"", "repository":"repo"}],"git_identity":{},"terminal_identity":null,"unstated":["git query timed out"]}),source(2));
    let facts = stated.finish();
    assert!(facts.repo.is_none());
    assert!(facts.objectives.is_empty());
    assert!(facts.git_identity.is_none());
    assert!(facts.terminal.is_none());
    assert_eq!(facts.sources, vec![source(2)]);
    assert_eq!(
        facts.unavailable,
        vec![
            "git query timed out",
            "invalid stated git_identity",
            "invalid stated objective reference"
        ]
    );
}
