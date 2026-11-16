use crate::protocol::line::{Rows, Step};
use crate::protocol::steps::NoteStep;
use serde_json::json;

fn steps() -> &'static command_surface::StepDeclarations {
    crate::commands::declarations()
        .steps("rows", "steps")
        .expect("rows declares its steps")
}

#[test]
fn step_parse_refuses_unknown_option_with_signature() {
    let failure = steps()
        .parse_step("tag", &["later".into(), "--colour".into(), "red".into()])
        .unwrap_err();
    assert_eq!(failure.hint.as_deref(), Some("tag(name)"));
}

#[test]
fn step_parse_reads_keywords_and_relations_into_the_typed_step() {
    let words = [
        "flaky clock",
        "--as",
        "finding",
        "--supports",
        "assertion:a1,assertion:a2",
    ]
    .map(String::from);
    let parsed = steps().parse_step("note", &words).unwrap();
    assert_eq!(
        serde_json::from_value::<Step>(parsed).unwrap(),
        Step::Note(Box::new(NoteStep {
            text: "flaky clock".into(),
            kind: Some("finding".into()),
            supports: vec!["assertion:a1".into(), "assertion:a2".into()],
            ..NoteStep::default()
        }))
    );
}

#[test]
fn step_serde_refuses_unknown_field() {
    let step = json!({"verb": "tag", "name": "later", "colour": "red"});
    assert!(serde_json::from_value::<Step>(step.clone()).is_err());
    assert!(
        serde_json::from_value::<Rows>(json!({"selection": "sessions", "steps": [step]})).is_err()
    );
    assert_eq!(
        serde_json::from_value::<Step>(
            json!({"verb": "resume", "in": "codex", "stop_source": true})
        )
        .unwrap()
        .verb(),
        "resume"
    );
}
