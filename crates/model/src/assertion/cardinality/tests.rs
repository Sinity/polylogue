use super::super::{LinkRole, NewAssertionLink};
use super::*;
fn r(text: &str) -> Reference {
    Reference::parse(text).unwrap()
}
fn input(
    kind: AssertionKind,
    key: Option<&str>,
    value: Json,
    links: &[(LinkRole, &str)],
) -> NewAssertion {
    NewAssertion {
        kind,
        key: key.map(str::to_owned),
        value: Some(value),
        valid_until: None,
        links: links
            .iter()
            .map(|(role, reference)| NewAssertionLink {
                role: *role,
                reference: r(reference),
            })
            .collect(),
    }
}
fn keys(input: &NewAssertion, author: &Author, rubric: Option<&Reference>) -> Vec<Key> {
    let value = values::decode(input.kind, input.key.as_deref(), input.value.as_ref()).unwrap();
    input.cardinality_keys(&value, author, rubric).unwrap()
}
#[test]
fn title_and_tag_cardinality_uses_targets_and_ignores_tag_confidence() {
    let title = input(
        AssertionKind::Title,
        None,
        serde_json::json!("first"),
        &[(LinkRole::About, "session:codex:s")],
    );
    let mut revised = title.clone();
    revised.value = Some(serde_json::json!("second"));
    assert_eq!(
        keys(&title, &Author::Operator, None),
        keys(&revised, &Author::Unattributed, None)
    );
    let tag = input(
        AssertionKind::Tag,
        Some("score"),
        serde_json::json!({"value":1,"confidence":0.5}),
        &[
            (LinkRole::About, "message:codex:a"),
            (LinkRole::About, "message:codex:b"),
        ],
    );
    let mut repeated = tag.clone();
    repeated.value = Some(serde_json::from_str(r#"{"value":1.0,"confidence":0.9}"#).unwrap());
    repeated.links.reverse();
    let original = keys(&tag, &Author::Operator, None);
    assert_eq!(original.len(), 2);
    assert_eq!(original, keys(&repeated, &Author::Unattributed, None));
    repeated.value = Some(serde_json::json!({"value":2}));
    assert_ne!(original, keys(&repeated, &Author::Operator, None));
    let note = input(
        AssertionKind::Note,
        None,
        serde_json::json!("free text"),
        &[(LinkRole::About, "message:codex:a")],
    );
    assert!(keys(&note, &Author::Operator, None).is_empty());
}
#[test]
fn annotation_cardinality_keeps_rubric_revision_and_computed_author_distinct() {
    let annotation = input(
        AssertionKind::Annotation,
        Some("clear"),
        serde_json::json!({"value":true}),
        &[
            (LinkRole::About, "message:codex:a"),
            (LinkRole::DerivedFrom, "assertion:rubric@first"),
        ],
    );
    let value = values::decode(
        annotation.kind,
        annotation.key.as_deref(),
        annotation.value.as_ref(),
    )
    .unwrap();
    assert!(matches!(
        annotation.cardinality_keys(&value, &Author::Operator, None),
        Err(Error::Rubric)
    ));
    let first = r("assertion:rubric@first");
    let second = r("assertion:rubric@second");
    let operator = keys(&annotation, &Author::Operator, Some(&first));
    assert_ne!(
        operator,
        keys(&annotation, &Author::Unattributed, Some(&first))
    );
    assert_ne!(
        operator,
        keys(&annotation, &Author::Operator, Some(&second))
    );
    assert!(matches!(
        annotation.cardinality_keys(&value, &Author::Operator, Some(&r("assertion:rubric"))),
        Err(Error::Rubric)
    ));
}
#[test]
fn memory_scope_and_setting_target_host_are_part_of_current_value_identity() {
    let global = input(
        AssertionKind::Memory,
        Some("notes.md"),
        serde_json::json!("text"),
        &[],
    );
    let mut local = global.clone();
    local.links.push(NewAssertionLink {
        role: LinkRole::Scope,
        reference: r("repo:project"),
    });
    assert_ne!(
        keys(&global, &Author::Operator, None),
        keys(&local, &Author::Operator, None)
    );
    let setting = input(
        AssertionKind::Setting,
        Some("signal.interrupts_per_hour"),
        serde_json::json!({"value":{"type":"interrupt_budget","value":4}}),
        &[],
    );
    let mut value =
        values::decode(setting.kind, setting.key.as_deref(), setting.value.as_ref()).unwrap();
    assert!(matches!(
        setting.cardinality_keys(&value, &Author::Operator, None),
        Err(Error::SettingTarget)
    ));
    let values::Value::Setting(raw) = value else {
        panic!("setting shape")
    };
    value = values::Value::Setting(Box::new(values::Structured::plain(
        raw.fields
            .resolve(
                "signal.interrupts_per_hour",
                "first-host",
                crate::settings::Owner::Standalone,
            )
            .unwrap(),
    )));
    let first = setting
        .cardinality_keys(&value, &Author::Operator, None)
        .unwrap();
    let values::Value::Setting(raw) = value else {
        unreachable!()
    };
    let mut other = *raw;
    other.fields.machine = Some("second-host".into());
    let second = setting
        .cardinality_keys(
            &values::Value::Setting(Box::new(other)),
            &Author::Operator,
            None,
        )
        .unwrap();
    assert_ne!(first, second);
}
