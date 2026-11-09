use super::super::{LinkRole, NewAssertionLink};
use super::*;
use crate::Json;
use authored::{SlotValue, UniqueSlot, WhenHeld};

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
fn unique(input: &NewAssertion, author: &Author, rubric: Option<&Reference>) -> Option<Unique> {
    let value = values::decode(input.kind, input.key.as_deref(), input.value.as_ref()).unwrap();
    input.unique(&value, author, rubric).unwrap()
}
fn slots(input: &NewAssertion, author: &Author, rubric: Option<&Reference>) -> Vec<UniqueSlot> {
    unique(input, author, rubric).unwrap().slots
}

#[test]
fn title_and_tag_slots_use_their_targets_and_ignore_tag_confidence() {
    let title = input(
        AssertionKind::Title,
        None,
        serde_json::json!("first"),
        &[(LinkRole::About, "session:codex:s")],
    );
    let mut revised = title.clone();
    revised.value = Some(serde_json::json!("second"));
    let title_slots = unique(&title, &Author::Operator, None).unwrap();
    assert_eq!(title_slots.when_held, WhenHeld::Revise);
    assert_eq!(
        title_slots.slots,
        vec![UniqueSlot {
            kind: "title".into(),
            parts: vec![("about".into(), SlotValue::One("session:codex:s".into()))],
        }]
    );
    assert_eq!(
        title_slots.slots,
        slots(&revised, &Author::Unattributed, None)
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
    let original = unique(&tag, &Author::Operator, None).unwrap();
    assert_eq!(original.when_held, WhenHeld::Already);
    assert_eq!(original.slots.len(), 2);
    assert_eq!(
        original.slots,
        slots(&repeated, &Author::Unattributed, None)
    );
    repeated.value = Some(serde_json::json!({"value":2}));
    assert_ne!(original.slots, slots(&repeated, &Author::Operator, None));
    let note = input(
        AssertionKind::Note,
        None,
        serde_json::json!("free text"),
        &[(LinkRole::About, "message:codex:a")],
    );
    assert_eq!(unique(&note, &Author::Operator, None), None);
}

#[test]
fn annotation_slots_keep_the_rubric_revision_and_computed_author_distinct() {
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
        annotation.unique(&value, &Author::Operator, None),
        Err(Error::Rubric)
    ));
    let first = r("assertion:rubric@first");
    let second = r("assertion:rubric@second");
    let operator = slots(&annotation, &Author::Operator, Some(&first));
    assert_ne!(
        operator,
        slots(&annotation, &Author::Unattributed, Some(&first))
    );
    assert_ne!(
        operator,
        slots(&annotation, &Author::Operator, Some(&second))
    );
    assert!(matches!(
        annotation.unique(&value, &Author::Operator, Some(&r("assertion:rubric"))),
        Err(Error::Rubric)
    ));
}

#[test]
fn memory_scope_and_setting_target_host_are_part_of_the_slot() {
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
        slots(&global, &Author::Operator, None),
        slots(&local, &Author::Operator, None)
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
        setting.unique(&value, &Author::Operator, None),
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
    let first = setting.unique(&value, &Author::Operator, None).unwrap();
    let values::Value::Setting(raw) = value else {
        unreachable!()
    };
    let mut other = *raw;
    other.fields.machine = Some("second-host".into());
    let second = setting
        .unique(
            &values::Value::Setting(Box::new(other)),
            &Author::Operator,
            None,
        )
        .unwrap();
    assert_ne!(first, second);
}

#[test]
fn every_kind_declares_only_components_the_writer_supplies() {
    for kind in AssertionKind::ALL {
        if let Some(uniqueness) = kind.spec().uniqueness {
            for component in uniqueness.components {
                assert!(
                    matches!(
                        *component,
                        "about" | "scope" | "key" | "author" | "value" | "rubric" | "machine"
                    ),
                    "{kind:?} declares {component}"
                );
            }
        }
    }
    assert!(AssertionKind::Hidden.spec().single_valued_mark());
    assert!(AssertionKind::Title.spec().single_valued_mark());
    assert!(AssertionKind::Outcome.spec().single_valued_mark());
    assert!(!AssertionKind::Tag.spec().single_valued_mark());
}

#[test]
fn blocker_is_not_a_kind_and_hidden_is() {
    assert_eq!(AssertionKind::from_name("blocker"), None);
    assert_eq!(
        AssertionKind::from_name("hidden"),
        Some(AssertionKind::Hidden)
    );
    let hidden = input(
        AssertionKind::Hidden,
        None,
        serde_json::json!("noise"),
        &[(LinkRole::About, "session:codex:s")],
    );
    hidden.validate_links().unwrap();
    assert_eq!(
        slots(&hidden, &Author::Operator, None),
        vec![UniqueSlot {
            kind: "hidden".into(),
            parts: vec![("about".into(), SlotValue::One("session:codex:s".into()))],
        }]
    );
}

#[test]
fn canonical_text_is_equal_exactly_when_values_are_equal() {
    for (left, right) in [
        ("1", "1.0"),
        ("10e-1", "1"),
        ("-0", "0"),
        (r#"{"b":1,"a":[2.50]}"#, r#"{"a":[2.5],"b":1.0}"#),
    ] {
        let left: Json = serde_json::from_str(left).unwrap();
        let right: Json = serde_json::from_str(right).unwrap();
        assert_eq!(values::canonical(&left), values::canonical(&right));
    }
    for (left, right) in [("1", "2"), ("0.1", "1"), (r#""1""#, "1"), ("100", "1e3")] {
        let left: Json = serde_json::from_str(left).unwrap();
        let right: Json = serde_json::from_str(right).unwrap();
        assert_ne!(values::canonical(&left), values::canonical(&right));
    }
}
