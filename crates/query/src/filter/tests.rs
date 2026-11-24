use super::*;
use crate::{Bindings, plan::prepare};

fn clock() -> Clock {
    Clock::new(
        jiff::Timestamp::from_millisecond(1_790_000_000_000).unwrap(),
        crate::time::DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap(),
    )
}

/// The whole logical day 2026-01-01 in a clock whose day starts at 04:00 UTC.
fn first_of_january() -> crate::time::ResolvedTime {
    let at = |text: &str| text.parse::<jiff::Timestamp>().unwrap();
    crate::time::ResolvedTime {
        expression: "2026-01-01".into(),
        value: crate::time::Time::Span {
            start: at("2026-01-01T04:00:00Z"),
            end: at("2026-01-02T04:00:00Z"),
        },
        notes: vec![],
    }
}

fn origins() -> Vec<String> {
    vec!["claude-code".into(), "codex".into()]
}

/// The filter a query's selection binds.
fn filter(text: &str) -> Result<Filter, QueryError> {
    let bindings = Bindings {
        origins: origins(),
        ..Bindings::default()
    };
    let plan = prepare(text, &bindings, |_| Ok(()))?;
    Ok(plan
        .bound(&crate::Options::fixture(clock(), origins()))?
        .selection
        .expect("a unit source")
        .filter
        .expect("a condition"))
}

/// The one field condition a query binds, without its position.
fn field(text: &str) -> (FieldId, Test) {
    match filter(text).unwrap() {
        Filter::Field(FieldTest { field, test, .. }) => (field, test),
        other => panic!("{text} bound {other:?}"),
    }
}

/// Values are held as the store keeps them: amounts in nano-dollars,
/// durations in milliseconds, closed sets in their declared spelling,
/// open text in lowercase.
#[test]
fn values_bind_in_stored_form() {
    assert_eq!(
        field("sessions where cost > 2.90"),
        (
            FieldId::SessionsCost,
            Test::NumberCompare(Op::Gt, "2900000000".into())
        )
    );
    assert_eq!(
        field("sessions where wall_span >= 90s"),
        (
            FieldId::SessionsWallSpan,
            Test::NumberCompare(Op::Ge, "90000".into())
        )
    );
    assert_eq!(
        field("messages where role:(USER|Assistant)"),
        (
            FieldId::MessagesRole,
            Test::OneOf {
                values: vec!["assistant".into(), "user".into()],
                unknown: false
            }
        )
    );
    assert_eq!(
        field("sessions where branch:Main"),
        (
            FieldId::SessionsBranch,
            Test::OneOf {
                values: vec!["main".into()],
                unknown: false
            }
        )
    );
    // A field that stores a word for unknown lists it apart.
    assert_eq!(
        field("messages where written_by:(unknown|human)").1,
        Test::OneOf {
            values: vec!["human".into()],
            unknown: true
        }
    );
    assert_eq!(field("sessions where repo:*").1, Test::Any);
}

/// A time is resolved once, against the query's clock, when it binds.
#[test]
fn times_resolve_against_the_clock() {
    let expected = first_of_january();
    assert_eq!(
        field("sessions where started > 2026-01-01"),
        (
            FieldId::SessionsStarted,
            Test::Time(TimeTest::Compare(Op::Gt, expected.clone()))
        )
    );
    assert_eq!(
        field("sessions where since:2026-01-01"),
        (
            FieldId::SessionsSince,
            Test::Time(TimeTest::Within(expected))
        )
    );
}

/// A bare session field on a row unit names the row's session; a prefixed
/// one goes through the declared relation. A comparison through a relation
/// binds as the field when it names a time, and is otherwise computed.
#[test]
fn fields_resolve_through_the_session_and_relations() {
    assert_eq!(field("messages where repo:morrow").0, FieldId::SessionsRepo);
    let related = |text: &str| match filter(text).unwrap() {
        Filter::Related {
            prefix,
            target,
            filter,
            ..
        } => (prefix, target, *filter),
        other => panic!("{text} bound {other:?}"),
    };
    let (prefix, target, inner) = related("messages where session.repo:Morrow");
    assert_eq!((prefix.as_str(), target), ("session", Unit::Sessions));
    assert!(matches!(
        inner,
        Filter::Field(FieldTest {
            field: FieldId::SessionsRepo,
            test: Test::Special(Special::One { ref text, .. }),
            ..
        }) if text == "Morrow"
    ));
    let (_, _, inner) = related("messages where session.started > 2026-01-01");
    let expected = first_of_january();
    assert!(matches!(
        inner,
        Filter::Field(FieldTest {
            field: FieldId::SessionsStarted,
            test: Test::Time(TimeTest::Compare(Op::Gt, ref at)),
            ..
        }) if *at == expected
    ));
    assert!(matches!(
        filter("messages where session.cost > 1").unwrap(),
        Filter::Compare {
            left: Expression::Field { ref name },
            op: Op::Gt,
            right: Expression::Value { value: Value::Scalar { ref text, .. } },
            ..
        } if name == "session.cost" && text == "1"
    ));
}

/// `origin:native` names a session, `block:…` rows by key, a pasted
/// ID-shaped word a session; words search.
#[test]
fn references_and_words_bind_apart_from_fields() {
    assert!(matches!(
        filter("sessions where codex:abc").unwrap(),
        Filter::Session { id, .. } if id == "codex:abc"
    ));
    assert!(matches!(
        filter("messages where block:codex:abc/1").unwrap(),
        Filter::Rows { reference, .. } if reference == "block:codex:abc/1"
    ));
    let Filter::All(parts) = filter("sessions where latest retr*").unwrap() else {
        panic!("not a conjunction");
    };
    assert!(matches!(&parts[0], Filter::SessionWord { word, .. } if word == "latest"));
    assert!(matches!(&parts[1], Filter::Word(Word { text, phrase: false, .. }) if text == "retr*"));
}

/// A closed set's value binds in its declared spelling; one outside the
/// set is refused when binding, naming the set.
#[test]
fn values_outside_the_field_are_refused() {
    let clock = clock();
    let origins = origins();
    let binder = Binder {
        text: "messages where role:robot",
        origins: &origins,
        clock: &clock,
        provider_catalog: None,
        saved: &Default::default(),
        catalog: &crate::catalog::catalog(&[]),
        expanding: &[],
        fields: &[],
        kept: &Default::default(),
        typing: None,
    };
    // The declared spelling is what the column holds, however it was written.
    assert!(matches!(
        binder
            .field(
                Unit::Messages,
                "role",
                Comparison::Match(Value::scalar("USER")),
                Span { start: 15, end: 24 },
            )
            .unwrap(),
        Filter::Field(FieldTest { test: Test::OneOf { ref values, unknown: false }, .. })
            if values == &["user".to_owned()]
    ));
    let refused = binder
        .field(
            Unit::Messages,
            "role",
            Comparison::Match(Value::scalar("robot")),
            Span { start: 15, end: 25 },
        )
        .unwrap_err();
    assert_eq!(
        refused,
        QueryError::BadValue {
            field: "role".into(),
            value: "robot".into(),
            position: 15,
            message: "role on messages is one of user, assistant, system".into(),
        }
    );
    assert_eq!(
        filter("sessions where cost > cheap").unwrap_err(),
        QueryError::BadValue {
            field: "cost".into(),
            value: "cheap".into(),
            position: 15,
            message: "cost is an amount in dollars such as 2.90".into(),
        }
    );
    assert_eq!(
        filter("sessions where colour:red").unwrap_err().kind(),
        polylogue_model::Code::UnknownField
    );
}

#[test]
fn sinex_local_reference_correction_reaches_selection_errors() {
    for kind in ["ev", "record", "key", "at", "window"] {
        let text = format!("{kind}:opaque");
        assert_eq!(
            filter(&text).unwrap_err(),
            QueryError::Invalid {
                message: format!("{text} is a Sinex reference; write sinex:{kind}/opaque"),
                position: 0,
            },
        );
    }
    let combined = "sessions where repo:morrow ev:opaque";
    assert_eq!(
        filter(combined).unwrap_err(),
        QueryError::Invalid {
            message: "ev:opaque is a Sinex reference; write sinex:ev/opaque".into(),
            position: combined.find("ev:").unwrap(),
        },
    );
    assert!(matches!(
        filter(&format!("{} where key:opaque", Unit::Annotations.name())).unwrap(),
        Filter::Field(FieldTest {
            field: FieldId::AnnotationsKey,
            ..
        }),
    ));
    assert!(matches!(
        filter("sessions where title:\"ev:opaque\"").unwrap(),
        Filter::Field(FieldTest {
            field: FieldId::SessionsTitle,
            ..
        }),
    ));
    let Filter::Word(word) = filter("\"ev:opaque\"").unwrap() else {
        panic!("quoted reference text is a search term")
    };
    assert_eq!(word.text, "ev:opaque");
}

#[test]
fn declared_annotation_request_field_precedes_request_object_refusal() {
    let (id, _) = field("annotations where request:write-root");
    assert_eq!(id, FieldId::AnnotationsRequest);
    let operation_clock = clock();
    let binder = Binder {
        text: "request:write-root",
        origins: &[],
        clock: &operation_clock,
        provider_catalog: None,
        saved: &Default::default(),
        catalog: &crate::catalog::catalog(&[]),
        expanding: &[],
        fields: &[],
        kept: &Default::default(),
        typing: None,
    };
    assert_eq!(
        binder
            .field(
                Unit::Sessions,
                "request",
                Comparison::Match(Value::scalar("write-root")),
                Span::default()
            )
            .unwrap_err(),
        QueryError::NotBuilt {
            what: "accepted request objects have no query row-domain producer".into()
        }
    );
}
