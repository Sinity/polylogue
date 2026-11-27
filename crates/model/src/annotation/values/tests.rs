use super::*;

#[test]
fn saved_denominator_is_a_query_hash_not_a_name() {
    let mut value = serde_json::json!({"text":"sessions","hash":BlobHash([2;32]),"denominator":format!("query:{}",BlobHash([3;32]))});
    decode(
        super::super::AnnotationKind::SavedQuery,
        Some("population"),
        Some(&value),
    )
    .unwrap();
    value["denominator"] = "query:population".into();
    value["predecessor"] = serde_json::json!({"denominator":format!("query:{}",BlobHash([3;32]))});
    assert!(
        decode(
            super::super::AnnotationKind::SavedQuery,
            Some("population"),
            Some(&value)
        )
        .is_err()
    );
}
fn number(text: &str) -> serde_json::Number {
    serde_json::from_str(text).unwrap()
}
fn numeric(kind: FieldType, minimum: Option<&str>, maximum: Option<&str>) -> Field {
    Field {
        name: "score".into(),
        value_type: kind,
        enum_values: None,
        minimum: minimum.map(number),
        maximum: maximum.map(number),
        required: None,
        description: None,
    }
}

#[test]
fn probabilities_and_rubric_ranges_preserve_exact_decimal_boundaries() {
    for text in [
        "0",
        "-0",
        "1",
        "1.0000000000000000000000000000000000000",
        "0.0000000000000000000000000000000000001",
        "1e-9999999999999999999999999999999999999",
    ] {
        let probability: Probability = serde_json::from_str(text).unwrap();
        assert_eq!(probability.number(), &number(text));
        let encoded = serde_json::to_string(&probability).unwrap();
        assert_eq!(
            serde_json::from_str::<Probability>(&encoded).unwrap(),
            probability
        );
    }
    for text in [
        "1.0000000000000000000000000000000000001",
        "-1e-9999999999999999999999999999999999999",
        "1e9999999999999999999999999999999999999",
        "-1",
        "false",
        "null",
        "\"0.5\"",
    ] {
        assert!(serde_json::from_str::<Probability>(text).is_err(), "{text}");
    }
    let field = numeric(FieldType::Number, Some("0"), Some("1"));
    field.validate().unwrap();
    assert!(field.allows(&serde_json::from_str("1e-1000000").unwrap()));
    assert!(!field.allows(&serde_json::from_str("-1e-1000000").unwrap()));
    assert!(!field.allows(&serde_json::from_str("1.000000000000000000000001").unwrap()));
    assert!(!field.allows(&Json::Bool(true)));
    let field = numeric(
        FieldType::Number,
        Some("9007199254740993"),
        Some("9007199254740994"),
    );
    assert!(!field.allows(&serde_json::from_str("9007199254740992.9999999999999999999").unwrap()));
    assert!(field.allows(&serde_json::from_str("9007199254740993.0000000000000000001").unwrap()));
    assert!(!field.allows(&serde_json::from_str("9007199254740994.0000000000000000001").unwrap()));
    assert!(
        numeric(FieldType::Number, Some("1.0000000000000000001"), Some("1"))
            .validate()
            .is_err()
    );
}
#[test]
fn numeric_ranges_compare_exponent_carries_without_expanding_them() {
    let range = numeric(
        FieldType::Number,
        Some("1e999999999999999999999999"),
        Some("1e1000000000000000000000000"),
    );
    range.validate().unwrap();
    for text in [
        "10e999999999999999999999999",
        "99e999999999999999999999998",
        "1e1000000000000000000000000",
    ] {
        assert!(range.allows(&serde_json::from_str(text).unwrap()), "{text}");
    }
    for text in [
        "9e999999999999999999999998",
        "10.0000000000000000001e999999999999999999999999",
        "1e1000000000000000000000001",
    ] {
        assert!(
            !range.allows(&serde_json::from_str(text).unwrap()),
            "{text}"
        );
    }
    let range = numeric(
        FieldType::Number,
        Some("-1e1000000000000000000000000"),
        Some("-1e999999999999999999999999"),
    );
    range.validate().unwrap();
    assert!(range.allows(&serde_json::from_str("-10e999999999999999999999999").unwrap()));
    assert!(!range.allows(
        &serde_json::from_str("-10.0000000000000000001e999999999999999999999999").unwrap()
    ));
}

#[test]
fn rubric_integer_ranges_and_enum_numbers_have_one_mathematical_meaning() {
    let field = numeric(FieldType::Integer, None, None);
    for text in [
        "0",
        "-0.0",
        "1.00",
        "1.5e1",
        "1e999999999999999999999999999999999999999",
    ] {
        assert!(field.allows(&serde_json::from_str(text).unwrap()), "{text}");
    }
    for text in [
        "0.1",
        "1.5000001e1",
        "-1e-999999999999999999999999999999999999999",
        "true",
    ] {
        assert!(
            !field.allows(&serde_json::from_str(text).unwrap()),
            "{text}"
        );
    }
    let mut field = Field {
        name: "label".into(),
        value_type: FieldType::Enum,
        enum_values: Some(vec![
            serde_json::from_str("1").unwrap(),
            serde_json::from_str("1.0").unwrap(),
        ]),
        minimum: None,
        maximum: None,
        required: None,
        description: None,
    };
    assert!(field.validate().is_err());
    field.enum_values = Some(vec![serde_json::from_str("1").unwrap()]);
    for text in ["1", "1.0", "10e-1"] {
        assert!(field.allows(&serde_json::from_str(text).unwrap()), "{text}");
    }
    assert!(!field.allows(&serde_json::from_str("2").unwrap()));
    assert!(!field.allows(&Json::String("1".into())));
    field.enum_values = Some(vec![
        Json::String("clear".into()),
        Json::String("uncertain".into()),
    ]);
    field.validate().unwrap();
    assert!(field.allows(&Json::String("clear".into())));
    assert!(!field.allows(&Json::String("invented".into())));
    assert!(!field.allows(&Json::Null));
    field.minimum = Some(number("0"));
    assert!(field.validate().is_err());
}
#[test]
fn comparison_values_round_trip_and_refuse_forged_verdict_shapes() {
    let reference = |text: &str| Reference::parse(text).unwrap();
    for (text, expected) in [
        (
            r#""prefer:message:codex:m""#,
            Verdict::Prefer(reference("message:codex:m")),
        ),
        (r#""tie""#, Verdict::Tie),
        (r#""incomparable""#, Verdict::Incomparable),
        (r#""abstain""#, Verdict::Abstain),
        (r#""insufficient_evidence""#, Verdict::InsufficientEvidence),
        (
            r#"{"ranking":["message:codex:a","message:codex:b"]}"#,
            Verdict::Ranking(vec![
                reference("message:codex:a"),
                reference("message:codex:b"),
            ]),
        ),
    ] {
        let verdict: Verdict = serde_json::from_str(text).unwrap();
        assert_eq!(verdict, expected, "{text}");
        assert_eq!(
            serde_json::to_value(&verdict).unwrap(),
            serde_json::from_str::<Json>(text).unwrap(),
            "{text}"
        );
    }
    for text in [
        r#""winner""#,
        r#""prefer:bad""#,
        r#"{"ranking":[],"author":"operator"}"#,
        r#"{"prefer":"message:codex:m"}"#,
    ] {
        assert!(serde_json::from_str::<Verdict>(text).is_err(), "{text}");
    }
}
#[test]
fn rubric_declarations_keep_python_types_and_ranges() {
    let rubric:Rubric=serde_json::from_value(serde_json::json!({
        "instructions":"Assess the visible evidence and abstain when it is missing.",
        "unit_query":"messages role:assistant",
        "fields":[
            {"name":"applicable","value_type":"boolean"},
            {"name":"confidence","value_type":"number","minimum":0,"maximum":1},
            {"name":"directive_mode","value_type":"enum","enum_values":["imperative","collaborative","not_observed"]}
        ],"abstain":true,"confidence":true,"rationale":true
    })).unwrap();
    rubric.validate().unwrap();
    let mut duplicate = rubric.clone();
    duplicate.fields.push(duplicate.fields[0].clone());
    assert!(duplicate.validate().is_err());
    assert!(serde_json::from_value::<Rubric>(serde_json::json!({"instructions":"text","unit_query":"messages","fields":[],"abstain":true,"confidence":true,"author":"operator"})).is_err());
}

#[test]
fn setting_resolves_target_host_keeping_accepting_host() {
    use crate::settings::{Error, SettingMode, Value};
    let archive = Setting {
        machine: None,
        value: Value::Boolean(true),
    }
    .resolve("browser.actions", "accepting", SettingMode::Standalone)
    .unwrap();
    assert_eq!(archive.machine, None);
    assert_eq!(archive.value, Value::Boolean(true));
    let host = Setting {
        machine: None,
        value: Value::QuietHours(Some([1320, 420])),
    }
    .resolve("signal.quiet_hours", "accepting", SettingMode::Standalone)
    .unwrap();
    assert_eq!(host.machine.as_deref(), Some("accepting"));
    let remote = Setting {
        machine: Some("other".into()),
        value: Value::QuietHours(None),
    }
    .resolve("signal.quiet_hours", "accepting", SettingMode::Standalone)
    .unwrap();
    assert_eq!(remote.machine.as_deref(), Some("other"));
    assert!(
        Setting {
            machine: Some("other".into()),
            value: Value::Boolean(true)
        }
        .resolve("browser.actions", "accepting", SettingMode::Standalone)
        .is_err()
    );
    assert!(
        Setting {
            machine: Some("*".into()),
            value: Value::QuietHours(None)
        }
        .resolve("signal.quiet_hours", "accepting", SettingMode::Standalone)
        .is_err()
    );
    assert!(matches!(
        Setting {
            machine: None,
            value: Value::QuietHours(None)
        }
        .resolve("signal.quiet_hours", "accepting", SettingMode::Primary),
        Err(Error::SinexOwned(_))
    ));
    assert!(
        Value::Text("off".into())
            .validate_for("embed.provider", SettingMode::Standalone)
            .is_err()
    );
    assert!(
        Value::Alternates(vec!["invented:pl".into()])
            .validate_for("reading.alternates", SettingMode::Standalone)
            .is_err()
    );
    assert!(
        Value::QuietHours(Some([1440, 0]))
            .validate_for("signal.quiet_hours", SettingMode::Standalone)
            .is_err()
    );
    assert!(
        serde_json::from_value::<Value>(
            serde_json::json!({"type":"nonnegative_number","value":-1})
        )
        .is_err()
    );
    assert!(
        Value::Boolean(true)
            .validate_for("alternates.enabled", SettingMode::Standalone)
            .is_err()
    );
    assert!(serde_json::from_value::<Setting>(serde_json::json!({"machine":null,"value":{"type":"boolean","value":true},"author":"operator"})).is_err());
}

#[test]
fn kind_shapes_decode_declared_and_refuse_malformed() {
    use super::super::AnnotationKind as K;
    let text = Json::String("authored text".into());
    let mut cases: Vec<(K, Option<&str>, Option<Json>)> = vec![
        (
            K::Tag,
            Some("research"),
            Some(serde_json::json!({"value":"indexed","confidence":0.8})),
        ),
        (K::Highlight, Some("yellow"), None),
        (K::Outcome, None, Some(serde_json::json!("partial"))),
        (
            K::Forecast,
            Some("completion"),
            Some(
                serde_json::json!({"probability":0.5,"horizon":"2026-11-01T12:00:00Z","resolves_by":"the published result"}),
            ),
        ),
        (
            K::Judgment,
            Some("accepted"),
            Some(
                serde_json::json!({"reason":"supported","blind":{"order":3,"hidden":["author","model"],"seed":"sample-a"}}),
            ),
        ),
        (
            K::Comparison,
            Some("clarity"),
            Some(serde_json::json!({"verdict":"tie"})),
        ),
        (
            K::Rubric,
            Some("clarity"),
            Some(
                serde_json::json!({"instructions":"Assess clarity","unit_query":"messages","fields":[{"name":"clear","value_type":"boolean"}],"abstain":true,"confidence":false}),
            ),
        ),
        (
            K::Pass,
            None,
            Some(
                serde_json::json!({"rubric":"annotation:rubric@revision","population":"result:population","counts":{"covered":2}}),
            ),
        ),
        (
            K::RubricValue,
            Some("clear"),
            Some(serde_json::json!({"value":true})),
        ),
        (
            K::Method,
            Some("check"),
            Some(
                serde_json::json!({"inputs":["archive"],"conditions":[{"name":"ready","check":{"query":"messages","expect":"rows"}}],"steps":["inspect"],"expected":"evidence","stop":["source unavailable"]}),
            ),
        ),
        (
            K::SavedQuery,
            Some("recent"),
            Some(serde_json::json!({"text":"messages","hash":BlobHash([2;32])})),
        ),
        (
            K::Signal,
            Some("changed"),
            Some(serde_json::json!({"name":"changed","query":"saved:recent"})),
        ),
        (
            K::Page,
            Some("project"),
            Some(serde_json::json!({"title":"Project","layout":"linear"})),
        ),
        (
            K::Slot,
            None,
            Some(
                serde_json::json!({"section":"Evidence","position":0.5,"mode":"follow","heading":"Findings"}),
            ),
        ),
        (K::Member, Some("reviewer"), None),
        (
            K::Role,
            Some("reviewer"),
            Some(serde_json::json!({"instructions":"Review the evidence","harness":"codex"})),
        ),
        (
            K::Message,
            None,
            Some(
                serde_json::json!({"text":"Please review","subject":"annotation:claim@revision","reply_to":{"page":"page:project","section":"Discussion"}}),
            ),
        ),
        (
            K::Seen,
            Some("session:codex:session"),
            Some(
                serde_json::json!({"position":{"item":"message:codex:item","at":"generation:2:3"}}),
            ),
        ),
        (K::Withdrawal, None, None),
        (K::Hidden, None, Some(Json::String("duplicate".into()))),
        (
            K::Setting,
            Some("browser.actions"),
            Some(serde_json::json!({"machine":null,"value":{"type":"boolean","value":true}})),
        ),
    ];
    for kind in [
        K::Note,
        K::Title,
        K::Claim,
        K::Finding,
        K::Decision,
        K::Lesson,
        K::Question,
        K::Goal,
        K::Problem,
        K::Handoff,
        K::Correction,
        K::Memory,
    ] {
        cases.push((
            kind,
            if kind == K::Memory {
                Some("notes.md")
            } else {
                None
            },
            Some(text.clone()),
        ));
    }
    for kind in K::ALL {
        let (_, key, value) = cases
            .iter()
            .find(|(candidate, _, _)| candidate == kind)
            .unwrap_or_else(|| panic!("missing value oracle for {kind:?}"));
        let decoded = decode(*kind, *key, value.as_ref()).unwrap_or_else(|error| panic!("{error}"));
        let expected = match kind {
            K::Tag => "tag",
            K::Highlight | K::Member | K::Withdrawal => "absent",
            K::Hidden
            | K::Note
            | K::Title
            | K::Lesson
            | K::Goal
            | K::Problem
            | K::Handoff
            | K::Correction
            | K::Memory => "text",
            K::Claim | K::Finding | K::Question => "claim",
            K::Decision => "decision",
            K::Outcome => "outcome",
            K::Forecast => "forecast",
            K::Judgment => "judgment",
            K::Comparison => "comparison",
            K::Rubric => "rubric",
            K::Pass => "pass",
            K::RubricValue => "rubric_value",
            K::Method => "method",
            K::SavedQuery => "saved_query",
            K::Signal => "signal",
            K::Page => "page",
            K::Slot => "slot",
            K::Role => "role",
            K::Message => "message",
            K::Setting => "setting",
            K::Seen => "seen",
        };
        let found = match decoded {
            Value::Absent => "absent",
            Value::Text(_) => "text",
            Value::Tag(_) => "tag",
            Value::Outcome(_) => "outcome",
            Value::Claim(_) => "claim",
            Value::Decision(_) => "decision",
            Value::Forecast(_) => "forecast",
            Value::Judgment(_) => "judgment",
            Value::Comparison(_) => "comparison",
            Value::Rubric(_) => "rubric",
            Value::Pass(_) => "pass",
            Value::RubricValue(_) => "rubric_value",
            Value::Method(_) => "method",
            Value::SavedQuery(_) => "saved_query",
            Value::Signal(_) => "signal",
            Value::Page(_) => "page",
            Value::Slot(_) => "slot",
            Value::Role(_) => "role",
            Value::Message(_) => "message",
            Value::Setting(_) => "setting",
            Value::Seen(_) => "seen",
        };
        assert_eq!(found, expected, "{kind:?}");
        assert!(
            decode(*kind, *key, Some(&serde_json::json!({"author":"operator"}))).is_err(),
            "{kind:?}"
        );
        let value_is_optional = matches!(
            kind,
            K::Tag | K::Highlight | K::Judgment | K::Member | K::Hidden | K::Withdrawal
        );
        assert_eq!(
            decode(*kind, *key, None).is_ok(),
            value_is_optional,
            "{kind:?}"
        );
    }
    assert!(decode(K::Outcome, None, Some(&serde_json::json!("complete"))).is_err());
    assert!(decode(K::Judgment, Some("approve"), None).is_err());
    assert!(
        decode(
            K::Page,
            Some("project"),
            Some(&serde_json::json!({"title":"Project","layout":"freeform"}))
        )
        .is_err()
    );
    assert!(
        decode(
            K::Signal,
            Some("different"),
            Some(&serde_json::json!({"name":"changed","query":"messages"}))
        )
        .is_err()
    );
    assert!(matches!(
        decode(
            K::RubricValue,
            Some("score"),
            Some(&serde_json::json!({"value":{"rubric_specific":"value"}}))
        )
        .unwrap(),
        Value::RubricValue(_)
    ));
}
#[test]
fn checked_claims_and_experiment_assignments_keep_their_declared_shapes() {
    use super::super::AnnotationKind as K;
    let claim = serde_json::json!({"text":"There is evidence","check":{"query":"messages","expect":"count >= 2"},"outcomes":[{"observation":"no rows","implies":"the source is absent"}]});
    assert!(decode(K::Claim, None, Some(&claim)).is_ok());
    assert!(decode(K::Finding, None, Some(&claim)).is_err());
    assert!(decode(K::Question, None, Some(&claim)).is_err());
    let outcomes_only = serde_json::json!({"text":"evidence","outcomes":[{"observation":"no rows","implies":"the source is absent"}]});
    assert!(decode(K::Claim, None, Some(&outcomes_only)).is_ok());
    assert!(decode(K::Finding, None, Some(&outcomes_only)).is_err());
    let check_only =
        serde_json::json!({"text":"what follows?","check":{"query":"messages","expect":"rows"}});
    assert!(decode(K::Question, None, Some(&check_only)).is_err());
    assert!(
        decode(
            K::Finding,
            None,
            Some(
                &serde_json::json!({"text":"evidence","check":{"query":"messages","expect":"rows"}})
            )
        )
        .is_ok()
    );
    assert!(decode(K::Question,None,Some(&serde_json::json!({"text":"what follows?","outcomes":[{"observation":"no rows","implies":"the source is absent"}]}))).is_ok());
    for expect in [
        "rows",
        "no_rows",
        "count >= 999999999999999999999999999999999999",
        "agg < -0.00000000000000000001",
    ] {
        Check {
            query: "messages".into(),
            expect: expect.into(),
        }
        .validate()
        .unwrap();
    }
    for expect in [
        "count >= -1",
        "count > 1.5",
        "agg = NaN",
        "count >",
        "unknown",
        "rows extra",
    ] {
        assert!(
            Check {
                query: "messages".into(),
                expect: expect.into()
            }
            .validate()
            .is_err(),
            "{expect}"
        );
    }
    let assignment = serde_json::json!({"units":"result:population","arms":["baseline","changed"],"method":"seeded:sample-a","assignment":{"message:codex:m":"baseline"}});
    assert!(matches!(
        decode(K::Decision, None, Some(&assignment)).unwrap(),
        Value::Decision(_)
    ));
    let mut invalid = assignment.clone();
    invalid["assignment"]["message:codex:m"] = serde_json::json!("undeclared");
    assert!(decode(K::Decision, None, Some(&invalid)).is_err());
    let setting: Setting = serde_json::from_value(
        serde_json::json!({"value":{"type":"nonnegative_number","value":0.125}}),
    )
    .unwrap();
    setting
        .value
        .validate_for(
            "embed.price_per_mtok",
            crate::settings::SettingMode::Standalone,
        )
        .unwrap();
}

#[test]
fn python_polylogue_text_keeps_only_its_content() {
    use super::super::{AnnotationFact, AnnotationKind as K};
    let original = "original a\u{301}\n\rβ";
    let metadata = serde_json::json!({"id":"note:older","author":"operator","scope":"repo:claimed","text":"not the body","unknown":{"nested":[1,"kept"]}});
    let raw = serde_json::json!({"text":original,"predecessor":metadata});
    let carried = decode(K::Note, None, Some(&raw)).unwrap();
    let plain = decode(K::Note, None, Some(&serde_json::json!(original))).unwrap();
    assert_eq!(carried, plain);
    let Value::Text(text) = carried else {
        panic!("Python Polylogue text")
    };
    assert_eq!(serde_json::to_value(text).unwrap(), raw);
    assert!(decode_live(K::Note, None, Some(&raw)).is_err());
    assert!(decode_live(K::Note, None, Some(&serde_json::json!(original))).is_ok());
    let annotation = AnnotationFact {
        id: "old".into(),
        root: "old".into(),
        kind: K::Note,
        key: None,
        value: Some(raw.clone()),
        author: crate::Author::Unattributed,
        via: crate::authorship::Channel::Import,
        machine: "importing-host".into(),
        created_at: Some(crate::Timestamp::from_unix_ms(0).unwrap()),
        valid_until: None,
        links: Vec::new(),
    };
    assert_eq!(annotation.text(), Some(original));

    for invalid in [
        serde_json::json!({"text":original}),
        serde_json::json!({"text":0,"predecessor":metadata}),
        serde_json::json!({"text":original,"predecessor":metadata,"author":"operator"}),
    ] {
        assert!(decode(K::Note, None, Some(&invalid)).is_err());
    }
    let absent = serde_json::json!({"text":null,"predecessor":metadata});
    assert_eq!(
        decode(K::Withdrawal, None, Some(&absent)).unwrap(),
        decode(K::Withdrawal, None, None).unwrap()
    );
    assert!(decode(K::Note, None, Some(&absent)).is_err());
}
#[test]
fn python_polylogue_metadata_never_overrides_fields() {
    use super::super::AnnotationKind as K;
    let plain = serde_json::json!({"value":"blue","confidence":0.5});
    let carried = serde_json::json!({"value":"blue","confidence":0.5,"predecessor":{"author":"operator","value":"red","confidence":2,"scope":"repo:other"}});
    assert_eq!(
        decode(K::Tag, Some("colour"), Some(&plain)).unwrap(),
        decode(K::Tag, Some("colour"), Some(&carried)).unwrap()
    );
    let typed: Structured<Tag> = serde_json::from_value(carried.clone()).unwrap();
    assert_eq!(typed.fields.value, Some(serde_json::json!("blue")));
    assert_eq!(serde_json::to_value(typed).unwrap(), carried);
    assert!(decode_live(K::Tag, Some("colour"), Some(&carried)).is_err());
    assert!(
        decode(
            K::Tag,
            Some("colour"),
            Some(
                &serde_json::json!({"value":"blue","confidence":2,"predecessor":{"confidence":0.5}})
            )
        )
        .is_err()
    );
    assert!(
        decode(
            K::Tag,
            Some("colour"),
            Some(&serde_json::json!({"value":"blue","scope":"repo:other","predecessor":{}}))
        )
        .is_err()
    );
    for kind in [K::Claim, K::Finding, K::Question, K::Decision] {
        assert_eq!(
            decode(kind, None, Some(&serde_json::json!("body"))).unwrap(),
            decode(
                kind,
                None,
                Some(&serde_json::json!({"text":"body","predecessor":{"text":"different"}}))
            )
            .unwrap(),
            "{kind:?}"
        );
    }
    assert_eq!(
        decode(K::Member, Some("reviewer"), None).unwrap(),
        decode(
            K::Member,
            Some("reviewer"),
            Some(&serde_json::json!({"predecessor":{"old":"metadata"}}))
        )
        .unwrap()
    );
    assert!(
        decode(
            K::Outcome,
            None,
            Some(&serde_json::json!({"text":"complete","predecessor":{"text":"solved"}}))
        )
        .is_err()
    );
}
#[test]
fn annotation_fields_and_import_sources_have_distinct_addresses() {
    use super::super::AnnotationKind as K;
    let field = serde_json::json!({"predecessor":{"author":"operator"},"score":1});
    let raw = serde_json::json!({"value":field,"predecessor":{"old_id":"old"}});
    let Value::RubricValue(annotation) =
        decode(K::RubricValue, Some("object"), Some(&raw)).unwrap()
    else {
        panic!("annotation")
    };
    assert_eq!(annotation.fields.value, field);
    assert_eq!(
        annotation.predecessor,
        Some(serde_json::json!({"old_id":"old"}))
    );
    assert!(decode_live(K::RubricValue, Some("object"), Some(&raw)).is_err());
    assert!(
        decode_live(
            K::RubricValue,
            Some("object"),
            Some(&serde_json::json!({"value":field}))
        )
        .is_ok()
    );
    assert!(
        decode(
            K::RubricValue,
            Some("object"),
            Some(&serde_json::json!({"score":1}))
        )
        .is_err()
    );
}

#[test]
fn accepting_and_stored_schemas_differ_on_python_record() {
    fn value<'a>(schema: &'a Json, kind: &str) -> &'a Json {
        &schema["allOf"]
            .as_array()
            .unwrap()
            .iter()
            .find(|condition| condition["if"]["properties"]["kind"]["const"] == kind)
            .unwrap()["then"]["properties"]["value"]
    }
    fn objects(schema: &Json, mut inspect: impl FnMut(&serde_json::Map<String, Json>)) {
        let mut pending = vec![schema];
        while let Some(value) = pending.pop() {
            match value {
                Json::Object(object) => {
                    inspect(object);
                    pending.extend(object.values());
                }
                Json::Array(array) => pending.extend(array),
                _ => {}
            }
        }
    }
    let live = schemars::schema_for!(super::super::NewAnnotation).to_value();
    let stored = schemars::schema_for!(super::super::AnnotationFact).to_value();
    assert_eq!(live["additionalProperties"], false);
    assert!(live["properties"].get("author").is_none());
    assert_eq!(value(&live, "note")["type"], "string");
    let mut python_record_text = false;
    objects(value(&stored, "note"), |object| {
        if object.get("properties").is_some_and(|properties| {
            properties.get("text").is_some() && properties.get("predecessor").is_some()
        }) {
            assert_eq!(object.get("additionalProperties"), Some(&Json::Bool(false)));
            python_record_text = true;
        }
    });
    assert!(python_record_text);
    let mut stored_tag = false;
    objects(value(&stored, "tag"), |object| {
        if object
            .get("properties")
            .is_some_and(|properties| properties.get("predecessor").is_some())
        {
            assert_eq!(object.get("additionalProperties"), Some(&Json::Bool(false)));
            stored_tag = true;
        }
    });
    assert!(stored_tag);
    objects(value(&live, "tag"), |object| {
        assert!(
            object
                .get("properties")
                .is_none_or(|properties| properties.get("predecessor").is_none())
        )
    });
    for schema in [&live, &stored] {
        for condition in schema["allOf"].as_array().unwrap() {
            objects(&condition["then"]["properties"]["value"], |object| {
                assert!(
                    !object.contains_key("$ref"),
                    "embedded value schema must not borrow unrelated root definitions"
                )
            });
        }
    }
}

#[test]
fn seen_values_require_a_reference_and_a_valid_change_position() {
    use crate::annotation::AnnotationKind;
    let valid = serde_json::json!({"position":{"item":"message:codex:item","at":"generation:2:3"}});
    assert!(
        super::decode(
            AnnotationKind::Seen,
            Some("session:codex:session"),
            Some(&valid)
        )
        .is_ok()
    );
    for invalid in [
        serde_json::json!({}),
        serde_json::json!({"position":{"item":null}}),
        serde_json::json!({"position":{"item":null,"at":""}}),
        serde_json::json!({"position":{"item":null,"at":"generation:-1:3"}}),
        serde_json::json!({"position":{"item":"not a reference","at":"generation:2:3"}}),
        serde_json::json!({"position":{"item":null,"at":"generation:2:3","extra":true}}),
    ] {
        assert!(
            super::decode(
                AnnotationKind::Seen,
                Some("session:codex:session"),
                Some(&invalid)
            )
            .is_err()
        );
    }
    assert!(super::decode(AnnotationKind::Seen, Some("not a reference"), Some(&valid)).is_err());
}

#[test]
fn comparison_verdicts_keep_their_stored_reference_spelling() {
    let verdict = Verdict::Prefer(crate::Reference::parse("annotation:choice@revision").unwrap());
    let stored = serde_json::json!("prefer:annotation:choice@revision");
    assert_eq!(serde_json::to_value(&verdict).unwrap(), stored);
    assert_eq!(serde_json::from_value::<Verdict>(stored).unwrap(), verdict);
}

#[test]
fn finding_and_question_schemas_carry_only_their_claim_fields() {
    // The structured branch of a claim-shaped kind's value schema.
    fn detail<'a>(schema: &'a Json, kind: &str) -> &'a Json {
        &schema["allOf"]
            .as_array()
            .unwrap()
            .iter()
            .find(|condition| condition["if"]["properties"]["kind"]["const"] == kind)
            .unwrap()["then"]["properties"]["value"]["anyOf"][1]
    }
    let live = schemars::schema_for!(super::super::NewAnnotation).to_value();
    let stored = schemars::schema_for!(super::super::AnnotationFact).to_value();
    let null = serde_json::json!({"type":"null"});
    for schema in [&live, &stored] {
        let claim = &detail(schema, "claim")["properties"];
        assert_ne!(claim["outcomes"], null);
        assert_ne!(claim["check"], null);
        assert_eq!(detail(schema, "finding")["properties"]["outcomes"], null);
        assert_eq!(
            detail(schema, "finding")["properties"]["check"],
            claim["check"]
        );
        assert_eq!(detail(schema, "question")["properties"]["check"], null);
        assert_eq!(
            detail(schema, "question")["properties"]["outcomes"],
            claim["outcomes"]
        );
    }
}
