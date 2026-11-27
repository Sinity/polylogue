use super::*;

#[test]
fn question_identity_is_only_a_full_query_hash() {
    let hash = crate::BlobHash([0xab; 32]);
    assert_eq!(
        Reference::parse(&format!("query:{hash}"))
            .unwrap()
            .question_id(),
        Some(query_language::identity::QuestionId(hash.0))
    );
    for reference in [
        "query:recent".to_owned(),
        "query:ab".to_owned(),
        format!("query:{}", hash.to_string().to_uppercase()),
        format!("query:{hash}(repo=morrow)"),
    ] {
        assert_eq!(
            Reference::parse(&reference).unwrap().question_id(),
            None,
            "{reference}"
        );
    }
}

#[test]
fn reference_full_forms_round_trip_without_losing_source_names() {
    for text in [
        "session:codex:thread:opaque",
        "message:codex:message:opaque",
        "message:codex:first..codex:last",
        "block:codex:message#2",
        "tool:codex:thread#call:opaque",
        "document:codex:document",
        "event:origin:event:1",
        "run:019a2f20-1234-7000-8000-0123456789ab",
        "attachment:codex:message#2",
        "attachment:codex:message#2.1",
        "annotation:root",
        "annotation:root@revision",
        "alternate:root",
        "alternate:root@revision",
        "alternate:root@revision#3",
        "page:name",
        "page:root@revision",
        "role:reviewer",
        "role:root@revision",
        "query:abc123",
        "query:work(p=value,q=other)",
        "result:opaque",
        "result:opaque#0",
        "delivery:opaque:1",
        "file:src/file.rs@abc123",
        "repo:project",
        "cwd:/home/op/project",
        "url:https://example.test/a?q=1#part",
        "sinex:ev/019a2f20-1234-7000-8000-0123456789ab",
        "span:message:codex:message#b=0-4",
        "span:document:claude-ai:document#u16=1-5",
        "span:block:codex:message#2#c=0-3",
        "span:tool:codex:session#call#u16=1-3",
        "span:annotation:root@revision#u16=0-5",
        "span:alternate:root@revision#3#b=0-4",
    ] {
        let reference = Reference::parse(text).unwrap_or_else(|error| panic!("{text}: {error}"));
        assert_eq!(reference.to_string(), text);
        assert_eq!(
            serde_json::from_str::<Reference>(&serde_json::to_string(&reference).unwrap()).unwrap(),
            reference
        );
    }
}
#[test]
fn malformed_references_are_typed_errors() {
    const SPAN_CONTAINER: &str = "span container must be a message, document, block, tool call, exact annotation or exact alternate part";
    let invalid = |text: &str, why| ReferenceError::Invalid(text.into(), why);
    let incomplete = |text: &str| ReferenceError::Incomplete(text.into());
    for (text, expected) in [
        ("work:7", ReferenceError::UnknownKind("work:7".into())),
        ("note:old", ReferenceError::UnknownKind("note:old".into())),
        (
            "alternate:root#-1",
            invalid("alternate:root#-1", "expected a nonnegative integer"),
        ),
        ("alternate:root@", incomplete("alternate:root@")),
        ("role:", incomplete("role:")),
        ("role:root@", incomplete("role:root@")),
        (
            "result:id#x",
            invalid("result:id#x", "expected a nonnegative integer"),
        ),
        (
            "block:codex:m#18446744073709551616",
            invalid(
                "block:codex:m#18446744073709551616",
                "integer is out of range",
            ),
        ),
        (
            "span:session:codex:s#b=0-2",
            invalid("span:session:codex:s#b=0-2", SPAN_CONTAINER),
        ),
        (
            "span:annotation:root#b=0-2",
            invalid("span:annotation:root#b=0-2", SPAN_CONTAINER),
        ),
        (
            "span:alternate:root@revision#b=0-2",
            invalid("span:alternate:root@revision#b=0-2", SPAN_CONTAINER),
        ),
        (
            "span:message:codex:m#b=5-1",
            invalid("span:message:codex:m#b=5-1", "span end precedes its start"),
        ),
        (
            "span:message:codex:m#bytes=1-2",
            invalid("span:message:codex:m#bytes=1-2", "unknown span unit"),
        ),
        ("file:path", incomplete("file:path")),
        (
            "url:/relative",
            invalid("url:/relative", "URL must be absolute"),
        ),
        (
            "query:name(p=1,p=2)",
            invalid("query:name(p=1,p=2)", "query parameter is repeated"),
        ),
    ] {
        assert_eq!(Reference::parse(text), Err(expected), "{text}");
    }
}

#[test]
fn containers_preserve_identity_without_guessing_message_sessions() {
    let parse = |text| Reference::parse(text).unwrap();
    let mut reference = parse("span:block:codex:message#9#u16=0-2");
    for expected in ["block:codex:message#9", "message:codex:message"] {
        reference = reference.container().unwrap();
        assert_eq!(reference, parse(expected));
    }
    assert_eq!(reference.container(), None);
    for (child, parent) in [
        (
            "span:document:claude-ai:document#b=1-2",
            "document:claude-ai:document",
        ),
        ("alternate:root@revision#3", "alternate:root@revision"),
        ("result:kept#0", "result:kept"),
        ("tool:codex:session#call", "session:codex:session"),
    ] {
        assert_eq!(parse(child).container(), Some(parse(parent)));
    }
    assert_eq!(parse("annotation:root@revision").container(), None);
}

#[test]
fn run_references_name_one_run_id() {
    let id = RunId::parse("019a2f20-1234-7000-8000-0123456789ab").unwrap();
    let reference = Reference::parse(&format!("run:{id}")).unwrap();
    assert_eq!(reference, Reference::Run(id));
    assert_eq!(reference.session(), None);
    for text in [
        "run:",
        "run:codex:session",
        "run:caller-key",
        "run:019a2f201234700080000123456789ab",
        "run:019A2F20-1234-7000-8000-0123456789AB",
        "run:019a2f20-1234-7000-8000-0123456789ab#0",
    ] {
        assert!(
            matches!(
                Reference::parse(text),
                Err(ReferenceError::Id(_, IdError::BadRun(_)))
            ),
            "{text}"
        );
    }
}

#[test]
fn authored_link_reference_forms_preserve_their_opaque_source_names() {
    for text in [
        "session:claude-code:abc",
        "message:codex:m:1",
        "tool:claude-code:abc#toolu_01",
        "result:018f",
        "annotation:018f.1",
        "repo:polylogue",
        "cwd:/workspace/project",
    ] {
        assert_eq!(Reference::parse(text).unwrap().to_string(), text);
    }
    assert!(matches!(
        Reference::parse("session"),
        Err(ReferenceError::NoKind(_))
    ));
    assert!(matches!(
        Reference::parse("tool:claude-code:abc"),
        Err(ReferenceError::Incomplete(_))
    ));
}

#[test]
fn accepted_request_reference_preserves_the_existing_opaque_identity() {
    for id in [
        "accepted-request",
        "a:b@revision#part..suffix",
        " spaces remain opaque ",
        "\0",
    ] {
        let text = format!("request:{id}");
        let reference = Reference::parse(&text).unwrap();
        assert_eq!(reference, Reference::Request(id.into()));
        assert_eq!(reference.to_string(), text);
        assert_eq!(
            serde_json::from_str::<Reference>(&serde_json::to_string(&reference).unwrap()).unwrap(),
            reference
        );
        assert_eq!(reference.container(), None);
    }
    assert!(Reference::parse("request:").is_err());
}

#[test]
fn sinex_global_reference_round_trips() {
    for text in [
        "sinex:record/machine/origin/type/0",
        "sinex:key/opaque%23key",
        "sinex:future-kind/opaque:identity",
    ] {
        let reference = Reference::parse(text).unwrap();
        let Reference::Sinex(object) = &reference else {
            panic!("not a Sinex object")
        };
        assert_eq!(object.product(), references::Product::Sinex);
        assert_eq!(reference.to_string(), text);
        assert_eq!(
            serde_json::to_string(&reference).unwrap(),
            serde_json::to_string(text).unwrap()
        );
        assert_eq!(
            serde_json::from_str::<Reference>(&serde_json::to_string(text).unwrap()).unwrap(),
            reference
        );
    }
}

#[test]
fn sinex_local_form_refused_with_global() {
    for kind in SINEX_LOCAL_KINDS {
        let text = format!("{kind}:opaque");
        assert_eq!(
            Reference::parse(&text).unwrap_err(),
            ReferenceError::SinexLocal {
                text: text.clone(),
                global: format!("sinex:{kind}/opaque")
            },
        );
        assert_eq!(
            Reference::parse(&text).unwrap_err().to_string(),
            format!("{text} is a Sinex reference; write sinex:{kind}/opaque")
        );
    }
}

#[test]
fn sinex_reference_with_selector_refused() {
    for text in [
        "sinex:ev/opaque#c=0-4",
        "sinex:ev/opaque@revision",
        "sinex:ev/",
        "sinex:ev",
    ] {
        assert!(
            matches!(Reference::parse(text), Err(ReferenceError::Invalid(_, _))),
            "{text}"
        );
    }
}

#[test]
fn sinex_read_names_its_reader() {
    let reference = Reference::parse("sinex:ev/opaque").unwrap();
    let error = crate::read::ReadTarget::try_from(reference).unwrap_err();
    assert_eq!(error.reference, "sinex:ev/opaque");
    assert_eq!(
        error.reason,
        "reading a Sinex object needs the Sinex reader"
    );
}

#[test]
fn unknown_reference_prefix_is_refused() {
    let text = "sx:ev/opaque";
    assert_eq!(
        Reference::parse(text),
        Err(ReferenceError::UnknownKind(text.into()))
    );
}

#[test]
fn authored_references_round_trip_through_text_and_serde() {
    for text in [
        "annotation:root",
        "annotation:root@revision",
        "alternate:root@revision#2",
        "span:annotation:root@revision#c=0-3",
        "span:alternate:root@revision#2#b=0-3",
        "url:https://example.test/annotation:kept",
    ] {
        let reference = Reference::parse(text).unwrap();
        assert_eq!(reference.to_string(), text);
        assert_eq!(
            serde_json::to_value(&reference).unwrap(),
            serde_json::json!(text)
        );
        assert_eq!(
            serde_json::from_value::<Reference>(serde_json::json!(text)).unwrap(),
            reference
        );
    }
}

#[test]
fn read_targets_refuse_what_has_no_selected_row_reading() {
    use crate::read::ReadTarget;
    let session = || crate::SessionId::parse("test:one").unwrap();
    let message = || crate::MessageId::parse("test:one:m1").unwrap();
    for (text, expected) in [
        ("session:test:one", ReadTarget::Session(session())),
        ("message:test:one:m1", ReadTarget::Message(message())),
        (
            "block:test:one:m1#0",
            ReadTarget::Block {
                message: message(),
                n: 0,
            },
        ),
        (
            "annotation:root@revision",
            ReadTarget::Annotation {
                root: "root".into(),
                revision: "revision".into(),
            },
        ),
    ] {
        assert_eq!(
            ReadTarget::try_from(Reference::parse(text).unwrap()),
            Ok(expected),
            "{text}"
        );
    }
    for (text, reason) in [
        (
            "annotation:root",
            "an annotation family is read through its current revision",
        ),
        (
            "span:block:test:one:m1#0#u16=0-2",
            "a span addresses text inside a record; read its container",
        ),
        ("tool:test:one#call", "it has no selected-row reading"),
    ] {
        let refused = ReadTarget::try_from(Reference::parse(text).unwrap()).unwrap_err();
        assert_eq!(refused.reason, reason, "{text}");
    }
}
