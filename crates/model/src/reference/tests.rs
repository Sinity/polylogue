use super::*;

#[test]
fn exact_question_identity_excludes_names_prefixes_and_parameterized_queries() {
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
        "attachment:blob:1",
        "assertion:root",
        "assertion:root@revision",
        "variant:root",
        "variant:root@revision",
        "variant:root@revision#3",
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
        "sx:material:opaque",
        "span:message:codex:message#b=0-4",
        "span:document:claude-ai:document#u16=1-5",
        "span:block:codex:message#2#c=0-3",
        "span:assertion:root@revision#u16=0-5",
        "span:variant:root@revision#3#b=0-4",
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
    for text in [
        "work:7",
        "note:old",
        "variant:root#-1",
        "variant:root@",
        "role:",
        "role:root@",
        "result:id#x",
        "block:codex:m#18446744073709551616",
        "span:session:codex:s#b=0-2",
        "span:assertion:root#b=0-2",
        "span:variant:root@revision#b=0-2",
        "span:message:codex:m#b=5-1",
        "span:message:codex:m#bytes=1-2",
        "file:path",
        "url:/relative",
        "query:name(p=1,p=2)",
    ] {
        assert!(Reference::parse(text).is_err(), "{text}");
    }
}

#[test]
fn containers_preserve_exact_identity_without_guessing_message_sessions() {
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
        ("variant:root@revision#3", "variant:root@revision"),
        ("result:kept#0", "result:kept"),
        ("tool:codex:session#call", "session:codex:session"),
    ] {
        assert_eq!(parse(child).container(), Some(parse(parent)));
    }
    assert_eq!(parse("assertion:root@revision").container(), None);
}

#[test]
fn run_references_are_explicit_native_identities() {
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
        "assertion:018f.1",
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
fn exact_accepted_request_reference_preserves_the_existing_opaque_identity() {
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
