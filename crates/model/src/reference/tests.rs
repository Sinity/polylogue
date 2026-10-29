use super::*;

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
        "attachment:blob:1",
        "assertion:root",
        "assertion:root@revision",
        "variant:root",
        "variant:root@revision",
        "variant:root@revision#3",
        "page:name",
        "page:root@revision",
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
