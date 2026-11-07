//! References as conditions: `block:` and `assertion:` name one thing.

use super::*;

#[test]
fn a_block_reference_names_its_block_message_and_session() {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = polylogue_store::fixture::Fixture::new(dir.path()).unwrap();
    let session = fx
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    fx.message(
        session,
        MessageFacts {
            id: "test:message",
            role: "user",
            written_by: "human",
            prompt_kind: Some("prose"),
            text: Some("synthetic"),
            ..Default::default()
        },
    )
    .unwrap();
    let w = TestArchive { _dir: dir, fx };
    let blocks = q(&w, "blocks where block:test:message#0");
    assert_eq!(blocks["count"], 1, "{blocks}");
    assert_eq!(
        q(&w, "messages where block:test:message#0")["rows"][0]["id"],
        "test:message"
    );
    assert_eq!(ids(&q(&w, "block:test:message#0")), ["test:session"]);
    assert_eq!(q(&w, "messages where block:test:message#999")["count"], 0);
    let bad = run_q(&w, ask("block:test:message:0")).unwrap_err();
    assert_eq!(bad.code(), "invalid_query");
}

#[test]
fn an_assertion_reference_names_the_current_revision_or_one_revision() {
    let w = morrow();
    let first = q(&w, "assertions where kind:decision");
    let id = first["rows"][0]["id"].as_str().unwrap().to_owned();
    let root = first["rows"][0]["root"].as_str().unwrap().to_owned();
    let by_root = q(&w, &format!("assertions where assertion:{root}"));
    assert_eq!(by_root["rows"][0]["id"], id, "{by_root}");
    let by_revision = q(&w, &format!("assertions where assertion:{root}@{id}"));
    assert_eq!(by_revision["count"], 1);
    let elsewhere = run_q(&w, ask(&format!("assertion:{root}"))).unwrap_err();
    assert_eq!(elsewhere.code(), "not_built");
    assert_eq!(
        elsewhere.message,
        "this reference at the selected row grain is not built yet"
    );
    // Pages and commits are not built: refused by name, never empty.
    for later in ["page:plan", "file:src/a.rs@HEAD"] {
        assert_eq!(
            run_q(&w, ask(later)).unwrap_err().code(),
            "not_built",
            "{later}"
        );
    }
}
