//! References as conditions: `block:` and `note:` name one thing.

use super::*;

#[test]
fn a_block_reference_names_its_block_message_and_session() {
    let w = morrow();
    let blocks = q(&w, "blocks where block:m2p0:0");
    assert_eq!(blocks["count"], 1, "{blocks}");
    assert_eq!(ids(&q(&w, "messages where block:m2p0:0")).len(), 1);
    assert_eq!(ids(&q(&w, "block:m2p0:0")), ["#2"]);
    let bad = run_q(&w, ask("block:m2p0")).unwrap_err();
    assert_eq!(bad.kind, "invalid_query");
}

#[test]
fn a_note_reference_names_the_current_revision_or_one_revision() {
    let w = morrow();
    let first = q(&w, "notes where kind:decision");
    let id = first["rows"][0]["id"].as_str().unwrap().to_owned();
    let root = first["rows"][0]["first_note"].as_str().unwrap().to_owned();
    let by_root = q(&w, &format!("notes where note:{root}"));
    assert_eq!(by_root["rows"][0]["id"], id, "{by_root}");
    let by_revision = q(&w, &format!("notes where note:{root}@{id}"));
    assert_eq!(by_revision["count"], 1);
    let elsewhere = run_q(&w, ask(&format!("note:{root}"))).unwrap_err();
    assert_eq!(elsewhere.kind, "invalid_query");
    // Pages and commits are not built: refused by name, never empty.
    for later in ["page:plan", "file:src/a.rs@HEAD"] {
        assert_eq!(
            run_q(&w, ask(later)).unwrap_err().kind,
            "not_built",
            "{later}"
        );
    }
}
