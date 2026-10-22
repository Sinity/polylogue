use super::*;
use crate::encoding::pack_text;

fn seed(a: &rusqlite::Connection, id: &str, refs: &[&str], text: &str) {
    a.execute("INSERT INTO delivery(id,recipient_kind,sender,event,at,content_hash,bytes,body,refs_json,commit_seq)
        VALUES(?1,'operator','cli','recall',1,?2,?3,?4,?5,1)",
        params![id,blake3::hash(text.as_bytes()).as_bytes().as_slice(),text.len() as i64,pack_text(text).unwrap(),serde_json::to_string(refs).unwrap()]).unwrap();
}

#[test]
fn delivery_redaction_uses_exact_references_and_preserves_preview_and_hashes() {
    let mut a = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&a).unwrap();
    let text = "before long private phrase extended after";
    for (id, refs) in [
        ("", vec!["session:claude-code:gone"]),
        ("\0id", vec!["claude-code:gone"]),
        ("message", vec!["message:opaque:id:with:colons"]),
        ("neighbor", vec!["session:claude-code:gone-too"]),
        ("unrelated", vec!["message:unrelated:claude-code:gone"]),
    ] {
        seed(&a, id, &refs, text);
    }
    let mut texts = SessionTexts::default();
    texts.insert("long private phrase extended").unwrap();
    texts.insert("long private phrase").unwrap();
    let ids = vec!["claude-code:gone".into()];
    let messages = vec!["opaque:id:with:colons".into()];
    let tx = a.transaction().unwrap();
    assert_eq!(
        redact_deliveries(&tx, &ids, &messages, &texts, true).unwrap(),
        3
    );
    for id in ["", "\0id", "message", "neighbor", "unrelated"] {
        let body: Vec<u8> = tx
            .query_row("SELECT body FROM delivery WHERE id=?1", [id], |r| r.get(0))
            .unwrap();
        assert_eq!(unpack_text(&body).unwrap(), text);
    }
    assert_eq!(
        redact_deliveries(&tx, &ids, &messages, &texts, false).unwrap(),
        3
    );
    let expected = format!(
        "before {} extended after",
        " ".repeat("long private phrase".len())
    );
    for id in ["", "\0id", "message", "neighbor", "unrelated"] {
        let (body, hash): (Vec<u8>, Vec<u8>) = tx
            .query_row(
                "SELECT body,content_hash FROM delivery WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        let actual = unpack_text(&body).unwrap();
        assert_eq!(
            actual,
            if ["neighbor", "unrelated"].contains(&id) {
                text
            } else {
                &expected
            }
        );
        assert_eq!(hash, blake3::hash(actual.as_bytes()).as_bytes());
    }
}

#[test]
fn unrelated_unreadable_delivery_is_not_decoded() {
    let mut a = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&a).unwrap();
    seed(
        &a,
        "neighbor",
        &["session:claude-code:gone-too"],
        "long private phrase",
    );
    a.execute("UPDATE delivery SET body=x'010203'", []).unwrap();
    let mut texts = SessionTexts::default();
    texts.insert("long private phrase").unwrap();
    let tx = a.transaction().unwrap();
    assert_eq!(
        redact_deliveries(&tx, &["claude-code:gone".into()], &[], &texts, false).unwrap(),
        0
    );
    tx.execute(
        "UPDATE delivery SET refs_json='[\"session:claude-code:gone\"]'",
        [],
    )
    .unwrap();
    assert!(redact_deliveries(&tx, &["claude-code:gone".into()], &[], &texts, false).is_err());
}
