use super::*;

fn fixture(native: &str, uuid: &str, text: &str) -> Vec<u8> {
    [
        json!({"type":"ai-title","aiTitle":"Synthetic reference guard"}).to_string(),
        json!({"type":"user","uuid":uuid,"sessionId":native,"promptSource":"typed",
            "timestamp":"2026-10-01T08:00:00Z","message":{"role":"user","content":text}})
        .to_string(),
        String::new(),
    ]
    .join("\n")
    .into_bytes()
}

#[test]
fn short_native_word_in_foreign_prose_is_not_a_full_session_reference() {
    let world = World::new();
    let own = world.import_as(&fixture("a", "own", "Own text"), "own-a", "a");
    world.wait_rows(&own, false);
    let ordinary = world.import_as(
        &fixture(
            "foreign-ordinary",
            "ordinary",
            "A marker contains a character",
        ),
        "ordinary",
        "foreign-ordinary",
    );
    world.wait_rows(&ordinary, false);
    let id = SessionId::new(&Origin::new("claude-code").unwrap(), "a").unwrap();
    let document = || {
        let view = world.reader.selected_session(&id, 65536).unwrap().unwrap();
        let mut bytes = Vec::new();
        write(&view, &mut bytes, Instant::now() + WAIT).unwrap();
        serde_json::from_slice::<Value>(&bytes).unwrap()
    };
    let clean = document();
    assert_eq!(clean["result"]["outcome"], "ok");
    assert_eq!(
        clean["result"]["completion"]["computed_from"]["inputs"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let mention = world.import_as(
        &fixture(
            "foreign-reference",
            "mention",
            "See claude-code:a for the source",
        ),
        "mention",
        "foreign-reference",
    );
    world.wait_rows(&mention, false);
    let affected = document();
    assert_eq!(affected["result"]["outcome"], "degraded");
    let basis = affected["result"]["completion"]["computed_from"]["inputs"]
        .as_array()
        .unwrap();
    assert!(
        basis
            .iter()
            .any(|input| input["input"] == mention.id.as_str())
    );
    assert!(
        !basis
            .iter()
            .any(|input| input["input"] == ordinary.id.as_str())
    );
    assert!(
        affected["result"]["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|note| note.as_str().unwrap().contains("Session links"))
    );
}
