use super::*;
use serde_json::json;
fn session() -> SessionId {
    SessionId::new(&polylogue_model::Origin::new("codex").unwrap(), "s").unwrap()
}
fn event(kind: &str, saw: bool, data: serde_json::Value) -> Event {
    Event {
        kind: kind.into(),
        at: None,
        after_message: None,
        model_saw: Some(saw),
        data: RawJson::parse(data.to_string().as_bytes()).unwrap(),
    }
}
#[test]
fn source_event_shapes_keep_their_paths() {
    let sid = session();
    let mut proof = FlatSessionEvents::new(&sid);
    for e in [
        event(
            "base_instructions",
            true,
            json!({"text":"Neutral instructions","provenance":{}}),
        ),
        event(
            "turn_start",
            false,
            json!({"type":"task_started","turn_id":"t","thread_id":"s"}),
        ),
        event(
            "world_state",
            true,
            json!({"full":true,"state":{"instructions":"Neutral"}}),
        ),
        event(
            "turn_context",
            false,
            json!({"turn_id":"t","model":"synthetic","session_id":"s"}),
        ),
        event(
            "rate_limit",
            false,
            json!({"primary":{"used_percent":1},"secondary":null,"credits":{}}),
        ),
        event(
            "turn_end",
            false,
            json!({"type":"task_complete","turn_id":"t"}),
        ),
        event("turn_context", false, json!({"turn_id":"later"})),
    ] {
        assert!(proof.accepts(&e), "{}", e.kind);
    }
}
#[test]
fn malformed_or_wrong_scope_events_of_the_same_kind_are_not_members() {
    let sid = session();
    for bad in [
        event("base_instructions", true, json!({"text":null})),
        event("base_instructions", false, json!("text")),
        event("world_state", true, json!({"full":null,"state":{}})),
        event("world_state", true, json!({"full":true,"state":[]})),
        event("rate_limit", false, json!([])),
        event("rate_limit", false, json!({"primary":"wrong"})),
        event("rate_limit", false, json!({"future_control":true})),
        event(
            "turn_start",
            false,
            json!({"type":"task_started","turn_id":null}),
        ),
        event(
            "turn_start",
            false,
            json!({"type":"thread_rolled_back","turn_id":"t"}),
        ),
        event(
            "turn_start",
            false,
            json!({"type":"task_started","turn_id":"t","thread_id":"foreign"}),
        ),
        event("turn_context", false, json!({"turn_id":"t","model":17})),
        event(
            "turn_context",
            false,
            json!({"turn_id":"t","session_id":null}),
        ),
        event(
            "turn_context",
            false,
            json!({"turn_id":"t","type":"future"}),
        ),
        event(
            "turn_end",
            false,
            json!({"type":"task_complete","turn_id":"t"}),
        ),
        event("compaction", true, json!({})),
        event("rollback", false, json!({"num_turns":1})),
        event("repeated_session_meta", false, json!({"id":"s"})),
        event("future", false, json!({})),
    ] {
        assert!(
            !FlatSessionEvents::new(&sid).accepts(&bad),
            "{} {:?}",
            bad.kind,
            bad.data
        );
    }
    let mut proof = FlatSessionEvents::new(&sid);
    assert!(proof.accepts(&event(
        "turn_start",
        false,
        json!({"type":"task_started","turn_id":"first"})
    )));
    assert!(!proof.accepts(&event("turn_context", false, json!({"turn_id":"other"}))));
    assert!(!proof.accepts(&event(
        "turn_end",
        false,
        json!({"type":"task_complete","turn_id":"other"})
    )));
    assert!(proof.accepts(&event(
        "turn_end",
        false,
        json!({"type":"task_complete","turn_id":"first"})
    )));
    let mut foreign = event("world_state", true, json!({"full":true,"state":{}}));
    foreign.after_message = Some(
        polylogue_model::MessageId::new(
            SessionId::new(&polylogue_model::Origin::new("codex").unwrap(), "other").unwrap(),
            "m",
        )
        .unwrap(),
    );
    assert!(!proof.accepts(&foreign));
}
#[test]
fn header_and_unit_codec_require_the_declared_event_family() {
    let bytes = serde_json::to_vec(&EffectsHeader {
        version: STATE_LAYOUT_VERSION,
        step: 3,
        counts: [1, 0, 0, 0, 0, 0, 0, 0],
    })
    .unwrap();
    let proof = FlatSessionStep::decode(&bytes, 3).unwrap().unwrap();
    assert_eq!(proof.events(), 1);
    assert_eq!(proof.event_key(0).unwrap(), effect_key(3, 1, 0));
    let e = event("rate_limit", false, json!({}));
    let encoded = serde_json::to_vec(&Effect::Event(e.clone())).unwrap();
    assert_eq!(proof.event(0, Some(&encoded)).unwrap(), Some(e));
    assert!(proof.event(0, None).is_err());
    assert!(proof.event(1, Some(&encoded)).is_err());
    assert!(proof.event(1, None).unwrap().is_none());
    assert!(proof.event_key(2).is_err());
    let wrong = serde_json::to_vec(&Effect::ReachedBack).unwrap();
    assert!(proof.event(0, Some(&wrong)).is_err());
    assert!(FlatSessionStep::decode(&bytes, 4).is_err());
    for index in [1, 3, 5] {
        let mut counts = [0; 8];
        counts[index] = 1;
        let bytes = serde_json::to_vec(&EffectsHeader {
            version: STATE_LAYOUT_VERSION,
            step: 3,
            counts,
        })
        .unwrap();
        assert!(FlatSessionStep::decode(&bytes, 3).unwrap().is_none());
    }
}
