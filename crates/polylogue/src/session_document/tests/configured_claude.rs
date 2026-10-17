use super::*;

#[test]
fn configured_claude_separate_agent_has_provider_owned_membership() {
    let world = World::new();
    let native = "actual-owner/a";
    let bytes = (json!({"type":"user","uuid":"u","sessionId":"actual-owner","promptSource":"typed","message":{"role":"user","content":"Separate agent independent text."}}).to_string()+"\n").into_bytes();
    let input = world.import_context(
        &bytes,
        "separate-agent",
        vec![
            "claude-subagent".into(),
            "a".into(),
            "folder-lineage-only".into(),
        ],
    );
    world.wait_rows(&input, true);
    let id = SessionId::new(&Origin::new("claude-code").unwrap(), native).unwrap();
    let view = world.reader.selected_session(&id, 65536).unwrap().unwrap();
    let qualification =
        polylogue_engine::session_qualification::qualify(&view, Instant::now() + WAIT, 8000)
            .unwrap();
    assert!(qualification.known_rows_readable(), "{qualification:?}");
    assert!(
        !qualification.complete(),
        "lineage products remain qualified separately"
    );
    assert_eq!(qualification.expected.len(), 1);
    assert_eq!(qualification.inputs.len(), 1);
    assert_eq!(qualification.basis[0].input, input.id);
    let mut bytes = vec![];
    write(&view, &mut bytes, Instant::now() + WAIT).unwrap();
    let document: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(document["session"]["id"], id.as_str());
    assert_eq!(
        document["messages"][0]["blocks"][0]["text"],
        "Separate agent independent text."
    );
    assert!(
        world
            .reader
            .selected_session(
                &SessionId::new(
                    &Origin::new("claude-code").unwrap(),
                    "folder-lineage-only/a"
                )
                .unwrap(),
                65536
            )
            .unwrap()
            .is_none()
    );
}

#[test]
fn configured_claude_exact_read_does_not_charge_streamed_foreign_population_as_live_metadata() {
    let world = World::new();
    let bytes = (json!({"type":"user","uuid":"u","sessionId":SID,"promptSource":"typed","message":{"role":"user","content":"Retained exact target."}}).to_string()+"\n").into_bytes();
    let own = world.import(&bytes, "own-target");
    world.wait_rows(&own, false);
    for i in 0..40 {
        let native = format!("foreign-{i:03}");
        let bytes = (json!({"type":"user","uuid":"u","sessionId":native,"promptSource":"typed","message":{"role":"user","content":"Independent foreign conversation."}}).to_string()+"\n").into_bytes();
        let input = world.import_as(&bytes, &format!("foreign-{i:03}"), &native);
        world.wait_rows(&input, false);
    }
    let view = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let qualification =
        polylogue_engine::session_qualification::qualify(&view, Instant::now() + WAIT, 8000)
            .unwrap();
    assert!(qualification.known_rows_readable(), "{qualification:?}");
    assert_eq!(qualification.basis.len(), 1);
    assert_eq!(qualification.basis[0].input, own.id);
}

#[test]
fn configured_claude_foreign_partial_main_does_not_make_requested_membership_pending() {
    let world = World::new();
    let bytes = (json!({"type":"user","uuid":"u","sessionId":SID,"promptSource":"typed","message":{"role":"user","content":"Independent readable main."}}).to_string()+"\n").into_bytes();
    let own = world.import(&bytes, "readable-main");
    world.wait_rows(&own, false);
    let mut partial = (json!({"type":"user","uuid":"f","sessionId":"other-main","promptSource":"typed","message":{"role":"user","content":"Other main has an unfinished suffix."}}).to_string()+"\n").into_bytes();
    partial.extend_from_slice(b"{\"type\":\"user\",\"uuid\":\"later\"");
    let other = world.import_as(&partial, "partial-foreign", "other-main");
    let until = Instant::now() + WAIT;
    loop {
        if world
            .progress(&other)
            .parser
            .is_some_and(|p| p.status == "waiting_tail")
        {
            break;
        }
        assert!(Instant::now() < until);
        thread::sleep(POLL);
    }
    let view = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let qualification =
        polylogue_engine::session_qualification::qualify(&view, Instant::now() + WAIT, 8000)
            .unwrap();
    assert!(qualification.known_rows_readable(), "{qualification:?}");
    assert!(!qualification.complete());
    let result = qualification.result(&view.catalog());
    assert_eq!(result["unknown"]["population_pending_inputs"], 1);
    assert!(result["unknown"].get("pending_inputs").is_none());
    assert_eq!(qualification.expected.len(), 1);
}

#[test]
fn configured_claude_observed_member_is_independent_of_incoming_agent_lineage() {
    let mut world = World::new();
    world.path = world._dir.path().join(format!("{SID}.jsonl"));
    let mut bytes = (json!({"type":"user","uuid":"u","sessionId":SID,"promptSource":"typed","message":{"role":"user","content":"Earlier certified main text."}}).to_string()+"\n").into_bytes();
    bytes.extend_from_slice(b"{\"type\":\"user\",\"uuid\":\"unfinished\"");
    fs::write(&world.path, bytes).unwrap();
    let observation = Observation {
        id: "observed-main".into(),
        machine: "synthetic-machine".into(),
        host: "synthetic-host".into(),
        path: world.path.to_str().unwrap().into(),
        seen_at: 0,
    };
    let input = world
        .owner
        .as_ref()
        .unwrap()
        .handle()
        .try_claude_acquisition(world.path.clone(), observation, 65536)
        .unwrap()
        .recv_timeout(WAIT)
        .unwrap()
        .unwrap()
        .input;
    world.wait_selected(&input, &["u"]);
    let until = Instant::now() + WAIT;
    while !world
        .progress(&input)
        .parser
        .is_some_and(|p| p.status == "waiting_tail")
    {
        assert!(Instant::now() < until);
        thread::sleep(POLL);
    }
    world.path = world._dir.path().join("agent.jsonl");
    let agent=(json!({"type":"user","uuid":"a","sessionId":SID,"promptSource":"typed","message":{"role":"user","content":"Separate agent text."}}).to_string()+"\n").into_bytes();
    let agent = world.import_context(
        &agent,
        "incoming-agent",
        vec!["claude-subagent".into(), "a".into(), SID.into()],
    );
    world.wait_rows(&agent, true);
    fn qualify(
        world: &World,
    ) -> (
        polylogue_engine::session_qualification::Qualification,
        Value,
    ) {
        let view = world
            .reader
            .selected_session(&world.session(), 65536)
            .unwrap()
            .unwrap();
        let polylogue_store::contributions::SessionStep::Ready { message, .. } =
            view.next(None, 65536).unwrap()
        else {
            panic!("expected selected main message")
        };
        let q = polylogue_engine::session_qualification::qualify_member_with_cancel(
            &view,
            &message,
            Instant::now() + WAIT,
            8000,
            &mut || false,
        )
        .unwrap();
        let result = q.result(&view.catalog());
        (q, result)
    }
    let (q, result) = qualify(&world);
    assert_eq!(result["settlement"]["target_basis"]["inputs_total"], 2);
    assert!(
        !q.complete(),
        "incoming relationship and partial suffix remain qualified"
    );
    assert!(q.member_rows_readable(), "{q:?}");
    // A second actual source of main membership must still deny the observed
    // exception, even if a consumer omits that source from display metadata.
    world.path = world._dir.path().join("second-main.jsonl");
    let second=(json!({"type":"user","uuid":"v","sessionId":SID,"promptSource":"typed","message":{"role":"user","content":"Second main membership source."}}).to_string()+"\n").into_bytes();
    let second = world.import(&second, "second-main");
    world.wait_rows(&second, false);
    let (mut q, result) = qualify(&world);
    assert_eq!(result["settlement"]["target_basis"]["inputs_total"], 3);
    q.basis.truncate(1);
    assert!(
        !q.member_rows_readable(),
        "a display sample must not grant membership permission: {q:?}"
    );
}
