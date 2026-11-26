use super::*;

fn command(daemon: &crate::test_daemon::TestDaemon, query: &str, yes: bool) -> Answered<Answer> {
    let mut words = vec!["polylogue", "embed", "--query", query];
    if yes {
        words.push("--yes");
    }
    let (_, _, arguments) = crate::cli::request_arguments(&words).unwrap();
    let request = crate::client::request(
        "embed",
        Value::Object(arguments),
        uuid::Uuid::now_v7().to_string(),
    );
    crate::client::call_ready(&daemon.daemon.archive, &request)
        .unwrap()
        .map(|value| serde_json::from_value(value).unwrap())
}

#[test]
fn embed_mixed_queries_keep_each_winning_rows_scope_and_kept_results_stay_exact() {
    let seed = |home: &std::path::Path| {
        let project = home.join(".claude/projects/-p");
        let continued = serde_json::json!({
            "type": "continued-in", "sessionId": "pick",
            "continuedInSessionId": "tail", "timestamp": "2026-03-01T10:00:02Z"
        });
        std::fs::write(
            project.join("pick.jsonl"),
            format!(
                "{}{}\n",
                polylogue_testkit::claude_user("root-prose", "pick", "root prose"),
                continued
            ),
        )
        .unwrap();
        std::fs::write(
            project.join("tail.jsonl"),
            polylogue_testkit::claude_user("continued-prose", "tail", "continued prose"),
        )
        .unwrap();
        std::fs::write(project.join("qroot.jsonl"), format!("{}{}\n",
            polylogue_testkit::claude_user("physical-root", "qroot", "physical root prose"),
            json!({"type":"continued-in", "sessionId":"qroot", "continuedInSessionId":"qtail", "timestamp":"2026-03-01T10:00:02Z"})
        )).unwrap();
        std::fs::write(
            project.join("qtail.jsonl"),
            polylogue_testkit::claude_user("excluded-continuation", "qtail", "must stay outside"),
        )
        .unwrap();
    };
    let mut daemon =
        crate::test_daemon::start_with_sources_and_workers("unselected prose", &seed, 4, 2);
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    daemon.set("embed.provider", "local");
    daemon.set("embed.model", "synthetic-logical-command");
    daemon.set(
        "embed.endpoint",
        &format!("http://{}", listener.local_addr().unwrap()),
    );
    let logical = "sessions | sort by id | limit 1";
    let physical = "sessions where physical:true | sort by id | limit 1";
    let mixed = "sessions | union (sessions where physical:true) | sort by id | limit 1";
    let physical_first = "sessions where physical:true | union (sessions) | sort by id | limit 1";
    let nested = "sessions | union (sessions where physical:true) | fields id | union (sessions where physical:true | fields id) | sort by id | limit 1";
    let aliased = "sessions | union (sessions where physical:true) | sort by id | limit 1 | fields text(id) as label";
    let distinct =
        "sessions where \"continued prose\" | union (sessions where id:claude-code:qroot)";
    let saved = crate::client::request(
        "rows",
        json!({
            "selection": logical, "steps": [{"verb":"save", "name":"captured-root"}]
        }),
        uuid::Uuid::now_v7().to_string(),
    );
    let saved =
        crate::test_answers::ok(crate::client::call_ready(&daemon.daemon.archive, &saved).unwrap());
    let crate::protocol::line::LineAnswer::Steps(mut reports) =
        serde_json::from_value(saved).unwrap()
    else {
        panic!("save reports");
    };
    let crate::protocol::line::StepOutcome::Done {
        reply: crate::protocol::line::StepAnswer::Saved(saved),
        ..
    } = reports.steps.remove(0).outcome
    else {
        panic!("kept selection");
    };
    let kept = saved.kept.result.as_str();
    daemon.restart();
    for (query, expected) in [
        (logical, 2),
        (physical, 1),
        (mixed, 2),
        (physical_first, 1),
        (nested, 2),
        (aliased, 2),
        (kept, 1),
        (distinct, 3),
    ] {
        let Answer::Preview(preview) =
            crate::protocol::into_value(command(&daemon, query, false)).unwrap()
        else {
            panic!("preview for {query}");
        };
        assert_eq!(
            (preview.texts, preview.chunks),
            (expected, expected),
            "{query}"
        );
    }
    daemon.set("embed.calls", "on");
    let server = std::thread::spawn(move || {
        let (peer, _) = listener.accept().unwrap();
        crate::test_daemon::embeddings::serve_request(
            peer,
            200,
            r#"{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,1]},{"index":2,"embedding":[1,1]}]}"#,
        )
    });
    let Answer::Built(built) =
        crate::protocol::into_value(command(&daemon, distinct, true)).unwrap()
    else {
        panic!("mixed purchase");
    };
    assert_eq!((built.texts, built.vectors, built.failed), (3, 3, 0));
    let sent = server.join().unwrap();
    let mut inputs: Vec<_> = sent["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    inputs.sort_unstable();
    assert_eq!(
        inputs,
        ["continued prose", "physical root prose", "root prose"]
    );
    // Every selected logical member is now complete; the closed provider is
    // not needed for either another logical build or its physical subset.
    for query in [logical, physical, mixed, physical_first, kept, distinct] {
        let Answer::Built(built) =
            crate::protocol::into_value(command(&daemon, query, true)).unwrap()
        else {
            panic!("repeat for {query}");
        };
        assert_eq!((built.texts, built.vectors, built.failed), (0, 0, 0));
    }
}

#[test]
fn embed_query_bound_selects_all_texts_and_chunks_of_physical_rows_without_preview_calls() {
    let long = "x".repeat(CHUNK_CHARS + 4);
    let selected = "second selected";
    let seed = |home: &std::path::Path| {
        let path = home.join(".claude/projects/-p/pick.jsonl");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            path,
            format!(
                "{}{}",
                polylogue_testkit::claude_user("pick-user", "pick", &long),
                polylogue_testkit::claude_assistant("pick-assistant", "pick", selected),
            ),
        )
        .unwrap();
    };
    let daemon = crate::test_daemon::start_with_sources_and_workers("not selected", &seed, 2, 2);
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    daemon.set("embed.provider", "local");
    daemon.set("embed.model", "synthetic-command");
    daemon.set(
        "embed.endpoint",
        &format!("http://{}", listener.local_addr().unwrap()),
    );
    daemon.set("embed.price_per_mtok", "2");
    let query = "sessions where physical:true | sort by id | limit 1";
    let (accepted, requests) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        let (peer, _) = listener.accept().unwrap();
        accepted.send(()).unwrap();
        crate::test_daemon::embeddings::serve_request(
            peer,
            200,
            r#"{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,1]},{"index":2,"embedding":[1,1]}]}"#,
        )
    });
    let preview = crate::protocol::into_value(command(&daemon, query, false)).unwrap();
    let Answer::Preview(preview) = preview else {
        panic!("without --yes embed only previews");
    };
    assert_eq!(
        (
            preview.texts,
            preview.chunks,
            preview.characters,
            preview.tokens_estimate
        ),
        (2, 3, 6019, 1505)
    );
    assert_eq!(preview.cost_estimate, Some(Amount(3_010_000)));
    assert!(matches!(
        requests.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    let off = command(&daemon, query, true);
    assert_eq!(
        off.failure().unwrap().code.as_deref(),
        Some("model_calls_off")
    );
    assert!(matches!(
        requests.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    daemon.set("embed.calls", "on");
    let preview = crate::protocol::into_value(command(&daemon, query, false)).unwrap();
    assert!(matches!(preview, Answer::Preview(_)));
    assert!(matches!(
        requests.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    let built = crate::protocol::into_value(command(&daemon, query, true)).unwrap();
    let Answer::Built(built) = built else {
        panic!("--yes builds selected text");
    };
    assert_eq!((built.texts, built.vectors, built.failed), (2, 3, 0));
    requests.recv().unwrap();
    let sent = server.join().unwrap();
    let mut inputs: Vec<&str> = sent["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    inputs.sort_unstable();
    let chunk = "x".repeat(CHUNK_CHARS);
    let mut expected = vec![selected, chunk.as_str(), "xxxx"];
    expected.sort_unstable();
    assert_eq!(inputs, expected);
    // The mock endpoint is closed. A repeat or an empty query must not call it.
    let repeated = crate::protocol::into_value(command(&daemon, query, true)).unwrap();
    assert!(matches!(repeated, Answer::Built(built) if built.texts==0 && built.vectors==0));
    let empty = crate::protocol::into_value(command(
        &daemon,
        "sessions where physical:true | limit 0",
        true,
    ))
    .unwrap();
    assert!(matches!(empty, Answer::Built(built) if built.texts==0 && built.vectors==0));
}

#[test]
fn embed_carries_query_limits_in_the_query() {
    let (_, _, values) =
        crate::cli::request_arguments(&["polylogue", "embed", "--query", "sessions | limit 2"])
            .unwrap();
    let arguments: crate::protocol::embed::Arguments =
        serde_json::from_value(Value::Object(values)).unwrap();
    assert_eq!(arguments.query.as_deref(), Some("sessions | limit 2"));
}
