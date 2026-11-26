use super::settings::merge_hooks;
use super::*;

pub(super) fn startup() -> Startup {
    let engine = polylogue_engine::Config::new(
        PathBuf::from("/synthetic/home"),
        Default::default(),
        "synthetic-machine".into(),
        "synthetic-host".into(),
        1,
        polylogue_model::WallClock::fixed("2026-01-01T00:00:00Z".parse().unwrap()),
    )
    .unwrap();
    Startup::for_test(engine, Path::new("/synthetic/archive"))
}

#[test]
fn hook_spool_files_get_one_line_per_event() {
    let dir = tempfile::tempdir().unwrap();
    append_to_spool(
        dir.path(),
        Harness::ClaudeCode,
        "Stop",
        &json!({"session_id": "a"}),
        &json!({}),
        &startup(),
    )
    .unwrap();
    append_to_spool(
        dir.path(),
        Harness::Codex,
        "SessionEnd",
        &json!({"session_id": "b"}),
        &json!({}),
        &startup(),
    )
    .unwrap();
    let file = dir
        .path()
        .join(format!("{}.ndjson", std::os::unix::process::parent_id()));
    let text = std::fs::read_to_string(file).unwrap();
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 2);
    let event_id = uuid::Uuid::parse_str(lines[0]["event_id"].as_str().unwrap()).unwrap();
    assert_eq!(event_id.get_timestamp().unwrap().to_unix().0, 1767225600);
    assert_eq!(lines[0]["at_ns"], "1767225600000000000");
    assert_ne!(lines[0]["event_id"], lines[1]["event_id"]);
    assert_eq!(lines[0]["provider"], "claude-code");
    assert_eq!(lines[0]["session_id"], "a");
    assert_eq!(lines[1]["event"], "SessionEnd");
    assert_eq!(lines[1]["provider"], "codex");
    assert_eq!(lines[1]["session_id"], "b");
}

#[test]
fn a_hermes_turn_keeps_its_event_without_the_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let payload = json!({
        "hook_event_name": "pre_llm_call",
        "session_id": "20260101_000000_abc",
        "cwd": "/tmp",
        "extra": {"is_first_turn": true, "user_message": "hi", "conversation_history": [{"role": "user"}]},
    });
    append_to_spool(
        dir.path(),
        Harness::Hermes,
        "pre_llm_call",
        &payload,
        &json!({}),
        &startup(),
    )
    .unwrap();
    let file = dir
        .path()
        .join(format!("{}.ndjson", std::os::unix::process::parent_id()));
    let line: Value = serde_json::from_str(std::fs::read_to_string(file).unwrap().trim()).unwrap();
    assert_eq!(line["provider"], "hermes");
    assert!(
        line["payload"]["extra"]
            .get("conversation_history")
            .is_none()
    );
    assert_eq!(line["payload"]["extra"]["user_message"], "hi");
    let native = line["session_id"].as_str().unwrap();
    assert!(
        native.starts_with("20260101_000000_abc@profile-"),
        "{native}"
    );
}

#[test]
fn hermes_session_uses_the_resolved_source_profile_and_machine() {
    let mut startup = startup();
    startup.engine.tool_dirs.insert(
        "HERMES_HOME".into(),
        PathBuf::from("/synthetic/hermes/profiles/work"),
    );
    assert_eq!(
        session_of(Harness::Hermes, &json!({"session_id":"turn-1"}), &startup),
        Some(SessionId::parse("hermes:turn-1@profile-work").unwrap())
    );
}

#[test]
fn context_uses_the_configured_archive_socket_and_clock() {
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let startup = Startup::for_test(startup().engine, dir.path());
    let listener = UnixListener::bind(Archive::at(dir.path()).socket()).unwrap();
    listener.set_nonblocking(true).unwrap();
    let peer = std::thread::spawn(move || {
        let until = std::time::Instant::now() + Duration::from_secs(2);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && std::time::Instant::now() < until =>
                {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("configured archive was not contacted: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = String::new();
        BufReader::new(&stream).read_line(&mut request).unwrap();
        let request: Value = serde_json::from_str(&request).unwrap();
        assert_eq!(request["operation"], "rows");
        assert_eq!(
            request["arguments"],
            json!({"steps": [{"verb":"brief", "cwd":"/synthetic/project", "for":"codex:turn-1", "source":"startup"}]})
        );
        // The caller's 400 ms budget, less the time spent reaching the call.
        assert!(
            request["deadline"]
                .as_u64()
                .is_some_and(|milliseconds| (250..=400).contains(&milliseconds)),
            "{}",
            request["deadline"]
        );
        assert!(request.get("budget_tokens").is_none());
        let id = uuid::Uuid::parse_str(request["request_id"].as_str().unwrap()).unwrap();
        assert_eq!(id.get_timestamp().unwrap().to_unix().0, 1767225600);
        let response = command_surface::Answer::ok(
            json!({"kind":"steps", "reply":{"resolved":null, "steps":[{"verb":"brief", "request_id":format!("{}-1", request["request_id"].as_str().unwrap()), "outcome":{"status":"done", "reply":{"kind":"brief", "reply":{"delivery":"context-1", "prepared":{"event":"session_start","empty":false,"text":"configured archive","refs":[],"left_out":[]}}}}}]}}),
        );
        write!(
            stream,
            "event: answer\ndata: {}\n\n",
            serde_json::to_string(&response).unwrap()
        )
        .unwrap();
    });
    let reply = respond(
        Harness::Codex,
        "SessionStart",
        &json!({"session_id":"turn-1","cwd":"/synthetic/project"}),
        &startup,
        Instant::now() + Duration::from_millis(400),
    );
    peer.join().unwrap();
    assert_eq!(
        reply,
        Some(("context-1".into(), "configured archive".into()))
    );
}

#[test]
fn pasted_and_search_briefs_use_rows() {
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixListener;
    let recipient = SessionId::parse("codex:receiving-session").unwrap();
    for (ask, expected) in [
        (
            Ask::Session {
                session: SessionId::parse("claude-code:parser-session").unwrap(),
                for_session: Some(recipient.clone()),
            },
            json!({"selection":"session:claude-code:parser-session", "steps":[{"verb":"brief", "for":"codex:receiving-session"}]}),
        ),
        (
            Ask::Words {
                text: "date parser".into(),
                cwd: Some("/synthetic/project".into()),
                for_session: Some(recipient.clone()),
                cursor: Some("next-page".into()),
            },
            json!({"selection":"date parser", "steps":[{"verb":"brief", "cwd":"/synthetic/project", "for":"codex:receiving-session", "cursor":"next-page"}]}),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let startup = Startup::for_test(startup().engine, dir.path());
        let listener = UnixListener::bind(Archive::at(dir.path()).socket()).unwrap();
        listener.set_nonblocking(true).unwrap();
        let peer = std::thread::spawn(move || {
            let until = std::time::Instant::now() + Duration::from_secs(2);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && std::time::Instant::now() < until =>
                    {
                        std::thread::sleep(Duration::from_millis(1))
                    }
                    Err(error) => panic!("brief exchange was not contacted: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request["operation"], "rows");
            assert_eq!(request["arguments"], expected);
            assert_eq!(request["budget_tokens"], 1000);
            let response = command_surface::Answer::ok(json!({
                "kind":"steps", "reply":{"resolved":null, "steps":[{
                    "verb":"brief", "request_id":format!("{}-1", request["request_id"].as_str().unwrap()),
                    "outcome":{"status":"done", "reply":{"kind":"brief", "reply":{
                        "delivery":"brief-1", "prepared":{"event":"session_brief", "empty":false, "text":"prepared context", "refs":[], "left_out":[]}
                    }}}
                }]}
            }));
            write!(
                stream,
                "event: answer\ndata: {}\n\n",
                serde_json::to_string(&response).unwrap()
            )
            .unwrap();
        });
        let reply = context(&startup, ask, Some(1000), Instant::now() + CONTEXT_BUDGET);
        peer.join().unwrap();
        assert_eq!(reply, Some(("brief-1".into(), "prepared context".into())));
    }
}

#[test]
fn hooks_fail_open_without_a_daemon() {
    // Without a daemon the hooks fail open: no context, no output, for an
    // event that asks the daemon for session-start context and for one that
    // asks it for an advisory.
    let start = json!({"session_id": "a", "cwd": "/nonexistent", "source": "startup"});
    assert!(
        respond(
            Harness::Codex,
            "SessionStart",
            &start,
            &startup(),
            Instant::now() + CONTEXT_BUDGET
        )
        .is_none()
    );
    let prompt =
        json!({"session_id": "a", "cwd": "/synthetic/project", "prompt": "Inspect src/parser.rs"});
    assert!(
        respond(
            Harness::ClaudeCode,
            "UserPromptSubmit",
            &prompt,
            &startup(),
            Instant::now() + CONTEXT_BUDGET
        )
        .is_none()
    );
}

#[test]
fn pasted_ids_are_found_in_prose() {
    let ids = pasted_session_ids(
        "pick up from claude-code:7f3e9a1c-0000-4000-8000-000000000001, and see \
         `0199a1b2-3c4d-7e8f-9012-345678901234`.",
    );
    let texts: Vec<String> = ids.iter().map(ToString::to_string).collect();
    assert_eq!(
        texts,
        [
            "claude-code:7f3e9a1c-0000-4000-8000-000000000001",
            "claude-code:0199a1b2-3c4d-7e8f-9012-345678901234",
            "codex:0199a1b2-3c4d-7e8f-9012-345678901234",
        ]
    );
    assert!(pasted_session_ids("no ids here: http://x").is_empty());
}

#[test]
fn hook_install_merges_and_is_idempotent() {
    let other = json!({
        "model": "opus",
        "hooks": {
            "Stop": [{"hooks": [{"type": "command", "command": "notify-send done"}]}]
        }
    });
    let once = merge_hooks(other.clone(), Harness::ClaudeCode, None, true).unwrap();
    let twice = merge_hooks(once.clone(), Harness::ClaudeCode, None, true).unwrap();
    assert_eq!(once, twice);
    assert_eq!(once["model"], "opus");
    let stop = once["hooks"]["Stop"].as_array().unwrap();
    assert_eq!(stop.len(), 2);
    assert_eq!(stop[0]["hooks"][0]["command"], "notify-send done");
    let removed = merge_hooks(once, Harness::ClaudeCode, None, false).unwrap();
    assert_eq!(removed["hooks"]["Stop"].as_array().unwrap().len(), 1);
    assert!(removed["hooks"].get("SessionStart").is_none());
}

#[test]
fn install_writes_a_file_and_refuses_symlinks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    install(&path, Harness::ClaudeCode, || None, true).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("polylogue internal hook SessionStart"));
    let link = dir.path().join("linked.json");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    assert!(install(&link, Harness::ClaudeCode, || None, true).is_err());
}

#[test]
fn hook_entries_call_internal_hook() {
    for (harness, event, expected) in [
        (
            Harness::ClaudeCode,
            "SessionStart",
            "polylogue internal hook SessionStart",
        ),
        (
            Harness::Codex,
            "SessionStart",
            "polylogue internal hook SessionStart --harness codex",
        ),
        (
            Harness::GeminiCli,
            "BeforeAgent",
            "polylogue internal hook BeforeAgent --harness gemini-cli",
        ),
    ] {
        let settings = merge_hooks(json!({}), harness, None, true).unwrap();
        assert_eq!(settings["hooks"][event][0]["hooks"][0]["command"], expected);
    }
}

/// docs/installing.md: recording entries run asynchronously where the
/// harness supports it (Claude Code's `async`); context entries do not.
#[test]
fn claude_code_recording_hooks_run_in_the_background() {
    let settings = merge_hooks(json!({}), Harness::ClaudeCode, None, true).unwrap();
    for (event, background) in [
        ("SessionStart", false),
        ("UserPromptSubmit", false),
        ("PostToolUse", false),
        ("PreToolUse", true),
        ("Notification", true),
        ("Stop", true),
        ("SubagentStop", true),
        ("PreCompact", true),
        ("SessionEnd", true),
    ] {
        let hook = &settings["hooks"][event][0]["hooks"][0];
        assert_eq!(
            hook.get("async"),
            background.then_some(&json!(true)),
            "{event}"
        );
    }
    let codex = merge_hooks(json!({}), Harness::Codex, None, true).unwrap();
    let stop = &codex["hooks"]["Stop"][0]["hooks"][0];
    assert!(stop["command"].is_string());
    assert!(stop.get("async").is_none());
}

#[test]
fn codex_hooks_name_the_harness_and_replace_their_own() {
    let once = merge_hooks(json!({}), Harness::Codex, None, true).unwrap();
    let start = &once["hooks"]["SessionStart"][0]["hooks"][0]["command"];
    assert_eq!(
        start,
        "polylogue internal hook SessionStart --harness codex"
    );
    assert!(
        once["hooks"].get("Notification").is_none(),
        "Codex has no such event"
    );
    let twice = merge_hooks(once.clone(), Harness::Codex, None, true).unwrap();
    assert_eq!(once, twice);
    let removed = merge_hooks(twice, Harness::Codex, None, false).unwrap();
    // A `hooks` object only Polylogue's commands filled goes with them.
    assert_eq!(removed, json!({}));
}

#[test]
fn install_preserves_permissions_and_ignores_an_old_temporary_name() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let other = dir.path().join("other.json");
    std::fs::write(&path, r#"{"model":"example"}"#).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    std::fs::write(&other, b"unchanged").unwrap();
    let stale = dir
        .path()
        .join(format!(".settings.json.polylogue-{}", std::process::id()));
    std::os::unix::fs::symlink(&other, stale).unwrap();
    install(&path, Harness::ClaudeCode, || None, true).unwrap();
    assert_eq!(std::fs::read(&other).unwrap(), b"unchanged");
    assert!(!path.is_symlink());
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o640
    );
    let settings: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(settings["model"], "example");
    assert!(settings["hooks"]["SessionStart"].is_array());
}

mod rollback;
mod settings;

#[test]
fn a_silent_daemon_cannot_hold_a_hook_past_its_deadline() {
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("daemon.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let peer = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        // Stay silent until the hook closes its end. Only its deadline
        // can finish the exchange while this connection is open.
        let mut request = Vec::new();
        stream.read_to_end(&mut request).unwrap();
        assert!(!request.is_empty());
    });
    let Err(error) = crate::client::call_with_budget(
        &socket,
        &crate::client::request("status", serde_json::json!({}), "status".into()),
        Duration::from_millis(20),
    ) else {
        panic!("a silent daemon cannot answer");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    peer.join().unwrap();
}

#[test]
fn web_links_resolve_to_the_same_sessions_as_captures() {
    let links = [
        ("https://chatgpt.com/c/demo?model=x", "chatgpt:demo"),
        ("https://chat.openai.com/c/demo", "chatgpt:demo"),
        (
            "https://chatgpt.com/g/g-project/c/demo#turn",
            "chatgpt:demo",
        ),
        ("https://gemini.google.com/u/0/app/demo", "gemini-app:demo"),
        (
            "https://aistudio.google.com/u/2/prompts/demo",
            "ai-studio:demo",
        ),
        ("https://claude.ai/chat/demo", "claude-ai:demo"),
        ("https://gemini.google.com/app/demo", "gemini-app:demo"),
        ("https://aistudio.google.com/prompts/demo", "ai-studio:demo"),
        ("https://grok.com/c/demo", "grok:demo"),
    ];
    for (url, expected) in links {
        assert_eq!(
            pasted_session_ids(&format!("continue [{url}]({url}).")),
            vec![SessionId::parse(expected).unwrap()]
        );
    }
    for url in [
        "https://chatgpt.com.evil.invalid/c/demo",
        "http://chatgpt.com/c/demo",
        "https://u@chatgpt.com/c/demo",
        "https://chatgpt.com/c/demo/extra",
        "https://chatgpt.com/share/demo",
        "https://claude.ai/g/project/chat/demo",
        "https://grok.com/u/0/c/demo",
        "https://gemini.google.com/u/account/app/demo",
        "https://aistudio.google.com/g/project/prompts/demo",
        "https://chatgpt.com/u/0/c/demo",
    ] {
        assert!(pasted_session_ids(url).is_empty(), "{url}");
    }
}

#[test]
fn gemini_hooks_use_the_documented_events_and_millisecond_timeouts() {
    let settings =
        json!({"hooks":{"BeforeAgent":[{"hooks":[{"type":"command", "command":"other"}]}]}});
    let once = merge_hooks(settings.clone(), Harness::GeminiCli, None, true).unwrap();
    assert_eq!(
        merge_hooks(once.clone(), Harness::GeminiCli, None, true).unwrap(),
        once
    );
    assert_eq!(
        once["hooks"]["SessionStart"][0]["hooks"][0]["timeout"],
        10000
    );
    assert_eq!(once["hooks"]["BeforeTool"][0]["hooks"][0]["timeout"], 2000);
    assert_eq!(
        once["hooks"]["BeforeAgent"][1]["hooks"][0]["command"],
        "polylogue internal hook BeforeAgent --harness gemini-cli"
    );
    assert_eq!(
        merge_hooks(once, Harness::GeminiCli, None, false).unwrap(),
        settings
    );
    assert_eq!(
        session_of(
            Harness::GeminiCli,
            &json!({"session_id":"test"}),
            &startup()
        )
        .unwrap()
        .to_string(),
        "gemini-cli:test"
    );
}

#[test]
fn gemini_context_output_matches_the_harness_protocol() {
    assert_eq!(
        output_for(Harness::GeminiCli, "SessionStart", "brief"),
        json!({"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"brief"}})
    );
    assert_eq!(
        output_for(Harness::GeminiCli, "BeforeAgent", "takeover"),
        json!({"hookSpecificOutput":{"hookEventName":"BeforeAgent","additionalContext":"takeover"}})
    );
}

#[test]
fn decoded_hook_context_has_visible_terminal_controls() {
    let text = "\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\u{202e}\n\tcontext";
    for harness in [
        Harness::ClaudeCode,
        Harness::Codex,
        Harness::Hermes,
        Harness::GeminiCli,
    ] {
        let mut bytes = Vec::new();
        assert!(write_context(&mut bytes, harness, "SessionStart", text));
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        let context = match harness {
            Harness::Hermes => &value["context"],
            _ => &value["hookSpecificOutput"]["additionalContext"],
        };
        assert_eq!(
            context,
            "\\x1b]8;;https://example.test\\x1b\\link\\x1b]8;;\\x1b\\\\x7f\\u{202e}\n\tcontext"
        );
    }
}

#[test]
fn context_output_records_a_parsable_emission() {
    for printed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let startup = startup();
        let payload = json!({"session_id":"emission-session","cwd":"/synthetic/project"});
        let observed = json!({"boot_id":"synthetic-boot"});
        let invocation = append_to_spool(
            dir.path(),
            Harness::ClaudeCode,
            "SessionStart",
            &payload,
            &observed,
            &startup,
        )
        .unwrap();
        let recorder = EmissionRecorder {
            dir: dir.path(),
            harness: Harness::ClaudeCode,
            event: "SessionStart",
            invocation: Some(&invocation),
            session: session_of(Harness::ClaudeCode, &payload, &startup),
            observed: &observed,
            startup: &startup,
        };
        let emitted = recorder.emit(
            &["first-receipt".into(), "second:receipt".into()],
            "context text is not an attribution marker",
            |harness, event, text| {
                assert_eq!(harness, Harness::ClaudeCode);
                assert_eq!(event, "SessionStart");
                assert_eq!(text, "context text is not an attribution marker");
                let path = std::fs::read_dir(dir.path())
                    .unwrap()
                    .next()
                    .unwrap()
                    .unwrap()
                    .path();
                assert_eq!(
                    std::fs::read_to_string(path).unwrap().lines().count(),
                    1,
                    "emission was recorded before output succeeded"
                );
                printed
            },
        );
        assert_eq!(emitted, printed);
        let path = std::fs::read_dir(dir.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let bytes = std::fs::read(&path).unwrap();
        assert!(polylogue_sources::hook_spool::open(&path).is_some());
        let step = polylogue_sources::parse::run(
            &polylogue_sources::hook_spool::HookSpool::default(),
            &mut polylogue_sources::hook_spool::ParserState::default(),
            &bytes,
            0,
        )
        .unwrap();
        assert_eq!(step.output.events.len(), if printed { 2 } else { 1 });
        assert!(step.output.messages.is_empty());
        let emissions = step
            .output
            .events
            .iter()
            .filter(|event| event.kind.ends_with("hook_context_emitted"))
            .collect::<Vec<_>>();
        if printed {
            assert_eq!(emissions.len(), 1);
            assert_eq!(emissions[0].data["invocation"], invocation);
            assert_eq!(emissions[0].data["session"], "claude-code:emission-session");
            assert_eq!(emissions[0].data["hook_event"], "SessionStart");
            assert_eq!(
                emissions[0].data["deliveries"],
                serde_json::json!(["delivery:first-receipt", "delivery:second:receipt"])
            );
            assert_ne!(step.output.events[1].data["event_id"], invocation);
        } else {
            assert!(emissions.is_empty());
        }
    }
}

/// The newer Claude Code events (permission requests, failed stops,
/// subagent starts, questions) are registered only for a version a probe
/// confirmed; an unknown or older version gets the base events, and
/// removal takes them out whatever version installed them.
#[test]
fn newer_events_are_registered_only_for_a_confirmed_version() {
    use super::settings::HarnessVersion;
    let newer = |settings: &Value| {
        [
            "PermissionRequest",
            "StopFailure",
            "SubagentStart",
            "Elicitation",
        ]
        .iter()
        .filter(|e| settings["hooks"].get(**e).is_some())
        .count()
    };
    assert_eq!(
        HarnessVersion::parse("2.1.316 (Claude Code)"),
        HarnessVersion::parse("x 2.1.316")
    );
    assert!(HarnessVersion::parse("2.1.40") < HarnessVersion::parse("2.1.316"));
    assert_eq!(HarnessVersion::parse("no version"), None);
    let unknown = merge_hooks(json!({}), Harness::ClaudeCode, None, true).unwrap();
    assert_eq!(newer(&unknown), 0);
    let older = HarnessVersion::parse("2.1.40").unwrap();
    let old = merge_hooks(json!({}), Harness::ClaudeCode, Some(&older), true).unwrap();
    assert_eq!(newer(&old), 0);
    assert!(old["hooks"].get("SessionStart").is_some());
    let current = HarnessVersion::parse("2.1.316").unwrap();
    let new = merge_hooks(json!({}), Harness::ClaudeCode, Some(&current), true).unwrap();
    assert_eq!(newer(&new), 4);
    assert_eq!(
        new["hooks"]["PermissionRequest"][0]["hooks"][0]["command"],
        "polylogue internal hook PermissionRequest"
    );
    // Installing again for an older version takes the newer events out.
    let downgraded = merge_hooks(new.clone(), Harness::ClaudeCode, Some(&older), true).unwrap();
    assert_eq!(newer(&downgraded), 0);
    let removed = merge_hooks(new, Harness::ClaudeCode, None, false).unwrap();
    // A `hooks` object only Polylogue's commands filled goes with them.
    assert_eq!(removed, json!({}));
}

/// Messages and notices are claimed only after a tool result or at a prompt,
/// never before a tool runs.
#[test]
fn the_mailbox_is_claimed_only_between_steps() {
    for harness in [Harness::ClaudeCode, Harness::Codex] {
        assert!(accepts_mail(harness, "PostToolUse"));
        assert!(accepts_mail(harness, "UserPromptSubmit"));
        assert!(!accepts_mail(harness, "PreToolUse"));
        assert!(!accepts_mail(harness, "SessionStart"));
        assert!(!accepts_mail(harness, "PermissionRequest"));
    }
    assert!(accepts_mail(Harness::GeminiCli, "AfterTool"));
    assert!(!accepts_mail(Harness::GeminiCli, "BeforeTool"));
    assert!(accepts_mail(Harness::Hermes, "pre_llm_call"));
}

/// A decision given ahead becomes Claude Code's permission decision; no
/// decision prints nothing, so the harness's own prompt stands.
#[test]
fn a_standing_answer_becomes_the_permission_decision() {
    assert_eq!(permission_output(&json!({"decision": null})), None);
    assert_eq!(
        permission_output(&json!({"decision": "approve", "text": null, "delivery": "d"})),
        Some(
            json!({"hookSpecificOutput": {"hookEventName": "PermissionRequest",
            "decision": {"behavior": "allow"}}})
        )
    );
    assert_eq!(
        permission_output(&json!({"decision": "deny", "text": "not on main"})),
        Some(
            json!({"hookSpecificOutput": {"hookEventName": "PermissionRequest",
            "decision": {"behavior": "deny", "message": "not on main"}}})
        )
    );
}

#[test]
fn a_context_timeout_emits_an_unreferenced_notice_and_names_only_the_original_prepare() {
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let startup = Startup::for_test(startup().engine, dir.path());
    let listener = UnixListener::bind(Archive::at(dir.path()).socket()).unwrap();
    listener.set_nonblocking(true).unwrap();
    let peer = std::thread::spawn(move || {
        let accept = || {
            let until = std::time::Instant::now() + Duration::from_secs(2);
            loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && std::time::Instant::now() < until =>
                    {
                        std::thread::sleep(Duration::from_millis(1))
                    }
                    Err(error) => panic!("context exchange was not contacted: {error}"),
                }
            }
        };
        let mut stream = accept();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut line = String::new();
        BufReader::new(&stream).read_line(&mut line).unwrap();
        let prepare: Value = serde_json::from_str(&line).unwrap();
        let failure = command_surface::Answer::ok(json!({
            "kind":"steps", "reply":{"resolved":null, "steps":[{
                "verb":"brief", "request_id":format!("{}-1", prepare["request_id"].as_str().unwrap()),
                "outcome":{"status":"failed", "error":crate::protocol::failure(polylogue_model::Code::DeadlineExceeded, "fixture timeout")}
            }]}
        }));
        write!(
            stream,
            "event: answer\ndata: {}\n\n",
            serde_json::to_string(&failure).unwrap()
        )
        .unwrap();
        let mut stream = accept();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        line.clear();
        BufReader::new(&stream).read_line(&mut line).unwrap();
        let outcome: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(outcome["operation"], "internal delivery-event");
        assert_eq!(
            outcome["arguments"]["delivery"],
            format!("{}-1", prepare["request_id"].as_str().unwrap())
        );
        assert_eq!(outcome["arguments"]["what"], "not_sent");
        assert_eq!(outcome["arguments"]["detail"], "timeout");
        write!(
            stream,
            "event: answer\ndata: {}\n\n",
            serde_json::to_string(&command_surface::Answer::ok(
                json!({"queued":false,"preparation":"unconfirmed"})
            ))
            .unwrap()
        )
        .unwrap();
    });
    let (deliveries, text) = context(
        &startup,
        Ask::SessionStart {
            cwd: "/synthetic/project".into(),
            session: None,
            source: StartSource::Startup,
        },
        None,
        Instant::now() + CONTEXT_BUDGET,
    )
    .unwrap();
    peer.join().unwrap();
    assert!(deliveries.is_empty());
    assert_eq!(text, TIMEOUT_NOTICE);
}

#[test]
fn file_advisory_hook_uses_the_shared_request_and_silences_timeout() {
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixListener;
    for timed_out in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let startup = Startup::for_test(startup().engine, dir.path());
        let listener = UnixListener::bind(Archive::at(dir.path()).socket()).unwrap();
        let (ready, listening) = std::sync::mpsc::sync_channel(0);
        let peer = std::thread::spawn(move || {
            ready.send(()).unwrap();
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request["operation"], "internal advisory");
            assert_eq!(
                request["arguments"],
                json!({"prompt":"Inspect src/parser.rs", "cwd":"/synthetic/project", "session":"codex:turn-1"})
            );
            assert_eq!(request["deadline"], 20);
            assert!(request.get("budget_tokens").is_none());
            let id = uuid::Uuid::parse_str(request["request_id"].as_str().unwrap()).unwrap();
            assert_eq!(id.get_timestamp().unwrap().to_unix().0, 1767225600);
            let answer = if timed_out {
                command_surface::Answer::error(crate::protocol::failure(
                    polylogue_model::Code::DeadlineExceeded,
                    "fixture timeout",
                ))
            } else {
                command_surface::Answer::ok(
                    json!({"delivery":request["request_id"],"prepared":{"event":"advice","empty":false,"text":"one pointer", "refs":[],"left_out":[]}}),
                )
            };
            write!(
                stream,
                "event: answer\ndata: {}\n\n",
                serde_json::to_string(&answer).unwrap()
            )
            .unwrap();
            listener.set_nonblocking(true).unwrap();
            listener
        });
        listening.recv().unwrap();
        let result = respond(
            Harness::Codex,
            "UserPromptSubmit",
            &json!({"session_id":"turn-1","cwd":"/synthetic/project","prompt":"Inspect src/parser.rs"}),
            &startup,
            Instant::now() + CONTEXT_BUDGET,
        );
        let listener = peer.join().unwrap();
        if timed_out {
            assert!(result.is_none(), "optional timeout must emit no notice");
        } else {
            assert_eq!(result.unwrap().1, "one pointer");
        }
        assert_eq!(
            listener.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock,
            "respond neither acknowledges emission nor retries preparation"
        );
    }
}

#[test]
fn a_forked_session_start_asks_for_no_context() {
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let startup = Startup::for_test(startup().engine, dir.path());
    let listener = UnixListener::bind(Archive::at(dir.path()).socket()).unwrap();
    listener.set_nonblocking(true).unwrap();
    assert!(
        respond(
            Harness::ClaudeCode,
            "SessionStart",
            &json!({"session_id":"fork-1","cwd":"/synthetic/project","source":"fork"}),
            &startup,
            Instant::now() + CONTEXT_BUDGET
        )
        .is_none()
    );
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn prompt_without_a_path_needs_no_advisory_exchange() {
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let startup = Startup::for_test(startup().engine, dir.path());
    let listener = UnixListener::bind(Archive::at(dir.path()).socket()).unwrap();
    listener.set_nonblocking(true).unwrap();
    assert!(respond(Harness::Codex, "UserPromptSubmit", &json!({"session_id":"turn-1","cwd":"/synthetic/project","prompt":"Continue the current work"}), &startup, Instant::now() + CONTEXT_BUDGET).is_none());
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn the_hook_spool_lives_under_the_state_directory() {
    let engine = polylogue_engine::Config::new(
        PathBuf::from("/synthetic/home"),
        [(
            "XDG_STATE_HOME".to_owned(),
            PathBuf::from("/synthetic/state"),
        )]
        .into(),
        "synthetic-machine".into(),
        "synthetic-host".into(),
        1,
        polylogue_model::WallClock::fixed("2026-01-01T00:00:00Z".parse().unwrap()),
    )
    .unwrap();
    assert_eq!(
        engine.spool_dir(polylogue_sources::hook_spool::ORIGIN),
        Path::new("/synthetic/state/polylogue/hooks")
    );
}

#[test]
fn an_elapsed_hook_deadline_sends_no_request_or_delivery() {
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let startup = Startup::for_test(startup().engine, dir.path());
    let listener = UnixListener::bind(Archive::at(dir.path()).socket()).unwrap();
    listener.set_nonblocking(true).unwrap();
    let until = Instant::now();
    let result = respond(
        Harness::Codex,
        "SessionStart",
        &json!({"session_id":"turn-1", "cwd":"/synthetic/project"}),
        &startup,
        until,
    );
    assert_eq!(result, Some((String::new(), TIMEOUT_NOTICE.into())));
    assert!(claim(&startup, &SessionId::parse("codex:turn-1").unwrap(), until).is_none());
    permission(
        &startup,
        &SessionId::parse("claude-code:turn-1").unwrap(),
        &json!({}),
        until,
    );
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

#[test]
fn expired_multi_session_context_emits_one_unreferenced_notice() {
    let startup = startup();
    let result = respond(
        Harness::Codex,
        "UserPromptSubmit",
        &json!({"session_id":"turn-1", "prompt":"codex:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee codex:bbbbbbbb-cccc-dddd-eeee-ffffffffffff"}),
        &startup,
        Instant::now(),
    );
    assert_eq!(result, Some((String::new(), TIMEOUT_NOTICE.into())));
}

/// docs/agent-context.md: three or more pasted sessions inject one index,
/// asked for in one request that names them all.
#[test]
fn three_pasted_sessions_give_one_index() {
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let startup = Startup::for_test(startup().engine, dir.path());
    let listener = UnixListener::bind(Archive::at(dir.path()).socket()).unwrap();
    let peer = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut line = String::new();
        BufReader::new(&stream).read_line(&mut line).unwrap();
        let request: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(request["operation"], "rows");
        assert_eq!(
            request["arguments"],
            json!({
                "references": ["session:claude-code:s1", "session:claude-code:s2", "session:claude-code:s3"],
                "steps": [{"verb": "brief", "for": "codex:receiving"}],
            })
        );
        let response = command_surface::Answer::ok(json!({
            "kind":"steps", "reply":{"resolved":null, "steps":[{
                "verb":"brief", "request_id":format!("{}-1", request["request_id"].as_str().unwrap()),
                "outcome":{"status":"done", "reply":{"kind":"brief", "reply":{
                    "delivery":"index-1", "prepared":{"event":"session_brief", "empty":false, "text":"session index", "refs":[], "left_out":[]}
                }}}
            }]}
        }));
        write!(
            stream,
            "event: answer\ndata: {}\n\n",
            serde_json::to_string(&response).unwrap()
        )
        .unwrap();
        // No second request arrives: one index, not a takeover per session.
        let mut rest = String::new();
        assert_eq!(BufReader::new(&stream).read_line(&mut rest).unwrap_or(0), 0);
    });
    let reply = pasted_sessions(
        Harness::Codex,
        &json!({"session_id": "receiving"}),
        "compare claude-code:s1, claude-code:s2 and claude-code:s3",
        &startup,
        Instant::now() + CONTEXT_BUDGET,
    );
    peer.join().unwrap();
    assert_eq!(reply, Some(("index-1".into(), "session index".into())));
}

#[test]
fn a_hook_that_cannot_reach_the_daemon_names_the_cause() {
    use std::os::unix::net::UnixListener;
    let dir = tempfile::tempdir().unwrap();
    let request = crate::client::request("status", json!({}), "status".into());
    let budget = Duration::from_millis(50);
    // Refused: no daemon listens on the socket.
    let missing = dir.path().join("missing.sock");
    let cause = exchange(&missing, &request, budget).unwrap_err();
    assert!(matches!(cause, Unanswered::NotRunning(_)), "{cause:?}");
    assert!(
        cause.to_string().starts_with("the daemon is not running ("),
        "{cause}"
    );
    // Timeout: a daemon that accepts and never answers.
    let silent = dir.path().join("silent.sock");
    let listener = UnixListener::bind(&silent).unwrap();
    let held = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let _ = stream.read_to_end(&mut request);
    });
    let cause = exchange(&silent, &request, budget).unwrap_err();
    assert_eq!(cause.to_string(), "the daemon did not answer within 50 ms");
    held.join().unwrap();
    // Decode: an answer line that is not an answer.
    let garbled = dir.path().join("garbled.sock");
    let listener = UnixListener::bind(&garbled).unwrap();
    let answering = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(&stream), &mut line).unwrap();
        writeln!(stream, "not an answer").unwrap();
    });
    let cause = exchange(&garbled, &request, budget).unwrap_err();
    assert!(matches!(cause, Unanswered::Undecodable(_)), "{cause:?}");
    assert!(
        cause
            .to_string()
            .starts_with("the daemon's answer could not be decoded: "),
        "{cause}"
    );
    answering.join().unwrap();
}

#[test]
fn a_morrow_session_start_without_context_is_stored_and_counted_after_daemon_start() {
    use crate::daemon::{server::Server, service::ArchiveMode};
    let archive = polylogue_testkit::Morrow::build(polylogue_testkit::Set::Main);
    let startup = Startup::for_test(archive.config(), &archive.root);
    let session = polylogue_testkit::session(1);
    let payload =
        json!({"session_id":session.native(), "cwd":"/home/op/morrow", "source":"startup"});
    assert_eq!(
        run_event(
            Harness::ClaudeCode,
            "SessionStart",
            payload.clone(),
            &startup,
            Instant::now() + CONTEXT_BUDGET,
            |_, _, _| panic!("no context to emit")
        ),
        ExitCode::SUCCESS
    );
    let mut runtime =
        crate::test_daemon::start_runtime(&archive.root, archive.config(), ArchiveMode::Standalone)
            .unwrap();
    let mut server = Server::start(std::sync::Arc::clone(&runtime.daemon)).unwrap();
    let status = crate::protocol::into_value(crate::test_daemon::call(
        &runtime.daemon,
        "status",
        json!({}),
    ))
    .unwrap();
    assert_eq!(
        status["session_starts_without_context"],
        json!({
            "since":"2026-10-07T10:00:00Z", "causes":{"daemon_unreachable":1}
        })
    );
    let rows = crate::protocol::into_value(crate::test_daemon::call(&runtime.daemon, "rows", json!({"selection":"events where event:hook_session_context", "fields":["data", "session"]}))).unwrap();
    assert_eq!(rows["count"], 1, "{rows}");
    assert_eq!(rows["rows"][0]["session"], session.to_string());
    let data = &rows["rows"][0]["data"];
    assert_eq!(data["outcome"], "not_given");
    assert_eq!(data["cause"], "daemon_unreachable");
    assert_eq!(data["cwd"], "/home/op/morrow");
    assert_eq!(
        data["detail"]
            .as_str()
            .unwrap()
            .starts_with("the daemon is not running ("),
        true
    );
    for (reference, expected) in [
        (
            "claude-code:absent",
            "session claude-code:absent is not in the archive (yet)",
        ),
        (
            "https://chatgpt.com/c/absent",
            "session chatgpt:absent is not in the archive (yet); capture it with `polylogue import chatgpt`",
        ),
    ] {
        let reply = respond(
            Harness::ClaudeCode,
            "UserPromptSubmit",
            &json!({"session_id":session.native(), "prompt":reference}),
            &startup,
            Instant::now() + CONTEXT_BUDGET,
        )
        .unwrap();
        assert_eq!(reply.1, expected);
    }
    let mut config = archive.config();
    config.clock = polylogue_model::WallClock::fixed("2026-10-07T10:00:01Z".parse().unwrap());
    let startup = Startup::for_test(config, &archive.root);
    let mut text = String::new();
    assert_eq!(
        run_event(
            Harness::ClaudeCode,
            "SessionStart",
            payload,
            &startup,
            Instant::now() + CONTEXT_BUDGET,
            |_, _, context| {
                text = context.into();
                true
            }
        ),
        ExitCode::SUCCESS
    );
    assert!(
        text.contains("Polylogue brief for /home/op/morrow"),
        "{text}"
    );
    server.shutdown().unwrap();
    runtime.shutdown().unwrap();
    let mut runtime =
        crate::test_daemon::start_runtime(&archive.root, archive.config(), ArchiveMode::Standalone)
            .unwrap();
    let status = crate::protocol::into_value(crate::test_daemon::call(
        &runtime.daemon,
        "status",
        json!({}),
    ))
    .unwrap();
    assert_eq!(
        status["session_starts_without_context"],
        json!({"since":null, "causes":{}})
    );
    runtime.shutdown().unwrap();
}

#[test]
fn a_session_start_names_a_wrong_archive_path() {
    let dir = tempfile::tempdir().unwrap();
    let startup = Startup::for_test(
        crate::test_daemon::engine_config(dir.path().join("home")),
        &dir.path().join("absent"),
    );
    assert_eq!(
        respond(
            Harness::Codex,
            "SessionStart",
            &json!({"session_id":"s1", "cwd":"/home/op/morrow"}),
            &startup,
            Instant::now() + CONTEXT_BUDGET
        ),
        None
    );
    let spool = startup
        .engine
        .spool_dir(polylogue_sources::hook_spool::ORIGIN);
    let path = spool.join(format!("{}.ndjson", std::os::unix::process::parent_id()));
    let text = std::fs::read_to_string(path).unwrap();
    let record: Value = serde_json::from_str(text.trim()).unwrap();
    assert_eq!(record["event"], "SessionContext");
    assert_eq!(record["payload"]["cause"], "archive_path");
    assert_eq!(
        record["payload"]["detail"],
        format!(
            "no archive at {}; check the archive path",
            startup.config.archive.display()
        )
    );
}
