use super::read_current::Items;
use super::*;
use polylogue_model::read::ReadPage;

#[test]
fn cli_read_pages_cover_utf8_once_and_refuse_malformed_and_revoked_cursors() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let project = home.join(".claude/projects/-pages");
    std::fs::create_dir_all(&project).unwrap();
    let expected = "λ🙂 synthetic prompt with several pages";
    std::fs::write(project.join("pages.jsonl"), line("u1", "pages", expected)).unwrap();
    let env = crate::Fixture::new(&archive, &home, &dir.path().join("config"));
    let daemon = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 1 && status["store"]["search_pending"] == 0
    });
    let whole = cli(
        &env,
        &[
            "--deadline",
            "none",
            "--json",
            "session:claude-code:pages",
            "read",
            "--budget",
            "100000",
        ],
    );
    assert!(whole.status.success(), "{whole:?}");
    let whole: ReadPage = read_page(json(&whole));
    let finite_source = serde_json::to_vec(&whole.document.items).unwrap().len();
    let args = [
        "--deadline".into(),
        "none".into(),
        "--json".into(),
        "session:claude-code:pages".into(),
        "read".into(),
        "--budget".into(),
        "1".into(),
    ];
    let mut joined = Items::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut first_next = None;
    let mut complete = false;
    let out = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(out.status.success(), "{out:?}");
    let mut page = read_page(json(&out));
    for _ in 0..=finite_source {
        assert!(
            !page.document.items.is_empty(),
            "continuation made no item progress"
        );
        // Items validates distinct complete items and contiguous, advancing
        // fragment coordinates; every page consumes the finite source.
        joined.push(&page);
        let Some(next) = page.document.footer.next else {
            complete = true;
            break;
        };
        assert!(
            seen.insert(serde_json::to_string(&next.request).unwrap()),
            "repeated continuation"
        );
        first_next.get_or_insert_with(|| next.clone());
        page = followed_page(&env, &next);
    }
    assert!(complete, "finite source delivery did not terminate");
    assert_eq!(joined.text(), expected);
    let next = first_next.expect("one-token page must continue");
    let mut invalid = next.clone();
    assert!(
        !invalid.request.arguments["steps"][0]["cursor"]
            .as_str()
            .unwrap()
            .is_empty()
    );
    invalid.request.arguments["steps"][0]["cursor"] = "broken".into();
    let invalid = read_failure(follow(&env, &invalid));
    assert_eq!(
        serde_json::to_value(invalid).unwrap()["kind"],
        "invalid_query"
    );
    drop(daemon);
    polylogue_testkit::delete_index(&archive);
    let _rebuilt = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 1 && status["store"]["search_pending"] == 0
    });
    // A cursor from an earlier index generation is `stale_cursor`, never
    // restarted and never confused with a broken one (docs/query.md, Paging).
    let stale = read_failure(follow(&env, &next));
    assert_eq!(serde_json::to_value(stale).unwrap()["kind"], "stale_cursor");
}

/// A session of three short prompts, read in pages too small for all three.
pub(super) fn three_prompts(dir: &std::path::Path) -> (crate::Fixture, super::Daemon) {
    let home = dir.join("home");
    let project = home.join(".claude/projects/-pager");
    std::fs::create_dir_all(&project).unwrap();
    let lines: String = [
        ("u1", "opening question"),
        ("u2", "second marker beta"),
        ("u3", "third marker gamma"),
    ]
    .iter()
    .map(|(uuid, text)| line(uuid, "pager", text))
    .collect();
    std::fs::write(project.join("pager.jsonl"), lines).unwrap();
    let env = crate::Fixture::new(&dir.join("archive"), &home, &dir.join("config"));
    let daemon = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 3 && status["store"]["search_pending"] == 0
    });
    (env, daemon)
}

const SMALL_PAGE: &[&str] = &[
    "--deadline",
    "none",
    "session:claude-code:pager",
    "read",
    "--budget",
    "12",
];

/// The continue command a piped read printed on standard error.
fn continue_command(out: &std::process::Output) -> Option<String> {
    let stderr = String::from_utf8(out.stderr.clone()).unwrap();
    let command = stderr
        .lines()
        .find_map(|line| line.strip_prefix("continue: "))?;
    Some(command.to_owned())
}

#[test]
fn read_without_json_on_pipe_prints_continue_command_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = three_prompts(dir.path());
    let out = cli(&env, SMALL_PAGE);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8(out.stdout.clone()).unwrap();
    assert!(stdout.contains("opening question"), "{stdout}");
    assert!(!stdout.contains("third marker gamma"), "{stdout}");
    assert!(!stdout.contains("--cursor"), "the page alone is on stdout");
    let next = continue_command(&out).expect("a continue command on stderr");
    assert!(next.contains("--cursor"), "{next}");
    let continued = printed_cli(&env, &next);
    assert!(continued.status.success(), "{continued:?}");
}

#[test]
fn read_pages_tile_with_no_gap_or_overlap() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = three_prompts(dir.path());
    let mut args: Vec<String> = SMALL_PAGE.iter().map(|word| (*word).to_owned()).collect();
    args.extend(["--format".to_owned(), "json".to_owned()]);
    let mut joined = Items::default();
    let mut pages = 0;
    let mut out = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
    loop {
        assert!(out.status.success(), "{out:?}");
        let document: polylogue_model::read::ReadDocument =
            serde_json::from_slice(&out.stdout).unwrap();
        assert!(!document.items.is_empty(), "a page made no progress");
        joined.push_document(&document);
        pages += 1;
        assert!(pages <= 20, "the reading did not end");
        match continue_command(&out) {
            Some(next) => out = printed_cli(&env, &next),
            None => break,
        }
    }
    assert!(pages > 1, "the small budget must page");
    assert_eq!(
        joined.text(),
        "opening questionsecond marker betathird marker gamma"
    );
}

#[test]
fn read_on_terminal_shows_whole_session_through_pager() {
    use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
    use std::io::Read;
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = three_prompts(dir.path());
    let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
    let master = openpt(flags).unwrap();
    grantpt(&master).unwrap();
    unlockpt(&master).unwrap();
    let slave = ioctl_tiocgptpeer(&master, flags).unwrap();
    let mut command = env.command();
    command
        .env("PAGER", "cat")
        .env("NO_COLOR", "")
        .args(SMALL_PAGE)
        .stdin(std::process::Stdio::null())
        .stdout(std::fs::File::from(slave))
        .stderr(std::process::Stdio::piped());
    let child = command.spawn().unwrap();
    drop(command);
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Err(error) = std::fs::File::from(master).read_to_end(&mut bytes) {
            assert_eq!(error.raw_os_error(), Some(5));
        }
        bytes
    });
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let shown = String::from_utf8(reader.join().unwrap()).unwrap();
    // One document under one header, every message once.
    assert_eq!(
        shown
            .matches("session:claude-code:pager · transcript")
            .count(),
        1,
        "{shown}"
    );
    assert!(shown.contains("opening question"), "{shown}");
    for marker in ["second marker beta", "third marker gamma"] {
        assert_eq!(shown.matches(marker).count(), 1, "{marker} in {shown}");
    }
    let order: Vec<_> = [
        "opening question",
        "second marker beta",
        "third marker gamma",
    ]
    .iter()
    .map(|marker| shown.find(marker).unwrap())
    .collect();
    assert!(order.is_sorted(), "messages out of order: {shown}");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("continue:"));
    assert_eq!(
        super::seen::count(&env),
        1,
        "all pager pages acknowledge display"
    );
    let after = cli(
        &env,
        &["--json", "claude-code:pager", "read", "--from", "seen"],
    );
    let page = read_page(json(&after));
    assert!(
        !page
            .document
            .items
            .iter()
            .any(|item| item.source.message.is_some())
    );
}

#[test]
fn read_budget_none_zero_and_default_survive_real_cli_mcp_and_reopen() {
    use polylogue_model::read::ReadDocument;
    use serde_json::{Value, json};
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-budget");
    std::fs::create_dir_all(&project).unwrap();
    let text = "a ".repeat(500);
    let lines: String = (0..12)
        .map(|n| line(&format!("u{n}"), "budget", &text))
        .collect();
    std::fs::write(project.join("budget.jsonl"), lines).unwrap();
    let large = "λ🙂 finite transfer ".repeat(2500);
    std::fs::write(
        project.join("large.jsonl"),
        line("large-u", "large", &large),
    )
    .unwrap();
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let daemon = start_daemon(&env);
    wait_status(&env, |s| {
        s["store"]["messages"] == 13 && s["store"]["search_pending"] == 0
    });
    let run = |budget: Option<&str>| {
        let mut args = vec![
            "--deadline",
            "none",
            "session:claude-code:budget",
            "read",
            "--format",
            "json",
        ];
        if let Some(budget) = budget {
            args.extend(["--budget", budget]);
        }
        let out = cli(&env, &args);
        assert!(out.status.success(), "{out:?}");
        let document: ReadDocument = serde_json::from_slice(&out.stdout).unwrap();
        (out, document)
    };
    let (_, default) = run(None);
    assert!(
        default.footer.next.is_some(),
        "the default token allowance is finite"
    );
    let (_, unlimited) = run(Some("none"));
    assert!(
        unlimited.footer.next.is_none(),
        "this source fits the unchanged item/byte allowance"
    );
    let mut whole = Items::default();
    whole.push_document(&unlimited);
    assert_eq!(whole.text(), text.repeat(12));
    let (zero_out, zero) = run(Some("0"));
    assert!(
        !zero.items.is_empty(),
        "zero still makes minimum forward progress"
    );
    let zero_next = zero
        .footer
        .next
        .as_ref()
        .expect("zero must not become the default");
    assert_eq!(zero_next.request.arguments["steps"][0]["budget"], 0);
    let zero_command = continue_command(&zero_out).unwrap();
    assert!(zero_command.contains("--budget 0"), "{zero_command}");
    drop(daemon);
    let _reopened = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"] == 13);
    let continued = printed_cli(&env, &zero_command);
    assert!(continued.status.success(), "{continued:?}");
    let second: ReadDocument = serde_json::from_slice(&continued.stdout).unwrap();
    let previous = second
        .footer
        .previous
        .as_ref()
        .expect("previous zero-budget page");
    assert_eq!(previous.request.arguments["steps"][0]["budget"], 0);
    let returned = followed_page(&env, previous);
    assert_eq!(returned.document.items, zero.items);

    let mut mcp = super::query_surfaces::Mcp::start(&env);
    for budget in [None, Some(Value::Null), Some(json!(0)), Some(json!(4000))] {
        let mut step = json!({"verb":"read"});
        if let Some(value) = &budget {
            step["budget"] = value.clone();
        }
        let reply = mcp.request("tools/call", json!({"name":"polylogue_read", "arguments":{
            "operation":"rows", "arguments":{"selection":"session:claude-code:budget", "steps":[step]},
            "request_id":uuid::Uuid::now_v7().to_string(), "deadline":null
        }}));
        assert!(reply["error"].is_null(), "{reply}");
        let answer: command_surface::Answer<Value> =
            serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap();
        assert_eq!(answer.exit_code(), 0, "{answer:?}");
        let page = read_page(answer.value().unwrap().clone());
        let expected = match budget {
            Some(Value::Null) => &unlimited.items,
            Some(Value::Number(ref n)) if n.as_u64() == Some(0) => &zero.items,
            _ => &default.items,
        };
        // Fresh invocations own different query references on transfer fragments.
        // Their encoded source item and advancing coordinates are the same.
        assert_eq!(
            page.document
                .items
                .iter()
                .map(|item| &item.kind)
                .collect::<Vec<_>>(),
            expected.iter().map(|item| &item.kind).collect::<Vec<_>>()
        );
        for item in &page.document.items {
            if matches!(item.kind, polylogue_model::read::ItemKind::Fragment { .. }) {
                assert_eq!(item.source.reference, page.snapshot.query);
            }
        }
    }
    let out = cli(
        &env,
        &[
            "--deadline",
            "none",
            "session:claude-code:large",
            "read",
            "--format",
            "json",
            "--budget",
            "none",
        ],
    );
    assert!(out.status.success(), "{out:?}");
    let mut document: ReadDocument = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        document.footer.next.is_some(),
        "unlimited tokens retain the transfer bound"
    );
    let mut items = Items::default();
    let mut output = out;
    let mut calls = std::collections::BTreeSet::new();
    loop {
        items.push_document(&document);
        let Some(command) = continue_command(&output) else {
            break;
        };
        assert!(command.contains("--budget none"), "{command}");
        assert!(calls.insert(command.clone()), "continuation must advance");
        output = printed_cli(&env, &command);
        assert!(output.status.success(), "{output:?}");
        document = serde_json::from_slice(&output.stdout).unwrap();
    }
    assert_eq!(items.text(), large);
    let mut request = command_surface::Request {
        operation: "rows".into(),
        arguments: json!({"selection":"session:claude-code:large", "steps":[{"verb":"read", "budget":null}]}),
        request_id: uuid::Uuid::now_v7().to_string(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let mut first_items = None;
    for page_number in 0..3 {
        let reply = mcp.request(
            "tools/call",
            json!({"name":"polylogue_read", "arguments":request}),
        );
        assert!(reply["error"].is_null(), "{reply}");
        let answer: command_surface::Answer<Value> =
            serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap();
        assert_eq!(answer.exit_code(), 0, "{answer:?}");
        let page = read_page(answer.value().unwrap().clone());
        if page_number == 2 {
            assert_eq!(page.document.items, first_items.unwrap());
            break;
        }
        let call = if page_number == 0 {
            first_items = Some(page.document.items.clone());
            page.document
                .footer
                .next
                .as_ref()
                .expect("MCP none still has a transfer continuation")
        } else {
            page.document
                .footer
                .previous
                .as_ref()
                .expect("MCP previous transfer")
        };
        assert!(
            call.request.arguments["steps"][0]
                .as_object()
                .unwrap()
                .contains_key("budget")
        );
        assert_eq!(call.request.arguments["steps"][0]["budget"], Value::Null);
        request = call.request.clone();
    }
}
