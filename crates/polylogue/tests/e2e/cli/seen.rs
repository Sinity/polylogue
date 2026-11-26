//! The actual output destination, rather than merely asking for a page, marks it.
use super::*;
use serde_json::{Value, json};

pub(super) fn count(env: &crate::Fixture) -> u64 {
    let barrier = cli(env, &["set", "browser.actions", "on"]);
    assert!(barrier.status.success(), "{barrier:?}");
    let output = cli(env, &["--json", "annotations where kind:seen"]);
    query_rows(&output)["count"].as_u64().unwrap()
}

#[test]
fn pipe_file_and_finite_stream_do_not_mark_but_explicit_pager_does() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = super::read_pages::three_prompts(dir.path());
    let target = "session:claude-code:pager";
    let pipe = cli(
        &env,
        &[
            "--json", target, "read", "--from", "seen", "--budget", "100000",
        ],
    );
    assert!(pipe.status.success(), "{pipe:?}");
    assert_eq!(count(&env), 0);
    let path = dir.path().join("reading.md");
    let file = cli(&env, &[target, "read", "--to", path.to_str().unwrap()]);
    assert!(file.status.success(), "{file:?}");
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("third marker gamma")
    );
    assert_eq!(count(&env), 0);
    let bundle = format!("{}/bundle/", dir.path().display());
    let stream = cli(&env, &[target, "read", "--to", &bundle]);
    assert!(stream.status.success(), "{stream:?}");
    assert_eq!(count(&env), 0);

    let displayed = env
        .command()
        .env("PAGER", "cat")
        .args([target, "read", "--to", "pager"])
        .output()
        .unwrap();
    assert!(displayed.status.success(), "{displayed:?}");
    assert_eq!(count(&env), 1);
    // What the session appends after the displayed reading is its remainder.
    let mut capture = std::fs::OpenOptions::new()
        .append(true)
        .open(dir.path().join("home/.claude/projects/-pager/pager.jsonl"))
        .unwrap();
    capture
        .write_all(line("u4", "pager", "fourth marker delta").as_bytes())
        .unwrap();
    drop(capture);
    wait_status(&env, |status| {
        status["store"]["messages"] == 4 && status["store"]["search_pending"] == 0
    });
    let after = cli(
        &env,
        &[
            "--json", target, "read", "--from", "seen", "--budget", "100000",
        ],
    );
    let after = read_page(json(&after));
    let mut text = super::read_current::Items::default();
    text.push(&after);
    assert_eq!(text.text(), "fourth marker delta");
    let file_after = dir.path().join("after.json");
    let out = cli(
        &env,
        &[
            target,
            "read",
            "--from",
            "seen",
            "--format",
            "json",
            "--to",
            file_after.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{out:?}");
    let document: polylogue_model::read::ReadDocument =
        serde_json::from_slice(&std::fs::read(file_after).unwrap()).unwrap();
    let mut items = super::read_current::Items::default();
    items.push_document(&document);
    assert_eq!(items.text(), "fourth marker delta");
    assert_eq!(
        count(&env),
        1,
        "exporting the remainder cannot advance its display mark"
    );
    let again = cli(
        &env,
        &[
            "--json", target, "read", "--from", "seen", "--budget", "100000",
        ],
    );
    let again = read_page(json(&again));
    let mut items = super::read_current::Items::default();
    items.push(&again);
    assert_eq!(
        items.text(),
        "fourth marker delta",
        "export must leave the accepted mark unchanged"
    );
}

#[test]
fn actual_mcp_returned_page_marks_and_from_seen_reads_the_remainder() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = super::read_pages::three_prompts(dir.path());
    let large = "large native MCP remainder ".repeat(2000);
    assert!(large.len() > polylogue_model::read::PageBudget::default().bytes.get());
    let mut capture = std::fs::OpenOptions::new()
        .append(true)
        .open(dir.path().join("home/.claude/projects/-pager/pager.jsonl"))
        .unwrap();
    capture
        .write_all(line("u4", "pager", &large).as_bytes())
        .unwrap();
    drop(capture);
    wait_status(&env, |status| {
        status["store"]["messages"] == 4 && status["store"]["search_pending"] == 0
    });
    let mut mcp = super::query_surfaces::Mcp::start(&env);
    let request = |step: Value| {
        command_surface::RunnableRequest::new(
            "rows",
            json!({"selection":"session:claude-code:pager", "steps":[step]}),
            uuid::Uuid::now_v7().to_string(),
        )
    };
    let mut read = |call: command_surface::RunnableRequest| {
        let reply = mcp.request(
            "tools/call",
            json!({"name":"polylogue_read", "arguments":call}),
        );
        assert!(reply["error"].is_null(), "{reply}");
        assert_eq!(reply["result"]["isError"], false, "{reply}");
        let answer: command_surface::Answer<Value> =
            serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap();
        read_page(answer.value().unwrap().clone())
    };
    // One item a page, until the first message is shown whole; each page
    // the MCP call returns is displayed.
    let mut page = read(request(
        json!({"verb":"read", "budget":1, "hide":["annotations"]}),
    ));
    let mut text = super::read_current::Items::default();
    text.push(&page);
    while text.text() != "opening question" {
        let next = page.document.footer.next.expect("a next page");
        page = read(next.request);
        text.push(&page);
    }
    assert_eq!(count(&env), 1);
    let mut page = read(request(
        json!({"verb":"read", "from":"seen", "budget":100000}),
    ));
    let mut items = super::read_current::Items::default();
    let mut saw_fragment = false;
    let mut cursors = std::collections::BTreeSet::new();
    loop {
        saw_fragment |= page
            .document
            .items
            .iter()
            .any(|item| matches!(item.kind, polylogue_model::read::ItemKind::Fragment { .. }));
        items.push(&page);
        let Some(next) = page.document.footer.next else {
            break;
        };
        assert!(
            cursors.insert(serde_json::to_string(&next.request).unwrap()),
            "native MCP continuation advances"
        );
        page = read(next.request);
    }
    assert!(
        saw_fragment,
        "the actual MCP body must cross its byte transfer bound"
    );
    assert!(!cursors.is_empty());
    assert_eq!(
        items.text(),
        format!("second marker betathird marker gamma{large}")
    );
    assert_eq!(count(&env), 1);
    let end = read(request(
        json!({"verb":"read", "from":"seen", "budget":100000}),
    ));
    assert!(
        !end.document
            .items
            .iter()
            .any(|item| item.source.message.is_some())
    );
}

#[test]
fn logical_read_reuses_physical_progress_through_cli_and_actual_mcp() {
    let dir = tempfile::tempdir().unwrap();
    let (env, _daemon) = super::read_pages::three_prompts(dir.path());
    let project = dir.path().join("home/.claude/projects/-pager");
    let mut source = std::fs::OpenOptions::new()
        .append(true)
        .open(project.join("pager.jsonl"))
        .unwrap();
    writeln!(
        source,
        "{}",
        json!({"type":"continued-in","sessionId":"pager","continuedInSessionId":"cont"})
    )
    .unwrap();
    drop(source);
    std::fs::write(
        project.join("cont.jsonl"),
        line("c1", "cont", "already read continuation"),
    )
    .unwrap();
    wait_status(&env, |status| {
        status["store"]["messages"] == 4 && status["store"]["search_pending"] == 0
    });
    let displayed = env
        .command()
        .env("PAGER", "cat")
        .args(["session:claude-code:cont", "read", "--to", "pager"])
        .output()
        .unwrap();
    assert!(displayed.status.success(), "{displayed:?}");
    assert_eq!(count(&env), 1);
    let target = "root:claude-code:pager";
    let pipe = cli(
        &env,
        &[
            "--json", target, "read", "--from", "seen", "--budget", "100000",
        ],
    );
    let mut items = super::read_current::Items::default();
    items.push(&read_page(json(&pipe)));
    assert_eq!(
        items.text(),
        "opening questionsecond marker betathird marker gamma"
    );
    assert_eq!(count(&env), 1);
    let bundle = format!("{}/logical-bundle/", dir.path().display());
    let stream = cli(&env, &[target, "read", "--from", "seen", "--to", &bundle]);
    assert!(stream.status.success(), "{stream:?}");
    assert_eq!(
        count(&env),
        1,
        "exporting a composed stream cannot mark its unread root"
    );
    let mut mcp = super::query_surfaces::Mcp::start(&env);
    let mut read = |selection: &str| {
        let request = command_surface::RunnableRequest::new(
            "rows",
            json!({"selection":selection,"steps":[{"verb":"read","from":"seen","budget":100000}]}),
            uuid::Uuid::now_v7().to_string(),
        );
        let reply = mcp.request(
            "tools/call",
            json!({"name":"polylogue_read","arguments":request}),
        );
        assert_eq!(reply["result"]["isError"], false, "{reply}");
        let answer: command_surface::Answer<Value> =
            serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap();
        read_page(answer.value().unwrap().clone())
    };
    // MCP's computed author is independent from the CLI's. Establish its own
    // physical continuation progress before reading that author's logical root.
    let mut items = super::read_current::Items::default();
    items.push(&read("session:claude-code:cont"));
    assert_eq!(items.text(), "already read continuation");
    assert_eq!(count(&env), 2);
    let mut items = super::read_current::Items::default();
    items.push(&read(target));
    assert_eq!(
        items.text(),
        "opening questionsecond marker betathird marker gamma"
    );
    assert_eq!(count(&env), 3, "MCP advances its own root physical mark");
    let mut items = super::read_current::Items::default();
    items.push(&read(target));
    assert_eq!(items.text(), "");
    let again = cli(
        &env,
        &[
            "--json", target, "read", "--from", "seen", "--budget", "100000",
        ],
    );
    let mut items = super::read_current::Items::default();
    items.push(&read_page(json(&again)));
    assert_eq!(
        items.text(),
        "opening questionsecond marker betathird marker gamma",
        "MCP cannot consume the CLI reader's root progress"
    );
}
