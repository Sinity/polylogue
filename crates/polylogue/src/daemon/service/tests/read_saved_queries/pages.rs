use super::*;
use command_surface::RunnableRequest;

fn first(d: &Daemon, target: &str, id: &str) -> ReadPage {
    let parsed = crate::cli::line::parse(
        &[
            target.to_owned(),
            "read".into(),
            "--budget".into(),
            "1024".into(),
        ],
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    let steps = parsed
        .steps
        .into_iter()
        .map(|step| step.step)
        .collect::<Vec<_>>();
    crate::test_answers::read(
        Call::new("rows", json!({"selection":target,"steps":steps}))
            .id(id)
            .answer(d),
    )
}
fn next(d: &Daemon, request: &RunnableRequest) -> ReadPage {
    crate::test_answers::read(Call::from_request(request).answer(d))
}
fn joined(pages: &[ReadPage]) -> Vec<ReadItem> {
    let mut items = Vec::new();
    let mut pending = String::new();
    let mut hash = None;
    for page in pages {
        assert!(
            serde_json::to_vec(&page.document).unwrap().len() <= PageBudget::default().bytes.get()
        );
        assert!(serde_json::to_vec(&page.document.header).unwrap().len() < 1024);
        for item in &page.document.items {
            if let ItemKind::Fragment {
                text,
                position,
                until,
                total,
            } = &item.kind
            {
                assert_eq!(position.character as usize, pending.chars().count());
                if let Some(expected) = hash {
                    assert_eq!(position.item_hash, expected)
                } else {
                    hash = Some(position.item_hash)
                }
                pending.push_str(text);
                assert_eq!(*until as usize, pending.chars().count());
                if until == total {
                    let complete: ReadItem = serde_json::from_str(&pending).unwrap();
                    assert_eq!(
                        polylogue_model::BlobHash::of(&serde_json::to_vec(&complete).unwrap()),
                        hash.unwrap()
                    );
                    items.push(complete);
                    pending.clear();
                    hash = None;
                }
            } else {
                assert!(pending.is_empty(), "no item may interrupt a fragment");
                items.push(item.clone());
            }
        }
    }
    assert!(pending.is_empty());
    items
}
fn projected(items: Vec<ReadItem>, template: &ReadPage) -> ReadPage {
    let mut page = template.clone();
    page.document.items = items;
    page
}

#[test]
fn saved_history_pages_preserve_text_actions_rows_and_both_directions_after_restart() {
    let mut d = test_daemon::start("history page source");
    d.pause_signals();
    for n in 0..3 {
        let text = format!(
            "sessions where title:{} OR kind:*",
            json!(format!("revision-{n}-{}", "x".repeat(10_000)))
        );
        save(&d.daemon, &format!("large-{n}"), "focus", &text, false);
    }
    save(
        &d.daemon,
        "dependent",
        "summary",
        "query:focus | sort by id",
        false,
    );
    let expected = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .saved_query_record("focus")
        .unwrap()
        .unwrap();
    assert!(serde_json::to_vec(&expected).unwrap().len() > PageBudget::default().bytes.get());
    let start = first(&d.daemon, "query:focus", "history-reading-one");
    assert!(start.document.footer.next.is_some());
    assert!(
        start
            .document
            .items
            .iter()
            .any(|item| matches!(item.kind, ItemKind::Fragment { .. }))
    );
    let mut forward = vec![start];
    d.restart();
    d.pause_signals();
    for _ in 0..128 {
        let Some(call) = forward.last().unwrap().document.footer.next.clone() else {
            break;
        };
        forward.push(next(&d.daemon, &call.request));
    }
    assert!(
        forward.last().unwrap().document.footer.next.is_none(),
        "finite history must finish"
    );
    // Old saved-history cursors omit the per-reading seed. Decode them,
    // then refuse the obsolete layout explicitly after daemon reopen.
    let mut old = forward[0]
        .document
        .footer
        .next
        .as_ref()
        .unwrap()
        .request
        .clone();
    let reader = d.daemon.reads.reader().unwrap();
    let seal = CursorSeal {
        key: reader.cursor_key().unwrap(),
        generation: reader.change_position().unwrap().generation,
    };
    let mut cursor =
        PageCursor::open(old.arguments["steps"][0]["cursor"].as_str().unwrap(), &seal).unwrap();
    let PagePosition::Record { history_seed, .. } = &mut cursor.position else {
        panic!("row cursor")
    };
    *history_seed = None;
    old.arguments["steps"][0]["cursor"] = json!(cursor.seal(&seal));
    old.request_id = "old-saved-history-layout".into();
    drop(reader);
    let failure = crate::test_answers::refusal(Call::from_request(&old).answer(&d.daemon));
    assert_eq!(failure.code.as_deref(), Some("stale_cursor"));
    let complete = projected(joined(&forward), &forward[0]);
    assert_eq!(history(&complete), expected);
    let commands = resaves(&complete);
    assert_eq!(commands.len(), 1);
    assert_eq!(
        commands[0].request.arguments,
        json!({"selection":"query:focus | sort by id","steps":[{"verb":"save","name":"summary"}]})
    );
    let data = data_items(&complete);
    assert_eq!(data, read(&d.daemon, "sessions").document.items);
    let mut backward = vec![forward.last().unwrap().clone()];
    for _ in 0..128 {
        let Some(call) = backward.last().unwrap().document.footer.previous.clone() else {
            break;
        };
        backward.push(next(&d.daemon, &call.request));
    }
    assert!(backward.last().unwrap().document.footer.previous.is_none());
    backward.reverse();
    assert_eq!(joined(&backward), complete.document.items);
    // Different accepted reads have independent actions; retrying the same
    // accepted first request retains the original reading and action IDs.
    let mut second = vec![first(&d.daemon, "query:focus", "history-reading-two")];
    for _ in 0..128 {
        let Some(call) = second.last().unwrap().document.footer.next.clone() else {
            break;
        };
        second.push(next(&d.daemon, &call.request));
    }
    assert!(second.last().unwrap().document.footer.next.is_none());
    let second = projected(joined(&second), &second[0]);
    assert_ne!(
        resaves(&second)[0].request.request_id,
        commands[0].request.request_id
    );
    let retried = first(&d.daemon, "query:focus", "history-reading-one");
    assert_eq!(retried.document.items, forward[0].document.items);
    let encoded = crate::commands::declarations()
        .cli_command(&commands[0].request)
        .unwrap();
    let (words, bytes) = crate::cli::line::tests::shell_arguments(&encoded);
    assert!(bytes.is_empty());
    crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    assert!(matches!(
        crate::test_answers::step(Call::from_request(&commands[0].request).answer(&d.daemon)),
        crate::protocol::line::StepAnswer::Saved(_)
    ));
}

#[test]
fn saved_history_cursor_rejects_changed_history_and_retains_authored_self_aliases() {
    let mut d = test_daemon::start("history identity");
    d.pause_signals();
    let long = format!(
        "sessions where title:{} OR kind:*",
        json!("x".repeat(10_000))
    );
    save(&d.daemon, "first-focus", "focus", &long, false);
    save(&d.daemon, "authored-alias", "alias", "query:focus", false);
    let alias_page = read(&d.daemon, "query:focus");
    assert_eq!(
        resaves(&alias_page).len(),
        1,
        "a named self-alias remains a real dependent"
    );
    let page = first(&d.daemon, "query:focus", "before-change");
    let call = page
        .document
        .footer
        .next
        .as_ref()
        .expect("small page has continuation")
        .request
        .clone();
    save(&d.daemon, "second-focus", "focus", "messages", false);
    let failure = crate::test_answers::refusal(Call::from_request(&call).answer(&d.daemon));
    assert_eq!(failure.code.as_deref(), Some("stale_cursor"));
    assert_eq!(
        resaves(&read(&d.daemon, "query:focus")).len(),
        1,
        "the alias is still a dependent after the history changed"
    );
}

#[test]
fn a_saved_empty_selection_still_delivers_its_history() {
    let mut d = test_daemon::start("empty saved selection");
    d.pause_signals();
    save(
        &d.daemon,
        "empty-focus",
        "focus",
        "sessions where kind:subagent",
        false,
    );
    let page = read(&d.daemon, "query:focus");
    assert_eq!(history(&page).revisions.len(), 1);
    assert_eq!(history(&page).kept.len(), 1);
    assert!(data_items(&page).is_empty());
    assert!(page.document.footer.next.is_none());
}

#[test]
fn saved_query_stream_emits_history_records_before_selected_rows() {
    let mut d = test_daemon::start("saved history stream");
    d.pause_signals();
    save(&d.daemon, "stream-focus", "focus", "sessions", false);
    let expected = read(&d.daemon, "query:focus");
    let request = crate::daemon::verbs::read::request_for(
        crate::protocol::steps::ReadStep {
            delivery: ReadDelivery::Stream,
            format: Some("jsonl".into()),
            ..Default::default()
        },
        "query:focus".into(),
    )
    .unwrap();
    let actor = polylogue_model::authorship::Authorship {
        author: polylogue_model::Author::Operator,
        via: polylogue_model::authorship::Channel::Cli,
    };
    let operation = d.daemon.operation_context(&actor).unwrap();
    let reader = d.daemon.reads.reader().unwrap();
    let mut bytes = Vec::new();
    polylogue_reading::stream(&reader, d.daemon.as_ref(), &request, &operation, &mut bytes)
        .unwrap();
    let records = std::str::from_utf8(&bytes)
        .unwrap()
        .split("\n\n")
        .filter(|frame| frame.lines().any(|line| line.trim() == "event: record"))
        .map(|frame| {
            let data = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim_start)
                .collect::<Vec<_>>()
                .join("\n");
            serde_json::from_str::<ReadRecord>(&data).unwrap()
        })
        .collect::<Vec<_>>();
    let prefix = records
        .iter()
        .take_while(|record| matches!(record, ReadRecord::Presentation { .. }))
        .flat_map(|record| record.parts())
        .map(|part| {
            let ReadPart::Item(item) = part else {
                panic!("history item")
            };
            item.clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(history(&projected(prefix, &expected)), history(&expected));
    let rows = records
        .iter()
        .skip_while(|record| matches!(record, ReadRecord::Presentation { .. }))
        .flat_map(|record| record.parts())
        .filter_map(|part| match part {
            ReadPart::Item(item) => Some(item.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(rows, data_items(&expected));
}
