use crate::test_daemon;
use command_surface::Suggestion;
use polylogue_model::read::View;

fn replace(daemon: &test_daemon::TestDaemon, source: String) {
    let status = test_daemon::Call::new("status", serde_json::json!({})).answer(&daemon.daemon);
    let before = status.value().unwrap()["store"]["revisions"]
        .as_u64()
        .unwrap();
    std::fs::write(daemon.home.join(".claude/projects/-p/s1.jsonl"), source).unwrap();
    daemon.wait_for(|status| {
        status["store"]["revisions"].as_u64().unwrap() > before
            && status["store"]["search_pending"] == 0
    });
}

#[test]
fn stale_cursor_failures_suggest_rerun() {
    for (case, target, view) in [
        (
            "rewritten session",
            "session:claude-code:s1",
            View::Transcript,
        ),
        (
            "removed row",
            "messages where words >= 5 | sort by id",
            View::Transcript,
        ),
        ("changed links", "session:claude-code:s1", View::Tree),
    ] {
        let daemon =
            test_daemon::start("the original synthetic question has enough text to continue");
        let user = |id: &str, text: &str| polylogue_testkit::claude_user(id, "s1", text);
        let link = |child: &str| {
            format!(
                "{}\n",
                serde_json::json!({"type":"continued-in", "sessionId":"s1", "continuedInSessionId":child})
            )
        };
        if case == "changed links" {
            replace(
                &daemon,
                format!("{}{}", user("u1", "the root session"), link("first-child")),
            );
        }
        let response = test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection": target, "steps":[{"verb":"read", "view": view, "budget":1}]}),
        )
        .answer(&daemon.daemon);
        let page = crate::test_answers::read(response);
        let cursor = page.document.footer.next.expect(case).request;
        match case {
            "rewritten session" => replace(
                &daemon,
                user("u1", "the question was rewritten after its first page"),
            ),
            "removed row" => replace(&daemon, user("u1", "short")),
            "changed links" => replace(
                &daemon,
                format!("{}{}", user("u1", "the root session"), link("second-child")),
            ),
            _ => unreachable!(),
        }
        let response = test_daemon::Call::from_request(&cursor).answer(&daemon.daemon);
        let failure = crate::test_answers::refusal(response);
        assert_eq!(
            failure.code.as_deref(),
            Some("stale_cursor"),
            "{case}: {failure:?}"
        );
        let view_option = if view == View::Transcript {
            String::new()
        } else {
            format!(" --view {view}")
        };
        let command = format!(
            "; run `polylogue {} read{view_option}` again to start over",
            polylogue_render::shell_quote(target)
        );
        assert!(failure.message.ends_with(&command), "{case}: {failure:?}");
        let Some(Suggestion::Request(repeated)) = failure.suggestion() else {
            panic!("{case}: {failure:?}")
        };
        assert_eq!(repeated.operation, cursor.operation, "{case}");
        let expected = crate::protocol::line::Rows {
            selection: Some(target.into()),
            steps: vec![crate::protocol::line::Step::Read(Box::new(
                crate::protocol::steps::ReadStep {
                    options: polylogue_model::read::ReadOptions {
                        view,
                        ..Default::default()
                    },
                    format: Some("json".into()),
                    budget: polylogue_model::read::ReadBudget::Tokens(1),
                    ..Default::default()
                },
            ))],
            ..Default::default()
        };
        let actual: crate::protocol::line::Rows =
            serde_json::from_value(repeated.arguments.clone()).unwrap();
        assert_eq!(actual, expected, "{case}");
        assert_eq!(repeated.cursor, None, "{case}");
        let restarted = test_daemon::Call::from_request(repeated).answer(&daemon.daemon);
        assert!(restarted.value().is_some(), "{case}: {restarted:?}");
        if case == "removed row" {
            let restarted = crate::test_answers::read(restarted);
            assert_eq!(restarted.document.items, []);
        }
    }
}
