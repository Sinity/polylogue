use crate::test_daemon;
use command_surface::{Outcome, Suggestion};
use polylogue_model::{
    SessionId,
    read::{ReadPage, View},
};

fn fingerprint(daemon: &test_daemon::TestDaemon) -> String {
    let reader = daemon.daemon.reads.reader().unwrap();
    let session = SessionId::parse("claude-code:s1").unwrap();
    let index = reader.item_index(&session).unwrap().unwrap();
    format!("{}:{}", index.rewrite_version, index.entries.len())
}

fn replace(daemon: &test_daemon::TestDaemon, source: String, messages: u64) {
    let before = fingerprint(daemon);
    std::fs::write(daemon.home.join(".claude/projects/-p/s1.jsonl"), source).unwrap();
    daemon.wait_for(|status| {
        status["store"]["messages"] == messages && fingerprint(daemon) != before
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
        ("removed row", "messages | sort by id", View::Transcript),
        ("changed links", "session:claude-code:s1", View::Links),
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
        if case == "removed row" {
            replace(
                &daemon,
                format!(
                    "{}{}",
                    user("u1", "the original first row has enough text to continue"),
                    user("u2", "the second row remains")
                ),
                2,
            );
        } else if case == "changed links" {
            replace(
                &daemon,
                format!("{}{}", user("u1", "the root session"), link("first-child")),
                1,
            );
        }
        let response = test_daemon::Call::new(
            "read",
            serde_json::json!({"target": target, "view": view, "page_tokens": 1}),
        )
        .answer(&daemon.daemon);
        let page: ReadPage = serde_json::from_value(response.value().expect(case).clone()).unwrap();
        let cursor = page.document.footer.next.expect(case).request;
        match case {
            "rewritten session" => replace(
                &daemon,
                user("u1", "the question was rewritten after its first page"),
                1,
            ),
            "removed row" => replace(&daemon, user("u2", "the second row remains"), 1),
            "changed links" => replace(
                &daemon,
                format!("{}{}", user("u1", "the root session"), link("second-child")),
                1,
            ),
            _ => unreachable!(),
        }
        let response = test_daemon::Call::from_request(&cursor).answer(&daemon.daemon);
        let Outcome::Error(failure) = &response.outcome else {
            panic!("{case}: {response:?}")
        };
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
            "; run `polylogue read {}{view_option}` again to start over",
            polylogue_render::shell_quote(target)
        );
        assert!(failure.message.ends_with(&command), "{case}: {failure:?}");
        let Some(Suggestion::Request(repeated)) = failure.suggestion() else {
            panic!("{case}: {failure:?}")
        };
        assert_eq!(repeated.operation, cursor.operation, "{case}");
        assert_eq!(repeated.arguments, cursor.arguments, "{case}");
        assert_eq!(repeated.cursor, None, "{case}");
        let restarted = test_daemon::Call::from_request(repeated).answer(&daemon.daemon);
        assert!(restarted.value().is_some(), "{case}: {restarted:?}");
    }
}
