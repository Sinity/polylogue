use super::read_current::Items;
use super::*;
use polylogue_model::read::ReadPage;

fn next_arguments(call: &polylogue_model::read::ReadyCall) -> Vec<String> {
    let stages = polylogue_model::command::effective_stages(&call.cli);
    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].program, "polylogue");
    let mut args = vec!["--json".to_owned()];
    args.extend(stages[0].args.clone());
    args
}
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
            "read",
            "session:claude-code:pages",
            "--page-tokens",
            "100000",
        ],
    );
    assert!(whole.status.success(), "{whole:?}");
    let whole: ReadPage = serde_json::from_value(json(&whole)).unwrap();
    let finite_source = serde_json::to_vec(&whole.document.items).unwrap().len();
    let mut args = vec![
        "--deadline".into(),
        "none".into(),
        "--json".into(),
        "read".into(),
        "session:claude-code:pages".into(),
        "--page-tokens".into(),
        "1".into(),
    ];
    let mut joined = Items::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut first_next = None;
    let mut complete = false;
    for _ in 0..=finite_source {
        let out = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(out.status.success(), "{out:?}");
        let page: ReadPage = serde_json::from_value(json(&out)).unwrap();
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
        assert!(seen.insert(next.cli.clone()), "repeated continuation");
        first_next.get_or_insert_with(|| next.clone());
        args = next_arguments(&next);
    }
    assert!(complete, "finite source delivery did not terminate");
    assert_eq!(joined.text(), expected);
    let next = first_next.expect("one-token page must continue");
    let mut invalid = next_arguments(&next);
    let cursor = invalid.iter().position(|word| word == "--cursor").unwrap() + 1;
    assert!(!invalid[cursor].is_empty());
    invalid[cursor] = "broken".into();
    let invalid = cli(
        &env,
        &invalid.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    assert!(!invalid.status.success());
    assert_eq!(json(&invalid)["error"]["kind"], "invalid_query");
    drop(daemon);
    polylogue_testkit::delete_index(&archive);
    let _rebuilt = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"] == 1 && status["store"]["search_pending"] == 0
    });
    // A cursor from an earlier index generation is `stale_cursor`, never
    // restarted and never confused with a broken one (docs/query.md, Paging).
    let args = next_arguments(&next);
    let stale = cli(&env, &args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(!stale.status.success());
    assert_eq!(json(&stale)["error"]["kind"], "stale_cursor");
}
