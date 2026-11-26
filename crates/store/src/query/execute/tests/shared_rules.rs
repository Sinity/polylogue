use super::*;
use polylogue_model::live::{self, LiveFacts, ProcessIdentity};

#[test]
fn an_idle_live_root_is_running_in_every_live_read() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let now = options().clock.now.as_millisecond();
    let pk = f
        .session(SessionFacts {
            id: "test:idle",
            cwd: Some("/a/b"),
            last: Some(now - 2 * live::STALE_FOR_MS),
            ..Default::default()
        })
        .unwrap();
    f.live(pk, "hook_stop", now - 2 * live::STALE_FOR_MS, None, None)
        .unwrap();
    let mut facts: LiveFacts = f
        .reads()
        .unwrap()
        .reader()
        .unwrap()
        .live_state("test:idle", now)
        .unwrap()
        .unwrap()
        .facts;
    facts.root = Some(ProcessIdentity {
        boot: "test-boot".into(),
        pid: 1,
        start: 1,
    });
    facts.edited_files.insert("src/main.rs".into());
    f.live_facts(pk, &facts).unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    assert_eq!(
        r.running_candidates(now).unwrap().at(now)[0].state,
        live::LiveState::Idle
    );
    assert_eq!(r.live_roots().unwrap()[0].session, "test:idle");
    assert_eq!(r.live_edited_files(now).unwrap()[0].files, ["src/main.rs"]);
    assert!(r.live_untils(now).unwrap().is_empty());
    assert_eq!(
        group_rows(&r, "sessions where running:true | count")[0]["count"],
        1
    );
}

#[test]
fn a_one_turn_session_costs_what_its_turn_costs() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let pk = f
        .session(SessionFacts {
            id: "codex:cost",
            ..Default::default()
        })
        .unwrap();
    for (id, role, by, cost, active) in [
        ("codex:prompt", "user", "human", None, true),
        ("codex:priced", "assistant", "model", Some(7), true),
        ("codex:unpriced", "assistant", "model", None, true),
        ("codex:other-branch", "assistant", "model", Some(11), false),
    ] {
        f.message(
            pk,
            MessageFacts {
                id,
                role,
                written_by: by,
                cost_nusd: cost,
                on_active_path: Some(active),
                ..Default::default()
            },
        )
        .unwrap();
    }
    rusqlite::Connection::open(f.archive.index_db())
        .unwrap()
        .execute("UPDATE message SET input_tokens=3,output_tokens=5", [])
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let mut opts = options();
    opts.origins.push("codex".into());
    opts.fields = ["cost", "cost_complete", "input_tokens", "output_tokens"]
        .map(str::to_owned)
        .to_vec();
    let rows = |text| {
        r.query_page(&crate::query::compile(text, &opts).unwrap(), None)
            .unwrap()
            .rows
    };
    let session = rows("sessions");
    let turn = rows("turns");
    assert_eq!(turn[0]["cost"], 18);
    assert_eq!(turn[0]["cost_complete"], 0);
    assert_eq!(turn[0]["cost"], session[0]["cost"]);
    assert_eq!(turn[0]["cost_complete"], session[0]["cost_complete"]);
    let mut turn_options = opts.clone();
    turn_options.fields = vec!["answer".into()];
    let answers = r
        .query_page(
            &crate::query::compile("turns", &turn_options).unwrap(),
            None,
        )
        .unwrap();
    assert_eq!(answers.rows[0]["answer"], "codex:unpriced");
    assert_eq!(
        group_rows(&r, "turns | agg sum(cost)")[0]["partial_cost"],
        1
    );

    for (column, expected) in [("input_tokens", 9), ("output_tokens", 15)] {
        assert_eq!(turn[0][column], expected);
        assert_eq!(turn[0][column], session[0][column]);
    }
}

#[test]
fn a_resume_row_always_names_its_reason() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let pk = f
        .session(SessionFacts {
            id: "test:slash",
            ..Default::default()
        })
        .unwrap();
    f.message(
        pk,
        MessageFacts {
            id: "test:command",
            role: "user",
            written_by: "human",
            prompt_kind: Some("command"),
            text: Some("/clear"),
            ..Default::default()
        },
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    assert!(
        !r.open_items("test:slash")
            .unwrap()
            .unwrap()
            .unanswered_prompt
    );
    assert!(!r.session_summary(pk).unwrap().unwrap().unanswered_prompt);
    assert_eq!(
        group_rows(&r, "sessions where ends_open:true | count")[0]["count"],
        0
    );
}

#[test]
fn path_under_agrees_in_every_read() {
    use crate::fixture::writer::{WriterArchive, prose, sid};
    use polylogue_model::{RepoLookup, Session, SessionKind};
    let mut w = WriterArchive::new();
    for (path, root) in [("/", "root"), ("/a/b/", "nested")] {
        w.repo_lookups.push(RepoLookup {
            path: path.into(),
            remote: None,
            root_commits: vec![root.into()],
        });
    }
    for (n, path) in ["/", "/a/b", "/a/b/", "/a/b/c", "/a/bc"].iter().enumerate() {
        let id = format!("path{n}");
        let mut session = Session::new(sid(&id), SessionKind::Main);
        session.offer_cwd(path);
        w.write_session(session, prose(&id, 1, 1000, "synthetic prompt"));
    }
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let index = crate::ScopeIndex::new(r.scope_changes(None).unwrap());
    for (directory, expected) in [
        ("/", vec![0, 1, 2, 3, 4]),
        ("/a/b", vec![1, 2, 3]),
        ("/a/b/", vec![1, 2, 3]),
    ] {
        let mut opts = options();
        opts.fields = vec!["id".into()];
        opts.page_size = Some(20);
        let page = r
            .query_page(
                &crate::query::compile(
                    &format!("sessions where cwd:{directory} | sort by id"),
                    &opts,
                )
                .unwrap(),
                None,
            )
            .unwrap();
        let wanted: Vec<_> = expected
            .iter()
            .map(|n| sid(&format!("path{n}")).to_string())
            .collect();
        assert_eq!(
            page.rows
                .iter()
                .map(|row| row["id"].as_str().unwrap().to_owned())
                .collect::<Vec<_>>(),
            wanted
        );
        let mut scoped: Vec<_> = index
            .sessions_under(&[directory.into()], None)
            .iter()
            .map(|s| s.session.id.to_string())
            .collect();
        scoped.sort();
        assert_eq!(scoped, wanted);
        for n in 0..5 {
            let path = ["/", "/a/b", "/a/b/", "/a/b/c", "/a/bc"][n];
            let under: bool = r
                .conn()
                .unwrap()
                .query_row(
                    "SELECT pl_query_under(?1,?2)",
                    rusqlite::params![path, directory],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(under, expected.contains(&n));
        }
    }
    assert_eq!(
        w.rows("SELECT repo_root FROM session ORDER BY id"),
        ["root", "nested", "nested", "nested", "root"]
            .map(|root| serde_json::json!({"repo_root":root}))
    );
    let facets = crate::measure::facets::query(
        "SELECT pk AS _pk FROM session",
        "SELECT pk AS _pk FROM session",
        "repositories",
    )
    .unwrap();
    let counts: Vec<(String, i64)> = r
        .conn()
        .unwrap()
        .prepare(&facets)
        .unwrap()
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(counts, [("nested".into(), 3), ("root".into(), 2)]);
}

#[test]
fn session_measurement_windows_use_stored_message_milliseconds() {
    let program = crate::query::compile(
        "sessions since:2026-10-29T12:00:00.0002Z until:2026-10-29T12:00:00.0012Z | measure usage",
        &options(),
    )
    .unwrap();
    let measure = program.measure.unwrap();
    let start = "2026-10-29T12:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .as_millisecond();
    assert_eq!(
        (measure.since, measure.until),
        (Some(start), Some(start + 1))
    );
}
