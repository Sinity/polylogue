use super::*;

fn input_parse(input: &str, version: u32, text: &str, obsolete: bool) -> Batch {
    let mut rows = vec![row(10, text_message(input, "s", Role::User, text, None))];
    if obsolete {
        rows.push(row(
            20,
            text_message(
                &format!("obsolete-{input}"),
                "s",
                Role::User,
                &format!("obsolete {input}"),
                None,
            ),
        ));
    }
    let mut b = batch(input, 0, 0, rows, &["s"]);
    b.change = if version == 1 {
        Change::First
    } else {
        Change::Reparse { chain: 1 }
    };
    b.parse[0].parser_version = version;
    b.parse[0].at_end = true;
    if obsolete {
        b.parse[0].output.sessions[0].tool_version = Some(format!("obsolete {input}"));
    }
    b
}

#[test]
fn a_session_keeps_omitted_rows_until_all_inputs_finish_the_new_parse() {
    for order in [["a", "b"], ["b", "a"]] {
        let w = world();
        apply(&w, input_parse("a", 1, "old a", true));
        apply(&w, input_parse("b", 1, "old b", true));
        w.writer.require_parsers(&[("claude-code", 2)]).unwrap();
        apply(
            &w,
            input_parse(order[0], 2, &format!("new {}", order[0]), false),
        );
        assert_eq!(
            ReadPool::open(&w.archive)
                .unwrap()
                .reader()
                .unwrap()
                .status()
                .unwrap()
                .parse_states
                .get("reparsing"),
            Some(&2)
        );
        let shown = texts(&w, "s");
        assert!(
            shown.contains(&format!("obsolete {}", order[0])),
            "{order:?}: a completed input must not retire its omitted rows before the session's other selected input finishes: {shown:?}"
        );
        apply(
            &w,
            input_parse(order[1], 2, &format!("new {}", order[1]), false),
        );
        w.writer.derive().unwrap();
        assert_eq!(texts(&w, "s"), ["new a", "new b"]);
        let reads = ReadPool::open(&w.archive).unwrap();
        assert_eq!(
            reads
                .reader()
                .unwrap()
                .query("SELECT harness_version FROM session", &[])
                .unwrap()[0]["harness_version"],
            Json::Null
        );
    }
}

#[test]
fn a_captured_prefix_and_writer_restart_do_not_finish_the_other_input() {
    let w = world();
    apply(&w, input_parse("a", 1, "old a", true));
    apply(&w, input_parse("b", 1, "old b", true));
    w.writer.require_parsers(&[("claude-code", 2)]).unwrap();
    apply(&w, input_parse("a", 2, "new a", false));
    let mut prefix = input_parse("b", 2, "new b", false);
    prefix.parse[0].at_end = false;
    apply(&w, prefix);
    assert!(texts(&w, "s").contains(&"obsolete a".to_owned()));
    let World {
        _dir,
        archive,
        _lock,
        writer,
    } = w;
    drop(writer);
    let w = World {
        _dir,
        writer: Writer::start(&archive).unwrap(),
        archive,
        _lock,
    };
    assert!(texts(&w, "s").contains(&"obsolete a".to_owned()));
    apply(&w, input_parse("b", 2, "new b", false));
    w.writer.derive().unwrap();
    assert_eq!(texts(&w, "s"), ["new a", "new b"]);
}

fn shared(version: u32, obsolete: bool) -> Batch {
    let mut b = input_parse("shared", version, "shared s", obsolete);
    b.parse[0]
        .output
        .sessions
        .push(Session::new(sid("t"), SessionKind::Main));
    b.parse[0].output.messages.push(row(
        30,
        text_message("shared-t", "t", Role::User, "shared t", None),
    ));
    if obsolete {
        b.parse[0].output.messages.push(row(
            40,
            text_message("obsolete-t", "t", Role::User, "obsolete t", None),
        ));
    }
    b.parse[0].accounting = accounting(b.parse[0].output.messages.len() as u64);
    b
}

#[test]
fn a_shared_input_retires_each_session_when_its_own_other_inputs_finish() {
    let w = world();
    apply(&w, shared(1, true));
    apply(&w, input_parse("b", 1, "old b", true));
    let mut other = input_parse("c", 1, "old c", true);
    other.parse[0].output.sessions = vec![Session::new(sid("t"), SessionKind::Main)];
    for row in &mut other.parse[0].output.messages {
        row.message.session = sid("t");
    }
    apply(&w, other);
    w.writer.require_parsers(&[("claude-code", 2)]).unwrap();
    apply(&w, shared(2, false));
    apply(&w, input_parse("b", 2, "new b", false));
    assert!(!texts(&w, "s").contains(&"obsolete shared".to_owned()));
    assert!(texts(&w, "t").contains(&"obsolete t".to_owned()));
    let mut last = input_parse("c", 2, "new c", false);
    last.parse[0].output.sessions = vec![Session::new(sid("t"), SessionKind::Main)];
    for row in &mut last.parse[0].output.messages {
        row.message.session = sid("t");
    }
    apply(&w, last);
    assert!(!texts(&w, "t").contains(&"obsolete t".to_owned()));
    w.writer.derive().unwrap();
    assert_eq!(texts(&w, "s"), ["new b", "shared s"]);
    assert_eq!(texts(&w, "t"), ["new c", "shared t"]);
}

#[test]
fn omitted_hooks_retire_with_the_session_and_links_wait_for_both_endpoints() {
    let w = world();
    let mut first = input_parse("a", 1, "old a", true);
    first.parse[0].output.events.push(Event {
        session: Some(sid("s")),
        kind: "hook_permission_request".into(),
        at: Timestamp::from_unix_ms(1000),
        after: None,
        range: Some((1, 2)),
        data: Json::Null,
    });
    first.parse[0].output.links.push(Link {
        kind: LinkKind::Spawn,
        parent: sid("s"),
        child: sid("t"),
        at: None,
        call_id: None,
        evidence: "synthetic previous link".into(),
    });
    apply(&w, first);
    let mut other = input_parse("b", 1, "old b", true);
    other.parse[0].output.sessions = vec![Session::new(sid("t"), SessionKind::Main)];
    for row in &mut other.parse[0].output.messages {
        row.message.session = sid("t");
    }
    apply(&w, other);
    w.writer.require_parsers(&[("claude-code", 2)]).unwrap();
    apply(&w, input_parse("a", 2, "new a", false));
    {
        let reads = ReadPool::open(&w.archive).unwrap();
        let r = reads.reader().unwrap();
        assert!(r.events(&sid("s")).unwrap().is_empty());
        assert_eq!(
            r.query("SELECT count(*) AS n FROM current_link", &[])
                .unwrap()[0]["n"],
            1
        );
        assert_eq!(
            r.query(
                "SELECT live_waiting_on FROM session WHERE id='claude-code:s'",
                &[]
            )
            .unwrap()[0]["live_waiting_on"],
            Json::Null
        );
    }
    let mut last = input_parse("b", 2, "new b", false);
    last.parse[0].output.sessions = vec![Session::new(sid("t"), SessionKind::Main)];
    for row in &mut last.parse[0].output.messages {
        row.message.session = sid("t");
    }
    apply(&w, last);
    let reads = ReadPool::open(&w.archive).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .query("SELECT count(*) AS n FROM current_link", &[])
            .unwrap()[0]["n"],
        0
    );
}

#[test]
fn observed_source_growth_keeps_retirement_pending_until_its_new_eof() {
    let w = world();
    let blobs = hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap();
    apply(&w, input_parse("a", 1, "old a", true));
    let mut old = input_parse("b", 1, "old b", true);
    old.piece = Some(Piece::prepare(&blobs, &[b'x'; 100]).unwrap());
    apply(&w, old);
    w.writer.require_parsers(&[("claude-code", 2)]).unwrap();
    apply(&w, input_parse("a", 2, "new a", false));
    let mut prefix = input_parse("b", 2, "new b", false);
    prefix.parse[0].at_end = false;
    apply(&w, prefix);
    let mut growth = batch(
        "b",
        0,
        100,
        vec![row(
            130,
            text_message("late-b", "s", Role::User, "late b", None),
        )],
        &["s"],
    );
    growth.piece = Some(Piece::prepare(&blobs, &[b'y'; 100]).unwrap());
    growth.parse[0].parser_version = 2;
    growth.parse[0].at_end = false;
    apply(&w, growth);
    assert!(texts(&w, "s").contains(&"obsolete a".to_owned()));
    let mut end = batch("b", 0, 200, vec![], &["s"]);
    end.change = Change::Reparse { chain: 1 };
    end.length = 200;
    end.parse[0].parsed_upto = 200;
    end.parse[0].parser_version = 2;
    apply(&w, end);
    assert_eq!(texts(&w, "s"), ["new a", "new b", "late b"]);
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(
        r.query(
            "SELECT length FROM a.revision WHERE input='b' ORDER BY number DESC LIMIT 1",
            &[]
        )
        .unwrap()[0]["length"],
        200
    );
    assert_eq!(r.status().unwrap().parse_states.get("reparsing"), None);
}
