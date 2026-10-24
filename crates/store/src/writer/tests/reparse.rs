use super::parser_versions::send;
use super::*;

#[test]
fn a_completed_new_parse_retires_messages_it_no_longer_produces() {
    let w = world();
    send(&w, Change::First, 1, 0, (1, 2), "obsolete interpretation");
    let mut next = batch("input", 0, 0, vec![], &["s"]);
    next.change = Change::Reparse { chain: 1 };
    next.parse[0].parser_version = 2;
    next.parse[0].at_end = true;
    next.parse[0].accounting = Accounting {
        framed: 1,
        ..Default::default()
    };
    next.parse[0].accounting.add(
        "old-record",
        &Disposition::unknown("no message in this record"),
    );
    w.writer.send(next).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(
        texts(&w, "s").is_empty(),
        "completed new interpretation retained an obsolete message"
    );
}

#[test]
fn a_new_parser_does_not_merge_metadata_from_its_old_interpretation() {
    let w = world();
    let mut old = batch("input", 0, 0, vec![], &["s"]);
    old.parse[0].output.sessions[0].tool_version = Some("obsolete".into());
    w.writer.send(old).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let mut next = batch("input", 0, 0, vec![], &["s"]);
    next.change = Change::Reparse { chain: 1 };
    next.parse[0].parser_version = 2;
    next.parse[0].at_end = true;
    w.writer.send(next).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    w.writer.derive().unwrap();
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(
        r.query("SELECT harness_version FROM session", &[]).unwrap()[0]["harness_version"],
        Json::Null
    );
}

fn apply(w: &World, b: Batch) {
    w.writer.send(b).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
}

fn completion(version: u32) -> Batch {
    let mut b = batch("input", 0, 0, vec![], &["s"]);
    b.change = Change::Reparse { chain: 1 };
    b.parse[0].parser_version = version;
    b.parse[0].at_end = true;
    b
}

#[test]
fn partial_reparse_survives_writer_restart_and_retires_only_on_completion() {
    let w = world();
    let mut old = batch(
        "input",
        0,
        0,
        vec![
            row(10, text_message("a", "s", Role::User, "old A", None)),
            row(20, text_message("b", "s", Role::User, "old B", None)),
        ],
        &["s"],
    );
    old.parse[0].output.sessions[0].tool_version = Some("old".into());
    apply(&w, old);
    let mut partial = completion(2);
    partial.parse[0].parsed_upto = 50;
    partial.parse[0].output.messages =
        vec![row(10, text_message("a", "s", Role::User, "new A", None))];
    apply(&w, partial);
    assert_eq!(texts(&w, "s"), ["new A", "old B"]);
    w.writer.derive().unwrap();
    let reads = ReadPool::open(&w.archive).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .query("SELECT harness_version FROM session", &[])
            .unwrap()[0]["harness_version"],
        "old"
    );
    drop(reads);
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
    let mut last = completion(2);
    last.parse[0].start = 50;
    last.parse[0].reset = false;
    apply(&w, last.clone());
    assert_eq!(texts(&w, "s"), ["new A"]);
    apply(&w, last);
    assert_eq!(texts(&w, "s"), ["new A"]);
    send(&w, Change::Reparse { chain: 1 }, 1, 0, (1, 2), "old A");
    assert_eq!(texts(&w, "s"), ["new A"]);
}

#[test]
fn retiring_one_input_promotes_the_other_inputs_message_and_metadata() {
    let w = world();
    send(&w, Change::First, 1, 0, (1, 2), "live interpretation");
    let mut copy = batch(
        "copy",
        2,
        0,
        vec![row(
            5,
            text_message("a", "s", Role::User, "copy interpretation", None),
        )],
        &["s"],
    );
    copy.parse[0].output.sessions[0].tool_version = Some("copy version".into());
    apply(&w, copy);
    assert_eq!(texts(&w, "s"), ["live interpretation"]);
    apply(&w, completion(2));
    assert_eq!(texts(&w, "s"), ["copy interpretation"]);
    w.writer.derive().unwrap();
    let reads = ReadPool::open(&w.archive).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .query("SELECT harness_version FROM session", &[])
            .unwrap()[0]["harness_version"],
        "copy version"
    );
}

#[test]
fn omitted_hook_and_link_leave_neither_live_wait_nor_legacy_link() {
    let w = world();
    let mut old = batch("input", 0, 0, vec![], &["s"]);
    old.parse[0].output.events.push(Event {
        session: Some(sid("s")),
        kind: "hook_permission_request".into(),
        at: Timestamp::from_unix_ms(1000),
        after: None,
        range: Some((1, 2)),
        data: Json::Null,
    });
    old.parse[0].output.links.push(Link {
        kind: LinkKind::Spawn,
        parent: sid("s"),
        child: sid("child"),
        at: None,
        call_id: None,
        evidence: "old field".into(),
    });
    apply(&w, old);
    let check = |expected: Json, events: usize, links: i64| {
        let reads = ReadPool::open(&w.archive).unwrap();
        let r = reads.reader().unwrap();
        assert_eq!(r.events(&sid("s")).unwrap().len(), events);
        assert_eq!(
            r.query(
                "SELECT live_waiting_on FROM session WHERE id='claude-code:s'",
                &[]
            )
            .unwrap()[0]["live_waiting_on"],
            expected
        );
        assert_eq!(
            r.query("SELECT count(*) AS n FROM current_link", &[])
                .unwrap()[0]["n"],
            links
        );
    };
    check(Json::String("permission".into()), 1, 1);
    apply(&w, completion(2));
    check(Json::Null, 0, 0);
}

#[test]
fn reparsing_an_older_chain_does_not_remove_the_newer_chains_contributions() {
    let w = world();
    send(&w, Change::First, 1, 0, (1, 2), "old chain");
    send(&w, Change::Rewritten, 1, 0, (1, 2), "new chain");
    apply(&w, completion(2));
    assert_eq!(texts(&w, "s"), ["new chain"]);
}

#[test]
fn a_removed_snapshot_no_longer_hides_the_remaining_new_interpretation() {
    let w = world();
    let mut seed = batch("input", 0, 0, vec![], &[]);
    seed.length = 200;
    seed.parse.clear();
    apply(&w, seed);
    for (start, text) in [(0, "first"), (100, "later")] {
        let mut b = completion(1);
        b.length = 200;
        b.parse[0].parsed_upto = start + 100;
        b.parse[0].snapshot = Some(polylogue_model::Snapshot {
            start,
            end: start + 100,
            covered: vec![sid("s")],
            complete: true,
        });
        b.parse[0].output.messages =
            vec![row(start, text_message("a", "s", Role::User, text, None))];
        if start == 100 {
            b.parse[0].output.sessions[0].tool_version = Some("removed scope".into());
        }
        apply(&w, b);
    }
    assert_eq!(texts(&w, "s"), ["later"]);
    let mut new = completion(2);
    new.length = 200;
    new.parse[0].parsed_upto = 200;
    new.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start: 0,
        end: 100,
        covered: vec![sid("s")],
        complete: true,
    });
    new.parse[0].output.messages = vec![row(
        0,
        text_message("a", "s", Role::User, "new first", None),
    )];
    apply(&w, new);
    assert_eq!(texts(&w, "s"), ["new first"]);
    w.writer.derive().unwrap();
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

#[test]
fn a_captured_prefix_is_not_completion_even_when_all_its_bytes_are_parsed() {
    let w = world();
    send(&w, Change::First, 1, 0, (1, 2), "old still needed");
    let mut prefix = completion(2);
    prefix.length = 50;
    prefix.parse[0].parsed_upto = 50;
    prefix.parse[0].at_end = false;
    apply(&w, prefix);
    assert_eq!(texts(&w, "s"), ["old still needed"]);
    apply(&w, completion(2));
    assert!(texts(&w, "s").is_empty());
}

#[test]
fn a_completed_empty_interpretation_leaves_no_ghost_session() {
    let w = world();
    send(&w, Change::First, 1, 0, (1, 2), "old session");
    let mut last = completion(2);
    last.parse[0].output.sessions.clear();
    apply(&w, last);
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(
        r.query("SELECT count(*) AS n FROM session", &[]).unwrap()[0]["n"],
        0
    );
}

#[test]
fn a_repository_lookup_omitted_by_the_new_parser_is_retired() {
    let w = world();
    let mut old = batch("input", 0, 0, vec![], &["s"]);
    old.parse[0]
        .output
        .repo_lookups
        .push(polylogue_model::RepoLookup {
            path: "/synthetic".into(),
            remote: None,
            root_commit: Some("a".repeat(40)),
        });
    apply(&w, old);
    apply(&w, completion(2));
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(
        r.query("SELECT count(*) AS n FROM repo_lookup", &[])
            .unwrap()[0]["n"],
        0
    );
}

#[test]
fn a_failed_completion_rolls_back_retirement_and_can_retry() {
    let w = world();
    send(&w, Change::First, 1, 0, (1, 2), "old remains on refusal");
    let c =
        crate::archive::open_write(&w.archive.index_db(), crate::archive::cache::SETUP).unwrap();
    c.execute_batch(
        "CREATE TRIGGER refuse_retirement BEFORE DELETE ON session_record
        WHEN old.parser_version=1 BEGIN SELECT RAISE(ABORT,'synthetic completion failure'); END;",
    )
    .unwrap();
    w.writer.send(completion(2)).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(
        report
            .errors
            .iter()
            .any(|e| e.contains("synthetic completion failure")),
        "{:?}",
        report.errors
    );
    assert_eq!(texts(&w, "s"), ["old remains on refusal"]);
    c.execute_batch("DROP TRIGGER refuse_retirement").unwrap();
    drop(c);
    apply(&w, completion(2));
    assert!(texts(&w, "s").is_empty());
}

mod multiple_inputs;
