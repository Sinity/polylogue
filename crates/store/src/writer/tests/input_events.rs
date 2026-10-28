use super::*;

fn event(range: (u64, u64), name: &str) -> Event {
    Event {
        session: None,
        kind: "hermes_observer".into(),
        at: Timestamp::from_unix_ms(1000),
        after: None,
        range: Some(range),
        data: serde_json::json!({"name":name,"result":null,"approved":false}),
    }
}

fn unbound(input: &str, from: u64, events: Vec<Event>, prepared: bool) -> Batch {
    let mut b = batch(input, 0, from, vec![], &[]);
    b.input.format = "hermes".into();
    b.parse[0].output.events = events;
    if prepared {
        b.parse[0].prepare();
    }
    b
}

fn send(writer: &Writer, b: Batch) {
    writer.send(b).unwrap();
    assert!(writer.flush().unwrap().errors.is_empty());
}

fn rows(archive: &Archive) -> Vec<Json> {
    ReadPool::open(archive)
        .unwrap()
        .reader()
        .unwrap()
        .query(
            "SELECT e.id,r.id AS input,s.id AS session,pl_block(e.data) AS data
         FROM event e JOIN input_ref r ON r.pk=e.input_pk
         LEFT JOIN session s ON s.pk=e.session_pk ORDER BY r.id,e.id",
            &[],
        )
        .unwrap()
}

#[test]
fn input_event_occurrences_survive_preparation_replay_restart_and_input_separation() {
    for prepared in [false, true] {
        let TestArchive {
            archive,
            writer,
            _dir,
            _lock,
        } = test_archive();
        let observed = event((0, 100), "same observation");
        send(
            &writer,
            unbound("a", 0, vec![observed.clone(), observed.clone()], prepared),
        );
        let first = rows(&archive);
        assert_eq!(
            first.len(),
            2,
            "one frame emitted two genuinely repeated observations"
        );
        assert_ne!(first[0]["id"], first[1]["id"]);
        assert!(first.iter().all(|r| r["session"].is_null()));
        let mut replay = unbound("a", 0, vec![observed.clone(), observed.clone()], prepared);
        replay.change = Change::Reparse { chain: 1 };
        send(&writer, replay);
        assert_eq!(rows(&archive), first);
        drop(writer);
        let writer = Writer::start(&archive).unwrap();
        let mut replay = unbound("a", 0, vec![observed.clone(), observed.clone()], prepared);
        replay.change = Change::Reparse { chain: 1 };
        send(&writer, replay);
        assert_eq!(rows(&archive), first);
        send(
            &writer,
            unbound("b", 0, vec![observed.clone(), observed], prepared),
        );
        assert_eq!(
            rows(&archive).len(),
            4,
            "unrelated inputs cannot coalesce observer facts"
        );
        send(
            &writer,
            unbound(
                "a",
                100,
                vec![event((100, 200), "same observation")],
                prepared,
            ),
        );
        assert_eq!(
            rows(&archive).len(),
            5,
            "a later record is another occurrence"
        );
        let pool = ReadPool::open(&archive).unwrap();
        let r = pool.reader().unwrap();
        assert_eq!(
            r.query("SELECT count(*) AS n FROM session", &[]).unwrap()[0]["n"],
            0
        );
        assert!(r.events(&sid("missing")).unwrap().is_empty());
        for row in rows(&archive) {
            let body: Json = serde_json::from_str(row["data"].as_str().unwrap()).unwrap();
            assert_eq!(
                body,
                serde_json::json!({"name":"same observation","result":null,"approved":false})
            );
        }
    }
}

#[test]
fn input_events_require_record_coordinates_and_cannot_claim_after_message_association() {
    for shape in ["after", "missing_range", "empty_range", "missing_input"] {
        let w = test_archive();
        let mut invalid = event((0, 100), "private observation");
        match shape {
            "after" => invalid.after = Some(MessageId::parse("hermes:m").unwrap()),
            "missing_range" => invalid.range = None,
            "empty_range" => invalid.range = Some((1, 1)),
            _ => {}
        }
        let input = if shape == "missing_input" {
            ""
        } else {
            "invalid"
        };
        w.writer
            .send(unbound(input, 0, vec![invalid], true))
            .unwrap();
        let result = w.writer.flush().unwrap();
        assert!(!result.errors.is_empty());
        assert!(rows(&w.archive).is_empty());
        let pool = ReadPool::open(&w.archive).unwrap();
        let r = pool.reader().unwrap();
        assert_eq!(
            r.query("SELECT count(*) AS n FROM session", &[]).unwrap()[0]["n"],
            0
        );
    }
}

#[test]
fn input_event_identity_includes_the_stored_revision_chain() {
    let w = test_archive();
    let observed = event((0, 100), "same recorded bytes");
    send(
        &w.writer,
        unbound("rewritten", 0, vec![observed.clone()], true),
    );
    let original = rows(&w.archive);
    let mut rewritten = unbound("rewritten", 0, vec![observed], true);
    rewritten.change = Change::Rewritten;
    send(&w.writer, rewritten);
    let after = rows(&w.archive);
    assert_eq!(
        after.len(),
        2,
        "equal offsets in different stored chains are different source occurrences"
    );
    assert!(after.iter().any(|row| row["id"] == original[0]["id"]));
    assert_ne!(after[0]["id"], after[1]["id"]);
}
