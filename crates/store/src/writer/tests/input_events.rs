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
         FROM current_event e JOIN input_ref r ON r.pk=e.input_pk
         LEFT JOIN session s ON s.pk=e.session_pk ORDER BY r.id,e.id",
            &[],
        )
        .unwrap()
}

#[test]
fn input_event_occurrences_survive_preparation_replay_restart_and_input_separation() {
    for prepared in [false, true] {
        let World {
            archive,
            writer,
            _dir,
            _lock,
        } = world();
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

fn snapshot(at: u64, complete: bool, events: Vec<Event>) -> Batch {
    let mut b = unbound("snapshots", at, events, false);
    b.change = Change::Reparse { chain: 1 };
    b.length = 400;
    b.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start: at,
        end: at + 100,
        complete,
        covered: vec![],
    });
    b
}

#[test]
fn complete_input_snapshots_retire_omitted_events_but_partial_and_failed_ones_do_not() {
    let World {
        archive,
        writer,
        _dir,
        _lock,
    } = world();
    let mut seed = unbound("snapshots", 0, vec![], false);
    seed.length = 400;
    seed.parse.clear();
    send(&writer, seed);
    send(&writer, snapshot(0, true, vec![event((0, 100), "first")]));
    send(
        &writer,
        snapshot(100, false, vec![event((100, 200), "partial")]),
    );
    assert_eq!(rows(&archive).len(), 2);
    drop(writer);
    let writer = Writer::start(&archive).unwrap();
    assert_eq!(rows(&archive).len(), 2);
    send(
        &writer,
        snapshot(200, true, vec![event((200, 300), "complete")]),
    );
    let before = rows(&archive);
    assert_eq!(before.len(), 1);
    let mut invalid = event((300, 400), "must roll back");
    invalid.after = Some(MessageId::parse("hermes:unrelated").unwrap());
    writer.send(snapshot(300, true, vec![invalid])).unwrap();
    assert!(!writer.flush().unwrap().errors.is_empty());
    assert_eq!(
        rows(&archive),
        before,
        "a rejected complete snapshot changes nothing"
    );
    send(&writer, snapshot(300, false, vec![]));
    assert_eq!(
        rows(&archive),
        before,
        "an empty partial snapshot does not claim omission"
    );
}

#[test]
fn input_events_require_record_coordinates_and_cannot_claim_after_message_association() {
    for shape in ["after", "missing_range", "empty_range", "missing_input"] {
        let w = world();
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
