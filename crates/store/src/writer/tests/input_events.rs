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
fn session_excision_refuses_a_file_that_also_holds_records_of_no_session() {
    use crate::blobs::NewBlob;
    use crate::excise::{ExciseRequest, ExciseTarget};
    use std::sync::Arc;
    for shared_record in [false, true] {
        let w = test_archive();
        let store =
            Arc::new(hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap());
        let first = b"{\"native\":\"bound observation\"}\n";
        let last = b"{\"native\":\"independent observation\"}\n";
        let raw = if shared_record {
            first.to_vec()
        } else {
            [first.as_slice(), last.as_slice()].concat()
        };
        let bound_end = first.len() as u64;
        let mut bound = event((0, bound_end), "bound");
        bound.session = Some(sid("conversation"));
        let unrelated_range = if shared_record {
            (0, bound_end)
        } else {
            (bound_end, raw.len() as u64)
        };
        let unrelated = event(unrelated_range, "independent");
        let mut b = batch("shared-observer", 0, 0, vec![], &["conversation"]);
        b.length = raw.len() as u64;
        b.new_blob = Some(NewBlob::prepare(&store, &raw).unwrap());
        b.parse[0].parsed_upto = b.length;
        b.parse[0].output.events = vec![bound, unrelated];
        send(&w.writer, b);
        let before = rows(&w.archive);
        let result = w.writer.excise(
            ExciseRequest {
                id: "remove-conversation".into(),
                target: ExciseTarget::Session(sid("conversation")),
                reason: None,
                dry_run: false,
            },
            store.clone(),
        );
        // Records of no session share the file: excising the session alone
        // would remove them too, so it refuses.
        assert!(
            matches!(result, Err(StoreError::ExcisionRefused { .. })),
            "{result:?}"
        );
        assert_eq!(rows(&w.archive), before);
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
