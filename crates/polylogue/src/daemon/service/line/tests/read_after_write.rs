use super::*;
use crate::daemon::service::ArchiveMode;
use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome};
use polylogue_model::{
    AlternateKind, AlternateRelation, Code, NewAlternate, NewAlternatePart, Reference,
    user::UserWrite,
};
use polylogue_store::{
    fixture::authored::operator_request,
    read::{Lane, limit::ReadLimit},
};
use polylogue_testkit::{Morrow, Set};
use std::time::Duration;

#[test]
fn checkout_after_a_write_reads_the_alternate_search_entry() {
    let morrow = Morrow::build(Set::Main);
    let runtime =
        test_daemon::start_runtime(&morrow.root, morrow.config(), ArchiveMode::Standalone).unwrap();
    let daemon = &runtime.daemon;
    let writer = daemon.writes.as_ref().unwrap();
    let source = "message:claude-code:c1000000-0000-4000-8000-000000000001";
    let part = "block:claude-code:c1000000-0000-4000-8000-000000000001#0";
    let alternate = {
        let reader = daemon.reads.reader().unwrap();
        let hash = |text| {
            reader
                .reference_hash(&Reference::parse(text).unwrap())
                .unwrap()
                .unwrap()
        };
        NewAlternate {
            base: None,
            source: source.into(),
            source_hash: hash(source),
            kind: AlternateKind::Translation,
            from_lang: Some("en".into()),
            to_lang: Some("pl".into()),
            method: None,
            parts: vec![NewAlternatePart {
                of: part.into(),
                of_hash: hash(part),
                relation: AlternateRelation::Translates,
                body: "Załóż bibliotekę do parsowania dat.".into(),
            }],
        }
    };

    let index = rusqlite::Connection::open(daemon.archive.index_db()).unwrap();
    index.execute_batch("BEGIN IMMEDIATE").unwrap();
    let written = writer
        .archive()
        .submit(
            operator_request("translation".into(), UserWrite::Alternate { alternate }),
            None,
        )
        .unwrap();
    let limit = ReadLimit::new(Duration::from_secs(10), Default::default());
    let bodies = std::thread::scope(|threads| {
        let read = threads.spawn(|| {
            limit.scope(|| {
                let reader = daemon
                    .reads
                    .checkout_after(writer, written.position, Lane::Ordinary, &limit)
                    .unwrap();
                let mut bodies = Vec::new();
                reader
                    .visit_alternate_search("parsowania", Some("pl"), |hit| {
                        bodies.push(hit.part.body);
                        Ok(())
                    })
                    .unwrap();
                bodies
            })
        });
        while limit.stage() != Some("waiting for the index")
            && !read.is_finished()
            && limit.check().is_ok()
        {
            std::thread::yield_now();
        }
        drop(index);
        read.join().unwrap()
    });
    assert_eq!(bodies, ["Załóż bibliotekę do parsowania dat."]);
}

#[test]
fn read_after_a_write_times_out_and_the_line_keeps_the_write_result() {
    let morrow = Morrow::build(Set::Main);
    let runtime =
        test_daemon::start_runtime(&morrow.root, morrow.config(), ArchiveMode::Standalone).unwrap();
    let daemon = &runtime.daemon;
    let index = rusqlite::Connection::open(daemon.archive.index_db()).unwrap();
    index.execute_batch("BEGIN IMMEDIATE").unwrap();

    for follow in [false, true] {
        let id = if follow {
            "note-before-follow"
        } else {
            "note-before-read"
        };
        let limit = ReadLimit::new(Duration::from_secs(3), Default::default());
        let (events, answer) = Call::new(
            "rows",
            json!({
                "selection": "session:claude-code:7f3a2c10-0001-4000-8000-000000000001",
                "steps": [
                    {"verb": "note", "text": "Check date parsing."},
                    {"verb": "read", "follow": follow},
                ],
            }),
        )
        .id(id)
        .within(&limit)
        .answer_with_events(daemon);

        let LineAnswer::Steps(result) = serde_json::from_value(
            answer
                .value()
                .unwrap_or_else(|| panic!("{answer:?}"))
                .clone(),
        )
        .unwrap() else {
            panic!("expected steps");
        };
        let mut steps = Vec::new();
        if follow {
            let frame = std::str::from_utf8(&events).unwrap();
            let kind: Vec<_> = frame
                .lines()
                .filter_map(|line| line.strip_prefix("event:").map(str::trim))
                .collect();
            assert_eq!(kind, ["steps"]);
            let data = frame
                .lines()
                .find_map(|line| line.strip_prefix("data:"))
                .unwrap();
            let reply: command_surface::Answer<LineAnswer> = serde_json::from_str(data).unwrap();
            let command_surface::Outcome::Ok(LineAnswer::Steps(written)) = reply.outcome else {
                panic!("expected completed steps");
            };
            steps.extend(written.steps);
        } else {
            assert!(events.is_empty());
        }
        steps.extend(result.steps);
        let [write, read] = steps.as_slice() else {
            panic!("expected a write and a read: {steps:?}");
        };
        let StepOutcome::Done {
            reply: StepAnswer::WroteEach(write),
            undo,
        } = &write.outcome
        else {
            panic!("expected the completed write: {:?}", write.outcome);
        };
        let write_id = format!("{id}-1");
        assert_eq!(write.id.as_deref(), Some(write_id.as_str()));
        assert_eq!(write.written, 1);
        assert_eq!(undo.len(), 1);
        let StepOutcome::Failed { error } = &read.outcome else {
            panic!("expected the read timeout: {:?}", read.outcome);
        };
        assert_eq!(error.code.as_deref(), Some(Code::DeadlineExceeded.as_str()));
        let reader = daemon.reads.reader().unwrap();
        let saved = reader.user_request(&write_id).unwrap().unwrap();
        let UserWrite::AssertQuery { annotation, .. } = saved.write else {
            panic!("expected a committed note");
        };
        assert_eq!(annotation.value, Some(json!("Check date parsing.")));
    }
    drop(index);
}
