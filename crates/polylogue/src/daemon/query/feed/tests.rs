use super::*;
use polylogue_store::fixture::{Fixture, SessionFacts};
use serde_json::json;
use std::io::{BufRead, BufReader};

fn lines(file: File) -> Vec<Value> {
    BufReader::new(file)
        .lines()
        .map(
            |s| match serde_json::from_str::<Response>(&s.unwrap()).unwrap() {
                Response::Ok { data, .. } => data,
                other => panic!("{other:?}"),
            },
        )
        .collect()
}

fn fixture_counters(fixture: &Fixture, index: i64, archive: i64) {
    for (path, counter) in [
        (fixture.archive.index_db(), index),
        (fixture.archive.archive_db(), archive),
    ] {
        rusqlite::Connection::open(path).unwrap().execute("INSERT INTO meta VALUES ('write_counter',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[counter.to_string()]).unwrap();
    }
}

fn batch(fixture: &Fixture, since: Option<&str>) -> Vec<Value> {
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    lines(prepare(&reader, since, 0, tempfile::tempfile().unwrap()).unwrap())
}

#[test]
fn replacement_delivers_every_reference_and_policy_before_its_only_new_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    for n in 0..300 {
        fixture
            .session(SessionFacts {
                id: &format!("test:{n:03}"),
                ..Default::default()
            })
            .unwrap();
    }
    fixture
        .note("note", None, Some("synthetic note"), "session:test:000")
        .unwrap();
    fixture
        .excision(
            "remove-a",
            "session:test:gone",
            &["test:gone"],
            &["test:gone1"],
        )
        .unwrap();
    fixture
        .excision("remove-b", "input:synthetic", &[], &[])
        .unwrap();
    fixture_counters(&fixture, 1, 3);
    let result = batch(&fixture, Some("old-generation:8:9"));
    assert_eq!(result[0]["type"], "begin");
    assert_eq!(result[0]["generation_changed"], true);
    assert_eq!(result[0]["replacement"], true);
    assert_eq!(result.last().unwrap()["type"], "complete");
    assert_eq!(result.last().unwrap()["count"], 303);
    assert_eq!(result.len(), 305);
    let events = &result[1..result.len() - 1];
    assert!(
        events
            .iter()
            .all(|line| line["position"] == "old-generation:8:9")
    );
    assert_eq!(
        events
            .iter()
            .filter(|line| line["component"] == "index")
            .count(),
        300
    );
    let archive: Vec<_> = events
        .iter()
        .filter(|line| line["component"] == "archive")
        .collect();
    assert_eq!(
        archive
            .iter()
            .map(|line| line["change"]["seq"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(archive[1]["change"]["sessions"], json!(["test:gone"]));
    assert_eq!(archive[1]["change"]["messages"], json!(["test:gone1"]));
    assert_eq!(archive[2]["change"]["sessions"], json!([]));
    assert_eq!(archive[2]["change"]["messages"], json!([]));
}

#[test]
fn an_output_failure_never_returns_a_deliverable_file() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    // Read-only file reaches the first private staging write, never delivery.
    let path = dir.path().join("read-only");
    std::fs::write(&path, []).unwrap();
    assert_eq!(
        prepare(&pool.reader().unwrap(), None, 0, File::open(path).unwrap())
            .unwrap_err()
            .kind,
        "io"
    );
}

#[test]
fn delta_uses_both_counters_keeps_peers_and_repeats_a_completed_checkpoint_idempotently() {
    use polylogue_model::{Output, Session, SessionKind};
    use polylogue_store::fixture::writer::{WriterArchive, sid};
    let mut fixture = WriterArchive::new();
    let reads = polylogue_store::ReadPool::open(&fixture.archive).unwrap();
    let initial = reads
        .reader()
        .unwrap()
        .change_position()
        .unwrap()
        .to_string();
    fixture.write_output(
        "group",
        "claude-code",
        "/synthetic",
        Output {
            sessions: ["a", "b", "c"]
                .iter()
                .map(|id| Session::new(sid(id), SessionKind::Main))
                .collect(),
            ..Default::default()
        },
        3,
    );
    let reader = reads.reader().unwrap();
    let result = lines(prepare(&reader, Some(&initial), 0, tempfile::tempfile().unwrap()).unwrap());
    assert_eq!(result[0]["replacement"], false);
    assert_eq!(result.last().unwrap()["count"], 3);
    let events = &result[1..result.len() - 1];
    assert!(events.iter().all(|line| line["position"] == initial));
    assert!(
        events
            .iter()
            .all(|line| line["change"]["seq"] == events[0]["change"]["seq"])
    );
    let completed = result.last().unwrap()["position"]
        .as_str()
        .unwrap()
        .to_owned();
    drop(reader);
    let reader = reads.reader().unwrap();
    let repeated =
        lines(prepare(&reader, Some(&completed), 0, tempfile::tempfile().unwrap()).unwrap());
    assert_eq!(repeated.len(), 2);
    assert_eq!(repeated[1]["count"], 0);
    assert_eq!(repeated[1]["position"], completed);
    let target = reader.change_position().unwrap();
    let future = format!(
        "{}:{}:{}",
        target.generation,
        target.index + 1,
        target.archive
    );
    assert_eq!(
        prepare(&reader, Some(&future), 0, tempfile::tempfile().unwrap())
            .unwrap_err()
            .kind,
        "bad_request"
    );
    assert_eq!(
        prepare(
            &reader,
            Some("not-a-position"),
            0,
            tempfile::tempfile().unwrap()
        )
        .unwrap_err()
        .kind,
        "bad_request"
    );
}

#[test]
fn archive_counter_peers_arrive_in_local_order_without_cross_component_clock_claims() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let initial = batch(&fixture, None).last().unwrap()["position"]
        .as_str()
        .unwrap()
        .to_owned();
    // A synthetic archive transaction with two notes sharing one counter.
    let mut db = rusqlite::Connection::open(fixture.archive.archive_db()).unwrap();
    let tx = db.transaction().unwrap();
    for id in ["a", "b"] {
        tx.execute("INSERT INTO note(id, first_note,kind,value,author,machine,created_at,commit_seq) VALUES (?1,?1,'note','synthetic','operator','test',0,1)",[id]).unwrap();
    }
    tx.execute(
        "INSERT INTO meta VALUES ('write_counter','1') ON CONFLICT(key) DO UPDATE SET value='1'",
        [],
    )
    .unwrap();
    tx.commit().unwrap();
    let result = batch(&fixture, Some(&initial));
    assert_eq!(result.last().unwrap()["count"], 2);
    assert!(
        result[1..3]
            .iter()
            .all(|line| line["component"] == "archive"
                && line["position"] == initial
                && line["change"]["seq"] == 1)
    );
    let next = result.last().unwrap()["position"].as_str().unwrap();
    assert_eq!(batch(&fixture, Some(next)).last().unwrap()["count"], 0);
}

fn prepare(
    reader: &Reader<'_>,
    since: Option<&str>,
    now: i64,
    file: File,
) -> Result<File, Failure> {
    prepare_with(reader, since, now, file, |response| response)
}
