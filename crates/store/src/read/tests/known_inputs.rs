use super::*;
use rusqlite::{Connection, params};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn known_inputs_choose_the_latest_eligible_observation_and_keep_parser_facts() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let archive = Connection::open(fixture.archive.archive_db()).unwrap();
    archive.execute_batch(
        "INSERT INTO input VALUES ('watched','codex-rollout'),('manual','codex-rollout'),('other','codex-rollout');
         INSERT INTO revision VALUES ('watched',1,'first',1,10,1),('watched',2,'appended',1,20,2),('manual',1,'imported',1,5,3),('manual',2,'appended',1,7,4),('other',1,'first',1,99,4);
         INSERT INTO observation(id,input,machine,path,class,device,inode,size,mtime_ns,ctime_ns,polylogue_version,seen_at) VALUES
           ('first','watched','m','/home/op/session.jsonl','first',1,2,10,3,4,'test',100),
           ('gone','watched','m','/home/op/session.jsonl','deleted',NULL,NULL,NULL,NULL,NULL,'test',50),
           ('manual','manual','m','/home/op/session.jsonl','imported',5,6,5,7,8,'test',300),
           ('manual-tail','manual','m','/home/op/session.jsonl','appended',5,6,7,7,8,'test',200),
           ('facts','watched','m','/home/op/renamed.jsonl','back',9,10,20,11,12,'test',40),
           ('other','other','other-machine','/home/op/session.jsonl','first',13,14,99,15,16,'test',500);"
    ).unwrap();
    let index = Connection::open(fixture.archive.index_db()).unwrap();
    index
        .execute_batch(
            "INSERT INTO input_ref(id) VALUES ('watched');
         INSERT INTO parse_state(input_pk,chain,status,parsed_upto,detail)
         SELECT pk,1,'waiting',17,'20' FROM input_ref WHERE id='watched';",
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let watched = reader.known_watched_inputs("m").unwrap();
    assert_eq!(watched.len(), 2);
    let gone = watched
        .iter()
        .find(|k| k.path == "/home/op/session.jsonl")
        .unwrap();
    assert_eq!(gone.input_id, "watched");
    assert!(gone.facts.is_none());
    assert_eq!(gone.chain, Some(1));
    assert_eq!(gone.stored, 20);
    assert_eq!(gone.parsed_upto, 17);
    assert!(gone.parsed && gone.waiting);
    let renamed = reader
        .known_watched_input_at("m", "/home/op/renamed.jsonl")
        .unwrap()
        .unwrap();
    let facts = renamed.facts.unwrap();
    assert_eq!((facts.device, facts.inode, facts.size), (9, 10, 20));
    assert_eq!((facts.mtime_ns, facts.ctime_ns), (11, 12));
    assert_eq!(renamed.parsed_upto, 17);
    let all = reader.known_inputs("m").unwrap();
    let imported = all
        .iter()
        .find(|k| k.path == "/home/op/session.jsonl")
        .unwrap();
    assert_eq!(imported.input_id, "manual");
    assert_eq!(imported.stored, 7);
    assert!(!imported.parsed);
    assert_eq!(
        reader.known_inputs("other-machine").unwrap()[0].input_id,
        "other"
    );
    assert!(
        reader
            .known_watched_input_at("m", "/home/op/missing")
            .unwrap()
            .is_none()
    );
}

fn lookup_work(history: usize, point: bool, unrelated: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut archive = Connection::open(fixture.archive.archive_db()).unwrap();
    let tx = archive.transaction().unwrap();
    tx.execute("INSERT INTO input VALUES ('watched','codex-rollout')", [])
        .unwrap();
    for number in 1..=history {
        tx.execute(
            "INSERT INTO revision VALUES ('watched',?1,'appended',1,?1,?1)",
            [number as i64],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO observation(id,input,machine,path,class,size,polylogue_version,seen_at)
             VALUES (?1,'watched','m','/home/op/session.jsonl','appended',?2,'test',?2)",
            params![number.to_string(), number as i64],
        )
        .unwrap();
    }
    if unrelated > 0 {
        tx.execute("INSERT INTO input VALUES ('unrelated','codex-rollout')", [])
            .unwrap();
        for number in 1..=unrelated {
            tx.execute(
                "INSERT INTO revision VALUES ('unrelated',?1,'appended',1,?1,?1)",
                [number as i64],
            )
            .unwrap();
        }
    }
    tx.commit().unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let steps = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&steps);
    reader
        .conn()
        .unwrap()
        .progress_handler(
            100,
            Some(move || observed.fetch_add(100, Ordering::Relaxed) >= 1_000_000),
        )
        .unwrap();
    let found = if point {
        reader
            .known_watched_input_at("m", "/home/op/session.jsonl")
            .map(|found| found.into_iter().collect::<Vec<_>>())
    } else {
        reader.known_watched_inputs("m")
    };
    reader
        .conn()
        .unwrap()
        .progress_handler(0, None::<fn() -> bool>)
        .unwrap();
    let found = found.expect("startup lookup must not revisit every history for every observation");
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].stored, history as u64);
    steps.load(Ordering::Relaxed)
}

#[test]
fn known_inputs_work_scales_with_history_rows() {
    let small = lookup_work(64, false, 0);
    let large = lookup_work(256, false, 0);
    assert!(
        large <= small * 6,
        "64 rows: {small} VM steps; 256 rows: {large}"
    );
}

#[test]
fn known_point_lookup_only_visits_its_paths_revision_histories() {
    let small = lookup_work(64, true, 0);
    let large = lookup_work(256, true, 0);
    assert!(
        large <= small * 6,
        "64 rows: {small} VM steps; 256 rows: {large}"
    );
    let unrelated = lookup_work(64, true, 8192);
    assert!(
        unrelated <= small + 1000,
        "base: {small} VM steps; unrelated history: {unrelated}"
    );
}
