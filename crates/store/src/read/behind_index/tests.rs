use super::*;
use crate::fixture::Fixture;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn expected(
    input: &str,
    chain: i64,
    length: u64,
    current: bool,
    source: (&str, Option<&str>),
    imported: bool,
    parsed_upto: Option<u64>,
) -> BehindChain {
    BehindChain {
        input: input.into(),
        current,
        reading_changed: false,
        previous_reading: false,
        stopped_attempt: None,
        stopped_at_length: false,
        source: StoredSource {
            format: source.0.into(),
            path: source.1.map(str::to_owned),
        },
        progress: ChainProgress {
            chain,
            length,
            parsed_upto: parsed_upto.unwrap_or(0),
            parsed: parsed_upto.is_some(),
            waiting: false,
            imported,
        },
    }
}

#[test]
fn behind_chains_preserve_revision_import_and_parser_choices() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let archive = Connection::open(fixture.archive.archive_db()).unwrap();
    archive.execute_batch(
        "INSERT INTO input VALUES ('z','fallback'),('a','fallback'),('b','native'),('c','native');
         INSERT INTO revision VALUES
           ('a',1,'imported',1,20,1),('a',2,'rewritten',2,90,2),
           ('a',3,'rewritten',3,30,3),('a',4,'appended',2,40,4),
           ('a',5,'rewritten',5,50,5),('a',6,'rewritten',6,0,6),
           ('a',7,'appended',1,15,7),
           ('b',1,'first',1,10,1),('c',1,'imported',1,10,1),
           ('z',1,'imported',1,10,1);
         INSERT INTO observation(id,input,machine,path,class,details_json,polylogue_version,seen_at) VALUES
           ('first','z','m','/home/op/first','first','{}','test',99),
           ('later','z','m','/home/op/later','first','{}','test',1),
           ('invalid','z','m','/home/op/invalid','imported','invalid','test',2),
           ('nontext','z','m','/home/op/nontext','imported','{\"import_format\":7}','test',3),
           ('old-format','a','m','/home/op/old','imported','{\"import_format\":\"old\"}','test',99),
           ('format','a','m','/home/op/format','imported','{\"import_format\":\"codex-rollout\"}','test',1),
           ('export','a','m','/home/op/export','imported','{\"import_format\":\"export-file\"}','test',100),
           ('export-only','c','m','/home/op/export-only','imported','{\"import_format\":\"export-file\"}','test',1);"
    ).unwrap();
    let index = Connection::open(fixture.archive.index_db()).unwrap();
    index
        .execute_batch(
            "INSERT INTO input_ref(id) VALUES ('a');
         INSERT INTO parse_state(input_pk,chain,status,parsed_upto)
         SELECT pk,2,'parsing',12 FROM input_ref WHERE id='a';
         INSERT INTO parse_state(input_pk,chain,status,parsed_upto,at_length)
         SELECT pk,3,'waiting',2,30 FROM input_ref WHERE id='a';
         INSERT INTO parse_state(input_pk,chain,status,parsed_upto)
         SELECT pk,5,'done',50 FROM input_ref WHERE id='a';",
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .chains_behind_index(
                &[],
                crate::StoppedChains::Settled,
                crate::RevisionChains::All,
                None
            )
            .unwrap(),
        vec![
            // Live inputs first, then most recently observed inputs and newest chains.
            expected("b", 1, 10, true, ("native", None), false, None),
            expected(
                "a",
                2,
                90,
                false,
                ("codex-rollout", Some("/home/op/format")),
                true,
                Some(12)
            ),
            expected(
                "a",
                1,
                20,
                true,
                ("codex-rollout", Some("/home/op/format")),
                true,
                None
            ),
            expected(
                "z",
                1,
                10,
                true,
                ("fallback", Some("/home/op/first")),
                true,
                None
            ),
            expected(
                "c",
                1,
                10,
                true,
                ("export-file", Some("/home/op/export-only")),
                true,
                None
            ),
        ]
    );
}

fn history_work(revisions: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut archive = Connection::open(fixture.archive.archive_db()).unwrap();
    let tx = archive.transaction().unwrap();
    tx.execute("INSERT INTO input VALUES ('synthetic','codex-rollout')", [])
        .unwrap();
    let chains = revisions / 4;
    for number in 1..=revisions {
        tx.execute(
            "INSERT INTO revision VALUES ('synthetic',?1,'appended',?2,?1,?1)",
            params![number as i64, ((number - 1) % chains + 1) as i64],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at)
             VALUES (?1,'synthetic','m','/home/op/session.jsonl','appended','test',?2)",
            params![number.to_string(), (revisions - number) as i64],
        )
        .unwrap();
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
            Some(move || {
                observed.fetch_add(100, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
    let found = reader
        .chains_behind_index(
            &[],
            crate::StoppedChains::Settled,
            crate::RevisionChains::All,
            None,
        )
        .unwrap();
    reader
        .conn()
        .unwrap()
        .progress_handler(0, None::<fn() -> bool>)
        .unwrap();
    let rows: Vec<_> = (1..=chains)
        .rev()
        .map(|chain| {
            expected(
                "synthetic",
                chain as i64,
                (revisions - chains + chain) as u64,
                chain == chains,
                ("codex-rollout", Some("/home/op/session.jsonl")),
                false,
                None,
            )
        })
        .collect();
    assert_eq!(found, rows);
    steps.load(Ordering::Relaxed)
}

#[test]
fn behind_chains_work_scales_with_revision_and_observation_history() {
    let small = history_work(128);
    let large = history_work(512);
    assert!(
        large <= small * 6,
        "128 revisions: {small} VM steps; 512 revisions: {large}"
    );
}

#[test]
fn incomplete_metadata_readings_wait_until_the_stored_length_changes() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let archive = Connection::open(fixture.archive.archive_db()).unwrap();
    archive
        .execute_batch(
            "INSERT INTO input VALUES ('metadata','no-parser');
        INSERT INTO revision VALUES ('metadata',1,'first',1,100,1);",
        )
        .unwrap();
    let index = Connection::open(fixture.archive.index_db()).unwrap();
    index
        .execute_batch(
            "INSERT INTO input_ref(id) VALUES ('metadata');
        INSERT INTO parse_state(input_pk,chain,status,parsed_upto,at_length,reading_version)
        SELECT pk,1,'waiting',40,100,'metadata=1' FROM input_ref;",
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert!(
        reader
            .chains_behind_index(
                &[],
                crate::StoppedChains::Settled,
                crate::RevisionChains::All,
                None
            )
            .unwrap()
            .is_empty()
    );
    assert!(reader.input_progress("metadata").unwrap().unwrap().waiting);
    archive
        .execute(
            "INSERT INTO revision VALUES ('metadata',2,'appended',1,140,2)",
            [],
        )
        .unwrap();
    assert!(
        reader
            .chains_behind_index(
                &[],
                crate::StoppedChains::Settled,
                crate::RevisionChains::All,
                None
            )
            .unwrap()
            .is_empty()
    );
    assert!(reader.input_progress("metadata").unwrap().unwrap().waiting);
    let fresh = pool.reader().unwrap();
    let behind = fresh
        .chains_behind_index(
            &[],
            crate::StoppedChains::Settled,
            crate::RevisionChains::All,
            None,
        )
        .unwrap();
    assert_eq!(behind.len(), 1);
    assert_eq!(
        (
            behind[0].progress.parsed_upto,
            behind[0].progress.length,
            behind[0].progress.waiting
        ),
        (40, 140, false)
    );
    assert!(!fresh.input_progress("metadata").unwrap().unwrap().waiting);
}

#[test]
fn stopped_selection_uses_actual_attempt_even_after_a_full_length_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let archive = Connection::open(fixture.archive.archive_db()).unwrap();
    archive.execute_batch("INSERT INTO input VALUES('failed','claude-code'),('refused','claude-code'); INSERT INTO revision VALUES('failed',1,'first',1,10,1),('refused',1,'first',1,10,1);").unwrap();
    let index = Connection::open(fixture.archive.index_db()).unwrap();
    let attempt = polylogue_model::batch::StoppedAttempt {
        executable: polylogue_model::BlobHash::of(b"first actual program"),
        parser_settings: None,
    };
    index.execute_batch("INSERT INTO input_ref(pk,id,format) VALUES(71,'failed','claude-code'),(72,'refused','claude-code'); INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto,at_length,reason) VALUES(71,1,'reading','failed',0,10,'original failure'),(72,1,'reading','refused',10,10,'original refusal');").unwrap();
    index
        .execute(
            "UPDATE parse_state SET stopped_attempt=?1 WHERE input_pk=71",
            [crate::encoding::pack_json(&attempt).unwrap()],
        )
        .unwrap();
    let reads = fixture.reads().unwrap();
    let select = |attempt| {
        reads
            .reader()
            .unwrap()
            .chains_behind_index(
                &[],
                crate::StoppedChains::ForAttempt(attempt),
                crate::RevisionChains::All,
                None,
            )
            .unwrap()
    };
    let first = select(attempt);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].input, "refused");
    assert!(first[0].stopped_at_length);
    assert_eq!(first[0].progress.parsed_upto, 10);
    assert_eq!(first[0].stopped_attempt, None);
    index
        .execute(
            "UPDATE parse_state SET stopped_attempt=?1 WHERE input_pk=72",
            [crate::encoding::pack_json(&attempt).unwrap()],
        )
        .unwrap();
    assert!(select(attempt).is_empty());
    let next = polylogue_model::batch::StoppedAttempt {
        executable: polylogue_model::BlobHash::of(b"next actual program"),
        parser_settings: None,
    };
    assert_eq!(select(next).len(), 2);
}

#[test]
fn selected_chains_keep_order_parser_upgrades_and_empty_selection() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let archive = Connection::open(fixture.archive.archive_db()).unwrap();
    archive.execute_batch(
        "INSERT INTO input VALUES ('a','native'),('b','native'),('c','native');
         INSERT INTO revision VALUES ('a',1,'first',1,10,1),('a',2,'rewritten',2,20,2),('b',1,'first',1,30,3),('c',1,'first',1,40,4);
         INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at) VALUES
         ('a','a','m','/home/op/a','first','test',30),('b','b','m','/home/op/b','first','test',40),('c','c','m','/home/op/c','first','test',20);"
    ).unwrap();
    let index = Connection::open(fixture.archive.index_db()).unwrap();
    index
        .execute_batch(
            "INSERT INTO input_ref(id) VALUES ('a');
         INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto)
         SELECT pk,2,'old','done',20 FROM input_ref WHERE id='a';",
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let versions = [("native".into(), "new".into())];
    let all = reader
        .chains_behind_index(
            &versions,
            StoppedChains::Settled,
            crate::RevisionChains::All,
            None,
        )
        .unwrap();
    assert_eq!(
        all.iter()
            .map(|c| (&*c.input, c.progress.chain, c.current, c.reading_changed))
            .collect::<Vec<_>>(),
        vec![
            ("b", 1, true, true),
            ("a", 2, true, true),
            ("a", 1, false, true),
            ("c", 1, true, true)
        ]
    );
    let selected = vec!["c".into(), "a".into(), "a".into(), "missing".into()];
    assert_eq!(
        reader
            .chains_behind_index(
                &versions,
                StoppedChains::Settled,
                crate::RevisionChains::All,
                Some(&selected)
            )
            .unwrap(),
        all.into_iter()
            .filter(|c| c.input != "b")
            .collect::<Vec<_>>()
    );
    assert!(
        reader
            .chains_behind_index(
                &versions,
                StoppedChains::Settled,
                crate::RevisionChains::All,
                Some(&[])
            )
            .unwrap()
            .is_empty()
    );
}

fn selected_history_work(unselected: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let mut archive = Connection::open(fixture.archive.archive_db()).unwrap();
    let tx = archive.transaction().unwrap();
    for n in 0..=unselected {
        let input = if n == 0 {
            "selected".to_owned()
        } else {
            format!("other-{n}")
        };
        tx.execute("INSERT INTO input VALUES (?1,'native')", [&input])
            .unwrap();
        for revision in 1..=4 {
            tx.execute(
                "INSERT INTO revision VALUES (?1,?2,'appended',1,?2,?2)",
                params![input, revision],
            )
            .unwrap();
            tx.execute(
                "INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at)
                VALUES (?1,?2,'m','/home/op/input','first','test',?3)",
                params![format!("{input}-{revision}"), input, revision],
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
            Some(move || {
                observed.fetch_add(100, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
    let found = reader
        .chains_behind_index(
            &[],
            StoppedChains::Settled,
            crate::RevisionChains::All,
            Some(&["selected".into()]),
        )
        .unwrap();
    reader
        .conn()
        .unwrap()
        .progress_handler(0, None::<fn() -> bool>)
        .unwrap();
    assert_eq!(
        found,
        vec![expected(
            "selected",
            1,
            4,
            true,
            ("native", Some("/home/op/input")),
            false,
            None
        )]
    );
    steps.load(Ordering::Relaxed)
}

#[test]
fn selected_chain_planning_does_not_scan_unselected_history() {
    let small = selected_history_work(32);
    let large = selected_history_work(512);
    assert!(
        large <= small * 2,
        "same selected input, 32 unrelated inputs: {small} VM steps; 512 unrelated inputs: {large}"
    );
}

mod selection;
