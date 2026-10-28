use super::*;
use crate::fixture::Fixture;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn expected(
    input: &str,
    chain: i64,
    length: u64,
    current: bool,
    imported: Option<(&str, Option<&str>)>,
    parsed_upto: Option<u64>,
) -> BehindChain {
    BehindChain {
        input: input.into(),
        current,
        reading_changed: false,
        imported: imported.map(|(format, path)| ImportedAs {
            format: format.into(),
            path: path.map(str::to_owned),
        }),
        progress: ChainProgress {
            chain,
            length,
            parsed_upto: parsed_upto.unwrap_or(0),
            parsed: parsed_upto.is_some(),
            waiting: false,
            arrival_rank: if imported.is_some() { 2 } else { 0 },
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
           ('excised','z','m','/home/op/excised','excised','{}','test',0),
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
         INSERT INTO parse_state(input_pk,chain,status,parsed_upto)
         SELECT pk,3,'waiting',2 FROM input_ref WHERE id='a';
         INSERT INTO parse_state(input_pk,chain,status,parsed_upto)
         SELECT pk,5,'done',50 FROM input_ref WHERE id='a';",
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    assert_eq!(
        pool.reader().unwrap().chains_behind_index(&[]).unwrap(),
        vec![
            expected(
                "a",
                1,
                20,
                true,
                Some(("codex-rollout", Some("/home/op/format"))),
                None
            ),
            expected(
                "a",
                2,
                90,
                false,
                Some(("codex-rollout", Some("/home/op/format"))),
                Some(12)
            ),
            expected("b", 1, 10, true, None, None),
            expected(
                "c",
                1,
                10,
                true,
                Some(("export-file", Some("/home/op/export-only"))),
                None
            ),
            expected(
                "z",
                1,
                10,
                true,
                Some(("fallback", Some("/home/op/first"))),
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
    let found = reader.chains_behind_index(&[]).unwrap();
    reader
        .conn()
        .unwrap()
        .progress_handler(0, None::<fn() -> bool>)
        .unwrap();
    let rows: Vec<_> = (1..=chains)
        .map(|chain| {
            expected(
                "synthetic",
                chain as i64,
                (revisions - chains + chain) as u64,
                chain == chains,
                None,
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
