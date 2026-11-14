use super::*;
use rusqlite::types::Value;

const ALL_CHAINS: &str = include_str!("all_chains.sql");

type Rows = Vec<Vec<Value>>;

struct Work {
    rows: Rows,
    page_reads: u64,
    steps: usize,
}

fn query(reader: &Reader<'_>, sql: &str, values: &[Value]) -> Work {
    let conn = reader.conn().unwrap();
    conn.execute_batch("PRAGMA cache_size=-64; PRAGMA a.cache_size=-64; PRAGMA shrink_memory;")
        .unwrap();
    reader.take_page_reads().unwrap();
    let steps = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&steps);
    conn.progress_handler(
        100,
        Some(move || {
            observed.fetch_add(100, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    let mut statement = conn.prepare(sql).unwrap();
    let columns = statement.column_count();
    let rows = statement
        .query_map(rusqlite::params_from_iter(values), |row| {
            (0..columns)
                .map(|i| row.get(i))
                .collect::<rusqlite::Result<Vec<Value>>>()
        })
        .unwrap()
        .collect::<rusqlite::Result<Rows>>()
        .unwrap();
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    Work {
        rows,
        page_reads: reader.take_page_reads().unwrap(),
        steps: steps.load(Ordering::Relaxed),
    }
}

fn retain(rows: &Rows, chains: RevisionChains, selected: Option<&[String]>) -> Rows {
    rows.iter()
        .filter(|row| {
            let current = row[3] == Value::Integer(1);
            (match chains {
                RevisionChains::All => true,
                RevisionChains::Newest => current,
                RevisionChains::Older => !current,
            }) && selected.is_none_or(|ids| ids.iter().any(|id| row[0] == Value::Text(id.clone())))
        })
        .cloned()
        .collect()
}

// This register includes histories the normal append writer would not create:
// earlier chains resume after rewrites, with a shorter last declared length.
fn history(fixture: &Fixture, chains: usize, appends: usize) {
    let mut archive = Connection::open(fixture.archive.archive_db()).unwrap();
    let tx = archive.transaction().unwrap();
    tx.execute_batch("INSERT INTO input VALUES ('a','native'),('b','native'),('zero','native');
        INSERT INTO revision VALUES ('b',1,'first',1,10,1),('zero',1,'first',1,0,1);
        INSERT INTO observation(id,input,machine,path,class,details_json,polylogue_version,seen_at) VALUES
        ('b','b','m','/home/op/b','first','{}','test',100),
        ('a-first','a','m','/home/op/first','imported','{\"import_format\":\"old\"}','test',1),
        ('a-format','a','m','/home/op/chosen','imported','{\"import_format\":\"native\"}','test',2),
        ('a-export','a','m','/home/op/export','imported','{\"import_format\":\"export-file\"}','test',99);").unwrap();
    let mut number = 0;
    for chain in 1..=chains {
        for append in 1..=appends {
            number += 1;
            tx.execute(
                "INSERT INTO revision VALUES ('a',?1,?2,?3,?4,?1)",
                params![
                    number,
                    if chain == 1 { "imported" } else { "appended" },
                    chain,
                    100 + append
                ],
            )
            .unwrap();
        }
    }
    number += 1;
    tx.execute(
        "INSERT INTO revision VALUES ('a',?1,'appended',1,1,?1)",
        [number],
    )
    .unwrap();
    tx.commit().unwrap();
    let mut index = Connection::open(fixture.archive.index_db()).unwrap();
    let tx = index.transaction().unwrap();
    tx.execute("INSERT INTO input_ref(id) VALUES ('a')", [])
        .unwrap();
    for chain in 1..=chains {
        tx.execute("INSERT INTO parse_state(input_pk,chain,status,parsed_upto,at_length,reading_version,checkpoint,reason)
            SELECT pk,?1,?2,?3,?4,'old',zeroblob(16384),?5 FROM input_ref WHERE id='a'",
            params![chain, match chain%4 {0=>"waiting",1=>"parsing",2=>"refused",_=>"done"},
                if chain%4==3 {100+appends} else {0},
                matches!(chain%4,0|2).then_some(100+appends),
                (chain%4==2).then_some("synthetic refusal")]).unwrap();
    }
    tx.commit().unwrap();
}

#[test]
fn newest_and_older_match_the_original_query_for_retained_histories() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    history(&fixture, 12, 7);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let selected = vec!["a".into(), "a".into(), "zero".into(), "missing".into()];
    for versions in ["[]", "[[\"native\",\"new\"]]"] {
        for stopped in [0, 1] {
            let old = query(
                &reader,
                ALL_CHAINS,
                &[Value::Text(versions.into()), Value::Integer(stopped)],
            );
            for chains in [
                RevisionChains::All,
                RevisionChains::Newest,
                RevisionChains::Older,
            ] {
                for selection in [None, Some(selected.as_slice()), Some(&[][..])] {
                    let actual = query(
                        &reader,
                        &planning_sql(selection.is_some(), chains),
                        &[
                            Value::Text(versions.into()),
                            Value::Integer(stopped),
                            selection
                                .map(|ids| Value::Text(serde_json::to_string(ids).unwrap()))
                                .unwrap_or(Value::Null),
                        ],
                    );
                    assert_eq!(
                        actual.rows,
                        retain(&old.rows, chains, selection),
                        "{chains:?} {selection:?} {versions} stopped={stopped}"
                    );
                }
            }
        }
    }
    let newest = reader
        .chains_behind_index(&[], StoppedChains::Settled, RevisionChains::Newest, None)
        .unwrap();
    let a = newest.iter().find(|c| c.input == "a").unwrap();
    assert_eq!((a.progress.chain, a.progress.length), (1, 107));
    assert_eq!(a.source.path.as_deref(), Some("/home/op/chosen"));
}

fn compare_work(chains: usize, appends: usize) -> (usize, usize) {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    history(&fixture, chains, appends);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let values = [
        Value::Text("[[\"native\",\"new\"]]".into()),
        Value::Integer(1),
    ];
    let old_newest = query(&reader, ALL_CHAINS, &values);
    let old_older = query(&reader, ALL_CHAINS, &values);
    let values = [values[0].clone(), values[1].clone(), Value::Null];
    let newest = query(
        &reader,
        &planning_sql(false, RevisionChains::Newest),
        &values,
    );
    let older = query(
        &reader,
        &planning_sql(false, RevisionChains::Older),
        &values,
    );
    assert_eq!(
        newest.rows,
        retain(&old_newest.rows, RevisionChains::Newest, None)
    );
    assert_eq!(
        older.rows,
        retain(&old_older.rows, RevisionChains::Older, None)
    );
    let before = old_newest.page_reads + old_older.page_reads;
    let after = newest.page_reads + older.page_reads;
    eprintln!(
        "{chains} chains x {appends} appends: original/newest+older SQLite page reads {before}/{after}, VM steps {}/{}; newest alone {}/{} page reads",
        old_newest.steps + old_older.steps,
        newest.steps + older.steps,
        old_newest.page_reads,
        newest.page_reads
    );
    assert!(
        newest.page_reads < old_newest.page_reads,
        "newest planning still reads unrelated chain states"
    );
    assert!(
        after < before,
        "split discovery reads {after} pages, original reads {before}"
    );
    let before_steps = old_newest.steps + old_older.steps;
    let after_steps = newest.steps + older.steps;
    assert!(
        after_steps < before_steps,
        "split discovery must reduce VM work too"
    );
    (before_steps, after_steps)
}

#[test]
fn discovery_does_not_read_both_chain_sets_parser_states_twice() {
    let small = compare_work(32, 2);
    let large = compare_work(256, 8);
    assert!(
        large.0 - large.1 > small.0 - small.1,
        "historical parser-state VM work saving must grow with history"
    );
}
