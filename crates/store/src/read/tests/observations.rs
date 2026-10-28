use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn history(count: i64) -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let archive = crate::archive::open_archive_writer(&fixture.archive).unwrap();
    let tx = archive.unchecked_transaction().unwrap();
    for input in ["source", "imported", "gone", "other"] {
        tx.execute("INSERT INTO input VALUES(?1,'codex')", [input]).unwrap();
    }
    for number in 1..=count {
        let produced = if number == 1 { "first" } else { "appended" };
        tx.execute("INSERT INTO revision VALUES('source',?1,?2,1,?1,?1)", params![number,produced]).unwrap();
        tx.execute("INSERT INTO observation(id,input,machine,path,class,revision,length,device,inode,size,mtime_ns,ctime_ns,polylogue_version,seen_at) VALUES(?1,'source','synthetic','/source',?4,?2,?2,1,1,?2,?2,?2,'test',?3)", params![format!("observation-{number}"),number,count-number,produced]).unwrap();
    }
    for (input, produced, length) in [("imported", "imported", 9000), ("gone", "first", 10), ("other", "first", 7000)] {
        tx.execute("INSERT INTO revision VALUES(?1,1,?2,1,?3,1)", params![input,produced,length]).unwrap();
    }
    tx.execute_batch("INSERT INTO observation(id,input,machine,path,class,polylogue_version,seen_at) VALUES ('import','imported','synthetic','/source','imported','test',1), ('removed','source','synthetic','/source','excised','test',2), ('gone','gone','synthetic','/gone','deleted','test',3), ('other','other','elsewhere','/source','first','test',4)").unwrap();
    tx.commit().unwrap();
    (dir, fixture)
}

#[test]
fn current_source_positions_use_commit_order_and_exclude_imports_and_excisions() {
    let (_dir, fixture) = history(80);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let known = reader.known_watched_inputs("synthetic").unwrap();
    assert_eq!(known.len(), 2);
    let source = known.iter().find(|input| input.path == "/source").unwrap();
    assert_eq!(source.input_id, "source");
    assert_eq!(source.stored, 80);
    assert_eq!(source.facts.unwrap().size, 80);
    assert!(known.iter().find(|input| input.path == "/gone").unwrap().facts.is_none());
    let at = reader.known_watched_input_at("synthetic", "/source").unwrap().unwrap();
    assert_eq!((at.input_id.as_str(), at.stored), ("source", 80));
    let imported = reader.known_inputs("synthetic").unwrap();
    assert_eq!(imported.iter().find(|input| input.path == "/source").unwrap().input_id, "imported");
    assert_eq!(reader.known_watched_inputs("elsewhere").unwrap()[0].input_id, "other");
}

fn read_work(count: i64) -> usize {
    let (_dir, fixture) = history(count);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let conn = reader.conn().unwrap();
    let work = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&work);
    conn.progress_handler(100, Some(move || {
        observed.fetch_add(100, Ordering::Relaxed);
        false
    }));
    assert_eq!(reader.known_watched_inputs("synthetic").unwrap().len(), 2);
    conn.progress_handler(0, None::<fn() -> bool>);
    work.load(Ordering::Relaxed)
}

#[test]
fn doubling_observation_history_does_not_rescan_it_for_each_observation() {
    let smaller = read_work(80);
    let larger = read_work(160);
    assert!(larger < smaller * 4, "SQLite work grew from {smaller} to {larger} steps");
}
