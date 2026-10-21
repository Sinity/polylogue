use super::*;
use crate::sinex_rows::{ExcisionPublication, units};
use crate::{
    Archive,
    archive::{cache, open_write},
};

fn note(conn: &Connection, id: &str, seq: i64) {
    conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES (?1,?1,'note','body','operator','test',1,?2)",params![id,seq]).unwrap();
}

#[test]
fn inventory_order_and_nonprefix_held_positions_match_the_reference_units() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    for (id, seq) in [("z", 1), ("a\0", 1), ("é", 1), ("last", 4)] {
        note(&conn, id, seq);
    }
    let published = Published::open(dir.path()).unwrap();
    let mut inventory = published.inventory().unwrap();
    let expected = units(&published.conn, 0).unwrap();
    assert_eq!(inventory.len(), expected.len() as u64);
    for (n, unit) in expected.iter().enumerate() {
        assert_eq!(inventory.unit(n as u64 + 1).unwrap(), *unit);
    }
    assert!(!inventory.has_excision());
    assert_eq!(inventory.last_missing().unwrap(), Some(4));
    inventory.mark_held(&[1]).unwrap();
    inventory.mark_held(&[4]).unwrap();
    inventory.mark_held(&[4]).unwrap();
    assert_eq!(inventory.held(), 2);
    assert_eq!(inventory.last_missing().unwrap(), Some(3));
    assert!(inventory.mark_held(&[0]).is_err());
    assert!(inventory.mark_held(&[2, 5]).is_err());
    assert_eq!(inventory.held(), 2);
    assert!(inventory.unit(u64::MAX).is_err());
    assert!(inventory.next_missing(u64::MAX).is_err());
    let mut cursor = 0;
    let mut missing = Vec::new();
    while let Some((position, unit)) = inventory.next_missing(cursor).unwrap() {
        missing.push((position, unit));
        inventory.mark_held(&[position]).unwrap();
        cursor = position;
    }
    assert_eq!(missing.iter().map(|r| r.0).collect::<Vec<_>>(), [2, 3]);
    assert_eq!(inventory.held(), 4);
    assert_eq!(inventory.last_missing().unwrap(), None);
    assert!(inventory.next_replacement(None).unwrap().is_none());
}

fn publication(conn: &Connection, id: &str, seq: i64, replacements: Vec<Replacement>) {
    let mut publication = ExcisionPublication {
        version: 1,
        replacements,
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    let data = serde_json::to_string(&serde_json::json!({"publication":publication})).unwrap();
    conn.execute("INSERT INTO excision(id,scope,created_at,commit_seq,refused_json) VALUES (?1,'text:synthetic',1,?2,?3)",params![id,seq,data]).unwrap();
}
fn replacement(seq: i64, kind: &str, key: &str, body: bool) -> Replacement {
    Replacement {
        unit: Unit {
            commit_seq: seq,
            kind: kind.into(),
            key: key.into(),
        },
        batch: body.then(|| format!("blake3:{}", "01".repeat(32))),
        bytes: if body { 10 } else { 0 },
        rows: u64::from(body),
        blobs: vec![],
    }
}

#[test]
fn deleted_identities_and_latest_replacements_are_selected_before_payload_access() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    publication(
        &conn,
        "erase2",
        3,
        vec![
            replacement(1, "note", "gone", false),
            replacement(2, "excision", "erase1", true),
        ],
    );
    publication(
        &conn,
        "erase1",
        2,
        vec![replacement(1, "note", "gone", true)],
    );
    let published = Published::open(dir.path()).unwrap();
    let inventory = published.inventory().unwrap();
    let expected = units(&published.conn, 0).unwrap();
    assert_eq!(inventory.len(), 3);
    assert!(inventory.has_excision());
    for (n, unit) in expected.iter().enumerate() {
        assert_eq!(inventory.unit(n as u64 + 1).unwrap(), *unit);
    }
    let first = inventory.next_replacement(None).unwrap().unwrap();
    assert_eq!(first, replacement(1, "note", "gone", false));
    let next = inventory
        .next_replacement(Some(&first.unit))
        .unwrap()
        .unwrap();
    assert_eq!(next, replacement(2, "excision", "erase1", true));
    assert!(
        inventory
            .next_replacement(Some(&next.unit))
            .unwrap()
            .is_none()
    );
}

#[test]
fn legacy_and_forward_replacement_identities_refuse_the_inventory() {
    for forward in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        if forward {
            publication(
                &conn,
                "erase",
                2,
                vec![replacement(3, "note", "future", true)],
            );
        } else {
            conn.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('legacy','text:synthetic',1,1)",[]).unwrap();
        }
        let published = Published::open(dir.path()).unwrap();
        assert!(matches!(published.inventory(), Err(StoreError::Invalid(_))));
    }
}

#[cfg(target_os = "linux")]
#[test]
fn many_unit_keys_stay_on_disk_while_positions_are_acknowledged() {
    fn peak() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|l| {
                l.strip_prefix("VmHWM:")
                    .map(|v| v.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    conn.pragma_update(None, "cache_size", -1024).unwrap();
    let suffix = "x".repeat(2048);
    {
        let tx = conn.transaction().unwrap();
        let mut insert=tx.prepare("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES (?1,?1,'note','body','operator','test',1,?2)").unwrap();
        for n in 1..=20_000 {
            insert
                .execute(params![format!("{n:05}:{suffix}"), n])
                .unwrap();
        }
        drop(insert);
        tx.commit().unwrap();
    }
    drop(conn);
    let published = Published::open(dir.path()).unwrap();
    let before = peak();
    let mut inventory = published.inventory().unwrap();
    assert_eq!(inventory.len(), 20_000);
    for start in (1..=20_000).step_by(1000) {
        let positions: Vec<_> = (start..start + 1000).step_by(2).collect();
        inventory.mark_held(&positions).unwrap();
    }
    assert_eq!(inventory.held(), 10_000);
    assert_eq!(inventory.last_missing().unwrap(), Some(20_000));
    let mut cursor = 0;
    let mut count = 0;
    while let Some((position, _)) = inventory.next_missing(cursor).unwrap() {
        assert_eq!(position % 2, 0);
        cursor = position;
        count += 1;
    }
    assert_eq!(count, 10_000);
    let growth = peak().saturating_sub(before);
    eprintln!("20,000 publication keys: high-water growth {growth}KiB");
    assert!(
        growth < 32 * 1024,
        "publication inventory grew by {growth}KiB"
    );
}
