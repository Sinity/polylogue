use super::*;

fn previous_generation_tail(path: &Path) -> bool {
    let bytes = std::fs::read(path).unwrap();
    if bytes.len() < 32 {
        return false;
    }
    let page = u32::from_be_bytes(bytes[8..12].try_into().unwrap()) as usize;
    assert!((512..=65536).contains(&page));
    bytes[32..]
        .chunks_exact(24 + page)
        .any(|frame| frame[8..16] != bytes[16..24])
}

#[test]
fn actor_closes_idle_readers_before_last_writer_and_plain_restart_accepts_reused_wal() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("archive");
    let writer = Writer::open(&root).unwrap();
    let (sender, _receiver) = mpsc::sync_channel(COMMANDS);
    let mut actor = Actor::new(
        writer,
        Arc::new(Inner {
            sender,
            accepting: Mutex::new(true),
            cancelled: Arc::new(AtomicBool::new(false)),
        }),
    );
    actor.canonical_reader = Some(actor.take_canonical_reader().unwrap());
    // Real SQLite auto-checkpoint/reuse, through the public store write path.
    let mut writes = 0u64;
    for n in 0u64..2048 {
        actor.writer.put_file_blob(&n.to_be_bytes()).unwrap();
        writes += 1;
        if writes % 64 == 0 && previous_generation_tail(&root.join("archive.db-wal")) {
            break;
        }
    }
    assert!(previous_generation_tail(&root.join("archive.db-wal")));
    let before = actor.writer.status().unwrap().write_counter;
    // An opened SQLite connection has not necessarily joined the WAL yet.
    // Exercise the cached capability, as canonical source verification does.
    assert_eq!(
        actor
            .canonical_reader
            .as_ref()
            .unwrap()
            .values
            .status()
            .unwrap()
            .write_counter,
        before
    );
    drop(actor);
    let reopened = Writer::open(&root).expect("clean actor shutdown must allow ordinary restart");
    assert_eq!(reopened.status().unwrap().write_counter, before);
}
