use super::*;
#[test]
fn main_and_multiple_small_companions_share_one_monotone_counter() {
    let temp = tempfile::tempdir().unwrap();
    let main = temp.path().join("session.jsonl");
    std::fs::write(&main, b"\n").unwrap();
    let directory = temp.path().join("session/tool-results");
    std::fs::create_dir_all(&directory).unwrap();
    let references = (0..3)
        .map(|index| {
            let path = directory.join(format!("{index}.txt"));
            std::fs::write(&path, vec![b'x'; 32768]).unwrap();
            (path.to_str().unwrap().to_owned(), None)
        })
        .collect();
    let writer = Writer::open(&temp.path().join("archive")).unwrap();
    let capture = writer.blob_capture().unwrap();
    let mut turns = Vec::new();
    let prepared = capture
        .prepare_claude_in_turns(
            &main,
            None,
            &AtomicBool::new(false),
            crate::history_pages::turns::Budget {
                bytes: 65536,
                checkpoints: 512,
            },
            |progress| {
                turns.push(progress);
                Ok(())
            },
            |_| Ok(()),
            || Ok(references),
        )
        .unwrap();
    assert_eq!(prepared.bodies.len(), 3);
    assert!(turns.len() > 3);
    assert!(turns.iter().all(|p| p.charged_bytes <= 65536));
    assert!(
        turns
            .windows(2)
            .all(|pair| pair[1].completed_turns == pair[0].completed_turns + 1)
    );
    assert!(turns.iter().any(|p| p.phase == "claude_body"));
    let selected: i64 = writer
        .archive
        .query_row("SELECT count(*) FROM input", [], |row| row.get(0))
        .unwrap();
    assert_eq!(selected, 0);
}
