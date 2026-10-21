use super::*;
use std::fs::File;
use std::io::{Read, Seek, Write};

struct Input {
    file: File,
    remaining: Option<usize>,
}
impl Read for Input {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        assert!(
            buffer.len() <= 64 * 1024,
            "restore requested a growing read buffer"
        );
        let length = match self.remaining {
            Some(0) => return Err(std::io::Error::other("injected source failure")),
            Some(n) => n.min(buffer.len()).min(37),
            None => buffer.len().min(37),
        };
        let n = self.file.read(&mut buffer[..length])?;
        if let Some(left) = &mut self.remaining {
            *left -= n;
        }
        Ok(n)
    }
}

fn batch(targets: usize, tail: Option<Value>) -> (File, RestoreUnit) {
    let mut file = tempfile::tempfile().unwrap();
    let mut hash = blake3::Hasher::new();
    let mut length = 0;
    let mut row = |value: Value| {
        let mut bytes = serde_json::to_vec(&value).unwrap();
        bytes.push(b'\n');
        file.write_all(&bytes).unwrap();
        hash.update(&bytes);
        length += bytes.len() as u64;
    };
    row(json!({"table":"note", "row": {
        "id":"note1", "root":"note1", "kind":"note", "value":"new body",
        "author":"operator", "machine":"test", "created_at":1, "commit_seq":1
    }}));
    for n in 0..targets {
        row(json!({"table":"note_target", "row": {
            "note_id":"note1", "ref":format!("session:neutral:{n:08}")
        }}));
    }
    let extra = u64::from(tail.is_some());
    if let Some(tail) = tail {
        row(tail);
    }
    file.rewind().unwrap();
    (
        file,
        RestoreUnit {
            unit: Unit {
                commit_seq: 1,
                kind: "note".into(),
                key: "note1".into(),
            },
            batch: Some(format!("blake3:{}", hash.finalize())),
            bytes: Some(length),
            rows: targets as u64 + 1 + extra,
            blobs: vec![],
            excision: None,
        },
    )
}

fn target() -> Connection {
    let target = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&target).unwrap();
    target.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('note1','note1','note','old body','operator','test',1,1)", []).unwrap();
    target
}

#[test]
fn a_many_row_batch_restores_from_small_fragments_and_repeats_exactly() {
    let (mut file, unit) = batch(10_000, None);
    let selected = select(&[unit]).unwrap();
    let mut target = target();
    for expected in [10_001, 0] {
        file.rewind().unwrap();
        let inserted = restore::install(&mut target, &selected, None, |_| {
            Ok(Some(Input {
                file: file.try_clone()?,
                remaining: None,
            }))
        })
        .unwrap();
        assert_eq!(inserted, expected);
    }
    assert_eq!(
        target
            .query_row("SELECT COUNT(*) FROM note_target", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        10_000
    );
    assert_eq!(
        target
            .query_row("SELECT MAX(ref) FROM note_target", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "session:neutral:00009999"
    );
}

#[test]
fn interrupted_stream_and_late_foreign_row_leave_the_previous_view_unchanged() {
    for bad_tail in [false, true] {
        let (file, unit) = batch(100, bad_tail.then(|| json!({
            "table":"note_target", "row":{"note_id":"unrelated", "ref":"session:neutral:other"}
        })));
        let selected = select(&[unit]).unwrap();
        let mut target = target();
        let result = restore::install(&mut target, &selected, None, |_| {
            Ok(Some(Input {
                file: file.try_clone()?,
                remaining: (!bad_tail).then_some(300),
            }))
        });
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains(if bad_tail {
                "outside its unit"
            } else {
                "injected source failure"
            }),
            "{error}"
        );
        assert_eq!(
            target
                .query_row("SELECT value FROM note WHERE id='note1'", [], |r| r
                    .get::<_, String>(0))
                .unwrap(),
            "old body"
        );
        assert_eq!(
            target
                .query_row("SELECT COUNT(*) FROM note_target", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            target
                .query_row(
                    "SELECT COUNT(*) FROM meta WHERE key='sinex_restored_view'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
    }
}
