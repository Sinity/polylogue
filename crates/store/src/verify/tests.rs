use super::*;
use crate::staging::PreparedBatch;
use crate::{Archive, Batch, Change, InFlight, InputKey, Observation, Writer};

struct Stored {
    archive: Archive,
    writer: Writer,
    _lock: crate::ArchiveLock,
    _dir: tempfile::TempDir,
}

impl Stored {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        Self {
            archive,
            writer,
            _lock: lock,
            _dir: dir,
        }
    }

    /// Stores `body` as the bytes of `input` from `from` on: its first
    /// revision at zero, an append after.
    fn store(&self, input: &str, from: u64, body: &[u8]) {
        let (blob, file) = self.archive.prepare_blob(body).unwrap();
        let batch = Batch {
            required_disposition: None,
            input: InputKey {
                id: input.into(),
                format: "test-bytes".into(),
                route: polylogue_model::Route::Live,
            },
            change: if from == 0 {
                Change::First
            } else {
                Change::Appended { from }
            },
            observation: Some(Observation {
                machine: "synthetic-machine".into(),
                host: "synthetic-host".into(),
                path: format!("/home/op/{input}"),
                path_bytes: format!("/home/op/{input}").into_bytes(),
                facts: None,
                seen_at_ms: 1,
                import: None,
                container: None,
                database: None,
            }),
            continues: false,
            new_blob: Some(blob),
            blobs: Vec::new(),
            length: from + body.len() as u64,
            parse: Vec::new(),
            acquisition: None,
            basis: None,
            release: InFlight::default(),
        };
        self.writer
            .sender()
            .unwrap()
            .store_group(
                vec![PreparedBatch {
                    batch,
                    files: file.into_iter().collect(),
                }],
                Arc::new(self.archive.blobs().unwrap()),
            )
            .unwrap()
            .pop()
            .unwrap()
            .unwrap();
    }

    fn verify(&self, manifest: Option<&mut dyn Write>) -> Verification {
        self.writer.flush().unwrap();
        let pool = ReadPool::open(&self.archive).unwrap();
        verify(
            &pool,
            &Arc::new(self.archive.blobs().unwrap()),
            manifest,
            &mut |_: &dyn Fn()| Ok(()),
        )
        .unwrap()
    }

    fn edit(&self, sql: &str) {
        self.writer.flush().unwrap();
        crate::archive::open_write(&self.archive.archive_db(), 1)
            .unwrap()
            .execute_batch(sql)
            .unwrap();
    }
}

/// Two inputs: `growing` written in three steps (one chain, three
/// revisions) and `whole` written once.
fn two_inputs() -> Stored {
    let stored = Stored::new();
    stored.store("growing", 0, b"first line\n");
    stored.store("growing", 11, b"second line\n");
    stored.store("growing", 23, b"third line\n");
    stored.store("whole", 0, b"one whole file\n");
    stored
}

fn kinds(found: &Verification) -> Vec<ProblemKind> {
    found.problems.iter().map(|problem| problem.kind).collect()
}

#[test]
fn a_consistent_archive_has_no_problems_and_counts_every_class() {
    let stored = two_inputs();
    let found = stored.verify(None);
    assert_eq!(found.problems, []);
    assert_eq!(
        found.counts,
        Counts {
            inputs: 2,
            revisions: 4,
            extents: 4,
            observations: 4,
            blobs: 4,
            blob_bytes: 11 + 12 + 11 + 15,
        }
    );
    assert_eq!(
        found.formats,
        [FormatCount {
            format: "test-bytes".into(),
            inputs: 2,
            revisions: 4,
            revision_bytes: 11 + 23 + 34 + 15,
        }]
    );
    assert_eq!(
        found.revision_kinds,
        [
            ClassCount {
                class: "appended".into(),
                count: 2
            },
            ClassCount {
                class: "first".into(),
                count: 2
            },
        ]
    );
    assert_eq!(
        found.observation_classes,
        [
            ClassCount {
                class: "appended".into(),
                count: 2
            },
            ClassCount {
                class: "first".into(),
                count: 2
            },
        ]
    );
    assert_eq!(
        found.blobs,
        BlobCheck {
            checked: 4,
            bytes: 11 + 12 + 11 + 15,
            stored_bytes: found.blobs.stored_bytes,
            mismatched: 0,
        }
    );
    assert!(found.blobs.stored_bytes > 0);
    let schemas: Vec<_> = found
        .databases
        .iter()
        .map(|database| (database.schema.as_str(), database.result.clone()))
        .collect();
    assert!(
        schemas.contains(&("main", vec!["ok".to_string()])),
        "{schemas:?}"
    );
    assert!(
        schemas.contains(&("a", vec!["ok".to_string()])),
        "{schemas:?}"
    );
    assert_eq!(found.manifest, None);
}

#[test]
fn the_manifest_names_each_revision_with_the_hash_of_its_bytes() {
    let stored = two_inputs();
    let mut manifest = Vec::new();
    let found = stored.verify(Some(&mut manifest));
    assert_eq!(found.problems, []);
    // Expected lines from the bytes themselves, not from the store.
    let growing = b"first line\nsecond line\nthird line\n";
    let line = |input: &str, number: u32, bytes: &[u8]| {
        format!(
            "{input}\t{number}\t{}\t{}\n",
            bytes.len(),
            blake3::hash(bytes).to_hex()
        )
    };
    let expected = [
        line("growing", 1, &growing[..11]),
        line("growing", 2, &growing[..23]),
        line("growing", 3, &growing[..]),
        line("whole", 1, b"one whole file\n"),
    ]
    .concat();
    assert_eq!(String::from_utf8(manifest).unwrap(), expected);
    assert_eq!(
        found.manifest,
        Some(ManifestSummary {
            revisions: 4,
            bytes_read: 34 + 15,
        })
    );
}

#[test]
fn a_corrupted_stored_byte_is_named_and_left_as_it_is() {
    let stored = two_inputs();
    stored.writer.flush().unwrap();
    let blob = blake3::hash(b"second line\n");
    let storage: Vec<u8> = crate::archive::open_read(&stored.archive.archive_db())
        .unwrap()
        .query_row(
            "SELECT storage FROM blob WHERE hash = ?1",
            [blob.as_bytes().as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    let extent = hashed_blobs::Extent::decode(&storage).unwrap();
    let segment = stored.archive.blobs().unwrap().root().join(&extent.segment);
    let mut bytes = std::fs::read(&segment).unwrap();
    let at = usize::try_from(extent.offset + extent.len / 2).unwrap();
    bytes[at] ^= 0xff;
    std::fs::write(&segment, &bytes).unwrap();

    let mut manifest = Vec::new();
    let found = stored.verify(Some(&mut manifest));
    let named: Vec<_> = found
        .problems
        .iter()
        .filter(|problem| problem.kind == ProblemKind::BlobBytes)
        .map(|problem| problem.subject.clone())
        .collect();
    assert_eq!(named, [blob.to_hex().to_string()]);
    assert_eq!(found.blobs.mismatched, 1);
    // The revisions that read the damaged blob are named; the others, the
    // one before it in the same chain included, keep their lines.
    let unreadable: Vec<_> = found
        .problems
        .iter()
        .filter(|problem| problem.kind == ProblemKind::RevisionUnreadable)
        .map(|problem| problem.subject.as_str())
        .collect();
    assert_eq!(unreadable, ["growing#2", "growing#3"]);
    let manifest = String::from_utf8(manifest).unwrap();
    let inputs: Vec<_> = manifest
        .lines()
        .map(|line| line.split('\t').take(2).collect::<Vec<_>>().join("#"))
        .collect();
    assert_eq!(inputs, ["growing#1", "whole#1"]);
    assert_eq!(
        std::fs::read(&segment).unwrap(),
        bytes,
        "verify rewrote a segment"
    );
}

#[test]
fn verifying_changes_no_row() {
    let stored = two_inputs();
    stored.writer.flush().unwrap();
    let snapshot = || {
        let conn = crate::archive::open_read(&stored.archive.archive_db()).unwrap();
        let tables: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        tables
            .into_iter()
            .map(|table| {
                let rows: i64 = conn
                    .query_row(&format!("SELECT count(*) FROM \"{table}\""), [], |row| {
                        row.get(0)
                    })
                    .unwrap();
                (table, rows)
            })
            .collect::<Vec<_>>()
    };
    let write_counter = || -> String {
        crate::archive::open_read(&stored.archive.archive_db())
            .unwrap()
            .query_row(
                "SELECT coalesce((SELECT value FROM meta WHERE key = 'write_counter'), '')",
                [],
                |row| row.get(0),
            )
            .unwrap()
    };
    let (before, counter) = (snapshot(), write_counter());
    let _ = stored.verify(Some(&mut Vec::new()));
    assert_eq!(snapshot(), before);
    assert_eq!(write_counter(), counter);
}

#[test]
fn a_counter_that_differs_from_its_rows_is_reported() {
    let stored = two_inputs();
    stored.edit("UPDATE row_count SET n = n + 1 WHERE name = 'revision'");
    let found = stored.verify(None);
    assert_eq!(kinds(&found), [ProblemKind::RowCount]);
    assert_eq!(found.problems[0].subject, "revision");
    assert_eq!(found.problems[0].detail, "kept 5, counted 4");
}

#[test]
fn a_missing_middle_extent_is_a_gap() {
    let stored = two_inputs();
    stored.edit("DELETE FROM extent WHERE input = 'growing' AND offset = 11");
    let found = stored.verify(None);
    assert_eq!(kinds(&found), [ProblemKind::ExtentGap]);
    assert_eq!(found.problems[0].subject, "growing/1@23");
    assert_eq!(found.problems[0].detail, "gap after 11");
}

#[test]
fn a_revision_longer_than_its_extents_is_reported() {
    let stored = two_inputs();
    stored.edit("DELETE FROM extent WHERE input = 'growing' AND offset = 23");
    let found = stored.verify(None);
    assert_eq!(kinds(&found), [ProblemKind::RevisionUncovered]);
    assert_eq!(found.problems[0].subject, "growing#3");
    assert_eq!(
        found.problems[0].detail,
        "length 34, extents of chain 1 cover 23"
    );
}

#[test]
fn an_extent_citing_an_absent_blob_is_reported() {
    let stored = two_inputs();
    let missing = "00".repeat(32);
    stored.edit(&format!(
        "PRAGMA foreign_keys = OFF;
         UPDATE extent SET blob = x'{missing}' WHERE input = 'whole'"
    ));
    let found = stored.verify(None);
    assert_eq!(kinds(&found), [ProblemKind::ExtentWithoutBlob]);
    assert_eq!(found.problems[0].subject, "whole/1@0");
    assert_eq!(found.problems[0].detail, format!("blob {missing}"));
}

#[test]
fn verification_stops_when_its_pace_refuses() {
    let stored = two_inputs();
    stored.writer.flush().unwrap();
    let pool = ReadPool::open(&stored.archive).unwrap();
    let mut calls = 0;
    let result = verify(
        &pool,
        &Arc::new(stored.archive.blobs().unwrap()),
        None,
        &mut |_: &dyn Fn()| {
            calls += 1;
            if calls > 3 {
                Err(StoreError::ReadCancelled)
            } else {
                Ok(())
            }
        },
    );
    assert!(
        matches!(result, Err(StoreError::ReadCancelled)),
        "{result:?}"
    );
}
